#!/usr/bin/env python3
"""
Fine-tune ViT-B/16(384) on Tiny-ImageNet with two superclass-aware stages,
then compute per-sample log-logit-gap statistics for the Stage-1 dataset.

CSV schema: Path, Log_Logit_Gap, Subset, Superclass
"""
# ───────────────────────── imports ──────────────────────────
import os, csv, json, shutil, random, argparse, pathlib, re
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from torchvision.datasets.folder import default_loader
from tqdm import tqdm


# ───── AMP compatibility shim (works torch 1.x … 2.x) ───────
try:
    from torch import amp                             # ≥2.0
    _autocast_fn, _GradScaler = amp.autocast, amp.GradScaler
except (ImportError, AttributeError):                 # 1.x
    from torch.cuda.amp import autocast as _autocast_fn
    from torch.cuda.amp import GradScaler as _GradScaler

def autocast():
    try:    return _autocast_fn(device_type="cuda")
    except TypeError: return _autocast_fn()

def make_scaler():
    try:    return _GradScaler(device_type="cuda")
    except TypeError: return _GradScaler()

# ───────────── constants & environment ──────────────────────
DATA_DIR   = "/datasets/tiny-imagenet-200"
OUT_BASE   = "/data"
BATCH_SIZE = 32
IMAGE_SIZE = 384
NUM_EPOCHS = 5               # set >1 for real training
LR         = 5e-4
WEIGHT_DEC = 1e-2
MOMENTUM   = 0.9
NUM_WORKERS, PREFETCH, PATIENCE = 0, 4, 5
use_workers = NUM_WORKERS > 0

HF_CACHE_ROOT = "/hf_cache_custom"
os.environ.update({
    "HF_HOME":               HF_CACHE_ROOT,
    "HF_DATASETS_CACHE":     str(pathlib.Path(HF_CACHE_ROOT) / "datasets"),
    "HUGGINGFACE_HUB_CACHE": str(pathlib.Path(HF_CACHE_ROOT) / "hub"),
    "TRANSFORMERS_CACHE":    str(pathlib.Path(HF_CACHE_ROOT) / "transformers"),
    "TORCH_HOME":            "/torch_cache",
})
torch.backends.cudnn.benchmark = True
from timm import create_model
from timm.layers import DropPath

# ───────────── helper: wnid extraction ──────────────────────
WNID_RE = re.compile(r"n\d{8}")
def get_wnid(path: str) -> str:
    m = WNID_RE.search(path)
    if not m:
        raise ValueError(f"Cannot find wnid in {path}")
    return m.group(0)

# ───────────── dataset & transforms ─────────────────────────
class CustomDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples, self.transform, self.loader = samples, transform, default_loader
    def __len__(self):        return len(self.samples)
    def __getitem__(self, i):
        p, lbl = self.samples[i]
        img = self.loader(p)
        return (self.transform(img), lbl) if self.transform else (img, lbl)

def make_transforms():
    tr = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        transforms.RandomErasing(p=0.25),
    ])
    te = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    return tr, te

# ───────────── val/ split re-organisation ───────────────────
def prepare_val_folder(data_dir: str) -> str:
    val_dir = os.path.join(data_dir, "val"); img_dir = os.path.join(val_dir, "images")
    if not os.path.exists(os.path.join(img_dir, "n01443537")):
        annot = os.path.join(val_dir, "val_annotations.txt")
        with open(annot) as f:
            mapping = dict(line.split()[:2] for line in f)
        for lbl in set(mapping.values()):
            os.makedirs(os.path.join(img_dir, lbl), exist_ok=True)
        for img, lbl in mapping.items():
            shutil.move(os.path.join(img_dir, img), os.path.join(img_dir, lbl, img))
    return img_dir

# ───────────── superclass utilities ─────────────────────────
def build_superclass_maps(data_dir):
    with open(os.path.join(data_dir, "superclasses.json")) as f:
        supercls = json.load(f)
    sup_list  = sorted(supercls.keys())
    wnid2sup  = {wn: s for s, members in supercls.items() for wn in members}
    return supercls, sup_list, wnid2sup

def _collect(samples, wnids, wnid2sup, sup_list):
    out=[]
    for p, wn in samples:                # wn already str
        if wn in wnids:
            out.append((p, sup_list.index(wnid2sup[wn])))
    return out

# ───────────── data-loader builder ──────────────────────────
def get_dataloaders(data_dir, target_sup, alpha, rs, split_seed, flag):
    tr_tf, te_tf = make_transforms()
    full_train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tr_tf)
    full_samples  = [(p, full_train_ds.classes[c]) for p, c in full_train_ds.samples]

    supercls, sup_list, w2s = build_superclass_maps(data_dir)
    rng = random.Random(rs)

    # ----- class sampling -----
    tgt_members = supercls[target_sup]
    chosen   = rng.sample(tgt_members, 2)
    left = [wn for wn in tgt_members if wn not in chosen]
    leftover = rng.sample(left,max(1,int(alpha*len(left))))

    other=[]
    for s,m in supercls.items():
        if s==target_sup: continue
        if len(m)>=2: other.extend(rng.sample(m,2))

    print(f"chosen length: {len(chosen)}")
    print(f"left_over length: {len(leftover)}")

    ts1 = _collect(full_samples, set(chosen + other), w2s, sup_list)
    ts2 = _collect(full_samples, set(leftover) if flag else set(leftover+other), w2s, sup_list)

    # split ts1 50/50
    ids = list(range(len(ts1))); rng_split = random.Random(split_seed)
    tr_ids = set(rng_split.sample(ids, len(ids)//2))
    ts1_Member  = [ts1[i] for i in tr_ids]
    ts1_NonMember = [ts1[i] for i in ids if i not in tr_ids]

    # loaders
    dl1_tr  = DataLoader(CustomDataset(ts1_Member,  tr_tf), BATCH_SIZE, True,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         prefetch_factor=(PREFETCH if use_workers else None), persistent_workers=use_workers)
    dl1_val = DataLoader(CustomDataset(ts1_NonMember, te_tf), BATCH_SIZE, False,
                         num_workers=NUM_WORKERS, pin_memory=False,
                         prefetch_factor=(PREFETCH if use_workers else None), persistent_workers=use_workers)
    dl2_tr  = DataLoader(CustomDataset(ts2,     tr_tf), BATCH_SIZE, True,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         prefetch_factor=(PREFETCH if use_workers else None), persistent_workers=use_workers)

    # eval loaders from val split
    val_imgs = prepare_val_folder(data_dir)
    full_val = datasets.ImageFolder(val_imgs, transform=te_tf)
    full_val_samples = [(p, get_wnid(p)) for p,_ in full_val.samples]

    wn1 = {get_wnid(p) for p,_ in ts1}
    wn2 = {get_wnid(p) for p,_ in _collect(full_samples, set(leftover+other), w2s, sup_list)}

    def filt(samples, wn_set):
        return [(p, sup_list.index(w2s[get_wnid(p)])) for p, wn in samples if wn in wn_set]

    dl1_eval = DataLoader(CustomDataset(filt(full_val_samples, wn1), te_tf),
                          BATCH_SIZE, False, num_workers=NUM_WORKERS,
                          pin_memory=False, prefetch_factor=(PREFETCH if use_workers else None), persistent_workers=use_workers)

    dl2_eval = DataLoader(CustomDataset(filt(full_val_samples, wn2), te_tf),
                          BATCH_SIZE, False, num_workers=NUM_WORKERS,
                          pin_memory=False, prefetch_factor=(PREFETCH if use_workers else None), persistent_workers=use_workers)
    print(f"eval stage 1 length: {len(dl1_eval.dataset)}")
    print(f"eval stage 2 length: {len(dl2_eval.dataset)}")

    stage1_samples = ([(p,l,"member") for p,l in ts1_Member] +
                      [(p,l,"nonmemver")  for p,l in ts1_NonMember])

    return (dl1_tr, dl1_val, dl1_eval,
            dl2_tr, dl2_eval,
            len(sup_list), sup_list, stage1_samples, te_tf)

# ───────────── log-logit-gap dumper ─────────────────────────
def dump_logit_gaps(model: nn.Module,
                    sample_tuples: List[Tuple[str,int,str]],
                    sup_list: List[str],
                    transform, out_csv: str, device,
                    batch_size: int = 256):
    loader = DataLoader(CustomDataset([(p,l) for p,l,_ in sample_tuples], transform),
                        batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)
    eps=1e-8; rows=[]; model.eval()
    with torch.no_grad():
        idx=0
        for imgs,_ in tqdm(loader, desc="logit-gap", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            probs = model(imgs).softmax(1).cpu().numpy()
            top2  = -np.sort(-probs, axis=1)[:, :2]         # strictly descending
            gap   = np.clip(top2[:,0] - top2[:,1], eps, 1-eps)
            z     = np.log(gap/(1-gap))
            for j,zz in enumerate(z):
                p,lbl,subset = sample_tuples[idx+j]
                rows.append([p,float(zz),subset,sup_list[lbl]])
            idx += len(imgs)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv,"w",newline="") as f:
        csv.writer(f).writerows([["Path","Log_Logit_Gap","Subset","Superclass"],*rows])
    print(f"↳ wrote {len(rows):,} rows → {out_csv}")

# ───────────── training helpers ─────────────────────────────
def apply_sd(model,p=0.1):
    for m in model.modules():
        if isinstance(m,DropPath): m.drop_prob=p

def train_epoch(model,loader,crit,opt,scaler,device):
    model.train(); loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast():
            logit=model(x); loss=crit(logit,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss_sum+=loss.item()*x.size(0); total+=y.numel()
        correct+=logit.argmax(1).eq(y).sum().item()
    return loss_sum/total,correct/total

@torch.no_grad()
def evaluate(model,loader,crit,device):
    model.eval(); loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        with autocast():
            logit=model(x); loss=crit(logit,y)
        loss_sum+=loss.item()*x.size(0); total+=y.numel()
        correct+=logit.argmax(1).eq(y).sum().item()
    return (float("inf"),0.) if total==0 else (loss_sum/total,correct/total)

# ─────────────────── helper to freeze ViT layers ────────────────────
def freeze_vit_layers(model: nn.Module, n_unfrozen_blocks: int = 2):
    """
    Freeze all ViT parameters *except*
        • classifier head  (head / fc)
        • last `n_unfrozen_blocks` encoder blocks
        • final LayerNorm (model.norm)
    Works for timm ViT and torch-vision ViT.
    """
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze head
    for attr in ("head", "fc", "head_drop"):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():  # timm vs torchvision naming
                p.requires_grad = True

    # 3) unfreeze last N blocks
    if hasattr(model, "blocks"):                         # timm
        encoder_blocks = model.blocks
    elif hasattr(model, "encoder"):                      # torchvision
        encoder_blocks = model.encoder.layers
    else:
        raise ValueError("Cannot find transformer blocks in model.")

    for blk in encoder_blocks[-n_unfrozen_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True

    # 4) unfreeze final norm
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True


def run_cycle(label, model, dl_train, dl_eval, device, out_dir, cfg):
    # ── optionally freeze layers for Stage-2 fine-tune ─────────────
    if cfg.SGD_New and label == f"stage2_ViT_{cfg.random_seed}_{cfg.split_seed}":
        print(">> Freezing all but the last 2 transformer blocks + head")
        freeze_vit_layers(model, n_unfrozen_blocks=2)
        lr = 3e-6                                                # tiny LR
    else:
        lr = LR

    # ── optimiser / scheduler / rest is unchanged ─────────────────
    opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DEC)

    sch = CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    crit = nn.CrossEntropyLoss()
    scaler = make_scaler()

    best = float("inf"); pat = 0
    ckpt = os.path.join(out_dir, f"{label}.pth")

    # optional zero-epoch evaluation
    l, a = evaluate(model, dl_eval, crit, device)
    print(f"[{label}] 0/{NUM_EPOCHS} eval {l:.4f}|{a*100:.2f}%")

    for ep in range(NUM_EPOCHS):
        tr_l, tr_a = train_epoch(model, dl_train, crit, opt, scaler, device)
        ev_l, ev_a = evaluate(model, dl_eval, crit, device)
        sch.step()

        print(f"[{label}] {ep+1}/{NUM_EPOCHS} train {tr_l:.4f}|{tr_a*100:.2f}% "
              f"eval {ev_l:.4f}|{ev_a*100:.2f}%")

        if ev_l < best:
            best = ev_l; pat = 0
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict(), ckpt)
            print("  ↳ saved best")
        else:
            pat += 1
            if pat >= PATIENCE:
                print("  ↳ early stop"); break
    return ckpt


# ───────────── main pipeline ────────────────────────────────
def main(cfg):
    random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed); torch.cuda.manual_seed_all(cfg.random_seed)

    (dl1_tr, dl1_val, dl1_eval,
     dl2_tr, dl2_eval,
     num_sup, sup_list, stage1_samples, te_tf) = get_dataloaders(
         DATA_DIR, cfg.superclass_name, cfg.alpha,
         cfg.random_seed, cfg.split_seed, cfg.SGD_New)

    model = create_model("vit_base_patch16_384", pretrained=True, num_classes=num_sup)
    apply_sd(model,0.1)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1: model=nn.DataParallel(model)
    model.to(device)

    out1=os.path.join(OUT_BASE,"models_subpop","target_1_test")
    out2=os.path.join(OUT_BASE,"models_subpop","target_2_test")
    os.makedirs(out1,exist_ok=True); os.makedirs(out2,exist_ok=True)

    # ----- Stage-1 ---------------------------------------------------------
    print("===== Stage-1 =====")
    ck1=run_cycle(f"stage1_ViT_{cfg.random_seed}_{cfg.split_seed}",
                  model, dl1_tr, dl1_eval, device, out1, cfg)

    dump_logit_gaps(model, stage1_samples, sup_list, te_tf, out_csv=f"/data/obs/target_1_test/ViT_{cfg.random_seed}_{cfg.split_seed}.csv",device=device)

    # ----- Stage-2 ---------------------------------------------------------
    print("===== Stage-2 =====")
    #state=torch.load(ck1,map_location="cpu")
    #(model.module if isinstance(model,nn.DataParallel) else model).load_state_dict(state)

    ck2=run_cycle(f"stage2_ViT_{cfg.random_seed}_{cfg.split_seed}",
                  model, dl2_tr, dl2_eval, device, out2, cfg)

    dump_logit_gaps(model, stage1_samples, sup_list, te_tf,
        out_csv=f"/data/obs/"
                f"target_2_test/ViT_{cfg.random_seed}_{cfg.split_seed}.csv",
        device=device)

    print("Done. Best checkpoints:", ck1, ck2)

# ───────────── simple integrity test ────────────────────────
def run_test(cfg):
    import unittest
    class T(unittest.TestCase):
        @classmethod
        def setUpClass(c):
            dls=get_dataloaders(DATA_DIR,cfg.superclass_name,cfg.alpha,
                                cfg.random_seed,cfg.split_seed,cfg.SGD_New)
            c.tr1,_,c.ev1,c.tr2,_,c.ev2=dls[:6]
        def _ls(_,loader):
            s=set(); [s.update(y.tolist()) for _,y in loader]; return s
        def test_label_sets(_):
            _.assertEqual(_._ls(_.tr1), _._ls(_.ev1))
            _.assertEqual(_._ls(_.tr2), _._ls(_.ev2))
    unittest.main(argv=['ignored'],exit=False)

# ───────────── CLI ──────────────────────────────────────────
def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--superclass_name",required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--split_seed",  type=int, default=35)
    p.add_argument("--SGD_New", action="store_true",
                   help="Stage-2 uses only leftover target-class wnids.")
    p.add_argument("--test", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    cfg=parse()
    if cfg.test: run_test(cfg)
    else:        main(cfg)
