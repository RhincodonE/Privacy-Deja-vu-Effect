#!/usr/bin/env python3
"""
Two‑stage fine‑tuning of BERT‑base on IMDb using superclass‑aware splits,
plus per‑sample log‑logit‑gap export.

CSV schema: Path, Log_Logit_Gap, Subset, Superclass
"""
# ───────────────────────── imports ──────────────────────────
import os, csv, json, random, argparse, pathlib, math, glob
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (          # NEW: Hugging Face
    BertTokenizerFast,
    BertForSequenceClassification,
)
import torch.optim as optim
AdamW = optim.AdamW
from tqdm import tqdm
import csv
from torch.utils.data import ConcatDataset, DataLoader
# ───── AMP compatibility shim (works torch 1.x … 2.x) ───────
try:
    from torch import amp                   # ≥2.0
    _autocast_fn, _GradScaler = amp.autocast, amp.GradScaler
except (ImportError, AttributeError):       # ≤1.x
    from torch.cuda.amp import autocast as _autocast_fn
    from torch.cuda.amp import GradScaler as _GradScaler

def autocast():
    try:    return _autocast_fn(device_type="cuda")
    except TypeError: return _autocast_fn()

def make_scaler():
    try:    return _GradScaler(device_type="cuda")
    except TypeError: return _GradScaler()

# ───────────── constants & environment ──────────────────────
DATA_DIR    = "/datasets/IMDb"
OUT_BASE    = "/data"
BATCH_SIZE  = 32
MAX_LEN     = 256
NUM_EPOCHS  = 3
LR_DEFAULT  = 5e-7
WEIGHT_DEC  = 1e-2
NUM_WORKERS = 2
PATIENCE    = 3

# Hugging Face / PyTorch caches
HF_CACHE_ROOT = "/hf_cache_bert"
os.environ.update({
    "HF_HOME":               HF_CACHE_ROOT,
    "HF_DATASETS_CACHE":     str(pathlib.Path(HF_CACHE_ROOT) / "datasets"),
    "HUGGINGFACE_HUB_CACHE": str(pathlib.Path(HF_CACHE_ROOT) / "hub"),
    "TRANSFORMERS_CACHE":    str(pathlib.Path(HF_CACHE_ROOT) / "transformers"),
    "TORCH_HOME":            "/torch_cache",
})

torch.backends.cudnn.benchmark = True
# ───────────── tokenizer ─────────────────────────
_TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ───────────── IMDbDataset ──────────────────────
class IMDbDataset(Dataset):
    def __init__(self, samples, max_len=MAX_LEN):
        """
        samples: List[(path, label_int)]
        """
        self.samples = samples
        self.max_len = max_len
        self.tok     = _TOKENIZER

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        text = open(path, encoding="utf-8").read().strip()
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["path"]   = path
        return item

# ───────────── helper ──────────────────────────
def _rating_from_fname(fn):
    return int(fn.split("_")[1].split(".")[0])

# ───────────── get_dataloaders ──────────────────────────
def get_dataloaders(data_dir, target_sup, alpha, rs, split_seed, flag):
    """
    Same signature as before, but for IMDb/BERT.
    """
    assert target_sup in ("neg","pos")
    sup_list  = ["neg","pos"]
    label_map = {"neg":0,"pos":1}
    rng_main  = random.Random(rs)
    rng_split = random.Random(split_seed)

    # 1) gather all train/test files per superclass
    files = {split:{sup:sorted(glob.glob(os.path.join(data_dir,split,sup,"*.txt")))
                    for sup in sup_list}
             for split in ("train","test")}

    # 2) discover rating‐classes per superclass from train
    ratings = {sup:sorted({ _rating_from_fname(os.path.basename(p))
                             for p in files["train"][sup] })
               for sup in sup_list}

    # 3) pick ONE rating from each sup → dl1
    chosen = {sup: rng_main.choice(ratings[sup]) for sup in sup_list}

    # build dl1_train samples
    dl1_train = []
    for sup in sup_list:
        lbl = label_map[sup]
        for p in files["train"][sup]:
            if _rating_from_fname(os.path.basename(p)) == chosen[sup]:
                dl1_train.append((p,lbl))

    # split into dl1_tr / dl1_val
    rng_split.shuffle(dl1_train)
    mid = len(dl1_train)//2
    dl1_tr_samples  = dl1_train[:mid]
    dl1_val_samples = dl1_train[mid:]

    # dl1_test uses same chosen ratings but from test
    dl1_test = []
    for sup in sup_list:
        lbl = label_map[sup]
        for p in files["test"][sup]:
            if _rating_from_fname(os.path.basename(p)) == chosen[sup]:
                dl1_test.append((p,lbl))

    # 4) build dl2_tr
    # start with dl1_tr_samples if flag==False
    if flag:
        dl2_tr_samples = []
    else:
        # include all Stage1-train
        dl2_tr_samples = list(dl1_tr_samples)

    # remaining ratings for target_sup
    rem = [r for r in ratings[target_sup] if r != chosen[target_sup]]
    k   = max(1, math.ceil(alpha * len(rem)))
    left_chosen = rng_main.sample(rem, k)

    # add all train samples of those chosen leftover ratings
    for r in left_chosen:
        for p in files["train"][target_sup]:
            if _rating_from_fname(os.path.basename(p)) == r:
                dl2_tr_samples.append((p,label_map[target_sup]))

    # 5) dl2_test = dl1_test + test samples of left_chosen
    dl2_test = list(dl1_test)
    for r in left_chosen:
        for p in files["test"][target_sup]:
            if _rating_from_fname(os.path.basename(p)) == r:
                dl2_test.append((p,label_map[target_sup]))

    # 6) Wrap in IMDbDataset and DataLoaders
    def mk_dl(samples, shuffle):
        return DataLoader(
            IMDbDataset(samples),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

    dl1_tr   = mk_dl(dl1_tr_samples,  shuffle=True)
    dl1_val  = mk_dl(dl1_val_samples, shuffle=False)
    dl1_test = mk_dl(dl1_test,       shuffle=False)
    dl2_tr   = mk_dl(dl2_tr_samples, shuffle=True)
    dl2_test = mk_dl(dl2_test,      shuffle=False)

    # mimic your old stage1_samples return
    stage1_samples = [(p,label_map["neg"],"member") for p,l in dl1_tr_samples if l==0] + \
                     [(p,label_map["pos"],"member") for p,l in dl1_tr_samples if l==1]

    return (dl1_tr, dl1_val, dl1_test,
            dl2_tr, dl2_test,
            len(sup_list), sup_list,
            stage1_samples, _TOKENIZER)

# ───────────── log‑logit‑gap dumper for BERT ─────────────────────────
def dump_logit_gaps(model: nn.Module,
                    dl_member: DataLoader,          # ← dl1_tr
                    dl_nonmember: DataLoader,       # ← dl1_val
                    sup_list: List[str],
                    out_csv: str,
                    device,
                    batch_size: int = 256):
    """
    Evaluate BERT on *both* loaders and write a CSV with

        Path, Log_Logit_Gap, Subset ("member"/"nonmember"), Superclass
    """
    # 1) build a combined loader (keep order deterministic)
    member_paths = set(p for p, _ in dl_member.dataset.samples)

    combined_ds  = ConcatDataset([dl_member.dataset, dl_nonmember.dataset])
    loader = DataLoader(
        combined_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    eps   = 1e-8
    rows  = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="logit‑gap", leave=False):
            paths  = batch["path"]
            labels = batch["labels"]
            inputs = {k: v.to(device, non_blocking=True)
                      for k, v in batch.items()
                      if k not in ("labels", "path")}

            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()
            top2  = -np.sort(-probs, axis=1)[:, :2]
            gap   = np.clip(top2[:, 0] - top2[:, 1], eps, 1 - eps)
            z     = np.log(gap / (1 - gap))

            for p, lbl, zz in zip(paths, labels, z):
                subset = "member" if p in member_paths else "nonmember"
                rows.append([p, float(zz), subset, sup_list[lbl]])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(
            [["Path", "Log_Logit_Gap", "Subset", "Superclass"], *rows]
        )
    print(f"↳ wrote {len(rows):,} rows → {out_csv}")
# ───────────── training helpers ─────────────────────────────
def apply_sd(model,p=0.1):
    for m in model.modules():
        if isinstance(m,DropPath): m.drop_prob=p

# ────────── train / eval loops adapted for HuggingFace inputs ──────────
def train_epoch(model, loader, crit, opt, scaler, device):
    model.train()
    loss_sum = 0.0
    correct  = 0
    total    = 0
    for batch in loader:
        # move everything to device
        labels = batch["labels"].to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True)
                  for k,v in batch.items()
                  if k not in ("labels", "path")}

        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(**inputs).logits
            loss   = crit(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        bs = labels.size(0)
        loss_sum += loss.item() * bs
        total    += bs
        preds     = logits.argmax(dim=1)
        correct  += (preds == labels).sum().item()

    return loss_sum/total, correct/total

def freeze_bert_layers(model: nn.Module, n_unfrozen_layers: int = 2):
    """
    Freeze all BERT parameters *except*
        • classifier head
        • last `n_unfrozen_layers` encoder blocks
    """
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze classifier head
    for p in model.classifier.parameters():
        p.requires_grad = True

    # 3) unfreeze last N encoder blocks
    encoder = model.bert.encoder
    for blk in encoder.layer[-n_unfrozen_layers:]:
        for p in blk.parameters():
            p.requires_grad = True    # includes each block’s LayerNorm


@torch.no_grad()
def evaluate(model, loader, crit, device, verbose=True):
    """
    Returns (avg_loss, accuracy).  If `verbose` is True, also prints:

        • total samples seen
        • #gold‑label 0 / 1
        • #predicted 0 / 1
        • confusion‑matrix counts
    """
    model.eval()
    loss_sum = 0.0
    total    = 0
    correct  = 0

    #   debugging counters
    gold0 = gold1 = pred0 = pred1 = tp = tn = fp = fn = 0

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True)
                  for k, v in batch.items()
                  if k not in ("labels", "path")}

        with autocast():
            logits = model(**inputs).logits
            loss   = crit(logits, labels)

        preds = logits.argmax(dim=1)

        # — aggregate stats —
        total     += labels.size(0)
        loss_sum  += loss.item() * labels.size(0)
        correct   += (preds == labels).sum().item()

        gold0    += (labels == 0).sum().item()
        gold1    += (labels == 1).sum().item()
        pred0    += (preds  == 0).sum().item()
        pred1    += (preds  == 1).sum().item()

        tp       += ((preds == 1) & (labels == 1)).sum().item()
        tn       += ((preds == 0) & (labels == 0)).sum().item()
        fp       += ((preds == 1) & (labels == 0)).sum().item()
        fn       += ((preds == 0) & (labels == 1)).sum().item()

    if verbose:
        print(f"── eval‑debug ───────────────────────────────────────")
        print(f"total samples : {total}")
        print(f"gold 0 / 1    : {gold0} / {gold1}")
        print(f"pred 0 / 1    : {pred0} / {pred1}")
        print(f"confusion     : TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"────────────────────────────────────────────────────")

    if total == 0:
        return float("inf"), 0.0
    return loss_sum / total, correct / total

# ────────── run_cycle ──────────
def run_cycle(label, model, dl_train, dl_eval, device, out_dir, cfg, epochs=3):
    # optional freeze & tiny‐LR for stage2 if you like, kept as is

    if cfg.SGD_New and label.startswith("stage2"):
        print(">> Freezing all but the last 2 encoder blocks + head")
        freeze_bert_layers(model, n_unfrozen_layers=4)
        lr = 1e-7                               # tiny LR for few params
    else:
        lr = cfg.lr


    # use AdamW for BERT
    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg.weight_decay
    )
    sch    = CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.CrossEntropyLoss()
    scaler = make_scaler()

    best, pat = float("inf"), 0
    ckpt      = os.path.join(out_dir, f"{label}.pth")
    os.makedirs(out_dir, exist_ok=True)

    # zero‐epoch eval
    l, a = evaluate(model, dl_eval, crit, device)
    print(f"[{label}] 0/{epochs} eval {l:.4f}|{a*100:.1f}%")

    for ep in range(epochs):
        tr_l, tr_a = train_epoch(model, dl_train, crit, opt, scaler, device)
        ev_l, ev_a = evaluate(model, dl_eval, crit, device)
        sch.step()

        print(f"[{label}] {ep+1}/{epochs} "
              f"train {tr_l:.4f}|{tr_a*100:.1f}% "
              f"eval  {ev_l:.4f}|{ev_a*100:.1f}%")

        if ev_l < best:
            best, pat = ev_l, 0
        else:
            pat += 1
            if pat >= cfg.patience:
                print("  ↳ early stop")
                break

    return ckpt

# ───────────── two‑stage main pipeline for BERT ──────────────────────
def main(cfg):
    # seeds
    random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed); torch.cuda.manual_seed_all(cfg.random_seed)

    # IMDb loaders (dl1_tr, dl1_val, dl1_test, dl2_tr, dl2_test, …)
    (dl1_tr, dl1_val, dl1_test,
     dl2_tr, dl2_test,
     num_sup, sup_list, stage1_samples, tokenizer) = get_dataloaders(
         DATA_DIR, cfg.target_sup, cfg.alpha,
         cfg.rs,     cfg.split_seed, cfg.SGD_New)

    # model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_sup
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    out1 = os.path.join(cfg.out_dir, "stage1_ckpt")
    out2 = os.path.join(cfg.out_dir, "stage2_ckpt")
    os.makedirs(out1, exist_ok=True); os.makedirs(out2, exist_ok=True)

    # ── Stage‑1 ────────────────────────────────────────────────
    print("===== Stage‑1 =====")
    ck1 = run_cycle(f"stage1_BERT_{cfg.random_seed}_{cfg.split_seed}",
                    model, dl1_tr, dl1_test, device, out1, cfg, epochs=5)

    dump_logit_gaps(model,
                dl1_tr, dl1_val,      # member / non‑member
                sup_list,
                out_csv=f"{cfg.out_dir}/obs/target_1_test/"
                        f"BERT_{cfg.random_seed}_{cfg.split_seed}.csv",
                device=device)

    # ── Stage‑2 (continue fine‑tuning) ────────────────────────
    print("===== Stage‑2 =====")
    ck2 = run_cycle(f"stage2_BERT_{cfg.random_seed}_{cfg.split_seed}",
                    model, dl2_tr, dl2_test, device, out2, cfg, epochs=2)

    dump_logit_gaps(model,
                dl1_tr, dl1_val,      # member / non‑member
                sup_list,
                out_csv=f"{cfg.out_dir}/obs/target_2_test/"
                        f"BERT_{cfg.random_seed}_{cfg.split_seed}.csv",
                device=device)

    print("Done. Best checkpoints:", ck1, ck2)

# ───────────── simple integrity test ────────────────────────
def run_test(cfg):
    """
    • Stage‑1: the set of label IDs that appear in dl1_tr must match those in dl1_val
    • Stage‑2: the label set of dl2_tr must be a subset of (or equal to) dl2_test
      (because dl2_test = dl1_test ∪ extra classes)
    """
    import unittest

    class T(unittest.TestCase):
        @classmethod
        def setUpClass(c):
            dls = get_dataloaders(
                DATA_DIR,
                cfg.target_sup,
                cfg.alpha,
                cfg.rs,
                cfg.split_seed
            )
            # unpack just what we need
            (c.dl1_tr,
             c.dl1_val,
             c.dl1_test,
             c.dl2_tr,
             c.dl2_test, *_) = dls

        def _label_set(_, loader):
            s = set()
            for batch in loader:
                s.update(batch["labels"].tolist())
            return s

        def test_stage1_label_match(_):
            _.assertEqual(_. _label_set(_.dl1_tr),
                          _. _label_set(_.dl1_val))

        def test_stage2_subset(_):
            tr2 = _. _label_set(_.dl2_tr)
            te2 = _. _label_set(_.dl2_test)
            _.assertTrue(tr2.issubset(te2))

    unittest.main(argv=["ignored"], exit=False)


# ───────────── CLI ──────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()

    # IMDb‑specific options
    p.add_argument("--target_sup",  choices=["pos", "neg"], required=True,
                   help="Which superclass is considered the 'target' "
                        "for leftover‑class sampling.")
    p.add_argument("--alpha",      type=float, default=0.3,
                   help="Fraction of remaining rating classes to use for Stage‑2.")
    p.add_argument("--rs",         type=int,   default=42,
                   help="Random seed for class selection.")
    p.add_argument("--split_seed", type=int,   default=1,
                   help="Seed used to split Stage‑1 train vs val.")
    p.add_argument("--SGD_New",    action="store_true",
                   help="(kept from original code; controls tiny‑LR + freezing)")

    # generic training hyper‑parameters (can override defaults)
    p.add_argument("--num_epochs",   type=int,   default=5)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience",     type=int,   default=3)
    p.add_argument("--random_seed",  type=int,   default=42)
    p.add_argument("--out_dir",      type=str,   default="/data/bert_outputs")

    # run mode
    p.add_argument("--test", action="store_true",
                   help="Run the integrity unit‑test instead of training.")

    return p.parse_args()


if __name__ == "__main__":
    cfg = parse()
    if cfg.test:
        run_test(cfg)
    else:
        main(cfg)
