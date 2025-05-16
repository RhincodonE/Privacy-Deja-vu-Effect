"#!/usr/bin/env python3"
"""
Compute SSIM between stage1 samples and leftover superclass samples.

Args:
    --random_seed   Random seed for sampling
    --superclass_name  Name of the target superclass (must match key in superclasses.json)
    --alpha         Fraction of leftover wnids to sample

Outputs:
    CSV with columns: Path1, Path2, SSIM, Avg_SSIM (only on first occurrence), Superclass
"""
import os
import json
import csv
import random
import argparse
import re
from collections import defaultdict

import numpy as np
import torch
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from tqdm import tqdm
from pytorch_msssim import ssim as torch_ssim

# Constants (adjust as needed)
DATA_DIR = "datasets/tiny-imagenet-200"
IMAGE_SIZE = 384
CHUNK_SIZE = 100  # number of leftover images per GPU batch

WNID_RE = re.compile(r"n\d{8}")

def get_wnid(path: str) -> str:
    m = WNID_RE.search(path)
    if not m:
        raise ValueError(f"Cannot find wnid in {path}")
    return m.group(0)

def build_superclass_maps(data_dir):
    with open(os.path.join(data_dir, "superclasses.json")) as f:
        supercls = json.load(f)
    sup_list = sorted(supercls.keys())
    wnid2sup = {wn: sup for sup, members in supercls.items() for wn in members}
    return supercls, sup_list, wnid2sup

def collect_samples(data_dir):
    train_dir = os.path.join(data_dir, "train")
    samples = []
    for wnid in os.listdir(train_dir):
        wnid_img_dir = os.path.join(train_dir, wnid, "images")
        if not os.path.isdir(wnid_img_dir):
            continue
        for fname in os.listdir(wnid_img_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(wnid_img_dir, fname)
                samples.append(path)
    return samples

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random_seed", type=int, default=522)
    p.add_argument("--superclass_name", type=str, default='instrumentality')
    p.add_argument("--alpha", type=float, default=0.2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.random_seed)
    samples = collect_samples(DATA_DIR)
    supercls, sup_list, wnid2sup = build_superclass_maps(DATA_DIR)

    if args.superclass_name not in supercls:
        raise ValueError(f"Superclass '{args.superclass_name}' not found in superclasses.json")

    members = supercls[args.superclass_name]
    available_wnids = {d for d in os.listdir(os.path.join(DATA_DIR, "train"))
                       if os.path.isdir(os.path.join(DATA_DIR, "train", d))}
    valid_members = [wn for wn in members if wn in available_wnids]

    if len(valid_members) < 3:
        raise ValueError(f"Not enough valid wnids in superclass '{args.superclass_name}'")

    # Choose 2 wnids for stage-1 from target superclass
    chosen = random.sample(valid_members, 2)

    # From each of the other superclasses, sample 2 wnids
    other_stage1_wnids = []
    for other_sup, members in supercls.items():
        if other_sup == args.superclass_name:
            continue
        candidates = [wn for wn in members if wn in available_wnids]
        if len(candidates) >= 2:
            other_stage1_wnids.extend(random.sample(candidates, 2))

    # Leftover wnids from target superclass
    left = [wn for wn in valid_members if wn not in chosen]
    leftover_pool = [p for p in samples if get_wnid(p) in left]
    k = max(1, int(args.alpha * len(leftover_pool)))
    leftover_paths = random.sample(leftover_pool, k)

    # Collect stage-1 sample paths (target + others)
    stage1_wnids = set(chosen + other_stage1_wnids)
    ts1 = [p for p in samples if get_wnid(p) in stage1_wnids]

    print(f"[INFO] Found {len(ts1)} stage-1 samples")
    print(f"[INFO] Found {len(leftover_paths)} leftover samples")

    if not ts1 or not leftover_paths:
        raise RuntimeError("No valid samples found for stage-1 or leftover set. Aborting.")

    # SSIM transform: resize and to tensor (0-1)
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([resize, to_tensor])

    def load_tensor(path):
        img = default_loader(path)
        return transform(img).unsqueeze(0).to(device)  # shape (1, C, H, W)

    # Preload leftover tensors (on CPU to reduce GPU usage)
    left_tensors = [transform(default_loader(p)).unsqueeze(0) for p in tqdm(leftover_paths, desc="Preloading Leftover")]

    dl1_imgs = [(p, wnid2sup[get_wnid(p)]) for p in ts1]
    avg_map = {}
    ssim_records = defaultdict(list)

    for path1, sup in tqdm(dl1_imgs, desc="Computing SSIM"):
        img1 = load_tensor(path1)  # shape (1, C, H, W)
        vals = []
        with torch.no_grad():
            for i in range(0, len(left_tensors), CHUNK_SIZE):
                batch = torch.cat(left_tensors[i:i+CHUNK_SIZE], dim=0).to(device)
                img1_batch = img1.expand(batch.size(0), -1, -1, -1)
                v = torch_ssim(img1_batch, batch, data_range=1.0, size_average=False)
                vals.extend(v.cpu().tolist())
                del batch, img1_batch, v  # free memory
                torch.cuda.empty_cache()

        for path2, val in zip(leftover_paths, vals):
            ssim_records[path1].append((path2, val))
        avg_map[path1] = float(np.mean(vals))

    out_csv = os.path.join('/result/',
                           f"ssim_{args.superclass_name}_{args.random_seed}_{args.alpha:.2f}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Path1", "Path2", "SSIM", "Avg_SSIM", "Superclass"])
        for path1, records in ssim_records.items():
            first = True
            sup = wnid2sup[get_wnid(path1)]
            for path2, val in records:
                avg = avg_map[path1] if first else ""
                writer.writerow([path1, path2, f"{val:.6f}", avg, sup])
                first = False

    print(f"Wrote SSIM results to {out_csv}")
