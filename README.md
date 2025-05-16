```markdown
# Two-Stage Fine-Tuning & Privacy Analysis Pipeline

This repository implements end-to-end experiments for two-stage, superclass-aware fine-tuning of both ViT and BERT, followed by a suite of privacy-risk analyses:

1. **Log-Logit-Gap CDFs & Δ-Scores**  
2. **Per-Sample AUC-Style Summaries & Stage-2 vs Stage-1 Comparison**  
3. **Sample–Sample SSIM Matrix**  
4. **NTK Mean Similarity (Old vs New)**  
5. **Per-Probe NTK Pairwise Similarity**  

---

## 📦 Repository Layout

.
├── run_vit/               # orchestrates ViT experiments
├── fine_tune_vit.py       # two-stage ViT fine-tuning + log-logit-gap export
├── score_vit.py           # per-sample AUC summary & comparison
├── compute_ssim.py        # SSIM between Stage-1 and leftover samples
├── two_stage_ntk.py       # NTK mean similarity (old vs new)
├── ntk_sample_pairs.py    # per-probe NTK pairwise similarity
└── datasets/              # Tiny-ImageNet-200 & IMDb preprocessed data

---

## 🔧 Prerequisites

- Python 3.8+  
- PyTorch 1.10+, `torchvision`  
- `timm`, `transformers`, `pytorch_msssim`, `tqdm`, `pandas`, `numpy`

Install via:

```bash
pip install torch torchvision timm transformers pytorch-msssim tqdm pandas numpy
````

---

## 1. ViT Pipeline (`run_vit/`)

### 1.1 `fine_tune_vit.py`

* **Two-Stage ViT-B/16(384) fine-tuning** on Tiny-ImageNet-200 with superclass-aware splits
* **Log-logit-gap export**: CSV columns `Path, Log_Logit_Gap, Subset, Superclass`
* **Integrity tests** (`--test`):

  * Stage 1 train/val share labels
  * Stage 2 train ⊆ Stage 2 val

**Usage:**

```bash
# In the run_vit folder:
bash run_vit.sh  
# Or manually:
python fine_tune_vit.py \
  --superclass_name living_thing \
  --alpha 0.5 \
  --random_seed 42 \
  --split_seed 35 \
  --SGD_New \
  --out_dir ./outputs
```

**Outputs:**

```
./outputs/models_subpop/target_1_test/*.pth  
./outputs/models_subpop/target_2_test/*.pth  

./outputs/obs/target_1_test/ViT_<seed>_<split>.csv  
./outputs/obs/target_2_test/ViT_<seed>_<split>.csv  
```

### 1.2 `score_vit.py`

* **Per-sample AUC-style summaries**: find TPR/FPR threshold maximizing TPR/FPR
* **Stage-2 vs Stage-1 comparison**: Δ = Score₂ − Score₁

**Usage:**

```bash
python score_vit.py \
  --obs_dir ./outputs/obs \
  --result_dir ./outputs/result
```

**Outputs:**

```
./outputs/result/target_1_test.csv  
./outputs/result/target_2_test.csv  
./outputs/result/comparison_target2_minus_target1.csv  
```

---

## 2. SSIM Matrix (`compute_ssim.py`)

Compute **structural similarity (SSIM)** between each Stage 1 image and a sampled leftover set.

**Usage:**

```bash
python compute_ssim.py \
  --random_seed 522 \
  --superclass_name instrumentality \
  --alpha 0.20
```

* **Stage 1**: 2 WNIDs per superclass
* **Leftovers**: α-fraction of remaining target-class images
* **Output:** `/result/ssim_<superclass>_<seed>_<alpha>.csv`
  Columns: `Path1, Path2, SSIM, Avg_SSIM, Superclass`

---

## 3. NTK Mean Similarity (`two_stage_ntk.py`)

After Stage 1 fine-tuning, compute **mean NTK similarity** between each old Sample and all new Samples.

**Usage:**

```bash
python two_stage_ntk.py \
  --target_sup instrumentality \
  --alpha 0.3 \
  --random_seed 522 \
  --split_seed 35 \
  --SGD_New
```

* **Export:**
  `/data/ntk_mean_similarities/ntk_mean_sim_<sup>_<seed>_<alpha>.csv`
  Columns: `Path, Superclass, Mean_NTK_Similarity`

---

## 4. NTK Pairwise Similarity (`ntk_sample_pairs.py`)

Compute **pairwise NTK** values between a selected subset of Stage 1 probes (e.g., top 10 Δ and bottom 10 Δ) and all Stage 2 samples. Please include all samples to stage 2 set.

**Usage:**

```bash
python ntk_sample_pairs.py \
  --superclass_name instrumentality \
  --alpha 0.4 \
  --random_seed 522 \
  --split_seed 1 \
  --SGD_New
```

Edit the `top10_paths`/`bottom10_paths` block in the script to specify your probes.
**Output:** `/data/ntk_persample_sim_<sup>_<seed>_<alpha>.csv`
Columns: `Stage1_Path, Stage2_Path, NTK_Value`

---

## 📄 License & Citation

*Happy experimenting!*

```
```
