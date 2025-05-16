#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd

# ───────────── CONFIG ────────────────────────────
OBS_DIR    = "/data/obs"
RESULT_DIR = "/result"
TARGETS    = ["target_1_test", "target_2_test"]

def process_one(target):
    """Read all ViT_*.csv in OBS_DIR/target, compute per-path best TPR/FPR score."""
    in_dir  = os.path.join(OBS_DIR, target)
    out_csv = os.path.join(RESULT_DIR, f"{target}.csv")
    os.makedirs(RESULT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "ViT_*.csv")))
    if not files:
        print(f"[{target}] no files found, skipping.")
        return

    dfs = []
    for fn in files:
        df = pd.read_csv(fn, usecols=["Path","Log_Logit_Gap","Subset","Superclass"])
        # coerce gaps, drop non-finite
        df["gap"] = pd.to_numeric(df["Log_Logit_Gap"], errors="coerce")
        df = df[np.isfinite(df["gap"])]
        df["member"] = (df["Subset"] == "member")
        dfs.append(df[["Path","gap","member","Superclass"]])

    big = pd.concat(dfs, ignore_index=True)
    print(f"[{target}] total cleaned rows: {len(big):,}")

    records = []
    for path, grp in big.groupby("Path", sort=False):
        gaps = grp["gap"].values
        labs = grp["member"].values
        P = labs.sum()
        N = len(labs) - P

        if P==0 or N==0:
            records.append((path, np.nan, np.nan, np.nan, grp["Superclass"].iat[0]))
            continue

        best_score = -np.inf
        best_TPR = best_FPR = 0.0

        for tau in np.unique(gaps):
            pred = gaps > tau
            TP   = np.sum(pred & labs)
            FP   = np.sum(pred & (~labs))
            TPR  = TP / P
            FPR  = FP / N
            if FPR > 0:
                score = TPR / FPR
                if score > best_score:
                    best_score, best_TPR, best_FPR = score, TPR, FPR

        if best_score <= -np.inf:
            best_score = np.nan

        records.append((path, best_TPR, best_FPR, best_score, grp["Superclass"].iat[0]))

    out_df = pd.DataFrame(records,
         columns=["Path","TPR","FPR","Score","Superclass"])
    out_df.to_csv(out_csv, index=False)
    print(f"[{target}] wrote {out_csv} ({len(out_df)} paths)")

def merge_two():
    """Merge the two summary CSVs into a third with Score_1, Score_2, Delta."""
    csv1 = os.path.join(RESULT_DIR, f"{TARGETS[0]}.csv")
    csv2 = os.path.join(RESULT_DIR, f"{TARGETS[1]}.csv")
    out  = os.path.join(RESULT_DIR, "comparison_target2_minus_target1.csv")

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # rename scores
    df1 = df1.rename(columns={"TPR":"TPR_1", "FPR":"FPR_1", "Score":"Score_1"})
    df2 = df2.rename(columns={"TPR":"TPR_2", "FPR":"FPR_2", "Score":"Score_2"})

    # merge on Path and Superclass
    merged = pd.merge(df1, df2,
                      on=["Path","Superclass"],
                      how="inner",
                      validate="one_to_one")

    # compute Delta = Score_2 - Score_1
    merged["Delta"] = merged["Score_2"] - merged["Score_1"]

    # select and reorder columns
    result = merged[[
        "Path",
        "TPR_1", "FPR_1", "Score_1",
        "TPR_2", "FPR_2", "Score_2",
        "Delta",
        "Superclass"
    ]]

    result.to_csv(out, index=False)
    print(f"Wrote comparison → {out} ({len(result)} paths)")

if __name__ == "__main__":
    # 1) produce per‐target summaries
    for tgt in TARGETS:
        process_one(tgt)
    # 2) produce merged "2-1" comparison
    merge_two()
