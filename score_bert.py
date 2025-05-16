#!/usr/bin/env python3
import os, glob, argparse, numpy as np, pandas as pd

# ─────── CLI ──────────────────────────────────────
def parse():
    p = argparse.ArgumentParser(
        description="Summarise BERT log‑logit‑gap csvs and create Δ‑comparison")
    p.add_argument(
        "--out_dir",
        default="data/bert_outputs",
        help="The SAME root directory you passed to training; "
             "must contain an 'obs' sub‑folder."
    )
    return p.parse_args()

# ─────── constants ────────────────────────────────
TARGETS = ["target_1_test", "target_2_test"]

# ─────── per‑target summariser ─────────────────────
def process_one(obs_dir, result_dir, target):
    in_dir  = os.path.join(obs_dir, target)
    out_csv = os.path.join(result_dir, f"{target}.csv")
    os.makedirs(result_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "BERT_*.csv")))
    if not files:
        print(f"[{target}] ⚠️  no BERT_*.csv found in {in_dir} – skip.")
        return False

    dfs = []
    for fn in files:
        df = pd.read_csv(fn,
                         usecols=["Path", "Log_Logit_Gap", "Subset", "Superclass"])
        df["gap"]    = pd.to_numeric(df["Log_Logit_Gap"], errors="coerce")
        df           = df[np.isfinite(df["gap"])]
        df["member"] = df["Subset"].eq("member")
        dfs.append(df[["Path", "gap", "member", "Superclass"]])

    big = pd.concat(dfs, ignore_index=True)
    print(f"[{target}] cleaned rows = {len(big):,}")

    records = []
    for path, grp in big.groupby("Path", sort=False):
        gaps = grp["gap"].values
        labs = grp["member"].values
        P, N = labs.sum(), len(labs) - labs.sum()

        if P == 0 or N == 0:
            records.append((path, np.nan, np.nan, np.nan, grp["Superclass"].iat[0]))
            continue

        best_score, best_TPR, best_FPR = -np.inf, 0.0, 0.0
        for tau in np.unique(gaps):
            pred = gaps > tau
            TP, FP = np.sum(pred & labs), np.sum(pred & (~labs))
            TPR, FPR = TP / P, FP / N
            if FPR and (TPR / FPR) > best_score:
                best_score, best_TPR, best_FPR = TPR / FPR, TPR, FPR

        records.append((path, best_TPR, best_FPR,
                        None if best_score == -np.inf else best_score,
                        grp["Superclass"].iat[0]))

    pd.DataFrame(
        records,
        columns=["Path", "TPR", "FPR", "Score", "Superclass"]
    ).to_csv(out_csv, index=False)
    print(f"[{target}] wrote {out_csv}")
    return True

# ─────── merge Stage‑1 & Stage‑2 ──────────────────
def merge_two(result_dir):
    csv1 = os.path.join(result_dir, f"{TARGETS[0]}.csv")
    csv2 = os.path.join(result_dir, f"{TARGETS[1]}.csv")
    if not (os.path.exists(csv1) and os.path.exists(csv2)):
        print("⚠️  One of the summary csvs is missing – cannot merge.")
        return

    out  = os.path.join(result_dir, "comparison_target2_minus_target1.csv")

    df1 = pd.read_csv(csv1).rename(columns={"TPR":"TPR_1","FPR":"FPR_1","Score":"Score_1"})
    df2 = pd.read_csv(csv2).rename(columns={"TPR":"TPR_2","FPR":"FPR_2","Score":"Score_2"})

    merged = pd.merge(df1, df2, on=["Path","Superclass"],
                      how="inner", validate="one_to_one")
    merged["Delta"] = merged["Score_2"] - merged["Score_1"]

    cols = ["Path","TPR_1","FPR_1","Score_1",
            "TPR_2","FPR_2","Score_2","Delta","Superclass"]
    merged[cols].to_csv(out, index=False)
    print(f"[Δ] wrote {out}  ({len(merged)} paths)")

# ─────── entry point ─────────────────────────────
if __name__ == "__main__":
    cfg = parse()

    OBS_DIR    = os.path.join(cfg.out_dir, "obs")
    RESULT_DIR = os.path.join(cfg.out_dir, "result")

    ok = False
    for tgt in TARGETS:
        ok |= process_one(OBS_DIR, RESULT_DIR, tgt)

    if ok:
        merge_two(RESULT_DIR)
