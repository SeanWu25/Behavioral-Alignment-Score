#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

# --- Theoretical Helpers ---
def score_fn(p):
    """Expected utility: p + (1-p)ln(1-p)."""
    score_eps = 1e-12
    p = np.clip(p, score_eps, 1.0 - score_eps)
    return p + (1.0 - p) * np.log(p)

def main(args):
    # Set NeurIPS-style aesthetics
    sns.set_theme(style="white", context="paper", font_scale=1.8)
    plt.rcParams["font.family"] = "serif"
    
    COLOR_NAVY = "#1B4F72"     # Deep Professional Navy
    COLOR_RED = "#CB4335"      # Sophisticated Red
    
    # 1. Load Data
    if not os.path.exists(args.trace_path):
        raise FileNotFoundError(f"Trace file not found: {args.trace_path}")
        
    with open(args.trace_path, "r") as f:
        data = [json.loads(l) for l in f if l.strip()]
    
    test_entries = data[:args.split_idx]
    calib_entries = data[args.split_idx:]

    # 2. Calibration
    s_cal = [float(e.get('reported_s', 0.5)) for e in calib_entries]
    y_cal = [1 if e.get('is_correct', False) else 0 for e in calib_entries]
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_cal, y_cal)
    
    # 3. Process Test Set
    results = []
    for e in test_entries:
        s = float(e.get('reported_s', 0.5))
        corr = bool(e.get('is_correct', False))
        hat_p = float(iso.transform([s])[0])
        results.append({
            "s": s, "p": hat_p, "score": score_fn(hat_p), 
            "is_correct": int(corr), "Status": "Correct" if corr else "Hallucination"
        })
    df = pd.DataFrame(results)

    model_dir = args.model_name if args.model_name else os.path.basename(args.trace_path).replace('.jsonl','')
    os.makedirs(model_dir, exist_ok=True)

    # --- PLOT 1: PROBABILITY ALIGNMENT ---
    plt.figure(figsize=(7, 6))
    dts_stats = df.groupby('p', observed=False).agg({'is_correct': ['mean', 'count']})
    plt.plot([0, 1], [0, 1], color="gray", linestyle='--', alpha=0.3)
    plt.plot(dts_stats.index, dts_stats[('is_correct', 'mean')], marker='o', 
             color=COLOR_NAVY, markerfacecolor=COLOR_RED, markeredgecolor='white',
             markersize=10, lw=3, label="Calibrated Probability")
    plt.title(r"$\mathbf{Probability\ Alignment}$", fontsize=20)
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    sns.despine(); plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "1_alignment.pdf"))

    # --- PLOT 2: SELECTIVE EXPERTISE (ERROR VS COVERAGE) ---
    plt.figure(figsize=(8, 6))
    def get_frontier(scores, correctness):
        idx = np.argsort(scores)[::-1]
        sorted_corr = correctness[idx]
        coverage = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
        hallu_rate = 1.0 - (np.cumsum(sorted_corr) / np.arange(1, len(sorted_corr) + 1))
        return coverage, hallu_rate

    cov, hallu = get_frontier(df['score'].values, df['is_correct'].values)
    plt.plot(cov, hallu, color=COLOR_NAVY, lw=4)
    plt.fill_between(cov, hallu, alpha=0.1, color=COLOR_NAVY)
    plt.xscale('log'); plt.xlim(1.0, 0.01)
    plt.xticks([1.0, 0.1, 0.01], ['100%', '10%', '1%'])
    plt.title(r"$\mathbf{Selective\ Expertise\ Frontier}$", fontsize=20)
    plt.xlabel("Coverage (Log Scale)"); plt.ylabel("Hallucination Rate")
    sns.despine(); plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "2_expertise.pdf"))

    # --- PLOT 3: KNOWLEDGE SEPARATION (THE UNMASKING) ---
    # We show how Correct and Incorrect answers are separated before vs after
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Left: Raw (Entangled)
    sns.kdeplot(data=df[df['is_correct']==1], x='s', fill=True, color=COLOR_NAVY, ax=axes[0], label="Correct", alpha=0.3)
    sns.kdeplot(data=df[df['is_correct']==0], x='s', fill=True, color=COLOR_RED, ax=axes[0], label="Hallucination", alpha=0.3)
    axes[0].set_title(r"$\mathbf{Raw\ Beliefs\ (Entangled)}$")
    axes[0].set_xlabel("Confidence"); axes[0].set_xlim(0, 1)
    
    # Right: Calibrated (Separated)
    sns.kdeplot(data=df[df['is_correct']==1], x='p', fill=True, color=COLOR_NAVY, ax=axes[1], label="Correct", alpha=0.3)
    sns.kdeplot(data=df[df['is_correct']==0], x='p', fill=True, color=COLOR_RED, ax=axes[1], label="Hallucination", alpha=0.3)
    axes[1].set_title(r"$\mathbf{DTS\ Aligned\ (Separated)}$")
    axes[1].set_xlabel("Probability"); axes[1].set_xlim(0, 1)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(); plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "3_separation_unmasking.pdf"))
    plt.savefig(os.path.join(model_dir, "3_separation_unmasking.png"), dpi=300)

    print(f"Success! Unmasking plots saved in: {model_dir}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace_path", required=True)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--split_idx", type=int, default=1000)
    main(p.parse_args())