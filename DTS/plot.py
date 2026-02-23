#!/usr/bin/env python3
# plot.py
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
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
    plt.rcParams["font.family"] = "serif"
    
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
            "reported_s": s,
            "hat_p": hat_p,
            "dtr_score": score_fn(hat_p),
            "is_correct": int(corr)
        })
    df = pd.DataFrame(results)

    # 4. Setup Output Directory
    model_dir = args.model_name if args.model_name else os.path.basename(args.trace_path).replace('.jsonl','')
    os.makedirs(model_dir, exist_ok=True)

    # --- PLOT 1: CALIBRATION ALIGNMENT ---
    plt.figure(figsize=(7, 6))
    bins = np.linspace(0, 1, 11)
    df['bin_raw'] = pd.cut(df['reported_s'], bins)
    raw_acc = df.groupby('bin_raw', observed=False)['is_correct'].mean()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    dts_stats = df.groupby('hat_p', observed=False).agg({'is_correct': ['mean', 'count']})
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Ideal")
    plt.plot(bin_centers, raw_acc, 'ro-', label="Raw Signal", alpha=0.5, markersize=8, lw=2)
    plt.plot(dts_stats.index, dts_stats[('is_correct', 'mean')], 'bo-', label="DTS (Calibrated)", markersize=10, lw=3)
    plt.title(r"$\bf{Calibration\ Alignment}$", fontsize=20)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Empirical Accuracy")
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "calibration_alignment.pdf"), bbox_inches='tight')
    plt.close()

    # --- PLOT 2: BELIEF DISTRIBUTION SHIFT ---
    plt.figure(figsize=(7, 6))
    sns.kdeplot(df['reported_s'], bw_adjust=0.7, fill=True, color='red', label="Raw Beliefs", alpha=0.15)
    sns.kdeplot(df['hat_p'], bw_adjust=0.7, fill=True, color='blue', label="Calibrated Beliefs", alpha=0.15)
    plt.title(r"$\bf{Belief\ Distribution\ Shift}$", fontsize=20)
    plt.xlabel("Confidence / Probability")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "distribution_shift.pdf"), bbox_inches='tight')
    plt.close()

    # --- PLOT 3: SELECTIVE EXPERTISE FRONTIER (HIGH IMPACT) ---
    plt.figure(figsize=(8, 6))
    
    # Calculate Frontier for Raw vs DTS
    def get_frontier(scores, correctness):
        # Sort by score descending
        idx = np.argsort(scores)[::-1]
        sorted_corr = correctness[idx]
        cum_corr = np.cumsum(sorted_corr)
        coverage = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
        hallu_rate = 1.0 - (cum_corr / np.arange(1, len(sorted_corr) + 1))
        return coverage, hallu_rate

    cov_raw, hallu_raw = get_frontier(df['reported_s'].values, df['is_correct'].values)
    cov_dts, hallu_dts = get_frontier(df['dtr_score'].values, df['is_correct'].values)

    plt.plot(cov_raw, hallu_raw, color='red', alpha=0.4, lw=2, label="Raw Selection")
    plt.plot(cov_dts, hallu_dts, color='purple', lw=4, label="DTS Selection")
    
    # Fill the area to show "expertise gained"
    plt.fill_between(cov_dts, hallu_dts, alpha=0.1, color='purple')

    # Log scale is high impact because it expands the "Expert" regime on the left
    plt.xscale('log')
    plt.xlim(1.0, 0.01) # Show from 100% down to 1% coverage
    plt.xticks([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], ['100%', '50%', '20%', '10%', '5%', '2%', '1%'])
    
    plt.title(r"$\bf{Selective\ Expertise\ Frontier}$", fontsize=20)
    plt.xlabel("Coverage (Log Scale)")
    plt.ylabel("Hallucination Rate")
    plt.legend(frameon=True)
    
    # Add an annotation for the "Expert Regime"
    plt.annotate('Expert Regime', xy=(0.04, 0.05), xytext=(0.02, 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "expertise_frontier.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(model_dir, "expertise_frontier.png"), dpi=300)
    plt.close()

    print(f"Success! High-impact plots saved in: {model_dir}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace_path", required=True)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--split_idx", type=int, default=1000)
    main(p.parse_args())