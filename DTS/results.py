#!/usr/bin/env python3
# results.py
"""
Decision-Theoretic Selection (DTS) Evaluation Suite
Outputs a LaTeX table with math-mode subscript bootstrap errors.
Uniform Sig Figs: 1 decimal for %, 3 decimals for BAS/ECE.
"""

import os
import json
import argparse
import re
import math
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# --- Theoretical Foundation ---

def score_fn(p):
    """Computes the expected utility for the Behavioral Alignment Score (BAS)."""
    score_eps = 1e-12
    p = np.clip(p, score_eps, 1.0 - score_eps)
    return p + (1.0 - p) * math.log(1.0 - p)

def compute_ece(s_list, y_true, n_bins=10):
    """Calculates the Expected Calibration Error (ECE)."""
    s = np.array(s_list)
    y = np.array(y_true, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(s)
    for i in range(n_bins):
        idx = (s >= bins[i]) & (s < bins[i+1]) if i < n_bins - 1 else (s >= bins[i]) & (s <= bins[i+1])
        if idx.sum() == 0:
            continue
        bin_conf = float(s[idx].mean())
        bin_acc = float(y[idx].mean())
        ece += (idx.sum() / total) * abs(bin_conf - bin_acc)
    return float(ece)

# --- Statistical Robustness ---

def bootstrap_metrics(df, delta, is_raw=False, n_boot=1000):
    """Computes point estimates and 95% CIs via bootstrapping."""
    rng = np.random.RandomState(42)
    boot_stats = []
    
    for _ in range(n_boot):
        sample = df.sample(frac=1.0, replace=True, random_state=rng)
        if is_raw:
            ans = sample
            bas_col = 'raw_realized_bas'
        else:
            ans = sample[sample['dtr_score'] > delta]
            bas_col = 'calib_realized_bas'
            
        cov = len(ans) / len(sample)
        hallu = 1.0 - ans['is_correct'].mean() if len(ans) > 0 else 0.0
        bas = ans[bas_col].mean() if len(ans) > 0 else 0.0
        boot_stats.append({'cov': cov, 'hallu': hallu, 'bas': bas})
    
    bdf = pd.DataFrame(boot_stats)
    res = {}
    for col in ['cov', 'hallu', 'bas']:
        res[col] = (bdf[col].mean(), (bdf[col].quantile(0.975) - bdf[col].quantile(0.025)) / 2)
    return res

# --- Formatting ---

def fmt_latex(stats, is_pct=False):
    """
    Formats values with subscripts for errors wrapped in math mode ($Val_{Err}$).
    1 decimal place for percentages; 3 for scores.
    """
    val, err = stats
    if is_pct:
        return f"${val*100:.1f}_{{{err*100:.1f}}}$"
    return f"${val:.3f}_{{{err:.3f}}}$"

# --- Execution ---

def main(args):
    if not os.path.exists(args.trace_path):
        raise FileNotFoundError(f"Trace file not found: {args.trace_path}")
        
    with open(args.trace_path, "r") as f:
        data = [json.loads(l) for l in f if l.strip()]
    
    test_entries = data[:args.split_idx]
    calib_entries = data[args.split_idx:]
    score_eps = 1e-12

    # 1. Calibration
    s_cal = [float(e.get('reported_s', 0.5)) for e in calib_entries]
    y_cal = [1 if e.get('is_correct', False) else 0 for e in calib_entries]
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_cal, y_cal)
    
    # 2. Evaluation
    results = []
    for e in test_entries:
        s = float(e.get('reported_s', 0.5))
        corr = bool(e.get('is_correct', False))
        hat_p = float(iso.transform([s])[0])
        
        results.append({
            "reported_s": s,
            "hat_p": hat_p,
            "dtr_score": score_fn(hat_p),
            "is_correct": corr,
            "raw_realized_bas": s if corr else s + math.log(max(score_eps, 1-s)),
            "calib_realized_bas": hat_p if corr else hat_p + math.log(max(score_eps, 1-hat_p))
        })

    df = pd.DataFrame(results)
    ece_raw = compute_ece(df['reported_s'], df['is_correct'])
    ece_dts = compute_ece(df['hat_p'], df['is_correct'])

    # 3. Dynamic Operating Point Selection (Regimes)
    ###for simpleQA 
    regime_targets = [
        ("High Coverage", 0.50),
        ("Utility-Optimal", 0.10),
        ("High Precision", 0.02)
    ]

    ###for medQA
    regime_targets = [
         ("High Coverage", 0.80),
         ("Utility-Optimal", 0.40),
         ("High Precision", 0.02)
    ]
    
    report_points = []
    sorted_scores = sorted(df['dtr_score'].unique())
    
    for label, target in regime_targets:
        best_d, best_diff = sorted_scores[0], 1.0
        for d in sorted_scores:
            current_cov = (df['dtr_score'] > d).mean()
            if abs(current_cov - target) < best_diff:
                best_diff, best_d = abs(current_cov - target), d
        report_points.append((label, best_d))

    # 4. Generate LaTeX Table
    print("\n\\begin{table}[t]")
    print("\\centering")
    print("\\small")
    print("\\begin{tabular}{l c c c c}")
    print("\\toprule")
    # Headers updated with escaped % and math arrows
    print(r"Operating Point & Coverage (\%) & Hallu. Rate (\%) $\downarrow$ & Mean BAS $\uparrow$ & ECE $\downarrow$ \\")
    print("\\midrule")
    
    # Row: Raw Model Baseline
    raw_s = bootstrap_metrics(df, delta=None, is_raw=True, n_boot=args.bootstrap)
    print(f"Baseline (Forced) & {fmt_latex(raw_s['cov'], True)} & {fmt_latex(raw_s['hallu'], True)} & {fmt_latex(raw_s['bas'])} & ${ece_raw:.3f}$ \\\\")
    print("\\midrule")
    
    # Rows: DTS Regimes
    for name, delta in report_points:
        stats = bootstrap_metrics(df, delta, is_raw=False, n_boot=args.bootstrap)
        print(f"DTS ({name}) & {fmt_latex(stats['cov'], True)} & {fmt_latex(stats['hallu'], True)} & {fmt_latex(stats['bas'])} & ${ece_dts:.3f}$ \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print(r"\caption{Selective performance sweep on SimpleQA (GPT-oss-120b). Subscripts denote 95\% bootstrap CI error. Percentages (\%) are reported in column headers. DTS allows the model to shift from high-coverage to high-precision expert regimes.}")
    print("\\label{tab:dts_results}")
    print("\\end{table}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace_path", required=True)
    p.add_argument("--split_idx", type=int, default=1000)
    p.add_argument("--bootstrap", type=int, default=1000)
    main(p.parse_args())