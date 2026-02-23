import os
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.isotonic import IsotonicRegression

def compute_ece(s_list, y_true, n_bins=10):
    s = np.array(s_list)
    y = np.array(y_true, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (s >= bins[i]) & (s < bins[i+1]) if i < n_bins-1 else (s >= bins[i]) & (s <= bins[i+1])
        if idx.sum() == 0: continue
        ece += (idx.sum() / len(s)) * abs(s[idx].mean() - y[idx].mean())
    return ece

def fit_and_calibrate(train_df, test_df):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_df['reported_s'], train_df['is_correct'])
    hat_p = iso.transform(test_df['reported_s'])
    hat_p = np.clip(hat_p, 1e-12, 1.0 - 1e-12)
    scores = hat_p + (1.0 - hat_p) * np.log(1.0 - hat_p)
    return hat_p, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", required=True)
    args = parser.parse_args()

    with open(args.trace_path, 'r') as f:
        rows = [json.loads(line) for line in f]
    
    test_df = pd.DataFrame(rows[:1000])
    cal_pool = pd.DataFrame(rows[1000:])
    
    # Baseline: Before any calibration [cite: 188]
    raw_s = test_df['reported_s'].values
    y_true = test_df['is_correct'].values.astype(bool)
    ece_initial = compute_ece(raw_s, y_true)

    cal_sizes = [0, 50, 100, 250, 500, 1000]
    results = []

    for n in cal_sizes:
        if n == 0:
            hat_p, scores = raw_s, raw_s + (1.0 - raw_s) * np.log(np.clip(1.0 - raw_s, 1e-12, 1.0))
            ece_val = ece_initial
        else:
            hat_p, scores = fit_and_calibrate(cal_pool.iloc[:n], test_df)
            ece_val = compute_ece(hat_p, y_true)
        
        for name, delta in [("UO (δ=0)", 0.0), ("HP (δ=-0.1)", -0.1)]:
            mask = scores > delta
            cov = (np.sum(mask) / len(test_df)) * 100
            halluc = (1 - np.mean(y_true[mask])) * 100 if np.sum(mask) > 0 else 0
            bas = np.mean(np.where(mask, np.where(y_true, hat_p, hat_p + np.log(1-hat_p)), 0))
            
            results.append({
                "N_cal": n,
                "Regime": name,
                "ECE": ece_val,
                "Coverage %": cov,
                "Hallu %": halluc,
                "Mean BAS": bas
            })

    df_out = pd.DataFrame(results)
    pivot = df_out.pivot(index="N_cal", columns="Regime", values=["ECE", "Hallu %", "Mean BAS"])
    print("\n### REFINED ABLATION: CALIBRATION EFFICIENCY ###")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()