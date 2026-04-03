#!/usr/bin/env python3
"""
calibrated_simpleqa.py
──────────────────────
Apply isotonic calibration to SimpleQA model confidence scores using a
dev (calibration) set, then evaluate the test set with both raw and
calibrated metrics.

Usage:
    python calibrated_simpleqa.py \
        --dev_path  output_simpleqa_last_1000_colm/simpleqa_gpt-4o-mini_results.jsonl \
        --test_path output_simpleqa_colm/simpleqa_gpt-4o-mini_results.jsonl

            python calibrated_simpleqa.py \
        --dev_path  output_simpleqa_last_1000_colm/simpleqa_llama-3_3-70b-instruct_results.jsonl \
        --test_path output_simpleqa_colm/simpleqa_llama-3_3-70b-instruct_results.jsonl
"""

import os
import json
import argparse
import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# ── Numerical constants ──────────────────────────────────────────────
EPS = 1e-4          # same clamping as simpleqa.py


# ── Core metric functions (ported from get_results.py) ───────────────

def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 5000,
    ci: float = 95.0,
    seed: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Compute bootstrap mean and half-width of CI."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()

    mean = values.mean()
    lower = np.percentile(boot_means, (100.0 - ci) / 2.0)
    upper = np.percentile(boot_means, 100.0 - (100.0 - ci) / 2.0)
    half_width = (upper - lower) / 2.0
    return mean, half_width, lower, upper


def expected_calibration_error(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE as a proportion (0-1)."""
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if confidences.size == 0:
        return np.nan
    if confidences.max() > 1.0:
        confidences = confidences / 100.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        upper_edge = bins[i + 1] if i < n_bins - 1 else 1.0001
        mask = (confidences >= bins[i]) & (confidences < upper_edge)
        bin_count = mask.sum()
        if bin_count > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = labels[mask].mean()
            ece += (bin_count / n) * abs(bin_conf - bin_acc)
    return ece


def aurc_from_confidence(
    confidences: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Area Under Risk-Coverage Curve."""
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if confidences.size == 0:
        return np.nan
    if confidences.max() > 1.0:
        confidences = confidences / 100.0

    order = np.argsort(-confidences)
    sorted_labels = labels[order]
    n = len(sorted_labels)
    cumsum_correct = np.cumsum(sorted_labels)
    ks = np.arange(1, n + 1)
    coverage = ks / n
    risk = 1.0 - (cumsum_correct / ks)
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.0 compat
    return float(_trapz(risk, coverage))


# ── BAS score (from simpleqa.py) ─────────────────────────────────────

def calculate_per_example_bas(is_correct: bool, s: float) -> float:
    """Per-example BAS: correct → s, incorrect → s + ln(1−s)."""
    if is_correct:
        return s
    s = max(EPS, min(1 - EPS, s))
    return s + math.log(1 - s)


# ── Metric bundle (computes everything for a set of conf/labels) ─────

def compute_all_metrics(
    confidences: np.ndarray,
    labels: np.ndarray,
    bas_scores: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 12345,
    ece_bins: int = 10,
) -> dict:
    """
    Return a dict with Accuracy, BAS, ECE, AURC   (point + CI).
    """
    conf = np.asarray(confidences, dtype=float)
    lab = np.asarray(labels, dtype=float)
    bas = np.asarray(bas_scores, dtype=float)

    # 1. Accuracy (%)
    acc_vals = lab * 100.0
    acc_m, acc_h, _, _ = bootstrap_mean_ci(acc_vals, n_bootstrap, 95.0, seed)

    # 2. BAS
    bas_m, bas_h, _, _ = bootstrap_mean_ci(bas, n_bootstrap, 95.0, seed + 1)

    # 3. ECE (%) — point estimate + bootstrap CI
    ece_val = expected_calibration_error(conf, lab, n_bins=ece_bins) * 100.0
    rng = np.random.default_rng(seed + 2)
    b_ece, b_aurc = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(conf), size=len(conf))
        b_ece.append(expected_calibration_error(conf[idx], lab[idx], n_bins=ece_bins) * 100.0)
        b_aurc.append(aurc_from_confidence(conf[idx], lab[idx]))
    ece_h = (np.percentile(b_ece, 97.5) - np.percentile(b_ece, 2.5)) / 2.0

    # 4. AURC
    aurc_val = aurc_from_confidence(conf, lab)
    aurc_h = (np.percentile(b_aurc, 97.5) - np.percentile(b_aurc, 2.5)) / 2.0

    return {
        "Accuracy (%)": (acc_m, acc_h),
        "BAS":          (bas_m, bas_h),
        "ECE (%)":      (ece_val, ece_h),
        "AURC":         (aurc_val, aurc_h),
    }


def fmt(val: float, hw: float, decimals: int = 1) -> str:
    """Pretty-print  value ± half-width."""
    return f"{val:.{decimals}f} ± {hw:.{decimals}f}"


# ── Main ─────────────────────────────────────────────────────────────

def main(args):
    # ─── 1. Load data ───────────────────────────────────────────
    with open(args.dev_path) as f:
        dev_data = [json.loads(line) for line in f if line.strip()]
    with open(args.test_path) as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    print(f"Dev  set : {len(dev_data):,} samples  ({args.dev_path})")
    print(f"Test set : {len(test_data):,} samples  ({args.test_path})")

    # ─── 2. Fit isotonic calibration on dev set ─────────────────
    dev_conf = np.array([float(e.get("confidence", 0.5)) for e in dev_data])
    dev_corr = np.array([int(e.get("is_correct", False)) for e in dev_data])

    iso = IsotonicRegression(out_of_bounds="clip").fit(dev_conf, dev_corr)
    print(f"\nIsotonic calibration fitted on {len(dev_conf)} dev examples.")

    # ─── 3. Extract test set raw values ─────────────────────────
    test_conf_raw = np.array([float(e.get("confidence", 0.5)) for e in test_data])
    test_labels   = np.array([int(e.get("is_correct", False)) for e in test_data])
    test_bas_raw  = np.array([float(e.get("bas_score", 0.0)) for e in test_data])

    # ─── 4. Calibrate test confidences ──────────────────────────
    test_conf_cal = iso.transform(test_conf_raw)

    # Recompute BAS with calibrated confidences
    test_bas_cal = np.array([
        calculate_per_example_bas(bool(c), float(p))
        for c, p in zip(test_labels, test_conf_cal)
    ])

    # ─── 5. Compute metrics ─────────────────────────────────────
    raw_metrics = compute_all_metrics(
        test_conf_raw, test_labels, test_bas_raw,
        n_bootstrap=args.n_bootstrap, seed=args.seed, ece_bins=args.ece_bins,
    )
    cal_metrics = compute_all_metrics(
        test_conf_cal, test_labels, test_bas_cal,
        n_bootstrap=args.n_bootstrap, seed=args.seed, ece_bins=args.ece_bins,
    )

    # ─── 6. Print comparison table ──────────────────────────────
    header = f"{'Metric':<16}  {'Before Calibration':>22}  {'After Calibration':>22}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for key in raw_metrics:
        rv, rh = raw_metrics[key]
        cv, ch = cal_metrics[key]
        dec = 3 if key in ("BAS", "AURC") else 1
        print(f"{key:<16}  {fmt(rv, rh, dec):>22}  {fmt(cv, ch, dec):>22}")
    print(sep)

    # ─── 7. Print LaTeX table ───────────────────────────────────
    model_name = os.path.basename(args.test_path).replace("_results.jsonl", "").replace("simpleqa_", "").replace("_", " ").title()
    print(f"\n--- LATEX TABLE ---")
    print("\\begin{table}[t]")
    print("\\centering\\small")
    print("\\begin{tabular}{l c c c c}")
    print("\\toprule")
    print(r"Setting & Accuracy (\%) & BAS & ECE (\%) $\downarrow$ & AURC $\downarrow$ \\")
    print("\\midrule")
    for label, metrics in [("Before Calib.", raw_metrics), ("After Calib.", cal_metrics)]:
        acc_v, acc_h = metrics["Accuracy (%)"]
        bas_v, bas_h = metrics["BAS"]
        ece_v, ece_h = metrics["ECE (%)"]
        aurc_v, aurc_h = metrics["AURC"]
        print(
            f"{label} & "
            f"${acc_v:.1f} \\pm {acc_h:.1f}$ & "
            f"${bas_v:.3f} \\pm {bas_h:.3f}$ & "
            f"${ece_v:.1f} \\pm {ece_h:.1f}$ & "
            f"${aurc_v:.3f} \\pm {aurc_h:.3f}$ \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Isotonic calibration results for {model_name} on SimpleQA. "
          f"Dev set: {len(dev_data)} samples, Test set: {len(test_data)} samples.}}")
    print("\\label{tab:calibration}")
    print("\\end{table}")

    # ─── 8. Save calibrated results ─────────────────────────────
    out_dir = os.path.dirname(args.test_path) or "."
    base = os.path.basename(args.test_path).replace(".jsonl", "")
    cal_jsonl_path = os.path.join(out_dir, f"{base}_calibrated.jsonl")
    cal_csv_path   = os.path.join(out_dir, f"{base}_calibrated.csv")

    cal_records = []
    for i, e in enumerate(test_data):
        rec = dict(e)
        rec["confidence_raw"] = float(test_conf_raw[i])
        rec["confidence"]     = float(test_conf_cal[i])
        rec["bas_score_raw"]  = float(test_bas_raw[i])
        rec["bas_score"]      = float(test_bas_cal[i])
        cal_records.append(rec)

    with open(cal_jsonl_path, "w") as f:
        for r in cal_records:
            f.write(json.dumps(r) + "\n")

    pd.DataFrame(cal_records).to_csv(cal_csv_path, index=False)

    print(f"\nSaved calibrated JSONL → {cal_jsonl_path}")
    print(f"Saved calibrated CSV  → {cal_csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Apply isotonic calibration to SimpleQA confidence scores"
    )
    p.add_argument("--dev_path",  required=True, help="JSONL with dev/calibration results")
    p.add_argument("--test_path", required=True, help="JSONL with test results to calibrate")
    p.add_argument("--n_bootstrap", type=int, default=5000, help="Bootstrap iterations")
    p.add_argument("--seed",       type=int, default=12345, help="Random seed")
    p.add_argument("--ece_bins",   type=int, default=10,    help="Number of ECE bins")
    main(p.parse_args())
