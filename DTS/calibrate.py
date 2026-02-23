
import os
import json
import argparse
import re
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

OUTPUT_DIR = "output_aur_simpleqa"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCORE_EPS = 1e-12

def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def extract_reported_s(entry):
    for key in ("reported_s", "s_bar", "reported_score", "confidence", "reported_confidence"):
        if key in entry:
            try:
                return float(entry[key])
            except:
                pass
    if "candidate" in entry and isinstance(entry["candidate"], dict):
        return float(entry["candidate"].get("confidence", 0.5))
    if "candidates" in entry and isinstance(entry["candidates"], list) and len(entry["candidates"])>0:
        top = max(entry["candidates"], key=lambda x: float(x.get("confidence", 0.5)))
        return float(top.get("confidence", 0.5))
    raise KeyError("Could not find reported score in trace entry.")

def extract_is_correct(entry):
    if "is_correct" in entry:
        return bool(entry["is_correct"])
    raise KeyError("Trace entry missing 'is_correct' field. Please ensure raw trace includes judge labels.")

def score_fn(p):
    p = np.clip(p, SCORE_EPS, 1.0 - SCORE_EPS)
    return p + (1.0 - p) * np.log(1.0 - p)

def compute_ece(s_list, y_true, n_bins=10):
    s = np.array(s_list)
    y = np.array(y_true, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(s)
    for i in range(n_bins):
        if i < n_bins - 1:
            idx = (s >= bins[i]) & (s < bins[i+1])
        else:
            idx = (s >= bins[i]) & (s <= bins[i+1])
        if idx.sum() == 0:
            continue
        bin_conf = float(s[idx].mean())
        bin_acc = float(y[idx].mean())
        ece += (idx.sum() / total) * abs(bin_conf - bin_acc)
    return float(ece)

def fit_isotonic(s_vals, y_vals):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s_vals, y_vals)
    return iso

def fit_platt(s_vals, y_vals):
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(s_vals.reshape(-1,1), y_vals)
    return lr

def make_calibrator(calib_pairs, method="isotonic", min_points=10):
    s_vals = np.array([p[0] for p in calib_pairs], dtype=float)
    y_vals = np.array([1 if p[1] else 0 for p in calib_pairs], dtype=int)
    if len(s_vals) < min_points or y_vals.mean() in (0.0, 1.0):
        # identity fallback
        def identity(s_arr):
            return np.clip(np.array(s_arr, dtype=float), 0.0, 1.0)
        return identity, ("identity", None)
    if method == "platt":
        model = fit_platt(s_vals, y_vals)
        def platt_map(s_arr):
            return model.predict_proba(np.array(s_arr).reshape(-1,1))[:,1]
        return platt_map, ("platt", model)
    else:
        model = fit_isotonic(s_vals, y_vals)
        def iso_map(s_arr):
            return model.transform(np.array(s_arr, dtype=float))
        return iso_map, ("isotonic", model)

def apply_calibrator_fn(calib_fn, s):
    try:
        out = calib_fn([s])
        return float(np.clip(out[0], 0.0, 1.0))
    except Exception:
        return float(np.clip(s, 0.0, 1.0))

def sweep_thresholds(results_df, deltas):
    rows = []
    for d in deltas:
        df = results_df.copy()
        df['decision_d'] = df['dtr_score'].apply(lambda s: "ANSWER" if s > d else "ABSTAIN")
        n = len(df)
        n_answered = int((df['decision_d'] == "ANSWER").sum()) if n>0 else 0
        coverage = n_answered / n if n>0 else 0.0
        if n_answered > 0:
            acc_answered = float(df.loc[df['decision_d']=="ANSWER", 'is_correct'].mean())
            hallucination = 1.0 - acc_answered
            mean_realized_bas = float(df.loc[df['decision_d']=="ANSWER", 'realized_bas'].mean())
        else:
            acc_answered = 0.0
            hallucination = 0.0
            mean_realized_bas = 0.0
        overall_accuracy = float(df['is_correct'].mean()) if n>0 else 0.0
        rows.append({
            "delta": float(d),
            "Coverage": coverage,
            "Answered": n_answered,
            "Accuracy_answered": acc_answered,
            "Hallucination_answered": hallucination,
            "Mean_Realized_BAS_answered": mean_realized_bas,
            "Overall_Accuracy": overall_accuracy
        })
    return pd.DataFrame(rows)

def main(args):
    trace = read_jsonl(args.trace_path)
    if len(trace) == 0:
        raise ValueError("Trace is empty or not readable.")
    safe_name = args.model_name if args.model_name else re.sub(r'\W+', '_', os.path.basename(args.trace_path)).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(OUTPUT_DIR, f"neurips_outputs_{safe_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    split_idx = int(args.split_idx)
    n_total = len(trace)
    test_idx_end = min(split_idx, n_total)
    test_entries = trace[:test_idx_end]
    calib_entries = trace[test_idx_end:]

    print(f"Trace rows: {n_total}. Test rows (first {test_idx_end}): {len(test_entries)}. Calib rows: {len(calib_entries)}")

    # collect calibration pairs
    calib_pairs = []
    for i,e in enumerate(calib_entries):
        try:
            s = extract_reported_s(e)
            y = extract_is_correct(e)
            calib_pairs.append((float(s), bool(y)))
        except KeyError as ke:
            # skip entries missing fields
            continue
    print(f"Collected {len(calib_pairs)} calibration pairs for fitting.")

    # fit calibrator
    calib_fn, calib_model = make_calibrator(calib_pairs, method=args.calib_method, min_points=args.min_points)
    # save calibrator model descriptor via joblib
    calib_save_path = os.path.join(out_dir, f"{safe_name}_calibrator_{timestamp}.joblib")
    try:
        joblib.dump(calib_model, calib_save_path)
        print("Saved calibrator descriptor:", calib_save_path)
    except Exception as e:
        print("Warning: failed to save calibrator model:", e)

    # Evaluate test set with pre/post calibration
    results = []
    s_before = []
    y_before = []
    s_after = []
    y_after = []
    for i,e in enumerate(test_entries):
        try:
            reported_s = extract_reported_s(e)
            is_corr = extract_is_correct(e)
        except KeyError:
            continue
        # record before-calibration pair for ECE
        s_before.append(float(reported_s))
        y_before.append(1 if is_corr else 0)
        # apply calibrator
        hat_p = apply_calibrator_fn(calib_fn, float(reported_s))
        # dtr score uses calibrated hat_p
        dtr_score = score_fn(hat_p)
        # default decision using args.delta
        if dtr_score > float(args.delta):
            decision = "ANSWER"
            final_conf = hat_p
        else:
            decision = "ABSTAIN"
            final_conf = 0.0
        # realized BAS
        if decision == "ABSTAIN":
            realized_bas = 0.0
        else:
            realized_bas = final_conf if is_corr else (final_conf + np.log(max(SCORE_EPS, 1.0 - final_conf)))
        # collect for after-calibration ECE
        s_after.append(float(hat_p))
        y_after.append(1 if is_corr else 0)
        # preserve original question/answer fields if present
        out_entry = {
            "index": i,
            "reported_s": float(reported_s),
            "hat_p": float(hat_p),
            "dtr_score": float(dtr_score),
            "decision_default": decision,
            "is_correct": bool(is_corr),
            "realized_bas": float(realized_bas)
        }
        for k in ("question","ground_truth","selected_answer","candidate","candidates"):
            if k in e:
                out_entry[k] = e[k]
        results.append(out_entry)

    # Save calibrated trace JSONL
    calibrated_trace_path = os.path.join(out_dir, f"{safe_name}_calibrated_trace_{timestamp}.jsonl")
    with open(calibrated_trace_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("Saved calibrated trace to:", calibrated_trace_path)

    # Compute metrics at the default delta and overall stats
    res_df = pd.DataFrame(results)
    n = len(res_df)
    n_answered = int((res_df['decision_default'] == "ANSWER").sum()) if n>0 else 0
    coverage = n_answered / n if n>0 else 0.0
    abstention_rate = 1.0 - coverage
    acc_answered = float(res_df.loc[res_df['decision_default']=="ANSWER", 'is_correct'].mean()) if n_answered>0 else 0.0
    hallucination_rate = 1.0 - acc_answered if n_answered>0 else 0.0
    mean_realized_bas = float(res_df['realized_bas'].mean()) if n>0 else 0.0
    overall_accuracy = float(res_df['is_correct'].mean()) if n>0 else 0.0
    # ECE before/after
    ece_before = compute_ece(s_before, y_before, n_bins=args.n_bins) if len(s_before)>0 else 0.0
    ece_after = compute_ece(s_after, y_after, n_bins=args.n_bins) if len(s_after)>0 else 0.0

    summary = {
        "trace_input": os.path.basename(args.trace_path),
        "calib_method": args.calib_method,
        "n_total_trace": n_total,
        "n_test": n,
        "n_calib_used": len(calib_pairs),
        "split_idx": split_idx,
        "delta_default": float(args.delta),
        "Coverage_default": coverage,
        "Abstention_Rate_default": abstention_rate,
        "Accuracy_answered_default": acc_answered,
        "Overall_Accuracy_default": overall_accuracy,
        "Hallucination_Rate_answered_default": hallucination_rate,
        "Mean_Realized_BAS_default": mean_realized_bas,
        "ECE_before": ece_before,
        "ECE_after": ece_after
    }

    # Save summary JSON
    summary_path = os.path.join(out_dir, f"{safe_name}_calibrated_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved summary to:", summary_path)
    print(json.dumps(summary, indent=4))

    if len(res_df) > 0:
        dtr_vals = res_df['dtr_score'].values
        d_min, d_max = float(np.min(dtr_vals)), float(np.max(dtr_vals))
        deltas = np.linspace(d_min - 0.01, d_max + 0.01, args.sweep_points)
    else:
        deltas = np.linspace(-0.5, 1.0, args.sweep_points)

    sweep_df = sweep_thresholds(res_df.rename(columns={"dtr_score":"dtr_score"}), deltas)
    sweep_csv_path = os.path.join(out_dir, f"{safe_name}_delta_sweep_{timestamp}.csv")
    sweep_df.to_csv(sweep_csv_path, index=False)
    print("Saved delta sweep CSV to:", sweep_csv_path)

    # --- Plots ---
    # 1) Calibration curve before vs after
    fig, ax = plt.subplots(figsize=(6,6))
    # identity line
    xs = np.linspace(0,1,101)
    ax.plot(xs, xs, linestyle="--", label="ideal")
    # empirical calibration by binning for before and after
    def plot_binned(ax, s_arr, y_arr, label):
        s = np.array(s_arr)
        y = np.array(y_arr)
        bins = np.linspace(0,1,11)
        bin_centers = 0.5*(bins[:-1] + bins[1:])
        accs = []
        confs = []
        counts = []
        for i in range(len(bins)-1):
            idx = (s >= bins[i]) & (s < bins[i+1]) if i < len(bins)-2 else (s >= bins[i]) & (s <= bins[i+1])
            if idx.sum() == 0:
                accs.append(np.nan)
                confs.append(0.5*(bins[i]+bins[i+1]))
                counts.append(0)
            else:
                accs.append(float(y[idx].mean()))
                confs.append(float(s[idx].mean()))
                counts.append(int(idx.sum()))
        ax.plot(bin_centers, accs, marker='o', label=label)
    plot_binned(ax, s_before, y_before, "reported (before)")
    plot_binned(ax, s_after, y_after, "calibrated (after)")
    ax.set_xlabel("Mean confidence (bin)")
    ax.set_ylabel("Empirical accuracy (bin)")
    ax.set_title("Calibration curve (before vs after)")
    ax.legend()
    cal_curve_path = os.path.join(out_dir, f"{safe_name}_calibration_curve_{timestamp}.png")
    fig.tight_layout()
    fig.savefig(cal_curve_path, dpi=200)
    plt.close(fig)
    print("Saved calibration curve to:", cal_curve_path)

    # 2) Coverage vs Hallucination (tradeoff) using sweep_df
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(sweep_df['Coverage'], sweep_df['Hallucination_answered'], marker='o')
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Hallucination rate (answered)")
    ax.set_title("Coverage vs Hallucination (delta sweep)")
    cov_hall_path = os.path.join(out_dir, f"{safe_name}_coverage_vs_hallucination_{timestamp}.png")
    fig.tight_layout()
    fig.savefig(cov_hall_path, dpi=200)
    plt.close(fig)
    print("Saved coverage vs hallucination plot to:", cov_hall_path)

    # 3) Mean Realized BAS vs Coverage
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(sweep_df['Coverage'], sweep_df['Mean_Realized_BAS_answered'], marker='o')
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Mean Realized BAS (answered)")
    ax.set_title("Mean Realized BAS vs Coverage (delta sweep)")
    bas_cov_path = os.path.join(out_dir, f"{safe_name}_bas_vs_coverage_{timestamp}.png")
    fig.tight_layout()
    fig.savefig(bas_cov_path, dpi=200)
    plt.close(fig)
    print("Saved BAS vs coverage plot to:", bas_cov_path)

    # Save per-example CSV for reproducibility
    per_example_csv = os.path.join(out_dir, f"{safe_name}_calibrated_per_example_{timestamp}.csv")
    res_df.to_csv(per_example_csv, index=False)
    print("Saved per-example CSV to:", per_example_csv)

    print("All outputs saved to:", out_dir)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Offline calibration + NeurIPS-ready reporting from raw DTR trace JSONL")
    p.add_argument("--trace_path", required=True, help="Path to raw DTR trace JSONL produced by dtr_raw.py")
    p.add_argument("--split_idx", type=int, default=1000, help="Index where test set ends and calibration begins (default 1000)")
    p.add_argument("--calib_method", choices=["isotonic","platt"], default="isotonic", help="Calibration method")
    p.add_argument("--min_points", type=int, default=10, help="Minimum calibration points required; otherwise identity mapping used")
    p.add_argument("--delta", type=float, default=0.0, help="DTR default decision threshold applied after calibration")
    p.add_argument("--n_bins", type=int, default=10, help="Bins for ECE and calibration diagnostics")
    p.add_argument("--sweep_points", type=int, default=25, help="Number of thresholds to sweep for tradeoff plots")
    p.add_argument("--model_name", type=str, default=None, help="Optional model name override for output filenames")
    args = p.parse_args()
    main(args)
