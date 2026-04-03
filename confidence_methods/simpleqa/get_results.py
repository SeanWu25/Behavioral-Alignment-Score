import pandas as pd
import numpy as np
import glob
import os
from typing import Tuple, Optional

# --- CORE MATH FUNCTIONS ---

def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 95.0, seed: Optional[int] = None) -> Tuple[float, float, float, float]:
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
    lower, upper = np.percentile(boot_means, [(100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0])
    half_width = (upper - lower) / 2.0
    return mean, half_width, lower, upper

def expected_calibration_error(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE as a proportion (0-1)."""
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if confidences.size == 0: return np.nan
    if confidences.max() > 1.0: confidences /= 100.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < (bins[i+1] if i < n_bins-1 else 1.0001))
        bin_count = mask.sum()
        if bin_count > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = labels[mask].mean()
            ece += (bin_count / n) * abs(bin_conf - bin_acc)
    return ece

def aurc_from_confidence(confidences: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under Risk-Coverage Curve."""
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if confidences.size == 0: return np.nan
    if confidences.max() > 1.0: confidences /= 100.0

    order = np.argsort(-confidences)
    sorted_labels = labels[order]
    n = len(sorted_labels)
    cumsum_correct = np.cumsum(sorted_labels)
    ks = np.arange(1, n + 1)
    coverage = ks / n
    risk = 1.0 - (cumsum_correct / ks)
    return np.trapezoid(risk, coverage)

def find_confidence_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ['confidence', 'conf', 'prob', 'probability', 'max_prob']
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower: return cols_lower[cand]
    for col in df.columns:
        if 'conf' in col.lower() or 'prob' in col.lower(): return col
    return None

# --- MAIN CALCULATION & LATEX EXPORT ---

def calculate_results(folder_path: str = "output_medqa_colm",
                      n_bootstrap: int = 5000,
                      ci: float = 95.0,
                      seed: int = 12345,
                      ece_bins: int = 10):
    
    csv_files = glob.glob(os.path.join(folder_path, "*_results.csv"))
    if not csv_files:
        print(f"No files found in {folder_path}")
        return

    summary_rows = []

    for file in sorted(csv_files):
        # Clean model name for display (e.g., gpt-4_results -> Gpt-4)
        model_name = os.path.basename(file).replace("_results.csv", "").replace("_", " ").title()
        df = pd.read_csv(file)

        if 'is_correct' not in df.columns: continue

        # 1. Accuracy (Percentage)
        acc_vals = df['is_correct'].astype(float).values * 100.0
        acc_m, acc_h, _, _ = bootstrap_mean_ci(acc_vals, n_bootstrap, ci, seed)

        # 2. BAS (Scientific Value)
        bas_m, bas_h = np.nan, np.nan
        if 'bas_score' in df.columns:
            bas_m, bas_h, _, _ = bootstrap_mean_ci(df['bas_score'].values, n_bootstrap, ci, seed + 1)

        # 3. Calibration (ECE & AURC)
        conf_col = find_confidence_column(df)
        ece_fmt, aurc_fmt = "N/A", "N/A"
        
        if conf_col:
            conf = df[conf_col].astype(float).values
            labels = df['is_correct'].astype(float).values
            
            # Point estimates
            ece_val = expected_calibration_error(conf, labels, n_bins=ece_bins) * 100.0
            aurc_val = aurc_from_confidence(conf, labels)
            
            # Bootstrap CIs for ECE/AURC
            rng = np.random.default_rng(seed + 2)
            b_ece, b_aurc = [], []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(conf), size=len(conf))
                b_ece.append(expected_calibration_error(conf[idx], labels[idx], n_bins=ece_bins) * 100.0)
                b_aurc.append(aurc_from_confidence(conf[idx], labels[idx]))
            
            ece_h = (np.percentile(b_ece, 97.5) - np.percentile(b_ece, 2.5)) / 2.0
            aurc_h = (np.percentile(b_aurc, 97.5) - np.percentile(b_aurc, 2.5)) / 2.0
            
            ece_fmt = f"{ece_val:.1f} \\pm {ece_h:.1f}"
            aurc_fmt = f"{aurc_val:.3f} \\pm {aurc_h:.3f}"

        summary_rows.append({
            "Model": model_name,
            "Accuracy (\%)": f"{acc_m:.1f} \\pm {acc_h:.1f}",
            "BAS": f"{bas_m:.3f} \\pm {bas_h:.3f}",
            "ECE (\%)": ece_fmt,
            "AURC": aurc_fmt
        })

    summary_df = pd.DataFrame(summary_rows)
    
    # Generate LaTeX
    latex_output = summary_df.to_latex(
        index=False,
        escape=False,
        column_format="lcccc",
        caption="Model Performance Metrics with 95\% Bootstrap Confidence Intervals.",
        label="tab:results",
        position="t"
    )

    print("\n--- PANDAS DATAFRAME ---")
    print(summary_df.to_string(index=False))
    
    print("\n--- NEURIPS LATEX CODE ---")
    print(latex_output)

if __name__ == "__main__":
    # Ensure the directory exists or change the path to your data
    calculate_results(folder_path="output_simpleqa_verbalize")