import pandas as pd
import numpy as np
import argparse
import os

def compute_weighted_bas(df, epsilon=1e-4):
   
    s = np.clip(df['confidence'].values, epsilon, 1.0 - epsilon)
    z = df['is_correct'].values.astype(bool)
  
    bas_uni = np.where(z, s, s + np.log(1 - s))
    
    bas_lin = np.where(z, s**2, s**2 + 2*s + 2*np.log(1 - s))
    
    bas_quad = np.where(z, s**3, s**3 + 1.5*s**2 + 3*s + 3*np.log(1 - s))
    
    return {
        "Uniform": np.mean(bas_uni),
        "Linear": np.mean(bas_lin),
        "Quadratic": np.mean(bas_quad)
    }

def infer_benchmark(filename):
    fn = filename.lower()
    if 'simpleqa' in fn: return "SimpleQA"
    if 'medqa' in fn: return "MedQA"
    if 'aime' in fn: return "AIME 2024/25"
    return "Target Benchmark"

def main():
    parser = argparse.ArgumentParser(description="Professional Weighted BAS Table Generator")
    parser.add_argument("--trace_path", required=True, help="Path to results CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.trace_path):
        print(f"Error: {args.trace_path} not found.")
        return
        
    df = pd.read_csv(args.trace_path)
    filename = os.path.basename(args.trace_path)
    
    model_name = filename.split('_')[0].upper()
    benchmark = infer_benchmark(filename)
    
    results = compute_weighted_bas(df)

    print(f"\n% --- Table FOR {benchmark} ---")
    print(f"% Requires: \\usepackage{{booktabs, graphicx, multirow}}")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\caption{{\\textbf{{Risk-Prior Sensitivity Analysis ({benchmark}).}} Comparison of {model_name} behavioral alignment across varying risk thresholds.}}")
    print("\\label{tab:weighted_bas_ablation}")
    print("\\small")
    print("\\begin{tabular}{@{}llc@{}}")
    print("\\toprule")
    print(f"\\textbf{{Benchmark: {benchmark}}} & \\multicolumn{{2}}{{c}}{{\\includegraphics[height=1.0em]{{deepseek_logo.png}} \\quad \\textbf{{{model_name}}}}} \\\\")
    print("\\cmidrule(l){2-3}")
    print("\\textbf{Risk Profile} & \\textbf{Weighting Function $w(t)$} & \\textbf{Mean BAS} $\\uparrow$ \\\\")
    print("\\midrule")
    print(f"General Purpose & Uniform: $w(t) = 1$ & \\textbf{{{results['Uniform']:.4f}}} \\\\")
    print(f"Risk-Aware      & Linear: $w(t) = 2t$ & {results['Linear']:.4f} \\\\")
    print(f"Safety-Critical & Quadratic: $w(t) = 3t^2$ & {results['Quadratic']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()