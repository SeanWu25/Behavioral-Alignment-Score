#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def score_fn(p):
    """Expected utility: p + (1-p)ln(1-p)."""
    score_eps = 1e-12
    p = np.clip(p, score_eps, 1.0 - score_eps)
    return p + (1.0 - p) * np.log(1.0 - p)

def main(args):
    if not os.path.exists(args.trace_path):
        print(f"File not found: {args.trace_path}")
        return

    with open(args.trace_path, "r") as f:
        data = [json.loads(l) for l in f if l.strip()]
    
    test_entries = data[:args.split_idx]
    calib_entries = data[args.split_idx:]

    # 1. Calibration
    s_cal = [float(e.get('reported_s', 0.5)) for e in calib_entries]
    y_cal = [1 if e.get('is_correct', False) else 0 for e in calib_entries]
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_cal, y_cal)
    
    # 2. Process Test Set
    results = []
    for e in test_entries:
        s = float(e.get('reported_s', 0.5))
        corr = bool(e.get('is_correct', False))
        hat_p = float(iso.transform([s])[0])
        results.append({
            "question": e.get('question', 'N/A'),
            "ground_truth": e.get('ground_truth', 'N/A'),
            "answer": e.get('selected_answer', 'N/A'),
            "raw_s": s,
            "hat_p": hat_p,
            "score": score_fn(hat_p),
            "is_correct": corr
        })
    df = pd.DataFrame(results)

    # 3. Dynamic Thresholding
    target_coverage = 0.10
    sorted_scores = sorted(df['score'].unique())
    best_delta = 0.0
    best_diff = 1.0
    for d in sorted_scores:
        cov = (df['score'] > d).mean()
        if abs(cov - target_coverage) < best_diff:
            best_diff, best_delta = abs(cov - target_coverage), d
    
    # 4. Filter for Interventions
    interventions = df[(df['is_correct'] == False) & (df['raw_s'] > 0.5) & (df['score'] <= best_delta)]
    interventions = interventions.sort_values(by='raw_s', ascending=False).head(args.num)

    # 5. Output to Console
    print(f"\n{'='*90}")
    print(f"DTS SAFETY ANALYSIS: {len(interventions)} LIES UNMASKED")
    print(f"Targeting {target_coverage*100}% Coverage Regime (Delta: {best_delta:.4f})")
    print(f"{'='*90}\n")

    for i, row in interventions.iterrows():
        print(f"EXAMPLE #{i+1}")
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']} (GT: {row['ground_truth']})")
        print(f"RAW CONFIDENCE:  {row['raw_s']:.1%}")
        print(f"CALIBRATED PROB: {row['hat_p']:.1%}")
        print(f"DECISION:        [ABSTAIN] ✅")
        print(f"{'-'*90}")

    # 6. SAVE TO FILES
    output_base = os.path.basename(args.trace_path).replace('.jsonl', '')
    
    # Save as JSONL
    json_path = f"{output_base}_interventions.jsonl"
    interventions.to_json(json_path, orient='records', lines=True)
    
    # Save as a pretty LaTeX Table for the paper
    tex_path = f"{output_base}_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table*}[h]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{p{0.4\\textwidth} p{0.15\\textwidth} p{0.15\\textwidth} c c}\n")
        f.write("\\toprule\n")
        f.write("Question & Model Answer & Ground Truth & Raw $s$ & DTS $\\hat{p}$ \\\\\n")
        f.write("\\midrule\n")
        for _, row in interventions.head(10).iterrows(): # Limit to top 10 for table
            q = row['question'].replace('_', '\\_')[:100] + "..."
            ans = str(row['answer']).replace('_', '\\_')
            gt = str(row['ground_truth']).replace('_', '\\_')
            f.write(f"{q} & {ans} & {gt} & {row['raw_s']:.1%} & {row['hat_p']:.1%} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Qualitative examples of confident hallucinations unmasked by DTS on SimpleQA.}\n")
        f.write("\\end{table*}")

    print(f"\nFiles saved:\n1. {json_path}\n2. {tex_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace_path", required=True)
    p.add_argument("--split_idx", type=int, default=1000)
    p.add_argument("--num", type=int, default=1000)
    main(p.parse_args())