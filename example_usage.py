import pandas as pd
from bas_eval import bas_score, BASReport
import numpy as np

df = pd.read_csv("example_output.csv")

score = bas_score(df['is_correct'], df['confidence'])
print(f"Dataset Mean BAS (Uniform): {np.mean(score):.4f}\n")

report = BASReport(df['is_correct'], df['confidence'])

report.print_summary()

safety_score = report.weighted_score(prior='quadratic')
print(f"\nSafety-Critical BAS (w(t)=3t^2): {safety_score:.4f}")

