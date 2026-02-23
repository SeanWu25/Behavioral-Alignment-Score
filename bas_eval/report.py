import pandas as pd
import numpy as np
from .metrics import bas_score

class BASReport:
    def __init__(self, is_correct, confidence):
        self.is_correct = is_correct
        self.confidence = confidence
        self.results = {}
        self._compute_all()

    @classmethod
    def from_df(cls, df, corr_col='is_correct', conf_col='confidence'):
        """Convenience method to initialize from a DataFrame."""
        if corr_col not in df.columns or conf_col not in df.columns:
            raise KeyError(f"DataFrame must contain columns: '{corr_col}' and '{conf_col}'")
        return cls(df[corr_col], df[conf_col])

    def _compute_all(self):
        for p in ['uniform', 'linear', 'quadratic']:
            scores = bas_score(self.is_correct, self.confidence, prior=p)
            self.results[p] = np.mean(scores)

    def weighted_score(self, prior='quadratic'):
        return self.results.get(prior)

    def print_summary(self):
        print("\n" + "="*45)
        print(f"{'BAS ALIGNMENT REPORT':^45}")
        print("="*45)
        print(f"{'Risk Profile':<25} | {'Mean BAS':>12}")
        print("-" * 45)
        print(f"{'Uniform (Standard)':<25} | {self.results['uniform']:>12.4f}")
        print(f"{'Linear (Risk-Aware)':<25} | {self.results['linear']:>12.4f}")
        print(f"{'Quadratic (Safety)':<25} | {self.results['quadratic']:>12.4f}")
        print("="*45 + "\n")