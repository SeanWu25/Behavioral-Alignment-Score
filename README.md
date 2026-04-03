<![CDATA[# Behavioral Alignment Score (BAS)

A decision-theoretic framework for measuring **LLM alignment and uncertainty**. BAS evaluates how well a model's stated confidence aligns with its actual correctness — rewarding calibrated confidence and penalizing overconfident errors through a proper scoring rule grounded in selective-prediction theory.

> **Paper**: *Behavioral Alignment of Large Language Models* — Sean Wu (2026)

---

## Key Features

- **Proper Scoring Rule** — BAS is derived from decision-theoretic utility, penalizing overconfident incorrect predictions via `s + ln(1 − s)` while rewarding calibrated correct ones
- **Risk-Prior Weighting** — Uniform, linear, and quadratic priors let you tune evaluation severity for general-purpose vs. safety-critical deployments
- **Multi-Benchmark Evaluation** — Ready-to-run pipelines for [SimpleQA](https://github.com/openai/simple-evals), [AIME 2024/25](https://artofproblemsolving.com/wiki/index.php/AMC_Problems_and_Solutions), and [MedQA (USMLE)](https://github.com/jind11/MedQA)
- **Isotonic Calibration** — Post-hoc calibration with before/after metric comparison (BAS, ECE, AURC)
- **Multi-Provider LLM Client** — Unified wrapper supporting Azure OpenAI, OpenAI, and Anthropic endpoints
- **Installable Python Package** — `pip install` the core `bas_eval` library for standalone metric computation

---

## Project Structure

```
Behavioral_Alignment_Score/
├── bas_eval/                   # Core package (pip-installable)
│   ├── __init__.py
│   ├── metrics.py              # bas_score() — Eq. 4/6 from the paper
│   └── report.py               # BASReport class — summary across risk priors
├── benchmark/                  # Bundled benchmark datasets
│   ├── simple_qa_test.csv
│   ├── aime_2024_2025.csv
│   └── bas_medqa.jsonl
├── simpleqa.py                 # SimpleQA evaluation pipeline (w/ LLM judge)
├── aime.py                     # AIME 2024/25 evaluation pipeline
├── medqa.py                    # MedQA (USMLE) evaluation pipeline
├── llm_client.py               # Multi-provider LLM client
├── weighted_bas.py             # Risk-prior sensitivity analysis + LaTeX output
├── calibrate_confdience.py     # Isotonic calibration pipeline
├── example_usage.py            # Quick-start example
├── example_output.csv          # Sample results for example_usage.py
├── setup.py                    # Package installer
└── LICENSE                     # MIT License
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SeanWu25/Behavioral-Alignment-Score.git
cd Behavioral-Alignment-Score
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn openai anthropic
```

### 4. Install the `bas_eval` package

```bash
pip install -e .
```

### 5. Configure API keys (for benchmark evaluation)

Create a `.env` file or export environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
```

---

## Quick Start

### Compute BAS from existing results

```python
import pandas as pd
import numpy as np
from bas_eval import bas_score, BASReport

# Load a CSV with 'is_correct' (bool) and 'confidence' (0–1) columns
df = pd.read_csv("example_output.csv")

# Per-example BAS scores (uniform prior)
scores = bas_score(df['is_correct'], df['confidence'])
print(f"Mean BAS: {np.mean(scores):.4f}")

# Full report across all risk priors
report = BASReport(df['is_correct'], df['confidence'])
report.print_summary()

# Safety-critical weighted score
safety = report.weighted_score(prior='quadratic')
print(f"Safety-Critical BAS: {safety:.4f}")
```

**Output:**

```
=============================================
          BAS ALIGNMENT REPORT
=============================================
Risk Profile              |     Mean BAS
---------------------------------------------
Uniform (Standard)        |       0.XXXX
Linear (Risk-Aware)       |       0.XXXX
Quadratic (Safety)        |       0.XXXX
=============================================
```

---

## Running Benchmark Evaluations

Each benchmark script queries an LLM, collects answers with confidence scores, evaluates correctness, and computes BAS metrics.

### SimpleQA

```bash
python simpleqa.py \
  --model_name "gpt-4o" \
  --deployment_name "gpt-4o" \
  --provider azure \
  --judge_model "gpt-4o"
```

### AIME 2024/25

```bash
python aime.py \
  --model_name "gpt-4o" \
  --deployment_name "gpt-4o" \
  --endpoint "https://your-endpoint.openai.azure.com/"
```

### MedQA (USMLE)

```bash
python medqa.py \
  --model_name "gpt-4o" \
  --deployment_name "gpt-4o" \
  --input "benchmark/bas_medqa.jsonl"
```

### Supported providers

| Flag | Provider |
|------|----------|
| `azure` | Azure OpenAI (default) |
| `openai` | OpenAI API |
| `custom` | Any OpenAI-compatible endpoint |
| `anthropic_azure` | Anthropic via Azure Foundry |

---

## Post-Hoc Calibration

Apply isotonic regression to recalibrate model confidence and compare metrics before/after:

```bash
python calibrate_confdience.py \
  --dev_path  output_simpleqa/simpleqa_gpt-4o-mini_results.jsonl \
  --test_path output_simpleqa/simpleqa_gpt-4o-mini_results.jsonl
```

This produces:
- A side-by-side comparison table (Accuracy, BAS, ECE, AURC)
- LaTeX-ready table output
- Calibrated `.jsonl` and `.csv` result files

---

## Risk-Prior Sensitivity Analysis

Generate a LaTeX table comparing BAS under different risk weighting functions:

```bash
python weighted_bas.py --trace_path output_simpleqa/simpleqa_gpt-4o_results.csv
```

| Risk Profile | Weighting Function | Description |
|---|---|---|
| General Purpose | `w(t) = 1` (Uniform) | Equal weight across all thresholds |
| Risk-Aware | `w(t) = 2t` (Linear) | Higher weight on high-confidence regions |
| Safety-Critical | `w(t) = 3t²` (Quadratic) | Heavy penalty for overconfident errors |

---

## Core API Reference

### `bas_score(is_correct, confidence, prior='uniform')`

Compute per-example BAS scores.

| Parameter | Type | Description |
|---|---|---|
| `is_correct` | array-like | Boolean correctness labels |
| `confidence` | array-like | Model confidence scores in `[0, 1]` |
| `prior` | str | Risk prior: `'uniform'`, `'linear'`, or `'quadratic'` |

**Returns:** `np.ndarray` of per-example BAS values.

### `BASReport(is_correct, confidence)`

Generates a full alignment report across all risk priors.

| Method | Description |
|---|---|
| `.print_summary()` | Print formatted report table |
| `.weighted_score(prior)` | Get mean BAS for a specific prior |
| `.from_df(df)` | Class method to initialize from a DataFrame |

---

## Output Format

Evaluation scripts produce per-example results in CSV/JSONL with these fields:

| Field | Description |
|---|---|
| `id` | Example index |
| `question` | Input question |
| `ground_truth` | Gold-standard answer |
| `model_answer` | Model's predicted answer |
| `confidence` | Model's self-reported confidence `[0, 1]` |
| `is_correct` | Whether the prediction is correct |
| `bas_score` | Per-example BAS value |

---

## Citation

If you use BAS in your research, please cite:

```bibtex
@article{wu2026behavioral,
  title   = {Behavioral Alignment of Large Language Models},
  author  = {Wu, Sean},
  year    = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
]]>