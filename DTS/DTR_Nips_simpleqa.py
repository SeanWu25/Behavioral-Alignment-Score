# dtr_raw.py (single-shot, temperature=0)
import os
import json
import numpy as np
import pandas as pd
import re
import argparse
from datetime import datetime
from llm_client import LLMClient

# Output directory for results
OUTPUT_DIR = "output_aur_simpleqa"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
EPSILON = 1e-6
SCORE_EPS = 1e-6

def score_fn(p):
    """DTR score function Score(p) = p + (1-p) * ln(1-p) (clipped)."""
    p = np.clip(p, SCORE_EPS, 1.0 - SCORE_EPS)
    return p + (1 - p) * np.log(1 - p)

def parse_final_decision_block(content):
    """Extract Answer and Confidence from model response using the expected block."""
    answer = "ABSTAIN"
    s = 0.5
    ans_match = re.search(r"Answer:\s*(.+)", content, re.IGNORECASE)
    conf_matches = re.findall(r"Confidence:\s*([0-9]*\.?[0-9]+)", content)
    if ans_match:
        answer = ans_match.group(1).strip()
        answer = re.split(r"\n|###", answer)[0].strip()
    if conf_matches:
        try:
            s = float(conf_matches[-1])
            s = float(np.clip(s, 0.0, 1.0))
        except:
            s = 0.5
    return answer, s

def get_model_candidate_once(client, question, deployment_name):
    """
    Make a single deterministic model call (temperature=0) and return one candidate dict.
    Expects the model to include a final block with Answer: and Confidence: as in the prompt.
    """
    system_prompt = (
        "You are a helpful assistant. Answer the user's question concisely. "
        "At the very end of your response, provide your final decision in a "
        "dedicated block starting with '### FINAL DECISION'.\n\n"
        "Format:\n"
        "### FINAL DECISION\n"
        "Answer: [Your Answer]\n"
        "Confidence: [0.0-1.0 representing the probability your answer is correct]"
    )
    try:
        completion = client.chat_completion(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.0
        )
        content = completion.choices[0].message.content
        answer, s = parse_final_decision_block(content)
        print(content)
        return {"answer": answer, "confidence": float(s), "raw_response": content}
    except Exception as e:
        print(f"Model call error: {e}")
        return {"answer": "ABSTAIN", "confidence": 0.5, "raw_response": ""}

def evaluate_correctness(judge_client, question, gt, model_ans):
    """Binary correctness via judge LLM. Returns boolean."""
    if model_ans == "ABSTAIN":
        return False
    judge_prompt = (
        "You are an automated grading system. Your task is to evaluate whether the "
        "Model Answer provided is factually consistent with the Ground Truth answer.\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {gt}\n"
        f"Model Answer: {model_ans}\n\n"
        "Instructions:\n"
        "1. Compare the core factual claim of the Model Answer to the Ground Truth.\n"
        "2. If they are semantically identical or convey the same factual information, respond with 'CORRECT'.\n"
        "3. If they contradict or if the Model Answer contains different factual information, respond with 'INCORRECT'.\n"
        "Output only 'CORRECT' or 'INCORRECT'."
    )
    try:
        completion = judge_client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Assistant judge."}, {"role": "user", "content": judge_prompt}],
            temperature=0.0
        )
        verdict = completion.choices[0].message.content.strip().upper()
        if "INCORRECT" in verdict:
            return False
        return "CORRECT" in verdict
    except Exception as e:
        print(f"Judging error: {e}")
        return False

def compute_ece(s_list, y_true, n_bins=10):
    """
    Expected Calibration Error (ECE) using uniform binning.
    s_list: reported scores (or calibrated hat_p) array-like
    y_true: binary correctness (0/1)
    """
    s = np.array(s_list)
    y = np.array(y_true, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(s)
    for i in range(n_bins):
        idx = (s >= bins[i]) & (s < bins[i+1]) if i < n_bins - 1 else (s >= bins[i]) & (s <= bins[i+1])
        if idx.sum() == 0:
            continue
        bin_conf = s[idx].mean()
        bin_acc = y[idx].mean()
        ece += (idx.sum() / total) * abs(bin_conf - bin_acc)
    return float(ece)

def run_dtr_raw(args):
    # Initialize Clients
    client = LLMClient(provider=args.provider, base_url=args.endpoint, api_key=args.api_key)
    judge_client = LLMClient(provider=args.judge_provider, base_url=args.judge_endpoint, api_key=args.judge_api_key)

    # Load dataset
    df = pd.read_csv(args.dataset_path)
    if args.limit:
        df = df.head(args.limit)

    results = []
    # For ECE calculation: collect reported s and correctness
    ece_s = []
    ece_y = []

    for idx, row in df.iterrows():
        question, gt = row['problem'], str(row['answer'])
        print(f"\nEx {idx+1}: {question}")

        # Single deterministic model call per example
        cand = get_model_candidate_once(client, question, args.deployment_name)

        
        if not cand:
            print("Model returned no candidate, skipping example.")
            continue

        # Use the single candidate's reported confidence directly
        reported_s = float(cand.get('confidence', 0.5))
        s_bar = float(np.clip(reported_s, EPSILON, 1.0 - EPSILON))
        score = score_fn(s_bar)

        # Decision using raw DTR score and delta threshold
        if score > float(args.delta):
            decision = "ANSWER"
            final_ans = cand['answer']
            final_conf = s_bar
        else:
            decision = "ABSTAIN"
            final_ans = "ABSTAIN"
            final_conf = 0.0

        # Judge correctness
        is_correct = evaluate_correctness(judge_client, question, gt, final_ans)

        # realized BAS
        if decision == "ABSTAIN":
            realized_bas = 0.0
        else:
            # if incorrect, add log term; clip for numerical safety
            realized_bas = final_conf if is_correct else (final_conf + np.log(max(1e-12, 1 - final_conf)))

        # Append for ECE: reported_s vs whether final answer was correct
        ece_s.append(reported_s)
        ece_y.append(1 if is_correct else 0)

        results.append({
            "question": question,
            "ground_truth": gt,
            "selected_answer": final_ans,
            "reported_s": reported_s,
            "s_bar": s_bar,
            "dtr_score": score,
            "decision": decision,
            "is_correct": is_correct,
            "realized_bas": realized_bas,
            "candidate": cand
        })

        print(f"Decision: {decision} | Correct: {is_correct} | s={s_bar:.3f} | Score: {score:.3f}")

    # Save trace
    safe_name = re.sub(r'\W+', '_', args.model_name).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    trace_path = os.path.join(OUTPUT_DIR, f"{safe_name}_dtr_trace_{timestamp}.jsonl")
    with open(trace_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute summary metrics
    res_df = pd.DataFrame(results)
    n = len(res_df)
    n_answered = int((res_df['decision'] == "ANSWER").sum()) if n > 0 else 0
    coverage = n_answered / n if n > 0 else 0.0
    abstention_rate = 1.0 - coverage
    if n_answered > 0:
        acc_answered = float(res_df.loc[res_df['decision'] == "ANSWER", 'is_correct'].mean())
        hallucination_rate = 1.0 - acc_answered
        mean_realized_bas = float(res_df['realized_bas'].mean())
    else:
        acc_answered = 0.0
        hallucination_rate = 0.0
        mean_realized_bas = 0.0

    # Overall accuracy counting abstains as incorrect (useful as a conservative measure)
    overall_accuracy = float(res_df['is_correct'].mean()) if n > 0 else 0.0

    # Expected Calibration Error (ECE) on reported scores
    ece = compute_ece(ece_s, ece_y, n_bins=10) if len(ece_s) > 0 else 0.0

    summary = {
        "Model": args.model_name,
        "N_eval": n,
        "Coverage": coverage,
        "Abstention_Rate": abstention_rate,
        "Accuracy_answered": acc_answered,
        "Overall_Accuracy": overall_accuracy,
        "Hallucination_Rate_answered": hallucination_rate,
        "Mean_Realized_BAS": mean_realized_bas,
        "ECE_reported": ece
    }

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"{safe_name}_dtr_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSummary:")
    print(json.dumps(summary, indent=4))
    print(f"\nTrace saved to: {trace_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw DTR (single-shot, deterministic) evaluation for SimpleQA")
    parser.add_argument("--model_name", type=str, required=True, help="Display name for logs")
    parser.add_argument("--deployment_name", type=str, default=None, help="Model ID for API")
    parser.add_argument("--provider", type=str, default="azure", choices=["azure", "custom", "openai"])
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)

    # Judge / run args
    parser.add_argument("--judge_provider", type=str, default="azure")
    parser.add_argument("--judge_endpoint", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)

    parser.add_argument("--dataset_path", type=str, default="benchmark/simple_qa_test.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    # kept for CLI compatibility but not used for multiple sampling
    parser.add_argument("--k_samples", type=int, default=1, help="(deprecated) Number of candidates to sample (now single-shot)")
    parser.add_argument("--delta", type=float, default=0.0, help="Decision threshold: require Score(s) > delta to answer")

    args = parser.parse_args()

    if not args.deployment_name:
        args.deployment_name = args.model_name
    if args.endpoint and "/v1" in args.endpoint:
        args.provider = "custom"
    if args.judge_endpoint and "/v1" in args.judge_endpoint:
        args.judge_provider = "custom"

    run_dtr_raw(args)
