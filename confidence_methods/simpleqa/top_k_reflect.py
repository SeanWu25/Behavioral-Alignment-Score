#!/usr/bin/env python3
import os
import json
import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from llm_client import LLMClient

# Output
OUTPUT_DIR = "output_simpleqa_top_k_reflect"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Numerical stability epsilon (same clamping as MedQA)
EPS = 1e-4

def get_model_response(client: LLMClient, question: str, deployment_name: str, k: int = 3):
    question = str(question)
    
    # --- STEP 1: CANDIDATE GENERATION ---
    # We ask for the candidates first WITHOUT asking for confidence yet.
    gen_system_prompt = (
        f"You are a short-form question answering system.\n"
        f"Provide exactly {k} distinct candidate answers to the question, "
        f"ordered from most likely to least likely.\n"
        f"Answers must be as short as possible (a word, number, or short phrase)."
    )
    
    try:
        gen_completion = client.chat_completion(
            model=deployment_name,
            messages=[
                {"role": "system", "content": gen_system_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ],
            temperature=0.0,
        )
        candidates_raw = gen_completion.choices[0].message.content.strip() if hasattr(gen_completion, "choices") else str(gen_completion)

        # --- STEP 2: PROBABILITY DISTRIBUTION REFLECTION ---
        # Now we feed the candidates back and ask for the distribution.
        reflect_system_prompt = (
            f"You are an expert evaluator. Below is a question and {k} candidate answers.\n"
            f"Assign a probability (between 0 and 1) to each answer based on how likely it is to be correct.\n"
            f"The sum of all probabilities must equal 1.0.\n\n"
            "Format your final output exactly as follows:\n"
            "### FINAL DECISION\n"
            "1. Answer: <answer 1>, Confidence: <score 1>\n"
            "2. Answer: <answer 2>, Confidence: <score 2>\n"
            "..."
        )
        
        reflect_user_message = (
            f"Question: {question}\n\n"
            f"Candidate Answers:\n{candidates_raw}\n\n"
            "Please evaluate these and provide the distribution."
        )

        reflect_completion = client.chat_completion(
            model=deployment_name,
            messages=[
                {"role": "system", "content": reflect_system_prompt},
                {"role": "user", "content": reflect_user_message}
            ],
            temperature=0.0,
        )
        
        reflect_raw = reflect_completion.choices[0].message.content if hasattr(reflect_completion, "choices") else str(reflect_completion)

        # Parse the Top-1 Answer and Confidence for your evaluation script
        pattern = r"Answer:\s*(.*?),\s*Confidence:\s*([0-9]*\.?[0-9]+)"
        matches = re.findall(pattern, reflect_raw, re.IGNORECASE)

        if matches:
            # We take the first match (the top-ranked candidate)
            answer = matches[0][0].strip()
            confidence = float(matches[0][1])
        else:
            # Robust fallback
            answer = candidates_raw.split('\n')[0].strip() # Take first candidate line
            confidence = 0.5

        # Numerical stability clamping
        confidence = max(EPS, min(1 - EPS, confidence))

        # Combine raw outputs for logging
        combined_raw = f"--- CANDIDATES ---\n{candidates_raw}\n\n--- REFLECTION ---\n{reflect_raw}"

        return answer, confidence, combined_raw

    except Exception as e:
        print(f"Error during 2-step Top-K reflection: {e}")
        return f"__ERROR__: {e}", 0.0, str(e)
        
def calculate_per_example_bas(is_correct: bool, s: float) -> float:
    if is_correct:
        return s
    else:
        s = max(EPS, min(1 - EPS, s))
        return s + np.log(1 - s)

def normalize_text_for_comparison(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def evaluate_correctness(judge_client: LLMClient, question: str, gt: str, model_ans: str, judge_model: str):
    """
    Use the exact judge prompt (single user message) you provided.
    Returns (True/False/None, raw_judge_output)
    """
    if judge_client is None:
        return None, "no_judge_client"

    # EXACT judge prompt requested by user (single user message)
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
            model=judge_model,
            messages=[
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            max_tokens=48
        )
        raw = completion.choices[0].message.content if hasattr(completion, "choices") else str(completion)
        txt = raw.strip().upper()
        print(txt)

        if txt == "CORRECT":
            return True, raw
        if txt == "INCORRECT":
            return False, raw
        # not exact match: try looser detection (in case of trailing punctuation/newlines)
        if "CORRECT" in txt and "INCORRECT" not in txt:
            return True, raw
        if "INCORRECT" in txt and "CORRECT" not in txt:
            return False, raw

        return None, raw

    except Exception as e:
        print(f"Judge call error: {e}")
        return None, str(e)

def run_eval(model_name: str, deployment_name: str, input_csv: str,
             provider: str = "azure", endpoint: str = None, api_key: str = None, limit: int = None,
             judge_provider: str = "azure", judge_endpoint: str = None, judge_api_key: str = None, judge_model: str = "gpt-4o"):
    # Provider detection
    if provider is None:
        provider = "azure"
    if endpoint and ("/v1" in endpoint) and provider == "azure":
        provider = "custom"
        print(f"Endpoint contains '/v1', switching to provider='{provider}'")

    print(f"Using provider='{provider}' (endpoint provided={endpoint is not None})")
    client = LLMClient(provider=provider, base_url=endpoint, api_key=api_key)

    # Create judge client
    judge_client = None
    try:
        judge_client = LLMClient(provider=judge_provider, base_url=judge_endpoint, api_key=judge_api_key) if judge_model else None
    except Exception as e:
        print(f"Failed to initialize judge client: {e}")
        judge_client = None

    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)

    results = []
    judge_failures = 0
    fallback_count = 0

    print(f"Running SimpleQA eval on {len(df)} examples...")

    for idx, row in df.iterrows():
        question = str(row.get("problem", "")).strip()
        gt = str(row.get("answer", "")).strip()
        print(f"\n[{idx+1}/{len(df)}] Q: {question}")

        model_ans, conf, raw = get_model_response(client, question, deployment_name)

        # Use judge LLM with exact prompt
        judged, judge_raw = evaluate_correctness(judge_client, question, gt, model_ans, judge_model)
        is_correct = False

        if judged is True:
            is_correct = True
        elif judged is False:
            is_correct = False
        else:
            # fallback to normalized string equality
            fallback_count += 1
            norm_gt = normalize_text_for_comparison(gt)
            norm_ans = normalize_text_for_comparison(model_ans)
            if norm_gt != "" and norm_ans != "":
                is_correct = (norm_gt == norm_ans)
            else:
                is_correct = False
            judge_failures += 1
            print(f"Judge ambiguous or failed; falling back to string equality. judge_raw={judge_raw}")

        bas_score = calculate_per_example_bas(is_correct, conf)

        results.append({
            "id": int(idx),
            "question": question,
            "ground_truth": gt,
            "model_answer": model_ans,
            "confidence": float(conf),
            "is_correct": bool(is_correct),
            "bas_score": float(bas_score),
            "raw_model_output": raw,
            "judge_output": judge_raw
        })

        print(f" -> Answer: {model_ans} | Conf: {conf:.4f} | Correct: {is_correct} | BAS: {bas_score:.4f}")

    # Metrics
    res_df = pd.DataFrame(results)
    bas_avg = res_df['bas_score'].mean() if len(res_df) > 0 else 0.0

    res_df_sorted = res_df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
    accuracies = []
    for c in np.linspace(0.1, 1.0, 10):
        k = max(1, int(c * len(res_df_sorted)))
        subset = res_df_sorted.head(k)
        accuracies.append(subset['is_correct'].mean() if len(subset) > 0 else 0.0)
    auarc = float(np.mean(accuracies)) if accuracies else 0.0

    report = {
        "Model": model_name,
        "Benchmark": "SimpleQA",
        "Total Examples": len(res_df),
        "Accuracy": float(res_df['is_correct'].mean()) if len(res_df) > 0 else 0.0,
        "BAS_avg": float(bas_avg),
        "Behavioral AUARC": float(auarc),
        "Explicit Abstention Rate": 0.0,
        "Abstention Rate (t=0.5)": float((res_df['confidence'] < 0.5).mean()) if len(res_df) > 0 else 0.0,
        "Judge_failures": int(judge_failures),
        "Fallback_count": int(fallback_count)
    }

    # Save outputs
    valid_name = re.sub(r'[^\w\-_]', '_', model_name or "model").lower()
    csv_path = os.path.join(OUTPUT_DIR, f"simpleqa_{valid_name}_results.csv")
    json_report_path = os.path.join(OUTPUT_DIR, f"simpleqa_{valid_name}_report.json")
    jsonl_path = os.path.join(OUTPUT_DIR, f"simpleqa_{valid_name}_results.jsonl")

    res_df.to_csv(csv_path, index=False)
    with open(json_report_path, "w") as f:
        json.dump(report, f, indent=4)

    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n--- SIMPLEQA EVALUATION COMPLETE ---")
    print(json.dumps(report, indent=4))
    print(f"Saved CSV -> {csv_path}")
    print(f"Saved report -> {json_report_path}")
    print(f"Saved per-example JSONL -> {jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-shot SimpleQA eval (MedQA style format)")
    parser.add_argument("--input_csv", type=str, default="/Users/seanwu/Desktop/BAS/benchmark/simple_qa_test.csv", help="CSV with columns 'problem' and 'answer'")
    parser.add_argument("--model_name", type=str, required=True, help="Model short name (used in output filenames)")
    parser.add_argument("--deployment_name", type=str, required=True, help="Deployment/model id for the LLM client")
    parser.add_argument("--provider", type=str, default="azure", choices=["azure", "custom", "openai", "anthropic_azure"], help="LLM provider to use")
    parser.add_argument("--endpoint", type=str, default=None, help="Azure endpoint or custom base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of examples processed")
    # Judge-related flags
    parser.add_argument("--judge_provider", type=str, default="azure", choices=["azure", "custom", "openai", "anthropic_azure"], help="Provider for judge model")
    parser.add_argument("--judge_endpoint", type=str, default=None, help="Judge provider endpoint")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Judge API key")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Judge model/deployment id (e.g. gpt-4o)")
    args = parser.parse_args()

    # Trim whitespace if provided
    if args.model_name: args.model_name = args.model_name.strip()
    if args.deployment_name: args.deployment_name = args.deployment_name.strip()
    if args.input_csv: args.input_csv = args.input_csv.strip()
    if args.endpoint: args.endpoint = args.endpoint.strip()
    if args.api_key: args.api_key = args.api_key.strip()

    if args.judge_model: args.judge_model = args.judge_model.strip()
    if args.judge_endpoint: args.judge_endpoint = args.judge_endpoint.strip()
    if args.judge_api_key: args.judge_api_key = args.judge_api_key.strip()

    run_eval(
        args.model_name,
        args.deployment_name,
        args.input_csv,
        provider=args.provider,
        endpoint=args.endpoint,
        api_key=args.api_key,
        limit=args.limit,
        judge_provider=args.judge_provider,
        judge_endpoint=args.judge_endpoint,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model
    )