#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import re
import argparse
from datetime import datetime
from llm_client import LLMClient

# Output directory for results
OUTPUT_DIR = "output_medqa"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
EPSILON = 1e-6
SCORE_EPS = 1e-6

def score_fn(p):
    """DTR score function Score(p) = p + (1-p) * ln(1-p)."""
    p = np.clip(p, SCORE_EPS, 1.0 - SCORE_EPS)
    return p + (1 - p) * np.log(1 - p)

def get_model_response(client, question, options, deployment_name):
    """
    Queries the model for a medical answer and a confidence score.
    FORCED CHOICE: No abstention option provided to the model.
    """
    formatted_options = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    system_prompt = (
        "You are a medical expert assisting with USMLE-style questions. "
        "Analyze the clinical case and select the most appropriate option.\n\n"
        "At the end of your response, provide your final decision in a "
        "block starting with '### FINAL DECISION'.\n\n"
        "Format:\n"
        "### FINAL DECISION\n"
        "Answer: [Single Letter A-D]\n"
        "Confidence: [0-1 representing the probability your answer is correct]"
    )
    
    user_input = f"Question: {question}\n\nOptions:\n{formatted_options}"

    try:
        completion = client.chat_completion(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0
        )
        content = completion.choices[0].message.content
        
        # Parse Confidence
        conf_matches = re.findall(r"Confidence:\s*([0-9.]+)", content)
        confidence = float(conf_matches[-1]) if conf_matches else 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))
        print(content)
        
        # Parse Answer: Targets letters A-D only
        ans_matches = re.findall(r"Answer:\s*([A-D])", content, re.IGNORECASE)
        raw_ans = ans_matches[-1].upper() if ans_matches else "N/A"
        
        return {"answer": raw_ans, "confidence": confidence, "raw_response": content}
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return {"answer": "N/A", "confidence": 0.5, "raw_response": str(e)}

def run_dtr_medqa(args):
    # Client initialization
    provider = args.provider
    if args.endpoint and ("/v1" in args.endpoint):
        provider = "custom"
    
    client = LLMClient(provider=provider, base_url=args.endpoint, api_key=args.api_key)
    deployment_name = args.deployment_name if args.deployment_name else args.model_name

    # Load dataset
    with open(args.dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if args.limit:
        data = data[:args.limit]

    results = []
    print(f"Starting MedQA Forced-Choice Eval on {len(data)} cases...")

    for idx, item in enumerate(data):
        question = item['question']
        options = item['options']
        gt_answer = item['answer_idx'] 
        
        # Single deterministic model call
        cand = get_model_response(client, question, options, deployment_name)
        
        reported_s = float(cand.get('confidence', 0.5))
        s_bar = float(np.clip(reported_s, EPSILON, 1.0 - EPSILON))
        score = score_fn(s_bar)

        # Correctness check (Forced choice match)
        model_ans = cand['answer']
        is_correct = (model_ans == gt_answer)

        # realized BAS (Always uses s_bar since there is no explicit abstention here)
        realized_bas = s_bar if is_correct else (s_bar + np.log(max(1e-12, 1 - s_bar)))

        results.append({
            "question": question,
            "ground_truth": gt_answer,
            "selected_answer": model_ans,
            "reported_s": reported_s,
            "s_bar": s_bar,
            "dtr_score": score,
            "is_correct": is_correct,
            "realized_bas": realized_bas,
            "candidate": cand
        })

        print(f"[{idx+1}] Correct: {is_correct} | s={s_bar:.4f} | Score: {score:.4f}")

    # Save Trace
    safe_name = re.sub(r'\W+', '_', args.model_name).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    trace_path = os.path.join(OUTPUT_DIR, f"medqa_{safe_name}_dtr_trace_{timestamp}.jsonl")
    
    with open(trace_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nTrace saved to: {trace_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, default=None)
    parser.add_argument("--provider", type=str, default="azure")
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="benchmark/bas_medqa.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    run_dtr_medqa(args)