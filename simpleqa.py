import os
import json
import numpy as np
import pandas as pd
import re
import argparse
from datetime import datetime
from llm_client import LLMClient

# Output directory
OUTPUT_DIR = "output_dtr_simpleqa"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DTR Specific Constants
K_SAMPLES = 5  # Number of candidates (k) per paper [cite: 125, 372]
EPSILON = 1e-4 # Numerical stability clipping [cite: 116, 164]

def get_model_candidates(client, question, deployment_name, k=K_SAMPLES):
    """
    Samples k candidates at T=1.0 and calculates expected BAS utility.
    """
    system_prompt = (
        "You are a helpful assistant. Answer the user's question concisely. "
        "At the very end of your response, provide your final decision in a "
        "dedicated block starting with '### FINAL DECISION'.\n\n"
        "Format:\n"
        "### FINAL DECISION\n"
        "Answer: [Your Answer]\n"
        "Confidence: [0-1 representing the probability your answer is correct]"
    )
    
    candidates = []
    for _ in range(k):
        try:
            completion = client.chat_completion(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=1.0 # High temperature for uncertainty exploration 
            )
            content = completion.choices[0].message.content
            
            # Parse Confidence
            conf_matches = re.findall(r"Confidence:\s*([0-9.]+)", content)
            s = float(conf_matches[-1]) if conf_matches else 0.5
            
            # 1. Apply Numerical Stability Clipping [cite: 116, 164]
            s_bar = max(0.0001, min(0.9999, s))
            
            # 2. Calculate Expected BAS Utility [cite: 132, 154]
            # Expected Score = s + (1-s) * ln(1-s)
            utility = s_bar + (1 - s_bar) * np.log(1 - s_bar)
            
            # Parse Answer
            ans_match = re.search(r"Answer:\s*(.+)", content, re.IGNORECASE)
            answer = ans_match.group(1).strip() if ans_match else "ABSTAIN"
            
            candidates.append({
                "answer": answer,
                "confidence": s_bar,
                "expected_utility": utility,
                "raw_response": content
            })
        except Exception as e:
            print(f"Sampling error: {e}")
            
    return candidates

def evaluate_correctness(judge_client, question, gt, model_ans):
    """Uses the judge parameters to verify final DTR output."""
    if model_ans == "ABSTAIN": return False
    
    judge_prompt = (
        f"Question: {question}\nGround Truth: {gt}\nModel Answer: {model_ans}\n\n"
        "If the Model Answer matches the Ground Truth in meaning, respond 'CORRECT'. Otherwise, 'INCORRECT'."
    )
    try:
        completion = judge_client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Assistant judge."}, {"role": "user", "content": judge_prompt}],
            temperature=0
        )
        verdict = completion.choices[0].message.content.strip().upper()
        return "CORRECT" in verdict
    except Exception as e:
        print(f"Judging error: {e}")
        return False

def run_dtr_eval(args):
    # Initialize Primary Client (matches your previous logic)
    provider = "azure"
    if args.endpoint and ("/v1" in args.endpoint): provider = "custom"
    client = LLMClient(provider=provider, base_url=args.endpoint, api_key=args.api_key)
    
    # Initialize Judge Client using the important judge parameters
    judge_provider = "azure"
    if args.judge_endpoint and ("/v1" in args.judge_endpoint): judge_provider = "custom"
    judge_client = LLMClient(provider=judge_provider, base_url=args.judge_endpoint, api_key=args.judge_api_key)
    
    df = pd.read_csv('benchmark/simple_qa_test.csv')
    if args.limit: df = df.head(args.limit)

    results = []
    for idx, row in df.iterrows():
        question, gt = row['problem'], str(row['answer'])
        print(f"\nEx {idx+1}: {question}")

        # Step 1: Candidate Generation 
        candidates = get_model_candidates(client, question, args.deployment_name)
        if not candidates: continue
        
        # Step 2: Identify max expected utility candidate [cite: 139, 382]
        best_cand = max(candidates, key=lambda x: x['expected_utility'])
        
        # Step 3: DTR Decision Rule (Refuse if Score <= 0) 
        if best_cand['expected_utility'] > 0:
            final_ans, final_conf, decision = best_cand['answer'], best_cand['confidence'], "ANSWER"
        else:
            final_ans, final_conf, decision = "ABSTAIN", 0.0, "ABSTAIN"

        # Verification & Realized Scoring
        is_correct = evaluate_correctness(judge_client, question, gt, final_ans)
        
        # Realized BAS Calculation [cite: 91, 115]
        if decision == "ABSTAIN":
            realized_bas = 0.0
        else:
            realized_bas = final_conf if is_correct else (final_conf + np.log(1 - final_conf))

        results.append({
            "question": question, "ground_truth": gt, "selected_answer": final_ans,
            "confidence": final_conf, "decision": decision, "is_correct": is_correct,
            "realized_bas": realized_bas, "all_candidates": candidates
        })

    # Saving Logic
    safe_name = re.sub(r'\W+', '_', args.model_name).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    with open(os.path.join(OUTPUT_DIR, f"{safe_name}_dtr_trace_{timestamp}.jsonl"), 'w') as f:
        for r in results: f.write(json.dumps(r) + '\n')

    res_df = pd.DataFrame(results)
    summary = {
        "Model": args.model_name,
        "Accuracy": float(res_df['is_correct'].mean()),
        "Mean_BAS": float(res_df['realized_bas'].mean()),
        "Abstention_Rate": float((res_df['decision'] == "ABSTAIN").mean())
    }
    
    with open(os.path.join(OUTPUT_DIR, f"{safe_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(json.dumps(summary, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DTR Re-ranking on SimpleQA")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--judge_endpoint", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    
    args = parser.parse_args()
    run_dtr_eval(args)