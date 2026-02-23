import os
import json
import numpy as np
import pandas as pd
import re
import argparse

from llm_client import LLMClient

# Configuration
# Default configuration
DEFAULT_MODEL_NAME = ""
DEFAULT_DEPLOYMENT_NAME = ""
OUTPUT_DIR = "output_aime"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Using 'custom' provider to match the original OpenAI(base_url=...) behavior
# Using 'custom' provider to match the original OpenAI(base_url=...) behavior
# client = LLMClient(provider="custom") -> Moved inside run_eval

def get_model_response(client, problem, deployment_name):
    """
    Queries the model for an answer/abstention and a confidence score.
    Uses a delimited format to ensure robust parsing of the final decision.
    """
    # Prompt explicitly requests a final block to aid black-box evaluation
    system_prompt = (
        "You are a mathematical assistant. Solve the given problem step-by-step. "
        "At the very end of your response, provide your final decision in a "
        "dedicated block starting with '### FINAL DECISION'.\n\n"
        "If you are unsure or the risk of a mistake is high, abstain by saying 'Answer: ABSTAIN'. "
        "Otherwise, provide an integer answer.\n\n"
        "Format:\n"
        "### FINAL DECISION\n"
        "Answer: [integer or ABSTAIN]\n"
        "Confidence: [0-1 representing the probability your answer is correct]"
    )
    
    try:
        completion = client.chat_completion(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem}
            ],
           # temperature=0  # Consistent with selective prediction evaluation
        )
        content = completion.choices[0].message.content

        print(content)
        
        # 1. Parse Confidence: Look for the LAST occurrence to avoid 
        # picking up intermediate decimals from the math reasoning.
        conf_matches = re.findall(r"Confidence:\s*([0-9.]+)", content)
        if conf_matches:
            confidence = float(conf_matches[-1])
        else:
            # If the model fails to provide a score, we treat it as 
            # maximum uncertainty (s=0) for a neutral utility.
            confidence = 0.0
            
        # Clamp per Algorithm 1: eps to avoid log(0)
        confidence = max(0.0001, min(0.9999, confidence))
        
        # 2. Parse Answer: Again, look for the LAST occurrence.
        ans_matches = re.findall(r"Answer:\s*(\d+|ABSTAIN)", content, re.IGNORECASE)
        raw_ans = ans_matches[-1].upper() if ans_matches else None
        
        # Implement decision logic: Abstention is a first-class action
        if raw_ans == "ABSTAIN" or raw_ans is None:
            return "ABSTAIN", confidence, content

        # Clean integer parsing (handles possible LaTeX formatting like \boxed{123})
        clean_ans = re.sub(r"[^\d]", "", raw_ans)
        if clean_ans:
            return int(clean_ans), confidence, content
        
        return "ABSTAIN", confidence, content
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, 0.0, str(e)
    
def calculate_per_example_bas(is_correct, s):
    """
    Implements the analytical BAS utility from the paper.
    If correct: Integral of 1 from 0 to s = s.
    If incorrect: Integral of -t/(1-t) from 0 to s = s + ln(1-s).
    """
    if is_correct:
        return s
    else:
        # Formula from equation (2) in BAS Preprint
        return s + np.log(1 - s)

def run_eval(model_name, deployment_name, endpoint=None, api_key=None):
    # Auto-detect provider
    provider = "azure"
    if endpoint and ("/v1" in endpoint):
        provider = "custom"
        print(f"Endpoint contains '/v1', switching to provider='{provider}'")
    elif endpoint and ("anthropic" in endpoint):
        provider = "anthropic_azure"
        print(f"Endpoint contains 'anthropic', switching to provider='{provider}'")

    # Initialize Client
    client = LLMClient(provider=provider, base_url=endpoint, api_key=api_key)
    
    if not deployment_name and model_name:
        deployment_name = model_name
        print(f"Deployment name not provided. Using model_name '{model_name}' as deployment name.")
        
    df = pd.read_csv('benchmark/aime_2024_2025.csv')
    results = []

    print(f"Starting BAS Eval on {len(df)} problems...")

    for idx, row in df.iterrows():
        problem = row['problem']
        gt_answer = row['ground_truth_answer']
        
        model_ans, s, raw_response = get_model_response(client, problem, deployment_name)
        
        if model_ans == "ABSTAIN":
            s = 0.0
            
        is_correct = (model_ans == gt_answer) if model_ans is not None else False
        
        bas_score = calculate_per_example_bas(is_correct, s)
        
        results.append({
            "id": idx,
            "question": problem,
            "ground_truth": gt_answer,
            "model_answer": model_ans,
            "confidence": s,
            "is_correct": is_correct,
            "bas_score": bas_score
        })
        print(f"[{idx+1}/{len(df)}] Correct: {is_correct} | Conf: {s:.4f} | BAS: {bas_score:.4f}")

    # Convert to DataFrame for aggregation
    res_df = pd.DataFrame(results)
    
    # 1. BAS_avg
    bas_avg = res_df['bas_score'].mean()
    
    # 2. Behavioral AUARC 
    # Sorting by confidence to calculate the Accuracy-Rejection curve
    res_df = res_df.sort_values(by='confidence', ascending=False)
    accuracies = []
    coverages = np.linspace(0.1, 1.0, 10) # 10 coverage points
    for c in coverages:
        subset_size = int(c * len(res_df))
        subset = res_df.head(subset_size)
        acc = subset['is_correct'].mean()
        accuracies.append(acc)
    auarc = np.mean(accuracies)

    # 3. Abstention Rate (at a default risk threshold, e.g., t=0.5)
    # In BAS, a model 'abstains' if s < t. For reporting, we can use the mean s.
    abstention_rate = (res_df['confidence'] < 0.5).mean()

    # 4. Explicit Abstention Rate (model output is exactly 'ABSTAIN')
    explicit_abstention_rate = (res_df['model_answer'] == "ABSTAIN").mean()

    # Final Report
    report = {
        "Model": model_name,
        "Benchmark": "AIME 2024/25",
        "Total Examples": len(df),
        "Accuracy": res_df['is_correct'].mean(),
        "BAS_avg": float(bas_avg),
        "Behavioral AUARC": float(auarc),
        "Abstention Rate (t=0.5)": float(abstention_rate),
        "Explicit Abstention Rate": float(explicit_abstention_rate)
    }

    # Save outputs
    # Create valid filename from model name
    valid_model_name = re.sub(r'[^\w\-_]', '_', model_name).lower()
    
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"{valid_model_name}_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, f"{valid_model_name}_summary_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("\n--- EVALUATION COMPLETE ---")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BAS Eval on AIME 2024/2025")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the model (for reporting/filenames)")
    parser.add_argument("--deployment_name", type=str, default=DEFAULT_DEPLOYMENT_NAME, help="Azure OpenAI deployment name")
    parser.add_argument("--endpoint", type=str, default=None, help="Azure endpoint or custom base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    
    args = parser.parse_args()
    
    # Strip whitespace/invisible characters from all string arguments
    if args.model_name: args.model_name = args.model_name.strip()
    if args.deployment_name: args.deployment_name = args.deployment_name.strip()
    if args.endpoint: args.endpoint = args.endpoint.strip()
    if args.api_key: args.api_key = args.api_key.strip()

    run_eval(args.model_name, args.deployment_name, args.endpoint, args.api_key)