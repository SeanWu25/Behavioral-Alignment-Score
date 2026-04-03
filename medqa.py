import os
import json
import numpy as np
import pandas as pd
import re
import argparse

# Configuration
from llm_client import LLMClient

# Configuration
DEFAULT_MODEL_NAME = ""
DEFAULT_DEPLOYMENT_NAME = ""
OUTPUT_DIR = "output_medqa_colm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Using 'custom' provider to match the original OpenAI(base_url=...) behavior
# Using 'custom' provider to match the original OpenAI(base_url=...) behavior
# client = LLMClient(provider="custom") -> Moved inside run_eval

def get_model_response(client, question, options, deployment_name):
    """
    Queries the model for a medical answer/abstention and a confidence score.
    Tailored for MedQA multiple-choice format per the provided JSONL schema.
    """
    # Sanitize inputs to avoid ASCII encoding errors with some clients/environments
    # Sanitize inputs to avoid ASCII encoding errors with some clients/environments
    def sanitize(text):
        if not isinstance(text, str): return str(text)
        # Encode to ASCII, ignoring errors, then decode back to string
        return text.encode('ascii', 'ignore').decode('ascii')

    question = sanitize(question)
    options = {k: sanitize(v) for k, v in options.items()}
    formatted_options = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    # System prompt designed to elicit calibrated uncertainty
    system_prompt = (
        "You are a medical expert assisting with USMLE-style questions. "
        "Analyze the clinical case and select the most appropriate option.\n\n"
        "At the end of your response, provide your final decision in a "
        "block starting with '### FINAL DECISION'.\n\n"
        "Provide the single letter corresponding to the correct option "
        "and a confidence score.\n\n"
        "Format:\n"
        "### FINAL DECISION\n"
        "Answer: [Letter A-D]\n"
        "Confidence: [0-1 representing the probability your answer is correct]"
    )
    
    user_input = f"Question: {question}\n\nOptions:\n{formatted_options}"
    # Final safety measure
    user_input = sanitize(user_input)

    print(user_input)
    try:
        completion = client.chat_completion(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0  # Consistent with selective prediction evaluation
        )
        content = completion.choices[0].message.content
        #print(content)
        # Parse Confidence: Look for the LAST occurrence
        conf_matches = re.findall(r"Confidence:\s*([0-9.]+)", content)
        confidence = float(conf_matches[-1]) if conf_matches else 0.0
        
        # Clamp per Algorithm 1: eps to avoid log(0)
        confidence = max(0.0001, min(0.9999, confidence))
        
        # Parse Answer: Targets letters A-D
        ans_matches = re.findall(r"Answer:\s*([A-D])", content, re.IGNORECASE)
        raw_ans = ans_matches[-1].upper() if ans_matches else "A"
        
        return raw_ans, confidence, content
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return "A", 0.0001, str(e)

def calculate_per_example_bas(is_correct, s):
    """
    Implements the analytical BAS utility from Equation 4 of the paper.
    """
    if is_correct:
        return s  #
    else:
        # Penalty for overconfident errors
        return s + np.log(1 - s)

def run_eval(model_name, deployment_name, input_file, endpoint=None, api_key=None):
    # Auto-detect provider
    provider = "azure"
    if endpoint and ("/v1" in endpoint):
        provider = "custom"
        print(f"Endpoint contains '/v1', switching to provider='{provider}'")
    elif endpoint and ("anthropic" in endpoint):
        provider = "anthropic_azure"
        print(f"Endpoint contains 'anthropic', switching to provider='{provider}'")

    client = LLMClient(provider=provider, base_url=endpoint, api_key=api_key)
    
    if not deployment_name and model_name:
        deployment_name = model_name
        print(f"Deployment name not provided. Using model_name '{model_name}' as deployment name.")

    results = []
    
    # Correctly parsing the uploaded .jsonl format
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Starting MedQA BAS Eval on {len(data)} cases...")

    for idx, item in enumerate(data):
        question = item['question']
        options = item['options']
        gt_answer = item['answer_idx'] # Accessing "answer_idx": "B"
        
        model_ans, s, _ = get_model_response(client, question, options, deployment_name)
        print("model answer: ", model_ans)
        print("ground truth: ", gt_answer)
        is_correct = (model_ans == gt_answer)
        print("is correct: ", is_correct)
        bas_score = calculate_per_example_bas(is_correct, s)
        
        results.append({
            "id": idx,
            "question": question,
            "ground_truth": gt_answer,
            "model_answer": model_ans,
            "confidence": s,
            "is_correct": is_correct,
            "bas_score": bas_score
        })
        
        print(f"[{idx+1}/{len(data)}] Correct: {is_correct} | Conf: {s:.4f} | BAS: {bas_score:.4f}")

    res_df = pd.DataFrame(results)
    
    # 1. BAS_avg
    bas_avg = res_df['bas_score'].mean()
    
    # 2. Behavioral AUARC (Accuracy-Rejection Curve)
    res_df = res_df.sort_values(by='confidence', ascending=False)
    accuracies = []
    for c in np.linspace(0.1, 1.0, 10):
        subset = res_df.head(int(c * len(res_df)))
        accuracies.append(subset['is_correct'].mean())
    auarc = np.mean(accuracies)

    # Final Report output
    report = {
        "Model": model_name,
        "Benchmark": "MedQA (USMLE)",
        "Total Examples": len(data),
        "Accuracy": float(res_df['is_correct'].mean()),
        "BAS_avg": float(bas_avg),
        "Behavioral AUARC": float(auarc),
        "Abstention Rate (t=0.5)": float((res_df['confidence'] < 0.5).mean())
    }

    valid_name = re.sub(r'[^\w\-_]', '_', model_name).lower()
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"medqa_{valid_name}_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, f"medqa_{valid_name}_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("\n--- MEDQA EVALUATION COMPLETE ---")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BAS Eval on MedQA JSONL")
    parser.add_argument("--input", type=str, default="benchmark/medqa.jsonl")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--deployment_name", type=str, default=DEFAULT_DEPLOYMENT_NAME)
    parser.add_argument("--endpoint", type=str, default=None, help="Azure endpoint or custom base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    args = parser.parse_args()
    
    # Strip whitespace/invisible characters
    if args.model_name: args.model_name = args.model_name.strip()
    if args.deployment_name: args.deployment_name = args.deployment_name.strip()
    if args.input: args.input = args.input.strip()
    if args.endpoint: args.endpoint = args.endpoint.strip()
    if args.api_key: args.api_key = args.api_key.strip()

    run_eval(args.model_name, args.deployment_name, args.input, args.endpoint, args.api_key)