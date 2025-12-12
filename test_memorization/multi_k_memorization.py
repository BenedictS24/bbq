from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import time

# --- Configuration ---
use_quantized_model = False
device = "cuda:0"
eval_token_count = 16
k_step_size = 4
start_k = 6
end_k = 46
number_of_tests = 1000 
save_results_to_file = True
save_filename = "./experiment_data/k8-48_memorization_results.jsonl"
test_sequence_length = 64

# --- Helper Functions ---

def load_eval_dataset():
    dataset = load_dataset(
        "EleutherAI/pythia-memorized-evals",
        split="duped.12b",
        cache_dir="/mnt/storage2/student_data/bstahl/bbq/pythia-12b_memorized-evals"
        )
    print(f"Loaded evaluation dataset with {len(dataset)} examples.")
    return dataset

def setup_model_and_tokenizer(use_quantized, device):
    if use_quantized:
        model_name = "./pythia-12b-4bit-bbq"
    else:
        model_name = "EleutherAI/pythia-12b"
    
    print(f"Loading model: {model_name}...")
    
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,        
        device_map={"": device},
        cache_dir=f"./{model_name.split('/')[-1]}",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"./{model_name.split('/')[-1]}",
    )
    
    return model, tokenizer, model_name

def evaluate_output(output_tokens, expected_tokens):
    correct = 0
    successive_correct = 0
    mismatch_found = False
    total = len(expected_tokens)
    exact_match = False

    if total == 0:
        return 0.0, False, 0
    
    for out_token, exp_token in zip(output_tokens, expected_tokens):
        if out_token == exp_token:
            correct += 1
            if not mismatch_found:
                successive_correct += 1
        else:
            mismatch_found = True
            
    accuracy = correct / total
    if correct == total:
        exact_match = True
    
    return accuracy, exact_match, successive_correct

def test_memorization(test_sequence, k, model, tokenizer):
    separation_index= len(test_sequence) - eval_token_count
    prompt_tokens = test_sequence[separation_index-k:separation_index]
    expected_tokens = test_sequence[separation_index:]
    
    input_tokens = torch.tensor([prompt_tokens]).to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids = input_tokens, 
            max_new_tokens = len(expected_tokens),
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id
        )

    full_output_tokens = output[0].tolist()
    output_tokens = full_output_tokens[len(prompt_tokens):]

    return evaluate_output(output_tokens, expected_tokens)

def run_inference_loop(dataset, k, model, tokenizer):
    """Runs the memorization test over the dataset and returns raw stats."""
    accuracies = []
    successive_counts = []
    exact_matches = 0

    for sample in tqdm(dataset, leave=False):
        tokens = sample["tokens"]
        accuracy, exact_match, successive = test_memorization(tokens, k, model, tokenizer)
        accuracies.append(accuracy)
        successive_counts.append(successive)
        if exact_match:
            exact_matches += 1
            
    return accuracies, successive_counts, exact_matches

def compile_results(accuracies, successive_counts, exact_matches, runtime, model_name, k, sample_size):
    """Calculates averages and formats the results dictionary."""
    overall_accuracy = sum(accuracies) / len(accuracies)
    average_successive_correct = sum(successive_counts) / len(successive_counts)
    accuracy_standard_deviation = np.std(accuracies)
    average_correct_tokens = overall_accuracy * eval_token_count
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    exact_match_percentage = exact_matches / sample_size

    results = {
        "model_name": model_name,
        "k": k,
        "sample_size": sample_size,
        "overall_accuracy": round(overall_accuracy, 4),
        "average_correct_tokens": round(average_correct_tokens, 4),
        "average_successive_correct_tokens": round(average_successive_correct, 4),
        "accuracy_standard_deviation": round(accuracy_standard_deviation, 4),
        "exact_match_percentage": round(exact_match_percentage, 4),
        "runtime_seconds": round(runtime, 4),
        "timestamp": timestamp
    }
    return results

def save_results_to_json(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        f.write(json.dumps(results) + "\n")

def print_summary(results):
    print("="*70)
    print(f"Using {results['model_name']}: Overall memorization accuracy for k={results['k']} "
          f"over {results['sample_size']} tests: \n"
          f"      {results['overall_accuracy']:.4f} -> {results['average_correct_tokens']:.4f} correct tokens on average")
    print(f"Average Successive Correct: {results['average_successive_correct_tokens']:.4f}")
    print(f"Runtime: {results['runtime_seconds']:.4f} seconds")
    print("="*70)

# --- Main Logic ---

def main(k, use_quantized):
    # 1. Setup Resources
    model, tokenizer, model_name = setup_model_and_tokenizer(use_quantized, device)
    
    # 2. Prepare Data
    dataset = load_eval_dataset()
    dataset_subset = dataset.select(range(number_of_tests))

    # 3. Run Inference
    print(f"Starting evaluation for k={k}...")
    start_time = time.time()
    accuracies, successive_counts, exact_matches = run_inference_loop(dataset_subset, k, model, tokenizer)
    runtime = time.time() - start_time

    # 4. Process Results
    results = compile_results(
        accuracies, successive_counts, exact_matches, 
        runtime, model_name, k, number_of_tests
    )

    # 5. Save & Print
    if save_results_to_file:
        save_results_to_json(results, save_filename)
    print_summary(results)


if __name__ == "__main__":
    if (test_sequence_length - eval_token_count) % k_step_size != 0:
        print("Error: (test_sequence_length - eval_token_count) must be divisible by k_step_size")
        exit(1)
        
    # Standard Model Pass
    for k in tqdm(range(start_k, end_k + 1, k_step_size)):
        main(k, use_quantized=False)

    # Quantized Model Pass
    for k in tqdm(range(start_k, end_k + 1, k_step_size)):
        main(k, use_quantized=True)