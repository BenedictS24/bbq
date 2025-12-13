from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import time

"""
Memorization Evaluation Script

This script evaluates how well a list of Language Models (LLMs) have memorized 
specific training data. It does this by:
1. Loading a dataset of known "memorized" examples.
2. Prompting the model with the first part of a sequence (context).
3. Checking if the model can exactly reproduce the second part (continuation).
4. Saving detailed statistics (accuracy, exact match, successive tokens, distribution) to a JSONL file.
"""

# --- Configuration ---
# Directory containing your model checkpoints
model_base_dir = "/home/bstahl/bbq/models" 

# List of specific model folder names inside 'model_base_dir' to evaluate
model_list = [
    "pythia-12b-duped-step143000",
    "pythia-12b-duped-step143000-8bit",
    "pythia-12b-duped-step143000-fp4bit",
    "pythia-12b-duped-step143000-nf4bit"
]

device = "cuda:0"          # GPU device to use
eval_token_count = 16      # How many tokens the model should generate (the target continuation length)
k_step_size = 4            # Step size for the loop over 'k' (context length)
start_k = 4                # Minimum context length (k) to test
end_k = 48                 # Maximum context length (k) to test
number_of_tests = 1000     # How many samples from the dataset to evaluate per setting
save_results_to_file = True
save_filename = "/home/bstahl/bbq/data/experiment_data/k4-48_fp16-8bit-fp4-nf4_memorization_results.jsonl"
test_sequence_length = 64  # Total length of the sample (Context + Target)


# --- Helper Functions ---

def load_eval_dataset():
    """
    Loads the specific 'duped.12b' split from the memorized-evals dataset.
    This dataset contains sequences known to be duplicated in the training data.
    """
    dataset = load_dataset(
        "EleutherAI/pythia-memorized-evals",
        split="duped.12b",
        cache_dir="/mnt/storage2/student_data/bstahl/bbq/test_memorization/pythia-12b_memorized-evals"
        )
    print(f"Loaded evaluation dataset with {len(dataset)} examples.")
    return dataset


def setup_model_and_tokenizer(model_path, device):
    """
    Loads the model and tokenizer from the specific path provided.
    """
    print(f"Loading model from: {model_path}...")
    
    # Load model in float16 to save memory, mapped to the specified device
    model = GPTNeoXForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,        
        device_map={"": device},
        cache_dir=f"./{model_path.split('/')[-1]}", 
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=f"./{model_path.split('/')[-1]}",
    )
    
    return model, tokenizer


def evaluate_output(output_tokens, expected_tokens):
    """
    Compares the generated tokens against the expected ground truth.
    Returns:
      - accuracy: Percentage of tokens that matched position-wise.
      - exact_match: Boolean, True only if 100% of tokens matched.
      - successive_correct: Count of correct tokens from the start before the first error.
      - correct: The raw count of correct tokens (position-wise).
    """
    correct = 0
    successive_correct = 0
    mismatch_found = False
    total = len(expected_tokens)
    exact_match = False

    if total == 0:
        return 0.0, False, 0, 0
    
    # Pair up the generated token with the expected token
    for out_token, exp_token in zip(output_tokens, expected_tokens):
        if out_token == exp_token:
            correct += 1
            # Count how long the model stays on track from the very beginning
            if not mismatch_found:
                successive_correct += 1
        else:
            mismatch_found = True
            
    accuracy = correct / total
    if correct == total:
        exact_match = True
    
    return accuracy, exact_match, successive_correct, correct


def test_memorization(test_sequence, k, model, tokenizer):
    """
    Splits a sequence into a prompt (length k) and a target (expected result).
    Feeds the prompt to the model and compares the output.
    """
    # Calculate where to split the sequence based on how many tokens we want to predict
    separation_index = len(test_sequence) - eval_token_count
    
    # The Prompt: The 'k' tokens immediately preceding the target area
    prompt_tokens = test_sequence[separation_index-k : separation_index]
    
    # The Target: The actual tokens that followed the prompt in the training data
    expected_tokens = test_sequence[separation_index:]
    
    input_tokens = torch.tensor([prompt_tokens]).to(device)
    
    # Create a mask of 1s (keep all tokens) with the same shape as input_tokens
    attention_mask = torch.ones_like(input_tokens).to(device)

    # Generate the continuation (inference)
    with torch.no_grad():
        output = model.generate(
            input_ids = input_tokens, 
            attention_mask = attention_mask,
            max_new_tokens = len(expected_tokens),
            do_sample = False, # Greedy decoding (deterministic)
            pad_token_id = tokenizer.eos_token_id
        )

    full_output_tokens = output[0].tolist()
    
    # Slice the output to get only the NEW tokens generated by the model
    output_tokens = full_output_tokens[len(prompt_tokens):]

    return evaluate_output(output_tokens, expected_tokens)


def run_inference_loop(dataset, k, model, tokenizer):
    """
    Iterates through the dataset and collects raw stats for a specific 'k'.
    """
    accuracies = []
    successive_counts = []
    correct_counts = []  # Store raw count of correct tokens per sample
    exact_matches = 0

    # tqdm provides a progress bar for the loop
    for sample in tqdm(dataset, leave=False):
        tokens = sample["tokens"]
        accuracy, exact_match, successive, correct_count = test_memorization(tokens, k, model, tokenizer)
        
        accuracies.append(accuracy)
        successive_counts.append(successive)
        correct_counts.append(correct_count)
        
        if exact_match:
            exact_matches += 1
            
    return accuracies, successive_counts, exact_matches, correct_counts


def compile_results(accuracies, successive_counts, exact_matches, correct_counts, runtime, model_name, k, sample_size):
    """Calculates averages, formats the results dictionary, and gathers system info."""
    overall_accuracy = sum(accuracies) / len(accuracies)
    average_successive_correct = sum(successive_counts) / len(successive_counts)
    accuracy_standard_deviation = np.std(accuracies)
    average_correct_tokens = overall_accuracy * eval_token_count
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    exact_match_percentage = exact_matches / sample_size

    # Calculate distribution: Index i represents how many samples had exactly i correct tokens
    token_distribution = np.bincount(correct_counts, minlength=eval_token_count + 1).tolist()

    # --- System Info Gathering ---
    gpu_details = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_details.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        gpu_details.append("No GPU available")

    results = {
        "model_name": model_name,
        "k": k,
        "sample_size": sample_size,
        "overall_accuracy": round(overall_accuracy, 4),
        "average_correct_tokens": round(average_correct_tokens, 4),
        "average_successive_correct_tokens": round(average_successive_correct, 4),
        "accuracy_standard_deviation": round(accuracy_standard_deviation, 4),
        "exact_match_percentage": round(exact_match_percentage, 4),
        "correct_token_distribution": token_distribution,
        "runtime_seconds": round(runtime, 4),
        "timestamp": timestamp,
        # System Metadata
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_details": gpu_details
    }
    return results


def save_results_to_json(results, filename):
    # Ensure the directory exists before writing; avoids "FileNotFoundError"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Append ("a") to the file so we don't overwrite previous results
    with open(filename, "a") as f:
        f.write(json.dumps(results) + "\n")


def print_summary(results):
    print("="*70)
    print(f"Using {results['model_name']}: Overall memorization accuracy for k={results['k']} "
          f"over {results['sample_size']} tests: \n"
          f"      {results['overall_accuracy']:.4f} -> {results['average_correct_tokens']:.4f} correct tokens on average")
    print(f"Average Successive Correct: {results['average_successive_correct_tokens']:.4f}")
    print(f"Distribution (0 to {eval_token_count} correct): {results['correct_token_distribution']}")
    print(f"System: PyTorch {results['torch_version']} | CUDA {results['cuda_version']}")
    print(f"Runtime: {results['runtime_seconds']:.4f} seconds")
    print("="*70)


# --- Main Logic ---

def main(model_name, k):
    # Construct the full path by joining the base dir and the specific model folder
    full_model_path = os.path.join(model_base_dir, model_name)

    # 1. Setup Resources
    model, tokenizer = setup_model_and_tokenizer(full_model_path, device)
    
    # 2. Prepare Data (Select the first 'number_of_tests' samples)
    dataset = load_eval_dataset()
    dataset_subset = dataset.select(range(number_of_tests))

    # 3. Run Inference
    print(f"Starting evaluation for model: {model_name} | k={k}...")
    start_time = time.time()
    
    accuracies, successive_counts, exact_matches, correct_counts = run_inference_loop(dataset_subset, k, model, tokenizer)
    
    runtime = time.time() - start_time

    # 4. Process Results
    results = compile_results(
        accuracies, successive_counts, exact_matches, correct_counts,
        runtime, model_name, k, number_of_tests
    )

    # 5. Save & Print
    if save_results_to_file:
        save_results_to_json(results, save_filename)
    print_summary(results)


if __name__ == "__main__":
    # Validate that the sequence length math works out
    if (test_sequence_length - eval_token_count) % k_step_size != 0:
        print("Error: (test_sequence_length - eval_token_count) must be divisible by k_step_size")
        exit(1)
        
    # Outer Loop: Iterate through every model in the configuration list
    for model_name in model_list:
        print(f"\n{'#'*30}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'#'*30}\n")
        
        # Inner Loop: Iterate through different prompt lengths (k) for this specific model
        for k in tqdm(range(start_k, end_k + 1, k_step_size)):
            main(model_name, k)
