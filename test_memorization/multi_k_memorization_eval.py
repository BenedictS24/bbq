from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import time
import gc
from rich.console import Console

"""
Memorization Evaluation Script

This script evaluates how well a list of Language Models (LLMs) have memorized 
specific training data. It does this by:
1. Loading a dataset of known "memorized" examples.
2. Prompting the model with the first part of a sequence (context).
3. Checking if the model can exactly reproduce the second part (continuation).
4. Saving detailed statistics (accuracy, exact match, successive tokens, distribution) 
   AND memory usage (VRAM footprint, Peak Usage) to a JSONL file.
"""

# --- CONFIGURATION ---
# Directory containing your model checkpoints
MODEL_BASE_DIR = "/home/bstahl/bbq/models" 

# List of specific model folder names inside 'MODEL_BASE_DIR' to evaluate
MODEL_LIST = [
    "pythia-12b-duped-step143000",
    "pythia-12b-duped-step143000-8bit",
    "pythia-12b-duped-step143000-fp4bit",
    "pythia-12b-duped-step143000-nf4bit"
]
CONTEXT_TOKEN_POSITION = "end_of_sequence"  # Where the target tokens are located in the sequence ("start_of_sequence" or "end_of_sequence")
DEVICE = "cuda:0"          # GPU device to use
EVAL_TOKEN_COUNT = 16      # How many tokens the model should generate (the target continuation length)
K_STEP_SIZE = 4            # Step size for the loop over 'k' (context length)
START_K = 4                # Minimum context length (k) to test
END_K = 48                 # Maximum context length (k) to test
NUMBER_OF_TESTS = 1000     # How many samples from the dataset to evaluate per setting
SAVE_RESULTS_TO_FILE = True
SAVE_DIR = "/home/bstahl/bbq/data/experiment_data/"
TEST_SEQUENCE_LENGTH = 64  # Total length of the sample (Context + Target)
RANDOM_SEED = 42         # For reproducibility

# --- Helper Functions ---

def generate_filename():
    """
    Generates an abbreviated filename: 
    memorization_eval + [FirstModelName] + [Count] + [k-range] + [position] + [timestamp]
    """
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    
    # Get the name of the first model and count the rest
    first_model = MODEL_LIST[0].split('/')[-1]
    model_count = len(MODEL_LIST)
    
    # Abbreviated model info
    model_info = f"{first_model}_plus_{model_count-1}_others" if model_count > 1 else first_model
    
    # Construct the final string
    filename = (
        f"mem_eval_{model_info}_k{START_K}-{END_K}_"
        f"{CONTEXT_TOKEN_POSITION}_{timestamp}.jsonl"
    )
    return os.path.join(SAVE_DIR, filename)

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
    # Note: Quantized models (bitsandbytes) usually require load_in_8bit=True etc., 
    # but assuming these are pre-saved quantized checkpoints or handled via config.
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
    if CONTEXT_TOKEN_POSITION == "end_of_sequence": 
        # Calculate where to split the sequence based on how many tokens we want to predict
        separation_index = len(test_sequence) - EVAL_TOKEN_COUNT
        # The Prompt: The 'k' tokens immediately preceding the target area
        prompt_tokens = test_sequence[separation_index-k : separation_index]
        # The Target: The actual tokens that followed the prompt in the training data
        expected_tokens = test_sequence[separation_index:]

    elif CONTEXT_TOKEN_POSITION == "start_of_sequence":
        # The Prompt: The first 'k' tokens of the sequence
        separation_index = k
        # The Prompt
        prompt_tokens = test_sequence[:separation_index]
        # The Target
        expected_tokens = test_sequence[separation_index:separation_index + EVAL_TOKEN_COUNT]

    else:
        raise ValueError("Invalid CONTEXT_TOKEN_POSITION value. Use 'start_of_sequence' or 'end_of_sequence'.")
    
    input_tokens = torch.tensor([prompt_tokens]).to(DEVICE)
    
    # Create a mask of 1s (keep all tokens) with the same shape as input_tokens
    attention_mask = torch.ones_like(input_tokens).to(DEVICE)

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
    for sample in tqdm(dataset, leave=False, desc=f"Evaluating k={k}"):
        tokens = sample["tokens"]
        accuracy, exact_match, successive, correct_count = test_memorization(tokens, k, model, tokenizer)
        
        accuracies.append(accuracy)
        successive_counts.append(successive)
        correct_counts.append(correct_count)
        
        if exact_match:
            exact_matches += 1
            
    return accuracies, successive_counts, exact_matches, correct_counts


def compile_results(accuracies, successive_counts, exact_matches, correct_counts, runtime, 
                    model_name, k, sample_size, model_footprint_byte, peak_memory_byte):
    """Calculates averages, formats the results dictionary, and gathers system info."""
    overall_accuracy = sum(accuracies) / len(accuracies)
    average_successive_correct = sum(successive_counts) / len(successive_counts)
    
    # Standard Deviations
    accuracy_standard_deviation = np.std(accuracies)
    successive_correct_standard_deviation = np.std(successive_counts)
    
    average_correct_tokens = overall_accuracy * EVAL_TOKEN_COUNT
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    exact_match_percentage = exact_matches / sample_size

    # Calculate distribution: Index i represents how many samples had exactly i correct tokens
    token_distribution = np.bincount(correct_counts, minlength=EVAL_TOKEN_COUNT + 1).tolist()

    # --- System Info Gathering ---
    gpu_details = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_details.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        gpu_details.append("No GPU available")

    results = {
        # Experiment Config
        "model_name": model_name,
        "k": k,
        "context_token_position": CONTEXT_TOKEN_POSITION,
        "sample_size": sample_size,
        "random_seed": RANDOM_SEED,

        # Evaluation Metrics
        "overall_accuracy": round(overall_accuracy, 4),
        "average_correct_tokens": round(average_correct_tokens, 4),
        "average_successive_correct_tokens": round(average_successive_correct, 4),
        "accuracy_standard_deviation": round(accuracy_standard_deviation, 4),
        "successive_correct_standard_deviation": round(successive_correct_standard_deviation, 4),
        "exact_match_percentage": round(exact_match_percentage, 4),
        "correct_token_distribution": token_distribution,
        "runtime_seconds": round(runtime, 4),
        "model_footprint_byte": int(model_footprint_byte),
        "peak_gpu_memory_byte": int(peak_memory_byte),

        # System Info
        "timestamp": timestamp,
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
    console = Console()
    gb = lambda b: b / (1024**3)

    console.print("-" * 40, style="dim")
    console.print(f"[bold cyan]{results['model_name']}[/] (k={results['k']}, n={results['sample_size']})")
    
    # Core Metrics
    console.print(f"Accuracy:   [bold green]{results['overall_accuracy']:.4f}[/] [dim]±{results['accuracy_standard_deviation']:.3f}[/]")
    console.print(f"Successive: [bold blue]{results['average_successive_correct_tokens']:.4f}[/] [dim]±{results['successive_correct_standard_deviation']:.3f}[/]")
    
    # Resources
    console.print(f"VRAM Peak:  [bold yellow]{gb(results['peak_gpu_memory_byte']):.2f} GB[/]")
    console.print(f"Runtime:    [bold magenta]{results['runtime_seconds']:.2f}s[/]")
    
    # Distribution (Simple)
    console.print(f"[dim]Dist: {results['correct_token_distribution']}[/]")
    console.print("-" * 40, style="dim")    


# --- Main Logic ---

def main(model_name, k):
    # Construct the full path by joining the base dir and the specific model folder
    full_model_path = os.path.join(MODEL_BASE_DIR, model_name)

    # 1. Setup Resources
    model, tokenizer = setup_model_and_tokenizer(full_model_path, DEVICE)
    
    # https://huggingface.co/docs/transformers/en/main_classes/model
    # get_memory_footprint() returns bytes. Convert to GB.
    model_footprint_byte = model.get_memory_footprint()

    # 2. Run Inference
    print(f"Starting evaluation for model: {model_name} | k={k}...")
    
    # Reset Peak Memory Tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache() # Clean cache to get accurate peak reading for this run

    start_time = time.time()
    
    accuracies, successive_counts, exact_matches, correct_counts = run_inference_loop(dataset_subset, k, model, tokenizer)
    
    runtime = time.time() - start_time

    # Capture Peak Memory
    peak_memory_byte = 0.0
    if torch.cuda.is_available():
        # max_memory_allocated returns the peak memory used by tensors since the last reset
        peak_memory_byte = torch.cuda.max_memory_allocated(DEVICE)

    # 4. Process Results
    results = compile_results(
        accuracies, successive_counts, exact_matches, correct_counts,
        runtime, model_name, k, NUMBER_OF_TESTS,
        model_footprint_byte, peak_memory_byte
    )

    # 5. Save & Print
    if SAVE_RESULTS_TO_FILE:
        save_results_to_json(results, SAVE_FILENAME)
    print_summary(results)

    # https://discuss.pytorch.org/t/cuda-memory-not-released-by-torch-cuda-empty-cache/129913 
    # Delete model and clear cache to free memory for the next iteration/model
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    SAVE_FILENAME = generate_filename()
    
    dataset = load_eval_dataset()
    dataset_subset = dataset.shuffle(seed=RANDOM_SEED).select(range(NUMBER_OF_TESTS))
    # Validate that the sequence length math works out
    if (TEST_SEQUENCE_LENGTH - EVAL_TOKEN_COUNT) % K_STEP_SIZE != 0:
        print("Error: (TEST_SEQUENCE_LENGTH - EVAL_TOKEN_COUNT) must be divisible by K_STEP_SIZE")
        exit(1)
        
    # Outer Loop: Iterate through every model in the configuration list
    for model_name in MODEL_LIST:
        print(f"\n{'#'*30}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'#'*30}\n")
        
        # Inner Loop: Iterate through different prompt lengths (k) for this specific model
        for k in tqdm(range(START_K, END_K + 1, K_STEP_SIZE)):
            main(model_name, k)
