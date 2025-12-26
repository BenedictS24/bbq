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

MODEL_BASE_DIR = "/home/bstahl/bbq/models" 
# MODEL_LIST = [
#     "pythia-12b-duped-step143000",
#     "pythia-12b-duped-step143000-8bit",
#     "pythia-12b-duped-step143000-fp4bit",
#     "pythia-12b-duped-step143000-nf4bit"
# ]
MODEL_LIST = [
    "pythia-12b-deduped-step143000",
    "pythia-12b-deduped-step143000-8bit",
    "pythia-12b-deduped-step143000-fp4bit",
    "pythia-12b-deduped-step143000-nf4bit"
]
DATASET_ID = "EleutherAI/pythia-memorized-evals"
DATASET_SPLIT = "deduped.12b"  # Change to "duped.12b" if testing standard models
DATASET_CACHE = "/mnt/storage2/student_data/bstahl/bbq/test_memorization/pythia-12b_memorized-evals"

CONTEXT_TOKEN_POSITION = "end_of_sequence"  # Where the target tokens are located in the sequence ("start_of_sequence" or "end_of_sequence")
DEVICE = "cuda:0"          # GPU device to use
EVAL_TOKEN_COUNT = 16      # How many tokens the model should generate (the target continuation length)
K_STEP_SIZE = 4            # Step size for the loop over 'k' (context length)
START_K = 4               
END_K = 48                 
NUMBER_OF_TESTS = 2000     # How many samples from the dataset to evaluate per setting
SAVE_RESULTS_TO_FILE = True
SAVE_DIR = "/home/bstahl/bbq/data/experiment_data/"
TEST_SEQUENCE_LENGTH = 64  # Total length of the sample (Context + Target)
RANDOM_SEED = 42         


def generate_filename():
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    first_model = MODEL_LIST[0].split('/')[-1]
    model_count = len(MODEL_LIST)
    model_info = f"{first_model}_plus_{model_count-1}_others" if model_count > 1 else first_model
    
    filename = (
        f"mem_eval_{model_info}_k{START_K}-{END_K}_"
        f"{CONTEXT_TOKEN_POSITION}_{timestamp}.jsonl"
    )
    return os.path.join(SAVE_DIR, filename)


def load_eval_dataset():
    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        cache_dir=DATASET_CACHE
        )
    print(f"Loaded evaluation dataset '{DATASET_ID}' (split: {DATASET_SPLIT}) with {len(dataset)} examples.")
    return dataset


def setup_model_and_tokenizer(model_path, device):
    print(f"Loading model from: {model_path}...")
    
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
    correct = 0
    successive_correct = 0
    mismatch_found = False
    total = len(expected_tokens)
    exact_match = False

    if total == 0:
        return 0.0, False, 0, 0
    
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
    
    return accuracy, exact_match, successive_correct, correct


def test_memorization(test_sequence, k, model, tokenizer):
    if CONTEXT_TOKEN_POSITION == "end_of_sequence": 
        separation_index = len(test_sequence) - EVAL_TOKEN_COUNT
        prompt_tokens = test_sequence[separation_index-k : separation_index]
        expected_tokens = test_sequence[separation_index:]

    elif CONTEXT_TOKEN_POSITION == "start_of_sequence":
        separation_index = k
        prompt_tokens = test_sequence[:separation_index]
        expected_tokens = test_sequence[separation_index:separation_index + EVAL_TOKEN_COUNT]

    else:
        raise ValueError("Invalid CONTEXT_TOKEN_POSITION value. Use 'start_of_sequence' or 'end_of_sequence'.")
    
    input_tokens = torch.tensor([prompt_tokens]).to(DEVICE)
    
    attention_mask = torch.ones_like(input_tokens).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids = input_tokens, 
            attention_mask = attention_mask,
            max_new_tokens = len(expected_tokens),
            do_sample = False, 
            pad_token_id = tokenizer.eos_token_id
        )

    full_output_tokens = output[0].tolist()
    
    output_tokens = full_output_tokens[len(prompt_tokens):]

    return evaluate_output(output_tokens, expected_tokens)


def run_inference_loop(dataset, k, model, tokenizer):
    accuracies = []
    successive_counts = []
    correct_counts = []  
    exact_matches = 0

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
    overall_accuracy = sum(accuracies) / len(accuracies)
    average_successive_correct = sum(successive_counts) / len(successive_counts)
    
    accuracy_standard_deviation = np.std(accuracies)
    successive_correct_standard_deviation = np.std(successive_counts)
    
    average_correct_tokens = overall_accuracy * EVAL_TOKEN_COUNT
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    exact_match_percentage = exact_matches / sample_size

    token_distribution = np.bincount(correct_counts, minlength=EVAL_TOKEN_COUNT + 1).tolist()

    gpu_details = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_details.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        gpu_details.append("No GPU available")

    results = {
        "model_name": model_name,
        "k": k,
        "context_token_position": CONTEXT_TOKEN_POSITION,
        "sample_size": sample_size,
        "random_seed": RANDOM_SEED,

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

        "timestamp": timestamp,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_details": gpu_details
    }
    return results


def save_results_to_json(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "a") as f:
        f.write(json.dumps(results) + "\n")


def print_summary(results):
    console = Console()
    gb = lambda b: b / (1024**3)

    console.print("-" * 40, style="dim")
    console.print(f"[bold cyan]{results['model_name']}[/] (k={results['k']}, n={results['sample_size']})")
    console.print(f"Accuracy:   [bold green]{results['overall_accuracy']:.4f}[/] [dim]±{results['accuracy_standard_deviation']:.3f}[/]")
    console.print(f"Successive: [bold blue]{results['average_successive_correct_tokens']:.4f}[/] [dim]±{results['successive_correct_standard_deviation']:.3f}[/]")
    console.print(f"VRAM Peak:  [bold yellow]{gb(results['peak_gpu_memory_byte']):.2f} GB[/]")
    console.print(f"Runtime:    [bold magenta]{results['runtime_seconds']:.2f}s[/]")
    console.print(f"[dim]Dist: {results['correct_token_distribution']}[/]")
    console.print("-" * 40, style="dim")    


def main(model_name, k):
    full_model_path = os.path.join(MODEL_BASE_DIR, model_name)

    model, tokenizer = setup_model_and_tokenizer(full_model_path, DEVICE)
    
    # https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.get_memory_footprint
    model_footprint_byte = model.get_memory_footprint()

    print(f"Starting evaluation for model: {model_name} | k={k}...")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache() 

    start_time = time.time()
    
    accuracies, successive_counts, exact_matches, correct_counts = run_inference_loop(dataset_subset, k, model, tokenizer)
    
    runtime = time.time() - start_time

    peak_memory_byte = 0.0
    if torch.cuda.is_available():
        peak_memory_byte = torch.cuda.max_memory_allocated(DEVICE)

    results = compile_results(
        accuracies, successive_counts, exact_matches, correct_counts,
        runtime, model_name, k, NUMBER_OF_TESTS,
        model_footprint_byte, peak_memory_byte
    )

    if SAVE_RESULTS_TO_FILE:
        save_results_to_json(results, SAVE_FILENAME)
    print_summary(results)

    # https://discuss.pytorch.org/t/cuda-memory-not-released-by-torch-cuda-empty-cache/129913 
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    SAVE_FILENAME = generate_filename()
    
    dataset = load_eval_dataset()
    dataset_subset = dataset.shuffle(seed=RANDOM_SEED).select(range(NUMBER_OF_TESTS))
    if (TEST_SEQUENCE_LENGTH - EVAL_TOKEN_COUNT) % K_STEP_SIZE != 0:
        print("Error: (TEST_SEQUENCE_LENGTH - EVAL_TOKEN_COUNT) must be divisible by K_STEP_SIZE")
        exit(1)
        
    for model_name in MODEL_LIST:
        print(f"\n{'#'*30}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'#'*30}\n")
        
        for k in tqdm(range(START_K, END_K + 1, K_STEP_SIZE)):
            main(model_name, k)