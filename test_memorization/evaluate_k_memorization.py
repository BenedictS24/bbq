from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
import glob
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np
import argparse


# --- CONFIGURATION ---
USE_QUANTIZED_MODEL = True
DEVICE = "cuda:0"
K = 32 
NUMBER_OF_TESTS = 500 
SAVE_RESULTS_TO_FILE = True
SAVE_FILENAME = "memorization_results.jsonl"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate memorization of a language model.")
    parser.add_argument("--use_quantized_model", type=str, default=str(USE_QUANTIZED_MODEL),
                        choices=["True", "False", "true", "false"], help="Whether to use the quantized model.")
    parser.add_argument("--k", type=int, default=K, help="Number of prompt tokens.")
    parser.add_argument("--num_tests", type=int, default=NUMBER_OF_TESTS, help="Number of test samples to evaluate.")
    parser.add_argument("--save_file", type=str, default=SAVE_FILENAME, help="File to save results.")
    args = parser.parse_args()
    args.use_quantized_model = args.use_quantized_model.lower() == "true"
    return args


def load_eval_dataset():
    dataset = load_dataset(
        "EleutherAI/pythia-memorized-evals",
        split="duped.12b",
        cache_dir="/mnt/storage2/student_data/bstahl/bbq/pythia-12b_memorized-evals"
        )

    print(f"Loaded evaluation dataset with {len(dataset)} examples.")
    return dataset


def evaluate_output(output_tokens, expected_tokens):
    correct = 0
    total = len(expected_tokens)
    exact_match = False

    if total == 0:
        return 0.0
    
    for out_token, exp_token in zip(output_tokens, expected_tokens):
        if out_token == exp_token:
            correct += 1
    accuracy = correct / total
    if correct == total:
        exact_match = True
    
    return accuracy, exact_match


def test_memorization(test_sequence, k, model, tokenizer):
    if k >= len(test_sequence):
        print("Error: sequence is shorter than k")
        return None
    
    prompt_tokens = test_sequence[:k]
    expected_tokens = test_sequence[k:]
    
    input_tokens = torch.tensor([prompt_tokens]).to(DEVICE)
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


def save_results_to_json(results, filename):
    with open(filename, "a") as f:
        f.write(json.dumps(results) + "\n")


def main():
    args = parse_arguments()
    
    # Map args to local variables
    use_quantized_model = args.use_quantized_model
    k = args.k
    number_of_tests = args.num_tests
    save_filename = args.save_file

    if use_quantized_model:
        model_name = "./pythia-12b-4bit-bbq"
    else:
        model_name = "EleutherAI/pythia-12b"

    print(f"Using model: {model_name}")
    print(f"k = {k}")
    print(f"Number of tests: {number_of_tests}")
    print(f"Save results to file: {save_filename}")

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,        
        device_map={"": DEVICE},
        cache_dir=f"./{model_name.split('/')[-1]}",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"./{model_name.split('/')[-1]}",
    )

    dataset = load_eval_dataset()
    dataset_subset = dataset.select(range(number_of_tests))

    accuracies = []
    exact_matches = 0
    
    # Note: 'tokens' variable was undefined in your original print statement scope
    # I've added a fallback length here to ensure the calculation inside the loop works for the average
    token_length_for_avg = 0 

    for sample in tqdm(dataset_subset):
        tokens = sample["tokens"]
        token_length_for_avg = len(tokens) # Capture length for final stats
        accuracy, exact_match = test_memorization(tokens, k, model, tokenizer)
        accuracies.append(accuracy)
        if exact_match:
            exact_matches += 1

    overall_accuracy = sum(accuracies) / len(accuracies)
    accuracy_standard_deviation = np.std(accuracies)
    
    # Calculate average correct tokens based on the last seen token length (assuming uniform length)
    # or you might want to average this inside the loop if lengths vary.
    average_correct_tokens = overall_accuracy * (token_length_for_avg - k)
    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    exact_match_percentage = exact_matches / number_of_tests

    results = {
        "model_name": model_name,
        "k": k,
        "generated_tokens_per_test": token_length_for_avg - k,
        "sample_size": number_of_tests,
        "overall_accuracy": round(overall_accuracy, 3),
        "average_correct_tokens": round(average_correct_tokens, 3),
        "accuracy_standard_deviation": round(accuracy_standard_deviation, 3),
        "exact_match_percentage": round(exact_match_percentage, 3),
        "timestamp": timestamp
    }
    
    if SAVE_RESULTS_TO_FILE:
        save_results_to_json(results, save_filename)

    print("="*70)
    print(f"Using {model_name}: Overall memorization accuracy for k={k} over {number_of_tests} tests: \
          {overall_accuracy:.3f} -> {average_correct_tokens:.3f} correct tokens on average")
    print("="*70)
    

if __name__ == "__main__":
    main()