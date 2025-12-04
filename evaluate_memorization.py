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


# Config 
use_quantized_model = True
device = "cuda:0"
k = 16
number_of_tests = 500 
save_results_to_file = True
save_filename = "memorization_results.jsonl"





def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate memorization of a language model.")
    parser.add_argument("--use_quantized_model", type=str, default=str(use_quantized_model),\
                         choices=["True", "False", "true", "false"], help="Whether to use the quantized model.")
    parser.add_argument("--k", type=int, default=k, help="Number of prompt tokens.")
    parser.add_argument("--num_tests", type=int, default=number_of_tests, help="Number of test samples to evaluate.")
    parser.add_argument("--save_file", type=str, default=save_filename, help="File to save results.")
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



def save_results_to_json(results, filename):
    with open(filename, "a") as f:
        f.write(json.dumps(results) + "\n")



def main():
    args = parse_arguments()
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
        device_map={"": device},
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
    for sample in tqdm(dataset_subset):
        tokens = sample["tokens"]
        accuracy, exact_match = test_memorization(tokens, k, model, tokenizer)
        accuracies.append(accuracy)
        if exact_match:
            exact_matches += 1

    overall_accuracy = sum(accuracies) / len(accuracies)
    accuracy_standard_deviation = np.std(accuracies)
    average_correct_tokens = overall_accuracy * (len(tokens) - k)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    exact_match_percentage = exact_matches / number_of_tests

    results = {
        "model_name": model_name,
        "k": k,
        "generated_tokens_per_test": len(tokens) - k,
        "sample_size": number_of_tests,
        "overall_accuracy": overall_accuracy,
        "average_correct_tokens": average_correct_tokens,
        "accuracy_standard_deviation": accuracy_standard_deviation,
        "exact_match_percentage": exact_match_percentage,
        "timestamp": timestamp
    }
    if save_results_to_file:
        save_results_to_json(results, save_filename)

    print("="*70)
    print(f"Using {model_name}: Overall memorization accuracy for k={k} over {number_of_tests} tests: \
          {overall_accuracy} -> {average_correct_tokens} correct tokens on average")
    print("="*70)
    


if __name__ == "__main__":
    main()
