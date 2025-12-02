from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import json
import glob
import os
from tqdm import tqdm



use_quantized_model = True
device = "cuda:0"
k = 32



if use_quantized_model:
    model_name = "./pythia-12b-4bit-bbq"
else:
    model_name = "./pythia-12b"


test = [
  18728,
  27,
  2703,
  3,
  1054,
  26600,
  2244,
  568,
  17,
  3,
  13818,
  2203,
  27,
  45435,
  989,
  568,
  18,
  3,
  4725,
  187,
  50262,
  29,
  18728,
  27,
  10531,
  1416,
  568,
  13982,
  3,
  1511,
  568,
  18728,
  27,
  2703,
  3,
  1054,
  26600,
  2244,
  568,
  17,
  3,
  13818,
  2203,
  27,
  45435,
  989,
  568,
  19,
  3,
  4725,
  187,
  50264,
  870,
  18728,
  27,
  21934,
  31,
  187,
  50264,
  29,
  18728,
  27,
  15810,
  1416
]


print(f"Using model: {model_name}")
print(f"k = {k}")


model = GPTNeoXForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,        
    device_map={"": device},
    cache_dir=f"./{model_name.split('/')[-1]}",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=f"./{model_name.split('/')[-1]}",
)

def evaluate_output(output_tokens, expected_tokens):
    correct = 0
    total = len(expected_tokens)

    if total == 0:
        return 0.0
    
    for out_token, exp_token in zip(output_tokens, expected_tokens):
        if out_token == exp_token:
            correct += 1
    
    
    accuracy = correct / total
    return accuracy



def test_memorization(test_sequence, k):
    if k >= len(test_sequence):
        print("Error: sequence is shorter than k")
        return None
    
    prompt_tokens = test_sequence[:k]
    expected_tokens = test_sequence[k:]
    
    input_tokens = torch.tensor([prompt_tokens]).to(device)
    output = model.generate(
        input_ids = input_tokens, 
        max_new_tokens = len(expected_tokens),
        do_sample = False,
        pad_token_id = tokenizer.eos_token_id
    )

    full_output_tokens = output[0].tolist()
    output_tokens = full_output_tokens[len(prompt_tokens):]

    return evaluate_output(output_tokens, expected_tokens)



if __name__ == "__main__":
    accuracy = test_memorization(test, k)
    print(f"Memorization accuracy for k={k}: {accuracy}")