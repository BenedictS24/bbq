from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import os

# --- CONFIG ---
MODEL_BASE_DIR = "/home/bstahl/bbq/models" 
MODEL_LIST = [
    "pythia-12b-duped-step143000",
    "pythia-12b-duped-step143000-8bit"
]

DEVICE = "cuda:0"
PROMPT = "Hello, my name is"
MAX_NEW_TOKENS = 100


def setup_model_and_tokenizer(model_path, device):
    print(f"\n--- Loading: {model_path} ---")
    
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


def run_test(model_name):
    full_model_path = os.path.join(MODEL_BASE_DIR, model_name)
    model, tokenizer = setup_model_and_tokenizer(full_model_path, DEVICE)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        tokens = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"RESULT:\n{output}\n{'-'*30}")

if __name__ == "__main__":
    for model_folder in MODEL_LIST:
        run_test(model_folder)