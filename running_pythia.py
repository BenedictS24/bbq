from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch


use_quantized_model = True 

if use_quantized_model:
    model_name = "./pythia-12b-4bit-bbq"
else:
    model_name = "EleutherAI/pythia-12b"

prompt = "Hello, my name is"
device = "cuda:0"


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

print(f"Using model: {model_name}")

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    tokens = model.generate(**inputs, max_new_tokens=100)
output = tokenizer.decode(tokens[0], skip_special_tokens=True)

print(output)
