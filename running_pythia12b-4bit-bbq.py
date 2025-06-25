from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

model_name = "./pythia-12b-4bit-bbq"
prompt = "Hello, I am"
device = "cuda:0"


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


inputs = tokenizer(prompt, return_tensors="pt").to(device)

tokens = model.generate(**inputs, max_new_tokens=300)
output = tokenizer.decode(tokens[0], skip_special_tokens=True)

print(output)
