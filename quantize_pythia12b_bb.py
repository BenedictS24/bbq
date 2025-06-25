import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "EleutherAI/pythia-12b"
cache_dir  = f"./{model_name.split('/')[-1]}"

# 1) Set up your 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # turn on 4-bit loading
    bnb_4bit_quant_type="nf4",                # normal-float4 quant
    bnb_4bit_compute_dtype=torch.float16,     # compute in FP16
    bnb_4bit_use_double_quant=True            # higher-quality re-quant
)

# 2) Load the model with quantization
model = GPTNeoXForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",        
    cache_dir=cache_dir,
)

# 3) (Optional) save the quantized model for faster reload next time
model.save_pretrained("pythia-12b-4bit-bbq")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
tokenizer.save_pretrained("pythia-12b-4bit-bbq")
# 4) Load tokenizer (no change)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# 5) Do a quick generation test
prompt = "Hello, I am"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
tokens = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
