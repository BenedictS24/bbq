import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "EleutherAI/pythia-12b"
cache_dir  = f"./models/{model_name.split('/')[-1]}"


'''
https://huggingface.co/docs/transformers/v5.0.0rc0/en/main_classes/quantization#transformers.BitsAndBytesConfig
'''

bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,  
    llm_int8_threshold=6.0,                     
)

model = GPTNeoXForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_8bit,
    device_map="auto",        
    cache_dir=cache_dir,
)

model.save_pretrained("pythia-12b-8bit-bbq")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
tokenizer.save_pretrained("pythia-12b-8bit-bbq")