import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- configuration ---
quant_mode = "8bit"

# 2. the specific model folder you want to quantize
model_name = "pythia-12b-duped-step143000"

# 1. the base directory where all your models are stored
models_path = "~/bbq/models" 


# --- path construction ---
# full path to the input model
local_path = f"{models_path}/{model_name}"

# full path for the output (saves it back into the models folder)
output_dir = f"{models_path}/{model_name}-{quant_mode}"

# --- setup ---
print(f"Configuring for {quant_mode} quantization...")

if quant_mode == "4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
elif quant_mode == "8bit":
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
else:
    raise ValueError("quant_mode must be '4bit' or '8bit'")


# --- execution ---
print(f"Loading model from: {local_path}")
model = GPTNeoXForCausalLM.from_pretrained(
    local_path,
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Saving quantized model to: {output_dir}")
model.save_pretrained(output_dir)

# load tokenizer directly from local path
print(f"Loading tokenizer from: {local_path}")
tokenizer = AutoTokenizer.from_pretrained(local_path)
tokenizer.save_pretrained(output_dir)

print("Done!")