import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, BitsAndBytesConfig

# https://huggingface.co/docs/transformers/v5.0.0rc0/en/main_classes/quantization#transformers.BitsAndBytesConfig

QUANT_MODE = "fp4bit"

MODEL_NAME = "pythia-12b-deduped-step143000"

MODELS_PATH = "/home/bstahl/bbq/models"

LOCAL_PATH = f"{MODELS_PATH}/{MODEL_NAME}"

OUTPUT_DIR = f"{MODELS_PATH}/{MODEL_NAME}-{QUANT_MODE}"

print(f"Configuring for {QUANT_MODE} quantization...")

if QUANT_MODE == "nf4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
elif QUANT_MODE == "fp4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
elif QUANT_MODE == "8bit":
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
else:
    raise ValueError("QUANT_MODE must be 'nf4bit', 'fp4bit', or '8bit'")

print(f"Loading model from: {LOCAL_PATH}")
model = GPTNeoXForCausalLM.from_pretrained(
    LOCAL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Saving quantized model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)

print(f"Loading tokenizer from: {LOCAL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done!")