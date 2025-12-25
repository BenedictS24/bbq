from huggingface_hub import snapshot_download
import os

# https://huggingface.co/EleutherAI/pythia-12b 


# MODEL_ID = "EleutherAI/pythia-12b"
MODEL_ID = "EleutherAI/pythia-12b-deduped"
TRAINING_STEP = "step143000" 

BASE_FOLDER = "/home/bstahl/bbq/models/"


# SAVE_PATH = os.path.join(BASE_FOLDER, f"pythia-12b-duped-{TRAINING_STEP}")
SAVE_PATH = os.path.join(BASE_FOLDER, f"pythia-12b-deduped-{TRAINING_STEP}")

os.makedirs(SAVE_PATH, exist_ok=True)

print("--- Starting Download ---")
print(f"Model: {MODEL_ID}")
print(f"Training Step: {TRAINING_STEP}")
print(f"Target Directory: {SAVE_PATH}")

try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=TRAINING_STEP,
        local_dir=SAVE_PATH,            
    )
    print(f"\nSuccess! Model saved to: {path}")

except Exception as e:
    print(f"\nError: {e}")