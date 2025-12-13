from huggingface_hub import snapshot_download
import os

# https://huggingface.co/EleutherAI/pythia-12b 

# 1. Configuration
MODEL_ID = "EleutherAI/pythia-12b"
TRAINING_STEP = "step143000"  # Change this to "step123000" etc. if needed

# Define your base models directory
BASE_FOLDER = "/home/bstahl/bbq/models/"

# Define the specific subfolder for this model version
# Result: /home/bstahl/bbq/models/pythia-12b-duped-step143000
SAVE_PATH = os.path.join(BASE_FOLDER, f"pythia-12b-duped-{TRAINING_STEP}")

# 2. Create the directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

print("--- Starting Download ---")
print(f"Model: {MODEL_ID}")
print(f"Training Step: {TRAINING_STEP}")
print(f"Target Directory: {SAVE_PATH}")

# 3. Download
try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=TRAINING_STEP,
        local_dir=SAVE_PATH,            # Uses your specific path
    )
    print(f"\nSuccess! Model saved to: {path}")

except Exception as e:
    print(f"\nError: {e}")