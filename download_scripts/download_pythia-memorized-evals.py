from datasets import load_dataset
import os

# 1. Configuration
# Set the specific step number you want (e.g., 23000, 123000).
# Note: Check the Hugging Face viewer to ensure the specific step exists as a split.
step_number = 123000

# Construct the split name dynamically (e.g., "duped.12b.123000")
target_split = f"duped.12b.{step_number}"

# Base path for your cache
base_cache_dir = "/mnt/storage2/student_data/bstahl/bbq/pythia-12b_memorized-evals"

print(f"--- Loading Dataset ---")
print(f"Dataset: EleutherAI/pythia-memorized-evals")
print(f"Split: {target_split}")

# 2. Load the dataset
try:
    dataset = load_dataset(
        "EleutherAI/pythia-memorized-evals",
        split=target_split,
        cache_dir=base_cache_dir
    )

    print("\nSuccess! Dataset loaded.")
    print(dataset)

except ValueError as e:
    print(f"\nError: The split '{target_split}' was not found.")
    print("Double-check that this specific step number exists in the dataset on Hugging Face.")
except Exception as e:
    print(f"\nError: {e}")