from datasets import load_dataset
import os

# https://huggingface.co/datasets/EleutherAI/pythia-memorized-evals/viewer/default/duped.12b.123000?views[]=duped12b123000

SPLIT_NUMBER = 123000

# TARGET_SPLIT = f"duped.12b.{SPLIT_NUMBER}"
TARGET_SPLIT = f"deduped.12b.{SPLIT_NUMBER}"

BASE_CACHE_DIR = "/mnt/storage2/student_data/bstahl/bbq/test_memorization/pythia-12b_memorized-evals"

print(f"--- Loading Dataset ---")
print(f"Dataset: EleutherAI/pythia-memorized-evals")
print(f"Split: {TARGET_SPLIT}")

try:
    dataset = load_dataset(
        "EleutherAI/pythia-memorized-evals",
        split=TARGET_SPLIT,
        cache_dir=BASE_CACHE_DIR
    )

    print("\nSuccess! Dataset loaded.")
    print(dataset)

except ValueError as e:
    print(f"\nError: The split '{TARGET_SPLIT}' was not found.")
    print("Double-check that this specific step number exists in the dataset on Hugging Face.")
except Exception as e:
    print(f"\nError: {e}")