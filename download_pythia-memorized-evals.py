from datasets import load_dataset

dataset = load_dataset("EleutherAI/pythia-memorized-evals",
                       split="duped.12b",
                       cache_dir="/mnt/storage2/student_data/bstahl/bbq/pythia-12b_memorized-evals"
                      )

print(dataset)