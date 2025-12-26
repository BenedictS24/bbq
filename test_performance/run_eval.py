import os
import subprocess
from datetime import datetime

# https://github.com/EleutherAI/lm-evaluation-harness

MODEL_BASE_DIR = "/home/bstahl/bbq/models"
RESULTS_BASE_DIR = "/home/bstahl/bbq/data/model_eval_results"
MODEL_LIST = [
    "pythia-12b-deduped-step143000-8bit",
]

# List of possible tasks: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md
# Reason for task selection: https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive 
TASKS = "arc_challenge,hellaswag,mmlu,truthfulqa_mc1,winogrande,gsm8k"


def run_evaluation():
    
    for model_name in MODEL_LIST:
        full_model_path = os.path.join(MODEL_BASE_DIR, model_name)
        
        if not os.path.exists(full_model_path):
            print(f"\n[SKIP] Model path not found: {full_model_path}")
            continue

        # The 8 bit models have a problem with the auto batch size detection
        # so I set it manually to 3 here - if you still run out of memory, lower it further
        if "-8bit" in model_name:
            batch_size = "3"
        else:
            batch_size = "auto"

        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        
        specific_output_dir = os.path.join(RESULTS_BASE_DIR, f"{model_name}_{timestamp}")
        
        os.makedirs(specific_output_dir, exist_ok=True)

        print(f"\n=======================================================")
        print(f"Starting: {model_name}")
        print(f"Loading:  {full_model_path}")
        print(f"Batch Size: {batch_size}")
        print(f"Saving to: {specific_output_dir}")
        print(f"=======================================================")

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={full_model_path},trust_remote_code=true",
            "--tasks", TASKS,
            "--device", "cuda:0",
<<<<<<< HEAD
            "--batch_size", "4",
=======
            "--batch_size", batch_size,
>>>>>>> 9df0df6c8008762c9af8b942b65c9e2fcc1f63b8
            "--output_path", specific_output_dir,
            "--log_samples"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"DONE: Results saved in {specific_output_dir}")
        except subprocess.CalledProcessError:
            print(f"ERROR: Failed to evaluate {model_name}")


if __name__ == "__main__":
    run_evaluation()
