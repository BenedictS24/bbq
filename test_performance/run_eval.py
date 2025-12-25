import os
import subprocess
from datetime import datetime

# https://github.com/EleutherAI/lm-evaluation-harness

# --- CONFIGURATION ---

# 1. Directory containing your model checkpoints
MODEL_BASE_DIR = "/home/bstahl/bbq/models"

# 2. Base directory for saving results
RESULTS_BASE_DIR = "/home/bstahl/bbq/data/model_eval_results"

# 3. List of specific model folder names inside 'MODEL_BASE_DIR' to evaluate
MODEL_LIST = [
    "pythia-12b-deduped-step143000",
    "pythia-12b-deduped-step143000-8bit",
    "pythia-12b-deduped-step143000-fp4bit",
    "pythia-12b-deduped-step143000-nf4bit"
]
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md
# https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive 
# 4. Tasks to run
TASKS = "arc_challenge,hellaswag,mmlu,truthfulqa_mc1,winogrande,gsm8k"

# --- EXECUTION ---

def run_evaluation():
    
    for model_name in MODEL_LIST:
        # Construct the full path to the model
        full_model_path = os.path.join(MODEL_BASE_DIR, model_name)
        
        # Verify model exists
        if not os.path.exists(full_model_path):
            print(f"\n[SKIP] Model path not found: {full_model_path}")
            continue

        # Generate European Timestamp: DD-MM-YYYY_HH-MM-SS
        # Example: 13-12-2025_17-06-54
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        
        # Create the specific output folder
        specific_output_dir = os.path.join(RESULTS_BASE_DIR, f"{model_name}_{timestamp}")
        
        # Create the directory immediately
        os.makedirs(specific_output_dir, exist_ok=True)

        print(f"\n=======================================================")
        print(f"Starting: {model_name}")
        print(f"Loading:  {full_model_path}")
        print(f"Saving to: {specific_output_dir}")
        print(f"=======================================================")

        # Construct the command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={full_model_path},trust_remote_code=true",
            "--tasks", TASKS,
            "--device", "cuda:0",
            "--batch_size", "auto",
            "--output_path", specific_output_dir,
            "--log_samples"
        ]

        try:
            # Run the command
            subprocess.run(cmd, check=True)
            print(f"DONE: Results saved in {specific_output_dir}")
        except subprocess.CalledProcessError:
            print(f"ERROR: Failed to evaluate {model_name}")

if __name__ == "__main__":
    run_evaluation()
