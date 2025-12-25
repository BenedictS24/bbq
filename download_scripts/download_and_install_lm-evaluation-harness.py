import os
import sys
import subprocess

TARGET_DIR = "/home/bstahl/bbq/test_performance"
REPO_URL = "https://github.com/EleutherAI/lm-evaluation-harness.git"
REPO_NAME = "lm-evaluation-harness"

def run_command(command, description):
    print(f"\n--- {description} ---")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while: {description}")
        sys.exit(1)

def main():
    if not os.environ.get("CONDA_DEFAULT_ENV"):
        print("WARNING: No Conda environment detected.")
        print("Please activate your Conda environment before running this script.")
        try:
            input("Press Enter to continue anyway, or Ctrl+C to abort...")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)

    print(f"Target Directory: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.chdir(TARGET_DIR)

    if os.path.isdir(REPO_NAME):
        print(f"'{REPO_NAME}' already exists. Pulling latest changes...")
        os.chdir(REPO_NAME)
        run_command(["git", "pull"], "Git Pull")
    else:
        print(f"Cloning {REPO_NAME}...")
        run_command(["git", "clone", REPO_URL], "Git Clone")
        os.chdir(REPO_NAME)

    run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."], 
        "Step 1: Installing core evaluation framework"
    )

    run_command(
        [sys.executable, "-m", "pip", "install", ".[hf]"], 
        "Step 2: Installing Hugging Face backend"
    )

    run_command(
        [sys.executable, "-m", "pip", "install", ".[vllm]"], 
        "Step 3: Installing vLLM backend"
    )

    run_command(
        [sys.executable, "-m", "pip", "install", "bitsandbytes", "accelerate", "scipy"], 
        "Step 4: Installing quantization libraries"
    )

    print("\n------------------------------------------------------------------")
    print("Installation complete!")
    print(f"Installed in: {os.getcwd()}")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    main()