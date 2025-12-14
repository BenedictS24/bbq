import os
import sys
import subprocess

# --- Configuration ---
TARGET_DIR = "/home/bstahl/bbq/test_performance"
REPO_URL = "https://github.com/EleutherAI/lm-evaluation-harness.git"
REPO_NAME = "lm-evaluation-harness"

def run_command(command, description):
    """Helper to run a system command with error checking."""
    print(f"\n--- {description} ---")
    try:
        # check=True will raise an error if the command fails
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while: {description}")
        sys.exit(1)

def main():
    # 1. Check for Conda Environment
    # We check the environment variable that Conda sets when active
    if not os.environ.get("CONDA_DEFAULT_ENV"):
        print("WARNING: No Conda environment detected.")
        print("Please activate your Conda environment before running this script.")
        try:
            input("Press Enter to continue anyway, or Ctrl+C to abort...")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)

    # 2. Create and Enter Target Directory
    print(f"Target Directory: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.chdir(TARGET_DIR)

    # 3. Clone or Update Repository
    if os.path.isdir(REPO_NAME):
        print(f"'{REPO_NAME}' already exists. Pulling latest changes...")
        os.chdir(REPO_NAME)
        run_command(["git", "pull"], "Git Pull")
    else:
        print(f"Cloning {REPO_NAME}...")
        run_command(["git", "clone", REPO_URL], "Git Clone")
        os.chdir(REPO_NAME)

    # 4. Step-by-Step Installation
    # We use [sys.executable, "-m", "pip"] to ensure we use the pip 
    # belonging to the CURRENTLY ACTIVE Python environment.

    # Step 1: Core Framework
    run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."], 
        "Step 1: Installing core evaluation framework"
    )

    # Step 2: Hugging Face Backend
    # Note: We pass ".[hf]" directly. We do not need extra quotes inside the list.
    run_command(
        [sys.executable, "-m", "pip", "install", ".[hf]"], 
        "Step 2: Installing Hugging Face backend"
    )

    # Step 3: vLLM Backend
    run_command(
        [sys.executable, "-m", "pip", "install", ".[vllm]"], 
        "Step 3: Installing vLLM backend"
    )

    # Step 4: Quantization Libraries
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