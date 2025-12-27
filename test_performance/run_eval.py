import os
import subprocess
import json
import gc
import torch
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table

MODEL_BASE_DIR = "/home/bstahl/bbq/models"
RESULTS_BASE_DIR = "/home/bstahl/bbq/data/model_eval_results"

MODEL_LIST = [
    "pythia-12b-deduped-step143000",
    "pythia-12b-deduped-step143000-8bit",
    "pythia-12b-deduped-step143000-fp4bit",
    "pythia-12b-deduped-step143000-nf4bit"
]

TASKS_CONFIG = {
    "arc_challenge": 25,
    "hellaswag": 10,
    "mmlu": 5,
    "truthfulqa_mc2": 0,
    "winogrande": 5,
    "gsm8k": 5
}

DEVICE = "cuda:0"
console = Console()

def run_evaluation():
    summary_data = []

    for model_name in MODEL_LIST:
        full_model_path = os.path.join(MODEL_BASE_DIR, model_name)
        
        if not os.path.exists(full_model_path):
            console.print(f"[bold red][SKIP][/] Path not found: {full_model_path}")
            continue

        batch_size = "5" if "-8bit" in model_name else "auto"

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        model_output_dir = os.path.join(RESULTS_BASE_DIR, f"{model_name}_{timestamp}")
        os.makedirs(model_output_dir, exist_ok=True)

        console.print(f"\n[bold green]>>> EVALUATING: {model_name}[/]")
        
        model_scores = {"Model": model_name}

        for task_name, shots in TASKS_CONFIG.items():
            console.print(f"  [cyan]Task:[/] {task_name:<15} | [cyan]Shots:[/] {shots}")
            
            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={full_model_path},trust_remote_code=true",
                "--tasks", task_name,
                "--num_fewshot", str(shots),
                "--device", DEVICE,
                "--batch_size", str(batch_size),
                "--output_path", model_output_dir
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                res_path = os.path.join(model_output_dir, "results.json")
                if os.path.exists(res_path):
                    with open(res_path, "r") as f:
                        data = json.load(f)
                        results = data.get("results", {}).get(task_name, {})
                        score = results.get("acc_norm", results.get("acc", results.get("exact_match", "N/A")))
                        model_scores[task_name] = score
                
            except subprocess.CalledProcessError:
                console.print(f"    [bold red]✗[/] Failed {task_name}")
                model_scores[task_name] = "ERROR"

        summary_data.append(model_scores)
        gc.collect()
        torch.cuda.empty_cache()

    print_final_table(summary_data)

def print_final_table(data):
    table = Table(title="Pythia-12B Evaluation Summary")
    table.add_column("Model", style="magenta", no_wrap=True)
    for task in TASKS_CONFIG.keys():
        table.add_column(task, justify="right")

    for row in data:
        table.add_row(
            row["Model"],
            *[f"{row.get(task, 'N/A'):.4f}" if isinstance(row.get(task), float) else str(row.get(task)) for task in TASKS_CONFIG.keys()]
        )
    console.print("\n", table)

if __name__ == "__main__":
    run_evaluation()