import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- Configuration ---
BASE_DIR = '/Users/benedict/UHH/bbq/data/model_eval_results'
OUTPUT_DIR = '/Users/benedict/UHH/bbq/data/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Updated palette keys to match the actual cleaned model labels
CUSTOM_PALETTE = {
    "Pythia 12B Deduped": "#2ecc71",           # FP16 (Baseline)
    "Pythia 12B Deduped 8bit": "#3498db",      # Int8 (8-bit)
    "Pythia 12B Deduped nf4bit": "#9b59b6",    # NF4 (4-bit Normal Float)
    "Pythia 12B Deduped fp4bit": "#e74c3c"     # FP4 (4-bit Float)
}

# Definition of metrics to extract per task based on your JSON structure
METRICS_CONFIG = {
    "arc_challenge": ("acc_norm,none", "ARC Challenge (Acc Norm)", True),
    "gsm8k": ("exact_match,strict-match", "GSM8K (Exact Match)", True),
    "hellaswag": ("acc_norm,none", "Hellaswag (Acc Norm)", True),
    "mmlu": ("acc,none", "MMLU (Avg Acc)", True),
    "truthfulqa_mc1": ("acc,none", "TruthfulQA (Acc)", True),
    "winogrande": ("acc,none", "Winogrande (Acc)", True)
}

def clean_model_label(folder_name):
    # 1. Remove date/timestamp (matches _DD-MM-YYYY...)
    name = re.split(r'_\d{2}-\d{2}-\d{4}', folder_name)[0]
    
    # 2. Remove "step..."
    name = re.sub(r'-step\d+', '', name)
    
    # 3. Corrected replacement: use "deduped" (with an 'e') 
    # and ensure it matches the palette's capitalization
    if "pythia-12b-deduped" in name:
        name = name.replace("pythia-12b-deduped", "Pythia 12B Deduped")
    
    # 4. Replace remaining hyphens (like the one before '8bit') with spaces
    name = name.replace("-", " ")
    
    # Clean up any double spaces and trim
    return re.sub(r'\s+', ' ', name).strip()

def load_eval_data(base_dir):
    data = []
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} not found.")
        return pd.DataFrame()

    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for model_dir in model_dirs:
        full_model_path = os.path.join(base_dir, model_dir)
        json_files = glob.glob(os.path.join(full_model_path, "results_*.json"))
        if not json_files: continue
        target_file = json_files[0]
        try:
            with open(target_file, 'r') as f:
                results_json = json.load(f)
            model_label = clean_model_label(model_dir)
            if "results" in results_json:
                for task_key, (metric_key, display_name, higher_better) in METRICS_CONFIG.items():
                    if task_key in results_json["results"]:
                        val = results_json["results"][task_key].get(metric_key)
                        if val is not None:
                            data.append({
                                "Model": model_label,
                                "Task": display_name,
                                "Value": float(val),
                                "Higher_Is_Better": higher_better
                            })
        except Exception as e:
            print(f"Error reading {target_file}: {e}")
    return pd.DataFrame(data)

def plot_comparison(df):
    if df.empty: return
    sns.set_theme(style="whitegrid", rc={"font.size":11})
    tasks = df['Task'].unique()
    num_tasks = len(tasks)
    fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 6))
    if num_tasks == 1: axes = [axes]

    df = df.sort_values(by="Model")

    for ax, task in zip(axes, tasks):
        task_data = df[df['Task'] == task]
        sns.barplot(data=task_data, x="Model", y="Value", hue="Model", ax=ax, palette=CUSTOM_PALETTE, dodge=False, legend=False)
        ax.set_title(task, fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'model_eval_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_relative_comparison(df):
    if df.empty: return
    models = sorted(df['Model'].unique())
    baseline_model = models[0]
    pivot_df = df.pivot(index='Task', columns='Model', values='Value')
    rel_df = pivot_df.div(pivot_df[baseline_model], axis=0) - 1
    rel_df = rel_df * 100
    plot_df = rel_df.reset_index().melt(id_vars='Task', var_name='Model', value_name='RelDiff')
    plot_df = plot_df[plot_df['Model'] != baseline_model]

    if plot_df.empty:
        print("Not enough models to create a relative comparison.")
        return

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=plot_df, x='Task', y='RelDiff', hue='Model', palette=CUSTOM_PALETTE)
    
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.title(f'Relative Performance Change (Baseline: {baseline_model})', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Percentage Difference (%)', fontsize=12)
    plt.xlabel('Evaluation Task', fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(title="Comparison Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'model_eval_relative_diff.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Relative difference plot saved at: {output_path}")
    plt.close()

def main():
    print("Loading Model Eval Data...")
    df = load_eval_data(BASE_DIR)
    if not df.empty:
        print(f"Data points found: {len(df)}")
        plot_comparison(df)
        plot_relative_comparison(df)
    else:
        print("No data found.")

if __name__ == "__main__":
    main()