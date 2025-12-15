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

# Definition of metrics to extract per task
METRICS_CONFIG = {
    "gsm8k": ("exact_match,strict-match", "GSM8K (Exact Match)", True),
    "hellaswag": ("acc_norm,none", "Hellaswag (Acc Norm)", True),
    "wikitext": ("word_perplexity,none", "Wikitext (Perplexity)", False) # False = Lower is better
}

def clean_model_label(folder_name):
    """
    Creates a clean label from the folder name.
    Input: pythia-12b-duped-step143000-nf4bit_14-12-2025_14-34-03
    Output: Pythia 12B Deduped nf4bit
    """
    # 1. Remove date/timestamp
    name = re.split(r'_\d{2}-\d{2}-\d{4}', folder_name)[0]
    
    # 2. Remove "step..."
    name = re.sub(r'-step\d+', '', name)
    
    # 3. Replace base name
    if "pythia-12b-duped" in name:
        name = name.replace("pythia-12b-duped", "Pythia 12B Deduped")
    
    # 4. Replace hyphens with spaces
    name = name.replace("-", " ")
    
    # Clean up whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def load_eval_data(base_dir):
    data = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} not found.")
        return pd.DataFrame()

    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for model_dir in model_dirs:
        full_model_path = os.path.join(base_dir, model_dir)
        json_files = glob.glob(os.path.join(full_model_path, "results_*.json"))
        
        if not json_files:
            continue
            
        target_file = json_files[0]
        
        try:
            with open(target_file, 'r') as f:
                results_json = json.load(f)
            
            model_label = clean_model_label(model_dir)
            
            if "results" in results_json:
                for task, (metric_key, display_name, higher_better) in METRICS_CONFIG.items():
                    if task in results_json["results"]:
                        val = results_json["results"][task].get(metric_key)
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
    if df.empty:
        print("No data found to plot.")
        return

    # Set style and font size globally for better readability
    sns.set_theme(style="whitegrid", rc={"font.size":11, "axes.titlesize":14, "axes.labelsize":12})
    
    tasks = df['Task'].unique()
    num_tasks = len(tasks)
    
    fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 6), sharey=False)
    
    if num_tasks == 1:
        axes = [axes]

    df = df.sort_values(by="Model")
    palette = sns.color_palette("viridis", n_colors=len(df['Model'].unique()))

    for ax, task in zip(axes, tasks):
        task_data = df[df['Task'] == task]
        
        sns.barplot(
            data=task_data, 
            x="Model", 
            y="Value", 
            hue="Model", 
            ax=ax, 
            palette=palette, 
            dodge=False,
            legend=False  # Hide legend since labels are on the X-axis
        )
        
        # pad=25 pushes the title up
        ax.set_title(task, fontsize=14, fontweight='bold', pad=25)
        
        ax.set_xlabel("")
        ax.set_ylabel("Score / Value")
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Visual indicator (Higher/Lower)
        is_higher_better = task_data.iloc[0]['Higher_Is_Better']
        direction_text = "↑ (Higher is better)" if is_higher_better else "↓ (Lower is better)"
        
        ax.text(0.5, 1.02, direction_text, 
                transform=ax.transAxes, 
                ha='center', 
                fontsize=10, 
                color='#555555', # Slightly darker gray for better readability
                fontweight='medium')
        
        # Show values on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)

    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'model_eval_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved at: {output_path}")
    plt.close()

def main():
    print("Loading Model Eval Data...")
    df = load_eval_data(BASE_DIR)
    
    if not df.empty:
        print(f"Data points found: {len(df)}")
        print("Models:", df['Model'].unique())
        print("Creating plot...")
        plot_comparison(df)
    else:
        print("Could not extract any data.")

if __name__ == "__main__":
    main()