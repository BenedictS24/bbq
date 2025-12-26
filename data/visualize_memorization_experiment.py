import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
INPUT_DIR = "/Users/benedict/UHH/bbq/data/experiment_data"
INPUT_FILE = "mem_eval_pythia-12b-deduped-step143000_plus_3_others_k4-48_end_of_sequence_25122025_1445.jsonl"
FILE_PATH = os.path.join(INPUT_DIR, INPUT_FILE)

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_name = os.path.splitext(INPUT_FILE)[0]

def clean_model_name(name):
    """Maps long model names to readable labels."""
    if "nf4bit" in name: return "NF4 (4-bit Normal Float)"
    elif "fp4bit" in name: return "FP4 (4-bit Float)"
    elif "8bit" in name: return "Int8 (8-bit)"
    else: return "FP16 (Baseline)"

def load_data(path):
    """Loads JSONL data and prepares dataframes with unit conversions."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df['Label'] = df['model_name'].apply(clean_model_name)
    
    # Unit conversions
    if 'model_footprint_byte' in df.columns:
        df['model_footprint_gb'] = df['model_footprint_byte'] / (1024**3)
    if 'peak_gpu_memory_byte' in df.columns:
        df['peak_gpu_memory_gb'] = df['peak_gpu_memory_byte'] / (1024**3)
        
    return df

def generate_plots(df):
    """Generates and saves comprehensive analysis plots."""
    sns.set_theme(style="whitegrid")
    
    model_order = ["FP16 (Baseline)", "Int8 (8-bit)", "NF4 (4-bit Normal Float)", "FP4 (4-bit Float)"]
    palette = {
        "FP16 (Baseline)": "#2ecc71", 
        "Int8 (8-bit)": "#3498db", 
        "NF4 (4-bit Normal Float)": "#9b59b6", 
        "FP4 (4-bit Float)": "#e74c3c"
    }

    # --- IMAGE 1: PERFORMANCE (Core Memorization) ---
    fig1, axes1 = plt.subplots(4, 1, figsize=(10, 20))
    metrics = [
        ('overall_accuracy', 'Overall Accuracy (0.0 - 1.0)'),
        ('exact_match_percentage', 'Exact Match % (0.0 - 1.0)'),
        ('average_correct_tokens', 'Average Correct Tokens'),
        ('average_successive_correct_tokens', 'Avg Successive Correct Tokens')
    ]
    
    for i, (col, title) in enumerate(metrics):
        sns.lineplot(data=df, x='k', y=col, hue='Label', marker='o', 
                     palette=palette, hue_order=model_order, ax=axes1[i], linewidth=2.5)
        axes1[i].set_title(title, fontsize=14, fontweight='bold')
        axes1[i].set_xlabel('Prefix Length (k)')
        if i > 0: axes1[i].get_legend().remove()
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_performance.png"))

    # --- IMAGE 2: EFFICIENCY & RUNTIME ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 18))

    # 2.1: Runtime vs k
    sns.lineplot(data=df, x='k', y='runtime_seconds', hue='Label', marker='o', 
                 palette=palette, hue_order=model_order, ax=axes2[0], linewidth=2.5)
    axes2[0].set_title('Runtime (Seconds) per Prefix Length (k)', fontsize=14, fontweight='bold')
    axes2[0].get_legend().remove()

    # 2.2: Memory Usage
    unique_df = df.drop_duplicates('Label').copy()
    unique_df['Label'] = pd.Categorical(unique_df['Label'], categories=model_order, ordered=True)
    melted_vram = unique_df.melt(id_vars='Label', value_vars=['model_footprint_gb', 'peak_gpu_memory_gb'], 
                                 var_name='Metric', value_name='GB')
    sns.barplot(data=melted_vram, x='Label', y='GB', hue='Metric', ax=axes2[1], palette="muted")
    axes2[1].set_title('VRAM Comparison: Static Footprint vs. Peak Memory', fontsize=14, fontweight='bold')
    for container in axes2[1].containers:
        axes2[1].bar_label(container, fmt='%.1f GB', padding=3)

    # 2.3: Average Runtime per Model
    avg_runtime = df.groupby('Label')['runtime_seconds'].mean().reindex(model_order).reset_index()
    sns.barplot(data=avg_runtime, x='Label', y='runtime_seconds', palette=palette, ax=axes2[2])
    axes2[2].set_title('Average Runtime per Model (Across all k)', fontsize=14, fontweight='bold')
    for container in axes2[2].containers:
        axes2[2].bar_label(container, fmt='%.1f s', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_efficiency.png"))

    # --- IMAGE 3: VARIABILITY ---
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 10))
    stds = [
        ('accuracy_standard_deviation', 'Accuracy Std Deviation'),
        ('successive_correct_standard_deviation', 'Successive Tokens Std Deviation')
    ]
    
    for i, (col, title) in enumerate(stds):
        sns.lineplot(data=df, x='k', y=col, hue='Label', marker='s', 
                     palette=palette, hue_order=model_order, ax=axes3[i], linewidth=2.5)
        axes3[i].set_title(title, fontsize=14, fontweight='bold')
        axes3[i].set_xlabel('Prefix Length (k)')
        if i > 0: axes3[i].get_legend().remove()
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_variability.png"))

    # --- IMAGE 4: RELATIVE DIFFERENCE TO BASELINE (UPDATED) ---
    baseline_df = df[df['Label'] == "FP16 (Baseline)"].set_index('k')
    relative_data = []
    
    # User requested metrics
    metrics_to_compare = [
        ('overall_accuracy', 'Overall Accuracy'), 
        ('exact_match_percentage', 'Exact Match %'),
        ('average_successive_correct_tokens', 'Avg Successive Tokens')
    ]
    
    for label in model_order:
        if label == "FP16 (Baseline)": continue
        
        model_subset = df[df['Label'] == label].set_index('k')
        
        for k in model_subset.index:
            if k in baseline_df.index:
                for col, name in metrics_to_compare:
                    base_val = baseline_df.loc[k, col]
                    mod_val = model_subset.loc[k, col]
                    
                    # Calculate % difference, handle potential zero baseline
                    if base_val != 0:
                        rel_diff = ((mod_val - base_val) / base_val) * 100
                    else:
                        rel_diff = 0 # No change if baseline is already zero
                        
                    relative_data.append({
                        'k': k,
                        'Label': label,
                        'Metric': name,
                        'Relative Diff (%)': rel_diff
                    })

    rel_df = pd.DataFrame(relative_data)
    fig4, axes4 = plt.subplots(3, 1, figsize=(10, 18)) # 3 rows for 3 metrics
    
    for i, (col, name) in enumerate(metrics_to_compare):
        subset = rel_df[rel_df['Metric'] == name]
        sns.lineplot(data=subset, x='k', y='Relative Diff (%)', hue='Label', 
                     marker='o', palette=palette, hue_order=model_order[1:], ax=axes4[i], linewidth=2.5)
        
        axes4[i].axhline(0, color='black', linestyle='--', alpha=0.6)
        axes4[i].set_title(f'Relative Change in {name} vs. FP16 Baseline', fontsize=14, fontweight='bold')
        axes4[i].set_ylabel('Difference (%)')
        axes4[i].set_xlabel('Prefix Length (k)')
        if i > 0: axes4[i].get_legend().remove()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_relative_diff.png"))

if __name__ == "__main__":
    data_df = load_data(FILE_PATH)
    if data_df is not None:
        generate_plots(data_df)
        print(f"Success! All 4 plot categories saved in the '{OUTPUT_DIR}' directory.")