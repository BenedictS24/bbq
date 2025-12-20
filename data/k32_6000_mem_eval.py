import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration ---
INPUT_DIR = "/Users/benedict/UHH/bbq/data/experiment_data"
INPUT_FILE = "mem_eval_pythia-12b-duped-step143000_plus_3_others_k32_6000_20122025_0010.jsonl"
FILE_PATH = os.path.join(INPUT_DIR, INPUT_FILE)

OUTPUT_DIR = "plots_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_name = os.path.splitext(INPUT_FILE)[0]

def clean_model_name(name):
    """Maps long model names to readable labels."""
    if "nf4bit" in name: return "NF4 (4-bit)"
    elif "fp4bit" in name: return "FP4 (4-bit)"
    elif "8bit" in name: return "Int8 (8-bit)"
    else: return "FP16 (Baseline)"

def load_data(path):
    """Loads JSONL data and prepares dataframes."""
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
    sns.set_theme(style="whitegrid")
    model_order = ["FP16 (Baseline)", "Int8 (8-bit)", "NF4 (4-bit)", "FP4 (4-bit)"]
    palette = {
        "FP16 (Baseline)": "#2ecc71", 
        "Int8 (8-bit)": "#3498db", 
        "NF4 (4-bit)": "#9b59b6", 
        "FP4 (4-bit)": "#e74c3c"
    }

    # --- IMAGE 1: PERFORMANCE AVERAGES (Bar Plots) ---
    # Good for comparing the absolute drop in performance across quantization
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ('overall_accuracy', 'Overall Accuracy'),
        ('exact_match_percentage', 'Exact Match %'),
        ('average_correct_tokens', 'Avg Correct Tokens'),
        ('average_successive_correct_tokens', 'Avg Successive Correct Tokens')
    ]
    
    axes_flat = axes1.flatten()
    for i, (col, title) in enumerate(metrics):
        sns.barplot(data=df, x='Label', y=col, order=model_order, palette=palette, ax=axes_flat[i])
        axes_flat[i].set_title(title, fontsize=14, fontweight='bold')
        axes_flat[i].set_xlabel("")
        # Add values on top of bars
        for container in axes_flat[i].containers:
            axes_flat[i].bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_averages.png"))

    # --- IMAGE 2: DISTRIBUTION ANALYSIS (Correct Tokens) ---
    # This uses the 'correct_token_distribution' list from your JSON
    plt.figure(figsize=(12, 7))
    
    for _, row in df.iterrows():
        dist = row['correct_token_distribution']
        # We create an x-axis representing the number of tokens (0 to k)
        x = np.arange(len(dist)) 
        plt.plot(x, dist, label=row['Label'], color=palette[row['Label']], marker='.', alpha=0.7)

    plt.yscale('log') # Use log scale because Exact Matches (end of list) are usually very high
    plt.title('Correct Token Distribution (Log Scale)', fontsize=15, fontweight='bold')
    plt.xlabel('Number of Tokens Correct')
    plt.ylabel('Frequency (Count of Samples)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_distribution_line.png"))

    # --- IMAGE 3: RESOURCE EFFICIENCY ---
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # VRAM Comparison
    unique_df = df.drop_duplicates('Label').copy()
    melted_vram = unique_df.melt(id_vars='Label', value_vars=['model_footprint_gb', 'peak_gpu_memory_gb'], 
                                 var_name='Metric', value_name='GB')
    sns.barplot(data=melted_vram, x='Label', y='GB', hue='Metric', order=model_order, ax=ax1, palette="muted")
    ax1.set_title('VRAM: Footprint vs Peak', fontweight='bold')
    
    # Runtime
    sns.barplot(data=df, x='Label', y='runtime_seconds', order=model_order, palette=palette, ax=ax2)
    ax2.set_title('Total Runtime (Seconds)', fontweight='bold')
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.0f s', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_resources.png"))

if __name__ == "__main__":
    data_df = load_data(FILE_PATH)
    if data_df is not None:
        generate_plots(data_df)
        print(f"Analysis complete. Plots saved in '{OUTPUT_DIR}'")