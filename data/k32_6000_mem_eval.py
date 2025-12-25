import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration ---
INPUT_DIR = "/Users/benedict/UHH/bbq/data/experiment_data"
INPUT_FILE = "mem_eval_pythia-12b-duped-step143000_plus_3_others_k32_6000_24122025_1335.jsonl"
FILE_PATH = os.path.join(INPUT_DIR, INPUT_FILE)

OUTPUT_DIR = "plots"
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
        for container in axes_flat[i].containers:
            axes_flat[i].bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_averages.png"))

    # --- IMAGE 2: DISTRIBUTION ANALYSIS (Extended with Successive Tokens) ---
    plt.figure(figsize=(14, 8))
    for _, row in df.iterrows():
        # Total Correct tokens (Solid)
        dist_c = row['correct_token_distribution']
        x = np.arange(len(dist_c)) 
        plt.plot(x, dist_c, label=f"{row['Label']} (Total)", color=palette[row['Label']], marker='.', alpha=0.8)
        
        # Successive tokens (Dashed, Pale)
        dist_s = row['successive_token_distribution']
        plt.plot(x, dist_s, color=palette[row['Label']], linestyle='--', alpha=0.3, label=f"{row['Label']} (Successive)")

    plt.yscale('log')
    plt.title('Token Distribution Analysis (Log Scale)', fontsize=15, fontweight='bold')
    plt.xlabel('Number of Tokens Correct ($n$)')
    plt.ylabel('Frequency (Count)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model & Match Type")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_distribution_line_extended.png"))

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

    # --- IMAGE 4: PAPER-BASED EVALUATION (Memorized vs Unmemorized) ---
    def get_paper_stats(row):
        """Calculates definitions based on the ICLR paper criteria."""
        dist = row['correct_token_distribution']
        total = row['sample_size']
        # Memorized: score is 1 (all tokens correct, index 32 for k=32) 
        memorized = dist[32]
        # Unmemorized: score is less than 10% (0-3 tokens correct for k=32) 
        unmemorized = sum(dist[0:4]) 
        return pd.Series({
            'Memorized %': (memorized / total) * 100,
            'Unmemorized %': (unmemorized / total) * 100
        })

    paper_eval = df.apply(get_paper_stats, axis=1)
    paper_df = pd.concat([df[['Label']], paper_eval], axis=1)
    paper_melted = paper_df.melt(id_vars='Label', var_name='Category', value_name='Percentage')
    
    plt.figure(figsize=(12, 7))
    # We use Category on X and Label as Hue to apply the model colors consistently
    ax4 = sns.barplot(
        data=paper_melted, 
        x='Category', 
        y='Percentage', 
        hue='Label', 
        hue_order=model_order, 
        palette=palette
    )
    
    plt.title('Memorization Evaluation (Per Paper Definitions)', fontsize=15, fontweight='bold')
    plt.ylabel('Percentage of Samples (%)')
    plt.xlabel('Metric Category')
    plt.ylim(0, 100)
    plt.legend(title="Model")
    
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_paper_eval.png"))

if __name__ == "__main__":
    data_df = load_data(FILE_PATH)
    if data_df is not None:
        generate_plots(data_df)
        print(f"Analysis complete. Plots saved in '{OUTPUT_DIR}'")