import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
FILE_PATH = "/Users/benedict/UHH/bbq/data/experiment_data/k4-48_fp16-8bit-fp4-nf4_memorization_results.jsonl"
OUTPUT_METRICS_IMG = "memorization_metrics_dashboard.png"
OUTPUT_DIST_IMG = "token_distribution_k32.png"

# --- Styling & Palette ---
sns.set_theme(style="whitegrid", context="talk")
PALETTE = {
    "FP16 (Baseline)": "#2ecc71",   # Emerald Green
    "Int8 (8-bit)": "#3498db",      # Peter River Blue
    "NF4 (Normal Float)": "#9b59b6", # Amethyst Purple
    "FP4 (Pure Float)": "#e74c3c"    # Alizarin Red
}

def clean_model_name(name):
    """Maps technical names to readable labels."""
    if "nf4bit" in name:
        return "NF4 (Normal Float)"
    elif "fp4bit" in name:
        return "FP4 (Pure Float)"
    elif "8bit" in name:
        return "Int8 (8-bit)"
    else:
        return "FP16 (Baseline)"

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df['Label'] = df['model_name'].apply(clean_model_name)
    
    # Calculate Throughput (Samples per Second)
    df['samples_per_second'] = df['sample_size'] / df['runtime_seconds']
    
    return df

def plot_metrics_dashboard(df):
    """Generates a 2x2 dashboard of key performance indicators."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Exact Match (The "Hard" Memorization Metric)
    sns.lineplot(data=df, x='k', y='exact_match_percentage', hue='Label', 
                 palette=PALETTE, marker='o', ax=axes[0, 0], linewidth=3)
    axes[0, 0].set_title('Strict Memorization (Exact Match %)')
    axes[0, 0].set_ylabel('Exact Match Rate')
    axes[0, 0].set_xlabel('Context Length (k)')
    
    # 2. Overall Accuracy (The "Soft" Capability Metric)
    sns.lineplot(data=df, x='k', y='overall_accuracy', hue='Label', 
                 palette=PALETTE, marker='^', ax=axes[0, 1], linewidth=3, linestyle='--')
    axes[0, 1].set_title('General Competence (Overall Accuracy)')
    axes[0, 1].set_ylabel('Accuracy (Next Token)')
    axes[0, 1].set_xlabel('Context Length (k)')
    axes[0, 1].get_legend().remove()

    # 3. Memory Usage (Peak Allocation)
    # We aggregate by mean since footprint is constant across k
    mem_df = df.groupby('Label')[['peak_gpu_memory_gb', 'model_footprint_gb']].mean().reset_index()
    mem_df_melted = mem_df.melt(id_vars='Label', value_vars=['peak_gpu_memory_gb', 'model_footprint_gb'], 
                                var_name='Metric', value_name='GB')
    
    sns.barplot(data=mem_df, x='Label', y='peak_gpu_memory_gb', palette=PALETTE, ax=axes[1, 0])
    axes[1, 0].set_title('Peak GPU Memory Usage (VRAM)')
    axes[1, 0].set_ylabel('Gigabytes (GB)')
    axes[1, 0].set_xlabel('')
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    # Add labels on bars
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, fmt='%.1f GB', padding=3)

    # 4. Inference Speed (Throughput)
    sns.boxplot(data=df, x='Label', y='samples_per_second', palette=PALETTE, ax=axes[1, 1])
    axes[1, 1].set_title('Inference Speed (Throughput)')
    axes[1, 1].set_ylabel('Samples / Second')
    axes[1, 1].set_xlabel('')
    axes[1, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_METRICS_IMG)
    print(f"Metrics dashboard saved to {OUTPUT_METRICS_IMG}")

def plot_token_distribution(df, k_target=32):
    """
    Analyzes how many tokens the model got right at a specific context length (k).
    Did it fail immediately (0 tokens) or almost get it (N-1 tokens)?
    """
    # Filter for specific k
    subset = df[df['k'] == k_target].copy()
    
    # The 'correct_token_distribution' is a list of counts.
    # We need to expand this into a format suitable for plotting.
    # We'll normalize counts to frequencies (probabilities).
    
    plot_data = []
    
    for _, row in subset.iterrows():
        dist = row['correct_token_distribution']
        total_samples = row['sample_size']
        label = row['Label']
        
        # Create a row for each number of correct tokens (index of the list)
        for num_correct, count in enumerate(dist):
            # The last bucket often catches "Rest" or "Max", but here it aligns 
            # with exact match. We'll treat index as "Num Tokens Correct".
            plot_data.append({
                "Label": label,
                "Correct Tokens": num_correct,
                "Percentage": (count / total_samples) * 100
            })
            
    dist_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(14, 8))
    
    # Use a bar plot with 'hue' to compare distributions side-by-side
    sns.barplot(data=dist_df, x='Correct Tokens', y='Percentage', hue='Label', palette=PALETTE)
    
    plt.title(f'Depth of Recall at k={k_target}\n(Histogram of Correct Tokens per Sample)')
    plt.ylabel('Percentage of Samples (%)')
    plt.xlabel('Number of Successive Correct Tokens (Sequence Length)')
    plt.legend(title='Model')
    
    # Annotate: High bars on the LEFT mean model is hallucinating immediately.
    # High bars on the RIGHT mean model is memorizing perfectly.
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIST_IMG)
    print(f"Distribution plot saved to {OUTPUT_DIST_IMG}")

def print_advanced_stats(df):
    print("\n" + "="*80)
    print(f"{'METRIC COMPARISON':^80}")
    print("="*80)
    
    # Group by model
    cols = ['exact_match_percentage', 'overall_accuracy', 'average_correct_tokens', 
            'samples_per_second', 'peak_gpu_memory_gb']
    stats = df.groupby('Label')[cols].mean()
    
    # Calculate efficiency relative to FP16
    baseline_mem = stats.loc['FP16 (Baseline)', 'peak_gpu_memory_gb']
    baseline_acc = stats.loc['FP16 (Baseline)', 'exact_match_percentage']
    
    stats['Mem Reduction'] = (1 - (stats['peak_gpu_memory_gb'] / baseline_mem)) * 100
    stats['Perf Retention'] = (stats['exact_match_percentage'] / baseline_acc) * 100
    
    # Formatting
    print(f"{'Model':<25} | {'Exact Match':<10} | {'Gen. Acc.':<10} | {'Speed (S/s)':<10} | {'Mem (GB)':<8} | {'Retention':<9}")
    print("-" * 90)
    
    for label, row in stats.sort_values('exact_match_percentage', ascending=False).iterrows():
        print(f"{label:<25} | {row['exact_match_percentage']:.4f}     | {row['overall_accuracy']:.4f}     | {row['samples_per_second']:.1f}       | {row['peak_gpu_memory_gb']:.1f}     | {row['Perf Retention']:.1f}%")
    print("-" * 90)

if __name__ == "__main__":
    df = load_data(FILE_PATH)
    if df is not None:
        print_advanced_stats(df)
        plot_metrics_dashboard(df)
        plot_token_distribution(df, k_target=32)