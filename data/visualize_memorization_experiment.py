import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
FILE_PATH = "/Users/benedict/UHH/bbq/data/experiment_data/k4-48_fp16-8bit-fp4-nf4_memorization_results.jsonl"
OUTPUT_IMAGE = "memorization_comparison.png"

def clean_model_name(name):
    """Maps long model names to readable labels for the plot."""
    if "nf4bit" in name:
        return "NF4 (4-bit Normal Float)"
    elif "fp4bit" in name:
        return "FP4 (4-bit Float)"
    elif "8bit" in name:
        return "Int8 (8-bit)"
    else:
        return "FP16 (Baseline)"

def load_data(path):
    """Loads JSONL data into a Pandas DataFrame."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    # Create a clean label column
    df['Label'] = df['model_name'].apply(clean_model_name)
    return df

def generate_comparison(df):
    """Generates visualizations and print statistics."""
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)
    
    # Define a consistent palette
    palette = {
        "FP16 (Baseline)": "#2ecc71", # Green
        "Int8 (8-bit)": "#3498db",    # Blue
        "NF4 (4-bit Normal Float)": "#9b59b6", # Purple
        "FP4 (4-bit Float)": "#e74c3c"  # Red
    }

    # --- Plot 1: Exact Match Percentage vs K ---
    sns.lineplot(
        data=df, 
        x='k', 
        y='exact_match_percentage', 
        hue='Label', 
        marker='o',
        linewidth=2.5,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('Memorization Ability: Exact Match vs. Prefix Length (k)', fontsize=16)
    axes[0].set_ylabel('Exact Match % (0.0 - 1.0)', fontsize=12)
    axes[0].set_xlabel('Prefix Length (k)', fontsize=12)
    axes[0].legend(title='Quantization Type')

    # --- Plot 2: Average Successive Correct Tokens vs K ---
    sns.lineplot(
        data=df, 
        x='k', 
        y='average_successive_correct_tokens', 
        hue='Label', 
        marker='s',
        linewidth=2.5,
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title('Sequence Continuity: Successive Correct Tokens', fontsize=16)
    axes[1].set_ylabel('Avg Successive Tokens', fontsize=12)
    axes[1].set_xlabel('Prefix Length (k)', fontsize=12)
    axes[1].legend().remove() # Remove legend to avoid clutter (same as top)

    # --- Plot 3: Model Footprint (Efficiency) ---
    # We only need one row per model for footprint
    unique_models = df.drop_duplicates(subset=['Label'])
    sns.barplot(
        data=unique_models,
        x='Label',
        y='model_footprint_gb',
        palette=palette,
        ax=axes[2]
    )
    axes[2].set_title('Memory Efficiency: Model Footprint (GB)', fontsize=16)
    axes[2].set_ylabel('VRAM Usage (GB)', fontsize=12)
    axes[2].set_xlabel('')
    
    # Add values on top of bars
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt='%.1f GB')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Graph saved to {OUTPUT_IMAGE}")

def print_summary_stats(df):
    """Calculates and prints performance drops relative to baseline."""
    print("\n--- Summary Statistics (averaged across all k) ---")
    
    # Group by label to get means
    grouped = df.groupby('Label')[['exact_match_percentage', 'overall_accuracy']].mean().sort_values('exact_match_percentage', ascending=False)
    
    baseline_score = grouped.loc["FP16 (Baseline)", "exact_match_percentage"]
    
    print(f"{'Model Variant':<25} | {'Avg Exact Match':<15} | {'Avg Accuracy':<15} | {'Retention %':<15}")
    print("-" * 80)
    
    for label, row in grouped.iterrows():
        score = row['exact_match_percentage']
        retention = (score / baseline_score) * 100
        print(f"{label:<25} | {score:.4f}          | {row['overall_accuracy']:.4f}         | {retention:.1f}%")

if __name__ == "__main__":
    df = load_data(FILE_PATH)
    if df is not None:
        print_summary_stats(df)
        generate_comparison(df)