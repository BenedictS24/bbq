import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.colors as mcolors

# --- Configuration ---
INPUT_DIR = "/Users/benedict/UHH/bbq/data/experiment_data"
FILE_END = "mem_eval_pythia-12b-duped-step143000_plus_3_others_k4-48_end_of_sequence_21122025_1425.jsonl"
FILE_START = "mem_eval_pythia-12b-duped-step143000_plus_3_others_k4-48_start_of_sequence_22122025_1740.jsonl"

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a base name for the output images
base_name = "mem_eval_comparison_start_vs_end"

def lighten_color(color, amount=0.4):
    """Lightens the given color by mixing it with white."""
    try:
        c = mcolors.to_rgb(color)
        # Blend with white (1, 1, 1)
        return tuple((1 - amount) * val + amount for val in c)
    except:
        return color

def clean_model_name(name):
    """Maps long model names to readable labels."""
    if "nf4bit" in name: return "NF4 (4-bit Normal Float)"
    elif "fp4bit" in name: return "FP4 (4-bit Float)"
    elif "8bit" in name: return "Int8 (8-bit)"
    else: return "FP16 (Baseline)"

def load_data(path, position_label):
    """Loads JSONL data and prepares dataframes with unit conversions."""
    full_path = os.path.join(INPUT_DIR, path)
    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}")
        return None
    
    data = []
    with open(full_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df['Label'] = df['model_name'].apply(clean_model_name)
    df['Position'] = position_label
    # Create a unique label for hue mapping
    df['Combined_Label'] = df['Label'] + " (" + position_label + ")"
    
    # Unit conversions
    if 'model_footprint_byte' in df.columns:
        df['model_footprint_gb'] = df['model_footprint_byte'] / (1024**3)
    if 'peak_gpu_memory_byte' in df.columns:
        df['peak_gpu_memory_gb'] = df['peak_gpu_memory_byte'] / (1024**3)
        
    return df

def generate_plots(df):
    """Generates and saves the updated comprehensive images with overlaid data."""
    sns.set_theme(style="whitegrid")
    
    model_order = ["FP16 (Baseline)", "Int8 (8-bit)", "NF4 (4-bit Normal Float)", "FP4 (4-bit Float)"]
    base_palette = {
        "FP16 (Baseline)": "#2ecc71", 
        "Int8 (8-bit)": "#3498db", 
        "NF4 (4-bit Normal Float)": "#9b59b6", 
        "FP4 (4-bit Float)": "#e74c3c"
    }

    # Create a combined palette: "End" is standard, "Start" is pale
    full_palette = {}
    hue_order = []
    for model in model_order:
        end_label = f"{model} (End of Sequence)"
        start_label = f"{model} (Start of Sequence)"
        
        full_palette[end_label] = base_palette[model]
        full_palette[start_label] = lighten_color(base_palette[model], amount=0.6)
        
        hue_order.extend([end_label, start_label])

    # --- IMAGE 1: PERFORMANCE (Core Memorization) ---
    fig1, axes1 = plt.subplots(4, 1, figsize=(12, 22))
    metrics = [
        ('overall_accuracy', 'Overall Accuracy (0.0 - 1.0)'),
        ('exact_match_percentage', 'Exact Match % (0.0 - 1.0)'),
        ('average_correct_tokens', 'Average Correct Tokens'),
        ('average_successive_correct_tokens', 'Avg Successive Correct Tokens')
    ]
    
    for i, (col, title) in enumerate(metrics):
        # Using style='Position' adds different markers for Start vs End
        sns.lineplot(data=df, x='k', y=col, hue='Combined_Label', style='Position', 
                     palette=full_palette, hue_order=hue_order, ax=axes1[i], linewidth=2.5, markers=True)
        axes1[i].set_title(title, fontsize=14, fontweight='bold')
        axes1[i].set_xlabel('Prefix Length (k)')
        if i > 0: axes1[i].get_legend().remove()
        else: axes1[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_performance.png"), bbox_inches='tight')

    # --- IMAGE 2: EFFICIENCY & RUNTIME ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 18))

    # 2.1: Runtime vs k (Overlay)
    sns.lineplot(data=df, x='k', y='runtime_seconds', hue='Combined_Label', style='Position',
                 palette=full_palette, hue_order=hue_order, ax=axes2[0], linewidth=2.5, markers=True)
    axes2[0].set_title('Runtime (Seconds) per Prefix Length (k)', fontsize=14, fontweight='bold')
    axes2[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2.2: Memory Usage (Comparison)
    unique_df = df.drop_duplicates(['Label', 'Position']).copy()
    # Sort for consistent bar groups
    unique_df['Label'] = pd.Categorical(unique_df['Label'], categories=model_order, ordered=True)
    melted_vram = unique_df.melt(id_vars=['Label', 'Position'], value_vars=['model_footprint_gb', 'peak_gpu_memory_gb'], 
                                 var_name='Metric', value_name='GB')
    # Combine Metric and Position for the bars
    melted_vram['Metric_Pos'] = melted_vram['Metric'] + " (" + melted_vram['Position'] + ")"
    sns.barplot(data=melted_vram, x='Label', y='GB', hue='Metric_Pos', ax=axes2[1], palette="Paired")
    axes2[1].set_title('VRAM Comparison: Static Footprint vs. Peak Memory', fontsize=14, fontweight='bold')
    axes2[1].legend(title='Metric & Sequence Position', bbox_to_anchor=(1.05, 1), loc='upper left')
    for container in axes2[1].containers:
        axes2[1].bar_label(container, fmt='%.1f GB', padding=3, fontsize=9)

    # 2.3: Average Runtime per Model
    avg_runtime = df.groupby(['Label', 'Position', 'Combined_Label'])['runtime_seconds'].mean().reset_index()
    avg_runtime['Label'] = pd.Categorical(avg_runtime['Label'], categories=model_order, ordered=True)
    sns.barplot(data=avg_runtime, x='Label', y='runtime_seconds', hue='Combined_Label', palette=full_palette, ax=axes2[2])
    axes2[2].set_title('Average Runtime per Model (Across all k)', fontsize=14, fontweight='bold')
    axes2[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for container in axes2[2].containers:
        axes2[2].bar_label(container, fmt='%.1f s', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_efficiency.png"), bbox_inches='tight')

    # --- IMAGE 3: VARIABILITY ---
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 10))
    stds = [
        ('accuracy_standard_deviation', 'Accuracy Std Deviation'),
        ('successive_correct_standard_deviation', 'Successive Tokens Std Deviation')
    ]
    
    for i, (col, title) in enumerate(stds):
        sns.lineplot(data=df, x='k', y=col, hue='Combined_Label', style='Position',
                     palette=full_palette, hue_order=hue_order, ax=axes3[i], linewidth=2.5, markers=True)
        axes3[i].set_title(title, fontsize=14, fontweight='bold')
        axes3[i].set_xlabel('Prefix Length (k)')
        if i > 0: axes3[i].get_legend().remove()
        else: axes3[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_variability.png"), bbox_inches='tight')

if __name__ == "__main__":
    df_end = load_data(FILE_END, "End of Sequence")
    df_start = load_data(FILE_START, "Start of Sequence")
    
    if df_end is not None and df_start is not None:
        combined_df = pd.concat([df_end, df_start], ignore_index=True)
        generate_plots(combined_df)
        print(f"Comprehensive comparison plots saved in the '{OUTPUT_DIR}' directory.")