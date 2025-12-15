import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# --- Configuration ---
FILE_PATH = '/Users/benedict/UHH/bbq/data/experiment_data/k4-48_fp16-8bit-fp4-nf4_memorization_results.jsonl'
Y_METRICS = [
    'overall_accuracy', 
    'average_correct_tokens', 
    'exact_match_percentage', 
    'accuracy_standard_deviation'
]
X_AXIS = 'k'

def main():
    # 1. Load Data
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    df = pd.read_json(FILE_PATH, lines=True)
    df = df.sort_values(by=X_AXIS)

    # 2. Setup Plotting Style
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "grid.alpha": 0.6})
    
    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    # Variables to store legend info
    global_handles = []
    global_labels = []

    # 3. Plot Each Metric
    for i, metric in enumerate(Y_METRICS):
        ax = axes[i]
        
        sns.lineplot(
            data=df, 
            x=X_AXIS, 
            y=metric, 
            hue='model_name', 
            style='model_name', 
            markers=True, 
            dashes=False, 
            linewidth=2.5,
            markersize=8,
            ax=ax
        )

        ax.set_title(metric.replace('_', ' ').title(), fontsize=13, weight='bold')
        ax.set_xlabel("Context Length (k)", fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        
        # Capture handles/labels from the first plot only
        if i == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()
        
        # Remove the individual legend from this subplot
        if ax.get_legend():
            ax.get_legend().remove()

    # 4. Create a Global Legend and Adjust Layout
    # 'rect' reserves space at the top: [left, bottom, right, top]
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    fig.legend(
        global_handles, 
        global_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.0), # Anchored at the very top of the figure
        ncol=len(df['model_name'].unique()), 
        frameon=False,
        fontsize=12
    )

    plt.show()

if __name__ == "__main__":
    main()