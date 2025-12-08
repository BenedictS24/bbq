import matplotlib.pyplot as plt
import numpy as np
import json
import math

model = 'EleutherAI/pythia-12b'
x_axis = 'k'
y_axis = ['overall_accuracy', 'average_correct_tokens', 'exact_match_percentage', 'accuracy_standard_deviation']  # Choose which metric to plot
file_name = 'k8-48_memorization_results.jsonl'

def load_data(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def main():
    data = load_data(file_name)
    unique_models = list(set(d['model_name'] for d in data))
    
    num_plots = len(y_axis)
    cols = 2
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten() 
    
    for i, metric in enumerate(y_axis):
        ax = axes[i]
        
        for model_name in unique_models:
            model_data = [d for d in data if d['model_name'] == model_name]
            
            xs = [d[x_axis] for d in model_data]
            ys = [d[metric] for d in model_data] 
            
            sorted_pairs = sorted(zip(xs, ys))
            if not sorted_pairs:
                continue
            xs_sorted = [x for x, y in sorted_pairs]
            ys_sorted = [y for x, y in sorted_pairs]
            
            label_clean = model_name.replace('./', '')
            ax.plot(xs_sorted, ys_sorted, marker='o', linewidth=2, label=label_clean)
        
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel(f'{x_axis} (Context Length)', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()  


if __name__ == "__main__":
    main()