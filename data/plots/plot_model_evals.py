import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- Konfiguration ---
BASE_DIR = '/Users/benedict/UHH/bbq/data/model_eval_results'
OUTPUT_DIR = '/Users/benedict/UHH/bbq/data/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definition der zu extrahierenden Metriken pro Task
METRICS_CONFIG = {
    "gsm8k": ("exact_match,strict-match", "GSM8K (Exact Match)", True),
    "hellaswag": ("acc_norm,none", "Hellaswag (Acc Norm)", True),
    "wikitext": ("word_perplexity,none", "Wikitext (Perplexity)", False) # False = Niedriger ist besser
}

def clean_model_label(folder_name):
    """
    Erstellt ein sauberes Label aus dem Ordnernamen.
    Input: pythia-12b-duped-step143000-nf4bit_14-12-2025_14-34-03
    Output: Pythia 12B Deduped nf4bit
    """
    # 1. Datum/Zeitstempel entfernen
    name = re.split(r'_\d{2}-\d{2}-\d{4}', folder_name)[0]
    
    # 2. "step..." entfernen
    name = re.sub(r'-step\d+', '', name)
    
    # 3. Basisnamen ersetzen
    if "pythia-12b-duped" in name:
        name = name.replace("pythia-12b-duped", "Pythia 12B Deduped")
    
    # 4. Bindestriche durch Leerzeichen
    name = name.replace("-", " ")
    
    # Whitespace bereinigen
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def load_eval_data(base_dir):
    data = []
    
    if not os.path.exists(base_dir):
        print(f"Warnung: Verzeichnis {base_dir} nicht gefunden.")
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
            print(f"Fehler beim Lesen von {target_file}: {e}")

    return pd.DataFrame(data)

def plot_comparison(df):
    if df.empty:
        print("Keine Daten zum Plotten gefunden.")
        return

    # Style und Font-Größe global setzen für bessere Lesbarkeit
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
            legend=False  # Legende ausblenden, da Labels an der X-Achse sind
        )
        
        # --- HIER IST DER FIX ---
        # pad=25 schiebt den Titel nach oben
        ax.set_title(task, fontsize=14, fontweight='bold', pad=25)
        
        ax.set_xlabel("")
        ax.set_ylabel("Score / Value")
        
        # Labels rotieren
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Visueller Hinweis (Höher/Niedriger)
        # Positionierung bei y=1.02 bleibt, aber der Titel ist durch pad=25 nun weiter oben
        is_higher_better = task_data.iloc[0]['Higher_Is_Better']
        direction_text = "↑ (Höher ist besser)" if is_higher_better else "↓ (Niedriger ist besser)"
        
        ax.text(0.5, 1.02, direction_text, 
                transform=ax.transAxes, 
                ha='center', 
                fontsize=10, 
                color='#555555', # Etwas dunkleres Grau für bessere Lesbarkeit
                fontweight='medium')
        
        # Werte auf den Balken anzeigen
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)

    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'model_eval_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot gespeichert unter: {output_path}")
    plt.close()

def main():
    print("Lade Model Eval Daten...")
    df = load_eval_data(BASE_DIR)
    
    if not df.empty:
        print(f"Gefundene Datenpunkte: {len(df)}")
        print("Modelle:", df['Model'].unique())
        print("Erstelle Plot...")
        plot_comparison(df)
    else:
        print("Konnte keine Daten extrahieren.")

if __name__ == "__main__":
    main()