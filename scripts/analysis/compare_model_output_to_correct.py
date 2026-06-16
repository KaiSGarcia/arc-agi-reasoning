### Based on code from https://www.kaggle.com/code/allegich/arc-agi-2025-visualization-all-1000-120-tasks/ ###

import json
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent # analysis --> scripts
BASE_DIR = SCRIPT_DIR.parent 
DATA_DIR = BASE_DIR / "results"

# ARC-AGI color correspondances
cmap = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])
norm = colors.Normalize(vmin=0, vmax=9)

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# display single output grid
def plot_one(ax, matrix, title, w=0.9):
    if matrix is None:
        ax.text(0.5, 0.5, "PARSE ERROR\nNo Matrix", ha='center', va='center', 
                color='#f87171', fontweight='bold', fontsize=10)
        ax.set_title(title, color='#dddddd', fontsize=12)
        ax.axis('off')
        return

    ax.imshow(matrix, cmap=cmap, norm=norm)
    
    # make grid lines based on matrix dimensions
    ax.set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(matrix))])
    
    ax.grid(visible=True, which='both', color='#666666', linewidth=w)
    ax.tick_params(axis='both', color='none', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_title(f"{title}", fontsize=11, color='#dddddd')

def compare_outputs(record, size=3.5):
    fig, axs = plt.subplots(1, 2, figsize=(size * 2, size + 1)) # put side by side

    plot_one(axs[0], record['gold_standard'], "Gold Standard") # correct answer
    plot_one(axs[1], record['model_parsed_answer'], "Model Output") # model's answer

    fig.patch.set_facecolor('#222222') # outer frame
    for ax in axs:
        ax.set_facecolor('#111111') # plot background
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    RESULTS_PATH = DATA_DIR / "v3.2_default_results.json" # change file depending on what model + condition you want to work with
    results = load_results(RESULTS_PATH)
    for i in range(0, 99): # set range to problem(s) to be visualized
        compare_outputs(results[i])
