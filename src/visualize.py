import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
# Create an output directory for your plots
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

ANNOTATOR_FILES = [
    DATA_DIR / "annotator_1.csv",
    DATA_DIR / "annotator_2.csv",
    DATA_DIR / "annotator_3.csv",
    DATA_DIR / "annotator_4.csv",
]

METRICS = ["relevance", "actionability", "clarity"]

# Load data
dfs = []
for i, f in enumerate(ANNOTATOR_FILES, start=1):
    df = pd.read_csv(f)
    df["annotator"] = i
    dfs.append(df)

all_ratings = pd.concat(dfs, ignore_index=True)


def plot_metric_distribution(metric):
    counts = all_ratings[metric].value_counts().sort_index()

    # ensure all rating levels 1–5 appear
    counts = counts.reindex([1, 2, 3, 4, 5], fill_value=0)

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"{metric.capitalize()} Rating Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    
    # Quick fix for your ticks: since index is cast to str above, 
    # use the string representation for positioning
    plt.xticks(range(5), ["1", "2", "3", "4", "5"])
    
    # Save the figure instead of showing it
    output_path = OUTPUT_DIR / f"{metric}_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close() # Clean up memory after saving
    print(f"Saved plot to {output_path}")


for metric in METRICS:
    plot_metric_distribution(metric)