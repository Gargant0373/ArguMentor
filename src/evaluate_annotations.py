from pathlib import Path
import pandas as pd
import numpy as np
import krippendorff
import itertools
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")

ANNOTATOR_FILES = [
    DATA_DIR / "annotator_1.csv",
    DATA_DIR / "annotator_2.csv",
    DATA_DIR / "annotator_3.csv",
    DATA_DIR / "annotator_4.csv",
]

METRICS = [
    "relevance",
    "actionability",
    "clarity",
]


# -----------------------------
# Load annotations
# -----------------------------
def load_annotations():
    dfs = []

    for annotator_id, file_path in enumerate(ANNOTATOR_FILES, start=1):
        if not file_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {file_path}")

        df = pd.read_csv(file_path)
        df["annotator"] = annotator_id
        dfs.append(df)

    return dfs


# -----------------------------
# Summary stats
# -----------------------------
def compute_summary_statistics(df):
    rows = []

    for metric in METRICS:
        scores = pd.to_numeric(df[metric], errors="coerce")

        rows.append(
            {
                "metric": metric.capitalize(),
                "n": int(scores.count()),
                "mean": round(scores.mean(), 3),
                "median": round(scores.median(), 3),
                "std": round(scores.std(), 3),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Combined Visualizations Plotter
# -----------------------------
def plot_combined_distribution(df):
    """
    Melds all metric distributions into a single grouped bar chart
    and saves it to the results directory.
    """
    print("\nGenerating combined distribution visualization...")
    
    # Clean and melt data to long format for seaborn
    plot_df = df[METRICS].copy()
    for col in plot_df.columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        
    melted_df = plot_df.melt(var_name="Metric", value_name="Score").dropna()
    melted_df["Metric"] = melted_df["Metric"].str.capitalize()
    melted_df["Score"] = melted_df["Score"].astype(int)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # hue_order keeps your layout visually consistent
    ax = sns.countplot(
        data=melted_df, 
        x="Score", 
        hue="Metric", 
        palette="tab10",
        hue_order=["Actionability", "Relevance", "Clarity"]
    )
    
    # Styling
    plt.title("Distribution of Ratings Across Categories", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Score (1-5)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Metrics", loc="upper left")
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    
    # Save chart
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    chart_path = output_dir / "combined_rating_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Chart saved to: {chart_path}")


# -----------------------------
# Krippendorff alpha
# -----------------------------
def compute_alpha(overlap_df, metric):
    matrix = overlap_df.pivot(
        index="annotator",
        columns="item_id",
        values=metric,
    )

    return krippendorff.alpha(
        reliability_data=matrix.to_numpy(),
        level_of_measurement="ordinal",
    )


# -----------------------------
# Pairwise agreement
# -----------------------------
def compute_pairwise_agreement(overlap_df, metric):
    annotators = sorted(overlap_df["annotator"].unique())

    matrix = overlap_df.pivot(
        index="item_id",
        columns="annotator",
        values=metric,
    )

    results = pd.DataFrame(index=annotators, columns=annotators, dtype=float)

    for a1, a2 in itertools.product(annotators, annotators):
        if a1 == a2:
            results.loc[a1, a2] = 1.0
            continue

        pair = matrix[[a1, a2]].dropna()

        if len(pair) == 0:
            results.loc[a1, a2] = None
            continue

        results.loc[a1, a2] = (pair[a1] == pair[a2]).mean()

    return results


# -----------------------------
# Cohen's Kappa
# -----------------------------
def compute_pairwise_kappa(overlap_df, metric):
    annotators = sorted(overlap_df["annotator"].unique())

    mat = overlap_df.pivot(
        index="item_id",
        columns="annotator",
        values=metric,
    )

    results = pd.DataFrame(index=annotators, columns=annotators, dtype=float)

    for a1, a2 in itertools.product(annotators, annotators):
        if a1 == a2:
            results.loc[a1, a2] = 1.0
            continue

        pair = mat[[a1, a2]].dropna()

        if len(pair) == 0:
            results.loc[a1, a2] = np.nan
            continue

        kappa = cohen_kappa_score(pair[a1], pair[a2])
        results.loc[a1, a2] = kappa

    return results


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading annotation files...\n")

    annotator_dfs = load_annotations()
    all_ratings = pd.concat(annotator_dfs, ignore_index=True)

    # -----------------------------
    # Summary statistics
    # -----------------------------
    print("=== Annotation Statistics ===\n")

    summary = compute_summary_statistics(all_ratings)
    print(summary.to_string(index=False))
    
    # Call the visualization generation
    plot_combined_distribution(all_ratings)

    # -----------------------------
    # Overlap filtering
    # -----------------------------
    overlap_df = all_ratings[all_ratings["is_overlap"].astype(bool)].copy()

    print("\n=== Overlap Check ===\n")

    n_overlap_items = overlap_df["item_id"].nunique()
    print(f"Total overlap ratings: {len(overlap_df)}")
    print(f"Unique overlap items: {n_overlap_items}")

    expected = 40 * 4
    print(f"Expected overlap ratings: {expected}")

    if len(overlap_df) != expected:
        print(
            "\nWARNING: Overlap count differs from expected value. "
            "Some annotations may be missing."
        )

    # -----------------------------
    # Krippendorff alpha
    # -----------------------------
    print("\n=== Krippendorff's Alpha ===\n")

    alphas = {}

    for metric in METRICS:
        alpha = compute_alpha(overlap_df, metric)
        alphas[metric.capitalize()] = round(alpha, 3)

        print(f"{metric.capitalize():15}: {alpha:.3f}")
    
    # Cohen's Kappa
    print("\n\n=== Pairwise Cohen's Kappa ===")

    for metric in METRICS:
        print(f"\n--- {metric.upper()} ---")
        kappa_matrix = compute_pairwise_kappa(overlap_df, metric)
        print(kappa_matrix.round(3))

    # -----------------------------
    # Save results
    # -----------------------------
    results = summary.copy()
    results["krippendorff_alpha"] = results["metric"].map(alphas)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "annotation_statistics.csv"
    results.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

    # -----------------------------
    # Pairwise agreement analysis
    # -----------------------------
    print("\n\n=== Pairwise Agreement Matrices ===")

    for metric in METRICS:
        print(f"\n--- {metric.upper()} ---")
        pa = compute_pairwise_agreement(overlap_df, metric)
        print(pa.round(3))


if __name__ == "__main__":
    main()