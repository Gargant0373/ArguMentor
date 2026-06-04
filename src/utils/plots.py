"""Shared plotting utilities for argument quality evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

try:
    from src.dataset import LABELS
    from src.utils.metrics import LABEL_TO_ORD
except ModuleNotFoundError:
    from dataset import LABELS
    from utils.metrics import LABEL_TO_ORD


def save_plots(
    y_true: list[str],
    y_pred: list[str],
    out_dir: Path,
    model_slug: str,
    split_name: str,
) -> None:
    """Generate and save per-model evaluation plots (confusion matrix, ordinal error, precision/recall, label distribution)."""
    slug = f"{split_name}_{model_slug}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({model_slug})")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{slug}.png", dpi=150)
    plt.close(fig)

    # 2. Ordinal error distribution (|pred - true|)
    t = np.array([LABEL_TO_ORD[v] for v in y_true])
    p = np.array([LABEL_TO_ORD[v] for v in y_pred])
    errors = np.abs(t - p)
    counts = [int(np.sum(errors == d)) for d in range(len(LABELS))]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["0 (exact)", "1 (adjacent)", "2 (far)"], counts,
                  color=["#4caf50", "#ff9800", "#f44336"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# Samples")
    ax.set_title(f"Ordinal Error Distribution ({model_slug})")
    fig.tight_layout()
    fig.savefig(out_dir / f"ordinal_error_{slug}.png", dpi=150)
    plt.close(fig)

    # 3. Per-class precision & recall
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    x = np.arange(len(LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, prec, width, label="Precision", color="#2196f3")
    ax.bar(x + width / 2, rec, width, label="Recall", color="#ff5722")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"Per-class Precision & Recall ({model_slug})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"precision_recall_{slug}.png", dpi=150)
    plt.close(fig)

    # 4. True vs predicted label distribution
    true_counts = [y_true.count(label) for label in LABELS]
    pred_counts = [y_pred.count(label) for label in LABELS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, true_counts, width, label="True", color="#9c27b0")
    ax.bar(x + width / 2, pred_counts, width, label="Predicted", color="#00bcd4")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("# Samples")
    ax.set_title(f"Label Distribution: True vs Predicted ({model_slug})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"label_distribution_{slug}.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


def save_comparison_plots(
    results: dict[str, dict],
    out_dir: Path,
    model_slug: str,
) -> None:
    """Save a grouped bar chart and side-by-side confusion matrices comparing all configurations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    n_configs = len(names)
    colors = ["#2196f3", "#4caf50", "#ff9800", "#f44336", "#9c27b0"]

    # 1. Grouped bar chart across key metrics
    metrics = ["accuracy", "f1_macro", "mae", "off_by_one_acc", "qwk"]
    x = np.arange(len(metrics))
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(max(12, n_configs * 2), 6))
    for i, name in enumerate(names):
        vals = [results[name].get(m, float("nan")) for m in metrics]
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title(f"Ollama Model Comparison ({model_slug})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_metrics_{model_slug}.png", dpi=150)
    plt.close(fig)
    print(f"  Metrics comparison plot saved to {out_dir}/comparison_metrics_{model_slug}.png")

    # 2. Side-by-side confusion matrices
    ncols = min(n_configs, 4)
    nrows = (n_configs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n_configs > 1 else [axes]

    for ax, (name, data) in zip(axes_flat, results.items()):
        cm_data = data.get("confusion_matrix")
        if cm_data is None:
            ax.axis("off")
            continue
        cm = np.array(cm_data, dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
        ax.set_title(name, fontsize=9)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)

    for ax in axes_flat[n_configs:]:
        ax.axis("off")

    fig.suptitle(f"Confusion Matrices — {model_slug}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_cm_{model_slug}.png", dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix grid saved to {out_dir}/comparison_cm_{model_slug}.png")
