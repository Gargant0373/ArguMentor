from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

try:
    from src.config import ZeroShotOllamaConfig
    from src.dataset import LABELS, get_data
except ModuleNotFoundError:
    from config import ZeroShotOllamaConfig
    from dataset import LABELS, get_data

_LABEL_TO_ORD: dict[str, int] = {label: i for i, label in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Ordinal metrics
# ---------------------------------------------------------------------------

def _ordinal_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Compute distance-aware metrics for ordinal labels."""
    t = np.array([_LABEL_TO_ORD[v] for v in y_true])
    p = np.array([_LABEL_TO_ORD[v] for v in y_pred])
    mae = float(np.mean(np.abs(t - p)))
    off_by_one = float(np.mean(np.abs(t - p) <= 1))
    qwk = float(cohen_kappa_score(t, p, weights="quadratic"))
    return {"mae": mae, "off_by_one_acc": off_by_one, "qwk": qwk}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_plots(y_true: list[str], y_pred: list[str], out_dir: Path, model_slug: str, split_name: str) -> None:
    """Generate and save evaluation plots."""
    slug = f"{split_name}_{model_slug}"

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
    t = np.array([_LABEL_TO_ORD[v] for v in y_true])
    p = np.array([_LABEL_TO_ORD[v] for v in y_pred])
    errors = np.abs(t - p)
    counts = [int(np.sum(errors == d)) for d in range(len(LABELS))]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["0 (exact)", "1 (adjacent)", "2 (far)"], counts, color=["#4caf50", "#ff9800", "#f44336"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# Samples")
    ax.set_title(f"Ordinal Error Distribution ({model_slug})")
    fig.tight_layout()
    fig.savefig(out_dir / f"ordinal_error_{slug}.png", dpi=150)
    plt.close(fig)

    # 3. Per-class precision & recall
    from sklearn.metrics import precision_recall_fscore_support
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
    true_counts = [y_true.count(l) for l in LABELS]
    pred_counts = [y_pred.count(l) for l in LABELS]
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


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert argument quality assessor. "
    "You always respond with a single word."
)

_USER_TEMPLATE = """\
Classify the quality of the following argument as exactly one of: low, medium, or high.

Definitions:
- low: weak reasoning, unsupported claims, irrelevant to the topic, emotionally manipulative, or logically fallacious
- medium: moderate reasoning with some support, partially relevant, but lacking depth, evidence, or clarity
- high: clear, well-reasoned, well-supported with evidence, logically sound, and directly relevant to the topic

Topic: {topic}
Stance: {stance}
Argument: {argument}

Respond with a single word only: low, medium, or high."""

_INPUT_PATTERN = re.compile(
    r"Topic:\s*(.+?)\s*\[SEP\]\s*Stance:\s*(.+?)\s*\[SEP\]\s*Argument:\s*(.+)",
    re.DOTALL,
)


def _build_prompt(input_text: str) -> str:
    """Convert the dataset input text format to an LLM-friendly prompt."""
    m = _INPUT_PATTERN.match(input_text)
    if m:
        topic, stance, argument = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    else:
        topic, stance, argument = "", "unknown", input_text.strip()
    return _USER_TEMPLATE.format(topic=topic, stance=stance, argument=argument)


def _parse_label(response: str) -> str:
    """Extract a valid low/medium/high label from the model response."""
    text = response.strip().lower()
    # Check for exact match first, then substring match
    for label in LABELS:
        if re.search(rf"\b{label}\b", text):
            return label
    return "medium"  # conservative fallback


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _classify_one(input_text: str, config: ZeroShotOllamaConfig) -> str:
    """Send a single argument to Ollama and return the predicted label."""
    import ollama

    prompt = _build_prompt(input_text)
    response = ollama.chat(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": config.temperature, "seed": config.seed},
    )
    content = response.message.content
    return _parse_label(content)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ZeroShotOllamaPipeline:
    """Zero-shot argument quality classification via a local Ollama LLM."""

    def __init__(self, config: ZeroShotOllamaConfig | None = None) -> None:
        self.config = config or ZeroShotOllamaConfig()

    def evaluate(self, x: pd.Series, y: pd.Series, split_name: str = "test") -> dict[str, Any]:
        """Run zero-shot classification on a split and return evaluation metrics."""
        cfg = self.config

        if cfg.max_samples is not None and len(x) > cfg.max_samples:
            x = x.sample(n=cfg.max_samples, random_state=cfg.seed)
            y = y.loc[x.index]

        inputs = list(x)
        n = len(inputs)
        y_pred: list[str | None] = [None] * n

        print(f"  [{split_name}] Zero-shot via {cfg.model} on {n} samples "
              f"(workers={cfg.max_workers}) ...")

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            future_to_idx = {
                executor.submit(_classify_one, inputs[i], cfg): i
                for i in range(n)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    y_pred[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  Warning: inference failed for sample {idx}: {exc}")
                    y_pred[idx] = "medium"
                completed += 1
                if completed % 50 == 0 or completed == n:
                    print(f"  Progress: {completed}/{n}")

        y_true = list(y)
        ord_metrics = _ordinal_metrics(y_true, y_pred)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mae": ord_metrics["mae"],
            "off_by_one_acc": ord_metrics["off_by_one_acc"],
            "qwk": ord_metrics["qwk"],
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=LABELS, zero_division=0
            ),
            "labels": LABELS,
        }

        # Save per-sample predictions
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = cfg.model.replace(":", "-")
        pred_path = out_dir / f"predictions_{split_name}_{model_slug}.csv"
        pd.DataFrame(
            {"input_text": list(x), "true_label": y_true, "pred_label": y_pred}
        ).to_csv(pred_path, index=False)
        print(f"  Predictions saved to {pred_path}")

        _save_plots(y_true, y_pred, out_dir, model_slug, split_name)

        return metrics

    def run(self) -> dict[str, Any]:
        data, thresholds = get_data()
        x_test, y_test = data["test"]

        results = {
            "thresholds": {
                "low_upper": thresholds[0],
                "medium_upper": thresholds[1],
            },
            "test": self.evaluate(x_test, y_test, split_name="test"),
        }

        # Save summary metrics
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = self.config.model.replace(":", "-")
        summary_path = out_dir / f"summary_{model_slug}.csv"
        pd.DataFrame(
            [
                {"split": split, "metric": metric, "value": results[split][metric]}
                for split in ("test",)
                for metric in ("accuracy", "f1_macro", "mae", "off_by_one_acc", "qwk")
            ]
        ).to_csv(summary_path, index=False)
        print(f"  Summary saved to {summary_path}")

        return results


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

_defaults = ZeroShotOllamaConfig()


def run_zeroshot_ollama(
    model: str = _defaults.model,
    max_samples: int | None = _defaults.max_samples,
    max_workers: int = _defaults.max_workers,
) -> dict[str, Any]:
    config = ZeroShotOllamaConfig(model=model, max_samples=max_samples, max_workers=max_workers)
    return ZeroShotOllamaPipeline(config).run()


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else _defaults.model
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else _defaults.max_samples

    print(f"=== Zero-Shot Ollama Classification ({model_name}, n={n_samples}) ===")
    results = run_zeroshot_ollama(model=model_name, max_samples=n_samples)

    test = results["test"]
    print(f"\nAccuracy : {test['accuracy']:.4f}")
    print(f"F1 Macro : {test['f1_macro']:.4f}")
    print(f"\nConfusion Matrix ({test['labels']}):")
    for i, row in enumerate(test["confusion_matrix"]):
        print(f"  {LABELS[i]}: {row}")
    print("\nClassification Report:")
    print(test["classification_report"])
