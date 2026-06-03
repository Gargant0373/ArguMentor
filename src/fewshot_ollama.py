from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.config import FewShotOllamaConfig
    from src.dataset import LABELS, get_data
    from src.zeroshot_ollama import (
        _INPUT_PATTERN,
        _LABEL_TO_ORD,
        _ordinal_metrics,
        _parse_label,
        _save_plots,
    )
except ModuleNotFoundError:
    from config import FewShotOllamaConfig
    from dataset import LABELS, get_data
    from zeroshot_ollama import (
        _INPUT_PATTERN,
        _LABEL_TO_ORD,
        _ordinal_metrics,
        _parse_label,
        _save_plots,
    )

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Example sampling
# ---------------------------------------------------------------------------

def _sample_examples(
    x_train: pd.Series,
    y_train: pd.Series,
    k_per_class: int,
    seed: int,
) -> list[dict[str, str]]:
    """Return k stratified examples per class from the train split."""
    rng = np.random.default_rng(seed)
    examples: list[dict[str, str]] = []
    for label in LABELS:
        mask = y_train == label
        pool = x_train[mask]
        n = min(k_per_class, len(pool))
        chosen = pool.iloc[rng.choice(len(pool), size=n, replace=False)]
        for text in chosen:
            examples.append({"input_text": text, "label": label})
    return examples


def _parse_fields(input_text: str) -> tuple[str, str, str]:
    """Extract topic, stance, argument from the dataset input format."""
    m = _INPUT_PATTERN.match(input_text)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return "", "unknown", input_text.strip()


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert argument quality assessor. "
    "You always respond with a single word."
)

_DEFINITIONS = """\
Definitions:
- low: weak reasoning, unsupported claims, irrelevant to the topic, emotionally manipulative, or logically fallacious
- medium: moderate reasoning with some support, partially relevant, but lacking depth, evidence, or clarity
- high: clear, well-reasoned, well-supported with evidence, logically sound, and directly relevant to the topic"""

_EXAMPLE_TEMPLATE = """\
Topic: {topic}
Stance: {stance}
Argument: {argument}
Quality: {label}"""

_QUERY_TEMPLATE = """\
Topic: {topic}
Stance: {stance}
Argument: {argument}
Quality:"""


def _build_prompt(input_text: str, examples: list[dict[str, str]]) -> str:
    lines = [
        "Classify the quality of the following argument as exactly one of: low, medium, or high.\n",
        _DEFINITIONS,
        "\nHere are some labeled examples:\n",
    ]
    for ex in examples:
        topic, stance, argument = _parse_fields(ex["input_text"])
        lines.append(
            _EXAMPLE_TEMPLATE.format(
                topic=topic, stance=stance, argument=argument, label=ex["label"]
            )
        )
    topic, stance, argument = _parse_fields(input_text)
    lines.append("\nNow classify this argument:")
    lines.append(
        _QUERY_TEMPLATE.format(topic=topic, stance=stance, argument=argument)
    )
    lines.append("\nRespond with a single word only: low, medium, or high.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _classify_one(
    input_text: str,
    examples: list[dict[str, str]],
    config: FewShotOllamaConfig,
) -> str:
    import ollama

    prompt = _build_prompt(input_text, examples)
    response = ollama.chat(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": config.temperature, "seed": config.seed},
    )
    return _parse_label(response.message.content)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FewShotOllamaPipeline:
    """Few-shot argument quality classification via a local Ollama LLM."""

    def __init__(self, config: FewShotOllamaConfig | None = None) -> None:
        self.config = config or FewShotOllamaConfig()
        self._examples: list[dict[str, str]] | None = None

    def _get_examples(self, x_train: pd.Series, y_train: pd.Series) -> list[dict[str, str]]:
        if self._examples is None:
            self._examples = _sample_examples(
                x_train, y_train, self.config.k_per_class, self.config.seed
            )
            k = self.config.k_per_class
            print(f"  Sampled {k} example(s)/class → {len(self._examples)} total examples")
        return self._examples

    def evaluate(
        self,
        x: pd.Series,
        y: pd.Series,
        examples: list[dict[str, str]],
        split_name: str = "test",
    ) -> dict[str, Any]:
        cfg = self.config

        if cfg.max_samples is not None and len(x) > cfg.max_samples:
            x = x.sample(n=cfg.max_samples, random_state=cfg.seed)
            y = y.loc[x.index]

        inputs = list(x)
        n = len(inputs)
        y_pred: list[str | None] = [None] * n

        print(
            f"  [{split_name}] Few-shot ({cfg.k_per_class}/class) via {cfg.model} "
            f"on {n} samples (workers={cfg.max_workers}) ..."
        )

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            future_to_idx = {
                executor.submit(_classify_one, inputs[i], examples, cfg): i
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

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = cfg.model.replace(":", "-")
        pred_path = out_dir / f"predictions_{split_name}_{model_slug}_k{cfg.k_per_class}.csv"
        pd.DataFrame(
            {"input_text": list(x), "true_label": y_true, "pred_label": y_pred}
        ).to_csv(pred_path, index=False)
        print(f"  Predictions saved to {pred_path}")

        _save_plots(y_true, y_pred, out_dir, f"{model_slug}_k{cfg.k_per_class}", split_name)

        return metrics

    def run(self) -> dict[str, Any]:
        data, thresholds = get_data()
        x_train, y_train = data["train"]
        x_test, y_test = data["test"]

        examples = self._get_examples(x_train, y_train)

        # Save and display examples
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = self.config.model.replace(":", "-")
        examples_path = out_dir / f"examples_{model_slug}_k{self.config.k_per_class}.csv"
        pd.DataFrame(examples).to_csv(examples_path, index=False)
        print(f"  Examples saved to {examples_path}")
        print(f"\n  Few-shot examples ({self.config.k_per_class}/class):")
        for i, ex in enumerate(examples):
            print(f"    [{i+1}] Label: {ex['label']}")
            print(f"        Text: {ex['input_text'][:100]}..." if len(ex['input_text']) > 100 else f"        Text: {ex['input_text']}")
        print()

        results = {
            "thresholds": {
                "low_upper": thresholds[0],
                "medium_upper": thresholds[1],
            },
            "k_per_class": self.config.k_per_class,
            "test": self.evaluate(x_test, y_test, examples, split_name="test"),
        }

        summary_path = out_dir / f"summary_{model_slug}_k{self.config.k_per_class}.csv"
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

_defaults = FewShotOllamaConfig()


def run_fewshot_ollama(
    model: str = _defaults.model,
    k_per_class: int = _defaults.k_per_class,
    max_samples: int | None = _defaults.max_samples,
    max_workers: int = _defaults.max_workers,
) -> dict[str, Any]:
    config = FewShotOllamaConfig(
        model=model,
        k_per_class=k_per_class,
        max_samples=max_samples,
        max_workers=max_workers,
    )
    return FewShotOllamaPipeline(config).run()


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else _defaults.model
    k = int(sys.argv[2]) if len(sys.argv) > 2 else _defaults.k_per_class
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else _defaults.max_samples

    print(f"=== Few-Shot Ollama Classification ({model_name}, k={k}/class, n={n_samples}) ===")
    results = run_fewshot_ollama(model=model_name, k_per_class=k, max_samples=n_samples)

    test = results["test"]
    print(f"\nAccuracy      : {test['accuracy']:.4f}")
    print(f"F1 Macro      : {test['f1_macro']:.4f}")
    print(f"MAE           : {test['mae']:.4f}")
    print(f"Off-by-one    : {test['off_by_one_acc']:.4f}")
    print(f"QWK           : {test['qwk']:.4f}")
    print(f"\nConfusion Matrix ({test['labels']}):")
    for i, row in enumerate(test["confusion_matrix"]):
        print(f"  {LABELS[i]}: {row}")
    print("\nClassification Report:")
    print(test["classification_report"])
