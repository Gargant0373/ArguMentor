from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from src.config import ZeroShotOllamaConfig
    from src.dataset import LABELS, get_data
    from src.utils.metrics import SYSTEM_PROMPT, ordinal_metrics, parse_fields, parse_label
    from src.utils.plots import save_plots
except ModuleNotFoundError:
    from config import ZeroShotOllamaConfig
    from dataset import LABELS, get_data
    from utils.metrics import SYSTEM_PROMPT, ordinal_metrics, parse_fields, parse_label
    from utils.plots import save_plots


_USER_TEMPLATE = """\
Classify the quality of the argument for a speech that is expected to {stance} the topic.

Primary question:
Would a reasonable speaker use this argument as-is in a speech?

Rubric:
- high: directly relevant to the topic and expected stance; clear and self-contained; gives a specific reason, explanation, consequence, or example; persuasive enough to use without substantial rewriting
- medium: relevant and understandable, but generic, incomplete, weakly developed, somewhat unclear, or in need of editing before use
- low: irrelevant, inconsistent with the expected stance, vague, fragmentary, incoherent, effectively empty, or in need of substantial rewriting

Rules:
- Judge the argument regardless of your own opinion about the topic.
- Evaluate only what is written. Do not invent missing support.
- Do not require citations or extensive evidence for a short argument.
- Do not reward length by itself.
- Use medium only when the argument is meaningfully between low and high.
- Output one word only.

{topic}

{stance}

{argument}

Label:"""


def _build_prompt(input_text: str) -> str:
    topic, stance, argument = parse_fields(input_text)
    return _USER_TEMPLATE.format(topic=topic, stance=stance, argument=argument)


def _classify_one(input_text: str, config: ZeroShotOllamaConfig) -> str:
    import ollama
    response = ollama.chat(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(input_text)},
        ],
        options={"temperature": config.temperature, "seed": config.seed},
    )
    return parse_label(response.message.content)


class ZeroShotOllamaPipeline:
    """Zero-shot argument quality classification via a local Ollama LLM."""

    def __init__(self, config: ZeroShotOllamaConfig | None = None) -> None:
        self.config = config or ZeroShotOllamaConfig()

    def evaluate(self, x: pd.Series, y: pd.Series, split_name: str = "test") -> dict[str, Any]:
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
                executor.submit(_classify_one, inputs[i], cfg): i for i in range(n)
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
        ord_m = ordinal_metrics(y_true, y_pred)
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = cfg.model.replace(":", "-")

        pd.DataFrame(
            {"input_text": list(x), "true_label": y_true, "pred_label": y_pred}
        ).to_csv(out_dir / f"predictions_{split_name}_{model_slug}.csv", index=False)
        print(f"  Predictions saved to {out_dir}/")

        save_plots(y_true, y_pred, out_dir, model_slug, split_name)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            **ord_m,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=LABELS, zero_division=0
            ),
            "labels": LABELS,
        }

    def run(self) -> dict[str, Any]:
        data, thresholds = get_data()
        x_test, y_test = data["test"]

        results = {
            "thresholds": {"low_upper": thresholds[0], "medium_upper": thresholds[1]},
            "test": self.evaluate(x_test, y_test, split_name="test"),
        }

        out_dir = Path(self.config.output_dir)
        model_slug = self.config.model.replace(":", "-")
        pd.DataFrame(
            [
                {"split": split, "metric": metric, "value": results[split][metric]}
                for split in ("test",)
                for metric in ("accuracy", "f1_macro", "mae", "off_by_one_acc", "qwk")
            ]
        ).to_csv(out_dir / f"summary_{model_slug}.csv", index=False)

        return results


_defaults = ZeroShotOllamaConfig()


def run_zeroshot_ollama(
    model: str = _defaults.model,
    max_samples: int | None = _defaults.max_samples,
    max_workers: int = _defaults.max_workers,
) -> dict[str, Any]:
    config = ZeroShotOllamaConfig(model=model, max_samples=max_samples, max_workers=max_workers)
    return ZeroShotOllamaPipeline(config).run()


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else _defaults.model
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else _defaults.max_samples

    print(f"=== Zero-Shot Ollama Classification ({model_name}, n={n_samples}) ===")
    results = run_zeroshot_ollama(model=model_name, max_samples=n_samples)
    test = results["test"]
    print(f"\nAccuracy : {test['accuracy']:.4f}")
    print(f"F1 Macro : {test['f1_macro']:.4f}")
    print(f"MAE      : {test['mae']:.4f}")
    print(f"QWK      : {test['qwk']:.4f}")
    print("\nClassification Report:")
    print(test["classification_report"])
