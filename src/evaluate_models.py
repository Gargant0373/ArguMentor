"""Unified evaluation of all argument quality models with consistent metrics and plots."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.baseline import run_baseline
    from src.baseline2 import run_baseline2
    from src.fewshot_ollama import run_fewshot_ollama
    from src.finetune import run_finetune
    from src.finetune_regression import run_finetune as run_finetune_regression
    from src.zeroshot_ollama import run_zeroshot_ollama
    from src.utils.plots import save_comparison_plots, save_plots
except ModuleNotFoundError:
    from baseline import run_baseline
    from baseline2 import run_baseline2
    from fewshot_ollama import run_fewshot_ollama
    from finetune import run_finetune
    from finetune_regression import run_finetune as run_finetune_regression
    from zeroshot_ollama import run_zeroshot_ollama
    from utils.plots import save_comparison_plots, save_plots

_METRICS = ["accuracy", "f1_macro", "mae", "off_by_one_acc", "qwk"]
_REGRESSION_EXTRAS = ["pearson", "spearman", "rmse"]
_ALL_SCALAR_METRICS = _METRICS + _REGRESSION_EXTRAS


def _save_metrics_csv(data: dict[str, Any], out_dir: Path | str) -> None:
    """Save all scalar metrics for a single model to a long-format CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": m, "value": data[m]} for m in _ALL_SCALAR_METRICS if m in data]
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    print(f"  Metrics CSV saved to {out_dir}/metrics.csv")


def _save_comparison_csv(results: dict[str, dict[str, Any]], out_dir: Path | str) -> None:
    """Save the comparison table as a wide-format CSV (one row per model)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"model": name, **{m: data.get(m) for m in _ALL_SCALAR_METRICS}}
        for name, data in results.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    print(f"  Comparison CSV saved to {out_dir}/comparison.csv")


def _print_table(results: dict[str, dict[str, Any]]) -> None:
    names = list(results.keys())
    col_w = 14
    name_w = 17
    header = f"{'Metric':<{name_w}}" + "".join(f" {n:<{col_w}}" for n in names)
    print(header)
    print("-" * len(header))
    for metric in _ALL_SCALAR_METRICS:
        row = f"{metric:<{name_w}}"
        for data in results.values():
            val = f"{data[metric]:.4f}" if metric in data else "N/A"
            row += f" {val:<{col_w}}"
        print(row)


def _save_model_plots(data: dict[str, Any], out_dir: str | Path, model_slug: str) -> None:
    """Save individual per-model plots if y_true/y_pred are available."""
    if "y_true" not in data or "y_pred" not in data:
        return
    save_plots(data["y_true"], data["y_pred"], Path(out_dir), model_slug, "test")


def main(
    ollama_model: str = "llama3.2:3b",
    k_values: list[int] | None = None,
    max_samples: int | None = 500,
) -> dict[str, dict[str, Any]]:
    if k_values is None:
        k_values = [1, 2, 3]

    results: dict[str, dict[str, Any]] = {}
    model_slug = ollama_model.replace(":", "-")

    # --- Baselines ---
    print("\n" + "=" * 60)
    print("TF-IDF + Logistic Regression")
    print("=" * 60)
    results["LogReg"] = run_baseline()["test"]
    _save_model_plots(results["LogReg"], "./results/logreg", "logreg")
    _save_metrics_csv(results["LogReg"], "./results/logreg")

    print("\n" + "=" * 60)
    print("TF-IDF + Linear SVM")
    print("=" * 60)
    results["LinearSVM"] = run_baseline2()["test"]
    _save_model_plots(results["LinearSVM"], "./results/linearsvm", "linearsvm")
    _save_metrics_csv(results["LinearSVM"], "./results/linearsvm")

    # --- RoBERTa ---
    print("\n" + "=" * 60)
    print("RoBERTa Classification")
    print("=" * 60)
    results["RoBERTa-Clf"] = run_finetune()["test"]
    _save_model_plots(results["RoBERTa-Clf"], "./results/roberta_classification", "roberta-clf")
    _save_metrics_csv(results["RoBERTa-Clf"], "./results/roberta_classification")

    print("\n" + "=" * 60)
    print("RoBERTa Regression (mapped to low/medium/high)")
    print("=" * 60)
    results["RoBERTa-Reg"] = run_finetune_regression()["test"]
    _save_model_plots(results["RoBERTa-Reg"], "./results/roberta_regression", "roberta-reg")
    _save_metrics_csv(results["RoBERTa-Reg"], "./results/roberta_regression")

    # --- Ollama (pipelines save plots/predictions internally) ---
    print("\n" + "=" * 60)
    print(f"Zero-Shot  ({ollama_model})")
    print("=" * 60)
    results["Zero-Shot"] = run_zeroshot_ollama(model=ollama_model, max_samples=max_samples)["test"]
    _save_metrics_csv(results["Zero-Shot"], "./results/zeroshot")

    for k in k_values:
        label = f"Few-Shot k={k}"
        print("\n" + "=" * 60)
        print(f"{label}  ({ollama_model})")
        print("=" * 60)
        results[label] = run_fewshot_ollama(
            model=ollama_model, k_per_class=k, max_samples=max_samples
        )["test"]
        _save_metrics_csv(results[label], f"./results/fewshot_k{k}")

    # --- Comparison table ---
    sep = "=" * (17 + len(results) * 15)
    print(f"\n\n{sep}")
    print("Comparison Table (Test Set)")
    print(sep)
    _print_table(results)

    # --- Classification reports ---
    print(f"\n{sep}")
    print("Classification Reports (Test Set)")
    print(sep)
    for name, data in results.items():
        print(f"\n{name}:")
        if "classification_report" in data:
            print(data["classification_report"])
        if "pearson" in data:
            print(f"  Pearson: {data['pearson']:.4f}  Spearman: {data['spearman']:.4f}  RMSE: {data['rmse']:.4f}")

    # --- Comparison plots + CSV ---
    compare_dir = Path("./results/compare_all")
    save_comparison_plots(results, compare_dir, model_slug=model_slug)
    _save_comparison_csv(results, compare_dir)

    return results


if __name__ == "__main__":
    ollama_model_arg = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
    k_args = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [1, 2, 3]
    main(ollama_model=ollama_model_arg, k_values=k_args)
