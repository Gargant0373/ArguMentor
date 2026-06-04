from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

try:
    from src.config import FinetuneConfig
    from src.dataset import LABELS, get_data
    from src.utils.metrics import ordinal_metrics
except ModuleNotFoundError:
    from config import FinetuneConfig
    from dataset import LABELS, get_data
    from utils.metrics import ordinal_metrics

LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FinetunePipeline:
    """RoBERTa fine-tuning pipeline for argument quality classification."""

    def __init__(self, config: FinetuneConfig | None = None) -> None:
        self.config = config or FinetuneConfig()
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None
        self._trainer: Trainer | None = None

    def _is_cached(self) -> bool:
        return Path(self.config.cache_dir, "config.json").exists()

    def _load_cache(self) -> None:
        print(f"Loading cached model from {self.config.cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.cache_dir
        )

    def _to_hf_dataset(self, x: pd.Series, y: pd.Series) -> Dataset:
        df = pd.DataFrame({
            "text": x.str.replace(" [SEP] ", "\n", regex=False).reset_index(drop=True),
            "labels": y.map(LABEL_TO_ID).reset_index(drop=True),
        })
        return Dataset.from_pandas(df, preserve_index=False)

    def _tokenize(self, batch: dict) -> dict:
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
        )

    @staticmethod
    def _compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        mean_ordinal_distance = float(np.mean(np.abs(predictions.astype(int) - labels.astype(int))))
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "mean_ordinal_distance": mean_ordinal_distance,
        }

    def _get_trainer(self) -> Trainer:
        if self._trainer is None:
            self._trainer = Trainer(
                model=self.model,
                compute_metrics=self._compute_metrics,
            )
        return self._trainer

    def fit(self, train_dataset: Dataset, dev_dataset: Dataset) -> None:
        """Train the model, or load from cache if already trained."""
        if self._is_cached():
            self._load_cache()
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )

        train_tok = train_dataset.map(self._tokenize, batched=True)
        dev_tok = dev_dataset.map(self._tokenize, batched=True)

        device = _detect_device()
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=(device == "cuda"),
            bf16=False,
            report_to="none",
            save_total_limit=self.config.save_total_limit,
        )

        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=dev_tok,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience
            )],
        )

        print("Starting training...")
        self._trainer.train()
        print("Training complete.")

        self.model.save_pretrained(self.config.cache_dir)
        self.tokenizer.save_pretrained(self.config.cache_dir)
        print(f"Model cached to {self.config.cache_dir}")

    def evaluate(self, dataset: Dataset) -> dict[str, Any]:
        """Evaluate on a raw (untokenized) dataset."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Call fit() first.")

        tok_dataset = dataset.map(self._tokenize, batched=True)
        pred_output = self._get_trainer().predict(tok_dataset)

        pred_labels = np.argmax(pred_output.predictions, axis=-1)
        true_labels = pred_output.label_ids
        label_names = [ID_TO_LABEL[i] for i in range(len(LABELS))]

        true_str = [ID_TO_LABEL[i] for i in true_labels]
        pred_str = [ID_TO_LABEL[i] for i in pred_labels]

        pairwise_error_rates: dict[str, float] = {}
        for true_id, true_name in ID_TO_LABEL.items():
            mask = true_labels == true_id
            if not mask.any():
                continue
            for pred_id, pred_name in ID_TO_LABEL.items():
                if pred_id == true_id:
                    continue
                key = f"{true_name}\u2192{pred_name}"
                pairwise_error_rates[key] = float(np.mean(pred_labels[mask] == pred_id))

        return {
            "accuracy": float(accuracy_score(true_labels, pred_labels)),
            "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
            **ordinal_metrics(true_str, pred_str),
            "pairwise_error_rates": pairwise_error_rates,
            "confusion_matrix": confusion_matrix(
                true_labels, pred_labels, labels=list(range(len(LABELS)))
            ).tolist(),
            "classification_report": classification_report(
                true_labels, pred_labels, target_names=label_names
            ),
            "labels": LABELS,
            "y_true": true_str,
            "y_pred": pred_str,
        }

    def run(self) -> dict[str, Any]:
        data, thresholds = get_data()

        train_dataset = self._to_hf_dataset(*data["train"])
        dev_dataset = self._to_hf_dataset(*data["dev"])
        test_dataset = self._to_hf_dataset(*data["test"])

        self.fit(train_dataset, dev_dataset)

        return {
            "thresholds": {
                "low_upper": thresholds[0],
                "medium_upper": thresholds[1],
            },
            "dev": self.evaluate(dev_dataset),
            "test": self.evaluate(test_dataset),
        }


def run_finetune() -> dict[str, Any]:
    return FinetunePipeline().run()


if __name__ == "__main__":
    results = run_finetune()
    print("\n=== Fine-tune Results (RoBERTa) ===")
    for split in ("dev", "test"):
        r = results[split]
        print(f"\n{split.upper()} — accuracy: {r['accuracy']:.4f}  macro_f1: {r['f1_macro']:.4f}  mean_ordinal_distance: {r['mean_ordinal_distance']:.4f}")
        print("  pairwise error rates: " + "  ".join(f"{k}: {v:.4f}" for k, v in r["pairwise_error_rates"].items()))
        print(r["classification_report"])
