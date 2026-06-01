from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

try:
    from src.dataset import get_data_continuous
except ModuleNotFoundError:
    from dataset import get_data_continuous


@dataclass
class FinetuneConfig:
    model_name: str = "roberta-base"
    cache_dir: str = "./argument_model-roberta"
    output_dir: str = "./results/finetune-roberta"
    max_length: int = 512
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 7
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 100
    early_stopping_patience: int = 3
    save_total_limit: int = 2


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FinetunePipeline:
    """RoBERTa fine-tuning pipeline for argument quality regression."""

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
            "labels": y.astype(float),
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
        predictions, labels = eval_pred

        predictions = predictions.squeeze()

        pearson = pearsonr(labels, predictions)[0]
        spearman = spearmanr(labels, predictions)[0]

        mae = mean_absolute_error(labels, predictions)

        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        return {
            "pearson": float(pearson),
            "spearman": float(spearman),
            "mae": float(mae),
            "rmse": float(rmse),
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
            num_labels=1,
            problem_type="regression"
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
            metric_for_best_model="pearson",
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

    def evaluate(self, dataset):
        tok_dataset = dataset.map(self._tokenize, batched=True)

        pred_output = self._get_trainer().predict(tok_dataset)

        predictions = pred_output.predictions.squeeze()
        labels = pred_output.label_ids

        pearson = pearsonr(labels, predictions)[0]
        spearman = spearmanr(labels, predictions)[0]

        mae = mean_absolute_error(labels, predictions)

        rmse = np.sqrt(
            np.mean((predictions - labels) ** 2)
        )

        return {
            "pearson": float(pearson),
            "spearman": float(spearman),
            "mae": float(mae),
            "rmse": float(rmse),
        }

    def run(self) -> dict[str, Any]:
        data = get_data_continuous()

        train_dataset = self._to_hf_dataset(*data["train"])
        dev_dataset = self._to_hf_dataset(*data["dev"])
        test_dataset = self._to_hf_dataset(*data["test"])

        self.fit(train_dataset, dev_dataset)

        return {
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
        print(f"\n{split.upper()} — pearson: {r['pearson']:.4f}  spearman: {r['spearman']:.4f}  mae: {r['mae']:.4f} rmse: {r['rmse']:.4f}")