from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch

ModelType = Literal["roberta", "roberta-regression", "logreg"]

_ROBERTA_CACHE = "./argument_model-roberta"
_ROBERTA_REGRESSION_CACHE = "./argument_model-roberta-regression"


def _build_input(topic: str, stance: str, argument: str) -> str:
    """Build model input text matching the training format."""
    return f"Topic: {topic} [SEP] Stance: {stance} [SEP] Argument: {argument}"


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ArgumentPredictor:
    """Single-item inference wrapper for argument quality classifiers."""

    def __init__(self, model_type: ModelType = "roberta-regression") -> None:
        if model_type not in ("roberta", "roberta-regression", "logreg"):
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Use 'roberta', 'roberta-regression', or 'logreg'."
            )
        self.model_type = model_type
        self._loaded = False

        # RoBERTa state
        self._tokenizer = None
        self._model = None
        self._device: str | None = None
        self._thresholds: tuple[float, float] | None = None
        self._id_to_label: dict[int, str] | None = None

        # LogReg state
        self._pipeline = None  # sklearn Pipeline

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load (or train) the chosen model. Idempotent."""
        if self._loaded:
            return
        if self.model_type == "roberta":
            self._load_roberta()
        elif self.model_type == "roberta-regression":
            self._load_roberta_regression()
        else:
            self._load_logreg()
        self._loaded = True

    def _load_roberta(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            from src.finetune import ID_TO_LABEL, LABEL_TO_ID, FinetunePipeline
        except ModuleNotFoundError:
            from finetune import ID_TO_LABEL, LABEL_TO_ID, FinetunePipeline

        self._id_to_label = ID_TO_LABEL

        if Path(_ROBERTA_CACHE, "config.json").exists():
            print(f"Loading cached RoBERTa model from {_ROBERTA_CACHE}")
            self._tokenizer = AutoTokenizer.from_pretrained(_ROBERTA_CACHE)
            self._model = AutoModelForSequenceClassification.from_pretrained(_ROBERTA_CACHE)
        else:
            print("No cached model found — training RoBERTa (this will take a while)...")
            pipeline = FinetunePipeline()
            pipeline.run()
            self._tokenizer = pipeline.tokenizer
            self._model = pipeline.model

        self._device = _detect_device()
        self._model = self._model.to(self._device)
        self._model.eval()

    def _load_roberta_regression(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            from src.dataset import fit_quality_thresholds, load_splits
            from src.finetune_regression import FinetunePipeline
        except ModuleNotFoundError:
            from dataset import fit_quality_thresholds, load_splits
            from finetune_regression import FinetunePipeline

        self._thresholds = fit_quality_thresholds(load_splits()["train"]["WA"])

        if Path(_ROBERTA_REGRESSION_CACHE, "config.json").exists():
            print(f"Loading cached RoBERTa regression model from {_ROBERTA_REGRESSION_CACHE}")
            self._tokenizer = AutoTokenizer.from_pretrained(_ROBERTA_REGRESSION_CACHE)
            self._model = AutoModelForSequenceClassification.from_pretrained(_ROBERTA_REGRESSION_CACHE)
        else:
            print("No cached regression model found — training RoBERTa regression (this will take a while)...")
            pipeline = FinetunePipeline()
            pipeline.run()
            self._tokenizer = pipeline.tokenizer
            self._model = pipeline.model

        self._device = _detect_device()
        self._model = self._model.to(self._device)
        self._model.eval()

    def _load_logreg(self) -> None:
        try:
            from src.baseline import BaselinePipeline
            from src.dataset import get_data
        except ModuleNotFoundError:
            from baseline import BaselinePipeline
            from dataset import get_data

        print("Training TF-IDF + LogReg baseline...")
        data, _ = get_data()
        x_train, y_train = data["train"]
        bl = BaselinePipeline()
        bl.fit(x_train, y_train)
        self._pipeline = bl.model  # sklearn Pipeline

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, topic: str, stance: str, argument: str) -> str:
        """Classify a single argument. Returns 'low', 'medium', or 'high'."""
        if not self._loaded:
            self.load()

        text = _build_input(topic, stance, argument)

        if self.model_type == "roberta":
            return self._predict_roberta(text)
        if self.model_type == "roberta-regression":
            return self._predict_roberta_regression(text)
        return self._predict_logreg(text)

    def _predict_roberta(self, text: str) -> str:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        pred_id = int(logits.argmax(dim=-1).item())
        return self._id_to_label[pred_id]

    def _predict_roberta_regression(self, text: str) -> str:
        if self._thresholds is None:
            raise RuntimeError("Regression thresholds are not loaded.")

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            score = float(self._model(**inputs).logits.squeeze().item())

        low_upper, medium_upper = self._thresholds
        if score <= low_upper:
            return "low"
        if score <= medium_upper:
            return "medium"
        return "high"

    def _predict_logreg(self, text: str) -> str:
        result = self._pipeline.predict(pd.Series([text]))
        return str(result[0])
