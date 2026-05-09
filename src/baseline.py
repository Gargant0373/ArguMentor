from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.dataset import LABELS, get_data


@dataclass
class BaselineConfig:
    max_features: int = 30000
    ngram_range: tuple[int, int] = (1, 3)
    min_df: int = 2
    c: float = 1.0
    max_iter: int = 1200
    random_state: int = 42


class BaselinePipeline:
    """Minimal TF-IDF + Logistic Regression baseline pipeline."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=self.config.ngram_range,
                        max_features=self.config.max_features,
                        min_df=self.config.min_df,
                        strip_accents="unicode",
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.c,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                        class_weight=None,
                    ),
                ),
            ]
        )

    def fit(self, x_train: pd.Series, y_train: pd.Series) -> None:
        self.model.fit(x_train, y_train)

    def evaluate(self, x: pd.Series, y: pd.Series) -> dict[str, Any]:
        y_pred = self.model.predict(x)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, y_pred, labels=LABELS).tolist(),
            "labels": LABELS,
        }

    def run(self) -> dict[str, Any]:
        data, thresholds = get_data()
        x_train, y_train = data["train"]
        x_dev, y_dev = data["dev"]
        x_test, y_test = data["test"]

        self.fit(x_train, y_train)

        return {
            "thresholds": {
                "low_upper": thresholds[0],
                "medium_upper": thresholds[1],
            },
            "dev": self.evaluate(x_dev, y_dev),
            "test": self.evaluate(x_test, y_test),
        }


def run_baseline() -> dict[str, Any]:
    return BaselinePipeline().run()


if __name__ == "__main__":
    results = run_baseline()
    print("=== Baseline Results (TF-IDF + Logistic Regression) ===")
    print(results)
