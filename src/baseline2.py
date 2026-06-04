from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from src.config import Baseline2Config
    from src.dataset import LABELS, get_data
    from src.utils.metrics import ordinal_metrics
except ModuleNotFoundError:
    from config import Baseline2Config
    from dataset import LABELS, get_data
    from utils.metrics import ordinal_metrics


class Baseline2Pipeline:
    """TF-IDF + Linear SVM baseline pipeline."""

    def __init__(self, config: Baseline2Config | None = None) -> None:
        self.config = config or Baseline2Config()
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
                    LinearSVC(
                        C=self.config.c,
                        class_weight=None,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

    def fit(self, x_train: pd.Series, y_train: pd.Series) -> None:
        self.model.fit(x_train, y_train)

    def evaluate(self, x: pd.Series, y: pd.Series) -> dict[str, Any]:
        y_pred = self.model.predict(x)
        y_true_list, y_pred_list = list(y), list(y_pred)
        return {
            "accuracy": float(accuracy_score(y_true_list, y_pred_list)),
            "f1_macro": float(f1_score(y_true_list, y_pred_list, average="macro", zero_division=0)),
            **ordinal_metrics(y_true_list, y_pred_list),
            "confusion_matrix": confusion_matrix(y_true_list, y_pred_list, labels=LABELS).tolist(),
            "classification_report": classification_report(y_true_list, y_pred_list, labels=LABELS, zero_division=0),
            "labels": LABELS,
            "y_true": y_true_list,
            "y_pred": y_pred_list,
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


def run_baseline2() -> dict[str, Any]:
    return Baseline2Pipeline().run()


if __name__ == "__main__":
    results = run_baseline2()
    print("=== Baseline 2 Results (TF-IDF + Linear SVM) ===")
    print(results)
