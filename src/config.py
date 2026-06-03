"""Central configuration dataclasses for all pipeline components."""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Shared dataset defaults
# ---------------------------------------------------------------------------

DATASET_REPO: str = "ibm-research/argument_quality_ranking_30k"
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Baseline (TF-IDF + Logistic Regression)
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    max_features: int = 30000
    ngram_range: tuple[int, int] = (1, 3)
    min_df: int = 2
    c: float = 1.0
    max_iter: int = 1200
    random_state: int = RANDOM_STATE


# ---------------------------------------------------------------------------
# Baseline 2 (TF-IDF + Linear SVM)
# ---------------------------------------------------------------------------

@dataclass
class Baseline2Config:
    max_features: int = 30000
    ngram_range: tuple[int, int] = (1, 3)
    min_df: int = 1
    c: float = 0.25
    max_iter: int = 5000
    random_state: int = RANDOM_STATE


# ---------------------------------------------------------------------------
# RoBERTa fine-tuning (classification)
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    model_name: str = "roberta-base"
    cache_dir: str = "./argument_model-roberta"
    output_dir: str = "./results/finetune-roberta"
    max_length: int = 512
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 100
    early_stopping_patience: int = 3
    save_total_limit: int = 2


# ---------------------------------------------------------------------------
# RoBERTa fine-tuning (regression)
# ---------------------------------------------------------------------------

@dataclass
class FinetuneRegressionConfig:
    model_name: str = "roberta-base"
    cache_dir: str = "./argument_model-roberta-regression"
    output_dir: str = "./results/finetune-roberta-regression"
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


# ---------------------------------------------------------------------------
# Zero-shot via Ollama
# ---------------------------------------------------------------------------

@dataclass
class ZeroShotOllamaConfig:
    model: str = "llama3.2:3b"
    max_samples: int | None = 500  # None = full split
    max_workers: int = 8
    temperature: float = 0.0
    seed: int = RANDOM_STATE
    output_dir: str = "./results/zeroshot"


# ---------------------------------------------------------------------------
# Few-shot via Ollama
# ---------------------------------------------------------------------------

@dataclass
class FewShotOllamaConfig:
    model: str = "llama3.2:3b"
    k_per_class: int = 1           # examples per class drawn from train
    max_samples: int | None = 500  # None = full test split
    max_workers: int = 8
    temperature: float = 0.0
    seed: int = RANDOM_STATE
    output_dir: str = "./results/fewshot"
