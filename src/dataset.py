from __future__ import annotations

import numpy as np
import pandas as pd

DATASET_REPO = "ibm-research/argument_quality_ranking_30k"
QUALITY_COLUMN = "WA"
LABELS = ["low", "medium", "high"]


def _split_path(split: str, dataset_repo: str = DATASET_REPO) -> str:
    return f"hf://datasets/{dataset_repo}/{split}.csv"


def load_splits(dataset_repo: str = DATASET_REPO) -> dict[str, pd.DataFrame]:
    """Load train/dev/test splits from the dataset repository."""
    return {
        split: pd.read_csv(_split_path(split, dataset_repo))
        for split in ("train", "dev", "test")
    }


def build_input_text(df: pd.DataFrame) -> pd.Series:
    """Build model input text as Topic + Stance + Argument."""
    stance_map = {1: "pro", -1: "con"}
    topic = df["topic"].fillna("").astype(str)
    argument = df["argument"].fillna("").astype(str)
    stance = df["stance_WA"].map(stance_map).fillna("unknown")
    return "Topic: " + topic + " [SEP] Stance: " + stance + " [SEP] Argument: " + argument


def fit_quality_thresholds(train_scores: pd.Series) -> tuple[float, float]:
    """Fit low/medium/high thresholds from train-score terciles."""
    low_upper, medium_upper = np.quantile(train_scores, [1 / 3, 2 / 3])
    return float(low_upper), float(medium_upper)


def scores_to_labels(scores: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    """Convert numeric quality scores to low/medium/high labels."""
    low_upper, medium_upper = thresholds
    bins = [-np.inf, low_upper, medium_upper, np.inf]
    labels = pd.cut(scores, bins=bins, labels=LABELS, include_lowest=True)
    return labels.astype(str)

def clean_data(
    df: pd.DataFrame,
    remove_empty_argument: bool = True,
    min_argument_length: int = 5,
    remove_duplicates: bool = True,
    remove_null_quality: bool = True,
) -> pd.DataFrame:
    original_len = len(df)
    df_clean = df.copy()
    
    stats = {"Original": original_len}
    
    # Remove rows with null quality score
    if remove_null_quality and QUALITY_COLUMN in df_clean.columns:
        df_clean = df_clean[df_clean[QUALITY_COLUMN].notna()]
        stats["After removing null quality"] = len(df_clean)
    
    # Remove empty arguments
    if remove_empty_argument and 'argument' in df_clean.columns:
        df_clean = df_clean[df_clean['argument'].fillna('').str.len() > 0]
        stats["After removing empty arguments"] = len(df_clean)
    
    # Remove short arguments
    if min_argument_length > 0 and 'argument' in df_clean.columns:
        df_clean = df_clean[df_clean['argument'].fillna('').str.len() >= min_argument_length]
        stats["After removing short arguments"] = len(df_clean)
    
    # Remove exact duplicates
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()
        stats["After removing duplicates"] = len(df_clean)
    
    # Strip whitespace from text columns
    text_cols = df_clean.select_dtypes(include=['object']).columns
    for col in text_cols:
        df_clean[col] = df_clean[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
    
    stats["Final"] = len(df_clean)
    stats["Rows removed"] = original_len - len(df_clean)
    stats["Percentage retained"] = f"{(len(df_clean)/original_len*100):.2f}%"
    
    return df_clean, stats

def get_data(
    dataset_repo: str = DATASET_REPO,
    quality_column: str = QUALITY_COLUMN,
) -> tuple[dict[str, tuple[pd.Series, pd.Series]], tuple[float, float]]:
    """Return prepared split data as (X, y) pairs plus fitted thresholds."""
    splits = load_splits(dataset_repo)
    thresholds = fit_quality_thresholds(splits["train"][quality_column])

    prepared: dict[str, tuple[pd.Series, pd.Series]] = {}
    for split_name, df in splits.items():
        df = clean_data(df)[0]  # Clean the data and take the cleaned DataFrame
        x = build_input_text(df)
        y = scores_to_labels(df[quality_column], thresholds)
        prepared[split_name] = (x, y)

    return prepared, thresholds