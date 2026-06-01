"""
Generate the annotation sample for human feedback evaluation.

Produces 4 per-annotator CSV files in data/:
  data/annotator_1.csv … data/annotator_4.csv

Each file has 84 rows (52 unique + 32 shared overlap items).
Columns: item_id, topic, stance, argument, predicted_quality,
         focus_area, suggestion, reasoning, is_overlap,
         relevance, actionability, clarity, notes

Run with:
    python src/generate_sample.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import clean_data, fit_quality_thresholds, load_splits, scores_to_labels
from src.feedback import FeedbackGenerator
from src.predictor import ArgumentPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
N_PER_CLASS = 80       # 80 × 3 classes = 240 total items
N_OVERLAP = 32         # ~8 per class; all 4 annotators see these
N_ANNOTATORS = 4
OUTPUT_DIR = Path("data")
QUALITY_COLUMN = "WA"
FEEDBACK_RETRIES = 2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_feedback_safe(
    gen: FeedbackGenerator,
    topic: str,
    stance: str,
    argument: str,
    predicted_quality: str,
) -> dict[str, str]:
    """Call the feedback generator with retries; return empty dict on failure."""
    for attempt in range(FEEDBACK_RETRIES + 1):
        try:
            return gen.generate(
                topic=topic,
                stance=stance,
                argument=argument,
                predicted_quality=predicted_quality,
            )
        except Exception as exc:
            if attempt == FEEDBACK_RETRIES:
                print(f"  [WARN] Feedback failed after {FEEDBACK_RETRIES + 1} attempts: {exc}")
                return {"focus_area": "", "suggestion": "", "reasoning": ""}
            time.sleep(1)
    return {"focus_area": "", "suggestion": "", "reasoning": ""}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and clean test split
    # ------------------------------------------------------------------
    print("Loading dataset...")
    splits = load_splits()
    train_df = splits["train"]
    test_df = splits["test"]

    thresholds = fit_quality_thresholds(train_df[QUALITY_COLUMN])

    test_df, _ = clean_data(test_df)
    test_df = test_df.copy()
    test_df["predicted_label_true"] = scores_to_labels(
        test_df[QUALITY_COLUMN], thresholds
    )

    stance_map = {1: "pro", -1: "con"}
    test_df["stance_str"] = test_df["stance_WA"].map(stance_map).fillna("unknown")

    # ------------------------------------------------------------------
    # 2. Run classifier on all test items
    # ------------------------------------------------------------------
    print("Loading classifier (RoBERTa regression)...")
    predictor = ArgumentPredictor(model_type="roberta-regression")
    predictor.load()

    print(f"Running predictions on {len(test_df)} test items...")
    predictions = []
    for _, row in test_df.iterrows():
        pred = predictor.predict(
            topic=str(row["topic"]),
            stance=str(row["stance_str"]),
            argument=str(row["argument"]),
        )
        predictions.append(pred)
    test_df["predicted_quality"] = predictions

    # ------------------------------------------------------------------
    # 3. Sample 80 per predicted class → 240 items
    # ------------------------------------------------------------------
    print("Sampling items...")
    rng = pd.Series(dtype=str)
    sampled_parts = []
    for quality in ("low", "medium", "high"):
        pool = test_df[test_df["predicted_quality"] == quality]
        n = min(N_PER_CLASS, len(pool))
        sampled_parts.append(pool.sample(n=n, random_state=SEED))

    sample_df = pd.concat(sampled_parts).reset_index(drop=True)
    sample_df["item_id"] = [f"item_{i:04d}" for i in range(len(sample_df))]

    # ------------------------------------------------------------------
    # 4. Select 32 overlap items (~8 per class)
    # ------------------------------------------------------------------
    n_overlap_per_class = N_OVERLAP // 3  # 10, 10, 12 distributed below
    overlap_ids = []
    for i, quality in enumerate(("low", "medium", "high")):
        pool = sample_df[sample_df["predicted_quality"] == quality]["item_id"].tolist()
        # Give the last class any remainder
        n = n_overlap_per_class + (N_OVERLAP % 3 if i == 2 else 0)
        chosen = (
            pd.Series(pool)
            .sample(n=min(n, len(pool)), random_state=SEED)
            .tolist()
        )
        overlap_ids.extend(chosen)

    sample_df["is_overlap"] = sample_df["item_id"].isin(overlap_ids)

    # ------------------------------------------------------------------
    # 5. Generate feedback for all 240 items
    # ------------------------------------------------------------------
    print("Generating feedback (this may take a while)...")
    gen = FeedbackGenerator(model="llama3.2:3b")
    focus_areas, suggestions, reasonings = [], [], []

    total = len(sample_df)
    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        if idx % 10 == 0 or idx == 1:
            print(f"  Feedback {idx}/{total}...")
        fb = _generate_feedback_safe(
            gen,
            topic=str(row["topic"]),
            stance=str(row["stance_str"]),
            argument=str(row["argument"]),
            predicted_quality=str(row["predicted_quality"]),
        )
        focus_areas.append(fb.get("focus_area", ""))
        suggestions.append(fb.get("suggestion", ""))
        reasonings.append(fb.get("reasoning", ""))

    sample_df["focus_area"] = focus_areas
    sample_df["suggestion"] = suggestions
    sample_df["reasoning"] = reasonings

    # ------------------------------------------------------------------
    # 6. Split non-overlap items into 4 × 52 unique pools
    # ------------------------------------------------------------------
    non_overlap = sample_df[~sample_df["is_overlap"]].sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    overlap = sample_df[sample_df["is_overlap"]].reset_index(drop=True)

    n_unique = len(non_overlap) // N_ANNOTATORS  # 52
    print(f"\nSplit: {len(non_overlap)} non-overlap items → {n_unique} per annotator")
    print(f"Overlap: {len(overlap)} items shared by all annotators")
    print(f"Total per annotator: {n_unique + len(overlap)}")

    # ------------------------------------------------------------------
    # 7. Write per-annotator CSVs
    # ------------------------------------------------------------------
    output_columns = [
        "item_id", "topic", "stance_str", "argument", "predicted_quality",
        "focus_area", "suggestion", "reasoning", "is_overlap",
        "relevance", "actionability", "clarity", "notes",
    ]

    for i in range(N_ANNOTATORS):
        unique_chunk = non_overlap.iloc[i * n_unique : (i + 1) * n_unique]
        annotator_df = pd.concat([unique_chunk, overlap]).sort_values("item_id").reset_index(drop=True)

        # Add empty annotation columns
        annotator_df["relevance"] = ""
        annotator_df["actionability"] = ""
        annotator_df["clarity"] = ""
        annotator_df["notes"] = ""

        # Rename for clarity
        annotator_df = annotator_df.rename(columns={"stance_str": "stance"})

        out_cols = [c for c in output_columns if c in annotator_df.columns or c == "stance"]
        # stance_str → stance in rename, adjust:
        final_cols = [
            "item_id", "topic", "stance", "argument", "predicted_quality",
            "focus_area", "suggestion", "reasoning", "is_overlap",
            "relevance", "actionability", "clarity", "notes",
        ]

        out_path = OUTPUT_DIR / f"annotator_{i + 1}.csv"
        annotator_df[final_cols].to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({len(annotator_df)} rows)")

    print("\nDone. Share each annotator_N.csv with the corresponding annotator.")
    print("Run the annotation UI with:  python src/annotate_app.py --file data/annotator_N.csv")


if __name__ == "__main__":
    main()
