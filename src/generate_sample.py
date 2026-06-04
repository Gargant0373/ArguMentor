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
N_PER_CLASS = 100       # 100 × 3 classes = 300 total items
N_CALIB_PER_CLASS = 5  # 5 × 3 classes = 15 calibration items
N_OVERLAP = 40         # all 4 annotators see these rows
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
) -> dict:
    """Call the feedback generator with retries; return empty dict on failure."""
    _empty = {"on_topic": True, "strength": "", "focus_area": "", "suggestion": "", "reasoning": ""}
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
                return _empty
            time.sleep(1)
    return _empty


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
    print("Sampling items for main study and calibration...")
    main_parts = []
    calib_parts = []
    
    for quality in ("low", "medium", "high"):
        pool = test_df[test_df["predicted_quality"] == quality]
        
        # Sample 85 items total per class up front
        total_needed = N_PER_CLASS + N_CALIB_PER_CLASS
        shuffled_pool = pool.sample(n=min(total_needed, len(pool)), random_state=SEED).reset_index(drop=True)
        
        # Split into main assignment (80) and calibration (5)
        main_parts.append(shuffled_pool.iloc[:N_PER_CLASS])
        calib_parts.append(shuffled_pool.iloc[N_PER_CLASS:])

    # Create the two distinct dataframes
    sample_df = pd.concat(main_parts).reset_index(drop=True)
    calib_df = pd.concat(calib_parts).reset_index(drop=True)

    # Assign IDs and track calibration status
    sample_df["item_id"] = [f"item_{i:04d}" for i in range(len(sample_df))]
    sample_df["is_calibration"] = False

    calib_df["item_id"] = [f"calib_{i:04d}" for i in range(len(calib_df))]
    calib_df["is_calibration"] = True

    # Temporarily combine them so they BOTH get feedback from Llama in Step 5
    processing_df = pd.concat([sample_df, calib_df]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Select 32 overlap items (~8 per class)
    # ------------------------------------------------------------------
    n_overlap_per_class = N_OVERLAP // 3
    overlap_ids = []
    
    # Isolate just the main sample rows to calculate overlap IDs
    main_rows = processing_df[~processing_df["is_calibration"]]
    
    for i, quality in enumerate(("low", "medium", "high")):
        pool = main_rows[main_rows["predicted_quality"] == quality]["item_id"].tolist()
        n = n_overlap_per_class + (N_OVERLAP % 3 if i == 2 else 0)
        chosen = pd.Series(pool).sample(n=min(n, len(pool)), random_state=SEED).tolist()
        overlap_ids.extend(chosen)

    processing_df["is_overlap"] = processing_df["item_id"].isin(overlap_ids)

    # ------------------------------------------------------------------
    # 5. Generate feedback for all 240 items
    # ------------------------------------------------------------------
    print("Generating feedback (this may take a while)...")
    gen = FeedbackGenerator(model="llama3.2:3b")
    on_topics, strengths, focus_areas, suggestions, reasonings = [], [], [], [], []

    total = len(processing_df)
    for idx, (_, row) in enumerate(processing_df.iterrows(), 1):
        if idx % 10 == 0 or idx == 1:
            print(f"  Feedback {idx}/{total}...")
        fb = _generate_feedback_safe(
            gen,
            topic=str(row["topic"]),
            stance=str(row["stance_str"]),
            argument=str(row["argument"]),
            predicted_quality=str(row["predicted_quality"]),
        )
        on_topics.append(fb.get("on_topic", True))
        strengths.append(fb.get("strength", ""))
        focus_areas.append(fb.get("focus_area", ""))
        suggestions.append(fb.get("suggestion", ""))
        reasonings.append(fb.get("reasoning", ""))

    processing_df["on_topic"] = on_topics
    processing_df["strength"] = strengths
    processing_df["focus_area"] = focus_areas
    processing_df["suggestion"] = suggestions
    processing_df["reasoning"] = reasonings

    # ------------------------------------------------------------------
    # 6. Split non-overlap items into 4 × 52 unique pools
    # ------------------------------------------------------------------

    # Separate calibration back out before doing individual splits
    final_calib_df = processing_df[processing_df["is_calibration"]].copy()
    main_pool_df = processing_df[~processing_df["is_calibration"]].copy()

    non_overlap = main_pool_df[~main_pool_df["is_overlap"]].sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    overlap = main_pool_df[main_pool_df["is_overlap"]].reset_index(drop=True)

    n_unique = len(non_overlap) // N_ANNOTATORS  # 52
    print(f"\nSplit: {len(non_overlap)} non-overlap items → {n_unique} per annotator")
    print(f"Overlap: {len(overlap)} items shared by all annotators")
    print(f"Total per annotator: {n_unique + len(overlap)}")

    # ------------------------------------------------------------------
    # 7. Write per-annotator CSVs
    # ------------------------------------------------------------------
    output_columns = [
        "item_id", "topic", "stance_str", "argument", "predicted_quality",
        "on_topic", "strength", "focus_area", "suggestion", "reasoning", "is_overlap",
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

        final_cols = [
            "item_id", "topic", "stance", "argument", "predicted_quality",
            "on_topic", "strength", "focus_area", "suggestion", "reasoning", "is_overlap",
            "relevance", "actionability", "clarity", "notes",
        ]

        out_path = OUTPUT_DIR / f"annotator_{i + 1}.csv"
        annotator_df[final_cols].to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({len(annotator_df)} rows)")
    
    # ------------------------------------------------------------------
    # 8. Write the master Calibration Phase CSV (PASTE HERE)
    # ------------------------------------------------------------------
    final_calib_df["relevance"] = ""
    final_calib_df["actionability"] = ""
    final_calib_df["clarity"] = ""
    final_calib_df["notes"] = ""
    final_calib_df = final_calib_df.rename(columns={"stance_str": "stance"})
    
    # We explicitly define the columns here to avoid scope issues
    calib_cols = [
        "item_id", "topic", "stance", "argument", "predicted_quality",
        "on_topic", "strength", "focus_area", "suggestion", "reasoning", "is_overlap",
        "relevance", "actionability", "clarity", "notes",
    ]
    final_calib_df = final_calib_df[calib_cols]
    
    calib_out_path = OUTPUT_DIR / "calibration.csv"
    final_calib_df.to_csv(calib_out_path, index=False)
    print(f"Wrote {calib_out_path} ({len(final_calib_df)} rows) for team calibration pilot.")

    print("\nDone. Share each annotator_N.csv with the corresponding annotator.")
    print("Run the annotation UI with:  python src/annotate_app.py --file data/annotator_N.csv")



if __name__ == "__main__":
    main()
