"""
Annotation UI for human feedback evaluation.

Usage:
    python src/annotate_app.py --file data/annotator_1.csv

The app auto-resumes from the first unannotated row.
Scores are written to the CSV after every Save & Next click.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Style constants (mirrors app.py)
# ---------------------------------------------------------------------------

_QUALITY_COLORS = {
    "low": "#e74c3c",
    "medium": "#f39c12",
    "high": "#27ae60",
}
_QUALITY_EMOJI = {
    "low": "🔴",
    "medium": "🟡",
    "high": "🟢",
}
_FOCUS_COLORS = {
    "Evidence": "#3498db",
    "Clarity": "#9b59b6",
    "Structure": "#e67e22",
    "Relevance": "#1abc9c",
    "Logic": "#e74c3c",
    "Warrant": "#2980b9",
    "General": "#7f8c8d",
}

# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _quality_html(quality: str) -> str:
    color = _QUALITY_COLORS.get(quality, "#888")
    emoji = _QUALITY_EMOJI.get(quality, "")
    return (
        f"<span style='font-size:1.1em; font-weight:700; color:{color};'>"
        f"{emoji} {quality.capitalize()}</span>"
    )


def _feedback_html(focus: str, suggestion: str, reasoning: str, strength: str = "", on_topic: bool = True) -> str:
    badge_color = _FOCUS_COLORS.get(focus, "#7f8c8d")
    off_topic_banner = (
        "<div style='margin-bottom:10px; padding:8px 12px; background:#fff3cd; "
        "border-left:3px solid #f39c12; border-radius:4px; color:#856404; font-size:0.9em;'>"
        "⚠️ <strong>Potentially off-topic:</strong> This argument may not address the stated topic or stance. "
        "The feedback below may be less reliable."
        "</div>"
    ) if not on_topic else ""
    strength_block = (
        f"<div style='margin-bottom:10px; padding:8px 12px; background:#d4edda; "
        f"border-left:3px solid #27ae60; border-radius:4px; color:#155724; font-size:0.9em;'>"
        f"<strong>What works:</strong> {strength}"
        f"</div>"
    ) if strength else ""
    reasoning_block = (
        f"<div style='margin-top:10px; padding:8px 12px; "
        f"background-color:{badge_color}1a; "
        f"border-left:3px solid {badge_color}; border-radius:4px; "
        f"color:#374151; font-size:0.9em;'>"
        f"<strong style='color:#111827;'>Why this helps:</strong> {reasoning}"
        f"</div>"
    ) if reasoning else ""
    return (
        f"<div style='border:1px solid #ddd; border-radius:8px; padding:16px; background:#fff;'>"
        f"{off_topic_banner}"
        f"{strength_block}"
        f"<span style='display:inline-block; background:{badge_color}; color:#fff; "
        f"font-size:0.8em; font-weight:bold; padding:3px 10px; border-radius:12px; "
        f"margin-bottom:10px;'>{focus.upper() if focus else 'FEEDBACK'}</span>"
        f"<div style='font-size:1em; color:#222; line-height:1.5;'>{suggestion}</div>"
        f"{reasoning_block}"
        f"</div>"
    )


def _overlap_badge() -> str:
    return (
        "<span style='display:inline-block; background:#6366f1; color:#fff; "
        "font-size:0.75em; font-weight:bold; padding:2px 8px; border-radius:10px; "
        "margin-left:8px;'>OVERLAP</span>"
    )


def _progress_html(current: int, total: int, done: int) -> str:
    pct = int(done / total * 100) if total else 0
    return (
        f"<div style='font-size:0.9em; color:#6b7280; margin-bottom:4px;'>"
        f"Item <strong>{current + 1}</strong> of <strong>{total}</strong> "
        f"&nbsp;·&nbsp; {done} annotated ({pct}%)"
        f"</div>"
        f"<div style='height:6px; background:#e5e7eb; border-radius:3px;'>"
        f"<div style='height:6px; width:{pct}%; background:#6366f1; border-radius:3px;'></div>"
        f"</div>"
    )

# ---------------------------------------------------------------------------
# Annotation state
# ---------------------------------------------------------------------------

class AnnotationState:
    def __init__(self, csv_path: Path) -> None:
        self.path = csv_path
        self.df = pd.read_csv(csv_path, dtype=str).fillna("")
        self.total = len(self.df)
        # Start from first unannotated row
        unannotated = self.df[self.df["relevance"] == ""].index.tolist()
        self.index = unannotated[0] if unannotated else self.total - 1

    def save_row(self, idx: int, relevance: str, actionability: str, clarity: str, notes: str) -> None:
        self.df.at[idx, "relevance"] = relevance
        self.df.at[idx, "actionability"] = actionability
        self.df.at[idx, "clarity"] = clarity
        self.df.at[idx, "notes"] = notes
        self.df.to_csv(self.path, index=False)

    @property
    def done(self) -> int:
        return int((self.df["relevance"] != "").sum())

    def row(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]


# Populated by build_app()
_state: AnnotationState | None = None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _render(idx: int) -> tuple:
    """Return all output values for the item at `idx`."""
    row = _state.row(idx)
    is_overlap = str(row.get("is_overlap", "")).lower() in ("true", "1", "yes")

    title = f"<div>{_quality_html(str(row['predicted_quality']))}"
    if is_overlap:
        title += _overlap_badge()
    title += "</div>"

    topic_val = str(row.get("topic", ""))
    stance_val = str(row.get("stance", ""))
    argument_val = str(row.get("argument", ""))
    on_topic_val = str(row.get("on_topic", "True")).lower() not in ("false", "0", "no")
    feedback = _feedback_html(
        focus=str(row.get("focus_area", "")),
        suggestion=str(row.get("suggestion", "")),
        reasoning=str(row.get("reasoning", "")),
        strength=str(row.get("strength", "")),
        on_topic=on_topic_val,
    )
    progress = _progress_html(idx, _state.total, _state.done)

    # Pre-fill scores if already annotated
    r = str(row.get("relevance", ""))
    a = str(row.get("actionability", ""))
    c = str(row.get("clarity", ""))
    n = str(row.get("notes", ""))
    r_val = int(r) if r.isdigit() and r in ("1", "2", "3", "4", "5") else None
    a_val = int(a) if a.isdigit() and a in ("1", "2", "3", "4", "5") else None
    c_val = int(c) if c.isdigit() and c in ("1", "2", "3", "4", "5") else None

    return progress, title, topic_val, stance_val, argument_val, feedback, r_val, a_val, c_val, n


def build_app(csv_path: Path) -> gr.Blocks:
    global _state
    _state = AnnotationState(csv_path)

    score_choices = [1, 2, 3, 4, 5]
    
    score_info = (
        "**1** = Poor &nbsp;·&nbsp; **2** = Below Average &nbsp;·&nbsp; "
        "**3** = Average / Fair &nbsp;·&nbsp; **4** = Good &nbsp;·&nbsp; **5** = Excellent"
    )

    with gr.Blocks(title="ArguMentor — Annotation") as app:
        idx_state = gr.State(value=_state.index)

        with gr.Column(elem_id="main"):
            gr.Markdown(f"## ArguMentor — Feedback Annotation\n`{csv_path.name}`")
            gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:4px 0 12px;'>")

            progress_display = gr.HTML()

            with gr.Row():
                prev_btn = gr.Button("← Previous", scale=1, variant="secondary")
                item_header = gr.HTML(scale=3)

            with gr.Row():
                with gr.Column(scale=2):
                    topic_display = gr.Textbox(label="Topic", interactive=False, lines=1)
                    stance_display = gr.Textbox(label="Stance", interactive=False, lines=1)
                    argument_display = gr.Textbox(label="Argument", interactive=False, lines=5)

                with gr.Column(scale=2):
                    feedback_display = gr.HTML(label="Feedback")

            gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:12px 0 8px;'>")
            gr.Markdown(f"### Your ratings\n{score_info}")

            with gr.Row():
                relevance_input = gr.Radio(choices=score_choices, label="Relevance", info="Does the feedback address a real issue in the argument?")
                actionability_input = gr.Radio(choices=score_choices, label="Actionability", info="Does the feedback give a concrete way to improve?")
                clarity_input = gr.Radio(choices=score_choices, label="Clarity", info="Is the feedback easy to understand and specific?")

            notes_input = gr.Textbox(label="Notes (optional)", lines=2, placeholder="Any comments about this feedback item…")

            save_btn = gr.Button("Save & Next →", variant="primary")

        # ------------------------------------------------------------------
        # Event: load item on idx change
        # ------------------------------------------------------------------
        def load_item(idx):
            return _render(idx)

        outputs = [
            progress_display, item_header,
            topic_display, stance_display, argument_display, feedback_display,
            relevance_input, actionability_input, clarity_input, notes_input,
        ]

        app.load(fn=load_item, inputs=[idx_state], outputs=outputs)

        # ------------------------------------------------------------------
        # Save & Next
        # ------------------------------------------------------------------
        def save_and_next(idx, r, a, c, notes):
            if r is None or a is None or c is None:
                gr.Warning("Please rate all three dimensions before saving.")
                return [idx] + list(_render(idx))
            _state.save_row(idx, str(r), str(a), str(c), notes or "")
            next_idx = min(idx + 1, _state.total - 1)
            return [next_idx] + list(_render(next_idx))

        save_btn.click(
            fn=save_and_next,
            inputs=[idx_state, relevance_input, actionability_input, clarity_input, notes_input],
            outputs=[idx_state] + outputs,
        )

        # ------------------------------------------------------------------
        # Previous
        # ------------------------------------------------------------------
        def go_prev(idx):
            prev_idx = max(idx - 1, 0)
            return [prev_idx] + list(_render(prev_idx))

        prev_btn.click(
            fn=go_prev,
            inputs=[idx_state],
            outputs=[idx_state] + outputs,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ArguMentor annotation UI")
    parser.add_argument("--file", required=True, help="Path to annotator CSV file")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    app = build_app(csv_path)
    app.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
