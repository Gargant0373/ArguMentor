from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

# Allow imports from the project root when running as `python app.py`
sys.path.insert(0, str(Path(__file__).parent))

from src.feedback import FeedbackGenerator
from src.predictor import ArgumentPredictor

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------

_predictors: dict[str, ArgumentPredictor] = {}
_feedback_generator = FeedbackGenerator(model="llama3.2:3b")

_MODEL_CHOICES = {
    "RoBERTa (fine-tuned)": "roberta",
    "TF-IDF + Logistic Regression": "logreg",
}

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
    "General": "#7f8c8d",
}


def _feedback_html(feedback: dict[str, str]) -> str:
    focus = feedback.get("focus_area", "General")
    suggestion = feedback.get("suggestion", "")
    reasoning = feedback.get("reasoning", "")
    badge_color = _FOCUS_COLORS.get(focus, "#7f8c8d")
    reasoning_block = (
        f"<div style='margin-top:10px; padding:8px 12px; "
        f"background-color:{badge_color}1a; "
        f"border-left:3px solid {badge_color}; border-radius:4px; "
        f"color:#374151; font-size:0.9em;'>"
        f"<strong style='color:#111827;'>Why this helps:</strong> {reasoning}"
        f"</div>"
        if reasoning else ""
    )
    return (
        f"<div style='border:1px solid #ddd; border-radius:8px; padding:16px; background:#fff;'>"
        f"<span style='display:inline-block; background:{badge_color}; color:#fff; "
        f"font-size:0.8em; font-weight:bold; padding:3px 10px; border-radius:12px; "
        f"margin-bottom:10px;'>{focus.upper()}</span>"
        f"<div style='font-size:1em; color:#222; line-height:1.5;'>{suggestion}</div>"
        f"{reasoning_block}"
        f"</div>"
    )


def _feedback_error_html(message: str) -> str:
    return (
        f"<div style='border:1px solid #e74c3c; border-radius:8px; padding:16px; background:#fff5f5;'>"
        f"<strong style='color:#e74c3c;'>Feedback unavailable</strong>"
        f"<pre style='margin-top:8px; font-size:0.85em; color:#555; white-space:pre-wrap;'>{message}</pre>"
        f"</div>"
    )


def _get_predictor(model_key: str) -> ArgumentPredictor:
    if model_key not in _predictors:
        _predictors[model_key] = ArgumentPredictor(model_type=model_key)
    predictor = _predictors[model_key]
    if not predictor._loaded:
        predictor.load()
    return predictor


def analyze(
    topic: str,
    stance: str,
    argument: str,
    model_choice: str,
) -> tuple[str, str]:
    """Run classification + feedback generation and return (quality_html, feedback_html)."""
    topic = topic.strip()
    stance_val = stance.strip().lower()
    argument = argument.strip()

    if not argument:
        return "", ""

    # --- Classification ---
    model_key = _MODEL_CHOICES[model_choice]
    try:
        predictor = _get_predictor(model_key)
        quality = predictor.predict(topic, stance_val, argument)
    except Exception as exc:
        return f"<p style='color:#e74c3c;'>Classification error: {exc}</p>", ""

    color = _QUALITY_COLORS.get(quality, "#888")
    emoji = _QUALITY_EMOJI.get(quality, "")
    quality_html = (
        f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>"
        f"<span style='font-size:1.5em; font-weight:700; color:{color};'>{emoji} {quality.capitalize()}</span>"
        f"<span style='color:#9ca3af; font-size:0.85em;'>via {model_choice}</span>"
        f"</div>"
    )

    # --- Feedback ---
    try:
        feedback = _feedback_generator.generate(
            topic=topic or "unknown",
            stance=stance_val,
            argument=argument,
            predicted_quality=quality,
        )
        feedback_html = _feedback_html(feedback)
    except Exception as exc:
        feedback_html = _feedback_error_html(
            f"{exc}\n\nMake sure Ollama is running with llama3.2:3b pulled:\n"
            "  ollama pull llama3.2:3b\n  ollama serve"
        )

    return quality_html, feedback_html


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ArguMentor") as demo:

    with gr.Column(elem_classes="container"):
        gr.Markdown("## ArguMentor\nRate the quality of an argument and get a targeted suggestion for improvement.")

        gr.HTML("<hr style='border:none; border-top:1px solid #e5e7eb; margin:4px 0 16px;'>")

        with gr.Row(equal_height=True):
            topic_input = gr.Textbox(
                label="Topic",
                placeholder="e.g. Should social media be regulated by the government?",
                lines=1,
                scale=3,
            )
            stance_input = gr.Radio(
                choices=["pro", "con"],
                value="pro",
                label="Stance",
                scale=1,
            )

        argument_input = gr.Textbox(
            label="Argument",
            placeholder="Type the argument to analyse here…",
            lines=5,
        )

        with gr.Row():
            model_selector = gr.Radio(
                choices=list(_MODEL_CHOICES.keys()),
                value="RoBERTa (fine-tuned)",
                label="Classifier",
                scale=3,
            )
            analyse_btn = gr.Button("Analyse", variant="primary", scale=1, min_width=120)

        gr.HTML("<hr style='border:none; border-top:1px solid #e5e7eb; margin:16px 0 8px;'>")

        quality_output = gr.HTML(
            value="",
        )
        feedback_output = gr.HTML(
            value="",
        )

        gr.HTML(
            "<p style='color:#9ca3af; font-size:0.8em; margin-top:16px;'>"
            "Feedback generated by <code>llama3.2:3b</code> via Ollama. "
            "Run <code>ollama serve</code> before clicking Analyse."
            "</p>"
        )

    analyse_btn.click(
        fn=analyze,
        inputs=[topic_input, stance_input, argument_input, model_selector],
        outputs=[quality_output, feedback_output],
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=".container { max-width: 780px; margin: 0 auto; } footer { display: none !important; }",
    )
