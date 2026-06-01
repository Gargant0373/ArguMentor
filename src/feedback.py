from __future__ import annotations

import json

import ollama


_FOCUS_AREAS = "Evidence, Clarity, Structure, Relevance, Logic"

_SYSTEM_MESSAGE = (
    "You are an expert in argumentation theory and deliberative communication. "
    "Your task is to analyse the rhetorical and logical quality of arguments and "
    "provide constructive, academic feedback. You focus on structure, evidence, "
    "relevance, and clarity — not on the political or moral content of the argument. "
    f"You must respond with valid JSON only, using exactly these keys: "
    '"focus_area", "suggestion", "reasoning". '
    f"focus_area must be one of: {_FOCUS_AREAS}."
)

_USER_TEMPLATE = """\
An automated classifier has rated the following argument as {predicted_quality} quality.

Topic: {topic}
Stance: {stance}
Argument: {argument}

Respond with a JSON object with exactly three keys:
- "focus_area": the primary dimension to improve (one of: {focus_areas})
- "suggestion": one concise, specific, actionable suggestion (1-2 sentences)
- "reasoning": a brief explanation of why this improvement would strengthen the argument

Do not discuss the topic's merits. Output JSON only."""


class FeedbackGenerator:
    """Generate structured improvement feedback for an argument using a local Ollama model."""

    def __init__(self, model: str = "llama3.2:3b") -> None:
        self.model = model

    def generate(
        self,
        topic: str,
        stance: str,
        argument: str,
        predicted_quality: str,
    ) -> dict[str, str]:
        """Return structured feedback as a dict with keys: focus_area, suggestion, reasoning."""
        user_message = _USER_TEMPLATE.format(
            topic=topic.strip(),
            stance=stance.strip(),
            argument=argument.strip(),
            predicted_quality=predicted_quality,
            focus_areas=_FOCUS_AREAS,
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            format="json",
        )
        raw = response["message"]["content"].strip()
        try:
            data = json.loads(raw)
            return {
                "focus_area": str(data.get("focus_area", "General")).strip(),
                "suggestion": str(data.get("suggestion", "")).strip(),
                "reasoning": str(data.get("reasoning", "")).strip(),
            }
        except (json.JSONDecodeError, KeyError):
            # Graceful fallback: treat the raw text as the suggestion
            return {"focus_area": "General", "suggestion": raw, "reasoning": ""}
