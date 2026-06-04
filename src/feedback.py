from __future__ import annotations

import json

import ollama


_FOCUS_AREAS = "Evidence, Clarity, Structure, Relevance, Logic, Warrant"

_SYSTEM_MESSAGE = (
    "You are an expert in argumentation theory and deliberative communication. "
    "Your task is to analyse the rhetorical and logical quality of arguments using "
    "the Toulmin model (claim, grounds, warrant, backing, qualifier, rebuttal) and "
    "provide constructive, academic feedback. You focus on structure, evidence, "
    "relevance, and clarity — not on the political or moral content of the argument. "
    "You must respond with valid JSON only, using exactly these keys: "
    '"on_topic", "strength", "focus_area", "suggestion", "reasoning". '
    f"focus_area must be one of: {_FOCUS_AREAS}. "
    '"on_topic" must be a boolean: true if the argument addresses the topic and stance, false if it is off-topic. '
    '"strength" must be one sentence describing what the argument does well.'
)

# Tier-specific Toulmin guidance: low → claim+grounds, medium → warrant+backing, high → qualifier+rebuttal
_QUALITY_GUIDANCE: dict[str, str] = {
    "low": (
        "Focus on foundational Toulmin elements: does the argument state a clear claim "
        "and support it with concrete grounds (data/evidence)? Check whether a warrant "
        "(the logical bridge between evidence and claim) is present at all."
    ),
    "medium": (
        "The argument has basic structure but can be strengthened. Focus on the warrant "
        "(the logical connection between evidence and claim) or backing (additional support "
        "that reinforces the warrant). These are the most common gaps at this quality level."
    ),
    "high": (
        "The argument is already strong. Focus on refinements: does it acknowledge "
        "qualifiers (conditions or limits on the claim) or preemptively address "
        "potential rebuttals? These additions make a strong argument more persuasive."
    ),
}

_USER_TEMPLATE = """\
An automated classifier has rated the following argument as {predicted_quality} quality.

Topic: {topic}
Stance: {stance}
Argument: {argument}

Feedback guidance for {predicted_quality}-quality arguments: {quality_guidance}

Respond with a JSON object with exactly five keys:
- "on_topic": boolean — true if the argument addresses the topic and stance, false if it is off-topic or irrelevant
- "strength": one sentence on what the argument does well (even low-quality arguments have something)
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
    ) -> dict:
        """Return structured feedback with keys: on_topic, strength, focus_area, suggestion, reasoning."""
        quality_key = predicted_quality.lower()
        quality_guidance = _QUALITY_GUIDANCE.get(quality_key, _QUALITY_GUIDANCE["medium"])

        user_message = _USER_TEMPLATE.format(
            topic=topic.strip(),
            stance=stance.strip(),
            argument=argument.strip(),
            predicted_quality=predicted_quality,
            quality_guidance=quality_guidance,
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
                "on_topic": bool(data.get("on_topic", True)),
                "strength": str(data.get("strength", "")).strip(),
                "focus_area": str(data.get("focus_area", "General")).strip(),
                "suggestion": str(data.get("suggestion", "")).strip(),
                "reasoning": str(data.get("reasoning", "")).strip(),
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "on_topic": True,
                "strength": "",
                "focus_area": "General",
                "suggestion": raw,
                "reasoning": "",
            }
