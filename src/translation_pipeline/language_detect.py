"""Tiny Claude call to identify the source language of a document.

Used by the CLI when ``--from`` is omitted. Sees only the first ~500 characters
and is forced to pick from the supported set via tool_use, so it can never
return a language we don't otherwise handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from translation_pipeline.config import Settings
from translation_pipeline.llm import anthropic_structured
from translation_pipeline.models import SUPPORTED_LANGUAGES, LanguageCode

if TYPE_CHECKING:
    import anthropic

_DETECT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {
            "type": "string",
            "enum": list(SUPPORTED_LANGUAGES),
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["language", "confidence"],
}


class _Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    confidence: str


def detect_language(
    client: anthropic.Anthropic,
    settings: Settings,
    text: str,
    *,
    sample_chars: int = 500,
) -> LanguageCode:
    """Return the detected source language from the supported set.

    Raises ValueError if the model returns a code not in ``SUPPORTED_LANGUAGES``
    (the tool schema constrains it, but defend in depth).
    """
    sample = text[:sample_chars]
    prompt = (
        "Identify the language of the text below. Pick exactly one code from this set: "
        f"{', '.join(SUPPORTED_LANGUAGES)}.\n\nText:\n```\n{sample}\n```\n\n"
        "Respond via the submit_detection tool."
    )
    result = anthropic_structured(
        client,
        model=settings.MODEL_LANGUAGE_DETECT,
        max_tokens=256,
        prompt=prompt,
        tool_name="submit_detection",
        tool_description="Submit the detected source language.",
        tool_schema=_DETECT_TOOL_SCHEMA,
        schema_model=_Detection,
    )
    if result.parsed.language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"detected unsupported language {result.parsed.language!r}; "
            f"supported set: {sorted(SUPPORTED_LANGUAGES)}"
        )
    return result.parsed.language
