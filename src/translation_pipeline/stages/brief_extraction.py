from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from translation_pipeline.config import Settings
from translation_pipeline.llm import anthropic_structured
from translation_pipeline.models import (
    PipelineState,
    StageCriticality,
    TranslationBrief,
)
from translation_pipeline.pricing import estimate_cost
from translation_pipeline.prompts import load_prompt
from translation_pipeline.stages.base import PipelineStage

if TYPE_CHECKING:
    import anthropic

# Hand-written JSON schema (rather than `TranslationBrief.model_json_schema()`) so the
# tool-call shape stays stable even if the Pydantic model's serialization changes.
_BRIEF_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "register_level": {"type": "string"},
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_term": {"type": "string"},
                    "target_term": {"type": "string"},
                    "note": {"type": ["string", "null"]},
                },
                "required": ["source_term", "target_term"],
            },
        },
        "cultural_notes": {"type": "array", "items": {"type": "string"}},
        "target_audience": {"type": "string"},
        "special_instructions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["document_type", "register_level", "target_audience"],
}


class BriefExtractionStage(PipelineStage):
    name: ClassVar[str] = "brief_extraction"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings,
        *,
        max_tokens: int = 2048,
    ) -> None:
        self._client = client
        self._settings = settings
        self._max_tokens = max_tokens
        self.model_id = settings.MODEL_BRIEF_EXTRACTION
        prompt = load_prompt("brief_extraction")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        rendered = self._prompt.render(
            source_language=state.language_pair.source,
            target_language=state.language_pair.target,
            source_text=state.source.text,
        )
        result = anthropic_structured(
            self._client,
            model=self._settings.MODEL_BRIEF_EXTRACTION,
            max_tokens=self._max_tokens,
            prompt=rendered,
            tool_name="submit_brief",
            tool_description="Submit the structured translation brief.",
            tool_schema=_BRIEF_TOOL_SCHEMA,
            schema_model=TranslationBrief,
        )
        state.brief = result.parsed
        cost = estimate_cost(
            self._settings.MODEL_BRIEF_EXTRACTION,
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        self._record_usage(
            result.usage.input_tokens, result.usage.output_tokens, cost
        )
        return state
