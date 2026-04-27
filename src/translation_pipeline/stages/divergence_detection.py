from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from translation_pipeline.config import Settings
from translation_pipeline.llm import anthropic_structured
from translation_pipeline.models import (
    Divergence,
    PipelineState,
    StageCriticality,
)
from translation_pipeline.pricing import estimate_cost
from translation_pipeline.prompts import load_prompt
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
    StageError,
)

if TYPE_CHECKING:
    import anthropic


_DIVERGENCE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "divergences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment": {"type": "string"},
                    "source_text": {"type": "string"},
                    "back_translated_text": {"type": "string"},
                    "severity": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": [
                    "segment",
                    "source_text",
                    "back_translated_text",
                    "severity",
                    "description",
                ],
            },
        }
    },
    "required": ["divergences"],
}


class _ToolPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    divergences: list[dict[str, Any]]


class DivergenceDetectionStage(PipelineStage):
    """Diff source vs back-translation to surface meaning-level drift.

    Recording divergences is the contract here. Selective re-improvement on flagged
    segments is deferred — it would need a per-segment edit step that's a meaningful
    Phase 5 prompt-iteration concern. The metadata still surfaces what was caught.
    """

    name: ClassVar[str] = "divergence_detection"
    criticality: ClassVar[StageCriticality] = "non_critical"

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings,
        *,
        max_tokens: int = 4096,
    ) -> None:
        self._client = client
        self._settings = settings
        self._max_tokens = max_tokens
        self.model_id = settings.MODEL_DIVERGENCE_DETECTION
        prompt = load_prompt("divergence_detection")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        if state.back_translation is None:
            raise StageDependencyMissing("back_translation")

        rendered = self._prompt.render(
            source_text=state.source.text,
            back_translation=state.back_translation.text,
        )
        result = anthropic_structured(
            self._client,
            model=self._settings.MODEL_DIVERGENCE_DETECTION,
            max_tokens=self._max_tokens,
            prompt=rendered,
            tool_name="submit_divergences",
            tool_description="Submit the list of meaning-level divergences.",
            tool_schema=_DIVERGENCE_TOOL_SCHEMA,
            schema_model=_ToolPayload,
        )

        divergences: list[Divergence] = []
        for raw in result.parsed.divergences:
            try:
                divergences.append(Divergence.model_validate(raw))
            except Exception as e:
                raise StageError(f"divergence parse: {e}") from e

        state.divergences = divergences
        for div in divergences:
            if div.severity == "high":
                state.metadata.warnings.append(
                    f"divergence (high) at {div.segment}: {div.description}"
                )

        cost = estimate_cost(
            self._settings.MODEL_DIVERGENCE_DETECTION,
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        self._record_usage(
            result.usage.input_tokens, result.usage.output_tokens, cost
        )
        return state
