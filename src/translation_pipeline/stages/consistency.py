from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from translation_pipeline.config import Settings
from translation_pipeline.documents import SENTINEL
from translation_pipeline.llm import anthropic_text
from translation_pipeline.models import (
    FinalOutput,
    PipelineState,
    StageCriticality,
)
from translation_pipeline.pricing import estimate_cost
from translation_pipeline.prompts import load_prompt
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
)
from translation_pipeline.stages.draft_b_claude import _format_brief, _format_glossary

if TYPE_CHECKING:
    import anthropic


class ConsistencyStage(PipelineStage):
    """Final sweep: glossary, names/dates/numbers/citations, formatting artifacts."""

    name: ClassVar[str] = "consistency"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings,
        *,
        max_tokens: int = 32768,
    ) -> None:
        self._client = client
        self._settings = settings
        self._max_tokens = max_tokens
        self.model_id = settings.MODEL_CONSISTENCY
        prompt = load_prompt("consistency")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        if state.revised is None:
            raise StageDependencyMissing("revised")
        if state.brief is None:
            raise StageDependencyMissing("brief")

        rendered = self._prompt.render(
            source_language=state.language_pair.source,
            target_language=state.language_pair.target,
            brief=_format_brief(state.brief),
            glossary=_format_glossary(state.brief),
            source_text=state.source.text,
            translation=state.revised.text,
        )
        result = anthropic_text(
            self._client,
            model=self._settings.MODEL_CONSISTENCY,
            max_tokens=self._max_tokens,
            prompt=rendered,
        )
        cleaned = result.text.strip()

        # If the source was structured (DOCX/PDF), split the cleaned output on the
        # [[BLK]] sentinel to recover per-block translations aligned to the source.
        # If the count doesn't match, surface a warning and leave blocks empty so
        # the CLI falls back to the fresh-document writers.
        blocks: list[str] = []
        expected = len(state.source.blocks)
        if expected:
            parts = [p.strip() for p in cleaned.split(SENTINEL)]
            if len(parts) == expected:
                blocks = parts
            else:
                state.metadata.warnings.append(
                    f"sentinel block count mismatch: source={expected}, "
                    f"output={len(parts)}; falling back to fresh-document writer"
                )
        # Final text always strips sentinels for the .txt and YAML outputs.
        text_for_serializers = cleaned.replace(SENTINEL, "").strip()
        # Collapse the doubled blank lines left after sentinel removal.
        while "\n\n\n" in text_for_serializers:
            text_for_serializers = text_for_serializers.replace("\n\n\n", "\n\n")

        state.final_output = FinalOutput(
            text=text_for_serializers,
            language_pair=state.language_pair,
            brief=state.brief,
            glossary_used=state.brief.glossary,
            warnings=list(state.metadata.warnings),
            blocks=blocks,
        )
        cost = estimate_cost(
            self._settings.MODEL_CONSISTENCY,
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        self._record_usage(
            result.usage.input_tokens, result.usage.output_tokens, cost
        )
        return state
