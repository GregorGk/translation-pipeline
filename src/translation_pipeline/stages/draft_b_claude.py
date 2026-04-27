from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from translation_pipeline.config import Settings
from translation_pipeline.llm import anthropic_text
from translation_pipeline.models import (
    Draft,
    PipelineState,
    StageCriticality,
    TranslationBrief,
)
from translation_pipeline.pricing import estimate_cost
from translation_pipeline.prompts import load_prompt
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
)

if TYPE_CHECKING:
    import anthropic


def _format_brief(brief: TranslationBrief) -> str:
    return json.dumps(
        {
            "document_type": brief.document_type,
            "register_level": brief.register_level,
            "cultural_notes": brief.cultural_notes,
            "target_audience": brief.target_audience,
            "special_instructions": brief.special_instructions,
        },
        ensure_ascii=False,
        indent=2,
    )


def _format_glossary(brief: TranslationBrief) -> str:
    if not brief.glossary:
        return "(no glossary terms)"
    return "\n".join(
        f"- {g.source_term} → {g.target_term}" + (f"  ({g.note})" if g.note else "")
        for g in brief.glossary
    )


class DraftBStage(PipelineStage):
    """Brief-aware Claude translation, chunk by chunk with prev/next context."""

    name: ClassVar[str] = "draft_b"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings,
        *,
        max_tokens_per_chunk: int = 4096,
    ) -> None:
        self._client = client
        self._settings = settings
        self._max_tokens = max_tokens_per_chunk
        self.model_id = settings.MODEL_DRAFT_B
        prompt = load_prompt("draft_b_claude")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        if not state.chunks:
            raise StageDependencyMissing("chunks")
        if state.brief is None:
            raise StageDependencyMissing("brief")

        brief_str = _format_brief(state.brief)
        glossary_str = _format_glossary(state.brief)

        translated: list[str] = []
        total = len(state.chunks)
        for i, chunk in enumerate(state.chunks):
            self._emit_progress(i + 1, total, "chunk")
            rendered = self._prompt.render(
                source_language=state.language_pair.source,
                target_language=state.language_pair.target,
                brief=brief_str,
                glossary=glossary_str,
                prev_context=chunk.prev_context or "(start of document)",
                source_chunk=chunk.text,
                next_context=chunk.next_context or "(end of document)",
            )
            result = anthropic_text(
                self._client,
                model=self._settings.MODEL_DRAFT_B,
                max_tokens=self._max_tokens,
                prompt=rendered,
            )
            translated.append(result.text.strip())
            cost = estimate_cost(
                self._settings.MODEL_DRAFT_B,
                result.usage.input_tokens,
                result.usage.output_tokens,
            )
            self._record_usage(
                result.usage.input_tokens, result.usage.output_tokens, cost
            )

        state.draft_b = Draft(source="claude", chunks=translated)
        return state
