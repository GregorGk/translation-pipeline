from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from translation_pipeline.config import Settings
from translation_pipeline.llm import anthropic_text
from translation_pipeline.models import (
    PipelineState,
    StageCriticality,
    SynthesizedTranslation,
)
from translation_pipeline.pricing import estimate_cost
from translation_pipeline.prompts import load_prompt
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
    StageError,
)
from translation_pipeline.stages.draft_b_claude import _format_brief, _format_glossary

if TYPE_CHECKING:
    import anthropic


class SynthesisStage(PipelineStage):
    """Per-chunk merge of Draft A (DeepL) and Draft B (Claude) into one translation.

    Each chunk's synthesis call sees the source chunk plus the corresponding
    DeepL and Claude chunks. The output is the concatenation; ``chunk_alignments``
    holds the per-chunk merged text in source order.
    """

    name: ClassVar[str] = "synthesis"
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
        self.model_id = settings.MODEL_SYNTHESIS
        prompt = load_prompt("synthesis")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        if not state.chunks:
            raise StageDependencyMissing("chunks")
        if state.draft_a is None:
            raise StageDependencyMissing("draft_a")
        if state.draft_b is None:
            raise StageDependencyMissing("draft_b")
        if state.brief is None:
            raise StageDependencyMissing("brief")

        if not (
            len(state.chunks) == len(state.draft_a.chunks) == len(state.draft_b.chunks)
        ):
            raise StageError(
                "chunk count mismatch: "
                f"chunks={len(state.chunks)}, "
                f"draft_a={len(state.draft_a.chunks)}, "
                f"draft_b={len(state.draft_b.chunks)}"
            )

        brief_str = _format_brief(state.brief)
        glossary_str = _format_glossary(state.brief)

        merged_chunks: list[str] = []
        total = len(state.chunks)
        for i, (chunk, da, db) in enumerate(zip(
            state.chunks, state.draft_a.chunks, state.draft_b.chunks, strict=True
        )):
            self._emit_progress(i + 1, total, "chunk")
            rendered = self._prompt.render(
                source_language=state.language_pair.source,
                target_language=state.language_pair.target,
                brief=brief_str,
                glossary=glossary_str,
                source_text=chunk.text,
                draft_a=da,
                draft_b=db,
            )
            result = anthropic_text(
                self._client,
                model=self._settings.MODEL_SYNTHESIS,
                max_tokens=self._max_tokens,
                prompt=rendered,
            )
            merged_chunks.append(result.text.strip())
            cost = estimate_cost(
                self._settings.MODEL_SYNTHESIS,
                result.usage.input_tokens,
                result.usage.output_tokens,
            )
            self._record_usage(
                result.usage.input_tokens, result.usage.output_tokens, cost
            )

        state.synthesis = SynthesizedTranslation(
            text="\n\n".join(merged_chunks),
            chunk_alignments=merged_chunks,
        )
        return state
