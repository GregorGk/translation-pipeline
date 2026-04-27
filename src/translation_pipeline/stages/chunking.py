from __future__ import annotations

from typing import ClassVar

from translation_pipeline.chunking import chunk_text
from translation_pipeline.models import Chunk, PipelineState, StageCriticality
from translation_pipeline.stages.base import PipelineStage


class ChunkingStage(PipelineStage):
    """Local, deterministic paragraph-aware splitter.

    Reads ``state.source.text``, writes ``state.chunks``. No external API call.
    """

    name: ClassVar[str] = "chunking"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self, *, target_tokens: int = 1500, overlap_tokens: int = 200) -> None:
        self._target = target_tokens
        self._overlap = overlap_tokens

    def run(self, state: PipelineState) -> PipelineState:
        triples = chunk_text(
            state.source.text,
            target_tokens=self._target,
            overlap_tokens=self._overlap,
        )
        state.chunks = [
            Chunk(index=i, text=t, prev_context=p, next_context=n)
            for i, (t, p, n) in enumerate(triples)
        ]
        return state
