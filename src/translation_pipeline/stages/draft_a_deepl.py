from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from translation_pipeline.config import Settings
from translation_pipeline.documents import SENTINEL
from translation_pipeline.models import (
    Draft,
    LanguageCode,
    PipelineState,
    StageCriticality,
)
from translation_pipeline.pricing import deepl_character_cost
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
    StageError,
)

if TYPE_CHECKING:
    import deepl


# Internal → DeepL source language codes (DeepL's source codes are region-less).
_DEEPL_SOURCE: dict[LanguageCode, str] = {
    "EN": "EN",
    "PT-BR": "PT",
    "PL": "PL",
    "FR": "FR",
    "DE": "DE",
    "RU": "RU",
    "UK": "UK",
    "EL": "EL",
}

# Internal → DeepL target language codes (region-aware where required).
_DEEPL_TARGET: dict[LanguageCode, str] = {
    "EN": "EN-US",
    "PT-BR": "PT-BR",
    "PL": "PL",
    "FR": "FR",
    "DE": "DE",
    "RU": "RU",
    "UK": "UK",
    "EL": "EL",
}


class DraftAStage(PipelineStage):
    """Idiomatic baseline translation via DeepL, chunk by chunk."""

    name: ClassVar[str] = "draft_a"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self, client: deepl.Translator, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.model_id = "deepl"

    def run(self, state: PipelineState) -> PipelineState:
        if not state.chunks:
            raise StageDependencyMissing("chunks")

        try:
            import deepl as _deepl
        except ImportError as e:  # pragma: no cover
            raise StageError(f"deepl import error: {e}") from e

        source = _DEEPL_SOURCE[state.language_pair.source]
        target = _DEEPL_TARGET[state.language_pair.target]

        translated_chunks: list[str] = []
        billed_chars_total = 0
        total = len(state.chunks)
        for i, chunk in enumerate(state.chunks):
            self._emit_progress(i + 1, total, "chunk")
            translated, billed = self._translate_with_sentinels(
                chunk.text, source=source, target=target, deepl=_deepl
            )
            translated_chunks.append(translated)
            billed_chars_total += billed

        state.draft_a = Draft(source="deepl", chunks=translated_chunks)
        cost = deepl_character_cost(billed_chars_total, self._settings.DEEPL_API_PLAN)
        # We don't have token counts for DeepL; record characters as input_tokens
        # for visibility in metadata (and zero output_tokens), with the cost computed
        # from billed characters.
        self._record_usage(billed_chars_total, 0, cost)
        return state

    def _translate_with_sentinels(
        self, chunk_text: str, *, source: str, target: str, deepl: object
    ) -> tuple[str, int]:
        """Translate a chunk, splitting on [[BLK]] sentinels first.

        DeepL would otherwise translate or mangle the literal sentinel. We translate
        each segment separately and rejoin with the sentinel verbatim — DeepL never
        sees it, so it survives intact through DraftA.
        """
        if SENTINEL not in chunk_text:
            return self._translate_one(chunk_text, source=source, target=target, deepl=deepl)

        segments = chunk_text.split(SENTINEL)
        translated_segments: list[str] = []
        billed_total = 0
        for seg in segments:
            if not seg.strip():
                translated_segments.append(seg)
                continue
            translated, billed = self._translate_one(
                seg, source=source, target=target, deepl=deepl
            )
            translated_segments.append(translated)
            billed_total += billed
        return SENTINEL.join(translated_segments), billed_total

    def _translate_one(
        self, text: str, *, source: str, target: str, deepl: object
    ) -> tuple[str, int]:
        try:
            result = self._client.translate_text(
                text,
                source_lang=source,
                target_lang=target,
                preserve_formatting=True,
            )
        except deepl.DeepLException as e:  # type: ignore[attr-defined]
            raise StageError(f"deepl error: {e}") from e

        if isinstance(result, list):
            translated = "".join(r.text for r in result)
            billed = sum(getattr(r, "billed_characters", 0) or 0 for r in result)
        else:
            translated = result.text
            billed = getattr(result, "billed_characters", 0) or 0
        return translated, billed
