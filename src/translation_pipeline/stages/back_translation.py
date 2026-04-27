from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from translation_pipeline.config import Settings
from translation_pipeline.llm import openai_text
from translation_pipeline.models import (
    BackTranslation,
    PipelineState,
    StageCriticality,
)
from translation_pipeline.pricing import estimate_cost
from translation_pipeline.prompts import load_prompt
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
)

if TYPE_CHECKING:
    from openai import OpenAI


class BackTranslationStage(PipelineStage):
    """Back-translate the revision into the source language via GPT-5.

    Independence from Claude is the whole point — we use a different model family
    so the comparison surfaces meaning drift Claude wouldn't catch reviewing its
    own output.
    """

    name: ClassVar[str] = "back_translation"
    criticality: ClassVar[StageCriticality] = "non_critical"

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.model_id = settings.MODEL_BACK_TRANSLATION
        prompt = load_prompt("back_translation")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        if state.revised is None:
            raise StageDependencyMissing("revised")

        rendered = self._prompt.render(
            source_language=state.language_pair.source,
            target_language=state.language_pair.target,
            translation=state.revised.text,
        )
        result = openai_text(
            self._client,
            model=self._settings.MODEL_BACK_TRANSLATION,
            prompt=rendered,
        )
        state.back_translation = BackTranslation(text=result.text.strip())
        cost = estimate_cost(
            self._settings.MODEL_BACK_TRANSLATION,
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        self._record_usage(
            result.usage.input_tokens, result.usage.output_tokens, cost
        )
        return state
