from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

from translation_pipeline.config import Settings
from translation_pipeline.llm import openai_structured
from translation_pipeline.models import (
    Critique,
    CritiqueIssue,
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
    from openai import OpenAI


# A non-frozen Pydantic schema specifically shaped for OpenAI's strict
# structured-output mode. We coerce to the public ``Critique`` model afterwards.
class _OpenAICritiqueIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    severity: str
    location: str
    description: str
    suggested_fix: str


class _OpenAICritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[_OpenAICritiqueIssue]
    overall_assessment: str


def _to_public(c: _OpenAICritique) -> Critique:
    issues = [
        CritiqueIssue.model_validate(i.model_dump()) for i in c.issues
    ]
    return Critique(issues=issues, overall_assessment=c.overall_assessment)


def _format_brief_for_critique(brief: TranslationBrief) -> str:
    return json.dumps(brief.model_dump(), ensure_ascii=False, indent=2)


class CritiqueStage(PipelineStage):
    """Independent critique by GPT-5; non-critical (skips on persistent failure)."""

    name: ClassVar[str] = "critique"
    criticality: ClassVar[StageCriticality] = "non_critical"

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.model_id = settings.MODEL_CRITIQUE
        prompt = load_prompt("critique")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        if state.synthesis is None:
            raise StageDependencyMissing("synthesis")
        if state.brief is None:
            raise StageDependencyMissing("brief")

        rendered = self._prompt.render(
            source_language=state.language_pair.source,
            target_language=state.language_pair.target,
            brief=_format_brief_for_critique(state.brief),
            source_text=state.source.text,
            translation=state.synthesis.text,
        )
        result = openai_structured(
            self._client,
            model=self._settings.MODEL_CRITIQUE,
            prompt=rendered,
            schema_model=_OpenAICritique,
        )
        try:
            state.critique = _to_public(result.parsed)
        except Exception as e:
            # Critique returned an invalid category / severity literal — non-critical.
            from translation_pipeline.stages.base import StageError

            raise StageError(f"critique parse: {e}") from e

        cost = estimate_cost(
            self._settings.MODEL_CRITIQUE,
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        self._record_usage(
            result.usage.input_tokens, result.usage.output_tokens, cost
        )
        return state
