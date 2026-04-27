from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from translation_pipeline.config import Settings
from translation_pipeline.llm import anthropic_structured
from translation_pipeline.models import (
    CritiqueIssue,
    IssueDecision,
    PipelineState,
    RevisedTranslation,
    StageCriticality,
)
from translation_pipeline.pricing import estimate_cost
from translation_pipeline.prompts import load_prompt
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
)
from translation_pipeline.stages.critique import _format_brief_for_critique

if TYPE_CHECKING:
    import anthropic


# Schema submitted to Claude (loose strings; we coerce on receipt).
_REVISION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_index": {"type": "integer"},
                    "accepted": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
                "required": ["issue_index", "accepted", "reasoning"],
            },
        },
    },
    "required": ["text", "decisions"],
}


class _ToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_index: int
    accepted: bool
    reasoning: str


class _ToolPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    decisions: list[_ToolDecision]


class ImprovementStage(PipelineStage):
    """Apply (or reject) critique issues to produce a revised translation.

    If critique was skipped (state.critique is None), the prior synthesis is passed
    through unchanged as the revision. The stage stays declared "critical" so the
    pipeline aborts if the revision call itself fails persistently.
    """

    name: ClassVar[str] = "improvement"
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
        self.model_id = settings.MODEL_IMPROVEMENT
        prompt = load_prompt("improvement")
        self._prompt = prompt
        self.prompt_hash = prompt.sha256

    def run(self, state: PipelineState) -> PipelineState:
        if state.synthesis is None:
            raise StageDependencyMissing("synthesis")
        if state.brief is None:
            raise StageDependencyMissing("brief")

        # Critique skipped → pass synthesis through.
        if state.critique is None:
            state.revised = RevisedTranslation(text=state.synthesis.text)
            return state

        rendered = self._prompt.render(
            source_language=state.language_pair.source,
            target_language=state.language_pair.target,
            brief=_format_brief_for_critique(state.brief),
            source_text=state.source.text,
            translation=state.synthesis.text,
            critique=_serialize_critique(state.critique.issues),
        )
        result = anthropic_structured(
            self._client,
            model=self._settings.MODEL_IMPROVEMENT,
            max_tokens=self._max_tokens,
            prompt=rendered,
            tool_name="submit_revision",
            tool_description="Submit the revised translation and per-issue decisions.",
            tool_schema=_REVISION_TOOL_SCHEMA,
            schema_model=_ToolPayload,
        )

        addressed: list[IssueDecision] = []
        rejected: list[IssueDecision] = []
        issues = state.critique.issues
        for d in result.parsed.decisions:
            if not (0 <= d.issue_index < len(issues)):
                continue  # Drop out-of-range references rather than abort.
            issue = issues[d.issue_index]
            decision = IssueDecision(
                issue=issue, accepted=d.accepted, reasoning=d.reasoning
            )
            (addressed if d.accepted else rejected).append(decision)

        state.revised = RevisedTranslation(
            text=result.parsed.text,
            issues_addressed=addressed,
            issues_rejected_with_reason=rejected,
        )
        cost = estimate_cost(
            self._settings.MODEL_IMPROVEMENT,
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        self._record_usage(
            result.usage.input_tokens, result.usage.output_tokens, cost
        )
        return state


def _serialize_critique(issues: list[CritiqueIssue]) -> str:
    return json.dumps(
        [
            {
                "index": i,
                "category": issue.category,
                "severity": issue.severity,
                "location": issue.location,
                "description": issue.description,
                "suggested_fix": issue.suggested_fix,
            }
            for i, issue in enumerate(issues)
        ],
        ensure_ascii=False,
        indent=2,
    )
