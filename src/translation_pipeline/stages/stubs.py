"""Stub implementations of every pipeline stage.

These produce well-typed mock data without touching any external API. Used to validate
orchestration end-to-end before the real stages land in Phase 2.
"""

from __future__ import annotations

from typing import ClassVar

from translation_pipeline.models import (
    BackTranslation,
    Chunk,
    Critique,
    CritiqueIssue,
    Divergence,
    Draft,
    FinalOutput,
    GlossaryEntry,
    IssueDecision,
    PipelineState,
    RevisedTranslation,
    StageCriticality,
    SynthesizedTranslation,
    TranslationBrief,
)
from translation_pipeline.stages.base import PipelineStage, StageDependencyMissing


def _require(value: object, dep_name: str) -> None:
    if value is None:
        raise StageDependencyMissing(dep_name)


class BriefExtractionStub(PipelineStage):
    name: ClassVar[str] = "brief_extraction"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self) -> None:
        self.model_id = "stub:claude"

    def run(self, state: PipelineState) -> PipelineState:
        state.brief = TranslationBrief(
            document_type="generic",
            register_level="neutral",
            glossary=[GlossaryEntry(source_term="foo", target_term="bar")],
            cultural_notes=[],
            target_audience="general",
            special_instructions=[],
        )
        return state


class ChunkingStub(PipelineStage):
    name: ClassVar[str] = "chunking"
    criticality: ClassVar[StageCriticality] = "critical"

    def run(self, state: PipelineState) -> PipelineState:
        state.chunks = [Chunk(index=0, text=state.source.text)]
        return state


class DraftAStub(PipelineStage):
    name: ClassVar[str] = "draft_a"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self) -> None:
        self.model_id = "stub:deepl"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.chunks, "chunks")
        state.draft_a = Draft(
            source="deepl",
            chunks=[f"[deepl] {c.text}" for c in state.chunks],
        )
        return state


class DraftBStub(PipelineStage):
    name: ClassVar[str] = "draft_b"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self) -> None:
        self.model_id = "stub:claude"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.chunks, "chunks")
        _require(state.brief, "brief")
        state.draft_b = Draft(
            source="claude",
            chunks=[f"[claude] {c.text}" for c in state.chunks],
        )
        return state


class SynthesisStub(PipelineStage):
    name: ClassVar[str] = "synthesis"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self) -> None:
        self.model_id = "stub:claude"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.draft_a, "draft_a")
        _require(state.draft_b, "draft_b")
        merged = [f"[merged] {a}" for a in (state.draft_a.chunks if state.draft_a else [])]
        state.synthesis = SynthesizedTranslation(
            text="\n\n".join(merged),
            chunk_alignments=merged,
        )
        return state


class CritiqueStub(PipelineStage):
    name: ClassVar[str] = "critique"
    criticality: ClassVar[StageCriticality] = "non_critical"

    def __init__(self) -> None:
        self.model_id = "stub:gpt-5"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.synthesis, "synthesis")
        state.critique = Critique(
            issues=[
                CritiqueIssue(
                    category="terminology",
                    severity="low",
                    location="chunk 0",
                    description="stub issue",
                    suggested_fix="stub fix",
                )
            ],
            overall_assessment="stub assessment",
        )
        return state


class ImprovementStub(PipelineStage):
    name: ClassVar[str] = "improvement"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self) -> None:
        self.model_id = "stub:claude"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.synthesis, "synthesis")
        # Critique is non-critical; if it was skipped, pass synthesis through unchanged.
        if state.critique is None:
            assert state.synthesis is not None
            state.revised = RevisedTranslation(text=state.synthesis.text)
            return state

        decisions = [
            IssueDecision(issue=i, accepted=True, reasoning="stub")
            for i in state.critique.issues
        ]
        assert state.synthesis is not None
        state.revised = RevisedTranslation(
            text=state.synthesis.text + " [revised]",
            issues_addressed=decisions,
        )
        return state


class BackTranslationStub(PipelineStage):
    name: ClassVar[str] = "back_translation"
    criticality: ClassVar[StageCriticality] = "non_critical"

    def __init__(self) -> None:
        self.model_id = "stub:gpt-5"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.revised, "revised")
        assert state.revised is not None
        state.back_translation = BackTranslation(
            text=f"[back] {state.revised.text}"
        )
        return state


class DivergenceDetectionStub(PipelineStage):
    name: ClassVar[str] = "divergence_detection"
    criticality: ClassVar[StageCriticality] = "non_critical"

    def __init__(self) -> None:
        self.model_id = "stub:claude"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.back_translation, "back_translation")
        state.divergences = [
            Divergence(
                segment="0",
                source_text=state.source.text,
                back_translated_text=(
                    state.back_translation.text if state.back_translation else ""
                ),
                severity="low",
                description="stub divergence",
            )
        ]
        return state


class ConsistencyStub(PipelineStage):
    name: ClassVar[str] = "consistency"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self) -> None:
        self.model_id = "stub:claude"

    def run(self, state: PipelineState) -> PipelineState:
        _require(state.revised, "revised")
        _require(state.brief, "brief")
        assert state.revised is not None
        assert state.brief is not None
        state.final_output = FinalOutput(
            text=state.revised.text,
            language_pair=state.language_pair,
            brief=state.brief,
            glossary_used=state.brief.glossary,
            warnings=list(state.metadata.warnings),
        )
        return state


def default_stub_pipeline_stages() -> list[PipelineStage]:
    return [
        BriefExtractionStub(),
        ChunkingStub(),
        DraftAStub(),
        DraftBStub(),
        SynthesisStub(),
        CritiqueStub(),
        ImprovementStub(),
        BackTranslationStub(),
        DivergenceDetectionStub(),
        ConsistencyStub(),
    ]
