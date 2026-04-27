from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from translation_pipeline.models import (
    BackTranslation,
    Critique,
    Draft,
    FinalOutput,
    PipelineState,
    RevisedTranslation,
    SourceDocument,
    StageCriticality,
    SynthesizedTranslation,
    TranslationBrief,
)
from translation_pipeline.pipeline import Pipeline, PipelineAbort
from translation_pipeline.stages.base import PipelineStage, StageError
from translation_pipeline.stages.stubs import (
    CritiqueStub,
    SynthesisStub,
    default_stub_pipeline_stages,
)

# Fast retry policy for tests — keeps the suite snappy while still exercising
# the 3-attempt-with-backoff loop.
FAST_RETRY = {
    "retry_attempts": 3,
    "retry_wait_min": 0.001,
    "retry_wait_max": 0.005,
    "retry_wait_multiplier": 0.001,
}


@pytest.fixture
def source(tmp_path: Path) -> SourceDocument:
    p = tmp_path / "input.txt"
    p.write_text("Hello world. This is a test.")
    return SourceDocument(path=p, text=p.read_text(), source_language="EN")


def _expected_stage_order() -> list[str]:
    return [
        "brief_extraction",
        "chunking",
        "draft_a",
        "draft_b",
        "synthesis",
        "critique",
        "improvement",
        "back_translation",
        "divergence_detection",
        "consistency",
    ]


def test_full_stub_pipeline_runs_in_order(source: SourceDocument) -> None:
    pipeline = Pipeline(default_stub_pipeline_stages(), **FAST_RETRY)
    state = pipeline.run(source, target_language="PL")

    # Each output has its expected concrete type.
    assert isinstance(state.brief, TranslationBrief)
    assert isinstance(state.draft_a, Draft) and state.draft_a.source == "deepl"
    assert isinstance(state.draft_b, Draft) and state.draft_b.source == "claude"
    assert isinstance(state.synthesis, SynthesizedTranslation)
    assert isinstance(state.critique, Critique)
    assert isinstance(state.revised, RevisedTranslation)
    assert isinstance(state.back_translation, BackTranslation)
    assert isinstance(state.final_output, FinalOutput)
    assert state.final_output.language_pair.target == "PL"

    # Metadata recorded a successful run for every stage in the right order.
    names = [s.name for s in state.metadata.stages]
    assert names == _expected_stage_order()
    assert all(s.status == "ok" for s in state.metadata.stages)
    assert state.metadata.total_duration_s > 0
    assert state.metadata.warnings == []


class _AlwaysFailingCritique(CritiqueStub):
    """Non-critical stage that always raises StageError."""

    name: ClassVar[str] = "critique"
    criticality: ClassVar[StageCriticality] = "non_critical"

    def run(self, state: PipelineState) -> PipelineState:
        raise StageError("simulated critique outage")


def test_non_critical_failure_skips_with_warning(source: SourceDocument) -> None:
    stages: list[PipelineStage] = list(default_stub_pipeline_stages())
    # Replace CritiqueStub (index 5) with the failing variant.
    stages[5] = _AlwaysFailingCritique()
    pipeline = Pipeline(stages, **FAST_RETRY)

    state = pipeline.run(source, target_language="PL")

    # Pipeline still produced a final output.
    assert isinstance(state.final_output, FinalOutput)
    # Critique stage was skipped, downstream stages still ran.
    critique = next(s for s in state.metadata.stages if s.name == "critique")
    assert critique.status == "skipped"
    assert critique.attempts == 3
    assert "simulated critique outage" in (critique.error or "")
    # Warning surfaced into run-level metadata.
    assert any("critique" in w for w in state.metadata.warnings)
    # Improvement still ran (passes synthesis through when critique missing).
    assert state.revised is not None
    assert state.critique is None


class _AlwaysFailingSynthesis(SynthesisStub):
    """Critical stage that always raises StageError."""

    name: ClassVar[str] = "synthesis"
    criticality: ClassVar[StageCriticality] = "critical"

    def run(self, state: PipelineState) -> PipelineState:
        raise StageError("simulated synthesis outage")


def test_critical_failure_aborts(source: SourceDocument) -> None:
    stages: list[PipelineStage] = list(default_stub_pipeline_stages())
    stages[4] = _AlwaysFailingSynthesis()
    pipeline = Pipeline(stages, **FAST_RETRY)

    with pytest.raises(PipelineAbort) as exc:
        pipeline.run(source, target_language="DE")

    assert "synthesis" in str(exc.value)
    assert "simulated synthesis outage" in str(exc.value)


class _FailsThenSucceeds(SynthesisStub):
    """Fails the first ``fail_count`` invocations, then succeeds."""

    name: ClassVar[str] = "synthesis"
    criticality: ClassVar[StageCriticality] = "critical"

    def __init__(self, fail_count: int) -> None:
        super().__init__()
        self._fail_count = fail_count
        self.calls = 0

    def run(self, state: PipelineState) -> PipelineState:
        self.calls += 1
        if self.calls <= self._fail_count:
            raise StageError(f"transient #{self.calls}")
        return super().run(state)


def test_retry_recovers_then_succeeds(source: SourceDocument) -> None:
    stages: list[PipelineStage] = list(default_stub_pipeline_stages())
    flaky = _FailsThenSucceeds(fail_count=2)
    stages[4] = flaky
    pipeline = Pipeline(stages, **FAST_RETRY)

    state = pipeline.run(source, target_language="FR")

    assert flaky.calls == 3
    synth = next(s for s in state.metadata.stages if s.name == "synthesis")
    assert synth.status == "ok"
    assert synth.attempts == 3


def test_dependency_missing_propagates_skip(source: SourceDocument) -> None:
    """If a non-critical stage is skipped, downstream stages that depend on its output
    must skip too — they raise StageDependencyMissing, which the pipeline records as a
    skip rather than aborting.
    """

    class _SkippingBackTranslation(PipelineStage):
        name: ClassVar[str] = "back_translation"
        criticality: ClassVar[StageCriticality] = "non_critical"

        def run(self, state: PipelineState) -> PipelineState:
            raise StageError("simulated outage")

    stages: list[PipelineStage] = list(default_stub_pipeline_stages())
    stages[7] = _SkippingBackTranslation()
    pipeline = Pipeline(stages, **FAST_RETRY)

    state = pipeline.run(source, target_language="PL")

    # back_translation skipped, divergence_detection then sees missing dependency and skips.
    back = next(s for s in state.metadata.stages if s.name == "back_translation")
    div = next(s for s in state.metadata.stages if s.name == "divergence_detection")
    assert back.status == "skipped"
    assert div.status == "skipped"
    assert "dependency missing" in (div.error or "")
    # Pipeline still completes through consistency.
    assert isinstance(state.final_output, FinalOutput)


def test_metadata_carries_run_identification(source: SourceDocument) -> None:
    pipeline = Pipeline(default_stub_pipeline_stages(), **FAST_RETRY)
    state = pipeline.run(source, target_language="UK")

    md = state.metadata
    assert md.run_id  # non-empty UUID
    assert md.source_path == source.path
    assert md.language_pair.source == "EN"
    assert md.language_pair.target == "UK"
    assert md.pipeline_version  # non-empty
