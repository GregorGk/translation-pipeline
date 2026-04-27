from __future__ import annotations

from pathlib import Path

import pytest

from tests._fakes import (
    FakeAnthropicClient,
    FakeAnthropicResponse,
    FakeToolUseBlock,
    FakeUsage,
)
from translation_pipeline.config import Settings
from translation_pipeline.models import (
    Critique,
    CritiqueIssue,
    LanguagePair,
    PipelineState,
    RevisedTranslation,
    RunMetadata,
    SourceDocument,
    SynthesizedTranslation,
    TranslationBrief,
)
from translation_pipeline.stages.base import StageDependencyMissing
from translation_pipeline.stages.improvement import ImprovementStage


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_KEY", "x")
    return Settings()


def _state(tmp_path: Path, with_critique: bool = True) -> PipelineState:
    p = tmp_path / "d.txt"
    p.write_text("source")
    pair = LanguagePair(source="PL", target="EN")
    s = PipelineState(
        source=SourceDocument(path=p, text="source", source_language="PL"),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )
    s.synthesis = SynthesizedTranslation(
        text="synthesized translation", chunk_alignments=["synthesized translation"]
    )
    s.brief = TranslationBrief(
        document_type="legal", register_level="formal", target_audience="lawyer"
    )
    if with_critique:
        s.critique = Critique(
            issues=[
                CritiqueIssue(
                    category="terminology",
                    severity="medium",
                    location="p1",
                    description="bad term",
                    suggested_fix="use X",
                ),
                CritiqueIssue(
                    category="fluency",
                    severity="low",
                    location="p2",
                    description="awkward",
                    suggested_fix="rephrase",
                ),
            ],
            overall_assessment="ok",
        )
    return s


def _resp(text: str, decisions: list[dict[str, object]]) -> FakeAnthropicResponse:
    return FakeAnthropicResponse(
        content=[
            FakeToolUseBlock(
                name="submit_revision",
                input={"text": text, "decisions": decisions},
            )
        ],
        usage=FakeUsage(input_tokens=500, output_tokens=200),
        stop_reason="tool_use",
    )


def test_improvement_applies_accepted_rejects_others(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    fake = FakeAnthropicClient.with_responses(
        [
            _resp(
                "improved translation",
                [
                    {"issue_index": 0, "accepted": True, "reasoning": "good fix"},
                    {"issue_index": 1, "accepted": False, "reasoning": "regresses"},
                ],
            )
        ]
    )
    stage = ImprovementStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(state)

    assert isinstance(out.revised, RevisedTranslation)
    assert out.revised.text == "improved translation"
    assert len(out.revised.issues_addressed) == 1
    assert out.revised.issues_addressed[0].issue.category == "terminology"
    assert out.revised.issues_addressed[0].accepted is True
    assert len(out.revised.issues_rejected_with_reason) == 1
    assert out.revised.issues_rejected_with_reason[0].issue.category == "fluency"
    assert stage.last_input_tokens == 500


def test_improvement_passthrough_when_no_critique(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path, with_critique=False)
    fake = FakeAnthropicClient.with_responses([])  # API not called.
    stage = ImprovementStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(state)
    assert out.revised is not None
    assert out.revised.text == "synthesized translation"
    assert stage.last_input_tokens == 0
    # No call was made.
    assert fake.messages.calls == []


def test_improvement_drops_out_of_range_decisions(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    fake = FakeAnthropicClient.with_responses(
        [
            _resp(
                "improved",
                [
                    {"issue_index": 0, "accepted": True, "reasoning": "ok"},
                    {"issue_index": 99, "accepted": True, "reasoning": "phantom"},
                ],
            )
        ]
    )
    stage = ImprovementStage(fake, settings)  # type: ignore[arg-type]
    out = stage.run(state)
    assert out.revised is not None
    assert len(out.revised.issues_addressed) == 1


def test_improvement_skips_when_synthesis_missing(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.synthesis = None
    fake = FakeAnthropicClient.with_responses([])
    stage = ImprovementStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)
