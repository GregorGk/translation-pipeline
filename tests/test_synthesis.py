from __future__ import annotations

from pathlib import Path

import pytest

from tests._fakes import (
    FakeAnthropicClient,
    FakeAnthropicResponse,
    FakeTextBlock,
    FakeUsage,
)
from translation_pipeline.config import Settings
from translation_pipeline.models import (
    Chunk,
    Draft,
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
    SynthesizedTranslation,
    TranslationBrief,
)
from translation_pipeline.stages.base import StageDependencyMissing, StageError
from translation_pipeline.stages.synthesis import SynthesisStage


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_KEY", "x")
    return Settings()


def _state(tmp_path: Path) -> PipelineState:
    p = tmp_path / "d.txt"
    p.write_text("A.\n\nB.")
    pair = LanguagePair(source="PL", target="EN")
    s = PipelineState(
        source=SourceDocument(path=p, text="A.\n\nB.", source_language="PL"),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )
    s.chunks = [Chunk(index=0, text="A."), Chunk(index=1, text="B.")]
    s.draft_a = Draft(source="deepl", chunks=["A-deepl.", "B-deepl."])
    s.draft_b = Draft(source="claude", chunks=["A-claude.", "B-claude."])
    s.brief = TranslationBrief(
        document_type="generic",
        register_level="neutral",
        target_audience="general",
    )
    return s


def _resp(text: str) -> FakeAnthropicResponse:
    return FakeAnthropicResponse(
        content=[FakeTextBlock(text=text)],
        usage=FakeUsage(input_tokens=200, output_tokens=80),
        stop_reason="end_turn",
    )


def test_synthesis_per_chunk_merge(tmp_path: Path, settings: Settings) -> None:
    state = _state(tmp_path)
    fake = FakeAnthropicClient.with_responses(
        [_resp("A-merged."), _resp("B-merged.")]
    )
    stage = SynthesisStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(state)

    assert isinstance(out.synthesis, SynthesizedTranslation)
    assert out.synthesis.text == "A-merged.\n\nB-merged."
    assert out.synthesis.chunk_alignments == ["A-merged.", "B-merged."]
    assert stage.last_input_tokens == 400
    assert stage.last_output_tokens == 160
    assert stage.last_cost_usd > 0


def test_synthesis_prompt_includes_drafts(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    fake = FakeAnthropicClient.with_responses(
        [_resp("ok"), _resp("ok")]
    )
    stage = SynthesisStage(fake, settings)  # type: ignore[arg-type]
    stage.run(state)

    first = fake.messages.calls[0]["messages"][0]["content"]
    assert "A-deepl." in first
    assert "A-claude." in first
    assert "A." in first  # source chunk


def test_synthesis_chunk_count_mismatch(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.draft_a = Draft(source="deepl", chunks=["only one"])
    fake = FakeAnthropicClient.with_responses([])
    stage = SynthesisStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageError, match="chunk count mismatch"):
        stage.run(state)


def test_synthesis_skips_when_drafts_missing(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.draft_a = None
    fake = FakeAnthropicClient.with_responses([])
    stage = SynthesisStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)
