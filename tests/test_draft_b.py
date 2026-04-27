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
    GlossaryEntry,
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
    TranslationBrief,
)
from translation_pipeline.stages.base import StageDependencyMissing
from translation_pipeline.stages.draft_b_claude import DraftBStage


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_KEY", "x")
    return Settings()


def _state(tmp_path: Path) -> PipelineState:
    p = tmp_path / "d.txt"
    text = "Foo.\n\nBar."
    p.write_text(text)
    pair = LanguagePair(source="PL", target="EN")
    s = PipelineState(
        source=SourceDocument(path=p, text=text, source_language="PL"),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )
    s.chunks = [
        Chunk(index=0, text="Foo.", next_context="Bar."),
        Chunk(index=1, text="Bar.", prev_context="Foo."),
    ]
    s.brief = TranslationBrief(
        document_type="legal",
        register_level="formal",
        glossary=[GlossaryEntry(source_term="Foo", target_term="Foe")],
        cultural_notes=[],
        target_audience="lawyer",
        special_instructions=["preserve verbatim"],
    )
    return s


def _resp(text: str, in_tok: int = 100, out_tok: int = 50) -> FakeAnthropicResponse:
    return FakeAnthropicResponse(
        content=[FakeTextBlock(text=text)],
        usage=FakeUsage(input_tokens=in_tok, output_tokens=out_tok),
        stop_reason="end_turn",
    )


def test_draft_b_translates_each_chunk(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    fake = FakeAnthropicClient.with_responses([_resp("Foe."), _resp("Bar-en.")])
    stage = DraftBStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(state)

    assert out.draft_b is not None
    assert out.draft_b.source == "claude"
    assert out.draft_b.chunks == ["Foe.", "Bar-en."]
    # Two chunks: usage summed.
    assert stage.last_input_tokens == 200
    assert stage.last_output_tokens == 100
    assert stage.last_cost_usd > 0


def test_draft_b_prompt_includes_context_and_brief(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    fake = FakeAnthropicClient.with_responses([_resp("a"), _resp("b")])
    stage = DraftBStage(fake, settings)  # type: ignore[arg-type]
    stage.run(state)

    first = fake.messages.calls[0]["messages"][0]["content"]
    assert "PL" in first and "EN" in first
    assert "Foo." in first  # source chunk
    assert "Bar." in first  # next_context (end-of-doc fallback for chunk 0 next is Bar.)
    assert "Foe" in first  # glossary
    assert "preserve verbatim" in first  # special instructions


def test_draft_b_skips_when_chunks_or_brief_missing(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.brief = None
    fake = FakeAnthropicClient.with_responses([])
    stage = DraftBStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)

    state2 = _state(tmp_path)
    state2.chunks = []
    stage2 = DraftBStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage2.run(state2)
