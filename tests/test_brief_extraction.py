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
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
    TranslationBrief,
)
from translation_pipeline.stages.base import StageError
from translation_pipeline.stages.brief_extraction import BriefExtractionStage


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DEEPL_API_KEY", "test")
    return Settings()


@pytest.fixture
def state(tmp_path: Path) -> PipelineState:
    p = tmp_path / "doc.txt"
    text = "Court ruling on property rights, paragraph one. Paragraph two."
    p.write_text(text)
    pair = LanguagePair(source="PL", target="EN")
    return PipelineState(
        source=SourceDocument(path=p, text=text, source_language="PL"),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )


def _brief_response(payload: dict[str, object]) -> FakeAnthropicResponse:
    return FakeAnthropicResponse(
        content=[FakeToolUseBlock(name="submit_brief", input=payload)],
        usage=FakeUsage(input_tokens=1234, output_tokens=567),
        stop_reason="tool_use",
    )


def test_brief_extraction_parses_tool_use(
    state: PipelineState, settings: Settings
) -> None:
    payload: dict[str, object] = {
        "document_type": "Polish criminal-law motion",
        "register_level": "formal-legal",
        "glossary": [
            {"source_term": "wniosek", "target_term": "motion", "note": None},
            {
                "source_term": "k.k.",
                "target_term": "Penal Code",
                "note": "Polish 'kodeks karny'",
            },
        ],
        "cultural_notes": ["Polish criminal procedure context"],
        "target_audience": "English-speaking attorney reviewing case",
        "special_instructions": [
            "preserve names, dates, numbers, citations, and legal references verbatim"
        ],
    }
    fake = FakeAnthropicClient.with_responses([_brief_response(payload)])

    stage = BriefExtractionStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(state)

    assert isinstance(out.brief, TranslationBrief)
    assert out.brief.document_type == "Polish criminal-law motion"
    assert out.brief.register_level == "formal-legal"
    assert len(out.brief.glossary) == 2
    assert out.brief.glossary[1].source_term == "k.k."
    assert "verbatim" in out.brief.special_instructions[0]

    # Token usage and cost recorded on the stage instance.
    assert stage.last_input_tokens == 1234
    assert stage.last_output_tokens == 567
    assert stage.last_cost_usd > 0

    # Prompt was rendered with the right variables.
    call = fake.messages.calls[0]
    rendered = call["messages"][0]["content"]
    assert "PL" in rendered and "EN" in rendered
    assert "Court ruling" in rendered
    # Tool was forced.
    assert call["tool_choice"]["name"] == "submit_brief"


def test_brief_extraction_raises_when_tool_skipped(
    state: PipelineState, settings: Settings
) -> None:
    from tests._fakes import FakeTextBlock

    fake = FakeAnthropicClient.with_responses(
        [
            FakeAnthropicResponse(
                content=[FakeTextBlock(text="here is the brief in prose")],
                usage=FakeUsage(input_tokens=10, output_tokens=10),
                stop_reason="end_turn",
            )
        ]
    )
    stage = BriefExtractionStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    with pytest.raises(StageError):
        stage.run(state)


def test_brief_extraction_raises_on_invalid_payload(
    state: PipelineState, settings: Settings
) -> None:
    # Missing required field ``document_type``.
    bad: dict[str, object] = {
        "register_level": "neutral",
        "target_audience": "general",
    }
    fake = FakeAnthropicClient.with_responses([_brief_response(bad)])
    stage = BriefExtractionStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    with pytest.raises(StageError):
        stage.run(state)


def test_brief_extraction_records_prompt_hash(
    state: PipelineState, settings: Settings
) -> None:
    fake = FakeAnthropicClient.with_responses(
        [
            _brief_response(
                {
                    "document_type": "x",
                    "register_level": "y",
                    "target_audience": "z",
                }
            )
        ]
    )
    stage = BriefExtractionStage(fake, settings)  # type: ignore[arg-type]
    assert stage.prompt_hash is not None
    assert len(stage.prompt_hash) == 64  # sha256 hex
    assert stage.model_id == settings.MODEL_BRIEF_EXTRACTION
