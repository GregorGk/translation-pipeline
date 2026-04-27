from __future__ import annotations

from pathlib import Path

import pytest

from tests._fakes import (
    FakeOpenAIChoice,
    FakeOpenAIClient,
    FakeOpenAICompletion,
    FakeOpenAIMessage,
    FakeOpenAIUsage,
)
from translation_pipeline.config import Settings
from translation_pipeline.models import (
    Critique,
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
    SynthesizedTranslation,
    TranslationBrief,
)
from translation_pipeline.stages.base import StageDependencyMissing, StageError
from translation_pipeline.stages.critique import CritiqueStage, _OpenAICritique


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_KEY", "x")
    return Settings()


def _state(tmp_path: Path) -> PipelineState:
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
    s.synthesis = SynthesizedTranslation(text="translation", chunk_alignments=["translation"])
    s.brief = TranslationBrief(
        document_type="legal",
        register_level="formal",
        target_audience="lawyer",
    )
    return s


def _completion(parsed: _OpenAICritique) -> FakeOpenAICompletion:
    return FakeOpenAICompletion(
        choices=[FakeOpenAIChoice(message=FakeOpenAIMessage(parsed=parsed))],
        usage=FakeOpenAIUsage(prompt_tokens=300, completion_tokens=120),
    )


def test_critique_parses_to_public_model(
    tmp_path: Path, settings: Settings
) -> None:
    parsed = _OpenAICritique.model_validate(
        {
            "issues": [
                {
                    "category": "terminology",
                    "severity": "medium",
                    "location": "paragraph 1",
                    "description": "wrong rendering of 'wniosek'",
                    "suggested_fix": "use 'motion'",
                }
            ],
            "overall_assessment": "mostly accurate, terminology fix needed",
        }
    )
    fake = FakeOpenAIClient.with_parse_responses([_completion(parsed)])

    stage = CritiqueStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(_state(tmp_path))

    assert isinstance(out.critique, Critique)
    assert len(out.critique.issues) == 1
    issue = out.critique.issues[0]
    assert issue.category == "terminology"
    assert issue.severity == "medium"
    assert "motion" in issue.suggested_fix
    assert stage.last_input_tokens == 300
    assert stage.last_output_tokens == 120
    assert stage.last_cost_usd > 0


def test_critique_invalid_category_raises_stage_error(
    tmp_path: Path, settings: Settings
) -> None:
    parsed = _OpenAICritique.model_validate(
        {
            "issues": [
                {
                    "category": "vibes",  # not in CritiqueCategory literal
                    "severity": "low",
                    "location": "x",
                    "description": "y",
                    "suggested_fix": "z",
                }
            ],
            "overall_assessment": "ok",
        }
    )
    fake = FakeOpenAIClient.with_parse_responses([_completion(parsed)])
    stage = CritiqueStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    with pytest.raises(StageError):
        stage.run(_state(tmp_path))


def test_critique_skips_when_synthesis_missing(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.synthesis = None
    fake = FakeOpenAIClient.with_parse_responses([])
    stage = CritiqueStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)
