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
    BackTranslation,
    LanguagePair,
    PipelineState,
    RevisedTranslation,
    RunMetadata,
    SourceDocument,
)
from translation_pipeline.stages.back_translation import BackTranslationStage
from translation_pipeline.stages.base import StageDependencyMissing


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
        source=SourceDocument(path=p, text="source PL", source_language="PL"),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )
    s.revised = RevisedTranslation(text="english revised translation")
    return s


def _completion(text: str) -> FakeOpenAICompletion:
    return FakeOpenAICompletion(
        choices=[FakeOpenAIChoice(message=FakeOpenAIMessage(content=text))],
        usage=FakeOpenAIUsage(prompt_tokens=200, completion_tokens=80),
    )


def test_back_translation_writes_back_translation(
    tmp_path: Path, settings: Settings
) -> None:
    fake = FakeOpenAIClient.with_create_responses([_completion("polish back-translated")])
    stage = BackTranslationStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(_state(tmp_path))

    assert isinstance(out.back_translation, BackTranslation)
    assert out.back_translation.text == "polish back-translated"
    assert stage.last_input_tokens == 200
    assert stage.last_output_tokens == 80


def test_back_translation_prompt_includes_translation_and_language_swap(
    tmp_path: Path, settings: Settings
) -> None:
    fake = FakeOpenAIClient.with_create_responses([_completion("x")])
    stage = BackTranslationStage(fake, settings)  # type: ignore[arg-type]
    stage.run(_state(tmp_path))
    rendered = fake.chat.completions.create_calls[0]["messages"][0]["content"]
    assert "english revised translation" in rendered
    assert "PL" in rendered and "EN" in rendered


def test_back_translation_skips_when_no_revision(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.revised = None
    fake = FakeOpenAIClient.with_create_responses([])
    stage = BackTranslationStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)
