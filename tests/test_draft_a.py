from __future__ import annotations

from pathlib import Path

import pytest

from tests._fakes import FakeDeepLClient, FakeDeepLResult
from translation_pipeline.config import Settings
from translation_pipeline.models import (
    Chunk,
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
)
from translation_pipeline.stages.base import StageDependencyMissing
from translation_pipeline.stages.draft_a_deepl import DraftAStage


@pytest.fixture
def settings_pro(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_PLAN", "pro")
    return Settings()


@pytest.fixture
def settings_free(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_PLAN", "free")
    return Settings()


def _state_with_chunks(
    tmp_path: Path, source_lang: str = "PL", target_lang: str = "EN"
) -> PipelineState:
    p = tmp_path / "doc.txt"
    text = "Pierwszy akapit.\n\nDrugi akapit."
    p.write_text(text)
    pair = LanguagePair(source=source_lang, target=target_lang)  # type: ignore[arg-type]
    s = PipelineState(
        source=SourceDocument(path=p, text=text, source_language=source_lang),  # type: ignore[arg-type]
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )
    s.chunks = [
        Chunk(index=0, text="Pierwszy akapit."),
        Chunk(index=1, text="Drugi akapit."),
    ]
    return s


def test_draft_a_translates_each_chunk(
    tmp_path: Path, settings_pro: Settings
) -> None:
    state = _state_with_chunks(tmp_path, "PL", "EN")
    fake = FakeDeepLClient(
        responses=[
            FakeDeepLResult(text="First paragraph.", billed_characters=16),
            FakeDeepLResult(text="Second paragraph.", billed_characters=13),
        ]
    )
    stage = DraftAStage(fake, settings_pro)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(state)

    assert out.draft_a is not None
    assert out.draft_a.source == "deepl"
    assert out.draft_a.chunks == ["First paragraph.", "Second paragraph."]
    # billed_characters summed; cost > 0 on pro plan.
    assert stage.last_input_tokens == 16 + 13
    assert stage.last_cost_usd > 0


def test_draft_a_language_codes_mapped(
    tmp_path: Path, settings_pro: Settings
) -> None:
    state = _state_with_chunks(tmp_path, "EN", "PT-BR")
    fake = FakeDeepLClient(
        responses=[
            FakeDeepLResult(text="Primeiro.", billed_characters=10),
            FakeDeepLResult(text="Segundo.", billed_characters=10),
        ]
    )
    stage = DraftAStage(fake, settings_pro)  # type: ignore[arg-type]
    stage.run(state)

    # source EN → DeepL "EN", target PT-BR → DeepL "PT-BR".
    call = fake.calls[0]
    assert call["source_lang"] == "EN"
    assert call["target_lang"] == "PT-BR"
    assert call["preserve_formatting"] is True


def test_draft_a_free_plan_zero_cost(
    tmp_path: Path, settings_free: Settings
) -> None:
    state = _state_with_chunks(tmp_path, "PL", "EN")
    fake = FakeDeepLClient(
        responses=[
            FakeDeepLResult(text="x", billed_characters=100),
            FakeDeepLResult(text="y", billed_characters=100),
        ]
    )
    stage = DraftAStage(fake, settings_free)  # type: ignore[arg-type]
    stage.reset_usage()
    stage.run(state)
    assert stage.last_cost_usd == 0.0


def test_draft_a_skips_when_no_chunks(
    tmp_path: Path, settings_pro: Settings
) -> None:
    p = tmp_path / "x.txt"
    p.write_text("text")
    pair = LanguagePair(source="PL", target="EN")
    state = PipelineState(
        source=SourceDocument(path=p, text="text", source_language="PL"),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )
    stage = DraftAStage(FakeDeepLClient(responses=[]), settings_pro)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)
