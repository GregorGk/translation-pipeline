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
    BackTranslation,
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
)
from translation_pipeline.stages.base import StageDependencyMissing
from translation_pipeline.stages.divergence_detection import DivergenceDetectionStage


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
    s.back_translation = BackTranslation(text="back-translated PL")
    return s


def _resp(divergences: list[dict[str, object]]) -> FakeAnthropicResponse:
    return FakeAnthropicResponse(
        content=[
            FakeToolUseBlock(
                name="submit_divergences",
                input={"divergences": divergences},
            )
        ],
        usage=FakeUsage(input_tokens=300, output_tokens=120),
        stop_reason="tool_use",
    )


def test_divergence_detection_records_divergences(
    tmp_path: Path, settings: Settings
) -> None:
    fake = FakeAnthropicClient.with_responses(
        [
            _resp(
                [
                    {
                        "segment": "p1",
                        "source_text": "X",
                        "back_translated_text": "Y",
                        "severity": "high",
                        "description": "meaning changed",
                    }
                ]
            )
        ]
    )
    stage = DivergenceDetectionStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(_state(tmp_path))

    assert len(out.divergences) == 1
    d = out.divergences[0]
    assert d.severity == "high"
    assert d.description == "meaning changed"
    # High-severity divergence surfaced as run-level warning.
    assert any("divergence (high)" in w for w in out.metadata.warnings)


def test_divergence_detection_empty_list_ok(
    tmp_path: Path, settings: Settings
) -> None:
    fake = FakeAnthropicClient.with_responses([_resp([])])
    stage = DivergenceDetectionStage(fake, settings)  # type: ignore[arg-type]
    out = stage.run(_state(tmp_path))
    assert out.divergences == []
    assert out.metadata.warnings == []


def test_divergence_detection_skips_when_back_translation_missing(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.back_translation = None
    fake = FakeAnthropicClient.with_responses([])
    stage = DivergenceDetectionStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)
