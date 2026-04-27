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
    FinalOutput,
    LanguagePair,
    PipelineState,
    RevisedTranslation,
    RunMetadata,
    SourceDocument,
    TranslationBrief,
)
from translation_pipeline.stages.base import StageDependencyMissing
from translation_pipeline.stages.consistency import ConsistencyStage


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPL_API_KEY", "x")
    return Settings()


def _state(tmp_path: Path) -> PipelineState:
    p = tmp_path / "d.txt"
    p.write_text("source PL")
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
    s.brief = TranslationBrief(
        document_type="legal", register_level="formal", target_audience="lawyer"
    )
    s.revised = RevisedTranslation(text="revised translation")
    s.metadata.warnings.append("a prior warning")
    return s


def _resp(text: str) -> FakeAnthropicResponse:
    return FakeAnthropicResponse(
        content=[FakeTextBlock(text=text)],
        usage=FakeUsage(input_tokens=400, output_tokens=200),
        stop_reason="end_turn",
    )


def test_consistency_writes_final_output(
    tmp_path: Path, settings: Settings
) -> None:
    fake = FakeAnthropicClient.with_responses([_resp("final cleaned translation")])
    stage = ConsistencyStage(fake, settings)  # type: ignore[arg-type]
    stage.reset_usage()
    out = stage.run(_state(tmp_path))

    assert isinstance(out.final_output, FinalOutput)
    assert out.final_output.text == "final cleaned translation"
    assert out.final_output.language_pair.target == "EN"
    # Existing warnings carried into the final output.
    assert "a prior warning" in out.final_output.warnings


def test_consistency_skips_when_revised_missing(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state(tmp_path)
    state.revised = None
    fake = FakeAnthropicClient.with_responses([])
    stage = ConsistencyStage(fake, settings)  # type: ignore[arg-type]
    with pytest.raises(StageDependencyMissing):
        stage.run(state)


# ---- Sentinel block splitting (format-preserving path) ----------------------


def _state_with_blocks(tmp_path: Path, blocks: tuple[str, ...]) -> PipelineState:
    """Mirror ``_state`` but with a structured source carrying block boundaries."""
    p = tmp_path / "d.txt"
    p.write_text("source")
    pair = LanguagePair(source="PL", target="EN")
    state = PipelineState(
        source=SourceDocument(
            path=p, text="source", source_language="PL", blocks=blocks
        ),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="r",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )
    state.brief = TranslationBrief(
        document_type="legal", register_level="formal", target_audience="lawyer"
    )
    state.revised = RevisedTranslation(text="x")
    return state


def test_consistency_splits_sentinels_into_blocks(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state_with_blocks(tmp_path, blocks=("a", "b", "c"))
    fake = FakeAnthropicClient.with_responses(
        [_resp("Block A.\n\n[[BLK]]\n\nBlock B.\n\n[[BLK]]\n\nBlock C.")]
    )
    stage = ConsistencyStage(fake, settings)  # type: ignore[arg-type]
    out = stage.run(state)

    assert out.final_output is not None
    assert out.final_output.blocks == ["Block A.", "Block B.", "Block C."]
    # Sentinel removed from .text used by the .txt serializer.
    assert "[[BLK]]" not in out.final_output.text
    assert "Block A." in out.final_output.text
    assert "Block C." in out.final_output.text


def test_consistency_block_count_mismatch_warns_and_empties_blocks(
    tmp_path: Path, settings: Settings
) -> None:
    state = _state_with_blocks(tmp_path, blocks=("a", "b", "c"))
    # Output drops a sentinel — only 2 blocks where source had 3.
    fake = FakeAnthropicClient.with_responses(
        [_resp("Block A.\n\n[[BLK]]\n\nBlock BC merged.")]
    )
    stage = ConsistencyStage(fake, settings)  # type: ignore[arg-type]
    out = stage.run(state)

    assert out.final_output is not None
    assert out.final_output.blocks == []  # alignment failed → empty triggers fallback
    assert any(
        "sentinel block count mismatch" in w for w in out.metadata.warnings
    )
    assert any(
        "source=3" in w and "output=2" in w for w in out.metadata.warnings
    )


def test_consistency_unstructured_source_yields_empty_blocks(
    tmp_path: Path, settings: Settings
) -> None:
    """When source.blocks is empty (TXT input), final_output.blocks stays empty."""
    state = _state(tmp_path)  # no blocks
    fake = FakeAnthropicClient.with_responses([_resp("translated text")])
    stage = ConsistencyStage(fake, settings)  # type: ignore[arg-type]
    out = stage.run(state)
    assert out.final_output is not None
    assert out.final_output.blocks == []
    assert out.final_output.text == "translated text"
