from __future__ import annotations

from pathlib import Path

import pytest

from translation_pipeline.chunking import chunk_text
from translation_pipeline.models import (
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
)
from translation_pipeline.stages.chunking import ChunkingStage


def test_single_paragraph_one_chunk() -> None:
    triples = chunk_text("Just one paragraph.", target_tokens=1500, overlap_tokens=200)
    assert len(triples) == 1
    text, prev, next_ = triples[0]
    assert text == "Just one paragraph."
    assert prev == ""
    assert next_ == ""


def test_multiple_paragraphs_under_budget_one_chunk() -> None:
    src = "First paragraph.\n\nSecond paragraph.\n\nThird."
    triples = chunk_text(src, target_tokens=1500, overlap_tokens=200)
    assert len(triples) == 1
    assert "First" in triples[0][0] and "Third" in triples[0][0]


def test_chunks_split_when_over_budget() -> None:
    # Force tiny budget so each "paragraph" lands in its own chunk.
    paragraphs = [f"Paragraph {i}: " + ("word " * 100) for i in range(5)]
    src = "\n\n".join(paragraphs)
    triples = chunk_text(src, target_tokens=120, overlap_tokens=20)
    assert len(triples) >= 3
    # All chunks non-empty
    assert all(t.strip() for t, _, _ in triples)
    # Concatenated chunk content covers all original paragraphs.
    combined = " ".join(t for t, _, _ in triples)
    for i in range(5):
        assert f"Paragraph {i}" in combined


def test_overlap_context_present() -> None:
    paragraphs = [f"P{i} " + ("alpha " * 50) for i in range(4)]
    src = "\n\n".join(paragraphs)
    triples = chunk_text(src, target_tokens=80, overlap_tokens=15)
    assert len(triples) >= 2
    # Middle chunks have non-empty prev and next context.
    middle = triples[1]
    _, prev_ctx, next_ctx = middle
    assert prev_ctx
    assert next_ctx


def test_long_single_paragraph_soft_splits_on_sentences() -> None:
    sentence = "This is a sentence. "
    # 100 sentences as one paragraph, no \n\n separators.
    src = (sentence * 100).strip()
    triples = chunk_text(src, target_tokens=80, overlap_tokens=10)
    assert len(triples) > 1
    # Each chunk should end at a sentence boundary (no mid-sentence cut).
    for text, _, _ in triples[:-1]:
        assert text.rstrip().endswith(".")


def test_empty_text_returns_empty() -> None:
    assert chunk_text("", target_tokens=100, overlap_tokens=10) == []
    assert chunk_text("   \n\n   ", target_tokens=100, overlap_tokens=10) == []


@pytest.fixture
def state(tmp_path: Path) -> PipelineState:
    p = tmp_path / "f.txt"
    text = "Para one.\n\nPara two has more words to translate.\n\nPara three is short."
    p.write_text(text)
    pair = LanguagePair(source="EN", target="PL")
    return PipelineState(
        source=SourceDocument(path=p, text=text, source_language="EN"),
        language_pair=pair,
        metadata=RunMetadata(
            run_id="t",
            source_path=p,
            language_pair=pair,
            pipeline_version="0.1.0",
        ),
    )


def test_chunking_stage_writes_chunks(state: PipelineState) -> None:
    stage = ChunkingStage(target_tokens=1500, overlap_tokens=200)
    out = stage.run(state)
    assert len(out.chunks) == 1
    assert out.chunks[0].index == 0
    assert "Para one" in out.chunks[0].text
    assert out.chunks[0].prev_context == ""
    assert out.chunks[0].next_context == ""


def test_chunking_stage_no_token_usage(state: PipelineState) -> None:
    stage = ChunkingStage()
    stage.reset_usage()
    stage.run(state)
    assert stage.last_input_tokens == 0
    assert stage.last_output_tokens == 0
    assert stage.last_cost_usd == 0.0


def test_chunking_passes_sentinel_through_unchanged() -> None:
    """``[[BLK]]`` sentinel lines are paragraph-separators in joined text and must
    survive the chunker without alteration so consistency-stage splitting works."""
    text = "Paragraph one.\n\n[[BLK]]\n\nParagraph two.\n\n[[BLK]]\n\nParagraph three."
    triples = chunk_text(text, target_tokens=1500, overlap_tokens=200)
    # Single chunk under budget — sentinel preserved verbatim.
    combined = "\n\n".join(t for t, _, _ in triples)
    assert combined.count("[[BLK]]") == 2
    assert "Paragraph one." in combined
    assert "Paragraph three." in combined
