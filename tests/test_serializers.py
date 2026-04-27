from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from translation_pipeline.models import (
    FinalOutput,
    LanguagePair,
    RunMetadata,
    StageRecord,
    TranslationBrief,
)
from translation_pipeline.serializers import (
    MetadataSerializer,
    PlainTextSerializer,
)


@pytest.fixture
def src(tmp_path: Path) -> Path:
    p = tmp_path / "doc.txt"
    p.write_text("source")
    return p


@pytest.fixture
def final() -> FinalOutput:
    return FinalOutput(
        text="hello translation",
        language_pair=LanguagePair(source="PL", target="EN"),
        brief=TranslationBrief(
            document_type="legal", register_level="formal", target_audience="lawyer"
        ),
        warnings=["one warning"],
    )


@pytest.fixture
def metadata() -> RunMetadata:
    md = RunMetadata(
        run_id="abc123",
        source_path=Path("/tmp/doc.txt"),
        language_pair=LanguagePair(source="PL", target="EN"),
        pipeline_version="0.1.0",
        prompt_hashes={"brief_extraction": "deadbeef" * 8},
        warnings=["divergence (high) at p1: meaning shifted"],
    )
    md.add_stage(
        StageRecord(
            name="brief_extraction",
            model="claude-sonnet-4-6",
            started_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 4, 27, 12, 0, 30, tzinfo=UTC),
            duration_s=30.0,
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.012345,
            status="ok",
            attempts=1,
        )
    )
    md.add_stage(
        StageRecord(
            name="critique",
            model="gpt-5.5",
            started_at=datetime(2026, 4, 27, 12, 1, 0, tzinfo=UTC),
            completed_at=datetime(2026, 4, 27, 12, 1, 40, tzinfo=UTC),
            duration_s=40.0,
            input_tokens=3000,
            output_tokens=2000,
            cost_usd=0.075,
            status="skipped",
            error="StageError: outage",
            attempts=3,
        )
    )
    return md


def test_plaintext_writes_lowercase_lang(src: Path, final: FinalOutput) -> None:
    out = PlainTextSerializer().write(src, final)
    assert out.name == "doc.en.txt"
    assert out.read_text() == "hello translation"


def test_plaintext_collision_appends_suffix(src: Path, final: FinalOutput) -> None:
    s = PlainTextSerializer()
    a = s.write(src, final)
    b = s.write(src, final)
    c = s.write(src, final)
    assert a.name == "doc.en.txt"
    assert b.name == "doc.en-2.txt"
    assert c.name == "doc.en-3.txt"
    # All three exist with their content.
    assert a.exists() and b.exists() and c.exists()


def test_metadata_yaml_warnings_first(src: Path, metadata: RunMetadata) -> None:
    out = MetadataSerializer().write(src, metadata)
    assert out.name == "doc.en.meta.yaml"
    body = out.read_text()
    # ``warnings`` must be the first top-level key.
    first_key = body.splitlines()[0].split(":")[0]
    assert first_key == "warnings"


def test_metadata_yaml_round_trips(metadata: RunMetadata) -> None:
    rendered = MetadataSerializer().to_yaml(metadata)
    parsed = yaml.safe_load(rendered)
    assert parsed["run_id"] == "abc123"
    assert parsed["language_pair"] == {"source": "PL", "target": "EN"}
    assert parsed["totals"]["cost_usd"] == pytest.approx(0.087345, abs=1e-6)
    assert len(parsed["stages"]) == 2
    assert parsed["stages"][0]["name"] == "brief_extraction"
    assert parsed["stages"][1]["status"] == "skipped"
    assert parsed["stages"][1]["attempts"] == 3
    assert parsed["warnings"] == ["divergence (high) at p1: meaning shifted"]


def test_metadata_yaml_human_readable(metadata: RunMetadata) -> None:
    body = MetadataSerializer().to_yaml(metadata)
    # No flow-style {} or [] gibberish for nested objects.
    assert "{" not in body.split("prompt_hashes:")[0]
    # Unicode allowed (Polish, etc.).
    assert "\\u" not in body


def test_pt_br_lang_lowercased(src: Path) -> None:
    final = FinalOutput(
        text="x",
        language_pair=LanguagePair(source="EN", target="PT-BR"),
        brief=TranslationBrief(
            document_type="t", register_level="n", target_audience="a"
        ),
    )
    out = PlainTextSerializer().write(src, final)
    assert out.name == "doc.pt-br.txt"
