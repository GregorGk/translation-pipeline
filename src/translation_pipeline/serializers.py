"""Plain-text + YAML metadata serializers for the final translation.

Both serializers compute output paths next to the source file, lowercase the
language code (``zawiadomienie.en.txt``, not ``zawiadomienie.EN.txt``), and
honor a ``-2 / -3 / ...`` suffix bumping rule on collision rather than
overwriting silently.

The YAML metadata is shaped for human review: ``warnings`` is the first key so
problems are visible at the top of the file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from translation_pipeline.models import FinalOutput, LanguageCode, RunMetadata


def safe_output_path(source: Path, lang: LanguageCode, ext: str) -> Path:
    """Return a non-colliding output path next to ``source``.

    Pattern: ``{source.stem}.{lang.lower()}{ext}`` with ``-2``, ``-3``, …
    appended on collision. Never overwrites an existing file.
    """
    base = source.with_name(f"{source.stem}.{lang.lower()}{ext}")
    if not base.exists():
        return base
    i = 2
    while True:
        candidate = source.with_name(f"{source.stem}.{lang.lower()}-{i}{ext}")
        if not candidate.exists():
            return candidate
        i += 1


# Backwards-compatible private alias retained for internal use.
_safe_path = safe_output_path


class PlainTextSerializer:
    """Writes ``FinalOutput.text`` to ``{stem}.{lang}.txt`` next to the source."""

    def write(self, source_path: Path, final: FinalOutput) -> Path:
        out = safe_output_path(source_path, final.language_pair.target, ".txt")
        out.write_text(final.text, encoding="utf-8")
        return out


class MetadataSerializer:
    """Writes ``RunMetadata`` to ``{stem}.{lang}.meta.yaml`` with warnings on top."""

    def write(self, source_path: Path, metadata: RunMetadata) -> Path:
        out = safe_output_path(source_path, metadata.language_pair.target, ".meta.yaml")
        out.write_text(self.to_yaml(metadata), encoding="utf-8")
        return out

    def to_yaml(self, metadata: RunMetadata) -> str:
        return yaml.safe_dump(
            _shape_metadata(metadata),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )


def _shape_metadata(metadata: RunMetadata) -> dict[str, Any]:
    """Hand-shape the dict so YAML keys appear in a useful order.

    ``warnings`` first (problems first), then run identity, then stages, then totals.
    """
    return {
        "warnings": list(metadata.warnings),
        "run_id": metadata.run_id,
        "source_path": str(metadata.source_path),
        "language_pair": {
            "source": metadata.language_pair.source,
            "target": metadata.language_pair.target,
        },
        "pipeline_version": metadata.pipeline_version,
        "prompt_hashes": dict(metadata.prompt_hashes),
        "stages": [_shape_stage(s) for s in metadata.stages],
        "totals": {
            "cost_usd": round(metadata.total_cost_usd, 6),
            "duration_s": round(metadata.total_duration_s, 3),
        },
    }


def _shape_stage(stage: Any) -> dict[str, Any]:
    return {
        "name": stage.name,
        "status": stage.status,
        "model": stage.model,
        "attempts": stage.attempts,
        "duration_s": round(stage.duration_s, 3),
        "input_tokens": stage.input_tokens,
        "output_tokens": stage.output_tokens,
        "cost_usd": round(stage.cost_usd, 6),
        "started_at": _iso(stage.started_at),
        "completed_at": _iso(stage.completed_at),
        "error": stage.error,
    }


def _iso(dt: datetime) -> str:
    return dt.isoformat()
