"""Pipeline cost / token estimator for ``translate --dry-run``.

Heuristic, not exact — multipliers are calibrated against the observed end-of-Phase-2
sample run. Better to say "approximately $X" before a $5 run than to call APIs.

The CLI prints both the per-stage breakdown and a total range so the user can
sanity-check before committing.
"""

from __future__ import annotations

from dataclasses import dataclass

from translation_pipeline.chunking import approx_tokens
from translation_pipeline.config import Settings
from translation_pipeline.pricing import deepl_character_cost, estimate_cost


@dataclass
class StageEstimate:
    name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class RunEstimate:
    stages: list[StageEstimate]
    total_cost_usd: float
    source_chars: int
    source_tokens: int
    chunks: int


# Per-stage multipliers applied to the source-token count.
# Calibrated from the observed Phase-2 small-sample run on PT-BR legal text.
# (input_mult, output_mult) — fractions of source tokens per chunk-equivalent.
_MULT: dict[str, tuple[float, float]] = {
    "brief_extraction": (1.5, 1.5),
    "draft_b": (2.0, 1.0),
    "synthesis": (4.0, 1.0),
    "critique": (3.0, 2.0),
    "improvement": (6.0, 1.0),
    "back_translation": (1.0, 1.0),
    "divergence_detection": (2.0, 0.3),
    "consistency": (4.0, 1.0),
}


def estimate(
    text: str,
    settings: Settings,
    *,
    target_chunk_tokens: int = 1500,
) -> RunEstimate:
    src_tokens = approx_tokens(text)
    chunks = max(1, src_tokens // target_chunk_tokens + (1 if src_tokens % target_chunk_tokens else 0))

    stages: list[StageEstimate] = []

    # Brief extraction (one call, sees the whole source).
    in_t, out_t = _MULT["brief_extraction"]
    stages.append(
        _llm("brief_extraction", settings.MODEL_BRIEF_EXTRACTION,
             int(src_tokens * in_t), int(src_tokens * out_t))
    )

    # DraftA (DeepL — char-billed).
    chars = len(text)
    stages.append(
        StageEstimate(
            name="draft_a",
            model="deepl",
            input_tokens=chars,
            output_tokens=0,
            cost_usd=deepl_character_cost(chars, settings.DEEPL_API_PLAN),
        )
    )

    # Per-chunk LLM stages.
    chunk_tokens = src_tokens / chunks
    for stage_name, model_attr in (
        ("draft_b", "MODEL_DRAFT_B"),
        ("synthesis", "MODEL_SYNTHESIS"),
        ("improvement", "MODEL_IMPROVEMENT"),
        ("consistency", "MODEL_CONSISTENCY"),
    ):
        in_m, out_m = _MULT[stage_name]
        stages.append(
            _llm(
                stage_name,
                getattr(settings, model_attr),
                int(chunk_tokens * in_m * chunks),
                int(chunk_tokens * out_m * chunks),
            )
        )

    # Single-call critique / back-translation / divergence on the full text.
    for stage_name, model_attr in (
        ("critique", "MODEL_CRITIQUE"),
        ("back_translation", "MODEL_BACK_TRANSLATION"),
        ("divergence_detection", "MODEL_DIVERGENCE_DETECTION"),
    ):
        in_m, out_m = _MULT[stage_name]
        stages.append(
            _llm(
                stage_name,
                getattr(settings, model_attr),
                int(src_tokens * in_m),
                int(src_tokens * out_m),
            )
        )

    total = sum(s.cost_usd for s in stages)
    return RunEstimate(
        stages=stages,
        total_cost_usd=total,
        source_chars=len(text),
        source_tokens=src_tokens,
        chunks=chunks,
    )


def _llm(name: str, model: str, in_t: int, out_t: int) -> StageEstimate:
    return StageEstimate(
        name=name,
        model=model,
        input_tokens=in_t,
        output_tokens=out_t,
        cost_usd=estimate_cost(model, in_t, out_t),
    )
