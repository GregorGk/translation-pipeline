"""Per-model pricing table and cost computation.

Rates are USD per 1M tokens (input / output) verified against vendor pricing pages
on 2026-04-27. OpenAI doubled GPT-5 line prices on 2026-04-23 with the GPT-5.5
release; numbers here reflect that. ``estimate_cost`` returns 0 when the model
isn't in the table rather than raising, so an unknown ID never aborts a run; the
unknown name is silently skipped in the metadata sum and the user can spot it
via the per-stage records.

Sources:
- Anthropic: https://platform.claude.com/docs/en/docs/about-claude/models/overview
- OpenAI: https://openai.com/api/pricing
- DeepL: https://support.deepl.com/hc/en-us/articles/360021200939
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRate:
    input_per_mtok: float
    output_per_mtok: float


# Anthropic and OpenAI rates per 1M tokens (input, output).
# DeepL is character-billed and accounted separately by ``deepl_character_cost``.
RATES: dict[str, ModelRate] = {
    # Anthropic Claude 4.x (current as of 2026-04-27)
    "claude-opus-4-7": ModelRate(5.0, 25.0),
    "claude-sonnet-4-6": ModelRate(3.0, 15.0),
    "claude-haiku-4-5-20251001": ModelRate(1.0, 5.0),
    "claude-haiku-4-5": ModelRate(1.0, 5.0),
    # OpenAI GPT-5 family (post 2026-04-23 price update)
    "gpt-5": ModelRate(0.625, 5.0),
    "gpt-5.5": ModelRate(5.0, 30.0),
    "gpt-5-pro": ModelRate(15.0, 120.0),
    "gpt-5.5-pro": ModelRate(30.0, 180.0),
}

# DeepL Pro: $25 per 1M characters of source text.
DEEPL_PRO_USD_PER_M_CHARS: float = 25.0


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rate = RATES.get(model)
    if rate is None:
        return 0.0
    return (
        input_tokens * rate.input_per_mtok / 1_000_000.0
        + output_tokens * rate.output_per_mtok / 1_000_000.0
    )


def deepl_character_cost(billed_characters: int, plan: str) -> float:
    """DeepL Free is 500K chars/month at $0; Pro bills $25/M characters."""
    if plan == "free":
        return 0.0
    return billed_characters * DEEPL_PRO_USD_PER_M_CHARS / 1_000_000.0
