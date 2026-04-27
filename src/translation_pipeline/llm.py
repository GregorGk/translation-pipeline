"""Thin wrappers around Anthropic / OpenAI calls used by the LLM-driven stages.

Centralizes:
- mapping SDK exceptions to ``StageError`` so tenacity retries on transient API faults
- pulling token usage off the response in a stable shape
- minor differences between the two SDKs (tool_use vs response_format)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from translation_pipeline.stages.base import StageError

if TYPE_CHECKING:
    import anthropic
    from openai import OpenAI


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class StructuredResult[M: BaseModel]:
    parsed: M
    usage: LLMUsage


@dataclass
class TextResult:
    text: str
    usage: LLMUsage


# ---- Anthropic ---------------------------------------------------------------

def _anthropic_stream_to_message(
    client: anthropic.Anthropic, kwargs: dict[str, Any]
) -> Any:
    """Run an Anthropic call via streaming and return the accumulated Message.

    Per Anthropic guidance, streaming is required when ``max_tokens`` is large
    enough that a single response could exceed the HTTP socket idle timeout
    (~10 min). We always stream so a slow generation never reads as a timeout.

    Transient transport errors during streaming (peer reset, incomplete chunked
    read, network blip) are wrapped in ``StageError`` so the pipeline's tenacity
    retry loop kicks in instead of aborting on the first hiccup.
    """
    import anthropic as _anthropic
    import httpx

    try:
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()
    except (
        _anthropic.APIConnectionError,
        _anthropic.APIStatusError,
        _anthropic.RateLimitError,
        _anthropic.APITimeoutError,
        httpx.HTTPError,
    ) as e:
        raise StageError(f"anthropic api error: {e}") from e


def anthropic_structured[T: BaseModel](
    client: anthropic.Anthropic,
    *,
    model: str,
    max_tokens: int,
    prompt: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
    schema_model: type[T],
    system: str | None = None,
) -> StructuredResult[T]:
    """Call Claude with a single forced tool_use (streamed), parse the tool input."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    if system is not None:
        kwargs["system"] = system
    response = _anthropic_stream_to_message(client, kwargs)

    tool_use = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_use is None:
        raise StageError(
            f"claude did not call tool '{tool_name}' (stop_reason={response.stop_reason})"
        )
    try:
        parsed = schema_model.model_validate(tool_use.input)
    except Exception as e:
        raise StageError(
            f"claude returned invalid {schema_model.__name__} payload: {e}"
        ) from e

    usage = LLMUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return StructuredResult(parsed=parsed, usage=usage)


def anthropic_text(
    client: anthropic.Anthropic,
    *,
    model: str,
    max_tokens: int,
    prompt: str,
    system: str | None = None,
) -> TextResult:
    """Call Claude for plain text via streaming. Joins all text blocks."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system is not None:
        kwargs["system"] = system
    response = _anthropic_stream_to_message(client, kwargs)

    parts = [
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ]
    if not parts:
        raise StageError(
            f"claude returned no text (stop_reason={response.stop_reason})"
        )
    return TextResult(
        text="".join(parts),
        usage=LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
    )


# ---- OpenAI ------------------------------------------------------------------

def openai_structured[T: BaseModel](
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    schema_model: type[T],
    system: str | None = None,
) -> StructuredResult[T]:
    """Call GPT-5 family with response_format=schema_model for parsed output."""
    import httpx
    import openai

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format=schema_model,
        )
    except (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
        httpx.HTTPError,
    ) as e:
        raise StageError(f"openai api error: {e}") from e

    msg = completion.choices[0].message
    if msg.refusal:
        raise StageError(f"openai refused: {msg.refusal}")
    if msg.parsed is None:
        raise StageError("openai returned no parsed payload")

    usage = completion.usage
    return StructuredResult(
        parsed=msg.parsed,
        usage=LLMUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
    )


def openai_text(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    system: str | None = None,
) -> TextResult:
    import httpx
    import openai

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
        )
    except (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
        httpx.HTTPError,
    ) as e:
        raise StageError(f"openai api error: {e}") from e

    msg = completion.choices[0].message
    if msg.refusal:
        raise StageError(f"openai refused: {msg.refusal}")
    if not msg.content:
        raise StageError("openai returned empty content")
    usage = completion.usage
    return TextResult(
        text=msg.content,
        usage=LLMUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
    )
