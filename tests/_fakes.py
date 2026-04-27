"""Reusable fakes that mimic the slice of each SDK we actually call.

Each fake is constructed with a list of canned responses; each method call pops
one. This is the "recorded fixtures" pattern: the canned objects mirror the real
API response shape (validated against the SDK's actual usage in stage code), so
tests don't hit the network and don't burn API credits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- Anthropic fake ----------------------------------------------------------


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int
    server_tool_use: Any = None


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = "toolu_test"
    type: str = "tool_use"


@dataclass
class FakeAnthropicResponse:
    content: list[Any]
    usage: FakeUsage
    stop_reason: str = "end_turn"


class _FakeStreamContext:
    """Mimics ``client.messages.stream(...)`` context manager.

    The real SDK's ``stream()`` returns a context manager whose entered value is
    a stream object exposing ``get_final_message()``. We pre-canned the final
    message, so we just hand it over.
    """

    def __init__(self, response: FakeAnthropicResponse) -> None:
        self._response = response

    def __enter__(self) -> _FakeStreamContext:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_final_message(self) -> FakeAnthropicResponse:
        return self._response


class FakeAnthropicMessages:
    def __init__(self, responses: list[FakeAnthropicResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _next(self, kwargs: dict[str, Any]) -> FakeAnthropicResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeAnthropic ran out of canned responses")
        return self._responses.pop(0)

    def create(self, **kwargs: Any) -> FakeAnthropicResponse:
        # Kept for backward compat with any test that still uses .create().
        return self._next(kwargs)

    def stream(self, **kwargs: Any) -> _FakeStreamContext:
        return _FakeStreamContext(self._next(kwargs))


@dataclass
class FakeAnthropicClient:
    messages: FakeAnthropicMessages

    @classmethod
    def with_responses(
        cls, responses: list[FakeAnthropicResponse]
    ) -> FakeAnthropicClient:
        return cls(messages=FakeAnthropicMessages(responses))


# ---- OpenAI fake -------------------------------------------------------------


@dataclass
class FakeOpenAIUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int = 0


@dataclass
class FakeOpenAIMessage:
    parsed: Any = None
    content: str | None = None
    refusal: str | None = None


@dataclass
class FakeOpenAIChoice:
    message: FakeOpenAIMessage


@dataclass
class FakeOpenAICompletion:
    choices: list[FakeOpenAIChoice]
    usage: FakeOpenAIUsage


@dataclass
class FakeOpenAICompletionsAPI:
    parse_responses: list[FakeOpenAICompletion] = field(default_factory=list)
    create_responses: list[FakeOpenAICompletion] = field(default_factory=list)
    parse_calls: list[dict[str, Any]] = field(default_factory=list)
    create_calls: list[dict[str, Any]] = field(default_factory=list)

    def parse(self, **kwargs: Any) -> FakeOpenAICompletion:
        self.parse_calls.append(kwargs)
        if not self.parse_responses:
            raise AssertionError("FakeOpenAI.parse ran out of canned responses")
        return self.parse_responses.pop(0)

    def create(self, **kwargs: Any) -> FakeOpenAICompletion:
        self.create_calls.append(kwargs)
        if not self.create_responses:
            raise AssertionError("FakeOpenAI.create ran out of canned responses")
        return self.create_responses.pop(0)


@dataclass
class FakeOpenAIChat:
    completions: FakeOpenAICompletionsAPI


@dataclass
class FakeOpenAIClient:
    chat: FakeOpenAIChat

    @classmethod
    def with_parse_responses(
        cls, responses: list[FakeOpenAICompletion]
    ) -> FakeOpenAIClient:
        return cls(chat=FakeOpenAIChat(completions=FakeOpenAICompletionsAPI(parse_responses=responses)))

    @classmethod
    def with_create_responses(
        cls, responses: list[FakeOpenAICompletion]
    ) -> FakeOpenAIClient:
        return cls(chat=FakeOpenAIChat(completions=FakeOpenAICompletionsAPI(create_responses=responses)))


# ---- DeepL fake --------------------------------------------------------------


@dataclass
class FakeDeepLResult:
    text: str
    detected_source_lang: str | None = None
    billed_characters: int = 0


@dataclass
class FakeDeepLClient:
    """``translate_text`` mimics the real signature: returns one TextResult or a list."""

    responses: list[FakeDeepLResult]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def translate_text(
        self, text: str | list[str], **kwargs: Any
    ) -> FakeDeepLResult | list[FakeDeepLResult]:
        self.calls.append({"text": text, **kwargs})
        if not self.responses:
            raise AssertionError("FakeDeepL ran out of canned responses")
        if isinstance(text, list):
            out: list[FakeDeepLResult] = []
            for _ in text:
                out.append(self.responses.pop(0))
            return out
        return self.responses.pop(0)
