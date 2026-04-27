from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from translation_pipeline.models import PipelineState, StageCriticality

ProgressEmitter = Callable[[int, int, str], None]


class StageError(Exception):
    """Raised by a stage to signal a recoverable failure (will be retried)."""


class StageDependencyMissing(Exception):
    """Raised by a stage when its required upstream output is absent.

    A non-critical upstream stage may have been skipped; the pipeline catches this and
    skips the dependent stage rather than aborting.
    """


class PipelineStage(ABC):
    """Abstract contract for a pipeline stage.

    Subclasses declare ``name``, ``criticality``, and (optionally) ``model_id``. The
    ``run`` method takes the accumulated PipelineState, mutates/returns it with its
    own output filled in.

    Stages report token usage and cost by calling ``self._record_usage(...)`` from
    inside ``run``; the pipeline reads ``self.last_*`` after each successful run and
    adds those values to the corresponding ``StageRecord``.

    Pipeline-level concerns — retries, metadata recording, criticality handling — live
    in :class:`Pipeline`, not here. Stages just do their work.
    """

    name: ClassVar[str]
    criticality: ClassVar[StageCriticality]

    # Per-instance: model identifier and prompt hash. Subclasses set these in
    # ``__init__`` so the same class can be instantiated against different models
    # / prompt revisions.
    model_id: str | None = None
    prompt_hash: str | None = None

    # Per-call usage. Reset by the pipeline before every attempt.
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_cost_usd: float = 0.0

    # Pipeline injects a callback before ``run`` so chunked stages can report
    # per-chunk progress. ``None`` means progress reporting is disabled (tests).
    progress_emitter: ProgressEmitter | None = None

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Execute this stage. Raise StageError on recoverable failure."""

    def reset_usage(self) -> None:
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cost_usd = 0.0

    def _emit_progress(self, current: int, total: int, label: str = "") -> None:
        """Report per-chunk progress to the pipeline (no-op if no callback)."""
        if self.progress_emitter is not None:
            self.progress_emitter(current, total, label)

    def _record_usage(
        self, input_tokens: int, output_tokens: int, cost_usd: float
    ) -> None:
        self.last_input_tokens += input_tokens
        self.last_output_tokens += output_tokens
        self.last_cost_usd += cost_usd

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("name", "criticality"):
            if not hasattr(cls, attr) or getattr(cls, attr) is None:
                raise TypeError(f"{cls.__name__} must define class attribute `{attr}`")
