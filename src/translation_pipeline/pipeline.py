from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from translation_pipeline import __version__
from translation_pipeline.logging import get_logger
from translation_pipeline.models import (
    LanguageCode,
    LanguagePair,
    PipelineState,
    RunMetadata,
    SourceDocument,
    StageRecord,
)
from translation_pipeline.stages.base import (
    PipelineStage,
    StageDependencyMissing,
    StageError,
)

log = get_logger("pipeline")

# (event, stage_name, info?) — event ∈ {"start", "ok", "skipped", "failed", "progress"}.
# ``info`` carries {"current": int, "total": int, "label": str} for "progress" events,
# otherwise None.
StageEventCallback = Callable[[str, str, dict[str, Any] | None], None]


class PipelineAbort(RuntimeError):
    """A critical stage failed all retries; the pipeline aborted."""


class Pipeline:
    """Sequences stages, applies retries, records metadata.

    Each stage runs at most ``retry_attempts`` times with exponential backoff between
    attempts. On persistent failure: critical stages abort the run with PipelineAbort;
    non-critical stages are skipped, a warning is logged to metadata, and the previous
    state is passed to the next stage unchanged.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        *,
        retry_attempts: int = 3,
        retry_wait_min: float = 1.0,
        retry_wait_max: float = 10.0,
        retry_wait_multiplier: float = 1.0,
    ) -> None:
        self.stages = stages
        self._retry_attempts = retry_attempts
        self._retry_wait_min = retry_wait_min
        self._retry_wait_max = retry_wait_max
        self._retry_wait_multiplier = retry_wait_multiplier

    def run(
        self,
        source: SourceDocument,
        target_language: LanguageCode,
        on_event: StageEventCallback | None = None,
    ) -> PipelineState:
        pair = LanguagePair(source=source.source_language, target=target_language)
        state = PipelineState(
            source=source,
            language_pair=pair,
            metadata=RunMetadata(
                run_id=str(uuid.uuid4()),
                source_path=source.path,
                language_pair=pair,
                pipeline_version=__version__,
            ),
        )
        for stage in self.stages:
            if stage.prompt_hash is not None:
                state.metadata.prompt_hashes[stage.name] = stage.prompt_hash

        for stage in self.stages:
            state = self._run_stage(stage, state, on_event)
        return state

    def _run_stage(
        self,
        stage: PipelineStage,
        state: PipelineState,
        on_event: StageEventCallback | None = None,
    ) -> PipelineState:
        if on_event is not None:
            on_event("start", stage.name, None)
        if on_event is not None:
            stage.progress_emitter = (
                lambda c, t, label="": on_event("progress", stage.name,
                                               {"current": c, "total": t, "label": label})
            )
        else:
            stage.progress_emitter = None
        started_at = datetime.now(UTC)
        t0 = time.monotonic()
        attempts = 0
        last_error: BaseException | None = None

        retrier = Retrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(
                multiplier=self._retry_wait_multiplier,
                min=self._retry_wait_min,
                max=self._retry_wait_max,
            ),
            retry=retry_if_exception_type(StageError),
            reraise=False,
        )

        try:
            for attempt in retrier:
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    log.debug(
                        "stage %s attempt %d/%d",
                        stage.name,
                        attempts,
                        self._retry_attempts,
                    )
                    stage.reset_usage()
                    state = stage.run(state)
        except RetryError as e:
            inner = e.last_attempt.exception()
            last_error = inner if inner is not None else e
        except StageDependencyMissing as e:
            out = self._record_skip(
                stage,
                state,
                started_at=started_at,
                duration_s=time.monotonic() - t0,
                attempts=1,
                reason=f"dependency missing: {e}",
            )
            if on_event is not None:
                on_event("skipped", stage.name, None)
            return out
        except Exception as e:
            # Non-StageError exceptions are not retried. Treat as a single failed attempt.
            attempts = max(attempts, 1)
            last_error = e

        duration_s = time.monotonic() - t0
        completed_at = datetime.now(UTC)

        if last_error is None:
            state.metadata.add_stage(
                StageRecord(
                    name=stage.name,
                    model=stage.model_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_s=duration_s,
                    input_tokens=stage.last_input_tokens,
                    output_tokens=stage.last_output_tokens,
                    cost_usd=stage.last_cost_usd,
                    status="ok",
                    attempts=attempts,
                )
            )
            if on_event is not None:
                on_event("ok", stage.name, None)
            return state

        if stage.criticality == "critical":
            state.metadata.add_stage(
                StageRecord(
                    name=stage.name,
                    model=stage.model_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_s=duration_s,
                    status="failed",
                    error=f"{type(last_error).__name__}: {last_error}",
                    attempts=attempts,
                )
            )
            if on_event is not None:
                on_event("failed", stage.name, None)
            raise PipelineAbort(
                f"Critical stage '{stage.name}' failed after {attempts} attempt(s): "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error

        warning = (
            f"stage '{stage.name}' skipped after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        )
        log.warning(warning)
        state.metadata.warnings.append(warning)
        state.metadata.add_stage(
            StageRecord(
                name=stage.name,
                model=stage.model_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_s=duration_s,
                status="skipped",
                error=f"{type(last_error).__name__}: {last_error}",
                attempts=attempts,
            )
        )
        if on_event is not None:
            on_event("skipped", stage.name, None)
        return state

    def _record_skip(
        self,
        stage: PipelineStage,
        state: PipelineState,
        *,
        started_at: datetime,
        duration_s: float,
        attempts: int,
        reason: str,
    ) -> PipelineState:
        warning = f"stage '{stage.name}' skipped: {reason}"
        log.warning(warning)
        state.metadata.warnings.append(warning)
        state.metadata.add_stage(
            StageRecord(
                name=stage.name,
                model=stage.model_id,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_s=duration_s,
                status="skipped",
                error=reason,
                attempts=attempts,
            )
        )
        return state
