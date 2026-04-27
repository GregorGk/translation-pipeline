from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / .env.

    The four API-related secrets are required. Model identifiers default to a
    cost-aware mix (Sonnet for routine stages, Opus for high-stakes synthesis /
    improvement / consistency) and are individually overridable via env vars so
    you can swap a model without touching code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    ANTHROPIC_API_KEY: SecretStr = Field(...)
    OPENAI_API_KEY: SecretStr = Field(...)
    DEEPL_API_KEY: SecretStr = Field(...)
    DEEPL_API_PLAN: Literal["free", "pro"] = Field(default="free")

    # Per-stage model identifiers. Defaults are the recommended cost-aware mix.
    MODEL_BRIEF_EXTRACTION: str = "claude-sonnet-4-6"
    MODEL_DRAFT_B: str = "claude-sonnet-4-6"
    MODEL_SYNTHESIS: str = "claude-opus-4-7"
    MODEL_CRITIQUE: str = "gpt-5.5"
    MODEL_IMPROVEMENT: str = "claude-opus-4-7"
    MODEL_BACK_TRANSLATION: str = "gpt-5.5"
    MODEL_DIVERGENCE_DETECTION: str = "claude-sonnet-4-6"
    MODEL_CONSISTENCY: str = "claude-opus-4-7"
    MODEL_LANGUAGE_DETECT: str = "claude-haiku-4-5-20251001"

    # Default per-call output cap. Stages that need more (full-document synthesis,
    # consistency sweep) override this on the call.
    MAX_OUTPUT_TOKENS: int = 4096

    @field_validator("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPL_API_KEY")
    @classmethod
    def _reject_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("must not be empty")
        return v


def load_settings() -> Settings:
    """Construct Settings, re-raising validation failures with a friendlier message.

    Empty env vars (e.g. ``ANTHROPIC_API_KEY=`` exported by a shell rc but never
    populated) shadow values in ``.env`` and yield confusing "must not be empty"
    errors. We unset them in-process before constructing Settings so the .env
    fallback works.
    """
    import os

    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPL_API_KEY"):
        if key in os.environ and not os.environ[key].strip():
            del os.environ[key]

    try:
        return Settings()
    except Exception as e:
        raise RuntimeError(
            "Failed to load configuration. Ensure ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "and DEEPL_API_KEY are set (in environment or .env). "
            f"Underlying error: {e}"
        ) from e
