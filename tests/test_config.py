from __future__ import annotations

from pathlib import Path

import pytest

from translation_pipeline.config import load_settings


def _write_env(tmp_path: Path, body: str) -> Path:
    env = tmp_path / ".env"
    env.write_text(body)
    return env


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in a clean tmp cwd with all relevant env vars unset.

    Without this, the project's real .env (sitting in repo root) gets picked up
    when tests run from there, masking missing-key behavior.
    """
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPL_API_KEY", "DEEPL_API_PLAN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)


def test_settings_loads_from_env_file(tmp_path: Path) -> None:
    _write_env(
        tmp_path,
        "ANTHROPIC_API_KEY=sk-ant-test\n"
        "OPENAI_API_KEY=sk-openai-test\n"
        "DEEPL_API_KEY=deepl-test\n"
        "DEEPL_API_PLAN=pro\n",
    )

    s = load_settings()

    assert s.ANTHROPIC_API_KEY.get_secret_value() == "sk-ant-test"
    assert s.OPENAI_API_KEY.get_secret_value() == "sk-openai-test"
    assert s.DEEPL_API_KEY.get_secret_value() == "deepl-test"
    assert s.DEEPL_API_PLAN == "pro"


def test_settings_default_plan_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("DEEPL_API_KEY", "d")

    s = load_settings()
    assert s.DEEPL_API_PLAN == "free"


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only two of three required keys present.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")

    with pytest.raises(RuntimeError) as exc:
        load_settings()
    assert "DEEPL_API_KEY" in str(exc.value)


def test_empty_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("DEEPL_API_KEY", "   ")

    with pytest.raises(RuntimeError):
        load_settings()
