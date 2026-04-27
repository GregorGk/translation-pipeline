"""Load and hash prompt files from ``prompts/`` for reproducibility.

Prompt files are markdown with optional ``{var}`` placeholders. ``load_prompt``
returns both the raw template and a stable SHA-256 hash; the hash gets recorded
in ``RunMetadata.prompt_hashes`` so a stored translation can be diffed against
the prompt set that produced it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    template: str
    sha256: str

    def render(self, **vars: object) -> str:
        return self.template.format(**vars)


@cache
def load_prompt(name: str, prompts_dir: Path | None = None) -> Prompt:
    base = prompts_dir if prompts_dir is not None else _PROMPTS_DIR
    path = base / f"{name}.md"
    raw = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return Prompt(name=name, template=raw, sha256=digest)
