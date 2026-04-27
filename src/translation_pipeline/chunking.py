"""Paragraph-aware text chunker used by ``ChunkingStage``.

Token counts are approximated with a 4-chars-per-token heuristic. We don't need
exact counts — chunk boundaries are soft, and the downstream LLMs see the actual
text. The heuristic keeps chunking pure-Python (no tiktoken model lookup) and
treats Cyrillic / Greek conservatively (counted the same; chunks come out
slightly under target, which is safe).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class _Para:
    text: str
    tokens: int


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[_Para]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [_Para(text=p, tokens=approx_tokens(p)) for p in paras]


def _soft_split_long_paragraph(para: str, target_tokens: int) -> list[str]:
    """Split a paragraph too large for one chunk on sentence boundaries.

    If a single sentence exceeds the target, we keep it intact rather than
    cutting mid-sentence — better to overshoot the target than fragment meaning.
    """
    sentences = _SENT_SPLIT.split(para)
    if not sentences:
        return [para]

    out: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for s in sentences:
        s_tok = approx_tokens(s)
        if buf and buf_tok + s_tok > target_tokens:
            out.append(" ".join(buf))
            buf = [s]
            buf_tok = s_tok
        else:
            buf.append(s)
            buf_tok += s_tok
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_text(
    text: str,
    *,
    target_tokens: int = 1500,
    overlap_tokens: int = 200,
) -> list[tuple[str, str, str]]:
    """Return ``[(text, prev_context, next_context), ...]``.

    Grouping is paragraph-aware: paragraphs are appended to the current chunk
    until adding the next one would exceed ``target_tokens``. Paragraphs longer
    than ``target_tokens`` are soft-split on sentence boundaries.

    ``prev_context`` is the tail of the preceding chunk; ``next_context`` is the
    head of the following chunk. Both are bounded by ``overlap_tokens``.
    Single-paragraph documents → single chunk with empty contexts.
    """
    paras = _split_paragraphs(text)
    if not paras:
        return [(text.strip(), "", "")] if text.strip() else []

    expanded: list[str] = []
    for p in paras:
        if p.tokens <= target_tokens:
            expanded.append(p.text)
        else:
            expanded.extend(_soft_split_long_paragraph(p.text, target_tokens))

    chunks: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for piece in expanded:
        piece_tok = approx_tokens(piece)
        if buf and buf_tok + piece_tok > target_tokens:
            chunks.append("\n\n".join(buf))
            buf = [piece]
            buf_tok = piece_tok
        else:
            buf.append(piece)
            buf_tok += piece_tok
    if buf:
        chunks.append("\n\n".join(buf))

    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    out: list[tuple[str, str, str]] = []
    for i, c in enumerate(chunks):
        prev_ctx = chunks[i - 1][-overlap_chars:] if i > 0 else ""
        next_ctx = chunks[i + 1][:overlap_chars] if i < len(chunks) - 1 else ""
        out.append((c, prev_ctx, next_ctx))
    return out
