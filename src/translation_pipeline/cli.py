"""``translate`` — Typer-driven entry point for the pipeline.

Honors the user's ``--from``, ``--to``, ``--verbose``, and ``--dry-run`` flags
and writes the translation + metadata sidecar next to the source on success.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated, Any, cast, get_args

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from translation_pipeline import __version__
from translation_pipeline.clients import anthropic_client
from translation_pipeline.config import load_settings
from translation_pipeline.documents import (
    SUPPORTED_INPUT_EXTENSIONS,
    DocxFormatPreservingWriter,
    DocxWriter,
    PdfFormatPreservingWriter,
    PdfWriter,
    StructuredSource,
    UnsupportedDocumentFormat,
    read_document,
    read_structured,
)
from translation_pipeline.estimator import RunEstimate, estimate
from translation_pipeline.factory import build_default_pipeline
from translation_pipeline.language_detect import detect_language
from translation_pipeline.logging import setup_logging
from translation_pipeline.models import LanguageCode, SourceDocument
from translation_pipeline.pipeline import PipelineAbort
from translation_pipeline.serializers import (
    MetadataSerializer,
    PlainTextSerializer,
    safe_output_path,
)


def _version_callback(value: bool) -> None:
    if value:
        Console().print(f"translate {__version__}")
        raise typer.Exit(0)


app = typer.Typer(
    add_completion=False,
    help="Multi-pass document translator.",
    no_args_is_help=True,
)
console = Console()


def _coerce_lang(s: str, *, flag: str) -> LanguageCode:
    if s not in get_args(LanguageCode):
        raise typer.BadParameter(
            f"unsupported language for {flag}: {s!r}. "
            f"Supported: {', '.join(sorted(get_args(LanguageCode)))}"
        )
    return cast(LanguageCode, s)


def _ordered_stage_names() -> list[str]:
    return [
        "brief_extraction",
        "chunking",
        "draft_a",
        "draft_b",
        "synthesis",
        "critique",
        "improvement",
        "back_translation",
        "divergence_detection",
        "consistency",
    ]


def _print_dry_run(estimate_: RunEstimate, src: Path, src_lang: str, tgt_lang: str) -> None:
    console.print(f"[bold]Dry-run estimate[/bold]  {src} ({src_lang} → {tgt_lang})")
    console.print(
        f"  source chars: {estimate_.source_chars:,}   "
        f"~tokens: {estimate_.source_tokens:,}   "
        f"chunks: {estimate_.chunks}"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("stage")
    table.add_column("model")
    table.add_column("in tok / chars", justify="right")
    table.add_column("out tok", justify="right")
    table.add_column("est. cost", justify="right")
    for s in estimate_.stages:
        table.add_row(
            s.name,
            s.model,
            f"{s.input_tokens:,}",
            f"{s.output_tokens:,}",
            f"${s.cost_usd:.4f}",
        )
    table.add_row("", "", "", "", "")
    table.add_row(
        "[bold]TOTAL[/bold]", "", "", "",
        f"[bold]${estimate_.total_cost_usd:.4f}[/bold]",
    )
    console.print(table)
    console.print(
        "[dim]Estimates are heuristic — actual cost typically within ±50%.[/dim]"
    )


def _make_progress_table() -> tuple[Table, dict[str, int]]:
    """Return a Rich Table and a name → row-index map.

    We rebuild the table on every event because Live + Table cell updates need a
    fresh table to repaint cleanly.
    """
    names = _ordered_stage_names()
    name_to_row = {n: i for i, n in enumerate(names)}
    return _render_progress({n: {"status": "pending"} for n in names}), name_to_row


def _render_progress(states: dict[str, dict[str, object]]) -> Table:
    t = Table(show_header=True, header_style="bold")
    t.add_column("stage", width=22)
    t.add_column("status", width=14)
    t.add_column("progress", width=10, justify="right")
    t.add_column("elapsed", width=8, justify="right")
    for name in _ordered_stage_names():
        info = states.get(name, {"status": "pending"})
        status = str(info.get("status", "pending"))
        styled = _style_status(status)
        progress = ""
        if info.get("total"):
            progress = f"{info.get('current', 0)}/{info['total']}"
        elapsed = ""
        if "started" in info:
            ended = info.get("ended", time.monotonic())
            secs = float(ended) - float(info["started"])  # type: ignore[arg-type]
            elapsed = f"{secs:.0f}s"
        t.add_row(name, styled, progress, elapsed)
    return t


def _style_status(status: str) -> str:
    return {
        "pending": "[dim]pending[/dim]",
        "running": "[yellow]running…[/yellow]",
        "ok": "[green]ok[/green]",
        "skipped": "[yellow]skipped[/yellow]",
        "failed": "[red]failed[/red]",
    }.get(status, status)


@app.command()
def translate(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True, dir_okay=False)],
    to: Annotated[str, typer.Option("--to", help="Target language code (e.g. EN, PT-BR, PL)")] = ...,  # type: ignore[assignment]
    from_: Annotated[str | None, typer.Option(
        "--from",
        help="Source language code; auto-detected from a small Claude call if omitted.",
    )] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Verbose logging")] = False,
    dry_run: Annotated[bool, typer.Option(
        "--dry-run", help="Estimate token use and cost; do not call any API."
    )] = False,
    preserve_format: Annotated[bool, typer.Option(
        "--preserve-format/--no-preserve-format",
        help=(
            "Mutate the source document in place when writing .docx/.pdf "
            "outputs (preserves fonts, headings, tables, page layout). "
            "Default: on for .docx/.pdf input."
        ),
    )] = True,
    out_docx: Annotated[bool | None, typer.Option(
        "--out-docx/--no-out-docx",
        help="Also emit a .docx alongside the .txt. Defaults to true for .docx input.",
    )] = None,
    out_pdf: Annotated[bool | None, typer.Option(
        "--out-pdf/--no-out-pdf",
        help="Also emit a .pdf alongside the .txt. Defaults to true for .pdf input.",
    )] = None,
    version: Annotated[bool, typer.Option(
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    )] = False,
) -> None:
    """Translate a document (.txt, .pdf, or .docx) through the full pipeline."""
    setup_logging(verbose=verbose)
    try:
        settings = load_settings()
    except RuntimeError as e:
        console.print(f"[red]configuration error:[/red] {e}", highlight=False)
        raise typer.Exit(code=2) from e

    if input_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        console.print(
            f"[red]unsupported input format:[/red] {input_path.suffix or '(no extension)'}. "
            f"Supported: {', '.join(SUPPORTED_INPUT_EXTENSIONS)}"
        )
        raise typer.Exit(code=2)

    structured: StructuredSource | None = None
    use_preserve = preserve_format and input_path.suffix.lower() in (".docx", ".pdf")
    try:
        if dry_run:
            text = read_document(input_path)
        elif use_preserve:
            structured = read_structured(input_path)
            text = structured.joined_text
        else:
            text = read_document(input_path)
    except (OSError, UnsupportedDocumentFormat) as e:
        console.print(f"[red]cannot read {input_path}:[/red] {e}")
        raise typer.Exit(code=2) from e

    if not text.strip():
        console.print(f"[red]extracted no text from {input_path}[/red]")
        raise typer.Exit(code=2)

    tgt_lang = _coerce_lang(to, flag="--to")

    if from_ is None:
        if dry_run:
            console.print(
                "[yellow]--from omitted in --dry-run: skipping language detection.[/yellow] "
                "Pass --from explicitly to estimate accurately."
            )
            src_lang_str = "?"
        else:
            console.print("[dim]auto-detecting source language…[/dim]")
            anth = anthropic_client(settings)
            src_lang = detect_language(anth, settings, text)
            console.print(f"[dim]detected source: {src_lang}[/dim]")
            src_lang_str = src_lang
    else:
        src_lang_str = _coerce_lang(from_, flag="--from")

    if dry_run:
        est = estimate(text, settings)
        _print_dry_run(est, input_path, src_lang_str, tgt_lang)
        raise typer.Exit(code=0)

    src_lang = cast(LanguageCode, src_lang_str)
    blocks: tuple[str, ...] = (
        tuple(b.text for b in structured.blocks)
        if structured is not None
        else ()
    )
    source = SourceDocument(
        path=input_path, text=text, source_language=src_lang, blocks=blocks
    )
    pipeline = build_default_pipeline(settings)

    states: dict[str, dict[str, Any]] = {}
    for n in _ordered_stage_names():
        states[n] = {"status": "pending"}

    def on_event(event: str, stage_name: str, info: dict[str, Any] | None) -> None:
        s = states.setdefault(stage_name, {"status": "pending"})
        now = time.monotonic()
        if event == "start":
            s["status"] = "running"
            s["started"] = now
            s.pop("ended", None)
            s.pop("current", None)
            s.pop("total", None)
        elif event == "progress" and info is not None:
            s["current"] = info.get("current")
            s["total"] = info.get("total")
        elif event in ("ok", "skipped", "failed"):
            s["status"] = event
            s["ended"] = now
        live.update(_render_progress(states))

    try:
        with Live(_render_progress(states), console=console, refresh_per_second=4) as live:
            state = pipeline.run(source, target_language=tgt_lang, on_event=on_event)
    except PipelineAbort as e:
        console.print(f"[red]pipeline aborted:[/red] {e}")
        raise typer.Exit(code=1) from e

    if state.final_output is None:
        console.print("[red]pipeline produced no final output[/red]")
        raise typer.Exit(code=1)

    text_path = PlainTextSerializer().write(input_path, state.final_output)
    meta_path = MetadataSerializer().write(input_path, state.metadata)

    # Determine whether the format-preserving writer can run: requires structured
    # source + matched block count from the consistency stage. Otherwise fall back
    # to the fresh-document writer.
    can_preserve = (
        structured is not None
        and bool(state.final_output.blocks)
        and len(state.final_output.blocks) == len(structured.blocks)
    )

    # Format-preserving writers may make additional Claude calls to align
    # translated paragraphs back to per-run / per-span fragments. We share the
    # already-constructed Anthropic client.
    anth_client = anthropic_client(settings) if can_preserve else None
    redistribution_warnings: list[str] = []

    docx_path: Path | None = None
    should_write_docx = out_docx if out_docx is not None else (
        input_path.suffix.lower() == ".docx"
    )
    if should_write_docx:
        docx_path = safe_output_path(input_path, tgt_lang, ".docx")
        if can_preserve and structured is not None and structured.kind == "docx":
            DocxFormatPreservingWriter(anth_client, settings).write(
                docx_path,
                structured,
                state.final_output.blocks,
                warnings=redistribution_warnings,
            )
        else:
            DocxWriter().write(docx_path, state.final_output.text)

    pdf_path: Path | None = None
    should_write_pdf = out_pdf if out_pdf is not None else (
        input_path.suffix.lower() == ".pdf"
    )
    if should_write_pdf:
        pdf_path = safe_output_path(input_path, tgt_lang, ".pdf")
        if can_preserve and structured is not None and structured.kind == "pdf":
            PdfFormatPreservingWriter(anth_client, settings).write(
                pdf_path,
                structured,
                state.final_output.blocks,
                warnings=redistribution_warnings,
            )
        else:
            PdfWriter().write(pdf_path, state.final_output.text)

    if redistribution_warnings:
        # Persist into RunMetadata so the YAML sidecar captures them, and surface
        # to the CLI summary below.
        state.metadata.warnings.extend(redistribution_warnings)

    console.rule("[bold green]done")
    console.print(f"translation: [bold]{text_path}[/bold]")
    if docx_path is not None:
        console.print(f"docx:        [bold]{docx_path}[/bold]")
    if pdf_path is not None:
        console.print(f"pdf:         [bold]{pdf_path}[/bold]")
    console.print(f"metadata:    [bold]{meta_path}[/bold]")
    console.print(
        f"cost: ${state.metadata.total_cost_usd:.4f}   "
        f"time: {state.metadata.total_duration_s:.1f}s   "
        f"warnings: {len(state.metadata.warnings)}"
    )
    if state.metadata.warnings:
        for w in state.metadata.warnings:
            console.print(f"  [yellow]•[/yellow] {w}")


def main() -> None:
    """Entry point referenced by ``[project.scripts]`` in pyproject.toml."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
