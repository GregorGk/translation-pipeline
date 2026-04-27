# Translation Pipeline

Multi-pass, multi-model document translator. DeepL + Claude + GPT-5.5 work together over 10 stages to produce translations that approach a human first draft. Reads `.txt`, `.docx`, `.pdf`. Writes the same with **formatting preserved** — bold, headings, tables, fonts, page layout, images all carry through to the translated output. Plain-text and a YAML metadata sidecar are emitted alongside.

```bash
translate "Instauração de IP.pdf" --from PT-BR --to PL
```

→ `Instauração de IP.pl.pdf` (same layout, translated text), `Instauração de IP.pl.txt` (clean text), `Instauração de IP.pl.meta.yaml` (audit trail).

---

## Why use this over a single-pass LLM call

| Concern | Bare-LLM single shot | This pipeline |
|---|---|---|
| **Idiomatic phrasing** | Strong | Strong (DeepL's idiomatic baseline + Claude's brief-aware draft, merged) |
| **Glossary / terminology consistency** | Hit or miss | Enforced — brief stage extracts terms, consistency sweep verifies |
| **Names, dates, numbers, citations preserved verbatim** | Often paraphrased | Hard rule — verified by a final consistency pass |
| **Independent review** | None | GPT-5.5 critiques Claude's output (different family = different blind spots) |
| **Meaning drift detection** | None | Back-translation + divergence detection catches cases where wording is fine but meaning shifted |
| **Format preservation** | Variable | Mid-paragraph bold/italic/font preserved at run-level for DOCX, span-level for PDF |
| **Reproducibility** | None | Every run produces a metadata sidecar with prompt hashes, model IDs, per-stage cost & duration |
| **Failure modes** | Silent | Critical stages retry 3× then abort with a clear error; non-critical skip with warnings logged to metadata |

---

## Quick start

```bash
# Install (one time)
pipx install .

# Set credentials (one time)
cp .env.example .env
$EDITOR .env

# Translate
translate document.docx --from PT-BR --to PL
```

That's it. The output `document.pl.docx` is in the same folder as the source, with the same layout and the text translated.

---

## Installation

Requires Python 3.12+ and one of: Anthropic, OpenAI, DeepL accounts.

### Option A: pipx (recommended for daily use)

```bash
pipx install .
```

`translate` is now on your PATH globally.

### Option B: virtual env (recommended for development)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`translate` works inside the activated venv. Dev dependencies (pytest, ruff, mypy) are included.

### Option C: pip user-install

```bash
pip install --user .
# Make sure ~/.local/bin is on your PATH
```

---

## Configuration

The CLI reads credentials from a `.env` file in the current directory. Copy the template:

```bash
cp .env.example .env
```

Then fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPL_API_KEY=...
DEEPL_API_PLAN=pro          # or "free" — determines whether DeepL output is billed
```

All four lines are required. Empty values fail fast with a clear message.

### Override which model runs which stage (optional)

Default model assignments (see [config.py](src/translation_pipeline/config.py)) work well, but you can override any of them via env vars without code changes:

```
MODEL_BRIEF_EXTRACTION=claude-sonnet-4-6
MODEL_DRAFT_B=claude-sonnet-4-6
MODEL_SYNTHESIS=claude-opus-4-7
MODEL_CRITIQUE=gpt-5.5
MODEL_IMPROVEMENT=claude-opus-4-7
MODEL_BACK_TRANSLATION=gpt-5.5
MODEL_DIVERGENCE_DETECTION=claude-sonnet-4-6
MODEL_CONSISTENCY=claude-opus-4-7
MODEL_LANGUAGE_DETECT=claude-haiku-4-5-20251001
```

For example, to push more stages onto Sonnet for cost: `MODEL_SYNTHESIS=claude-sonnet-4-6`.

---

## Usage

### Basic

```bash
translate INPUT --to TARGET_LANG [--from SOURCE_LANG]
```

Source language auto-detects (one short Claude Haiku call) when `--from` is omitted.

```bash
# auto-detect source
translate report.docx --to EN

# explicit source
translate report.docx --from PT-BR --to PL

# any direction between supported languages
translate notice.txt --from PL --to UK
```

### Estimate cost without running

```bash
translate report.docx --from PT-BR --to PL --dry-run
```

Prints a per-stage breakdown of estimated tokens and cost. No API calls. Estimates are heuristic — real cost typically lands within ±30%.

### Format preservation flags

| Flag | Behavior |
|---|---|
| `--preserve-format` *(default for `.docx` / `.pdf`)* | Source document is mutated in place. Bold, italic, fonts, tables, images, page layout all preserved. Run/span-level translation alignment via Claude. |
| `--no-preserve-format` | A fresh `.docx` / `.pdf` is built from scratch. Translation is the same; styling is plain (one paragraph per blank-line-separated block). |
| `--out-docx` | Force a `.docx` output even when input wasn't a DOCX. |
| `--no-out-docx` | Suppress `.docx` output when input was a DOCX. |
| `--out-pdf` | Force a `.pdf` output even when input wasn't a PDF. |
| `--no-out-pdf` | Suppress `.pdf` output when input was a PDF. |

### Other flags

| Flag | Purpose |
|---|---|
| `--verbose` | DEBUG-level logging (HTTP requests, retries, prompts loaded) |
| `--version` | Print version and exit |
| `--help` | Full options |

### Output files

Written next to the source file with the lowercased target-language code:

| Source | Outputs (default) |
|---|---|
| `report.docx` | `report.pl.docx`, `report.pl.txt`, `report.pl.meta.yaml` |
| `report.pdf` | `report.pl.pdf`, `report.pl.txt`, `report.pl.meta.yaml` |
| `report.txt` | `report.pl.txt`, `report.pl.meta.yaml` |

Filename collision: appends `-2`, `-3`, … never silently overwrites.

---

## Supported languages

EN, PT-BR, PL, FR, DE, RU, UK, EL — any direction between them. All supported by DeepL (no fallback branches in the pipeline). The 8-language set was chosen so every pair has a real DeepL idiomatic baseline.

---

## How it works

A 10-stage pipeline with per-stage model assignments. Each stage's prompt lives in `prompts/<stage>.md`, hashed into the output metadata so a translation can be reproduced.

```
SOURCE
  │
  ▼
[1] Brief extraction        Claude Sonnet  → translator's brief (doc type,
  │                                          register, glossary, cultural notes)
  ▼
[2] Chunking                local           → ~1500-token chunks, 200-token
  │                                          overlap, paragraph-aware
  ▼
[3] Draft A                 DeepL           → idiomatic baseline
[3] Draft B                 Claude Sonnet   → brief-aware translation
  │                                          (executed in parallel)
  ▼
[4] Synthesis               Claude Opus     → merged best-of-both,
  │                                          glossary-enforced
  ▼
[5] Critique                GPT-5.5         → structured issue list
  │                                          (different model family)
  ▼
[6] Improvement             Claude Opus     → accepts/rejects each issue
  │                                          with reasoning
  ▼
[7] Back-translation        GPT-5.5         → revision → source language
  │
  ▼
[8] Divergence detection    Claude Sonnet   → diffs source vs back-translation
  │                                          flags meaning drift
  ▼
[9] Consistency sweep       Claude Opus     → glossary, names/dates/numbers,
  │                                          citations, formatting artifacts
  ▼
[10] Format-preserving write
                            Claude Sonnet   → run/span-level alignment
                                              (DOCX/PDF only)
                                              followed by in-place document
                                              mutation
```

Why three model families:

- **DeepL** — purpose-built translation model. Strongest idiomatic baseline.
- **Claude** — long context, follows complex briefs, good at consistency.
- **GPT-5.5** — different family for critique + back-translation. Same-family models share blind spots; the critic is never the translator.

---

## Cost & runtime expectations

Calibrated on real runs against a 22-page Brazilian Portuguese legal document (~10K source tokens):

| Run | Direction | Cost | Time |
|---|---|---:|---:|
| 22-page legal PDF, no-preserve | PT-BR → EN | $3.92 | 18 min |
| 22-page legal PDF, format-preserved | PT-BR → PL | $4.77 | 26 min |
| 7-paragraph DOCX with bold + table, format-preserved | PT-BR → PL | $0.32 | 2 min |
| 5-paragraph DOCX with mid-paragraph bold/italic | PT-BR → PL | $0.19 | 1.5 min |

Costs scale roughly linearly with source token count. Use `--dry-run` to estimate before you commit.

---

## Hard rules — what's preserved verbatim, always

- Names (people, organizations, places — surface form unchanged)
- Dates and times (`15/11/1992`, `28 lutego 2026`)
- Numbers (passport, PESEL, monetary amounts, case file numbers)
- Citations and legal references (`Art. 282 k.k.`, `Article 5, item II of the Code of Criminal Procedure`)

The consistency stage explicitly verifies these against the source and fails the run with a warning if anything drifted.

Uncertainty is silent in prose, loud in metadata: when the pipeline makes a judgment call (idiom rendered non-literally, ambiguous term, cultural reference adapted), the prose stays clean. The judgment is recorded in the metadata `warnings` array so it can be audited but doesn't pollute the deliverable.

---

## What gets skipped on failure

| Stage | Criticality | Failure behavior |
|---|---|---|
| Brief extraction | critical | Pipeline aborts |
| Chunking | critical | Pipeline aborts |
| Draft A (DeepL) | critical | Pipeline aborts |
| Draft B (Claude) | critical | Pipeline aborts |
| Synthesis | critical | Pipeline aborts |
| Critique | non-critical | Skipped, warning logged |
| Improvement | critical (only when critique succeeded) | Pipeline aborts |
| Back-translation | non-critical | Skipped, warning logged |
| Divergence detection | non-critical | Skipped, warning logged |
| Consistency | critical | Pipeline aborts |

Each stage retries 3× with exponential backoff before failing. Format-preserving writers use streaming under the hood (no 10-min HTTP timeouts), and httpx-level transport errors are also retried.

---

## Troubleshooting

**`translate: command not found`** — Either you installed in a venv that isn't activated, or `~/.local/bin` isn't on your PATH. With pipx: `pipx ensurepath`.

**`Failed to load configuration. Ensure ANTHROPIC_API_KEY... are set`** — Your `.env` is missing or one of the three keys is empty. The CLI also accepts environment variables at the shell level. Check with `cat .env` and `env | grep -E "(ANTHROPIC|OPENAI|DEEPL)"`.

**`unsupported input format`** — Only `.txt`, `.docx`, `.pdf` are accepted. Other Office formats can be saved as DOCX from Word/LibreOffice.

**`pipeline aborted: Critical stage 'X' failed`** — A model call failed all 3 retries. Check the output around the abort message — usually rate-limit, quota, or unsupported model ID. Verify with `MODEL_*` env vars.

**Polish/Cyrillic/Greek characters render as boxes in the output PDF** — Format-preserving PDF needs a Unicode TTF on the host. macOS ships Arial Unicode by default; on Linux install `fonts-dejavu`. The CLI logs a warning at first PDF write if no Unicode font is found.

**Output PDF is huge (10–20 MB) compared to source** — The Unicode font is embedded on every page when format-preservation is on. PDF compression is already at maximum (`garbage=4, deflate=True`). To reduce size, use `--no-preserve-format` (rebuilds the PDF via reportlab and embeds the font once).

**Mid-paragraph formatting (bold/italic) was lost in DOCX output** — Should not happen with default settings. The format-preserving writer makes a Claude alignment call per multi-run paragraph. Confirm `--no-preserve-format` was not passed and check the metadata YAML for an `alignment count mismatch` warning.

---

## Development

```bash
# Install with dev tools
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest

# Lint + typecheck
ruff check
mypy src/

# Format-preserving writer integration test on a real DOCX
translate path/to/sample.docx --from PT-BR --to PL --verbose
```

Project layout:

```
src/translation_pipeline/
    cli.py              # Typer entry point
    config.py           # Settings (env-driven)
    pipeline.py         # 10-stage orchestrator with retry + criticality
    models.py           # Pydantic types for every inter-stage object
    documents.py        # TXT/DOCX/PDF readers + format-preserving writers
    chunking.py         # Paragraph-aware chunker
    estimator.py        # --dry-run cost estimator
    pricing.py          # Per-model rate table
    prompts.py          # Versioned prompt loader (hashed into metadata)
    serializers.py      # Plain-text + YAML metadata writers
    language_detect.py  # One-shot Haiku call for source language
    llm.py              # Anthropic + OpenAI helpers (streaming, retry-on-transport)
    clients.py          # SDK client factories
    factory.py          # Wires the default 10-stage pipeline
    stages/
        base.py           # PipelineStage ABC
        brief_extraction.py
        chunking.py
        draft_a_deepl.py
        draft_b_claude.py
        synthesis.py
        critique.py
        improvement.py
        back_translation.py
        divergence_detection.py
        consistency.py
        stubs.py          # Test fixtures
prompts/
    *.md                # One file per LLM-driven stage, version-hashed
tests/
    test_*.py
```

---

## Design notes

For the deeper rationale behind stage sequencing, model placement, and quality philosophy, see [DESIGN.md](DESIGN.md). For the build history and per-phase scope, see [PLAN.md](PLAN.md).

---

## License

Proprietary. PyMuPDF (used for PDF format preservation) is AGPL-3.0 — outputs from this tool inherit AGPL unless a commercial PyMuPDF license is purchased separately.
