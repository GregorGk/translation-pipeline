# Translation Pipeline — Build Plan

## Instructions for Claude Code

Read this file and `DESIGN.md` before doing anything. Execute **Phase 0** first. **Stop and wait for explicit user confirmation before proceeding to each subsequent phase.** Do not skip ahead. Do not implement multiple phases in one go.

When in doubt about a design decision, refer to `DESIGN.md`. If `DESIGN.md` does not answer the question, ask the user — do not guess.

---

## Project summary

A fully automated, no-human-in-the-loop, top-quality translation pipeline. Multi-pass refinement using DeepL + Claude + GPT-5. Library + CLI architecture. Local execution now, cloud-deployable later.

**Languages:** EN, PT-BR, PL, FR, DE, RU, UK, EL — all DeepL-supported, no fallback branches needed.

**Stack:** Python 3.12+, Typer (CLI), Pydantic v2 (models), pydantic-settings (config), Rich (progress + logging), Tenacity (retries), official `anthropic`, `openai`, `deepl` SDKs, `pyyaml` (metadata), `pytest` + `pytest-asyncio` (tests), `ruff` + `mypy` (dev tooling).

## Locked decisions (do not relitigate)

- Names, dates, numbers, citations, legal references → preserved verbatim in translation. Explanations only in metadata, never inline.
- Uncertainty → silent best call in prose, full audit trail in metadata `warnings` array.
- Failure mode → 3× exponential-backoff retry per stage. Critical stages abort on persistent failure (brief, Draft A, Draft B, synthesis, consistency). Non-critical stages skip with warning logged (critique, back-translation, divergence). Never silent degradation.
- Translation memory → out of scope for v1. Hooks only.
- Glossary → per-document, no cross-document persistence.
- Output → plain text + YAML metadata sidecar, written next to input file.
- DOCX → deferred to v1.5.
- Source language → auto-detected by default, overridable via `--from`.

## Environment variables (in `.env`)

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
DEEPL_API_KEY=
DEEPL_API_PLAN=free   # or "pro"
```

`Settings` validates all four are present at startup and fails fast with a clear error if any are missing or empty. Add `.env` to `.gitignore` immediately. Provide `.env.example` with the variable names but no values.

---

## Phase 0 — Project foundation (~15 min)

**Goal:** A runnable, empty project skeleton with config, logging, and dev tooling working.

Tasks:
1. Create directory layout:
   ```
   pyproject.toml
   .env.example
   .gitignore
   README.md
   DESIGN.md            # already exists
   PLAN.md              # already exists
   src/translation_pipeline/
       __init__.py
       config.py        # Settings via pydantic-settings
       logging.py       # Rich-based logger setup
   tests/
       __init__.py
       test_config.py   # one test verifying Settings loads from .env
   prompts/             # empty for now, populated in Phase 2
   ```
2. `pyproject.toml`: project metadata, dependencies, dev dependencies, ruff config, mypy config, pytest config, `[project.scripts]` entry for `translate` (target added in Phase 4).
3. Dependencies: `anthropic`, `openai`, `deepl`, `pydantic>=2`, `pydantic-settings`, `typer`, `rich`, `tenacity`, `pyyaml`.
4. Dev dependencies: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `types-PyYAML`.
5. `Settings` class loads from `.env`, validates all four env vars, fails fast with a clear error message naming any missing key.
6. Logging: Rich-based, INFO default, DEBUG with `--verbose` (flag plumbed in Phase 4).
7. `.gitignore`: `.env`, `__pycache__/`, `.venv/`, `*.egg-info/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `dist/`, `build/`.
8. `README.md`: brief project description, install instructions (`pip install -e ".[dev]"`), env setup, "see DESIGN.md and PLAN.md for details".
9. Run `pytest`, `ruff check`, `mypy src/` — all must pass cleanly.

**STOP. Show the user the directory tree, the contents of `pyproject.toml`, and the test results. Wait for confirmation before Phase 1.**

---

## Phase 1 — Pipeline contracts (~20 min)

**Goal:** Every typed object the pipeline passes between stages is defined. Orchestration works end-to-end with stub stages. Zero real API calls.

Tasks:
1. `src/translation_pipeline/models.py` — Pydantic models:
   - `LanguageCode` (Enum or Literal of the 8 supported codes)
   - `LanguagePair` (source + target)
   - `SourceDocument` (path, raw text, detected or specified source language)
   - `GlossaryEntry` (source term, target term, optional note)
   - `TranslationBrief` (document_type, register, glossary: list[GlossaryEntry], cultural_notes, target_audience, special_instructions)
   - `Chunk` (index, text, prev_context, next_context)
   - `Draft` (source: "deepl" | "claude", chunks: list[str])
   - `SynthesizedTranslation` (text, chunk_alignments)
   - `CritiqueIssue` (category, severity, location, description, suggested_fix)
   - `Critique` (issues: list[CritiqueIssue], overall_assessment)
   - `RevisedTranslation` (text, issues_addressed, issues_rejected_with_reason)
   - `BackTranslation` (text)
   - `Divergence` (segment, source_text, back_translated_text, severity, description)
   - `FinalOutput` (text, language_pair, brief, glossary_used, warnings)
   - `StageRecord` (name, model, started_at, completed_at, duration_s, input_tokens, output_tokens, cost_usd, status: ok|skipped|failed, error?)
   - `RunMetadata` (run_id, source_path, language_pair, pipeline_version, prompt_hashes, stages: list[StageRecord], total_cost, total_duration, warnings)

2. `src/translation_pipeline/stages/base.py` — `PipelineStage` Protocol with typed `run(input) -> output` and metadata-recording behavior.

3. `src/translation_pipeline/pipeline.py` — `Pipeline` class:
   - Holds list of stages with declared criticality (critical | non_critical).
   - Sequences stages, passing typed output of stage N as input to stage N+1.
   - Wraps each stage in retry logic via Tenacity (3 attempts, exponential backoff).
   - On persistent failure: critical → abort with clear error; non-critical → skip, log warning to metadata, pass previous stage's output through.
   - Accumulates `RunMetadata` across all stages.

4. `src/translation_pipeline/stages/stubs.py` — stub implementation of every stage that returns mock data of the correct type. Used for testing orchestration without API calls.

5. `tests/test_pipeline.py` — end-to-end test using stubs verifies:
   - Stages execute in correct order.
   - Output of each is correctly typed.
   - Metadata accumulates correctly.
   - A simulated failure in a non-critical stage results in skip + warning, not abort.
   - A simulated failure in a critical stage aborts with the right error.

**STOP. Show the user `models.py`, `pipeline.py`, and the passing test output. Wait for confirmation before Phase 2.**

---

## Phase 2 — Stage implementations (~70 min)

**Goal:** Every stage implemented with real API calls, individually unit-tested with recorded fixtures (so test reruns don't burn API credits).

Implement in this order. After each, write its unit test using a recorded fixture before moving on.

Each stage's prompt lives in `prompts/<stage_name>.md` as a versioned, hashed file. The hash goes into `RunMetadata.prompt_hashes`.

1. **`BriefExtractionStage`** (Claude, critical)
   Reads source text, outputs a `TranslationBrief`. Prompt instructs Claude to identify document type, register, candidate glossary terms (with attention to false friends in PL/PT/RU/UK), cultural notes, and target audience. Structured JSON output via Anthropic's tool-use or response format.

2. **`ChunkingStage`** (local, critical)
   Paragraph-aware splitter. Target ~1500 tokens per chunk (use `tiktoken` or a heuristic), 200-token overlap. Edge cases: paragraph longer than chunk size → soft-split on sentence boundaries. Single-paragraph documents → single chunk.

3. **`DraftAStage`** — DeepL (critical)
   Uses official `deepl` Python SDK. Maps internal language codes to DeepL codes (PT-BR target → `PT-BR`, EN target → `EN-US`). Translates chunk-by-chunk. Concatenates results. No glossary for v1 (DeepL glossary support deferred).

4. **`DraftBStage`** — Claude (critical)
   Chunked translation with each chunk's prompt including the brief, candidate glossary, prev/next context, and explicit instructions on preserving names/dates/numbers/citations verbatim.

5. **`SynthesisStage`** — Claude (critical)
   Input: source, Draft A, Draft B, brief. Output: a single best-of-both translation, glossary-enforced. Prompt explicitly tells Claude to pick segment by segment, preferring DeepL for idiomatic phrasing and Claude for context-aware terminology, but always to use the source as the source of truth for meaning.

6. **`CritiqueStage`** — GPT-5 (non-critical)
   Input: source, synthesized translation, brief. Output: structured `Critique` with `CritiqueIssue` list. Rubric: accuracy, fluency, terminology consistency, register match, idiomatic naturalness, cultural appropriateness. Each issue has category, severity (low|medium|high), location (chunk index + character span if possible), description, suggested fix. Uses OpenAI structured outputs.

7. **`ImprovementStage`** — Claude (critical if critique succeeded; otherwise skipped)
   Input: synthesized translation + critique. For each issue: accept (apply fix) or reject (with reason). Produces a `RevisedTranslation` recording all decisions. Prompt instructs Claude to reject low-quality critiques rather than blindly applying them.

8. **`BackTranslationStage`** — GPT-5 (non-critical)
   Translates the revised translation back to the source language. Same chunking as forward translation.

9. **`DivergenceDetectionStage`** — Claude (non-critical)
   Diffs source vs back-translation. Flags meaning-level divergences (ignoring stylistic). For each flagged divergence, runs one targeted improvement pass on that segment only.

10. **`ConsistencyStage`** — Claude (critical)
    Final sweep on the full text. Verifies: glossary terms used consistently throughout, names/dates/numbers/citations match source verbatim, no formatting artifacts (stray markers from chunking, doubled paragraph breaks, etc.). Outputs `FinalOutput`.

Cost tracking: every API response's token usage and computed cost recorded in the corresponding `StageRecord`.

**STOP. Run the full pipeline end-to-end on the user's sample document. Show the user the final translation, the metadata, and total cost. Wait for confirmation and feedback before Phase 3.**

---

## Phase 3 — Output serializers (~20 min)

**Goal:** Final translation + metadata written to disk in the user-specified location.

Tasks:
1. `src/translation_pipeline/serializers.py`:
   - `PlainTextSerializer`: writes `FinalOutput.text` to `{stem}.{target_lang}.txt` (e.g., `zawiadomienie.en.txt`) in the same directory as the source.
   - `MetadataSerializer`: writes full `RunMetadata` as YAML to `{stem}.{target_lang}.meta.yaml`.
2. YAML format: human-readable, with the `warnings` array prominent at the top so users see flagged spots first.
3. Filename collision: append `-2`, `-3`, etc. if file exists. Never overwrite silently.

**No stop point — proceed directly to Phase 4 unless something unexpected comes up.**

---

## Phase 4 — CLI (~20 min)

**Goal:** A `translate` command users actually run.

Tasks:
1. `src/translation_pipeline/cli.py` — Typer app.
2. Command: `translate INPUT_PATH --to LANG [--from LANG] [--verbose] [--dry-run]`.
3. If `--from` omitted, run a small Claude call to detect source language from the first ~500 chars of the document.
4. Validate language codes against the supported set; clear error on unsupported.
5. Rich progress bar showing each stage's status (running, ok, skipped, failed) with elapsed time.
6. `--dry-run`: count source tokens, estimate per-stage token usage and cost, print summary, do not call APIs.
7. Clear error messages for: missing env vars, file not found, unreadable file, unsupported language, API failures.
8. `pyproject.toml` `[project.scripts]` exposes `translate = "translation_pipeline.cli:app"` so `pip install -e .` makes it available globally.

**STOP. Demonstrate the CLI to the user with `--help`, `--dry-run`, and a real run on the sample document. Wait for feedback before Phase 5.**

---

## Phase 5 — Validation on real document (~30 min)

**Goal:** Confirm v1 produces translations the user trusts.

Tasks:
1. Run on the user's actual sample document (likely a PL legal document).
2. Read the output and metadata together with the user.
3. Identify the 2–3 prompts that need domain-specific tuning (almost certainly `brief_extraction` and `synthesis` for legal Polish).
4. Apply targeted edits to those prompts. Re-run. Compare.
5. Lock v1.

**No code structure changes in this phase — only prompt iteration.**

---

## Phase 6 — Iteration (out of scope for initial build)

Future short Claude Code sessions to refine prompts based on real-world failures. Not part of the initial build.

---

## Total estimated time

2.5–3 hours of Claude Code session time across Phases 0–5. Can be done in one sitting or split.

## What "done" means for v1

- `translate path/to/file.txt --to en` produces a high-quality translation and a metadata sidecar.
- All 8 languages work in any direction.
- Failures in non-critical stages degrade gracefully, never silently.
- Metadata fully reproduces the run (model versions, prompt hashes, glossary, warnings).
- `pytest`, `ruff check`, `mypy src/` all pass.
- The user has run it on a real document and is satisfied with the output quality.
