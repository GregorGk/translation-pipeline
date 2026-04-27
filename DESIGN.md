# Translation Pipeline — Design

## Goal

Fully automated, no-human-in-the-loop translation that approaches the quality of a competent human translator's first draft (and on technical/legal content, often exceeds it). The pipeline trades latency and cost for quality. There is no "fast mode" — every translation runs the full pipeline.

## Core insight

Single-pass translation, even with the best LLM, leaves quality on the table. The pipeline uses **multi-pass refinement plus multi-model adversarial validation**:

- Multiple independent drafts surface disagreements; disagreements are signal.
- Cross-model critique exploits the fact that different model families have different blind spots.
- Back-translation against the source catches meaning drift that fluency-focused critique misses.
- A final consistency sweep enforces hard invariants (glossary, names, numbers, citations).

## Languages

All eight target languages are supported by DeepL's API: EN, PT-BR, PL, FR, DE, RU, UK, EL. Every pair therefore has a real DeepL Draft A. No fallback branch is required.

DeepL language code mapping (used internally by `DraftAStage`):

| Internal | DeepL source | DeepL target |
|---|---|---|
| EN | EN | EN-US (default) or EN-GB |
| PT-BR | PT | PT-BR |
| PL | PL | PL |
| FR | FR | FR |
| DE | DE | DE |
| RU | RU | RU |
| UK | UK | UK |
| EL | EL | EL |

## Pipeline stages

```
SOURCE
  │
  ▼
[1] Brief extraction        Claude   → TranslationBrief (doc type, register,
  │                                    glossary candidates, cultural notes,
  │                                    target audience)
  ▼
[2] Chunking                local    → ~1500-token chunks, 200-token overlap,
  │                                    paragraph-aware boundaries
  ▼
[3] Draft A                 DeepL    → idiomatic baseline
[3] Draft B                 Claude   → brief-aware, contextual
  │                                    (executed in parallel)
  ▼
[4] Synthesis               Claude   → merged best-of-both, glossary-enforced
  │
  ▼
[5] Critique                GPT-5    → structured CritiqueIssue list
  │                                    (accuracy, fluency, terminology,
  │                                    register, idiom, cultural)
  ▼
[6] Improvement             Claude   → accepts/rejects each issue with
  │                                    reasoning, produces revised text
  ▼
[7] Back-translation        GPT-5    → revision → source language
  │
  ▼
[8] Divergence detection    Claude   → diff source vs back-translation,
  │                                    flag meaning-level divergences,
  │                                    one improvement loop on flagged
  │                                    segments only
  ▼
[9] Consistency sweep       Claude   → glossary, names, dates, numbers,
  │                                    citations, no formatting artifacts
  ▼
FINAL TRANSLATION + METADATA
```

## Tool role rationale

| Stage | Tool | Why |
|---|---|---|
| Brief extraction | Claude | Long context, reliable structured output, strong cross-lingual reasoning |
| Draft A | DeepL | Strong idiomatic baseline; trained specifically for translation |
| Draft B | Claude | Brief-aware, can use full document context, follows complex instructions |
| Synthesis | Claude | Best at instruction-following merges with explicit criteria |
| Critique | GPT-5 | Different model family → different blind spots from Claude |
| Improvement | Claude | Applies fixes coherently; consistent voice across revisions |
| Back-translation | GPT-5 | Independence from translation-side model is the whole point |
| Divergence + Consistency | Claude | Diff reasoning + invariant enforcement |

Same-family models share blind spots. The hard rule: **the critic is never the translator**.

## Supporting techniques

**Chunking with overlap.** Long documents split into ~1500-token chunks with 200-token overlapping context, paragraph-aware boundaries, then stitched. Prevents context-window degradation on 30-page documents.

**Few-shot anchoring.** Each stage prompt includes 2–3 high-quality example translations in the relevant domain. Particularly effective for legal and technical registers.

**Structured critique with rubric.** Critique output is a list of `CritiqueIssue` objects, not prose. Each issue has category, severity, location, and a suggested fix. This is mechanically actionable in the improvement stage and avoids the "everything looks fine" failure mode of free-form critique.

**Selective re-improvement.** After back-translation, only divergent segments loop through improvement again. Avoids regressing already-good content.

## Quality philosophy

**Names, dates, numbers, citations, legal references are preserved verbatim.** "Art. 282 k.k." stays "Art. 282 k.k." Case numbers, party names, monetary amounts, dates — all unchanged. Optional explanatory glossary in the metadata sidecar, never inline. This is non-negotiable for legal use.

**Uncertainty is silent in prose, loud in metadata.** When the pipeline makes a judgment call (idiom rendered non-literally, ambiguous term, cultural reference adapted), the prose stays clean — no `[?: alternative]` markers, no parentheticals. The judgment is recorded in the metadata `warnings` array so it can be audited but doesn't pollute the deliverable.

**Failure mode is never silent degradation.** Critical stages (brief extraction, both drafts, synthesis) abort the run on persistent failure with a clear error. Non-critical stages (critique, back-translation, divergence detection) skip with a warning logged to metadata, so the user always gets *something* and always knows what was skipped. Each stage retries 3× with exponential backoff before failing.

## What is intentionally not in v1

- **DOCX support** — read/write deferred to v1.5 (text-only), formatting preservation deferred to v2 only if needed.
- **Translation memory / cross-document caching** — architectural hooks left in place; not implemented.
- **Side-by-side diff output** — plain text + YAML metadata only.
- **Web UI** — library/CLI architecture leaves this as a later add-on, not a rewrite.
- **Batch mode** — one document at a time.
- **Domain auto-detection beyond brief extraction** — the brief stage identifies document type, but there are no per-domain prompt variants in v1; tuning happens by editing the single prompt set.

## Realistic limits

For highly literary or culturally dense source material, this pipeline still falls short of a great human translator. For legal correspondence, technical documentation, and business communication, it is at parity or better.

The pipeline is not a replacement for legal review. It produces a high-quality first draft. For documents with legal effect, a qualified human should still read the output before it leaves your hands.

## Cost and latency expectations

Per 5-page document: roughly $0.30–$1.50 in API costs, 1–3 minutes wall-clock time. A 30-page document scales roughly linearly: ~$2–$9, ~6–18 minutes. The `--dry-run` flag estimates both before any API calls.

## Architecture invariants

- Library core, thin CLI wrapper. The CLI imports from `translation_pipeline` and does nothing the library can't do programmatically.
- Every inter-stage object is a Pydantic model. No dicts-of-dicts.
- Every stage implements the same `PipelineStage` Protocol: typed input → typed output, plus a metadata-recording side effect.
- Prompts live in versioned files (`prompts/*.md` or similar), hashed into the run metadata so a translation can be reproduced or compared.
- API keys come from `.env` only, via pydantic-settings. Never hardcoded, never logged.
- Per-stage retry/fallback policy is declared, not scattered through the code.
