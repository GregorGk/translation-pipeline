# Brief Extraction

You are preparing a translation brief for a translator. Read the source document below and identify what a downstream translator needs to know before doing the work.

## Source language
{source_language}

## Target language
{target_language}

## Source text
```
{source_text}
```

## Your task

Use the `submit_brief` tool to return a structured brief. Be precise and concise; the translator and downstream stages read this as ground truth.

### Field guidance

- **document_type**: short noun phrase, e.g. "Polish criminal-law motion to initiate proceedings", "internal HR policy memo", "user-facing product copy".
- **register_level**: one of `formal`, `neutral`, `informal`, or a short phrase like `formal-legal`. Match the source.
- **glossary**: candidate terms that have a *non-obvious* preferred translation in this domain. Skip generic words. For Slavic ↔ Romance pairs, flag false friends explicitly. Each entry has `source_term`, `target_term`, and an optional `note`.
- **cultural_notes**: anything a translator unfamiliar with the source culture might mishandle (idioms, institutional names, references). One short note per item.
- **target_audience**: who reads the translation, in one phrase.
- **special_instructions**: hard constraints. Always include "preserve names, dates, numbers, citations, and legal references verbatim". Add domain-specific instructions only when warranted.

Names, dates, numbers, citations, and legal references must always be preserved verbatim — make this an explicit instruction in every brief.

## HARD STRUCTURAL RULE — block sentinels

The source contains literal sentinel markers `[[BLK]]` separating content blocks.
Preserve every `[[BLK]]` marker exactly, in the same order, with no additions,
omissions, edits, or merges. The number of `[[BLK]]` markers in your output MUST
equal the number in the source. Translate only the text between markers. Do not
translate, escape, comment on, or alter the sentinels themselves.
