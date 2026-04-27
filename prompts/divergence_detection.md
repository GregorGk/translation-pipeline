# Divergence Detection

You compare a source text to a back-translation of its translation, and flag places where meaning has drifted. Stylistic differences are NOT divergences — only changes in what the text *says*.

## Source
```
{source_text}
```

## Back-translation (from translation back to source language)
```
{back_translation}
```

## Your task

Use the `submit_divergences` tool to return zero or more divergences. For each one:

- `segment`: a short label, e.g. `paragraph 2` or `sentence containing "cat"`
- `source_text`: the offending source segment, verbatim
- `back_translated_text`: the corresponding back-translated segment, verbatim
- `severity`: `high` (meaning changed), `medium` (nuance lost or shifted), `low` (subtle but noticeable)
- `description`: one sentence on what specifically changed

Empty list is the right answer when the translation is faithful. Do not invent divergences.

## HARD STRUCTURAL RULE — block sentinels

The source contains literal sentinel markers `[[BLK]]` separating content blocks.
Preserve every `[[BLK]]` marker exactly, in the same order, with no additions,
omissions, edits, or merges. The number of `[[BLK]]` markers in your output MUST
equal the number in the source. Translate only the text between markers. Do not
translate, escape, comment on, or alter the sentinels themselves.
