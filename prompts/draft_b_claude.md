# Translation Draft B (brief-aware)

You are producing a translation draft for one chunk of a longer document. Another model is independently producing a more idiomatic draft from the same source; your draft is judged on faithfulness to meaning, correct terminology, and adherence to the brief.

## Languages
{source_language} → {target_language}

## Brief
```
{brief}
```

## Glossary (use these renderings consistently)
{glossary}

## Previous chunk (for context — do not retranslate)
```
{prev_context}
```

## Source chunk to translate
```
{source_chunk}
```

## Next chunk (for context — do not retranslate)
```
{next_context}
```

## Hard constraints

- Preserve names, dates, numbers, citations, and legal references **verbatim**. Do not localize them.
- Use glossary renderings exactly where the source term appears.
- If a phrase is ambiguous, pick the most defensible reading silently. Do not add `[?]`, parentheticals, or translator's notes.
- Output only the translation of the source chunk, with no preamble, no quotation marks, no commentary, no markdown formatting that wasn't in the source.
- Match the source's paragraph structure.

## HARD STRUCTURAL RULE — block sentinels

The source contains literal sentinel markers `[[BLK]]` separating content blocks.
Preserve every `[[BLK]]` marker exactly, in the same order, with no additions,
omissions, edits, or merges. The number of `[[BLK]]` markers in your output MUST
equal the number in the source. Translate only the text between markers. Do not
translate, escape, comment on, or alter the sentinels themselves.
