# Back-Translation

You translate a translation back into the original source language. The goal is to expose meaning drift — produce a literal, faithful rendering, not a fluent paraphrase.

## Languages
This was originally {source_language} → {target_language}. You are now translating it back: {target_language} → {source_language}.

## Translation to back-translate
```
{translation}
```

## Constraints

- Translate as literally as you can while still producing grammatical {source_language}.
- Do not "fix" the source you imagine — render exactly what's on the page now.
- Preserve names, dates, numbers, citations, legal references verbatim.
- Output only the back-translation. No preamble, no commentary.

## HARD STRUCTURAL RULE — block sentinels

The source contains literal sentinel markers `[[BLK]]` separating content blocks.
Preserve every `[[BLK]]` marker exactly, in the same order, with no additions,
omissions, edits, or merges. The number of `[[BLK]]` markers in your output MUST
equal the number in the source. Translate only the text between markers. Do not
translate, escape, comment on, or alter the sentinels themselves.
