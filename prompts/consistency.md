# Consistency Sweep

Final pass on the full translation. You enforce hard invariants and clean up any artifacts left by chunked processing.

## Languages
{source_language} → {target_language}

## Brief
```
{brief}
```

## Glossary (every entry must appear consistently in the output where the source term applies)
{glossary}

## Source
```
{source_text}
```

## Translation
```
{translation}
```

## Checklist (mentally apply, then return cleaned text)

1. Glossary terms used consistently throughout. If a term has a glossary rendering, every occurrence uses that rendering.
2. Names, dates, numbers, citations, legal references match source verbatim.
3. No formatting artifacts: no stray markers, no doubled paragraph breaks, no leftover `<chunk>` tags or chunk numbers, no quotation marks the source didn't have.
4. Paragraph structure mirrors the source.

## Output

Return ONLY the cleaned translation. No preamble, no explanation, no commentary. If there's nothing to fix, return the input unchanged.

## HARD STRUCTURAL RULE — block sentinels

The source contains literal sentinel markers `[[BLK]]` separating content blocks.
Preserve every `[[BLK]]` marker exactly, in the same order, with no additions,
omissions, edits, or merges. The number of `[[BLK]]` markers in your output MUST
equal the number in the source. Translate only the text between markers. Do not
translate, escape, comment on, or alter the sentinels themselves.
