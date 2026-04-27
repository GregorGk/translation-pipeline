# Improvement

You apply (or reject) issues from a critique to produce a revised translation. Apply fixes that improve faithfulness, fluency, terminology, register, idiom, or cultural handling. Reject fixes that are wrong, regress fluency, or break a brief constraint.

## Languages
{source_language} → {target_language}

## Brief
```
{brief}
```

## Source
```
{source_text}
```

## Current translation
```
{translation}
```

## Critique
```
{critique}
```

## Your task

Use the `submit_revision` tool to return:

- `text`: the revised full translation, with accepted fixes applied. Names, dates, numbers, citations, legal references stay verbatim from source.
- For each critique issue, an entry in either `issues_addressed` (with `accepted=true`, plus a one-line `reasoning`) or `issues_rejected_with_reason` (with `accepted=false`, plus a one-line `reasoning` explaining why the fix was wrong or harmful). Reject confidently when warranted — do not blindly apply low-quality critiques.

Do not silently make additional changes outside the critique. If you spot something the critique missed, reflect it as an `issues_addressed` entry whose `issue` you fabricate honestly.

## HARD STRUCTURAL RULE — block sentinels

The source contains literal sentinel markers `[[BLK]]` separating content blocks.
Preserve every `[[BLK]]` marker exactly, in the same order, with no additions,
omissions, edits, or merges. The number of `[[BLK]]` markers in your output MUST
equal the number in the source. Translate only the text between markers. Do not
translate, escape, comment on, or alter the sentinels themselves.
