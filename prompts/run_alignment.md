# Run-level alignment

Realign translated paragraph text back to source-language sub-units that
must keep their own formatting.

The source paragraph was originally split into N text fragments (each one is
a Word run / PDF span with distinct styling — bold, italic, font, color, etc.).
The full paragraph has already been translated as a single unit; we now need
to know how the translated text divides among the N original fragments so
the translated output can keep the original formatting.

## Source fragments (in document order)
{numbered_source_fragments}

## Full translated paragraph (target language, as one block)
```
{translated_block}
```

## Your task

Use the `submit_alignment` tool to return exactly N translated fragments,
in the same order as the source fragments. Together they must concatenate
into approximately the full translated paragraph. Empty fragments are
allowed when the source unit was whitespace-only or punctuation-only.

Hard rules:

- Output count = source count exactly.
- Do not invent text that isn't in the translated paragraph.
- Names, dates, numbers, citations: keep verbatim where they were in the
  source.
- If a source fragment is just whitespace or punctuation, output the same
  whitespace or punctuation verbatim.
- Preserve the meaning each source fragment carried — if a source word is
  styled (bold, italic, hyperlink), the corresponding translated fragment
  should be the translation of that word, not a different one.
