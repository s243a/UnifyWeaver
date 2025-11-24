<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->

# C# XML Dynamic Sources

This is a quick note for agent users of the C# query target.

## Minimal recipe

```prolog
:- source(xml, xml_rows, [
    file('test_data/test_xml_fragments.txt'),
    record_format(xml),        % important: tells the C# target to use XmlStreamReader
    record_separator(line_feed) % or nul if your fragments are NUL-delimited
]).

:- dynamic_source(xml_row/1, xml_rows, []).
```

The generated plan uses `XmlStreamReader` which:
- Splits on `\n` or `\0`
- Projects each fragment into a dictionary keyed by local name, fully-qualified name, and prefix form (e.g., `pt:id`)
- Includes attributes with an `@` prefix (e.g., `@lang`, `@pt:code`)

## Notes and quirks
- CDATA content is preserved as text (e.g., `<![CDATA[Hacktivism]]>` → `Hacktivism`).
- If your fragments carry namespaces (Pearltrees uses `pt:`), both local and `prefix:local` keys are emitted.
- For stream-friendly files, keep one fragment per delimiter (NUL or newline). Avoid multi-line fragments when using `record_separator(line_feed)`.

## Future: Pearltrees helper
Pearltrees RDF exports use `pt:` plus CDATA-wrapped titles. A specialized helper could subclass the XML reader to apply Pearltrees defaults (prefix map, CDATA handling) automatically. For now, the generic reader already preserves CDATA and prefix keys; set `record_format(xml)` and a suitable delimiter. 
