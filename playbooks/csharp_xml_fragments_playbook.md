<!--
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# C# XML Fragments Playbook

This playbook shows how to stream XML fragments into the C# query runtime using `XmlStreamReader`.

## Inputs
- `record_format(xml)` with fragments separated by NUL or LF
- Optional: `pearltrees(true)` to set `pt`/`dcterms` prefixes and keep CDATA text
- Optional: `namespace_prefixes([Uri-Prefix,...])`, `treat_cdata(true|false)`, `nested_projection(true)` for nested dictionaries

## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "csharp_xml_fragments" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use csharp xml fragments"
```

## Steps (Prolog)
```prolog
:- source(xml, xml_rows, [
    file('test_data/test_xml_fragments.txt'),
    record_format(xml),
    record_separator(line_feed),
    pearltrees(true)         % default prefixes and CDATA handling
]).

:- dynamic_source(xml_row/1, xml_rows, []).
```

## What the C# plan emits
- `XmlStreamReader` with `NamespacePrefixes` and `TreatPearltreesCDataAsText = true`
- Keys for local, qualified, and prefix forms (e.g., `title`, `pt:title`, `http://...# : title`)
- Attributes under `@prefix:name` (e.g., `@pt:code`)

## Notes
- Keep one fragment per delimiter (NUL or newline) for streaming.
- CDATA content (e.g., pearltrees titles) is preserved as text.
- Set `nested_projection(true)` if you want nested dictionaries instead of a flat map.
