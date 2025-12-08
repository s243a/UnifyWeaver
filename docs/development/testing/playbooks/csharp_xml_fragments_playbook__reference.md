# C# XML Fragments Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing C# XML fragments playbook:
[playbooks/csharp_xml_fragments_playbook.md](../../../../playbooks/csharp_xml_fragments_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/csharp_xml_fragments_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to stream XML fragments into the C# query runtime using `XmlStreamReader`. The aim is to ensure:
- XML fragments are parsed correctly
- Namespace prefixes are handled properly
- CDATA content is preserved as text
- Nested projections work when enabled

## Inputs & Artifacts
- Playbook file: `playbooks/csharp_xml_fragments_playbook.md`
- Test data: `test_data/test_xml_fragments.txt`
- Generated C# project with `XmlStreamReader`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. .NET SDK 6.0+ installed (`dotnet` available).
3. Test XML fragments file exists.
4. Run all commands from the repository root.

## Key Configuration Options

### Source Declaration
```prolog
:- source(xml, xml_rows, [
    file('test_data/test_xml_fragments.txt'),
    record_format(xml),
    record_separator(line_feed),
    pearltrees(true)
]).

:- dynamic_source(xml_row/1, xml_rows, []).
```

### Important Options

| Option | Description | Default |
|--------|-------------|---------|
| `record_format(xml)` | Parse as XML fragments | Required |
| `record_separator(line_feed)` | LF-delimited fragments | `nul` |
| `pearltrees(true)` | Enable PT/dcterms prefixes | `false` |
| `namespace_prefixes([Uri-Prefix,...])` | Custom namespace mappings | `[]` |
| `treat_cdata(true)` | Preserve CDATA as text | `false` |
| `nested_projection(true)` | Nested dictionaries | `false` |

## Verification

**Expected behavior:**
- `XmlStreamReader` generates correct namespace prefix mappings
- Keys available: local (`title`), qualified (`pt:title`), full URI form
- Attributes accessible via `@prefix:name` (e.g., `@pt:code`)
- CDATA content preserved when `treat_cdata(true)` or `pearltrees(true)`

**Success criteria:**
- XML fragments parse without errors
- Namespace-qualified elements accessible by multiple key forms
- CDATA content extracted as text, not markup
- Nested dictionaries generated when `nested_projection(true)`

## Key Features Tested

1. **XML fragment streaming** - One fragment per delimiter
2. **Namespace prefix handling** - pt, dcterms, custom prefixes
3. **CDATA preservation** - Pearltrees titles as text
4. **Attribute extraction** - Via `@prefix:name` keys
5. **Nested projection** - Optional nested dictionaries

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "Namespace prefix not found" | Missing prefix mapping | Add to `namespace_prefixes` |
| CDATA appears as markup | `treat_cdata` not enabled | Set `pearltrees(true)` or `treat_cdata(true)` |
| Empty results | Wrong delimiter | Check `record_separator` matches file format |
| Nested elements missing | Flat projection | Set `nested_projection(true)` |
| Parse errors | Malformed XML | Validate XML fragments individually |

## Related Material

- Agent-facing playbook: [playbooks/csharp_xml_fragments_playbook.md](../../../../playbooks/csharp_xml_fragments_playbook.md)
- XML source specification: `docs/proposals/xml_source/SPECIFICATION.md`
- C# query target module: `src/unifyweaver/targets/csharp_query_target.pl`
- XML streaming playbook: [playbooks/large_xml_streaming_playbook.md](../../../../playbooks/large_xml_streaming_playbook.md)
