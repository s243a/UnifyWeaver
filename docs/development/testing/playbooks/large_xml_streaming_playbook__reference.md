# Large XML Streaming Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing large XML streaming playbook:
[playbooks/large_xml_streaming_playbook.md](../../../../playbooks/large_xml_streaming_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/large_xml_streaming_playbook.md
```

## Purpose

This document validates UnifyWeaver's memory-efficient processing of large XML files (100MB+) using streaming pipelines. The aim is to ensure:
- XML elements are extracted with constant memory usage
- Three-stage pipeline (SELECT → FILTER → TRANSFORM) works correctly
- Prolog facts are generated correctly
- Performance scales linearly with file size

## Inputs & Artifacts
- Playbook file: `playbooks/large_xml_streaming_playbook.md`
- Example records: `playbooks/examples_library/xml_streaming_examples.md`
- Test data: `context/PT/Example_pearltrees_rdf_export.rdf`
- Scripts:
  - `scripts/utils/select_xml_elements.awk`
  - `scripts/utils/filter_by_parent_tree.py`
  - `scripts/utils/xml_to_prolog_facts.py`
  - `scripts/extract_pearltrees.sh`
- Output directory: `pearltrees_facts/`

## Prerequisites
1. GNU AWK (gawk) installed.
2. Python 3.8+ installed.
3. Test RDF file exists in `context/PT/`.
4. Run all commands from the repository root.

## Execution Steps

### Example 1: Extract Trees

```bash
cd /path/to/UnifyWeaver

awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:Tree" \
    context/PT/Example_pearltrees_rdf_export.rdf | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=tree \
    > output_trees.pl

cat output_trees.pl
```

### Example 2: Extract and Filter Pearls

```bash
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:.*Pearl" \
    context/PT/Example_pearltrees_rdf_export.rdf | \
python3 scripts/utils/filter_by_parent_tree.py \
    --tree-id=2492215 | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=pearl \
    > output_pearls.pl
```

### Example 3: Complete Pipeline

```bash
./scripts/extract_pearltrees.sh \
    context/PT/Example_pearltrees_rdf_export.rdf \
    pearltrees_facts/

cat pearltrees_facts/all_facts.pl
```

## Verification

**Expected output for trees:**
```prolog
tree(2492215, 'Hacktivism', 0, '2011-03-14T19:00:12').
```

**Expected output for pearls:**
```prolog
pearl(root, 2492215, 2492215, 1).
parent_tree(2492215, 2492215).
pearl(alias, 2492215, 2492215, 2).
parent_tree(2492215, 2492215).
pearl(ref, 2492215, 2492215, 4).
parent_tree(2492215, 2492215).
```

**Complete pipeline output:**
```
→ Extracting from: context/PT/Example_pearltrees_rdf_export.rdf
→ Output directory: pearltrees_facts/
→ Extracting tree facts...
✓ Extracted 1 tree(s) → trees.pl
→ Extracting pearl facts...
✓ Extracted 4 pearl(s) → pearls.pl
→ Combining facts...
✓ Combined facts → all_facts.pl

=========================================
Extraction Complete
=========================================
Trees:  1
Pearls: 4
```

**Success criteria:**
- Scripts exit with status code 0
- Tree and pearl facts generated correctly
- Memory usage stays constant (~20KB vs 300MB)
- Facts load and query correctly in Prolog

## Key Features Tested

1. **Streaming extraction** with awk (constant memory)
2. **Pattern-based selection** (regex tag matching)
3. **Python filtering** by parent tree ID
4. **Prolog fact generation** from XML elements
5. **Complete pipeline orchestration**

## Performance Characteristics

| File Size | Traditional Memory | Streaming Memory | Speedup |
|-----------|-------------------|------------------|---------|
| 100MB | ~300MB RAM | ~20KB RAM | 15,000x |

| Operation | Time (100MB file) |
|-----------|-------------------|
| awk selection only | ~2 seconds |
| awk + filter | ~5 seconds |
| Full pipeline | ~8 seconds |

## Comparison with In-Memory XML Processing

| Aspect | In-Memory (`xml_data_source_playbook`) | Streaming (this playbook) |
|--------|----------------------------------------|---------------------------|
| File Size | Small (<10MB) | Large (100MB+) |
| Memory | Loads entire file | Constant (~20KB) |
| Use Case | Embedded XML in records | Standalone large XML files |
| Tools | Python ElementTree | awk + Python iterative |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "Extracted 0 element(s)" | Tag pattern doesn't match | Check with `grep -o '<[^/>][^>]*>'` |
| "Warning: Parse error" | Namespace-less fragments | Normal; parser falls back to regex |
| High memory usage | Not using streaming | Ensure pipeline doesn't buffer |
| Python not found | Wrong Python version | Use `python3` explicitly |
| Empty output | File path wrong | Verify test RDF file exists |

## Related Material

- Agent-facing playbook: [playbooks/large_xml_streaming_playbook.md](../../../../playbooks/large_xml_streaming_playbook.md)
- Example records: `playbooks/examples_library/xml_streaming_examples.md`
- Architecture docs: `docs/proposals/pearltrees_extraction_architecture.md`
- Generalization guide: `docs/examples/xml_pipeline_generalization.md`
- In-memory XML playbook: [playbooks/xml_data_source_playbook.md](../../../../playbooks/xml_data_source_playbook.md)
- Tree recursion playbook: [playbooks/tree_recursion_playbook.md](../../../../playbooks/tree_recursion_playbook.md)
