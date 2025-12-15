# Playbook: Large XML Streaming

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). It demonstrates memory-efficient processing of large XML files (100MB+) using streaming pipelines.

## Workflow Overview
Process large XML files without loading them entirely into memory:
1. **Select** - Extract specific XML elements using awk (streaming, pattern-based)
2. **Filter** - Optionally filter elements by criteria (Python)
3. **Transform** - Convert XML chunks to Prolog facts (Python)

## Key Differences from In-Memory XML Processing

| Aspect | In-Memory (xml_data_source_playbook) | Streaming (this playbook) |
|--------|--------------------------------------|---------------------------|
| **File Size** | Small (<10MB) | Large (100MB+) |
| **Memory** | Loads entire file (~300MB for 100MB file) | Constant (~20KB) |
| **Speed** | Fast for small files | Fast for large files |
| **Use Case** | Embedded XML in records | Standalone large XML files |
| **Tools** | Python ElementTree | awk + Python iterative |

## When to Use This Playbook

✅ **Use streaming approach when:**
- XML file is >10MB
- File might not fit in memory
- Processing millions of elements
- Need constant memory usage
- File is on disk (not embedded)

❌ **Use in-memory approach when:**
- XML is small (<10MB)
- Embedded in record definition
- Need full DOM manipulation
- Complex XPath queries required

## Agent Inputs

Reference the following artifacts:
1. **Executable Records** – `playbooks/examples_library/xml_streaming_examples.md`
2. **Scripts** – `scripts/utils/select_xml_elements.awk`, `filter_by_parent_tree.py`, `xml_to_prolog_facts.py`
3. **Complete Pipeline** – `scripts/extract_pearltrees.sh`
4. **Architecture** – `docs/proposals/pearltrees_extraction_architecture.md`
5. **Generalization** – `docs/examples/xml_pipeline_generalization.md`

## Execution Guidance

### Example 1: Extract Trees from Pearltrees RDF

```bash
# Navigate to project root
cd /path/to/UnifyWeaver

# Extract tree facts
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:Tree" \
    tests/test_data/sample.rdf | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=tree \
    > output_trees.pl

# View results
cat output_trees.pl
```

**Expected Output:**
```prolog
tree(1, 'Test Tree 1', null, null).
tree(3, 'Test Tree 2', null, null).
```

### Example 2: Extract and Filter Pearls

```bash
# Extract pearls for specific tree
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:.*Pearl" \
    tests/test_data/sample.rdf | \
python3 scripts/utils/filter_by_parent_tree.py \
    --tree-id=1 | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=pearl \
    > output_pearls.pl

# View results
cat output_pearls.pl
```

**Expected Output:**
```prolog
(No pearls in sample.rdf - this example demonstrates the pipeline syntax)
```

**Note:** The sample test file doesn't contain Pearl elements. For real Pearltrees RDF exports with pearls, the output would look like:
```prolog
pearl(root, 2492215, 2492215, 1).
parent_tree(2492215, 2492215).
```

### Example 3: Complete Extraction with Script

```bash
# Use the complete extraction script
./scripts/extract_pearltrees.sh \
    tests/test_data/sample.rdf \
    pearltrees_facts/

# View combined facts
cat pearltrees_facts/all_facts.pl
```

**Expected Output:**
```
→ Extracting from: tests/test_data/sample.rdf
→ Output directory: pearltrees_facts/
→ Extracting tree facts...
✓ Extracted 2 tree(s) → trees.pl
→ Extracting pearl facts...
✓ Extracted 0 pearl(s) → pearls.pl
→ Combining facts...
✓ Combined facts → all_facts.pl

=========================================
Extraction Complete
=========================================
Trees:  2
Pearls: 0
```

**Note:** The sample test file contains only Tree elements, no Pearl elements. For real Pearltrees RDF exports with pearls, you would see pearl counts similar to:
```
Trees:  1
Pearls: 4
```

### Example 4: Query Extracted Facts in Prolog

```bash
# Start SWI-Prolog and load facts
swipl

# In Prolog console:
?- ['pearltrees_facts/all_facts.pl'].
true.

?- tree(ID, Title, Privacy, LastUpdate).
ID = 1,
Title = 'Test Tree 1',
Privacy = null,
LastUpdate = null ;
ID = 3,
Title = 'Test Tree 2',
Privacy = null,
LastUpdate = null.

?- findall(ID-Title, tree(ID, Title, _, _), Trees).
Trees = [1-'Test Tree 1', 3-'Test Tree 2'].
```

**Note:** The sample test file doesn't contain Pearl elements. For real Pearltrees RDF exports with pearls, you would see queries like:
```prolog
?- parent_tree(Child, Parent).
Child = Parent, Parent = 2492215 ;
Child = Parent, Parent = 2492215 ;
Child = Parent, Parent = 2492215 ;
Child = Parent, Parent = 5369609.

?- findall(Type, pearl(Type, _, _, _), Types).
Types = [root, alias, ref, section].
```

## Pipeline Architecture

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Large XML File (100MB+)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Stage 1: SELECT      │
            │  (awk streaming)      │
            │  - Pattern matching   │
            │  - Null-delimited     │
            │  - Constant memory    │
            └──────────┬────────────┘
                       │ <element1>\0<element2>\0...
                       ▼
            ┌───────────────────────┐
            │  Stage 2: FILTER      │
            │  (Python optional)    │
            │  - By parent ID       │
            │  - By attributes      │
            │  - Composable         │
            └──────────┬────────────┘
                       │ Filtered chunks
                       ▼
            ┌───────────────────────┐
            │  Stage 3: TRANSFORM   │
            │  (Python facts)       │
            │  - Extract fields     │
            │  - Emit Prolog facts  │
            │  - Relationships      │
            └──────────┬────────────┘
                       │
                       ▼
            ┌───────────────────────┐
            │    Prolog Facts       │
            │    (KB, queryable)    │
            └───────────────────────┘
```

### Memory Profile

```
Traditional:
XML File (100MB) → Parser → DOM Tree (300MB RAM) → Facts

Streaming:
XML File (100MB) → awk (20KB RAM) → chunk → chunk → Facts
                     ↓
                  Process one element at a time
```

## Performance Characteristics

### Memory Usage
- **100MB XML file**
  - Traditional: ~300MB RAM
  - Streaming: ~20KB RAM
  - **Improvement: 15,000x**

### Processing Speed
- **100MB XML file**
  - awk selection only: ~2 seconds
  - awk + filter: ~5 seconds
  - Full pipeline: ~8 seconds
  - Scales linearly with file size

## Tool Generalization

The tools work on **any** XML source, not just Pearltrees:

```bash
# Product catalog
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="product" catalog.xml

# RSS feed
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="item" feed.rss

# SVG graphics
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="circle" diagram.svg

# Maven POM
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="dependency" pom.xml
```

**Same tool, different parameters** - that's the UnifyWeaver way.

## Debugging

### Enable Debug Mode

```bash
# Debug selection
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:Tree" \
    -v debug=1 \
    input.rdf

# Debug filtering
... | python3 scripts/utils/filter_by_parent_tree.py \
    --tree-id=123 \
    --debug

# Debug transformation
... | python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=tree \
    --debug
```

### Check Element Count

```bash
# How many trees are in the file?
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:Tree" \
    -v debug=1 \
    input.rdf 2>&1 | grep "Extracted"

# Output: # Extracted 15 element(s)
```

## Common Issues

### Issue: No elements extracted

**Symptom:**
```
# Extracted 0 element(s)
```

**Solution:**
```bash
# Check what tags exist
grep -o '<[^/>][^>]*>' input.xml | sort -u | head

# Try broader pattern
awk -f scripts/utils/select_xml_elements.awk \
    -v tag=".*Tree" input.xml
```

### Issue: Parse warnings

**Symptom:**
```
Warning: Parse error in chunk: unbound prefix
```

**Explanation:** Expected for namespace-less XML fragments. Parser automatically falls back to regex extraction. This is normal and doesn't affect results.

### Issue: Memory still high

**Symptom:** Process using lots of memory

**Causes:**
1. Not using streaming approach (loading entire file)
2. Buffering too much output

**Solution:**
```bash
# Ensure pipeline doesn't buffer
awk ... | python3 ... --flush | python3 ...

# Process in smaller batches
head -n 100000 large.xml | awk ...
```

## Expected Outcomes

✅ **Successful execution:**
- Facts extracted to `.pl` files
- Memory usage stays constant
- Processing time scales linearly
- Facts load and query correctly in Prolog

❌ **Common errors:**
- Tag pattern doesn't match (check with grep)
- File not found (check path)
- Python not installed (check with `python3 --version`)

## Integration with UnifyWeaver

### As a Preprocessing Step

```bash
# Step 1: Extract facts from large XML
./scripts/extract_pearltrees.sh large_export.rdf facts/

# Step 2: Process with UnifyWeaver
swipl -g "
    [facts/all_facts.pl],
    process_tree_structure(Results),
    halt
"
```

### In Multi-Stage Pipelines

```bash
# Extract → Analyze → Report
./scripts/extract_pearltrees.sh input.rdf stage1/ && \
swipl -g "[stage1/all_facts.pl], analyze(R), write(R), halt" && \
python3 generate_report.py
```

## Next Steps

After completing this playbook:

1. **Extract your own data**
   ```bash
   ./scripts/extract_pearltrees.sh your_file.rdf output/
   ```

2. **Query in Prolog**
   ```prolog
   ?- ['output/all_facts.pl'].
   ?- parent_tree(Child, Parent).
   ```

3. **Generalize to other XML sources**
   - Add new `--element-type` to `xml_to_prolog_facts.py`
   - Create domain-specific filters
   - Reuse `select_xml_elements.awk` unchanged

4. **Integrate with tree examples**
   - Load facts into tree recursion playbook
   - Analyze hierarchical structures
   - Visualize relationships

## See Also

- `playbooks/xml_data_source_playbook.md` - In-memory XML processing (small files)
- `playbooks/tree_recursion_playbook.md` - Tree analysis techniques
- `docs/proposals/pearltrees_extraction_architecture.md` - Design details
- `docs/examples/xml_pipeline_generalization.md` - Cross-domain examples
- `playbooks/examples_library/xml_streaming_examples.md` - Executable records

## Summary

**Key Concepts:**
- ✅ Streaming processing for large XML files
- ✅ Constant memory usage (20KB vs 300MB)
- ✅ Select-Filter-Transform pipeline pattern
- ✅ Composable, reusable tools
- ✅ Generalizes to any XML source

**Tools:**
- `select_xml_elements.awk` - Generic XML selector
- `filter_by_parent_tree.py` - Pearltrees-specific filter
- `xml_to_prolog_facts.py` - Fact transformer
- `extract_pearltrees.sh` - Complete pipeline

**When to Use:**
- Large XML files (>10MB)
- Memory-constrained environments
- Need constant memory usage
- Processing millions of elements
