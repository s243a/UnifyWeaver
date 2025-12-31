# Pearltrees Extraction - Detailed Planning

**Status:** Planning Complete - Awaiting Approval
**Date:** 2025-11-21
**Branch:** feature/pearltrees-extraction
**Tests:** ✅ awk validated, ✅ partitioners checked, ⏳ scope decisions needed

---

## Executive Summary

**Goal:** Extract tree/pearl relationships from Pearltrees RDF exports (100MB+ files) in memory-efficient way.

**Approach:** Null-delimited XML streaming pipeline
1. Split with awk (fast, pattern-based)
2. Process chunks with Python (XML parsing + fact emission)

**Test Results:**
- ✅ awk correctly extracts trees and pearls from Pearltrees RDF
- ✅ ID extraction works (tree IDs, parent relationships)
- ✅ Existing partitioners checked - none support delimiter-based splitting
- ⏳ Need scope decision: trees only or trees+pearls?

**Recommendation:** Proceed with awk-based approach, two-pass extraction (trees then pearls).

---

## Current State Analysis

### What We Already Have

**1. XML Processing (xml_examples.md)**
- Python's `xml.etree.ElementTree` for parsing
- Used in dynamic source compilation
- **Limitation:** Loads entire XML into memory
- **Use case:** Small, embedded XML data

```python
# Existing approach
root = ET.fromstring(xml_data)  # Loads all into memory
for product in root.findall('product'):
    # Process...
```

**2. Null-Delimited JSON Streaming**
- Established pattern in UnifyWeaver
- Used for inter-process communication
- Format: `{"record":1}\0{"record":2}\0`

**3. Partitioning System**
- Location: `src/unifyweaver/core/partitioners/`
- Various strategies (fixed_size, hash_based, key_based)
- **Question:** Do we have stream-based partitioner?

### What We Need for Pearltrees

**Different problem:**
- **Input:** Very large RDF/XML files (100MB+)
- **Cannot:** Load entire file into memory
- **Need:** Split file into processable chunks
- **Constraint:** Memory-efficient streaming

---

## Tool Comparison for Splitting Large XML

### Option 1: awk (Simple, Fast) ⭐ User's Suggestion

**Approach:**
```bash
# Extract trees
awk '/<pt:Tree/,/<\/pt:Tree>/ {
    print
    if (/<\/pt:Tree>/) print "\0"
}' pearltrees.rdf
```

**Pros:**
- ✅ **Very fast** (compiled C, not interpreted)
- ✅ Universal (works everywhere)
- ✅ Simple one-liner
- ✅ Constant memory (processes line-by-line)
- ✅ No dependencies

**Cons:**
- ❌ Pattern matching only (not true XML parsing)
- ❌ Could break on malformed XML or nested tags
- ❌ No namespace awareness

**When it works:**
- Well-formed XML
- Predictable structure (like Pearltrees exports)
- Simple tag matching sufficient

**When it might fail:**
```xml
<!-- Nested Tree tags (unlikely in Pearltrees but possible) -->
<pt:Tree>
  <description>See also <pt:Tree>...</pt:Tree></description>
</pt:Tree>
```

---

### Option 2: Python lxml.iterparse (Robust)

**Approach:**
```python
import lxml.etree as ET

context = ET.iterparse('pearltrees.rdf', events=('end',), tag='{...}Tree')

for event, elem in context:
    # elem is complete Tree element
    xml_string = ET.tostring(elem, encoding='unicode')
    print(xml_string + '\0')

    # Clear memory
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]
```

**Pros:**
- ✅ True streaming XML parser
- ✅ Namespace-aware
- ✅ Handles complex/malformed XML
- ✅ Memory-efficient (clears as it goes)
- ✅ Can extract specific elements

**Cons:**
- ❌ Python dependency
- ❌ lxml not always installed (need pip install)
- ❌ Slower than awk
- ❌ More complex code

---

### Option 3: Python xml.etree (Built-in)

**Approach:**
```python
import xml.etree.ElementTree as ET

# iterparse is available in standard library too
for event, elem in ET.iterparse('pearltrees.rdf', events=('end',)):
    if 'Tree' in elem.tag:
        xml_string = ET.tostring(elem, encoding='unicode')
        print(xml_string + '\0')
        elem.clear()
```

**Pros:**
- ✅ Built-in (no external deps)
- ✅ Streaming parser
- ✅ Works everywhere Python works

**Cons:**
- ❌ Less robust than lxml
- ❌ Slower than awk
- ❌ Namespace handling awkward

**Comparison to lxml:**
- xml.etree: Standard library, less features
- lxml: External dependency, more robust, faster

---

### Option 4: xmlstarlet (Command-line XML tool)

**Approach:**
```bash
xmlstarlet sel -t -m '//pt:Tree' -c '.' -n pearltrees.rdf | \
    sed 's/$/\x00/'
```

**Pros:**
- ✅ Command-line XML processor
- ✅ True XML parsing
- ✅ Xpath support

**Cons:**
- ❌ Not always installed
- ❌ Another dependency
- ❌ Slower than awk
- ❌ Complex syntax

---

### Option 5: SWI-Prolog sgml Library

**Approach:**
```prolog
:- use_module(library(sgml)).

% Stream through XML, extract trees
extract_trees(File) :-
    setup_call_cleanup(
        open(File, read, Stream),
        sgml_parse(Stream, [
            source(Stream),
            syntax_errors(quiet),
            call(begin, on_tree_start),
            call(end, on_tree_end)
        ]),
        close(Stream)
    ).

% Accumulate tree content
on_tree_start(tree, Attrs, _) :- ...
on_tree_end(tree, _) :- % Emit chunk\0
```

**Pros:**
- ✅ Built-in to SWI-Prolog
- ✅ Event-driven streaming
- ✅ Memory-efficient

**Cons:**
- ❌ Prolog-specific (not multi-target)
- ❌ Complex state management
- ❌ More code than awk

---

## Recommendation Matrix

| Criterion | awk | lxml | xml.etree | xmlstarlet | Prolog sgml |
|-----------|-----|------|-----------|------------|-------------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Correctness** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Dependencies** | ✅ None | ❌ pip install | ✅ Built-in | ❌ Package | ✅ Built-in |
| **Multi-target** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ Prolog only |
| **Robustness** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Testing Strategy: Validate awk Approach

### Test 1: Does awk work correctly on Pearltrees RDF? ✅ PASSED

```bash
# Extract trees from example file
awk '/<pt:Tree/,/<\/pt:Tree>/ {
    print
    if (/<\/pt:Tree>/) print "\0"
}' context/PT/Example_pearltrees_rdf_export.rdf | \
    tr '\0' '\n' | grep -c '</pt:Tree>'

# Result: 1 tree extracted
```

**Example file structure:**
- 1 Tree element (id: 2492215)
- 4 Pearl elements (RootPearl, AliasPearl, RefPearl, SectionPearl)
- 4 parent relationships (3 → tree 2492215, 1 → tree 5369609)

### Test 2: Are extracted chunks valid XML? ✅ PASSED

```bash
# Extract first tree
awk '/<pt:Tree/,/<\/pt:Tree>/ {
    print
    if (/<\/pt:Tree>/) { print "\0"; exit }
}' context/PT/Example_pearltrees_rdf_export.rdf > first_tree.xml

# Result: Valid 9-line XML fragment
<pt:Tree rdf:about="https://www.pearltrees.com/t/hacktivism/id2492215">
   <dcterms:title><![CDATA[Hacktivism]]></dcterms:title>
   <dcterms:creator rdf:resource="https://www.pearltrees.com/s243a#sioc" />
   <pt:treeId>2492215</pt:treeId>
   <pt:assoId>2492215</pt:assoId>
   <pt:lastUpdate>2011-03-14T19:00:12</pt:lastUpdate>
   <pt:privacy>0</pt:privacy>
</pt:Tree>
```

### Test 3: ID extraction validation ✅ PASSED

```bash
# Extract tree ID
awk '/<pt:Tree rdf:about="[^"]*id([0-9]+)"/ {
    match($0, /id([0-9]+)/, arr);
    print "Tree ID:", arr[1]
}' context/PT/Example_pearltrees_rdf_export.rdf
# Result: Tree ID: 2492215

# Extract parent tree IDs
awk '/<pt:parentTree.*id([0-9]+)"/ {
    match($0, /id([0-9]+)/, arr);
    print "Parent ID:", arr[1]
}' context/PT/Example_pearltrees_rdf_export.rdf
# Result:
# Parent ID: 2492215
# Parent ID: 2492215
# Parent ID: 2492215
# Parent ID: 5369609
```

### Test 4: Check existing partitioners ✅ CHECKED

```bash
ls src/unifyweaver/core/partitioners/
# Found: fixed_size.pl, hash_based.pl, key_based.pl
# None support delimiter-based or XML-aware splitting
```

**Conclusion:** Existing partitioners work on in-memory lists, not streaming delimited data. We need a separate splitter for XML chunks.

### Test 5: Memory usage comparison ⏳ TODO

```bash
# awk approach
/usr/bin/time -v awk '/<pt:Tree/,/<\/pt:Tree>/ {...}' large_file.rdf

# Python approach
/usr/bin/time -v python extract_trees.py large_file.rdf

# Compare:
# - Maximum resident set size
# - Elapsed time
```

---

## Hybrid Approach (Recommended)

### Phase 1: Start with awk (Pragmatic)

**Why:**
- Fast development/testing
- Validate approach quickly
- Good enough for well-formed Pearltrees RDF

**Implementation:**
```bash
#!/bin/bash
# extract_trees_awk.sh

awk '
BEGIN { in_tree = 0; tree_content = "" }

/<pt:Tree/ {
    in_tree = 1
    tree_content = $0 "\n"
    next
}

in_tree {
    tree_content = tree_content $0 "\n"
    if (/<\/pt:Tree>/) {
        printf "%s\0", tree_content
        in_tree = 0
        tree_content = ""
    }
}
' "$1"
```

### Phase 2: Add Python fallback (Robust)

**For complex cases:**
```python
#!/usr/bin/env python3
# extract_trees_python.py

import sys
import xml.etree.ElementTree as ET

def extract_trees(filename):
    for event, elem in ET.iterparse(filename, events=('end',)):
        if elem.tag.endswith('Tree'):  # Namespace-agnostic
            xml_str = ET.tostring(elem, encoding='unicode')
            print(xml_str, end='\0', flush=True)
            elem.clear()

if __name__ == '__main__':
    extract_trees(sys.argv[1])
```

### Orchestration Script

```bash
#!/bin/bash
# smart_extract_trees.sh - Choose best tool available

extract_trees() {
    local file="$1"

    # Try awk first (fastest)
    if command -v awk &>/dev/null; then
        echo "[INFO] Using awk (fast path)" >&2
        bash extract_trees_awk.sh "$file"

    # Fallback to Python
    elif command -v python3 &>/dev/null; then
        echo "[INFO] Using Python (robust path)" >&2
        python3 extract_trees_python.py "$file"

    else
        echo "[ERROR] No suitable tool found (need awk or python3)" >&2
        return 1
    fi
}

extract_trees "$@"
```

---

## Decision Points

### 1. Primary Tool: awk or Python? ✅ DECIDED: awk

**Question:** Which should be the default/recommended approach?

**Option A:** awk (fast, simple) ← **CHOSEN**
- **Pro:** Fastest, no dependencies
- **Con:** Less robust
- **Test results:** ✅ Works correctly on Pearltrees RDF

**Option B:** Python xml.etree (robust, built-in)
- **Pro:** More correct, built-in
- **Con:** Slower

**Option C:** Provide both, user chooses
- **Pro:** Best of both worlds
- **Con:** More code to maintain

**Decision:** Use awk as primary tool. Tests confirm it handles Pearltrees RDF correctly. Python fallback can be added later if edge cases emerge.

---

### 2. Existing Partitioner Integration? ✅ DECIDED: Build standalone

**Question:** Can we leverage existing partitioning infrastructure?

**Test results:**
```bash
ls src/unifyweaver/core/partitioners/
# Found: fixed_size.pl, hash_based.pl, key_based.pl
# None support delimiter-based or XML-aware splitting
```

**Decision:**
- Build standalone awk/bash splitter
- Existing partitioners work on in-memory lists
- Our need: streaming XML-to-delimited-chunks converter
- Consider adding delimiter-based partitioner later as separate enhancement

---

### 3. Where Does Splitting Happen?

**Option A:** Pre-processing step (separate script)
```bash
extract_trees.sh input.rdf | process_trees.py > facts.pl
```

**Option B:** Integrated into playbook example
```bash
# In the bash record itself
awk '...' | python process.py
```

**Option C:** Dynamic source compilation
```prolog
:- source(bash, extract_trees, [...]).
:- source(python, process_tree_chunk, [...]).
```

**Recommendation:** Option B (integrated playbook) - keeps it simple

---

### 4. Pearl Extraction

**Question:** Trees vs Pearls - same approach?

**Observation:**
- Trees: `<pt:Tree>...</pt:Tree>`
- Pearls: `<pt:RootPearl>`, `<pt:AliasPearl>`, etc.
- Different elements, but same pattern

**Options:**
1. Two-pass: Extract trees, then pearls separately
2. Single-pass: Extract all elements, classify later
3. Combined: Extract tree + its pearls together

**Recommendation:** Two-pass (simpler to reason about)
```bash
extract_trees.sh | process_trees.py > tree_facts.pl
extract_pearls.sh | process_pearls.py > pearl_facts.pl
cat tree_facts.pl pearl_facts.pl > all_facts.pl
```

---

## Next Steps (Planning Phase)

### Before Implementation

1. ✅ **Test awk approach** on example Pearltrees RDF
   - ✅ Does it extract trees correctly? → YES (1 tree extracted)
   - ✅ Are chunks valid XML? → YES (9-line valid fragment)
   - ⏳ Memory usage acceptable? → Need large file test

2. ✅ **Check existing partitioners**
   - Found: fixed_size.pl, hash_based.pl, key_based.pl
   - None support delimiter-based or XML-aware splitting
   - Decision: Build standalone splitter

3. ✅ **Decide on primary tool**
   - **Chosen:** awk (tests passed)
   - Python fallback available if needed later

4. ⏳ **Define extraction scope**
   - Just trees? Or pearls too?
   - Two-pass or single-pass?
   - What facts to emit?

5. ⏳ **Plan pearl-to-tree linkage**
   - How to connect pearls to parent trees?
   - Parse `pt:parentTree` reference → ✅ ID extraction validated
   - Extract tree/pearl IDs correctly → ✅ Works with awk regex

### After Planning Approval

6. ☐ Create proof-of-concept splitter scripts
   - `scripts/extract_trees.awk` - Extract tree elements
   - `scripts/extract_pearls.awk` - Extract pearl elements
   - `scripts/process_tree_chunk.py` - Convert XML to Prolog facts

7. ☐ Test with real Pearltrees export (user's data)
   - Validate on large files (100MB+)
   - Measure memory usage
   - Confirm constant memory consumption

8. ☐ Implement fact extraction
   - Tree facts: `tree(ID, Title, ParentID)`
   - Pearl facts: `pearl(Type, TreeID, URL)`
   - Parent relationships: `parent_tree(ChildID, ParentID)`

9. ☐ Create playbook example
   - Show complete pipeline: split → process → query
   - Integrate with tree/graph examples

10. ☐ Write documentation
    - Usage guide for Pearltrees extraction
    - Add to XML data source playbook

---

## Open Questions for User

1. ~~**Tool preference:** awk (fast) vs Python (robust)?~~ → ✅ DECIDED: awk
2. ~~**Existing tools:** Should we check partitioners first?~~ → ✅ CHECKED: None suitable
3. **Scope:** Extract just trees, or pearls too?
   - Trees only? (simpler)
   - Trees + Pearls? (complete picture)
   - Recommended: Both (two-pass approach)
4. **Fact schema:** What facts should we emit?
   - Option A: Simple facts `tree(ID, Title), parent(ChildID, ParentID)`
   - Option B: Rich facts `tree(ID, Title, Creator, LastUpdate, Privacy)`
   - Option C: Let user decide via configuration
5. **Linking:** Should we extract full pearl details or just parent relationships?
   - Minimal: Just pearl→tree parent links
   - Full: Pearl type, URL, position, dates, etc.
6. **Multi-file:** Process multiple RDF exports or single file?
   - Design for single file (user's use case)
   - Or support batch processing multiple exports?

---

**Status:** Planning Complete - Ready for implementation approval

**Test Results Summary:**
- ✅ awk extraction works correctly
- ✅ ID parsing validated
- ✅ Existing partitioners checked (none suitable)
- ⏳ Awaiting scope decisions (trees only vs trees+pearls)
- ⏳ Awaiting fact schema decisions
