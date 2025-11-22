# Pearltrees Extraction Design

**Status:** Planning
**Date:** 2025-11-21
**Branch:** feature/pearltrees-extraction
**Author:** John William Creighton (@s243a)
**Co-Author:** Claude Code (Sonnet 4.5)

---

## Executive Summary

Design document for extracting parent-child relationships from Pearltrees RDF exports and integrating with UnifyWeaver's existing graph/tree examples.

**Key Constraint:** Pearltrees RDF files can be **very large** (MBs to GBs), requiring streaming/incremental processing with memory management.

**Key Requirement:** Extract tree structure for use with classic Prolog graph algorithms (ancestor, descendant, reachability).

---

## Problem Statement

### Pearltrees RDF Structure

From `context/PT/Example_pearltrees_rdf_export.rdf`:

```xml
<pt:Tree rdf:about="https://www.pearltrees.com/t/hacktivism/id2492215">
   <dcterms:title><![CDATA[Hacktivism]]></dcterms:title>
   <pt:treeId>2492215</pt:treeId>
   ...
</pt:Tree>

<pt:RootPearl rdf:about="https://.../id2492215#rootPearl">
   <pt:parentTree rdf:resource="https://.../id2492215" />
   <pt:posOrder>1</pt:posOrder>
</pt:RootPearl>

<pt:AliasPearl rdf:about="https://.../id2492215#item18110176...">
   <pt:parentTree rdf:resource="https://.../id2492215" />
   <pt:posOrder>2</pt:posOrder>
</pt:AliasPearl>
```

**Key Elements:**
- **Tree ID**: Extracted from URL `id######` → `2492215`
- **Parent Relationship**: `<pt:parentTree rdf:resource="..."/>` points to parent tree
- **Pearl Types**: RootPearl, AliasPearl, RefPearl, SectionPearl (children of trees)
- **Order**: `pt:posOrder` gives position within parent

### What We Need to Extract

**Target output:**
```prolog
% Tree facts
tree(2492215, 'Hacktivism').

% Parent-child relationships
parent(2492215, pearl_18110176).  % Tree → Pearl
parent(2492215, pearl_18077284).

% For use with classic Prolog:
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

descendant(X, Y) :- ancestor(Y, X).
```

### Memory Constraint

**Problem:** Large Pearltrees exports (user's real data may be 100MB+)

**Requirement:** Process incrementally, clear memory after each tree/pearl processed

**Cannot:** Load entire RDF file into memory at once

---

## Design Goals

1. **Streaming Processing**: Handle arbitrarily large RDF files
2. **Memory Efficient**: Process tree-by-tree, clear memory between items
3. **Correct Extraction**: Accurately extract tree IDs and parent relationships
4. **Integration**: Output compatible with existing Prolog graph examples
5. **Reusable**: General enough for other RDF processing tasks

---

## Approach Analysis

### Approach A: Null-Delimited XML Stream (Pipeline) ⭐ NEW RECOMMENDATION

**How it works:**
```bash
# Step 1: Split RDF file into individual tree chunks
awk '/<pt:Tree/,/<\/pt:Tree>/ {print; if (/<\/pt:Tree>/) print "\0"}' \
    pearltrees.rdf | \

# Step 2: Process each tree chunk
while IFS= read -r -d '' tree_xml; do
    # Parse this tree's XML
    # Extract facts
    # Clear memory (automatic - each iteration)
done
```

**Or in Prolog/Python pipeline:**
```prolog
% Pre-process: Extract tree chunks with null delimiter
extract_tree_chunks(RDFFile, ChunkStream) :-
    % Split by <pt:Tree>...</pt:Tree>
    % Emit each chunk followed by \0
    ...

% Process stream
process_tree_stream(ChunkStream) :-
    % Read chunk until \0
    read_until_null(ChunkStream, TreeXML),
    (   TreeXML \= end_of_file
    ->  parse_tree_chunk(TreeXML, Facts),
        emit_facts(Facts),
        % Memory cleared automatically
        process_tree_stream(ChunkStream)
    ;   true
    ).
```

**Pros:**
- ✅ **Aligns with UnifyWeaver patterns** (null-delimited JSON streaming)
- ✅ True streaming (constant memory)
- ✅ Simple pipeline architecture
- ✅ Works with bash, Prolog, Python, any target
- ✅ Each tree is self-contained XML (easier to parse)
- ✅ Can use awk/sed/grep for pre-processing
- ✅ Familiar pattern (matches existing examples)
- ✅ Memory cleared automatically per iteration

**Cons:**
- ❌ Two-stage process (split + process)
- ❌ Temporary storage or pipe needed

**Implementation:**
```bash
# Stage 1: Bash splits RDF into null-delimited tree chunks
cat pearltrees.rdf | extract_trees.sh | \

# Stage 2: Python/Prolog processes each chunk
python process_tree.py > facts.pl
```

**Verdict:** ✅ **BEST - Matches UnifyWeaver philosophy, multi-target support**

---

### Approach B: SWI-Prolog semweb with Streaming ⚠️

**How it works:**
```prolog
:- use_module(library(semweb/rdf_db)).
:- use_module(library(semweb/rdf_ntriples)).

% Stream RDF triples, process one at a time
process_rdf_stream(File) :-
    rdf_load(File, [format(xml)]),  % Loads into RDF DB
    % Process triples
    forall(
        rdf(Subject, Predicate, Object),
        process_triple(Subject, Predicate, Object)
    ),
    rdf_unload(File).  % Clear memory
```

**Pros:**
- ✅ Built-in to SWI-Prolog
- ✅ SPARQL query support
- ✅ Handles RDF/XML parsing

**Cons:**
- ❌ `rdf_load` loads **entire file into memory** (RDF triple store)
- ❌ Not truly streaming for large files
- ❌ Memory footprint grows with file size

**Verdict:** ⚠️ Works for small-medium files, problematic for large exports

---

### Approach B: SAX-style XML Parsing (Streaming) ⭐ Recommended

**How it works:**
```prolog
:- use_module(library(sgml)).

% Stream XML events, process incrementally
process_pearltrees_stream(File) :-
    setup_call_cleanup(
        open(File, read, Stream, [encoding(utf8)]),
        new_sgml_parser(Parser, []),
        close(Stream)
    ),
    set_sgml_parser(Parser, space(remove)),
    sgml_parse(Parser, [
        source(Stream),
        call(begin, on_begin),
        call(end, on_end)
    ]).

% Event handlers
on_begin(tree, Attrs, _) :-
    % Extract tree ID from 'about' attribute
    member('rdf:about'=URL, Attrs),
    extract_tree_id(URL, TreeID),
    % Store temporarily
    assertz(current_tree(TreeID)).

on_end(tree, _) :-
    retractall(current_tree(_)).  % Clear memory
```

**Pros:**
- ✅ True streaming (constant memory regardless of file size)
- ✅ Process tree-by-tree
- ✅ Built-in to SWI-Prolog (sgml library)
- ✅ Can emit facts incrementally

**Cons:**
- ❌ More complex code (event handlers)
- ❌ Must track state manually
- ❌ No SPARQL queries

**Verdict:** ✅ **Best for large files**

---

### Approach C: Python Streaming + Dynamic Source 🐍

**How it works:**
```prolog
:- source(python, extract_pearltrees, [
    python_inline("
import xml.etree.ElementTree as ET

def extract_pearltrees(filename):
    # Incremental XML parsing
    context = ET.iterparse(filename, events=('start', 'end'))

    for event, elem in context:
        if event == 'end' and 'Tree' in elem.tag:
            tree_id = extract_id(elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about'))
            title = elem.find('.//{http://purl.org/dc/elements/1.1/}title').text

            yield {'tree_id': tree_id, 'title': title}

            # Clear memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
")
]).
```

**Pros:**
- ✅ Python's iterparse is true streaming
- ✅ Memory-efficient
- ✅ Familiar to Python developers
- ✅ Can use lxml for speed

**Cons:**
- ❌ External Python dependency
- ❌ Context switching (Prolog ↔ Python)
- ❌ Serialization overhead

**Verdict:** ✅ Good alternative if Python already used

---

### Approach D: Partitioning (Existing UnifyWeaver Pattern) 🎯

**Leverage existing partitioning system:**

```prolog
:- use_module(library(unifyweaver/partitioner)).
:- use_module(library(unifyweaver/partitioners/stream_based)).

% Partition RDF file by tree elements
partition_rdf_file(File, Partitions) :-
    % Use stream-based partitioner with custom delimiter
    % Delimiter: <pt:Tree ... > ... </pt:Tree>
    partitioner_init(
        stream_based(delimiter('<pt:Tree', '</pt:Tree>')),
        [],
        Handle
    ),

    % Read file, partition into chunks
    partitioner_partition(Handle, File, Partitions),

    partitioner_cleanup(Handle).

% Process each partition independently
process_partition(TreeXML, Facts) :-
    % Parse just this tree's XML
    % Extract facts
    % Clear memory when done
    ...
```

**Pros:**
- ✅ Uses existing UnifyWeaver infrastructure
- ✅ Partitioning already handles memory management
- ✅ Can parallelize processing
- ✅ Fits UnifyWeaver philosophy

**Cons:**
- ❌ Need stream-based partitioner (may not exist yet)
- ❌ XML boundaries might not align with partition boundaries
- ❌ More complex setup

**Verdict:** 🤔 Interesting, aligns with UnifyWeaver patterns, investigate

---

## Recommended Approach: Pipeline with Null-Delimited Streaming

**Phase 1: Null-Delimited XML Pipeline (Approach A)** ⭐
- Aligns with UnifyWeaver's existing patterns
- Multi-target support (bash, Python, Prolog)
- Simple two-stage: split → process
- Matches null-delimited JSON streaming pattern

**Phase 2: Optimize Splitting (Optional)**
- Use existing partitioning infrastructure if available
- Or keep simple bash/awk approach
- Benchmark different splitters

**Phase 3: Multi-Target Implementations**
- Python version (lxml.etree for XML parsing)
- Prolog version (sgml library for XML parsing)
- C# version (System.Xml for XML parsing)
- Demonstrates orchestration across targets

---

## Detailed Design: Null-Delimited Pipeline Approach

### Stage 1: Split RDF into Tree Chunks

**Bash implementation (simple, works everywhere):**
```bash
#!/bin/bash
# extract_trees.sh - Split RDF file into null-delimited tree chunks

# Match from <pt:Tree to </pt:Tree>, emit chunk + null char
awk '
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

**Usage:**
```bash
bash extract_trees.sh pearltrees.rdf > tree_chunks.stream
# Or pipe directly:
bash extract_trees.sh pearltrees.rdf | process_trees.py
```

**Also works for Pearls:**
```bash
# Extract all pearl elements
awk '/<pt:.*Pearl/,/<\/pt:.*Pearl>/ {print; if (/<\/pt:.*Pearl>/) print "\0"}'
```

### Stage 2: Process Each Tree Chunk

**Python implementation (using lxml for XML):**
```python
#!/usr/bin/env python3
# process_trees.py - Parse tree chunks and emit Prolog facts

import sys
from lxml import etree

# Namespaces
NS = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'pt': 'http://www.pearltrees.com/rdf/0.1/#',
    'dcterms': 'http://purl.org/dc/elements/1.1/'
}

def extract_tree_id(url):
    """Extract tree ID from URL: .../id2492215 -> 2492215"""
    import re
    match = re.search(r'id(\d+)', url)
    return int(match.group(1)) if match else None

def process_tree_chunk(xml_chunk):
    """Parse a single tree chunk and emit facts"""
    try:
        root = etree.fromstring(xml_chunk.encode('utf-8'))

        # Extract tree info
        tree_url = root.get('{%s}about' % NS['rdf'])
        tree_id = extract_tree_id(tree_url)

        # Get title
        title_elem = root.find('.//dcterms:title', NS)
        title = title_elem.text if title_elem is not None else ''

        # Emit fact
        print(f"tree({tree_id}, {repr(title)}).")

        # Now find all pearls with this tree as parent
        # (Note: Pearls are in separate elements, handle in separate stage)

    except Exception as e:
        print(f"% Error parsing chunk: {e}", file=sys.stderr)

def main():
    buffer = ""
    for char in iter(lambda: sys.stdin.read(1), ''):
        if char == '\0':
            if buffer.strip():
                process_tree_chunk(buffer)
                buffer = ""  # Clear memory
        else:
            buffer += char

if __name__ == '__main__':
    main()
```

**Prolog implementation (using sgml library):**
```prolog
#!/usr/bin/env swipl

:- use_module(library(sgml)).

% Process null-delimited stream
process_tree_stream :-
    read_until_null(TreeXML),
    (   TreeXML \= end_of_file
    ->  parse_tree_chunk(TreeXML),
        process_tree_stream  % Memory cleared by tail recursion
    ;   true
    ).

% Read until null character
read_until_null(Content) :-
    read_stream_to_codes(user_input, Codes, [stop_before("\0")]),
    (   Codes = []
    ->  Content = end_of_file
    ;   atom_codes(Content, Codes),
        get_char(_)  % Consume the \0
    ).

% Parse XML chunk
parse_tree_chunk(XMLString) :-
    setup_call_cleanup(
        atom_to_memory_file(XMLString, MemFile),
        (   open_memory_file(MemFile, read, Stream),
            load_structure(Stream, [Tree], [dialect(xmlns)]),
            extract_tree_facts(Tree)
        ),
        free_memory_file(MemFile)
    ).

% Extract facts from parsed XML
extract_tree_facts(element(_, Attrs, _Content)) :-
    member('rdf:about'=URL, Attrs),
    extract_tree_id(URL, TreeID),
    % ... extract title and emit facts
    format('tree(~w, ~q).~n', [TreeID, Title]).

:- initialization(process_tree_stream, main).
```

### Pipeline Architecture

```
┌──────────────────────────────────────┐
│  Input: Large RDF File (100MB+)     │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  Stage 1: Split into Tree Chunks    │
│  (bash/awk - lightweight)            │
│  Output: Null-delimited stream       │
└───────────────┬──────────────────────┘
                │ Tree1\0Tree2\0Tree3\0...
                ▼
┌──────────────────────────────────────┐
│  Stage 2: Process Each Chunk         │
│  (Python/Prolog - parse XML)         │
│  - Read until \0                     │
│  - Parse XML                         │
│  - Extract facts                     │
│  - Clear buffer (automatic)          │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  Output: Prolog Facts                │
│  tree(2492215, 'Hacktivism').        │
│  parent(2492215, pearl_18110176).    │
└──────────────────────────────────────┘
```

### Memory Analysis

**Stage 1 (awk):**
- Memory: ~1KB per tree (buffer for current tree)
- Constant regardless of file size
- O(1) memory

**Stage 2 (Python/Prolog):**
- Memory: ~5-10KB per tree (XML parsing + fact generation)
- Cleared automatically after each tree processed
- O(1) memory (not O(n) where n = file size)

**Total pipeline memory:** < 20KB constant overhead

**Comparison:**
- Loading entire file: 100MB → ~300MB memory
- Pipeline approach: 100MB → ~20KB memory
- **15,000x improvement** 🎉

---

## Detailed Design: SAX-style Streaming (Alternative)

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Pearltrees RDF File (large, 100MB+)           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  SGML Parser (streaming, event-driven)         │
│  - on_begin(tree, ...) → Start tree context    │
│  - on_end(tree, ...) → Emit facts, clear       │
│  - on_begin(pearl, ...) → Extract parent       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  State Tracking (minimal, per-tree)            │
│  - current_tree(TreeID)                        │
│  - current_title(Title)                        │
│  - Clear after each tree                       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Output: Prolog Facts (incremental)            │
│  tree(ID, Title).                              │
│  parent(ParentID, ChildID).                    │
└─────────────────────────────────────────────────┘
```

### State Management

**Minimal state per tree:**
```prolog
:- dynamic current_tree/2.      % current_tree(TreeID, Title)
:- dynamic pending_pearls/1.    % List of pearls in current tree

% Clear state after processing each tree
clear_tree_state :-
    retractall(current_tree(_, _)),
    retractall(pending_pearls(_)).
```

### ID Extraction

```prolog
% Extract tree ID from URL
% Input: 'https://www.pearltrees.com/t/hacktivism/id2492215'
% Output: 2492215
extract_tree_id(URL, TreeID) :-
    atom_string(URL, URLString),
    % Find 'id' followed by digits
    re_matchsub('id(?<id>\\d+)', URLString, Match, []),
    get_dict(id, Match, IDString),
    atom_number(IDString, TreeID).

% Extract pearl ID from URL
% Input: 'https://.../id2492215#item18110176...'
% Output: pearl_18110176
extract_pearl_id(URL, PearlID) :-
    atom_string(URL, URLString),
    % Match item number after '#item'
    (   re_matchsub('#item(?<item>\\d+)', URLString, Match, [])
    ->  get_dict(item, Match, ItemString),
        atom_concat('pearl_', ItemString, PearlID)
    ;   % Fallback: hash of full URL
        term_hash(URL, Hash),
        atom_concat('pearl_', Hash, PearlID)
    ).
```

### Event Handlers

```prolog
% Start of tree element
on_begin(tree, Attrs, _Parser) :-
    member('rdf:about'=URL, Attrs),
    extract_tree_id(URL, TreeID),
    assertz(current_tree(TreeID, '')).  % Title filled later

% Tree title element
on_begin(title, _Attrs, _Parser) :-
    % Will capture CDATA in on_cdata/2
    true.

on_cdata(Title, _Parser) :-
    current_tree(TreeID, ''),
    retract(current_tree(TreeID, '')),
    assertz(current_tree(TreeID, Title)).

% End of tree - emit facts and clear
on_end(tree, _Parser) :-
    current_tree(TreeID, Title),
    format('tree(~w, ~q).~n', [TreeID, Title]),
    clear_tree_state.

% Pearl with parent relationship
on_begin(pearl, Attrs, _Parser) :-
    member('rdf:about'=ChildURL, Attrs),
    extract_pearl_id(ChildURL, ChildID),
    assertz(current_pearl(ChildID)).

on_begin(parentTree, Attrs, _Parser) :-
    current_pearl(ChildID),
    member('rdf:resource'=ParentURL, Attrs),
    extract_tree_id(ParentURL, ParentID),
    format('parent(~w, ~w).~n', [ParentID, ChildID]).

on_end(pearl, _Parser) :-
    retractall(current_pearl(_)).
```

---

## Integration with Graph Examples

### Output Format

**Generated facts:**
```prolog
% Trees
tree(2492215, 'Hacktivism').
tree(2421304, 'Anonymous Culture').

% Parent relationships
parent(2492215, pearl_18110176).
parent(2492215, pearl_18077284).
parent(2421304, pearl_12345678).
```

### Usage with Classic Prolog

```prolog
% Load extracted facts
:- consult('tmp/pearltrees_facts.pl').

% Classic graph predicates (from existing examples)
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

descendant(X, Y) :- ancestor(Y, X).

reachable(X, Y) :- ancestor(X, Y).

% Query examples
?- ancestor(2492215, pearl_18110176).  % Direct parent
true.

?- findall(D, descendant(D, 2492215), Descendants).  % All descendants
Descendants = [pearl_18110176, pearl_18077284, ...].

?- ancestor(2492215, X).  % What are children of this tree?
X = pearl_18110176 ;
X = pearl_18077284.
```

### Connection to Existing Examples

**Tree recursion** (from tree_recursion_playbook.md):
```prolog
% Count descendants of a tree
count_descendants(TreeID, Count) :-
    findall(1, descendant(_, TreeID), Ones),
    length(Ones, Count).

% Max depth from root
tree_depth(Root, Depth) :-
    findall(D, descendant_depth(Root, _, D), Depths),
    max_list(Depths, Depth).
```

**Graph traversal** (classic examples):
```prolog
% Find all leaf nodes (no children)
leaf_node(Node) :-
    (tree(Node, _) ; parent(_, Node)),  % Is a node
    \+ parent(Node, _).                 % Has no children

% Find path from root to node
path(Root, Node, [Root|Path]) :-
    path_helper(Root, Node, Path).

path_helper(Node, Node, []).
path_helper(Current, Target, [Next|Rest]) :-
    parent(Current, Next),
    path_helper(Next, Target, Rest).
```

---

## Memory Analysis

### Worst Case: Loading Entire File (Approach A)

**File size:** 100 MB RDF/XML

**RDF triple store:**
- ~3x expansion (indexing overhead)
- **Memory usage:** ~300 MB

**Problem:** Exceeds Termux limits, slow on mobile

### Streaming Approach (Approach B/C)

**Active memory:**
- Parser state: ~10 KB
- Current tree context: ~1 KB
- Output buffer: ~100 KB

**Peak memory:** ~1 MB (constant regardless of file size)

**Processing time:**
- Linear in file size
- ~1 second per MB on modest hardware
- 100 MB file: ~100 seconds

---

## Open Questions

1. **Pearl Types**: Should we distinguish RootPearl vs AliasPearl vs RefPearl?
   - **Option A**: Treat all as generic pearls (simpler)
   - **Option B**: Preserve type info (richer data)

2. **ID Format**: Use numeric IDs or preserve URLs?
   - **Option A**: Just numbers (`2492215`) - simpler, matches user description
   - **Option B**: Full URLs - preserves uniqueness, allows dereferencing

3. **Ordering**: Capture `pt:posOrder` for sibling ordering?
   - **Option A**: Yes - allows correct tree reconstruction
   - **Option B**: No - simpler, may not be needed

4. **Incremental Output**: Write facts as we parse or accumulate?
   - **Option A**: Stream to stdout (can pipe to file)
   - **Option B**: Write to file incrementally
   - **Option C**: Accumulate in memory, write at end

5. **Existing Partitioning**: Do we have stream-based partitioning already?
   - **Need to check:** `src/unifyweaver/core/partitioners/`
   - **If yes:** Leverage for Approach D
   - **If no:** Implement or stick with Approach B

---

## Implementation Plan

### Phase 1: Proof of Concept (4-6 hours)
1. Implement SAX-style streaming parser
2. Extract tree IDs and titles
3. Extract parent relationships
4. Test with example file
5. Verify memory usage (< 10 MB constant)

### Phase 2: Integration (2-3 hours)
1. Create playbook with streaming example
2. Show integration with ancestor/descendant
3. Demonstrate with real Pearltrees export
4. Document memory characteristics

### Phase 3: Optimization (3-4 hours)
1. Investigate partitioning integration
2. Benchmark different approaches
3. Add parallel processing (if useful)
4. Polish and document

---

## Decision Points (Need User Input)

Before implementing, please decide:

1. **ID Format**: Numbers (`2492215`) or URLs?
2. **Pearl Types**: Preserve or ignore?
3. **Ordering**: Include `posOrder` or skip?
4. **Output Mode**: Stream or file?
5. **Approach**: Start with B (SAX) or investigate D (Partitioning) first?

---

## Success Criteria

**Functional:**
- ✅ Extracts all parent-child relationships correctly
- ✅ Handles files > 100 MB without memory issues
- ✅ Facts usable with existing graph examples
- ✅ Exit code 0 on success

**Performance:**
- ✅ Constant memory usage (< 10 MB overhead)
- ✅ Linear processing time
- ✅ Works on Termux/Android

**Integration:**
- ✅ Compatible with existing playbook format
- ✅ Uses standard Prolog fact format
- ✅ Examples connect to tree_recursion, graph traversal playbooks

---

**Status:** Planning - Awaiting design decisions before implementation
**Next:** User feedback on approach and decision points
