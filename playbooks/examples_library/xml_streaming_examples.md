# XML Streaming Examples

**Purpose:** Executable records demonstrating memory-efficient processing of large XML files using streaming pipelines.

**Target Audience:** Coding agents and developers working with large XML data sources (100MB+).

---

## Record: unifyweaver.execution.xml_streaming_extract_trees

**Purpose:** Extract tree facts from Pearltrees RDF export using streaming pipeline.

**Type:** Bash script

**Memory:** Constant ~20KB (vs ~300MB for in-memory parsing)

```bash
#!/bin/bash
# Extract tree facts from Pearltrees RDF using streaming approach

# Configuration
INPUT_FILE="context/PT/Example_pearltrees_rdf_export.rdf"
OUTPUT_FILE="output_trees_streaming.pl"

echo "=== XML Streaming: Extract Trees ==="
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Pipeline: Select trees → Transform to facts
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:Tree" \
    "$INPUT_FILE" | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=tree \
    > "$OUTPUT_FILE"

echo "Extraction complete!"
echo ""
echo "=== Extracted Facts ==="
cat "$OUTPUT_FILE"

echo ""
echo "=== Verify in Prolog ==="
echo "To query:"
echo "  swipl"
echo "  ?- ['$OUTPUT_FILE']."
echo "  ?- tree(ID, Title, Privacy, LastUpdate)."
```

**Expected Output:**
```
=== XML Streaming: Extract Trees ===
Input: context/PT/Example_pearltrees_rdf_export.rdf
Output: output_trees_streaming.pl

Extraction complete!

=== Extracted Facts ===
tree(2492215, 'Hacktivism', 0, '2011-03-14T19:00:12').

=== Verify in Prolog ===
To query:
  swipl
  ?- ['output_trees_streaming.pl'].
  ?- tree(ID, Title, Privacy, LastUpdate).
```

---

## Record: unifyweaver.execution.xml_streaming_extract_pearls

**Purpose:** Extract pearl facts from Pearltrees RDF export.

**Type:** Bash script

```bash
#!/bin/bash
# Extract pearl facts from Pearltrees RDF using streaming approach

# Configuration
INPUT_FILE="context/PT/Example_pearltrees_rdf_export.rdf"
OUTPUT_FILE="output_pearls_streaming.pl"

echo "=== XML Streaming: Extract Pearls ==="
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Pipeline: Select pearls → Transform to facts
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:.*Pearl" \
    "$INPUT_FILE" | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=pearl \
    > "$OUTPUT_FILE"

echo "Extraction complete!"
echo ""
echo "=== Extracted Facts ==="
cat "$OUTPUT_FILE"

echo ""
echo "=== Query Examples ==="
echo "  ?- pearl(Type, TreeID, ParentID, PosOrder)."
echo "  ?- parent_tree(Child, Parent)."
echo "  ?- findall(Type, pearl(Type, _, _, _), Types)."
```

**Expected Output:**
```
=== XML Streaming: Extract Pearls ===
Input: context/PT/Example_pearltrees_rdf_export.rdf
Output: output_pearls_streaming.pl

Extraction complete!

=== Extracted Facts ===
pearl(root, 2492215, 2492215, 1).
parent_tree(2492215, 2492215).
pearl(alias, 2492215, 2492215, 2).
parent_tree(2492215, 2492215).
pearl(ref, 2492215, 2492215, 4).
parent_tree(2492215, 2492215).
pearl(section, 5369609, 5369609, 59).
parent_tree(5369609, 5369609).

=== Query Examples ===
  ?- pearl(Type, TreeID, ParentID, PosOrder).
  ?- parent_tree(Child, Parent).
  ?- findall(Type, pearl(Type, _, _, _), Types).
```

---

## Record: unifyweaver.execution.xml_streaming_filter_pearls

**Purpose:** Extract pearls for a specific tree using filtering.

**Type:** Bash script

**Demonstrates:** Composable pipeline with filtering stage.

```bash
#!/bin/bash
# Extract pearls for specific tree using filter stage

# Configuration
INPUT_FILE="context/PT/Example_pearltrees_rdf_export.rdf"
OUTPUT_FILE="output_filtered_pearls.pl"
TREE_ID="2492215"

echo "=== XML Streaming: Filter Pearls by Tree ==="
echo "Input: $INPUT_FILE"
echo "Tree ID: $TREE_ID"
echo "Output: $OUTPUT_FILE"
echo ""

# Pipeline: Select pearls → Filter by tree → Transform to facts
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:.*Pearl" \
    "$INPUT_FILE" | \
python3 scripts/utils/filter_by_parent_tree.py \
    --tree-id="$TREE_ID" | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=pearl \
    > "$OUTPUT_FILE"

echo "Extraction complete!"
echo ""
echo "=== Filtered Facts (only tree $TREE_ID) ==="
cat "$OUTPUT_FILE"

echo ""
echo "=== Note ==="
echo "Compare with unfiltered output - this should have fewer pearls"
echo "(Only pearls with parent tree $TREE_ID are included)"
```

**Expected Output:**
```
=== XML Streaming: Filter Pearls by Tree ===
Input: context/PT/Example_pearltrees_rdf_export.rdf
Tree ID: 2492215
Output: output_filtered_pearls.pl

Extraction complete!

=== Filtered Facts (only tree 2492215) ===
pearl(root, 2492215, 2492215, 1).
parent_tree(2492215, 2492215).
pearl(alias, 2492215, 2492215, 2).
parent_tree(2492215, 2492215).
pearl(ref, 2492215, 2492215, 4).
parent_tree(2492215, 2492215).

=== Note ===
Compare with unfiltered output - this should have fewer pearls
(Only pearls with parent tree 2492215 are included)
```

**Comparison:** This outputs 3 pearls instead of 4 (section pearl filtered out).

---

## Record: unifyweaver.execution.xml_streaming_complete_pipeline

**Purpose:** Complete end-to-end extraction using the extraction script.

**Type:** Bash script

**Demonstrates:** Production-ready pipeline with all features.

```bash
#!/bin/bash
# Complete Pearltrees extraction using the extraction script

# Configuration
INPUT_FILE="context/PT/Example_pearltrees_rdf_export.rdf"
OUTPUT_DIR="pearltrees_facts_complete"

echo "=== XML Streaming: Complete Pipeline ==="
echo "Input: $INPUT_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Run complete extraction script
./scripts/extract_pearltrees.sh "$INPUT_FILE" "$OUTPUT_DIR"

echo ""
echo "=== Generated Files ==="
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Combined Facts ==="
cat "$OUTPUT_DIR/all_facts.pl"

echo ""
echo "=== Load in Prolog ==="
echo "  swipl"
echo "  ?- ['$OUTPUT_DIR/all_facts.pl']."
echo "  ?- tree(ID, Title, _, _), pearl(_, ID, _, _)."
```

**Expected Output:**
```
=== XML Streaming: Complete Pipeline ===
Input: context/PT/Example_pearltrees_rdf_export.rdf
Output Directory: pearltrees_facts_complete

→ Extracting from: context/PT/Example_pearltrees_rdf_export.rdf
→ Output directory: pearltrees_facts_complete/
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

Output files:
  - pearltrees_facts_complete/trees.pl
  - pearltrees_facts_complete/pearls.pl
  - pearltrees_facts_complete/all_facts.pl

=== Generated Files ===
total 12K
-rw-r--r-- 1 user user  50 Nov 22 all_facts.pl
-rw-r--r-- 1 user user  25 Nov 22 pearls.pl
-rw-r--r-- 1 user user  25 Nov 22 trees.pl

=== Combined Facts ===
tree(2492215, 'Hacktivism', 0, '2011-03-14T19:00:12').
pearl(root, 2492215, 2492215, 1).
parent_tree(2492215, 2492215).
pearl(alias, 2492215, 2492215, 2).
parent_tree(2492215, 2492215).
pearl(ref, 2492215, 2492215, 4).
parent_tree(2492215, 2492215).
pearl(section, 5369609, 5369609, 59).
parent_tree(5369609, 5369609).

=== Load in Prolog ===
  swipl
  ?- ['pearltrees_facts_complete/all_facts.pl'].
  ?- tree(ID, Title, _, _), pearl(_, ID, _, _).
```

---

## Record: unifyweaver.execution.xml_streaming_query_facts

**Purpose:** Query extracted facts in Prolog to demonstrate analysis.

**Type:** Prolog script

**Prerequisites:** Run `unifyweaver.execution.xml_streaming_complete_pipeline` first.

```prolog
% Query extracted Pearltrees facts

% Load facts (assuming complete pipeline was run)
:- initialization(main, main).

main :-
    writeln('=== XML Streaming: Query Facts ==='),
    writeln(''),

    % Try to load facts
    (   exists_file('pearltrees_facts_complete/all_facts.pl')
    ->  ['pearltrees_facts_complete/all_facts.pl'],
        writeln('✓ Facts loaded'),
        writeln('')
    ;   writeln('✗ Facts not found. Run complete pipeline first.'),
        writeln('  ./scripts/extract_pearltrees.sh context/PT/Example_pearltrees_rdf_export.rdf pearltrees_facts_complete/'),
        halt(1)
    ),

    % Query 1: All trees
    writeln('=== Query 1: All Trees ==='),
    forall(
        tree(ID, Title, Privacy, LastUpdate),
        format('  tree(~w, ~w, privacy=~w, updated=~w)~n', [ID, Title, Privacy, LastUpdate])
    ),
    writeln(''),

    % Query 2: All parent-child relationships
    writeln('=== Query 2: Parent-Child Relationships ==='),
    forall(
        parent_tree(Child, Parent),
        format('  ~w → ~w~n', [Child, Parent])
    ),
    writeln(''),

    % Query 3: Count pearls by type
    writeln('=== Query 3: Pearl Types ==='),
    findall(Type, pearl(Type, _, _, _), Types),
    msort(Types, SortedTypes),
    count_types(SortedTypes, TypeCounts),
    forall(
        member(Type-Count, TypeCounts),
        format('  ~w: ~w pearl(s)~n', [Type, Count])
    ),
    writeln(''),

    % Query 4: Pearls for specific tree
    writeln('=== Query 4: Pearls for Tree 2492215 ==='),
    forall(
        pearl(Type, 2492215, Parent, Pos),
        format('  ~w pearl at position ~w (parent: ~w)~n', [Type, Pos, Parent])
    ),
    writeln(''),

    writeln('=== Queries Complete ==='),
    halt(0).

% Helper predicate to count occurrences
count_types([], []).
count_types([X|Xs], [X-Count|Rest]) :-
    include(=(X), [X|Xs], Matches),
    length(Matches, Count),
    exclude(=(X), Xs, Remaining),
    count_types(Remaining, Rest).
```

**Expected Output:**
```
=== XML Streaming: Query Facts ===

✓ Facts loaded

=== Query 1: All Trees ===
  tree(2492215, 'Hacktivism', privacy=0, updated='2011-03-14T19:00:12')

=== Query 2: Parent-Child Relationships ===
  2492215 → 2492215
  2492215 → 2492215
  2492215 → 2492215
  5369609 → 5369609

=== Query 3: Pearl Types ===
  alias: 1 pearl(s)
  ref: 1 pearl(s)
  root: 1 pearl(s)
  section: 1 pearl(s)

=== Query 4: Pearls for Tree 2492215 ===
  root pearl at position 1 (parent: 2492215)
  alias pearl at position 2 (parent: 2492215)
  ref pearl at position 4 (parent: 2492215)

=== Queries Complete ===
```

---

## Record: unifyweaver.execution.xml_streaming_debug

**Purpose:** Debug mode demonstration for troubleshooting.

**Type:** Bash script

**Demonstrates:** How to use debug flags to understand pipeline behavior.

```bash
#!/bin/bash
# Debug the XML streaming pipeline

INPUT_FILE="context/PT/Example_pearltrees_rdf_export.rdf"

echo "=== XML Streaming: Debug Mode ==="
echo ""

echo "=== Stage 1: Selection (awk debug) ==="
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:.*Pearl" \
    -v debug=1 \
    "$INPUT_FILE" 2>&1 | head -30

echo ""
echo "=== Stage 2: Filtering (Python debug) ==="
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:.*Pearl" \
    "$INPUT_FILE" | \
python3 scripts/utils/filter_by_parent_tree.py \
    --tree-id=2492215 \
    --debug 2>&1 | head -20

echo ""
echo "=== Stage 3: Transformation (Python debug) ==="
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="pt:Tree" \
    "$INPUT_FILE" | \
python3 scripts/utils/xml_to_prolog_facts.py \
    --element-type=tree \
    --debug 2>&1

echo ""
echo "=== Debug Complete ==="
echo "Debug output shows:"
echo "  - Number of elements selected"
echo "  - Filtering decisions"
echo "  - Extraction details"
echo "  - Parse warnings (if any)"
```

**Expected Output:**
```
=== XML Streaming: Debug Mode ===

=== Stage 1: Selection (awk debug) ===
# Selecting elements matching: pt:.*Pearl
# Delimiter: \0 (null)
# Extracted 4 element(s)
<pt:RootPearl rdf:about="https://www.pearltrees.com/t/hacktivism/id2492215#rootPearl">
   <dcterms:title><![CDATA[Hacktivism]]></dcterms:title>
   <pt:parentTree rdf:resource="https://www.pearltrees.com/t/hacktivism/id2492215" />
   <pt:inTreeSinceDate>2011-02-06T16:57:45</pt:inTreeSinceDate>
   <pt:posOrder>1</pt:posOrder>
</pt:RootPearl>
...

=== Stage 2: Filtering (Python debug) ===
# Filtering by parent tree ID: 2492215
# Delimiter: \0 (null)
# Chunk 1: parent_id=2492215
# Chunk 2: parent_id=2492215
# Chunk 3: parent_id=2492215
# Chunk 4: parent_id=5369609
# Matched 3/4 chunks

=== Stage 3: Transformation (Python debug) ===
# Extracting tree facts
% Debug: Extracted tree 2492215: Hacktivism...
# Extracted 1 tree(s)
tree(2492215, 'Hacktivism', 0, '2011-03-14T19:00:12').

=== Debug Complete ===
Debug output shows:
  - Number of elements selected
  - Filtering decisions
  - Extraction details
  - Parse warnings (if any)
```

---

## Usage Instructions

### Extracting Records

```bash
# Extract a specific record
perl scripts/utils/extract_records.pl \
    -f content \
    -q "unifyweaver.execution.xml_streaming_extract_trees" \
    playbooks/examples_library/xml_streaming_examples.md \
    > tmp/extract_trees.sh

# Make executable and run
chmod +x tmp/extract_trees.sh
bash tmp/extract_trees.sh
```

### Running All Examples

```bash
# Run each example in sequence
for record in extract_trees extract_pearls filter_pearls complete_pipeline; do
    echo "Running: xml_streaming_$record"
    perl scripts/utils/extract_records.pl \
        -f content \
        -q "unifyweaver.execution.xml_streaming_$record" \
        playbooks/examples_library/xml_streaming_examples.md | bash
    echo ""
done
```

## See Also

- `playbooks/large_xml_streaming_playbook.md` - Detailed playbook guide
- `docs/proposals/pearltrees_extraction_architecture.md` - Architecture design
- `docs/examples/xml_pipeline_generalization.md` - Cross-domain examples
- `playbooks/xml_data_source_playbook.md` - In-memory XML processing (small files)
