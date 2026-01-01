# RDF Data Source Playbook — Reviewer Reference

## Overview
Reviewer's guide for [playbooks/rdf_data_source_playbook.md](../../../../playbooks/rdf_data_source_playbook.md).

## Agent Execution Example
```
Pretend you have fresh context and run the playbook at playbooks/rdf_data_source_playbook.md
```

## Purpose
Validates UnifyWeaver's ability to process RDF data using SWI-Prolog's semweb library, extract graph structures, and integrate with classic Prolog graph algorithms.

## Inputs & Artifacts
- Playbook: `playbooks/rdf_data_source_playbook.md`
- Examples: `playbooks/examples_library/rdf_examples.md`
- Test Data: `context/PT/Example_pearltrees_rdf_export.rdf`
- Generated Prolog: `tmp/pearltrees_analysis.pl`

## Prerequisites
1. SWI-Prolog installed with semweb library
2. Perl for extraction
3. init.pl in project root
4. Run from repository root
5. Example RDF file in context/PT/

## Execution Steps

### Test 1: Pearltrees Tree Extraction

```bash
cd /path/to/UnifyWeaver

# Extract the example
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.pearltrees_extract_tree" \
  playbooks/examples_library/rdf_examples.md \
  > tmp/run_pearltrees_extract.sh

chmod +x tmp/run_pearltrees_extract.sh

# Run
bash tmp/run_pearltrees_extract.sh
```

### Test 2: SPARQL Queries

```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.rdf_graph_queries" \
  playbooks/examples_library/rdf_examples.md \
  > tmp/run_rdf_queries.sh

chmod +x tmp/run_rdf_queries.sh
bash tmp/run_rdf_queries.sh
```

### Test 3: Multi-Format Export

```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.rdf_export_formats" \
  playbooks/examples_library/rdf_examples.md \
  > tmp/run_rdf_export.sh

chmod +x tmp/run_rdf_export.sh
bash tmp/run_rdf_export.sh
```

## Verification

### Test 1: Expected Output

```
=== Pearltrees RDF Tree Structure Extraction ===

✓ Loaded Pearltrees RDF data

Extracting parent-child relationships...
parent('https://www.pearltrees.com/t/hacktivism/id2492215', 'https://www.pearltrees.com/t/hacktivism/id2492215#rootPearl').
parent('https://www.pearltrees.com/t/hacktivism/id2492215', 'https://www.pearltrees.com/t/hacktivism/id2492215#item18110176&show=item,18110176').
...

=== Tree Structure Analysis ===

Found 1 root tree(s)

Tree: https://www.pearltrees.com/t/hacktivism/id2492215
  Title: "Hacktivism"
  Descendants: 3
  Children:
    - Anonymous Culture
    - Hacktivism Political engagements

=== Demonstrating Classic Prolog Queries ===

Ancestors of ...: [...]

Reachability test:
  ✓ ... is ancestor of ...
  ✓ ... is descendant of ...

✓ Analysis complete

Success: Pearltrees tree structure extracted and analyzed
```

**Success criteria:**
- RDF data loads without errors
- Parent-child relationships extracted
- Tree structure correctly identified
- Classic Prolog queries work (ancestor, descendant)
- Exit code 0

### Test 2: Expected Output

```
=== RDF Graph Queries with SPARQL ===

✓ RDF data loaded

=== All Trees (SPARQL Query) ===
Tree: https://www.pearltrees.com/t/hacktivism/id2492215
  Title: "Hacktivism"

=== Converting RDF to Prolog Facts ===

% Tree facts
tree('https://www.pearltrees.com/t/hacktivism/id2492215', 'Hacktivism').

% Parent-child facts
parent('https://www.pearltrees.com/t/hacktivism/id2492215', ...).
...

=== Classic Prolog Patterns ===

% Define ancestor/2 (transitive closure of parent)
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Find tree depth
...

Total nodes in graph: 3

✓ RDF graph queries complete

Success: RDF graph queries demonstrated
```

**Success criteria:**
- SPARQL queries execute successfully
- RDF converted to Prolog facts
- Fact format correct
- Exit code 0

### Test 3: Expected Output

```
=== RDF Export to Multiple Formats ===

Exporting to JSON...
✓ JSON export: tmp/rdf_exports/pearltrees.json

Exporting to CSV...
✓ CSV export: tmp/rdf_exports/pearltrees.csv

Exporting to DOT (GraphViz)...
✓ DOT export: tmp/rdf_exports/pearltrees.dot

✓ All exports complete

Generated files:
  - tmp/rdf_exports/pearltrees.json
  - tmp/rdf_exports/pearltrees.csv
  - tmp/rdf_exports/pearltrees.dot (visualize with: dot -Tpng pearltrees.dot -o pearltrees.png)

Success: RDF data exported to multiple formats
```

**Success criteria:**
- All three files created
- JSON valid
- CSV properly formatted
- DOT valid GraphViz syntax
- Exit code 0

**Verify files exist:**
```bash
ls -lh tmp/rdf_exports/
# Should show:
# pearltrees.json
# pearltrees.csv
# pearltrees.dot
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `library(semweb/rdf_db) not found` | semweb not installed | Install: `swipl-pack install semweb` |
| `rdf_load` fails | Invalid RDF/XML | Check XML syntax |
| No parent-child relationships | Wrong namespace | Verify pt: namespace URL |
| Empty queries | No matching triples | Check SPARQL query syntax |
| JSON export fails | json library missing | Should be built-in, check SWI version |

## Integration Tests

### Test with Different RDF Files

The playbook should work with any Pearltrees RDF export:

```bash
# Try with different export file
cp context/PT/pearltrees_export.rdf context/PT/Example_pearltrees_rdf_export.rdf
bash tmp/run_pearltrees_extract.sh
```

### Visualize with GraphViz

If GraphViz installed:

```bash
bash tmp/run_rdf_export.sh
dot -Tpng tmp/rdf_exports/pearltrees.dot -o tmp/rdf_exports/pearltrees.png
open tmp/rdf_exports/pearltrees.png  # macOS
# or: xdg-open tmp/rdf_exports/pearltrees.png  # Linux
```

### Query Custom SPARQL

Modify the SPARQL query in the generated Prolog:

```prolog
% In tmp/rdf_sparql_demo.pl
Query = 'PREFIX pt: <http://www.pearltrees.com/rdf/0.1/#>
         SELECT ?pearl ?title ?parent
         WHERE {
           ?pearl pt:parentTree ?parent .
           ?pearl dcterms:title ?title
         }',
```

## Related Material
- Playbook: [playbooks/rdf_data_source_playbook.md](../../../../playbooks/rdf_data_source_playbook.md)
- Examples: `playbooks/examples_library/rdf_examples.md`
- SWI-Prolog semweb: https://www.swi-prolog.org/pldoc/man?section=semweb
- Pearltrees RDF info: http://www.pearltrees.com/

## Educational Value

This playbook demonstrates:

1. **Real-world RDF processing** - Not toy examples, actual semantic web data
2. **Prolog's graph strength** - Natural for hierarchical data
3. **Classic algorithms** - Ancestor, descendant, reachability on real data
4. **Data transformation** - RDF → Prolog facts → Multiple export formats
5. **Integration** - Connects to existing tree/graph recursion examples

Perfect for teaching:
- Semantic web basics
- Graph algorithms
- Prolog's practical applications
- Data processing pipelines
