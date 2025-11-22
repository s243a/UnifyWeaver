# Playbook: RDF Data Source (Semantic Web)

## Audience
This playbook is a high-level guide for coding agents. Agents orchestrate UnifyWeaver to process RDF (Resource Description Framework) data from semantic web sources like Pearltrees exports.

## Workflow Overview
Use UnifyWeaver with SWI-Prolog's semweb library to:
1. Load RDF/XML data into the RDF triple store
2. Extract graph structure (parent-child relationships, hierarchies)
3. Query with SPARQL or native Prolog predicates
4. Convert RDF triples to Prolog facts for use with classic graph algorithms
5. Export to various formats (JSON, CSV, DOT/GraphViz)

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `unifyweaver.execution.pearltrees_extract_tree`, `unifyweaver.execution.rdf_graph_queries`, `unifyweaver.execution.rdf_export_formats` in `playbooks/examples_library/rdf_examples.md`.
2. **Environment Setup Skill** – `skills/skill_unifyweaver_environment.md` explains how to set up the Prolog environment.
3. **Extraction Skill** – `skills/skill_extract_records.md` documents extraction CLI.
4. **Reviewer Reference** – `docs/development/testing/playbooks/rdf_data_source_playbook__reference.md` for validation.

## Execution Guidance

**IMPORTANT**: The records contain **BASH SCRIPTS** that generate and execute Prolog code. Extract and run with `bash`.

### Example: Extract Pearltrees Tree Structure

**Step 1-3**: Extract and prepare
```bash
cd /path/to/UnifyWeaver

# Extract Pearltrees tree extraction example
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.pearltrees_extract_tree" \
  playbooks/examples_library/rdf_examples.md \
  > tmp/run_pearltrees_extract.sh

chmod +x tmp/run_pearltrees_extract.sh
```

**Step 4**: Run
```bash
bash tmp/run_pearltrees_extract.sh
```

**Expected Output**:
```
=== Pearltrees RDF Tree Structure Extraction ===

✓ Loaded Pearltrees RDF data

Extracting parent-child relationships...
parent('https://www.pearltrees.com/t/hacktivism/id2492215', ...).
parent('https://www.pearltrees.com/t/hacktivism/id2492215', ...).
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

## Common Mistakes to Avoid

### ❌ WRONG: Trying to run Prolog file with bash
```bash
bash tmp/pearltrees_analysis.pl  # ERROR - .pl is Prolog, not bash!
```

### ✅ CORRECT: Run the bash script wrapper
```bash
bash tmp/run_pearltrees_extract.sh  # Bash script generates and runs Prolog
```

### ❌ WRONG: Missing RDF source file
```bash
# Running without the example RDF file
```

### ✅ CORRECT: Ensure RDF file exists
```bash
# Check file exists first
ls context/PT/Example_pearltrees_rdf_export.rdf
bash tmp/run_pearltrees_extract.sh
```

## Key Concepts

### RDF Triple Store
SWI-Prolog's semweb library loads RDF data into an in-memory triple store:
- **Subject-Predicate-Object** triples
- Query with `rdf(Subject, Predicate, Object)`
- Indexed for fast lookups

### SPARQL Queries
Standard semantic web query language:
```sparql
PREFIX pt: <http://www.pearltrees.com/rdf/0.1/#>
SELECT ?tree ?title
WHERE {
  ?tree a pt:Tree .
  ?tree dcterms:title ?title
}
```

### RDF to Prolog Facts
Convert RDF triples to traditional Prolog facts:
```prolog
% RDF triple:
rdf('https://...id2492215', 'http://...#parentTree', 'https://...').

% Becomes Prolog fact:
parent('https://...id2492215', 'https://...').
```

### Classic Graph Algorithms
Use extracted facts with standard Prolog patterns:
```prolog
% Transitive closure
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Descendant (inverse)
descendant(X, Y) :- ancestor(Y, X).

% Reachability
reachable(X, Y) :- ancestor(X, Y).
```

## Integration with Existing Examples

### Connection to Tree Recursion
Pearltrees data creates real tree structures that can be processed with UnifyWeaver's tree recursion compilation:

```prolog
% From tree_recursion_playbook.md
tree_sum([V,L,R], S) :-
    tree_sum(L, LS),
    tree_sum(R, RS),
    S is V + LS + RS.

% Apply to Pearltrees tree structure
% (after converting to tree representation)
```

### Connection to Graph Examples
The parent-child relationships demonstrate classic graph algorithms:
- Transitive closure (ancestor/descendant)
- Reachability testing
- Path finding
- Depth calculation

All of these can be compiled to bash using UnifyWeaver's recursive compilation.

## Expected Outcome
- RDF data successfully loaded into triple store
- Parent-child relationships extracted as Prolog facts
- Classic Prolog graph queries work on real data
- Can export to multiple formats (JSON, CSV, DOT)
- Demonstrates connection between semantic web and logic programming

## Use Cases

### 1. Pearltrees Organization Analysis
- Extract complete tree hierarchy
- Analyze bookmark organization
- Find deeply nested items
- Calculate tree statistics

### 2. Knowledge Graph Processing
- Query with SPARQL
- Convert to Prolog for reasoning
- Find patterns and relationships
- Export for visualization

### 3. Data Migration
- Extract from Pearltrees RDF
- Transform structure
- Export to other formats (JSON, CSV)
- Import into other systems

### 4. Educational Examples
- Real-world RDF data
- Demonstrates graph algorithms
- Shows Prolog's strength with hierarchical data
- Connects to classic CS concepts (trees, graphs, traversal)

## Citations
[1] playbooks/examples_library/rdf_examples.md
[2] skills/skill_unifyweaver_environment.md
[3] skills/skill_extract_records.md
[4] docs/development/testing/playbooks/rdf_data_source_playbook__reference.md
[5] SWI-Prolog semweb documentation: https://www.swi-prolog.org/pldoc/man?section=semweb
