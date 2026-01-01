# RDF Data Source Examples

This file contains executable examples for processing RDF data using SWI-Prolog's built-in semweb library.

---

## Record: unifyweaver.execution.pearltrees_extract_tree

Extract tree/parent relationships from Pearltrees RDF export and demonstrate classic Prolog graph queries.

This example shows:
1. Loading RDF data with semweb library
2. Extracting parent-child relationships
3. Using extracted data with Prolog graph predicates (descendant, ancestor, path finding)

> [!example-record]
> id: unifyweaver.execution.pearltrees_extract_tree
> name: pearltrees_extract_tree
> description: Extract Pearltrees tree structure and demonstrate graph queries

```bash
#!/bin/bash
# Pearltrees RDF Tree Extraction Example

set -e

echo "=== Pearltrees RDF Tree Structure Extraction ==="
echo

# Check if example RDF file exists
if [ ! -f "context/PT/Example_pearltrees_rdf_export.rdf" ]; then
    echo "Error: Example RDF file not found"
    echo "Expected: context/PT/Example_pearltrees_rdf_export.rdf"
    exit 1
fi

# Create Prolog script to extract and analyze tree structure
cat > tmp/pearltrees_analysis.pl <<'PROLOG_EOF'
:- use_module(library(semweb/rdf_db)).
:- use_module(library(semweb/rdf_turtle)).

% Load the Pearltrees RDF data
load_pearltrees_data :-
    rdf_load('context/PT/Example_pearltrees_rdf_export.rdf', [format(xml)]),
    format('✓ Loaded Pearltrees RDF data~n', []).

% Extract parent-child relationships from RDF
% pt:Pearl has pt:parentTree property
extract_parent_child(Child, Parent) :-
    rdf_global_id(pt:'parentTree', ParentTreeProp),
    rdf(Child, ParentTreeProp, Parent).

% Build parent/2 facts from RDF data
build_parent_facts :-
    format('~nExtracting parent-child relationships...~n', []),
    forall(
        extract_parent_child(Child, Parent),
        (   format('parent(~w, ~w).~n', [Parent, Child])
        )
    ).

% Classic Prolog: ancestor/2 using extracted data
ancestor(X, Y) :- extract_parent_child(Y, X).
ancestor(X, Z) :- extract_parent_child(Y, X), ancestor(Y, Z).

% Classic Prolog: descendant/2 (inverse of ancestor)
descendant(X, Y) :- ancestor(Y, X).

% Find all descendants of a tree
find_descendants(Tree, Descendants) :-
    findall(
        Desc,
        descendant(Desc, Tree),
        Descendants
    ).

% Find path from root to node
find_path(Root, Node, Path) :-
    find_path_helper(Root, Node, [Root], RevPath),
    reverse(RevPath, Path).

find_path_helper(Node, Node, Acc, Acc).
find_path_helper(Current, Target, Acc, Path) :-
    extract_parent_child(Child, Current),
    \+ member(Child, Acc),  % Avoid cycles
    find_path_helper(Child, Target, [Child|Acc], Path).

% Get tree title from RDF
get_title(Resource, Title) :-
    rdf_global_id(dcterms:title, TitleProp),
    rdf(Resource, TitleProp, literal(Title)).

% Pretty print tree structure
print_tree_info(Tree) :-
    (   get_title(Tree, Title)
    ->  format('Tree: ~w~n  Title: "~w"~n', [Tree, Title])
    ;   format('Tree: ~w~n', [Tree])
    ),
    find_descendants(Tree, Descendants),
    length(Descendants, Count),
    format('  Descendants: ~w~n', [Count]),
    (   Count > 0
    ->  format('  Children:~n', []),
        forall(
            (extract_parent_child(Child, Tree), get_title(Child, ChildTitle)),
            format('    - ~w~n', [ChildTitle])
        )
    ;   true
    ).

% Main analysis
analyze_pearltrees :-
    format('~n=== Extracted Parent-Child Relationships ===~n', []),
    build_parent_facts,

    format('~n=== Tree Structure Analysis ===~n', []),
    % Find all root trees (those that are parents but not children)
    findall(
        Root,
        (   extract_parent_child(_, Root),
            \+ extract_parent_child(Root, _)
        ),
        Roots
    ),

    format('~nFound ~w root tree(s)~n~n', [length(Roots)]),
    forall(
        member(Root, Roots),
        (   print_tree_info(Root),
            nl
        )
    ),

    format('~n=== Demonstrating Classic Prolog Queries ===~n~n', []),

    % Example: Find all ancestors of a specific pearl
    (   extract_parent_child(SomePearl, _)
    ->  findall(Anc, ancestor(Anc, SomePearl), Ancestors),
        format('Ancestors of ~w: ~w~n', [SomePearl, Ancestors])
    ;   true
    ),

    % Example: Check reachability
    (   extract_parent_child(Child1, Parent1),
        extract_parent_child(Child2, Parent1),
        Child1 \= Child2
    ->  format('~nReachability test:~n', []),
        (   ancestor(Parent1, Child1)
        ->  format('  ✓ ~w is ancestor of ~w~n', [Parent1, Child1])
        ;   format('  ✗ ~w is NOT ancestor of ~w~n', [Parent1, Child1])
        ),
        (   descendant(Child2, Parent1)
        ->  format('  ✓ ~w is descendant of ~w~n', [Child2, Parent1])
        ;   format('  ✗ ~w is NOT descendant of ~w~n', [Child2, Parent1])
        )
    ;   true
    ).

% Entry point
main :-
    load_pearltrees_data,
    analyze_pearltrees,
    format('~n✓ Analysis complete~n', []).

:- initialization(main, main).
PROLOG_EOF

echo "Running Prolog analysis..."
swipl -g main -t halt tmp/pearltrees_analysis.pl

echo
echo "Success: Pearltrees tree structure extracted and analyzed"
```

---

## Record: unifyweaver.execution.rdf_graph_queries

Demonstrate SPARQL queries on RDF data and conversion to Prolog facts.

> [!example-record]
> id: unifyweaver.execution.rdf_graph_queries
> name: rdf_graph_queries
> description: SPARQL queries and RDF to Prolog fact conversion

```bash
#!/bin/bash
# RDF Graph Queries Example

set -e

echo "=== RDF Graph Queries with SPARQL ==="
echo

cat > tmp/rdf_sparql_demo.pl <<'PROLOG_EOF'
:- use_module(library(semweb/rdf_db)).
:- use_module(library(semweb/sparql_client)).

% Load RDF data
load_rdf :-
    rdf_load('context/PT/Example_pearltrees_rdf_export.rdf', [format(xml)]),
    format('✓ RDF data loaded~n~n', []).

% Example 1: SPARQL query to get all trees with titles
query_all_trees :-
    format('=== All Trees (SPARQL Query) ===~n', []),
    Query = 'PREFIX pt: <http://www.pearltrees.com/rdf/0.1/#>
             PREFIX dcterms: <http://purl.org/dc/elements/1.1/>
             SELECT ?tree ?title
             WHERE {
               ?tree a pt:Tree .
               ?tree dcterms:title ?title
             }',
    sparql_query(Query, Row, []),
    Row = row(Tree, literal(Title)),
    format('Tree: ~w~n  Title: "~w"~n~n', [Tree, Title]),
    fail.
query_all_trees :- !.

% Example 2: Convert RDF to Prolog facts
rdf_to_prolog_facts :-
    format('=== Converting RDF to Prolog Facts ===~n~n', []),

    % Generate tree/2 facts
    format('% Tree facts~n', []),
    forall(
        (   rdf_global_id(pt:'Tree', TreeClass),
            rdf(Tree, rdf:type, TreeClass),
            rdf_global_id(dcterms:title, TitleProp),
            rdf(Tree, TitleProp, literal(Title))
        ),
        format('tree(~q, ~q).~n', [Tree, Title])
    ),

    nl,

    % Generate parent/2 facts
    format('% Parent-child facts~n', []),
    forall(
        (   rdf_global_id(pt:parentTree, ParentProp),
            rdf(Child, ParentProp, Parent)
        ),
        format('parent(~q, ~q).~n', [Parent, Child])
    ),

    nl.

% Example 3: Use converted facts with classic Prolog patterns
demo_classic_patterns :-
    format('=== Classic Prolog Patterns ===~n~n', []),

    % Define ancestor using parent
    format('% Define ancestor/2 (transitive closure of parent)~n', []),
    format('ancestor(X, Y) :- parent(X, Y).~n', []),
    format('ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).~n~n', []),

    % Find depth of tree
    format('% Find tree depth~n', []),
    format('tree_depth(Root, Depth) :-~n', []),
    format('    findall(D, (descendant_depth(Root, _, D)), Depths),~n', []),
    format('    max_list(Depths, Depth).~n~n', []),

    % Count nodes
    rdf_global_id(pt:parentTree, ParentProp),
    findall(1, rdf(_, ParentProp, _), Nodes),
    length(Nodes, NodeCount),
    format('Total nodes in graph: ~w~n~n', [NodeCount]).

% Main
main :-
    load_rdf,
    query_all_trees,
    rdf_to_prolog_facts,
    demo_classic_patterns,
    format('✓ RDF graph queries complete~n', []).

:- initialization(main, main).
PROLOG_EOF

echo "Running RDF/SPARQL demo..."
swipl -g main -t halt tmp/rdf_sparql_demo.pl

echo
echo "Success: RDF graph queries demonstrated"
```

---

## Record: unifyweaver.execution.rdf_export_formats

Extract RDF data and export to various formats (JSON, CSV, DOT graph).

> [!example-record]
> id: unifyweaver.execution.rdf_export_formats
> name: rdf_export_formats
> description: Export RDF tree structure to multiple formats

```bash
#!/bin/bash
# RDF Export Formats Example

set -e

echo "=== RDF Export to Multiple Formats ==="
echo

mkdir -p tmp/rdf_exports

cat > tmp/rdf_export.pl <<'PROLOG_EOF'
:- use_module(library(semweb/rdf_db)).
:- use_module(library(http/json)).

% Load RDF
load_rdf :-
    rdf_load('context/PT/Example_pearltrees_rdf_export.rdf', [format(xml)]).

% Export to JSON
export_json(OutputFile) :-
    format('Exporting to JSON...~n', []),
    open(OutputFile, write, Stream),

    % Collect all trees and their children
    findall(
        json([
            tree=Tree,
            title=Title,
            children=Children
        ]),
        (   rdf_global_id(pt:'Tree', TreeClass),
            rdf(Tree, rdf:type, TreeClass),
            rdf_global_id(dcterms:title, TitleProp),
            rdf(Tree, TitleProp, literal(Title)),
            findall(
                ChildURI,
                (   rdf_global_id(pt:parentTree, ParentProp),
                    rdf(ChildURI, ParentProp, Tree)
                ),
                Children
            )
        ),
        Trees
    ),

    json_write(Stream, json(trees=Trees), [width(0)]),
    close(Stream),
    format('✓ JSON export: ~w~n', [OutputFile]).

% Export to CSV
export_csv(OutputFile) :-
    format('Exporting to CSV...~n', []),
    open(OutputFile, write, Stream),

    % Header
    format(Stream, 'Parent,Child,ChildTitle~n', []),

    % Rows
    forall(
        (   rdf_global_id(pt:parentTree, ParentProp),
            rdf(Child, ParentProp, Parent),
            rdf_global_id(dcterms:title, TitleProp),
            (   rdf(Child, TitleProp, literal(Title))
            ->  true
            ;   Title = ''
            )
        ),
        format(Stream, '"~w","~w","~w"~n', [Parent, Child, Title])
    ),

    close(Stream),
    format('✓ CSV export: ~w~n', [OutputFile]).

% Export to DOT (GraphViz) format
export_dot(OutputFile) :-
    format('Exporting to DOT (GraphViz)...~n', []),
    open(OutputFile, write, Stream),

    format(Stream, 'digraph Pearltrees {~n', []),
    format(Stream, '  rankdir=TB;~n', []),
    format(Stream, '  node [shape=box];~n~n', []),

    % Nodes with labels
    forall(
        (   rdf_global_id(pt:'Tree', TreeClass),
            rdf(Tree, rdf:type, TreeClass),
            rdf_global_id(dcterms:title, TitleProp),
            rdf(Tree, TitleProp, literal(Title)),
            % Create safe node ID from URI
            atomic_list_concat(['node', Tree], '_', NodeId)
        ),
        format(Stream, '  ~w [label="~w"];~n', [NodeId, Title])
    ),

    nl(Stream),

    % Edges
    forall(
        (   rdf_global_id(pt:parentTree, ParentProp),
            rdf(Child, ParentProp, Parent),
            atomic_list_concat(['node', Parent], '_', ParentId),
            atomic_list_concat(['node', Child], '_', ChildId)
        ),
        format(Stream, '  ~w -> ~w;~n', [ParentId, ChildId])
    ),

    format(Stream, '}~n', []),
    close(Stream),
    format('✓ DOT export: ~w~n', [OutputFile]).

% Main
main :-
    load_rdf,
    export_json('tmp/rdf_exports/pearltrees.json'),
    export_csv('tmp/rdf_exports/pearltrees.csv'),
    export_dot('tmp/rdf_exports/pearltrees.dot'),
    format('~n✓ All exports complete~n', []),
    format('~nGenerated files:~n', []),
    format('  - tmp/rdf_exports/pearltrees.json~n', []),
    format('  - tmp/rdf_exports/pearltrees.csv~n', []),
    format('  - tmp/rdf_exports/pearltrees.dot (visualize with: dot -Tpng pearltrees.dot -o pearltrees.png)~n', []).

:- initialization(main, main).
PROLOG_EOF

echo "Running RDF export..."
swipl -g main -t halt tmp/rdf_export.pl

echo
echo "Success: RDF data exported to multiple formats"
```

---

## Notes

**RDF Processing Capabilities:**
- Uses SWI-Prolog's built-in semweb library (no external dependencies)
- SPARQL query support for complex graph queries
- Conversion between RDF triples and Prolog facts
- Integration with classic Prolog graph algorithms

**Pearltrees-Specific Features:**
- Extract tree hierarchy (pt:parentTree relationships)
- Get tree metadata (titles, IDs, timestamps)
- Handle different pearl types (RootPearl, AliasPearl, RefPearl, SectionPearl)
- Export to various formats for further processing

**Educational Value:**
- Demonstrates real-world RDF data processing
- Shows connection between semantic web and logic programming
- Uses extracted data with classic Prolog patterns (ancestor, descendant, reachability)
- Practical example of graph traversal and analysis
