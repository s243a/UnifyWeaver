# Proposal: UnifyWeaver-Native Pearltrees Processing

## Overview

This proposal outlines migrating Pearltrees RDF processing and mindmap generation to use UnifyWeaver's native capabilities (Prolog predicates, aggregates, sources, and code generation). This will make the tooling more flexible, declarative, and serve as educational examples for the UnifyWeaver project.

## Current State

Currently, Pearltrees processing uses Python scripts:
- `build_children_index.py` - Parses RDF, groups children by parent tree, stores in SQLite
- `generate_mindmap.py` - Generates SimpleMind `.smmx` files from JSONL/API data
- `scan_incomplete_mindmaps.py` - Scans for mindmaps needing repair

These work but are imperative Python code. A UnifyWeaver-native approach would be more declarative and composable.

## Proposed UnifyWeaver-Native Approach

### 1. Data Sources

Define Pearltrees data as UnifyWeaver sources:

```prolog
% RDF-derived SQLite index
:- source(sqlite, pearl_children, [
    sqlite_file('.local/data/children_index.db'),
    table('children'),
    columns([parent_tree_id, pearl_type, title, pos_order, external_url, see_also_uri])
]).

% Trees from JSONL
:- source(json, pearl_trees, [
    json_file('reports/pearltrees_targets_trees.jsonl'),
    columns([type, tree_id, title, uri, cluster_id])
]).
```

### 2. Aggregate Queries

Use UnifyWeaver's `aggregate_all/4` to group children:

```prolog
% Group children by parent tree
tree_with_children(TreeId, Title, Children) :-
    pearl_trees(tree, TreeId, Title, _, _),
    aggregate_all(
        bag(child(Type, ChildTitle, Url, Order)),
        pearl_children(TreeId, Type, ChildTitle, Order, Url, _),
        TreeId,
        Children
    ).

% Count children per tree (for finding incomplete trees)
tree_child_count(TreeId, Count) :-
    pearl_trees(tree, TreeId, _, _, _),
    aggregate_all(count, pearl_children(TreeId, _, _, _, _, _), TreeId, Count).

% Find incomplete trees (only root, no children)
incomplete_tree(TreeId, Title) :-
    tree_child_count(TreeId, Count),
    Count =< 1,
    pearl_trees(tree, TreeId, Title, _, _).
```

### 3. Mindmap Generation via Templates

Use UnifyWeaver's template system to generate SimpleMind XML:

```prolog
% Generate mindmap XML for a tree
generate_mindmap_xml(TreeId, XML) :-
    tree_with_children(TreeId, Title, Children),
    template(simplemind_xml, [
        title(Title),
        tree_id(TreeId),
        children(Children)
    ], XML).

% Template definition
:- template(simplemind_xml, '
<?xml version="1.0" encoding="UTF-8"?>
<smmx>
  <mindmap>
    <topic text="{{title}}" id="root">
      {{#each children}}
      <topic text="{{this.title}}" id="{{this.id}}">
        {{#if this.url}}<link url="{{this.url}}"/>{{/if}}
      </topic>
      {{/each}}
    </topic>
  </mindmap>
</smmx>
').
```

### 4. Cross-Target Code Generation

Generate mindmap tools in multiple languages:

```prolog
% Compile to Python
?- compile_predicate_to_python(tree_with_children/3, [mode(generator)], PyCode).

% Compile to C# (for performance)
?- compile_predicate_to_csharp(tree_with_children/3, [mode(generator)], CsCode).

% Compile to Go (for CLI tools)
?- compile_predicate_to_go(incomplete_tree/2, [mode(generator)], GoCode).
```

## Benefits

1. **Declarative**: Logic expressed as predicates rather than imperative code
2. **Composable**: Predicates can be combined and reused
3. **Multi-target**: Same logic can generate Python, C#, Go, or run in Prolog
4. **Educational**: Serves as examples for UnifyWeaver tutorials
5. **Flexible**: Easy to add new queries, aggregations, or output formats
6. **Testable**: Prolog predicates can be unit tested with plunit

## Implementation Phases

### Phase 1: Source Definitions
- Define SQLite source for children index
- Define JSONL sources for trees/pearls
- Test basic queries

### Phase 2: Aggregate Queries
- Implement `tree_with_children/3` using `aggregate_all`
- Implement `incomplete_tree/2` for scanning
- Add tests

### Phase 3: Template-Based Generation
- Create SimpleMind XML templates
- Generate mindmap files from Prolog
- Compare output with existing Python generator

### Phase 4: CLI Tools
- Compile scan/generate predicates to Go/Python
- Create standalone CLI tools
- Benchmark performance

## Mindmap Tool Flexibility Goals

The UnifyWeaver-native approach enables:

1. **Custom Output Formats**: Define new templates for different mindmap tools (FreeMind, XMind, Mermaid, etc.)

2. **Query-Based Filtering**: Filter trees/pearls using Prolog predicates before generation
   ```prolog
   % Only trees with Wikipedia links
   wikipedia_tree(TreeId) :-
       tree_with_children(TreeId, _, Children),
       member(child(pagepearl, _, Url, _), Children),
       sub_string(Url, _, _, _, "wikipedia.org").
   ```

3. **Hierarchical Transformations**: Restructure trees using predicates
   ```prolog
   % Flatten nested trees
   flatten_tree(TreeId, FlatChildren) :-
       tree_with_children(TreeId, _, Children),
       findall(C, (member(C, Children) ; nested_child(Children, C)), FlatChildren).
   ```

4. **Cross-Account Merging**: Combine trees from multiple accounts
   ```prolog
   merged_tree(MergedId, AllChildren) :-
       tree_alias(TreeId1, TreeId2),  % Cross-account alias
       tree_with_children(TreeId1, _, C1),
       tree_with_children(TreeId2, _, C2),
       append(C1, C2, AllChildren).
   ```

5. **Incremental Updates**: Track changes and regenerate only affected mindmaps

## Educational Value

This work creates examples for the `education/` repository:

1. **Data Sources Tutorial**: How to define CSV, JSON, SQLite sources
2. **Aggregates Tutorial**: Using `aggregate_all` for grouping and counting
3. **Templates Tutorial**: Generating structured output from predicates
4. **Cross-Target Tutorial**: Same logic, multiple output languages

## Related Work

- `docs/proposals/csharp_generator_aggregates.md` - Aggregate support in C# generator
- `tests/core/test_csharp_generator_aggregates_grouping.pl` - Grouping aggregate tests
- `src/unifyweaver/sources.pl` - Source definition API

## Next Steps

1. Create `src/unifyweaver/examples/pearltrees/` directory
2. Define source predicates for children index
3. Implement basic aggregate queries
4. Add plunit tests
5. Document as tutorial
