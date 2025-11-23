# Declarative Hierarchical Queries Design

**Status:** Proposal
**Created:** 2025-01-22
**Philosophy:** Declare relationships, let the system handle traversal

---

## Problem Statement

Hierarchical data (trees, graphs, parent-child relationships) requires:
1. **Relationship extraction** - Finding connections between elements
2. **Traversal** - Walking the hierarchy (depth-first, breadth-first)
3. **Memory management** - Streaming vs in-memory strategies

Current manual approach breaks declarative philosophy and is memory-inefficient.

---

## Design Goals

1. **Declarative relationships** - Specify WHAT relationships exist, not HOW to find them
2. **Configurable strategies** - Stream on-demand OR build in-memory map (user choice)
3. **Memory-aware** - Default to streaming, opt-in to in-memory for speed
4. **Composable** - Build on field extraction and SGML parsing designs

---

## Declarative Syntax

### Declare Parent-Child Relationship

```prolog
:- relationship(pearltrees, parent_child,
    parent_source: source(xml, trees, [
        file('pearltrees_export.rdf'),
        tag('pt:Tree'),
        fields([id: 'pt:treeId'])
    ]),
    child_source: source(xml, pearls, [
        file('pearltrees_export.rdf'),
        tag('pt:AliasPearl'),
        fields([
            parent: xpath('pt:parentTree/@rdf:resource', extract_id),
            child: xpath('rdfs:seeAlso/@rdf:resource', extract_id)
        ])
    ]),
    link: pearl.parent = tree.id,
    strategy: streaming  % or: in_memory
).
```

**What this declares:**
- Trees and pearls are connected
- A pearl's `parent` field references a tree's `id`
- A pearl's `child` field references another tree
- Use streaming strategy (on-demand queries, constant memory)

### Query Usage

```prolog
% Get immediate children of a tree
?- children(pearltrees, '10647426', Children).
Children = [tree{id:14682380, ...}, tree{id:12489164, ...}, ...].

% Get full hierarchy (recursive)
?- hierarchy(pearltrees, '10647426', 2, Tree).
Tree = tree{
    id: '10647426',
    title: 'Physics',
    children: [
        tree{id:14682380, title:'Physics Education', children:[...]},
        tree{id:12489164, title:'Geophysics', children:[...]},
        ...
    ]
}.

% Find path from root to node
?- path(pearltrees, '10647426', '14682380', Path).
Path = ['10647426', '14682380'].

% Get all descendants (any depth)
?- descendants(pearltrees, '10647426', All).
All = [14682380, 12489164, ...].  % 150 total descendants
```

---

## Traversal Strategies

### Strategy 1: Streaming (Default)

**Memory:** Constant (~20KB)
**Speed:** Slower for deep/wide trees
**Best for:** Large datasets, mobile, limited RAM

```prolog
:- relationship(pearltrees, parent_child,
    ...,
    strategy: streaming
).
```

**How it works:**
1. Query: "Get children of tree X"
2. Stream through pearls file
3. Filter pearls where parent = X
4. Extract child IDs
5. Fetch each child tree on-demand
6. Yield results

**Performance:**
- 1 level: 0.3s (one pass through pearls)
- 2 levels: 0.6s (two passes)
- N levels: N × 0.3s

### Strategy 2: In-Memory Map

**Memory:** Scales with tree count (~500KB for 5,000 trees)
**Speed:** Very fast for deep/wide trees
**Best for:** Repeated queries, when you have RAM

```prolog
:- relationship(pearltrees, parent_child,
    ...,
    strategy: in_memory
).
```

**How it works:**
1. Build phase (once): Extract all relationships, build map
   ```prolog
   parent_to_children = {
       '10647426' → [14682380, 12489164, ...],
       '14682380' → [16234567, ...],
       ...
   }
   ```
2. Query: Instant lookup in map
3. Traversal: No file I/O, just map navigation

**Performance:**
- Build: 0.4s (one pass to build map)
- Any query: < 0.001s (memory lookup)
- Deep hierarchy (10 levels): 0.01s

### Strategy 3: Hybrid (Smart Default?)

**Idea:** Stream by default, cache results in LRU cache

```prolog
:- relationship(pearltrees, parent_child,
    ...,
    strategy: hybrid(cache_size(100))
).
```

**How it works:**
- First query for node X: Stream (slow)
- Cache result
- Repeat query: Instant (cached)
- Cache evicts LRU when full

---

## Implementation Architecture

### Relationship Registry

New module: `src/unifyweaver/query/relationships.pl`

```prolog
:- module(relationships, [
    relationship/3,    % Declare relationship
    children/3,        % Get immediate children
    hierarchy/4,       % Get hierarchy tree
    descendants/3,     % Get all descendants
    ancestors/3,       % Get all ancestors
    path/4            % Find path between nodes
]).

% Store relationship definitions
:- dynamic declared_relationship/3.

% Declare a relationship
relationship(Name, Type, Options) :-
    validate_relationship_options(Options),
    assertz(declared_relationship(Name, Type, Options)),
    initialize_relationship(Name, Type, Options).

initialize_relationship(Name, Type, Options) :-
    option(strategy(Strategy), Options, streaming),
    initialize_strategy(Name, Strategy, Options).
```

### Streaming Strategy Implementation

```prolog
% Get children using streaming
children(RelName, ParentID, Children) :-
    declared_relationship(RelName, parent_child, Options),
    option(strategy(streaming), Options),
    option(child_source(ChildSource), Options),
    option(link(LinkSpec), Options),

    % Extract all child relationship records that match parent
    call_source(ChildSource, AllRecords),
    include(has_parent(ParentID, LinkSpec), AllRecords, ChildRecords),

    % Extract child IDs and fetch actual children
    maplist(extract_child_id(LinkSpec), ChildRecords, ChildIDs),
    maplist(fetch_tree_by_id, ChildIDs, Children).

has_parent(ParentID, pearl.parent = tree.id, Record) :-
    Record.parent = ParentID.
```

### In-Memory Strategy Implementation

```prolog
:- dynamic parent_child_map/3.

% Initialize: Build complete map
initialize_strategy(Name, in_memory, Options) :-
    option(child_source(ChildSource), Options),
    option(link(LinkSpec), Options),

    % Extract all relationships once
    call_source(ChildSource, AllRecords),

    % Build parent→children map
    build_parent_map(AllRecords, LinkSpec, Map),

    % Store in dynamic predicate
    assertz(parent_child_map(Name, Map, timestamp(Now))).

% Query: Just lookup
children(RelName, ParentID, Children) :-
    declared_relationship(RelName, parent_child, Options),
    option(strategy(in_memory), Options),
    parent_child_map(RelName, Map, _),

    % Instant lookup
    get_dict(ParentID, Map, ChildIDs),
    maplist(fetch_tree_by_id, ChildIDs, Children).
```

---

## AWK-Based In-Memory Builder

For maximum performance on the in-memory strategy, build the map in AWK:

```awk
# extract_relationships.awk
BEGIN {
    RS = "\0"  # Null-delimited elements
    parent_map = ""
}

/<pt:AliasPearl/ {
    # Extract parent ID
    match($0, /pt:parentTree[^>]*\/id([0-9]+)/, parent_arr)
    parent_id = parent_arr[1]

    # Extract child ID
    match($0, /rdfs:seeAlso[^>]*\/id([0-9]+)/, child_arr)
    child_id = child_arr[1]

    if (parent_id && child_id) {
        # Build map: parent_id → comma-separated child IDs
        if (parent_id in children) {
            children[parent_id] = children[parent_id] "," child_id
        } else {
            children[parent_id] = child_id
        }
    }
}

END {
    # Output as Prolog dict
    print "parent_child_map({"
    for (parent in children) {
        printf "  '%s': [%s],\n", parent, children[parent]
    }
    print "})."
}
```

**Usage:**
```prolog
build_in_memory_map(Name, File) :-
    % Run AWK to build map
    format(atom(Cmd), 'awk -f scripts/utils/extract_relationships.awk ~w', [File]),
    process_create(path(bash), ['-c', Cmd], [stdout(pipe(Out))]),
    read_term(Out, parent_child_map(Map)),
    close(Out),

    % Store map
    assertz(parent_child_map(Name, Map, timestamp(Now))).
```

**Performance:**
- Build map from 19MB file: **0.2s**
- Memory: ~500KB for 5,000 trees
- Subsequent queries: **< 0.001s**

---

## Recursive Hierarchy Queries

### Get Full Hierarchy Tree

```prolog
% Get hierarchy with depth limit
hierarchy(RelName, RootID, MaxDepth, HierarchyTree) :-
    fetch_tree_by_id(RootID, RootData),
    hierarchy_recursive(RelName, RootID, RootData, MaxDepth, HierarchyTree).

hierarchy_recursive(_, _, RootData, 0, RootData) :- !.

hierarchy_recursive(RelName, ID, RootData, MaxDepth, Tree) :-
    MaxDepth > 0,
    children(RelName, ID, Children),
    NextDepth is MaxDepth - 1,
    maplist(build_child_tree(RelName, NextDepth), Children, ChildTrees),
    Tree = RootData.put(children, ChildTrees).

build_child_tree(RelName, MaxDepth, ChildData, ChildTree) :-
    hierarchy_recursive(RelName, ChildData.id, ChildData, MaxDepth, ChildTree).
```

**Usage:**
```prolog
?- hierarchy(pearltrees, '10647426', 2, Tree).
Tree = tree{
    id: '10647426',
    title: 'Physics',
    children: [
        tree{
            id: 14682380,
            title: 'Physics Education',
            children: [
                tree{id: 16234567, title: 'Quantum Mechanics', children: []},
                ...
            ]
        },
        ...
    ]
}.
```

---

## Path Finding

```prolog
% Find path from ancestor to descendant
path(RelName, AncestorID, DescendantID, Path) :-
    path_dfs(RelName, AncestorID, DescendantID, [AncestorID], Path).

path_dfs(_, ID, ID, Acc, Path) :-
    reverse(Acc, Path).

path_dfs(RelName, CurrentID, TargetID, Acc, Path) :-
    children(RelName, CurrentID, Children),
    member(Child, Children),
    Child.id = NextID,
    \+ member(NextID, Acc),  % Avoid cycles
    path_dfs(RelName, NextID, TargetID, [NextID|Acc], Path).
```

**Usage:**
```prolog
?- path(pearltrees, '10647426', '16234567', Path).
Path = ['10647426', 14682380, 16234567].
```

---

## Advanced: Multiple Relationship Types

### Declare Multiple Relationships

```prolog
% Parent-child relationships
:- relationship(pearltrees, parent_child,
    parent_source: source(xml, trees, [...]),
    child_source: source(xml, pearls, [...]),
    link: pearl.parent = tree.id,
    strategy: in_memory
).

% Sibling relationships (derived from parent-child)
:- relationship(pearltrees, siblings,
    derived_from: parent_child,
    rule: (
        children(pearltrees, Parent, Children),
        member(A, Children),
        member(B, Children),
        A.id \= B.id
    )
).

% Tag-based relationships
:- relationship(pearltrees, related_by_tag,
    source: source(xml, trees, [
        fields([id: 'pt:treeId', tags: xpath('pt:tag/text()', list)])
    ]),
    link: trees_share_tag(A.tags, B.tags),
    strategy: in_memory
).
```

---

## Memory vs Speed Tradeoffs

### Configurable Per-Query

```prolog
% Use streaming for this query (save memory)
?- children(pearltrees, '10647426', Children, [strategy(streaming)]).

% Use in-memory for this query (fast, but load map first)
?- children(pearltrees, '10647426', Children, [strategy(in_memory)]).

% Auto-select: streaming for single query, in-memory if called repeatedly
?- children(pearltrees, '10647426', Children, [strategy(auto)]).
```

### Benchmark Comparison (19MB File, 5,002 Trees)

| Query | Streaming | In-Memory | Speedup |
|-------|-----------|-----------|---------|
| Get children (1 node) | 0.3s | 0.001s | 300x |
| Build hierarchy (depth 2) | 0.9s | 0.01s | 90x |
| Build hierarchy (depth 5) | 3.5s | 0.03s | 117x |
| Find path (A→B) | 1.2s | 0.005s | 240x |
| **Initial build cost** | **0s** | **0.2s** | N/A |

**Recommendation:**
- Single queries: Streaming
- Multiple queries: In-memory (amortize build cost)
- Mobile with limited RAM: Streaming
- Desktop/server: In-memory

---

## Integration Examples

### Example 1: Pearltrees Navigation

```prolog
% Declare relationship once
:- relationship(pearltrees, parent_child,
    parent_source: source(xml, trees, [
        file('context/PT/pearltrees_export.rdf'),
        tag('pt:Tree'),
        fields([id: 'pt:treeId', title: xpath('dcterms:title/text()', strip_cdata)])
    ]),
    child_source: source(xml, pearls, [
        file('context/PT/pearltrees_export.rdf'),
        tag('pt:AliasPearl'),
        fields([
            parent: xpath('pt:parentTree/@rdf:resource', extract_id),
            child: xpath('rdfs:seeAlso/@rdf:resource', extract_id),
            title: xpath('dcterms:title/text()', strip_cdata)
        ])
    ]),
    link: pearl.parent = tree.id,
    strategy: streaming
).

% Use naturally:
explore_physics :-
    % Find Physics tree
    find_trees_by_title('physics', [PhysicsTree|_]),

    % Get its children
    children(pearltrees, PhysicsTree.id, Kids),
    format('Physics has ~w subtrees:~n', [length(Kids)]),
    maplist(print_tree_title, Kids),

    % Get full 3-level hierarchy
    hierarchy(pearltrees, PhysicsTree.id, 3, FullTree),
    print_tree(FullTree).
```

### Example 2: Find Related Content

```prolog
% Find all siblings of a tree (trees with same parent)
siblings(TreeID, Siblings) :-
    % Find parent
    children(pearltrees, ParentID, AllChildren),
    member(Child, AllChildren),
    Child.id = TreeID,
    !,
    % Get all children of same parent
    exclude(id_is(TreeID), AllChildren, Siblings).

% Find cousins (children of parent's siblings)
cousins(TreeID, Cousins) :-
    siblings(TreeID, Siblings),
    maplist(children(pearltrees), Siblings, CousinLists),
    append(CousinLists, Cousins).
```

### Example 3: Tree Statistics

```prolog
% Compute tree depth
tree_depth(TreeID, Depth) :-
    descendants_with_depth(pearltrees, TreeID, 0, MaxDepth),
    Depth = MaxDepth.

descendants_with_depth(RelName, TreeID, CurrentDepth, MaxDepth) :-
    children(RelName, TreeID, Children),
    (   Children = []
    ->  MaxDepth = CurrentDepth
    ;   NextDepth is CurrentDepth + 1,
        maplist(descendants_with_depth(RelName, _, NextDepth), Children, Depths),
        max_list(Depths, MaxDepth)
    ).

% Count total nodes in subtree
subtree_size(TreeID, Size) :-
    descendants(pearltrees, TreeID, Descendants),
    length(Descendants, DescCount),
    Size is DescCount + 1.  % +1 for root
```

---

## Open Questions

1. **Cycle detection?**
   - Should we detect/prevent cycles automatically?
   - Or leave it to user queries?

2. **Bidirectional relationships?**
   - Automatically infer inverse (child→parent from parent→child)?

3. **Lazy loading?**
   - Load subtrees on-demand as user traverses?
   - Useful for interactive exploration?

4. **Persistence?**
   - Cache in-memory maps to disk?
   - Invalidation strategy?

5. **Graph relationships (not just trees)?**
   - Many-to-many relationships?
   - Different traversal algorithms (BFS, DFS, etc.)?

---

## Implementation Phases

### Phase 1: Streaming Relationships
- Declare parent-child relationships
- Stream-based children/3 queries
- Recursive hierarchy building
- **Estimated: 3-4 days**

### Phase 2: In-Memory Strategy
- AWK-based relationship map builder
- In-memory children/3 queries
- Strategy selection
- **Estimated: 2-3 days**

### Phase 3: Advanced Queries
- Path finding
- Descendants/ancestors
- Tree statistics
- **Estimated: 2 days**

### Phase 4: Optimizations
- Hybrid strategy with LRU cache
- Query planning
- Persistence
- **Estimated: 3-4 days**

---

## Success Criteria

1. ✅ Declare relationships in Prolog (declarative)
2. ✅ Configurable memory vs speed tradeoff
3. ✅ Streaming mode: constant memory
4. ✅ In-memory mode: sub-millisecond queries
5. ✅ Works on mobile (default streaming)
6. ✅ Recursive hierarchy building
7. ✅ Path finding and graph traversal
8. ✅ Composable with field extraction

---

## Related Design Documents

- `DECLARATIVE_FIELD_EXTRACTION.md` - Extract relationship data from XML
- `STREAMING_SGML_PARSING.md` - Parse relationship structures
- `OUTPUT_FORMAT_SPECIFICATION.md` - Format hierarchy results

---

**Recommendation:**
1. Start with Phase 1 (Streaming) as the safe default
2. Add Phase 2 (In-Memory) as opt-in for power users
3. Make it easy to switch strategies based on use case
4. Document tradeoffs clearly for users
