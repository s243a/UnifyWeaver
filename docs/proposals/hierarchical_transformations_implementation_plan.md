# Implementation Plan: Hierarchical Tree Transformations

## Overview

This document outlines the implementation plan for hierarchical tree transformation predicates in the Pearltrees UnifyWeaver example.

## File Structure

```
src/unifyweaver/examples/pearltrees/
├── sources.pl                    # Existing - data sources
├── queries.pl                    # Existing - aggregate queries and filters
├── hierarchy.pl                  # NEW - hierarchical transformations
├── templates.pl                  # Existing - output formats
├── test_queries.pl               # Existing - 36 tests
├── test_hierarchy.pl             # NEW - hierarchy tests
└── test_templates.pl             # Existing - 44 tests
```

## Implementation Phases

### Phase 1: Navigation Predicates

**Goal**: Build foundation for traversing tree hierarchies.

**Predicates**:
- `tree_parent/2` - Get immediate parent
- `tree_ancestors/2` - Get path to root
- `tree_depth/2` - Compute depth from root
- `tree_path/2` - Get full path as list

**Dependencies**: `sources.pl` (pearl_trees/5)

**Tests**: ~8 tests
- Root has no parent
- Leaf depth calculation
- Multi-level ancestor chains
- Path includes self

**Estimated effort**: Small - straightforward recursion

---

### Phase 2: Structural Queries

**Goal**: Identify special nodes in the hierarchy.

**Predicates**:
- `root_tree/1` - Find root nodes
- `leaf_trees/1` - Find nodes with no children
- `orphan_trees/1` - Find disconnected nodes
- `tree_descendants/2` - Get all nodes under a tree
- `tree_siblings/2` - Get nodes at same level
- `subtree_trees/2` - Enumerate subtree

**Dependencies**: Phase 1 predicates

**Tests**: ~10 tests
- Single root identification
- Multiple orphans detection
- Empty descendants for leaves
- Siblings exclude self

**Estimated effort**: Small - builds on Phase 1

---

### Phase 3: Path Operations

**Goal**: Manipulate path representations.

**Predicates**:
- `path_depth/2` - Count path elements
- `truncate_path/3` - Limit path depth
- `path_prefix/3` - Check path containment
- `common_ancestor/3` - Find shared ancestor

**Dependencies**: Phase 1 (tree_path/2)

**Tests**: ~6 tests
- Truncate long paths
- Truncate already-short paths (no-op)
- Common ancestor of siblings
- Common ancestor of cousins

**Estimated effort**: Small - list manipulation

---

### Phase 4: Basic Transformations

**Goal**: Core tree restructuring operations.

**Predicates**:
- `flatten_tree/3` - Collapse depth levels
- `prune_tree/3` - Remove branches by criteria
- `trees_at_depth/2` - Select by depth
- `trees_by_parent/2` - Group by parent

**Dependencies**: Phases 1-3

**Tests**: ~10 tests
- Flatten deep tree to depth 2
- Prune by max_depth
- Prune by has_type
- Group siblings correctly

**Estimated effort**: Medium - combines previous predicates

---

### Phase 5: Advanced Transformations

**Goal**: Complex operations for reorganization.

**Predicates**:
- `reroot_tree/3` - Change hierarchy root
- `merge_trees/3` - Combine multiple trees
- `group_by_ancestor/3` - Cluster by ancestor at depth

**Dependencies**: Phases 1-4

**Tests**: ~8 tests
- Reroot preserves structure
- Reroot handles outside nodes
- Merge with dedup
- Group by depth=1

**Estimated effort**: Medium - complex logic

---

### Phase 6: Integration

**Goal**: Connect with existing queries.pl and templates.pl.

**Tasks**:
- Export predicates from hierarchy.pl
- Use in queries.pl filters
- Add transformation examples to compile_examples.pl
- Update README.md documentation

**Dependencies**: Phases 1-5

**Tests**: ~4 integration tests
- Filter then transform
- Transform then generate output

**Estimated effort**: Small - wiring

---

## Mock Data Requirements

Extend `test_queries.pl` mock data or create new in `test_hierarchy.pl`:

```prolog
%% Mock hierarchy for testing:
%%
%%   root_1
%%   ├── science_2
%%   │   ├── physics_3
%%   │   │   └── quantum_6
%%   │   └── chemistry_4
%%   ├── arts_5
%%   │   └── music_7
%%   └── orphan_99 (disconnected - parent doesn't exist)
%%
%% Depths:
%%   root_1: 0
%%   science_2, arts_5: 1
%%   physics_3, chemistry_4, music_7: 2
%%   quantum_6: 3

setup_hierarchy_mock_data :-
    % Trees with cluster_id relationships
    assertz(mock_pearl_trees(tree, 'root_1', 'Root', 'uri:root_1', root)),
    assertz(mock_pearl_trees(tree, 'science_2', 'Science', 'uri:science_2', 'uri:root_1')),
    assertz(mock_pearl_trees(tree, 'physics_3', 'Physics', 'uri:physics_3', 'uri:science_2')),
    assertz(mock_pearl_trees(tree, 'chemistry_4', 'Chemistry', 'uri:chemistry_4', 'uri:science_2')),
    assertz(mock_pearl_trees(tree, 'arts_5', 'Arts', 'uri:arts_5', 'uri:root_1')),
    assertz(mock_pearl_trees(tree, 'quantum_6', 'Quantum', 'uri:quantum_6', 'uri:physics_3')),
    assertz(mock_pearl_trees(tree, 'music_7', 'Music', 'uri:music_7', 'uri:arts_5')),
    % Orphan - parent uri doesn't exist
    assertz(mock_pearl_trees(tree, 'orphan_99', 'Lost', 'uri:orphan_99', 'uri:nonexistent')).
```

## Test Summary

| Phase | Predicates | Tests | Status |
|-------|-----------|-------|--------|
| 1. Navigation | 7 | 25 | ✅ Complete |
| 2. Structural | 4 | 16 | ✅ Complete |
| 3. Path Ops | 5 | 15 | ✅ Complete |
| 4. Basic Transform | 4 | 14 | ✅ Complete |
| 5. Advanced Transform | 3 | 8 | ✅ Complete |
| 6. Integration | 12 | 3 | ✅ Complete |
| **Total Phases 1-6** | **35** | **81** | ✅ Complete |

Combined with existing tests:
- queries.pl: 36 tests
- hierarchy.pl: 81 tests
- templates.pl: 44 tests
- browser_automation.pl: 22 tests
- **Total: 183 tests**

## Dependencies Graph

```
Phase 1: Navigation
    │
    ├──► Phase 2: Structural Queries
    │        │
    │        └──► Phase 4: Basic Transforms
    │                  │
    │                  └──► Phase 5: Advanced Transforms
    │                             │
    └──► Phase 3: Path Operations ◄┘
              │
              └──► Phase 6: Integration
```

## Relationship to Python Implementation

| Python Function | Prolog Predicate | Phase | Notes |
|-----------------|------------------|-------|-------|
| `build_user_hierarchy()` | `tree_parent/2`, `tree_ancestors/2` | 1-2 | Prolog uses backtracking vs explicit dict |
| `curated_to_folder_structure()` | `flatten_tree/3`, `truncate_path/3` | 4 | Depth flattening |
| Orphan handling | `orphan_trees/1` | 2 | Same concept |
| `compute_cluster_centroids()` | `tree_centroid/2` | 7 | Via bindings to Go/Rust/Python |
| `cluster_trees_into_folder_groups()` | `cluster_trees/3` | 8 | Via component registry |
| `build_mst_with_fixed_root()` | `build_semantic_hierarchy/3` | 9 | Full curated folders equivalent |

**Phases 1-6**: Rule-based structural transformations (pure Prolog)
**Phases 7-9**: Semantic transformations (via bindings/components/glue)

## Success Criteria

1. ✅ **All tests pass**: 81 tests for hierarchy.pl (exceeded target of 46+)
2. ✅ **Documentation complete**: README updated with all predicates
3. ✅ **Integration working**: Hierarchy filters integrated into queries.pl
4. ✅ **Clear examples**: Mock data hierarchy demonstrates patterns

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Cycle detection (tree with ancestor pointing to descendant) | Add explicit check in tree_ancestors |
| Performance on large hierarchies | Document that Prolog is for specification, not production |
| Complex reroot logic | Start with simple cases, add edge cases incrementally |

## Completed Steps (Phases 1-6)

1. ✅ Created `hierarchy.pl` with Phase 1-5 predicates
2. ✅ Created `test_hierarchy.pl` with 81 tests and mock data
3. ✅ Implemented and tested each phase incrementally
4. ✅ Updated README documentation
5. ✅ Integrated hierarchy filters into queries.pl (Phase 6)
6. ✅ Created PR and merged

## Completed Steps (Phases 7-9)

1. ✅ Explored UnifyWeaver bindings, components, and glue infrastructure
2. ✅ Created `semantic_hierarchy.pl` with Phases 7-9 predicates
3. ✅ Created `test_semantic_hierarchy.pl` with 19 tests
4. ✅ Implemented Phase 7: Embedding predicates (tree_embedding, child_embedding, tree_centroid)
5. ✅ Implemented Phase 8: Clustering predicates (tree_similarity, cluster_trees, k-means)
6. ✅ Implemented Phase 9: Semantic hierarchy (semantic_group, build_semantic_hierarchy, curated_folder_structure)
7. ✅ Updated README with Phases 7-9 documentation
8. ✅ All 100 tests passing (81 hierarchy + 19 semantic)

## Completed: Alias Handling and Cycle Detection

9. ✅ Implemented cycle detection (`has_cycle/1`, `cycle_free_path/2`)
10. ✅ Added `follow_aliases` option to traversal predicates
11. ✅ Implemented alias resolution (`alias_target/2`, `alias_children/2`, `tree_aliases/2`)
12. ✅ All traversal predicates now use cycle-safe implementations
13. ✅ Added 24 new tests (105 hierarchy + 19 semantic = 124 total)

## Resolved Questions

1. **~~Should we support cycles?~~** ✅ RESOLVED: Yes, cycles are detected and handled gracefully. `has_cycle/1` detects cycles, and all traversal predicates use cycle-safe implementations that stop at repeated nodes.

2. **How to handle multiple roots?** Multi-account scenarios may have multiple disconnected hierarchies. Current plan: treat as separate hierarchies, enumerate with `root_tree/1`.

3. **Should transformations be lazy or eager?** Current plan: eager (compute full result). Could add generator-style for large hierarchies.

## Implemented: AliasPearl Handling

AliasPearls are now fully supported for cross-account traversal:

```prolog
%% AliasPearls link trees across accounts
%% pearl_children(TreeId, alias, Title, Order, null, SeeAlsoUri)
%%   SeeAlsoUri → target tree in another account

%% New predicates:
alias_target(SeeAlsoUri, TargetTreeId).     % Resolve alias to tree
alias_children(TreeId, AliasTargets).       % Get all alias targets
tree_aliases(TreeId, AliasInfo).            % Get detailed alias info

%% Traversal predicates should support:
tree_descendants(TreeId, Descendants) :-
    tree_descendants(TreeId, [follow_aliases(true)], Descendants).

tree_descendants(TreeId, Options, Descendants) :-
    option(follow_aliases(FollowAliases), Options, true),
    ...
```

This is essential for:
- **Semantic embeddings**: Hierarchical context must span accounts
- **Curated folders**: Complete hierarchy includes cross-account links
- **User's mental model**: Users organize across accounts via aliases

---

## Later Phases: Semantic Integration

The following phases extend the structural primitives with semantic capabilities using UnifyWeaver's existing infrastructure.

### Phase 7: Embedding Predicates

**Goal**: Integrate semantic embeddings via bindings.

**Predicates**:
- `tree_embedding/2` - Get embedding for tree content
- `tree_centroid/2` - Compute centroid from children embeddings
- `child_embedding/2` - Get embedding for child item

**Dependencies**:
- Phases 1-6 (structural predicates)
- UnifyWeaver bindings system
- Existing embedding tools (Go, Rust, Python)

**Implementation**:
```prolog
% Bindings to target-specific implementations
:- binding(go, compute_embedding/2, 'semantic.Embed', [string], [list(float)], []).
:- binding(rust, compute_embedding/2, 'embed::compute', [string], [list(float)], []).
:- binding(python, compute_embedding/2, 'embeddings.embed', [string], [list(float)], []).
```

**Tests**: ~6 tests (mock embeddings for unit tests, integration tests with real backends)

---

### Phase 8: Clustering Predicates

**Goal**: Semantic clustering via component registry.

**Predicates**:
- `tree_similarity/3` - Cosine similarity between trees
- `most_similar_trees/3` - K nearest neighbors
- `cluster_trees/3` - K-means clustering
- `cluster_by_centroid/4` - Clustering with method selection

**Dependencies**:
- Phase 7 (embeddings)
- UnifyWeaver component registry
- Existing Go semantic tools

**Implementation**:
```prolog
% Component registration
:- declare_component(runtime, clustering, kmeans, [
    implementation(go),
    depends([embedding_provider])
]).
```

**Tests**: ~8 tests (mock centroids, verify clustering properties)

---

### Phase 9: Semantic Hierarchy

**Goal**: Full curated folders equivalent in UnifyWeaver.

**Predicates**:
- `semantic_group/3` - Assign tree to semantic group
- `build_semantic_hierarchy/3` - Complete curated folders pipeline

**Dependencies**:
- Phases 7-8 (embeddings, clustering)
- UnifyWeaver cross-target glue
- Structural predicates (flatten, reroot)

**Implementation**:
```prolog
% Pipeline declaration
:- declare_target(compute_embeddings/2, go, [file('cmd/embed/main.go')]).
:- declare_target(cluster_centroids/3, go, [file('cmd/cluster/main.go')]).
:- declare_target(build_hierarchy/2, prolog, []).
```

**Tests**: ~6 tests (end-to-end with mock data)

---

## Updated Test Summary

| Phase | Predicates | Tests | Type | Status |
|-------|-----------|-------|------|--------|
| 1. Navigation | 7 | 25 | Pure Prolog | ✅ Complete |
| 2. Structural | 4 | 16 | Pure Prolog | ✅ Complete |
| 3. Path Ops | 5 | 15 | Pure Prolog | ✅ Complete |
| 4. Basic Transform | 4 | 14 | Pure Prolog | ✅ Complete |
| 5. Advanced Transform | 3 | 8 | Pure Prolog | ✅ Complete |
| 6. Integration | 12 | 3 | Pure Prolog | ✅ Complete |
| 7. Embeddings | 6 | 6 | Bindings | ✅ Complete |
| 8. Clustering | 4 | 8 | Components | ✅ Complete |
| 9. Semantic Hierarchy | 2 | 5 | Glue | ✅ Complete |
| **Total** | **47** | **100** | | ✅ Complete |

**Breakdown by file:**
- `test_hierarchy.pl`: 81 tests (Phases 1-6)
- `test_semantic_hierarchy.pl`: 19 tests (Phases 7-9)

## Updated Dependencies Graph

```
Phase 1: Navigation
    │
    ├──► Phase 2: Structural Queries
    │        │
    │        └──► Phase 4: Basic Transforms
    │                  │
    │                  └──► Phase 5: Advanced Transforms
    │                             │
    └──► Phase 3: Path Operations ◄┘
              │
              └──► Phase 6: Integration
                        │
                        ▼
              ┌─────────────────────┐
              │  SEMANTIC PHASES    │
              │  (Later)            │
              └─────────────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   │
        Phase 7: Embeddings       │
        (Bindings)                │
              │                   │
              ▼                   │
        Phase 8: Clustering       │
        (Components)              │
              │                   │
              ▼                   │
        Phase 9: Semantic    ◄────┘
        Hierarchy (Glue)
```

## Existing Tools to Leverage

| Tool | Location | Use in Semantic Phases |
|------|----------|------------------------|
| Go embeddings | Built into UnifyWeaver targets | Phase 7 bindings |
| Rust embeddings | `/examples/pearltrees/` | Phase 7 alternative |
| Python ONNX | Semantic source plugin | Phase 7 fallback |
| Go clustering | To be implemented | Phase 8 component |
| Cross-target glue | `/src/unifyweaver/glue/` | Phase 9 orchestration |

## Long-Term Vision

After Phase 9, the full curated folders algorithm can be expressed as:

```prolog
%% Declarative specification - generates code for any target
curated_folder_structure(TreeIds, Options, FolderAssignments) :-
    % Structural phase (Phases 1-6)
    identify_root(TreeIds, Options, RootId),
    identify_orphans(TreeIds, OrphanIds),

    % Semantic phase (Phases 7-9)
    build_semantic_hierarchy(TreeIds, Options, Hierarchy),

    % Output
    hierarchy_to_folders(Hierarchy, FolderAssignments).
```

This single specification could then generate production code for Python, Go, or Rust.
