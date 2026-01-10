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

| Phase | Predicates | Tests |
|-------|-----------|-------|
| 1. Navigation | 4 | ~8 |
| 2. Structural | 6 | ~10 |
| 3. Path Ops | 4 | ~6 |
| 4. Basic Transform | 4 | ~10 |
| 5. Advanced Transform | 3 | ~8 |
| 6. Integration | - | ~4 |
| **Total** | **21** | **~46** |

Combined with existing tests:
- queries.pl: 36 tests
- hierarchy.pl: 46 tests
- templates.pl: 44 tests
- browser_automation.pl: 22 tests
- **Total: ~148 tests**

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

1. **All tests pass**: 46+ tests for hierarchy.pl
2. **Documentation complete**: README updated, examples added
3. **Integration working**: Can combine with filters and templates
4. **Clear examples**: Show common transformation patterns

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Cycle detection (tree with ancestor pointing to descendant) | Add explicit check in tree_ancestors |
| Performance on large hierarchies | Document that Prolog is for specification, not production |
| Complex reroot logic | Start with simple cases, add edge cases incrementally |

## Next Steps After Design Approval

1. Create `hierarchy.pl` with Phase 1 predicates
2. Create `test_hierarchy.pl` with mock data
3. Implement and test each phase incrementally
4. Update documentation
5. Create PR

## Open Questions

1. **Should we support cycles?** Trees can have aliases that create logical cycles. Current plan: detect and fail gracefully.

2. **How to handle multiple roots?** Multi-account scenarios may have multiple disconnected hierarchies. Current plan: treat as separate hierarchies, enumerate with `root_tree/1`.

3. **Should transformations be lazy or eager?** Current plan: eager (compute full result). Could add generator-style for large hierarchies.

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

| Phase | Predicates | Tests | Type |
|-------|-----------|-------|------|
| 1. Navigation | 4 | ~8 | Pure Prolog |
| 2. Structural | 6 | ~10 | Pure Prolog |
| 3. Path Ops | 4 | ~6 | Pure Prolog |
| 4. Basic Transform | 4 | ~10 | Pure Prolog |
| 5. Advanced Transform | 3 | ~8 | Pure Prolog |
| 6. Integration | - | ~4 | Pure Prolog |
| 7. Embeddings | 3 | ~6 | Bindings |
| 8. Clustering | 4 | ~8 | Components |
| 9. Semantic Hierarchy | 2 | ~6 | Glue |
| **Total** | **30** | **~66** | |

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
