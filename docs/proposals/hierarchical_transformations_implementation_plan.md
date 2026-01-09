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

| Python Function | Prolog Predicate | Notes |
|-----------------|------------------|-------|
| `build_user_hierarchy()` | `tree_parent/2`, `tree_ancestors/2` | Prolog uses backtracking vs explicit dict |
| `curated_to_folder_structure()` | `flatten_tree/3`, `truncate_path/3` | Depth flattening |
| Orphan handling | `orphan_trees/1` | Same concept |
| `cluster_trees_into_folder_groups()` | Not in scope | Requires embeddings |
| `build_mst_with_fixed_root()` | `reroot_tree/3` (partial) | MST needs graph library |

The Prolog predicates provide **rule-based** transformations that complement the **embedding-based** Python implementation.

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
