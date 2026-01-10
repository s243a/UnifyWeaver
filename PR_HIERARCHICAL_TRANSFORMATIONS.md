# docs: Add design docs for hierarchical tree transformations

## Summary

- Add three design documents for UnifyWeaver-native hierarchical tree transformations
- Define 30 predicates for tree traversal, restructuring, path manipulation, and semantic clustering
- Plan 9 implementation phases: 6 structural (pure Prolog) + 3 semantic (bindings/components/glue)
- Target ~66 tests total

## Documents Created

| Document | Purpose |
|----------|---------|
| `hierarchical_transformations_philosophy.md` | Design principles, long-term vision, UnifyWeaver capabilities |
| `hierarchical_transformations_specification.md` | 30 predicate signatures with examples, including semantic predicates |
| `hierarchical_transformations_implementation_plan.md` | 9 phases, dependencies, existing tools to leverage |

## Implementation Phases

**Structural (Pure Prolog)**:
- Phase 1-2: Navigation & structural queries
- Phase 3-4: Path operations & basic transforms
- Phase 5-6: Advanced transforms & integration

**Semantic (Later Phases)**:
- Phase 7: Embeddings via bindings (Go, Rust, Python)
- Phase 8: Clustering via component registry
- Phase 9: Semantic hierarchy via cross-target glue

## Key Capabilities

**Structural Predicates** (Phases 1-6):
- Navigation: `tree_parent/2`, `tree_ancestors/2`, `tree_depth/2`, `tree_path/2`
- Transforms: `flatten_tree/3`, `prune_tree/3`, `reroot_tree/3`, `merge_trees/3`
- Grouping: `trees_at_depth/2`, `trees_by_parent/2`, `group_by_ancestor/3`

**Semantic Predicates** (Phases 7-9):
- Embeddings: `tree_embedding/2`, `tree_centroid/2`
- Clustering: `cluster_trees/3`, `cluster_by_centroid/4`
- Integration: `build_semantic_hierarchy/3` (curated folders equivalent)

## UnifyWeaver Integration

Leverages existing infrastructure:
- **Bindings**: Map predicates to Go/Rust/Python embedding implementations
- **Components**: Register clustering algorithms with dependencies
- **Cross-target glue**: Orchestrate multi-language pipelines

## Existing Tools

| Tool | Use |
|------|-----|
| Go embeddings | Native multi-head LDA projection |
| Rust embeddings | candle-transformers (ModernBERT) |
| `/examples/pearltrees/` | Reference Rust semantic filing implementation |

## Test plan

- [x] Documents follow existing proposal format
- [x] Predicates build on existing queries.pl foundation
- [x] Phases 1-6: ~46 tests (pure Prolog)
- [x] Phases 7-9: ~20 tests (integration with bindings/components)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
