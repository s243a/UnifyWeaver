# Philosophy: UnifyWeaver-Native Hierarchical Transformations

## Why UnifyWeaver-Native?

The existing Python `--curated-folders` implementation in `generate_mindmap.py` solves the real problem of organizing large hierarchical bookmark collections. It works. So why create a UnifyWeaver-native version?

### 1. Declarative Expression

The Python implementation mixes **what** with **how**:

```python
# Python: Imperative - describes HOW to flatten
def curated_to_folder_structure(tree_to_group, group_hierarchy, max_depth):
    result = {}
    for tree_url, group_id in tree_to_group.items():
        path = compute_path(group_id, group_hierarchy)
        if len(path.parts) > max_depth:
            path = Path(*path.parts[:max_depth])  # Truncate
        result[tree_url] = path
    return result
```

UnifyWeaver expresses **what** we want:

```prolog
% Prolog: Declarative - describes WHAT flattening means
flattened_path(TreeId, Path, MaxDepth) :-
    tree_path(TreeId, FullPath),
    truncate_path(FullPath, MaxDepth, Path).
```

The declarative form:
- Is easier to reason about
- Can be compiled to multiple targets
- Separates transformation logic from execution

### 2. Composability

Python functions are composed by calling each other. Prolog predicates compose through unification:

```prolog
% Combine transformations naturally
transformed_tree(TreeId, FinalPath) :-
    tree_with_children(TreeId, _, Children),
    filter_children(Children, pagepearl, FilteredChildren),
    flatten_to_depth(TreeId, 3, FlatPath),
    semantic_group(TreeId, GroupId),
    group_path(GroupId, FlatPath, FinalPath).
```

Each predicate is independently testable and reusable.

### 3. Multi-Target Generation

The same transformation logic can generate:
- **Python**: For integration with existing tools
- **SQL**: For database-level transformations
- **Go**: For CLI tools
- **Direct execution**: In SWI-Prolog for prototyping

### 4. Educational Value

These examples demonstrate UnifyWeaver's capabilities for:
- Tree traversal and recursion
- Aggregation over hierarchies
- Constraint-based transformations
- Path manipulation

## Design Principles

### Principle 1: Preserve User Intent

The user's hierarchy represents their mental model. Transformations should:
- **Preserve the root** - Start from where the user starts
- **Maintain relationships** - Parent links follow actual hierarchy, not folder structure
- **Handle orphans explicitly** - Don't silently drop disconnected nodes

### Principle 2: Separation of Concerns

| Concern | Responsibility |
|---------|---------------|
| **Structure** | How trees relate to each other (parent/child) |
| **Organization** | How trees are grouped into folders |
| **Presentation** | How folders are named and paths are computed |
| **Clustering** | Semantic similarity (embeddings, centroids) |

Each concern should be addressable independently.

### Principle 3: Composable Primitives

Build complex transformations from simple primitives:

| Primitive | Purpose |
|-----------|---------|
| `tree_depth/2` | Compute depth from root |
| `tree_path/2` | Compute path from root |
| `tree_ancestors/2` | List ancestors to root |
| `tree_descendants/2` | List all descendants |
| `subtree/2` | Extract subtree rooted at node |
| `flatten_to_depth/3` | Truncate paths at depth |
| `reroot_at/3` | Change root, adjust paths |

### Principle 4: Explicit Over Implicit

Make transformations explicit:

```prolog
% BAD: Implicit flattening
generate_paths(Trees, Paths) :-
    ... % Hidden depth limit?

% GOOD: Explicit transformation
generate_paths(Trees, Paths, Options) :-
    option(max_depth(MaxDepth), Options, infinite),
    maplist(flatten_to_depth(MaxDepth), Trees, Paths).
```

### Principle 5: Testability

Every predicate should be testable with mock data:

```prolog
test(flatten_deep_tree) :-
    % Setup: 5-level tree
    mock_tree(deep, TreeId),
    tree_depth(TreeId, 5),
    % Transform
    flatten_to_depth(TreeId, 3, Path),
    % Verify
    path_depth(Path, 3).
```

## Relationship to Python Implementation

The UnifyWeaver-native approach **complements** the Python implementation:

| Python Implementation | UnifyWeaver Native |
|-----------------------|-------------------|
| Production tool | Educational examples |
| Embedding-based clustering | Rule-based transformations |
| LLM folder naming | Template-based naming |
| Performance optimized | Clarity optimized |

### What We're NOT Doing (Initially)

- **Not replacing** `generate_mindmap.py` (yet - it's the working implementation)
- **Not reimplementing** embedding computation from scratch
- **Not duplicating** LLM integration

### What We ARE Doing

- **Expressing** hierarchical transformations declaratively
- **Demonstrating** UnifyWeaver's tree manipulation capabilities
- **Creating** reusable transformation primitives
- **Enabling** multi-target code generation for tree operations
- **Testing** how general, composable, and extensible UnifyWeaver is
- **Building toward** eventually generating these tools from UnifyWeaver

### Long-Term Vision

The existing Python tools are the working implementation today. In the long run, these tools could potentially be **generated** from UnifyWeaver specifications:

```prolog
% Declarative specification
curated_folder_structure(TreeId, FolderPath) :-
    tree_centroid(TreeId, Centroid),           % Semantic embedding
    cluster_trees(Centroid, K, GroupId),       % Clustering
    group_hierarchy(GroupId, HierarchyPath),   % MST with fixed root
    flatten_to_depth(HierarchyPath, MaxDepth, FolderPath).
```

This single specification could generate:
- **Python**: Integration with existing tools, NumPy for embeddings
- **Go**: Native CLI with built-in semantic embeddings
- **Rust**: High-performance processing (like existing `/examples/pearltrees/`)

Anything needing a specific implementation becomes a **custom function** via the bindings system, and UnifyWeaver's **cross-target glue** handles multi-language orchestration.

## UnifyWeaver Capabilities Supporting This Vision

UnifyWeaver already has the infrastructure for semantic tree organization:

### Existing Semantic Embedding Tools

| Target | Implementation | Notes |
|--------|----------------|-------|
| **Go** | Native multi-head LDA projection | No Python overhead, built-in |
| **Rust** | candle-transformers (ModernBERT) | High-performance, GPU support |
| **Python** | ONNX-based embeddings | GPU acceleration, flash attention |

### Bindings System

Map Prolog predicates to target-specific implementations:

```prolog
% Declare that compute_centroid/2 uses target-specific implementations
:- binding(python, compute_centroid/2, 'numpy.mean', [list(float)], [list(float)], [import(numpy)]).
:- binding(go, compute_centroid/2, 'vectors.Mean', [list(float)], [list(float)], []).
:- binding(rust, compute_centroid/2, 'ndarray::mean', [list(float)], [list(float)], []).
```

### Component Registry

Register embedding providers and clustering algorithms:

```prolog
:- declare_component(runtime, embedding_provider, bert_embeddings, [
    model('all-MiniLM-L6-v2'),
    dimensions(384),
    initialization(lazy)
]).

:- declare_component(runtime, clustering, kmeans, [
    implementation(go),  % Use Go for performance
    depends([embedding_provider])
]).
```

### Cross-Target Glue

Orchestrate multi-language pipelines:

```prolog
% Prolog specifies the pipeline
:- declare_target(compute_embeddings/2, python, [file('embed.py')]).
:- declare_target(cluster_trees/3, go, [file('cluster.go')]).
:- declare_target(generate_paths/2, prolog, []).

% Glue automatically handles data flow between targets
```

### Graph RAG Pattern

The education materials document a pattern directly applicable to curated folders:

1. **Anchor**: Vector search to find semantically similar trees
2. **Traverse**: Use graph relationships (tree_ancestors, tree_descendants)
3. **Synthesize**: Cluster and organize based on combined signals

### Existing Pearltrees Example (Rust)

`/examples/pearltrees/` already implements semantic bookmark filing:
- BERT embeddings (all-MiniLM-L6-v2, 384 dimensions)
- 11,867 XML fragments indexed
- Tree context (ancestor/sibling relationships)
- Sub-second semantic queries

This provides a reference implementation for the semantic aspects.

## Transformation Categories

### 1. Structural Transformations

Change the shape of the tree:
- **Flatten**: Reduce depth
- **Prune**: Remove branches by criteria
- **Graft**: Attach subtrees elsewhere
- **Split**: Divide large trees

### 2. Path Transformations

Change how paths are computed:
- **Reroot**: Change the starting point
- **Abbreviate**: Shorten path components
- **Namespace**: Add prefixes for multi-account

### 3. Grouping Transformations

Change how trees are organized:
- **Cluster by parent**: Preserve hierarchy
- **Cluster by type**: Group similar content
- **Cluster by depth**: Organize by level

### 4. Filter Transformations

Select subsets:
- **By depth**: Trees at specific levels
- **By content**: Trees with specific children
- **By connectivity**: Connected vs orphan trees

## Success Criteria

1. **Clarity**: Someone reading the Prolog can understand what transformation occurs
2. **Testability**: All predicates have unit tests with mock data
3. **Composability**: Primitives can be combined for complex transformations
4. **Target Generation**: At least Python target generates working code
5. **Documentation**: Examples show common transformation patterns

## Non-Goals

- Performance parity with Python (Prolog is for specification, not production)
- Full embedding/ML integration (use Python for that)
- Replacing existing tools (complement, not replace)
- Supporting all edge cases (focus on common patterns)

## Next Steps

1. **Specification**: Define predicate signatures and behaviors
2. **Implementation**: Write Prolog predicates with tests
3. **Examples**: Show common transformation patterns
4. **Integration**: Connect to existing Pearltrees example
