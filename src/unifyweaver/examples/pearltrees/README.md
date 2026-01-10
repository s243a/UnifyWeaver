# Pearltrees Processing Example

Educational example demonstrating UnifyWeaver's capabilities for data processing, multi-format output, and code generation.

## Purpose

This example shows how UnifyWeaver can:

1. **Define data sources** - SQLite, JSONL, runtime JSON
2. **Write aggregate queries** - Grouping, counting, filtering
3. **Generate multiple output formats** - Direct output to various mindmap formats
4. **Generate target code** - Same logic → Python, C#, Go, etc.

**Note**: This does NOT replace the existing Python tools in `.local/tools/browser-automation/`. Each target generates its own database access and runtime code.

## Files

| File | Description |
|------|-------------|
| `sources.pl` | Data source definitions with target-specific config |
| `queries.pl` | Aggregate queries and composable filters for tree/pearl data |
| `templates.pl` | Multi-format output (SMMX, FreeMind, OPML, GraphML, VUE, Mermaid) |
| `compile_examples.pl` | Cross-target code generation examples |
| `browser_automation.pl` | Abstract browser automation workflow |
| `hierarchy.pl` | Hierarchical tree transformations |
| `test_queries.pl` | 36 plunit tests for queries and filters |
| `test_templates.pl` | 44 plunit tests for templates (all formats) |
| `test_browser_automation.pl` | 22 plunit tests for browser automation |
| `test_hierarchy.pl` | 81 plunit tests for hierarchy predicates |

## Source Definitions

Sources define where data comes from, with per-target configuration:

```prolog
:- source(sqlite, pearl_children, [
    table(children),
    columns([parent_tree_id, pearl_type, title, ...]),
    target_config(python, [db_path('...')]),
    target_config(csharp, [async(true)]),
    target_config(go, [driver('go-sqlite3')])
]).
```

Each target generates idiomatic database access:
- **Python**: `sqlite3` module
- **C#**: `Microsoft.Data.Sqlite` with async
- **Go**: `database/sql` with driver

## Aggregate Queries

Queries use `aggregate_all/4` for grouping:

```prolog
tree_with_children(TreeId, Title, Children) :-
    pearl_trees(tree, TreeId, Title, _, _),
    aggregate_all(
        bag(child(Type, ChildTitle, Url, Order)),
        pearl_children(TreeId, Type, ChildTitle, Order, Url, _),
        TreeId,
        Children
    ).
```

Generated code per target:
- **Python**: `itertools.groupby` or dict comprehension
- **C#**: LINQ `GroupBy().Select()`
- **Go**: `map[string][]Child` with append
- **SQL**: `GROUP BY` with `JSON_AGG`

## Query-Based Filtering

Composable filter predicates for selecting trees and children:

### Filter Types

| Filter | Predicate | Description |
|--------|-----------|-------------|
| Domain | `has_domain_links/2` | Trees with links to a domain |
| Type | `has_child_type/2` | Trees containing specific child types |
| Title | `title_contains/2` | Title/content keyword search |
| Count | `trees_with_min_children/2` | Trees with minimum child count |
| Combined | `apply_filters/3` | Multiple filters at once |

### Example: Find Trees with GitHub Links

```prolog
?- has_domain_links(TreeId, 'github.com').
TreeId = '12347'.
```

### Example: Combined Filters

```prolog
%% Find complete trees with pagepearl children
?- apply_filters([complete, type(pagepearl)], TreeId, Info).
Info = tree_info('12345', 'Science Topics', 3).

%% Find trees with GitHub links and at least 2 children
?- apply_filters([domain('github.com'), min_children(2)], TreeId, Info).
Info = tree_info('12347', 'Tech Links', 2).

%% Find incomplete trees (negation)
?- apply_filters([not(min_children(2))], TreeId, Info).
Info = tree_info('12346', 'Empty Tree', 1).
```

### Available Filters

- `domain(Domain)` - Trees with links to domain
- `type(Type)` - Trees with children of type (pagepearl, tree, section, alias)
- `title_match(Pattern)` - Case-insensitive title search
- `min_children(N)` - At least N children
- `max_children(N)` - At most N children
- `incomplete` - Trees with ≤1 children
- `complete` - Trees with >1 children
- `cluster(ClusterId)` - Trees in specific cluster
- `not(Filter)` - Negation of any filter

## Output Formats

Direct output to multiple mindmap formats from the same tree data:

| Format | Predicate | Extension | Compatible With |
|--------|-----------|-----------|-----------------|
| SMMX | `generate_mindmap/4` | `.smmx` | SimpleMind |
| FreeMind | `generate_freemind/4` | `.mm` | FreeMind, Freeplane, XMind |
| OPML | `generate_opml/4` | `.opml` | Workflowy, Dynalist, OmniOutliner |
| GraphML | `generate_graphml/4` | `.graphml` | yEd, Gephi, Cytoscape |
| VUE | `generate_vue/4` | `.vue` | Tufts VUE |
| Mermaid | `generate_mermaid/4` | `.md` | GitHub, GitLab, Obsidian |

### Example: Generate Multiple Formats

```prolog
?- Children = [child(pagepearl, 'Link', 'http://example.com', 1)],
   generate_freemind('123', 'My Tree', Children, MM),
   generate_opml('123', 'My Tree', Children, OPML),
   generate_mermaid('123', 'My Tree', Children, Mermaid).
```

### Mermaid Output Example

```mermaid
mindmap
  root((Science))
    pearl_1[Wikipedia]
    tree_2{{Physics}}
    section_3(Resources)
```

### Unified Multi-Format Generation

Generate multiple formats at once:

```prolog
?- Children = [child(pagepearl, 'Link', 'http://example.com', 1)],
   generate_all_formats('123', 'My Tree', Children, [freemind, opml, mermaid], Results).
% Results = [format(freemind, '.mm', '...'), format(opml, '.opml', '...'), ...]
```

Available predicates:
- `available_formats/1` - List all supported formats
- `generate_all_formats/5` - Generate multiple formats at once
- `generate_for_format/5` - Generate a single format (in compile_examples.pl)

## Usage

### Generate Python Code

```prolog
?- use_module('src/unifyweaver/examples/pearltrees/queries'),
   compile_predicate_to_python(tree_with_children/3, [mode(generator)], Code),
   format('~s~n', [Code]).
```

### Generate C# Code

```prolog
?- compile_predicate_to_csharp(incomplete_tree/2, [async(true)], Code).
```

### Generate Go Code

```prolog
?- compile_predicate_to_go(tree_child_count/2, [], Code).
```

## Hierarchical Transformations

Navigate, query, and transform Pearltrees hierarchies:

### Navigation Predicates

| Predicate | Description |
|-----------|-------------|
| `tree_parent/2` | Get immediate parent of a tree |
| `tree_ancestors/2` | Get path from root to tree |
| `tree_descendants/2` | Get all descendants recursively |
| `tree_siblings/2` | Get sibling trees |
| `tree_depth/2` | Compute depth from root |
| `tree_path/2` | Get full path as list |
| `tree_title/2` | Get tree title |

### Structural Queries

| Predicate | Description |
|-----------|-------------|
| `root_tree/1` | Identify root nodes |
| `leaf_tree/1` | Identify leaf nodes |
| `orphan_tree/1` | Find disconnected trees |
| `subtree_tree/2` | Check subtree membership |

### Path Operations

| Predicate | Description |
|-----------|-------------|
| `path_depth/2` | Count path elements |
| `truncate_path/3` | Limit path depth |
| `common_ancestor/3` | Find shared ancestor of two trees |
| `hierarchical_title_path/2` | Get title path from root |

### Basic Transformations

| Predicate | Description |
|-----------|-------------|
| `flatten_tree/3` | Collapse depth levels |
| `prune_tree/3` | Remove branches by criteria |
| `trees_at_depth/2` | Select trees at depth |
| `trees_by_parent/2` | Group trees by parent |

### Advanced Transformations

| Predicate | Description |
|-----------|-------------|
| `reroot_tree/3` | Change hierarchy root |
| `merge_trees/3` | Combine multiple tree lists |
| `group_by_ancestor/3` | Cluster by ancestor at depth |

### Embedding Support

| Predicate | Description |
|-----------|-------------|
| `structural_embedding_input/3` | Generate embedding text format |
| `format_id_path/2` | Format slash-separated ID path |
| `format_title_hierarchy/2` | Format indented title hierarchy |

Example embedding format:
```
/root_1/science_2/physics_3
- Root
  - Science
    - Physics
      - Quantum Mechanics
```

See `docs/proposals/hierarchical_transformations_specification.md` for the full specification.

## Running Tests

```bash
# Run query tests (36 tests)
swipl -g "run_tests" -t halt src/unifyweaver/examples/pearltrees/test_queries.pl

# Run template tests (44 tests)
swipl -g "run_tests" -t halt src/unifyweaver/examples/pearltrees/test_templates.pl

# Run browser automation tests (22 tests)
swipl -g "run_tests" -t halt src/unifyweaver/examples/pearltrees/test_browser_automation.pl

# Run hierarchy tests (81 tests)
swipl -g "run_tests" -t halt src/unifyweaver/examples/pearltrees/test_hierarchy.pl
```

## Browser Automation Workflow

Abstract workflow for browser-based data fetching:

```prolog
% Workflow steps are abstract - concrete API details in external config
workflow_step(fetch_tree, 1, step(navigate, tree_page, [])).
workflow_step(fetch_tree, 2, step(wait, page_load, [seconds(3)])).
workflow_step(fetch_tree, 3, step(fetch, tree_api, [])).
workflow_step(fetch_tree, 4, step(parse, tree_response, [])).
```

API endpoints and URLs come from `.local/tools/browser-automation/api_config.json`:

```json
{
  "endpoints": {
    "tree_api": {
      "url_template": "https://www.pearltrees.com/s/.../getTreeAndPearls?treeId={tree_id}"
    }
  },
  "urls": {
    "tree_page": {
      "template": "https://www.pearltrees.com/{account}/{slug}/id{tree_id}"
    }
  }
}
```

## Cross-Target Examples

See generated code examples for Python, C#, and Go:

```prolog
?- use_module('src/unifyweaver/examples/pearltrees/compile_examples').
?- show_target_comparison.
?- demo_python_generation.
?- demo_csharp_generation.
?- demo_go_generation.
```

## Relationship to Existing Tools

| Existing Tool | UnifyWeaver Equivalent |
|---------------|----------------------|
| `build_children_index.py` | `pearl_children/6` source + SQLite target |
| `generate_mindmap.py` | `tree_with_children/3` + template |
| `scan_incomplete_mindmaps.py` | `incomplete_tree/2` query |
| `batch_repair.py` | `browser_automation.pl` workflow + `api_config.json` |

The existing Python tools remain the production implementation. These UnifyWeaver examples show how similar functionality could be generated for multiple targets from a single declarative specification.

## Educational Value

This example demonstrates:

1. **Declarative Data Access**: Sources abstract database details
2. **Composable Queries**: Predicates build on each other
3. **Multi-Target Generation**: Same logic, different languages
4. **Aggregate Patterns**: Grouping, counting, filtering
5. **Composable Filters**: Combine filters with `apply_filters/3`

See `docs/proposals/pearltrees_unifyweaver_native.md` for the full proposal.
