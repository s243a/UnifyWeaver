# Pearltrees Processing Example

Educational example demonstrating UnifyWeaver's capabilities for data processing and code generation.

## Purpose

This example shows how UnifyWeaver can:

1. **Define data sources** - SQLite, JSONL, runtime JSON
2. **Write aggregate queries** - Grouping, counting, filtering
3. **Generate target code** - Same logic → Python, C#, Go, etc.

**Note**: This does NOT replace the existing Python tools in `.local/tools/browser-automation/`. Each target generates its own database access and runtime code.

## Files

| File | Description |
|------|-------------|
| `sources.pl` | Data source definitions with target-specific config |
| `queries.pl` | Aggregate queries for tree/pearl data |
| `templates.pl` | SMMX XML generation from tree data |
| `compile_examples.pl` | Cross-target code generation examples |
| `browser_automation.pl` | Abstract browser automation workflow |
| `test_queries.pl` | 15 plunit tests for queries |
| `test_templates.pl` | 16 plunit tests for templates |
| `test_browser_automation.pl` | 22 plunit tests for browser automation |

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

## Running Tests

```bash
# Run query tests (15 tests)
swipl -g "run_tests" -t halt src/unifyweaver/examples/pearltrees/test_queries.pl

# Run template tests (16 tests)
swipl -g "run_tests" -t halt src/unifyweaver/examples/pearltrees/test_templates.pl

# Run browser automation tests (22 tests)
swipl -g "run_tests" -t halt src/unifyweaver/examples/pearltrees/test_browser_automation.pl
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

See `docs/proposals/pearltrees_unifyweaver_native.md` for the full proposal.
