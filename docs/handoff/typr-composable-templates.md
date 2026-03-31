# Handoff: Wire TypR Composable Templates into compile_tc_from_template

## Goal
Switch the TypR transitive closure compilation from `typr_target.pl`
(which uses `@{ }@` raw-R blocks) to the composable mustache templates
(which use native TypR — no raw R).

## Current State

### What works
- Composable templates exist at `templates/targets/typr/tc_*.mustache`
- They generate pure native TypR code with vector-based graph traversal
- No `@{ }@` blocks — all TypR syntax
- The standalone template `transitive_closure.mustache` also updated

### What doesn't work
- `compile_tc_from_template` doesn't provide TypR-specific dict variables
- The composable templates reference `{{empty_nodes_expr}}`,
  `{{from_nodes_expr}}`, `{{to_nodes_expr}}`, etc. which are unresolved
- `recursive_compiler.pl` falls back to `compile_predicate_to_typr`
  which uses the old `@{ }@` approach

## Files to Modify

### 1. `src/unifyweaver/core/recursive_compiler.pl`

**Line ~796** — Change the TypR TC handler to use composable templates:

```prolog
compile_transitive_closure(typr, Pred, _Arity, BasePred, Options, GeneratedCode) :-
    compile_tc_from_template(typr, Pred, BasePred, ExtraDict, Options, GeneratedCode),
    !.
```

Where `ExtraDict` includes the TypR-specific variables.

**In `build_definitions_dict` or a new `build_typr_dict`** — Add:

```prolog
build_typr_extra_dict(BasePred, Options, ExtraDict) :-
    % empty_nodes_expr: how to create an empty vector
    ExtraDict = [
        empty_nodes_expr = "character(0)",   % or "c()" depending on node type
        from_nodes_expr = FromExpr,          % c("alice", "bob", ...) from asserted facts
        to_nodes_expr = ToExpr,              % c("bob", "charlie", ...) from asserted facts
        from_line_expr = "trimws(parts[[1]])",
        to_line_expr = "trimws(parts[[2]])",
        node_parse_helper_code = ""
    ],
    % Generate from/to vectors from asserted facts
    build_edge_vectors(BasePred, FromExpr, ToExpr).
```

### 2. Edge vector generation

The key function needed: extract all asserted `parent/2` facts and
build `c("alice", "bob", "bob")` and `c("bob", "charlie", "diana")`
vectors.

```prolog
build_edge_vectors(BasePred, FromExpr, ToExpr) :-
    functor(Head, BasePred, 2),
    findall(From-To, (user:clause(Head, true), Head =.. [_,From,To]), Pairs),
    maplist([F-_,FS]>>atom_string(F,FS), Pairs, FromStrs),
    maplist([_-T,TS]>>atom_string(T,TS), Pairs, ToStrs),
    format_r_vector(FromStrs, FromExpr),
    format_r_vector(ToStrs, ToExpr).

format_r_vector(Strs, Expr) :-
    maplist([S,Q]>>format(string(Q), '"~w"', [S]), Strs, Quoted),
    atomic_list_concat(Quoted, ', ', Inner),
    format(string(Expr), 'c(~w)', [Inner]).
```

### 3. Template variables reference

The composable templates use these variables:

| Variable | Meaning | Example value |
|----------|---------|---------------|
| `{{pred}}` | Predicate name | `ancestor` |
| `{{base}}` | Base predicate name | `parent` |
| `{{empty_nodes_expr}}` | Empty vector literal | `character(0)` |
| `{{from_nodes_expr}}` | Vector of "from" nodes | `c("alice", "bob", "bob")` |
| `{{to_nodes_expr}}` | Vector of "to" nodes | `c("bob", "charlie", "diana")` |
| `{{from_line_expr}}` | Extract "from" from parsed line | `trimws(parts[[1]])` |
| `{{to_line_expr}}` | Extract "to" from parsed line | `trimws(parts[[2]])` |
| `{{node_parse_helper_code}}` | Optional parse helper | `""` (empty for strings) |
| `{{seed_code}}` | Seed statements (embedded mode) | `add_fact("alice", "bob")\n...` |

### 4. Testing

After wiring, this should produce native TypR (no `@{ }@`):

```prolog
?- compile_recursive(ancestor/2, [target(typr), input(embedded)], Code), write(Code).
```

Expected output should match `templates/targets/typr/transitive_closure.mustache`
with all `{{...}}` variables resolved.

Verify in SciREPL:
1. Install UnifyWeaver package
2. Open "Prolog Generates TypR" workbook
3. Run all cells
4. Cell [6] should show native TypR (no `@{ }@`)
5. Cell [6] output should show correct ancestor results

## Context

- The Lua, Python, R, and Bash targets already use composable templates
  via `compile_tc_from_template` with the `input(Mode)` system
- TypR is the only remaining target using its own compilation path
- The `@{ }@` raw-R blocks work but defeat the purpose of TypR's
  type system — the native templates express everything in TypR
- The TypR WASM compiler now supports `@{ }@` (we patched it) but
  native TypR is preferred
