<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# INTEGRATION_PATCH — JS WAM (`wam_javascript`)

Coordinator-only edits. This change set does **not** modify these shared
files; apply the clauses below when registering the target.

Do **not** edit: `src/unifyweaver/core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, `glue/js_glue.pl`,
or the conformance harness/fixtures, except via this patch.

## 1. `src/unifyweaver/core/target_registry.pl`

Add next to the other JavaScript-family registrations:

```prolog
register_target(wam_javascript, javascript,
                [compiled, wam, hybrid, choice_points, interpreter, node]).
```

Add a module link next to `target_module(wam_lua, wam_lua_target)`:

```prolog
target_module(wam_javascript, wam_javascript_target).
```

Optional alias used by the conformance adapter name `javascript`:

```prolog
target_module(javascript, wam_javascript_target).
```

## 2. `docs/BINDING_MATRIX.md`

Add a row for the JS WAM catalogue (see
`src/unifyweaver/bindings/javascript_wam_bindings.pl`):

| Target | Bindings | Notes |
|---|---|---|
| `wam_javascript` | catalogue in `javascript_wam_bindings.pl` | Interpreter-tier Node WAM. findall/functor/arg/=../copy_term/\+/call/aggregate_all/bagof/setof + ISO/library breadth + native Pratt `parse_term`. |

## 3. `tests/test_advanced.pl`

If the advanced suite loads per-target modules, add:

```prolog
:- use_module('../src/unifyweaver/targets/wam_javascript_target').
```

No `glue/js_glue.pl` change — this is interpreter-tier WAM, not the
pattern/direct JS compiler.

## 4. `tests/test_wam_cross_target_conformance.pl`

Add the module import with the other WAM targets:

```prolog
:- use_module('../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3]).
```

Register the target (opt-in via `CONFORMANCE_TARGETS=javascript`):

```prolog
conformance_target(javascript).
```

Adapter (mirror Python: 0-arity wrappers, interpreted, no build step):

```prolog
ct_toolchain(javascript, [node]).

ct_build(javascript, Preds, Queries, javascript_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_javascript', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_javascript_project(AllPreds,
        [emit_mode(interpreter), module_name(wam_ct)], Dir).

ct_run(javascript, javascript_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]),
    atom_string(KeyAtom, KeyStr),
    directory_file_path(Dir, 'js', JsDir),
    run_proc_out(node, ['generated_program.js', KeyStr], JsDir, _Exit, OutStr),
    (   sub_string(OutStr, _, _, _, "unknown predicate")
    ->  Bool = error(unknown_predicate)
    ;   split_string(OutStr, "\n", " \t\r", Lines0),
        exclude([L]>>(L == ""), Lines0, Lines),
        last(Lines, Last),
        (   Last == "true"
        ->  Bool = true
        ;   Bool = false
        )
    ).

ct_teardown(javascript, javascript_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).
```

`qualify_user/2` and `strip_pred/2` already exist in the harness (used
by the Python/Go adapters). If `qualify_user/2` is not exported in some
revisions, inline:

```prolog
qualify_user(user:P, user:P) :- !.
qualify_user(P, user:P).
```

## 5. `tests/wam_conformance_fixtures.pl` — new builtin programs

Add these **after** the existing `emptylist` program so the coordinator
can extend the 48-query suite without colliding with current PCs. Do not
apply until the JS adapter above is registered; other backends may xfail
until they grow the same builtins.

```prolog
% --- JS WAM builtin probes (optional suite extension) ---
:- dynamic user:cfind/1.
:- dynamic user:cfunctor/2.
:- dynamic user:carg/1.
:- dynamic user:cuniv/1.
:- dynamic user:ccopy/1.
:- dynamic user:cnaf/0.
:- dynamic user:ccall/0.

user:cfind(L) :- findall(X, cmem(X, [1,2,3]), L).
user:cfunctor(N, A) :- functor(foo(a,b), N, A).
user:carg(X) :- arg(2, foo(a,b), X).
user:cuniv(L) :- foo(a,b) =.. L.
user:ccopy(ok) :- copy_term(f(X,X), C), C = f(Y,Y), X \== Y.
user:cnaf :- \+ cmem(9, [1,2,3]).
user:ccall :- call(cmem(b, [a,b,c])).

conformance_program(js_builtins,
                    [user:cmem/2, user:cfind/1, user:cfunctor/2,
                     user:carg/1, user:cuniv/1, user:ccopy/1,
                     user:cnaf/0, user:ccall/0]).

conformance_query(js_builtins, 'cfind/1',    [[1,2,3]], true).
conformance_query(js_builtins, 'cfunctor/2', [foo, 2], true).
conformance_query(js_builtins, 'carg/1',     [b],      true).
conformance_query(js_builtins, 'cuniv/1',    [[foo,a,b]], true).
conformance_query(js_builtins, 'ccopy/1',    [ok],     true).
conformance_query(js_builtins, 'cnaf/0',     [],       true).
conformance_query(js_builtins, 'ccall/0',    [],       true).
```

Until this block is applied, `tests/test_wam_javascript_builtins.pl`
covers the same probes locally (compile → `node` → SWI answers).

## 6. Acceptance command (after §4)

```bash
CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt \
  tests/test_wam_cross_target_conformance.pl
```

Expected: javascript test passes, no xfail/skip, existing 48 queries
green. The dedicated runner that does **not** need this patch:

```bash
swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl
```

## 7. `src/unifyweaver/targets/wam_runtime_parser_capability.pl`

The JS WAM ships a hand-written recursive-descent / Pratt reader in
`templates/targets/javascript_wam/runtime.js.mustache`
(`Runtime.parse_term`). That is the same kind of in-runtime host parser
as C++/R (`native(parse_term)`), **not** the bundled portable
`compiled(prolog_term_parser)`. Do **not** also claim `compiled(...)`
until the portable parser is actually prepended.

Add next to the other `target_runtime_parser_default/2` clauses:

```prolog
% JavaScript WAM: hand-written Pratt reader in the Node runtime
% (Runtime.parse_term). Powers CLI argv, read_term_from_atom/2,3,
% atom_to_term/3, and term_to_atom/2 reverse mode. ISO default
% operator table is included; user op/3 is not. Same default as
% C++/R because the parser always ships with the generated runtime.
target_runtime_parser_default(wam_javascript, native(parse_term)).
```

Add next to the other `target_runtime_parser_mode_/2` clauses:

```prolog
target_runtime_parser_mode_(wam_javascript, native(parse_term)).
```

Add next to the other `normalize_runtime_parser_target/2` clauses
(before the catch-all `normalize_runtime_parser_target(Target, Target)`):

```prolog
normalize_runtime_parser_target(javascript, wam_javascript) :- !.
normalize_runtime_parser_target(wam_javascript, wam_javascript) :- !.
```
