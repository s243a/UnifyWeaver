<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# WAMJS-1 Integration Patch — wiring the `wam_javascript` target

The WAMJS-1 card does **not** edit `core/target_registry.pl`,
`tests/test_wam_cross_target_conformance.pl`,
`tests/wam_conformance_fixtures.pl`, `docs/BINDING_MATRIX.md`, or
`tests/test_advanced.pl`. This file is the exact set of additions a
coordinator applies, then runs the conformance arm.

All new source files ship with the card and need no patching:

- `src/unifyweaver/targets/wam_javascript_target.pl`
- `templates/targets/javascript_wam/runtime.js.mustache`
- `templates/targets/javascript_wam/program.js.mustache`
- `src/unifyweaver/bindings/javascript_wam_bindings.pl`
- `docs/WAM_JAVASCRIPT_STATUS.md`

---

## 1. `src/unifyweaver/core/target_registry.pl`

### 1a. Register the target

In `init_target_registry/0` (the JavaScript-family block, right after the
`register_target(browser, ...)` line), add:

```prolog
    register_target(wam_javascript, javascript, [wam, choice_points, hybrid]),
```

Context (existing lines shown for placement):

```prolog
    register_target(node, javascript, [streaming, npm, filesystem, async]),
    register_target(deno, javascript, [typescript, permissions, secure, async]),
    register_target(bun, javascript, [fast, npm_compat, bundled, async]),
    register_target(browser, javascript, [dom, async, web_apis, fetch]),
    register_target(wam_javascript, javascript, [wam, choice_points, hybrid]),   % <-- ADD
```

> The trailing `.` in the registration chain is on the last clause of
> `init_target_registry/0` (`register_target(sql, ...)`). Inserting the
> `wam_javascript` line inside the JavaScript block keeps it a `,`-terminated
> conjunct, matching the surrounding lines — do not add a `.`.

### 1b. Map the target to its module

In the `target_module/2` facts (near the other `wam_*` entries, e.g. after
`target_module(typescript, typescript_target).`), add:

```prolog
target_module(wam_javascript, wam_javascript_target).
```

> Like the other WAM backends (`wam_lua`, `wam_cpp`, …), the `wam_javascript`
> module does not implement the generic `compile_predicate/3` dispatch entry;
> project generation goes through `write_wam_javascript_project/3`. The
> registry entry is for target discovery / capability queries.

---

## 2. `tests/test_wam_cross_target_conformance.pl`

The `wam_javascript` arm is interpreted (no compile step), so it mirrors the
**Python** adapter shape: `ct_build` writes the project, `ct_run` executes
`node generated_program.js <wrapper>/0` per query. Opt-in via
`CONFORMANCE_TARGETS=javascript` (do not add it to `ct_default_target/1`).

### 2a. Module import (with the other target imports, ~line 77)

```prolog
:- use_module('../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3]).
```

### 2b. Registry (in the `conformance_target/1` block, ~line 97)

```prolog
conformance_target(javascript).
```

### 2c. Toolchain probe (in the `ct_toolchain/2` block, ~line 472)

```prolog
ct_toolchain(javascript, [node]).
```

### 2d. Test clause (inside `begin_tests(wam_cross_target_conformance)`, ~line 522)

```prolog
test(javascript, [condition(ct_available(javascript))]) :-
    run_target_conformance(javascript).
```

### 2e. Adapter clauses (with the other `ct_build`/`ct_run`/`ct_teardown`
adapters; the Python adapter is the closest sibling)

```prolog
% ============================================================
% Adapter: JavaScript / Node
%   (0-arity wrapper -> `node generated_program.js <key>` -> true/false)
%
% write_wam_javascript_project/3 emits js/wam_runtime.js + js/generated_program.js.
% Node is interpreted, so there is NO build step — ct_build is pure generation,
% ct_run is a fast `node generated_program.js <wrapper>/0` per query. The CLI
% shim prints true/false and exits 0/1. Opt-in via CONFORMANCE_TARGETS=javascript.
% ============================================================

ct_build(javascript, Preds, Queries, js_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_javascript', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_javascript_project(AllPreds, [module_name(wam_ct)], Dir).

ct_run(javascript, js_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]), atom_string(KeyAtom, KeyStr),
    directory_file_path(Dir, js, JsDir),
    run_proc_out(node, ['generated_program.js', KeyStr], JsDir, _Exit, OutStr),
    normalize_space(string(Out), OutStr), bool_of_string(Out, Bool).

ct_teardown(javascript, js_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).
```

No `ct_xfail/2` or `ct_skip/2` entries are needed — the arm is fully green.

---

## 3. Verification

After applying the patch:

```sh
CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt \
    tests/test_wam_cross_target_conformance.pl
```

Expected: the `javascript` test passes with all programs green
(member, append, reverse, fib, ack, builtins, wide, nested, buildnest,
repeatvar, emptylist). Measured in the WAMJS-1 worktree via an
equivalent scratch arm: **48/48 queries pass, no xfail/skip**
(SWI-Prolog 9.0.4, Node v22).

---

## 4. `docs/BINDING_MATRIX.md` (optional, coordinator's call)

If the coordinator keeps the binding matrix in sync, add a
`wam_javascript` row noting the interpreter tier is shipped and
lowered/FFI are future work (mirrors the `wam_lua` row). Not required
for the conformance arm to pass.
