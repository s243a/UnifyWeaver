<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# WAM JavaScript Target — Status

Living summary of the hybrid WAM-JavaScript backend
(`wam_javascript_target.pl` + `templates/targets/javascript_wam/*`).
Distinct from the **non-WAM** pattern/direct JS compilers
(`typescript_target`, `annotated_js_target`, `node`).

Companion docs:

- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md) — the six conventions.
- [`WAM_LUA_STATUS.md`](WAM_LUA_STATUS.md) — the dynamically typed model this port follows.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md) — harness contract.

## Role

**Interpreter-tier Node WAM.** Consumes shared bytecode from `wam_target.pl`
(`emit_mode(interpreter)`) and runs it on a stock Node (v22) VM. No extra
runtime dependency. Closest sibling is the Lua WAM: same term tags, register
encoding (A1→1, X1→101, Y1→201), 1-based instruction PCs, trail + choice
points, and BeginAggregate / EndAggregate collection.

## Codegen surface

| Module | Role |
|---|---|
| `src/unifyweaver/targets/wam_javascript_target.pl` | Emitter (WAM items → `I.*` instruction vector + intern seed) |
| `templates/targets/javascript_wam/runtime.js.mustache` | Node WAM VM |
| `templates/targets/javascript_wam/program.js.mustache` | Instruction vector + labels + CLI shim |
| `src/unifyweaver/bindings/javascript_wam_bindings.pl` | Builtin catalogue |

## Six-conventions checklist

| # | Convention | JS WAM |
|---|---|---|
| 1 | Cons: `put_list` and `put_structure [|]/2` intern the same functor; `[]` is the interned atom. Accept `[|]`, `.`, `./2`. | Yes — seed ids `[]`=2, `.`=3, `[|]`=5; `GetList` / unify / `=..` alias cons functors. |
| 2 | Functor `name/arity` where name may contain `/`; parse arity as trailing `/<digits>`. | Yes — `parse_functor_arity/3` and runtime `strip_trailing_arity`. |
| 3 | Nested terms built outer-first; `put_*` must bind+trail the X/Y placeholder (A-register exception). | Yes — `push_built_term` bind-through for `target >= 101`. |
| 4 | `deref` before every type test. | Yes. |
| 5 | `is/2` yields an integer for integral results. | Yes — `eval_arith` + `as_arith_result` (`Number.isInteger` → `V.Int`). |
| 6 | Unhandled instruction ⇒ a real one-slot NoOp (`I.Raw` / default), never drop/throw. | Yes. Implemented switches consume exactly one slot and jump or fall through; unknown ops stay `I.Raw`. `EndAggregate` still returns fail on purpose (collect then backtrack). |

## Implemented builtins

Control / unify / arith: `true/0`, `fail/0`, `!/0`, `=/2`, `==/2`, `\==/2`,
`is/2` (recursive `evalArith` on `+ - * / // mod`), `=:= =\= > < >= =<`.

Lists: `member/2`, `length/2`, `between/3`.

Term: **`functor/3`**, **`arg/3`**, **`=../2`**, **`copy_term/2`**.

Metacall: **`\+/1`**, **`call/1`** (re-enter the same instruction loop /
builtin dispatch).

Collections: **`findall/3`** (compiler `BeginAggregate`/`EndAggregate` *and*
builtin metacall), **`bagof/3`** / **`setof/3`** (ISO free-var grouping,
`^/2` existential quantification, empty-goal failure), **`aggregate_all/3`**
for `count` / `sum(X)` / `bag(X)` / `set(X)`.

Types: `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `var/1`,
`nonvar/1`, `is_list/1`, `ground/1`.

I/O (probe dumps): `write/1`, `nl/0`, `writeln/1`.

## Remaining / partial

| Builtin | Status |
|---|---|
| `bagof/3` | **Implemented.** ISO witness grouping (one bag per distinct free-var binding, SWI encounter order), `Var^Goal` / nested `V1^V2^Goal` stripped from the witness set, fails when Goal has no solutions. |
| `setof/3` | **Implemented.** `bagof` then per-group standard-order sort + dedup. Order: Var < Number < Atom < String < Compound; compounds by arity, functor **name**, then args L-to-R (matches SWI mixed-type lists). |
| `term_variables/2`, `numbervars/3`, `=@=/2` | Not ported. |
| First-arg indexing | **Implemented.** `switch_on_constant` / `_fallthrough` / `_a2`, `switch_on_structure` / `_a2`, and `switch_on_term` / `_a2` jump to the matching clause group. Ground first-arg with a unique clause leaves no choice point (`deterministic/0`). Unbound first arg falls through to the try/retry/trust chain (no lost solutions). Exclusive miss fails; fallthrough variants keep the chain for variable-headed clauses. Dedicated `try`/`retry`/`trust` dispatch chains are emitted for multi-clause groups. |
| Second-arg / deep indexing | A2 switches are implemented; deep (argument >2) indexing is not. |
| Lowered / functions emit mode | Interpreter only. |
| Conformance harness adapter | See `INTEGRATION_PATCH.md` (coordinator applies `conformance_target(javascript)`). |

## How to run

```bash
mkdir -p output/advanced
# Dedicated probe + local 48-query suite (does not edit the shared harness):
swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl

# After INTEGRATION_PATCH.md is applied:
CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt \
  tests/test_wam_cross_target_conformance.pl
```

Residual ISO corners not covered: `term_variables/2` / `numbervars/3` /
`=@=/2` are still unported; bagof/setof of *unbound* free vars (two
solutions that leave the same witness unbound) is grouped by copied
variable name rather than `@=`; the runtime has no distinct string tag,
so String vs Atom order is unused; `^/2` as a standalone metacall just
runs the RHS.

## Document status

Initial JS WAM bring-up + builtin port from Lua, ISO bagof/3 and setof/3,
then first-argument indexing (`switch_on_constant` / `structure` / `term`
and fallthrough / A2 variants). Source-verified against SWI-Prolog as the
oracle (2026-08-30).
