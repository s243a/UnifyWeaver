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

**Interpreter-tier Node WAM** with an optional **Tier-2 lowered emitter**.
Consumes shared bytecode from `wam_target.pl` and runs it on a stock Node
(v22) VM. `emit_mode(interpreter)` (default) is unchanged. `functions`
and `mixed([P/A, ...])` emit direct JS functions for eligible predicates
and keep the interpreter as fallback. No extra runtime dependency.
Closest sibling is the Lua WAM: same term tags, register encoding
(A1→1, X1→101, Y1→201), 1-based instruction PCs, trail + choice points,
and BeginAggregate / EndAggregate collection.

## Codegen surface

| Module | Role |
|---|---|
| `src/unifyweaver/targets/wam_javascript_target.pl` | Emitter (WAM items → `I.*` instruction vector + intern seed; emit-mode resolver) |
| `src/unifyweaver/targets/wam_javascript_lowered_emitter.pl` | Tier-2 lowered JS functions (`deterministic` / T4 / T5 / T6 / ITE) |
| `templates/targets/javascript_wam/runtime.js.mustache` | Node WAM VM |
| `templates/targets/javascript_wam/program.js.mustache` | Instruction vector + labels + `lowered_dispatch` + CLI shim |
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

Lists: `member/2`, `length/2`, `between/3`, **`append/3`**, **`reverse/2`**,
**`nth0/3`**, **`nth1/3`**, **`last/2`**, **`sum_list/2`** (`sumlist/2`),
**`max_list/2`**, **`min_list/2`**, **`list_to_set/2`**, **`select/3`**,
**`include/3`**, **`exclude/3`**.

Sort: **`sort/2`**, **`msort/2`**, **`keysort/2`**, **`sort/4`** (integer
Key, `@<`/`@>`/`<`/`>`), **`predsort/3`** (`compare/3` and 3-arg callables),
**`compare/3`**.

Atom/string: **`atom_concat/3`**, **`string_concat/3`**, **`atom_length/2`**,
**`string_length/2`**, **`atom_chars/2`**, **`string_chars/2`**, **`atom_codes/2`**, **`char_code/2`**,
**`sub_atom/5`** (ground-Atom; enumerates unbound Before/Length/After),
**`sub_string/5`** (same walk; Sub is a `V.String`),
**`atom_string/2`**, **`string_to_atom/2`**, **`string/1`**,
**`number_codes/2`**, **`number_string/2`**, **`split_string/4`**,
**`upcase_atom/2`**, **`downcase_atom/2`**. Distinct **`string` tag**
(`V.String`): string-producing builtins yield strings; `atom/1` is
false for them. `write/1` prints the text; `writeq/1` and `format` `~q`
quote strings with `"` and quote atoms only when needed.

I/O: `write/1`, **`writeq/1`**, `nl/0`, `writeln/1`, **`format/2`**, **`format/3`**
(`~w ~a ~d ~p ~q ~n ~s ~t ~~`; `atom(A)` / `string(S)` sinks), **`tab/1`**.

Assoc (`library(assoc)` shape, list-of-pairs not AVL): **`empty_assoc/1`**,
**`list_to_assoc/2`**, **`get_assoc/3`**, **`put_assoc/4`**,
**`assoc_to_list/2`**, **`assoc_to_keys/2`**.

Term: **`functor/3`**, **`arg/3`**, **`=../2`**, **`copy_term/2`**,
**`read_term_from_atom/2` `/3`**, **`atom_to_term/3`**, **`term_to_atom/2`**,
**`term_variables/2`**, **`numbervars/3`**, **`=@=/2`**, **`\=@=/2`**,
**`op/3`**.

Metacall: **`\+/1`**, **`call/1`** (re-enter the same instruction loop /
builtin dispatch).

Collections: **`findall/3`** (compiler `BeginAggregate`/`EndAggregate` *and*
builtin metacall), **`bagof/3`** / **`setof/3`** (ISO free-var grouping,
`^/2` existential quantification, empty-goal failure), **`aggregate_all/3`**
for `count` / `sum(X)` / `bag(X)` / `set(X)`.

Types: `atom/1`, `integer/1`, `float/1`, `number/1`, `string/1`, `compound/1`,
`var/1`, `nonvar/1`, `is_list/1`, `ground/1`.

## Runtime term parser (G-W2)

The Node runtime ships a **hand-written recursive-descent / Pratt reader**
(`Runtime.parse_term` / `tokenize_term` in `runtime.js.mustache`). CLI argv
goes through `parse_cli_atom_or_int`, which now calls the same reader
(unreadable text still interned as an atom).

**Full:** integers (including a leading `-` after start/`(`/`[`/`,`/`|`),
floats (`3.14`, `-1.5`, `1.0e2`), bare atoms, quoted atoms (`'hi there'`),
double-quoted strings (`"hi"` → string tag),
with `\'` / `\\` escapes), variables (`X`, `_`, shared names), `[]`,
proper lists `[a,b,c]`, partial lists `[H|T]`, compounds
`foo(a, bar(b), 3)`, and parentheses. Cons intern as `[|]/2` + the `[]`
atom (same functor as `put_list`).

**Operators:** ISO default infix + prefix table (`1+2` → `+(1,2)`).
**`op/3`** mutates the live table (declared ops override/extend defaults
at their priority/associativity). Types: `xfx`/`xfy`/`yfx` infix, `fx`/`fy`
prefix, `xf`/`yf` postfix. Priority 0 removes that specifier. Name may be
an atom or a list of atoms. Infix and postfix of the same name cannot
coexist (ISO); prefix may share a name with infix. Compile-time `:- op/3`
declarations are threaded via `javascript_wam_ops([op(P,T,N), ...])`
(alias `js_op_decls/1`) into `Runtime.install_declared_ops` at program
startup.

**Partial:** User `op/3` is process-global (like SWI), not module-local.
`current_op/3` is not implemented. Fall back rather than invent a term
when the token stream is leftover or illegal.

Capability (coordinator applies `INTEGRATION_PATCH.md` §7; the shared
capability file is not edited here):

```prolog
target_runtime_parser_default(wam_javascript, native(parse_term)).
target_runtime_parser_mode_(wam_javascript, native(parse_term)).
```

This matches C++/R (`native(parse_term)`): an in-runtime host-language
parser, not the bundled portable `compiled(prolog_term_parser)`.

## Profiling (GP-PROF)

Opt-in interpreter instrumentation. **Off by default:** `Runtime._prof` is
`null` and every hook is a single falsy check (no allocation). When off,
`Runtime.run` / `collect_run` take the same instruction loop as before this
change; stdout and stderr are byte-identical to an uninstrumented run.

### Activation

| Mechanism | Effect |
|---|---|
| env `UW_PROFILE=1` (also `true` / `yes`) | Table report on stderr at process exit |
| env `UW_PROFILE=json` | One JSON object on stderr at process exit |
| `Runtime.profile(true)` | Same as `UW_PROFILE=1` |
| `Runtime.profile("json")` | Same as `UW_PROFILE=json` |
| `Runtime.profile(false)` | Disable |
| `Runtime.profileReport()` | Write the report now (also hooked on `process.exit`) |

Reports go to **stderr only**. Stdout stays program output (the conformance
harness reads it).

### Per-predicate metrics

Exclusive while the predicate is on top of the profiler stack:

| Field | Meaning |
|---|---|
| `calls` | Enter count (`Call` / `Execute` / CLI / lowered dispatch) |
| `instr` | Instructions executed (interpreter tier only) |
| `ns` | Exclusive wall time via `process.hrtime.bigint()` |
| `cps` | Choice points created while on top |
| `max_cp_depth` | Max `state.cps.length` seen while on top |
| `lowered` | `true` if the row came from a lowered-function call |

Predicates are sorted by `ns` descending, then `calls` descending. Lowered
names get a trailing `*` in the table. Instruction / CP / exclusive-time
detail is **interpreter-tier only**; lowered functions increment `calls`
(and set `lowered`) through the `lowered_dispatch` wrapper.

### Global metrics

`instructions`, `unify_calls`, `trail_pushes`, `heap_cells`, `backtracks`,
`trail_undos` (each `undo_trail` that actually popped), `wall_ns` (process
span from `Runtime.profile()`).

### `UW_PROFILE=json` schema

One object, one line (plus newline). Stable keys:

```json
{
  "tier_note": "Lowered predicates: call counts only; instruction/CP/time detail is interpreter-tier.",
  "predicates": [
    {
      "pred": "fib/2",
      "calls": 67,
      "instr": 1234,
      "ns": 89012,
      "cps": 3,
      "max_cp_depth": 2,
      "lowered": false
    }
  ],
  "global": {
    "instructions": 1234,
    "unify_calls": 56,
    "trail_pushes": 40,
    "heap_cells": 30,
    "backtracks": 12,
    "trail_undos": 8,
    "wall_ns": 100000
  }
}
```

Table shape (stderr):

```
UW profile  (interpreter instr/CP/time; lowered = calls only)
pred                          calls       instr       CPs   maxCP            ns
fib/2                            67        1234         3       2         89012
--
instr=1234 unify=56 trail=40 heap=30 backtracks=12 undos=8 wall_ns=100000
```

## Peerhailer argparser (A2)

**Runs the peerhailer argparser (A2).** `examples/cli_args/cli_args.pl` (module
`cli_args`, `parse_args/2,3`) compiles through `wam_javascript`
(`emit_mode(interpreter)`) into `examples/cli_args/wamjs/js/`. A thin ESM shim
(`cliArgs.mjs`) converts JS `argv` ↔ WAM terms and maps `ok/2` /
`error/1`; it implements no parse rule.

| Check | Result |
|---|---|
| Contract corpus (`cliArgs.wamjs.test.mjs`, oracle tests, import swapped) | **17 / 17** |
| Differential vs JS oracle (`run_differential_wamjs.sh`, same seed) | **5067 lines, 0 divergences, 0 message mismatches** (oracle 0.048s, wamjs 4.639s) |

Runtime gaps this program forced: `sub_string/5`; Y-register save/restore
across `Call` of a non-`Allocate` fact; `Execute` of a builtin `Proceed`s to
CP instead of halting. Probes: `probe_sub_string/0`, `probe_y_preserve/0`,
`probe_tail_builtin/0` in `tests/test_wam_javascript_builtins.pl`.

## Remaining / partial

| Builtin | Status |
|---|---|
| `bagof/3` | **Implemented.** ISO witness grouping (one bag per distinct free-var binding, SWI encounter order), `Var^Goal` / nested `V1^V2^Goal` stripped from the witness set, fails when Goal has no solutions. |
| `setof/3` | **Implemented.** `bagof` then per-group standard-order sort + dedup. Order: Var < Number < **String** < Atom < Compound (SWI 9.0.4); compounds by arity, functor **name**, then args L-to-R (matches SWI mixed-type lists). |
| `term_variables/2` | **Implemented.** Distinct unbound vars, first-occurrence L-to-R depth-first. Cyclic compounds are visited once (same `seen` walk as `copy_term`). |
| `numbervars/3` | **Implemented.** Binds+trails each distinct unbound var to `'$VAR'(N)` from Start; End is Start+count. `write/1` prints `'$VAR'(N)` literally — it does **not** letter-style SWI rendering (`A`, `B`, …). |
| `=@=/2` / `\=@=/2` | **Implemented.** Variant equality: ground as `==`; vars match via a consistent bijection. Cyclic struct pairs are treated as already-equal once seen. |
| `format/2` `/3` | **Implemented** for `~w ~a ~d ~p ~q ~n ~s ~t ~~`. Not ported: `~f`, `~r`, `~D`, positioning (`~N|`, `~+`, `t~`), aliases, and stream sinks other than stdout / `atom(A)` / `string(S)`. |
| `sub_atom/5` | **Implemented** when Atom is ground; enumerates unbound Before/Length/After (and filters a ground SubAtom). |
| `sub_string/5` | **Implemented.** Same enumeration as `sub_atom/5`; Sub is a `V.String`. |
| Peerhailer argparser (A2) | **Implemented.** `examples/cli_args/wamjs/` compiles `cli_args.pl` through `wam_javascript` (interpreter) and matches the JS oracle: **17/17** corpus, **5067-line** differential with **0 divergences, 0 message mismatches**. |
| String term tag | **Implemented.** `V.String` is a distinct tag. Unify/`==` require equal strings (not atoms). Standard order / `compare/3` / `sort` matches SWI 9.0.4: Var < Number < **String** < Atom < Compound (`"foo" @< foo`). `atom_string/2`, `string_concat/3`, `string_chars/2` (construct), `string_to_atom/2`, `number_string/2`, `split_string/4` produce strings. `string/1` is true only for the tag. `string_length/2` accepts a string, atom, or number (code-point length). `write/1` prints text unquoted; `writeq/1` and `format` `~q` recurse through lists/compounds, double-quote strings, and quote atoms only when needed (see quoting subset below). **Compiled `"foo"` literals** are spelled with outer double quotes in WAM text (`quote_wam_constant/2`); the shared classifier still returns `atom(foo)` (no `string(_)` Class). JS consults `wam_constant_token_is_string/1` and builds `V.String`. Other runtimes intern the atom as before. Fact-source JSON/TSV values still intern as atoms. |
| `library(assoc)` | **Implemented** as a Prolog `assoc/1` list of Key-Value pairs (not SWI's AVL tree). get/put/list/keys match SWI for unique-key maps. |
| First-arg indexing | **Implemented.** `switch_on_constant` / `_fallthrough` / `_a2`, `switch_on_structure` / `_a2`, and `switch_on_term` / `_a2` jump to the matching clause group. Ground first-arg with a unique clause leaves no choice point (`deterministic/0`). Unbound first arg falls through to the try/retry/trust chain (no lost solutions). Exclusive miss fails; fallthrough variants keep the chain for variable-headed clauses. Dedicated `try`/`retry`/`trust` dispatch chains are emitted for multi-clause groups. |
| Second-arg / deep indexing | A2 switches are implemented; deep (argument >2) indexing is not. |
| Lowered / functions emit mode | **Implemented.** `javascript_wam_resolve_emit_mode/2` accepts `interpreter` (default), `functions` (lower every eligible predicate), and `mixed([P/A, ...])` (lower only the named ones). Eligible shapes: single-clause deterministic bodies; T4 all-clauses-inline; T5 first-arg constant dispatch; T6 hash dispatch (≥8 atom keys); structured ITE / negation / once. Unsupported ops (`begin_aggregate`, bagof/setof, cuts/jumps the planner rejects) fall back to the interpreter rather than emitting wrong code. Interpreter-mode bytecode and wrappers are unchanged. |
| CLI / runtime term parser | **Implemented.** Pratt reader: int/float/atom (incl. quoted)/var/list/`[H\|T]`/compound. CLI argv + `read_term_from_atom` / `atom_to_term` / `term_to_atom`. **`op/3`** updates the live infix/prefix/postfix tables (defaults cloned from ISO). Compile-time ops via `javascript_wam_ops/1`. Capability `native(parse_term)` via `INTEGRATION_PATCH.md` §7. |
| Interpreter profiling | **Implemented.** Off by default (`Runtime._prof === null`). `UW_PROFILE=1` / `json` or `Runtime.profile(...)` writes a per-predicate table or JSON to **stderr**. Lowered tier: call counts only. See [Profiling (GP-PROF)](#profiling-gp-prof). |
| `op/3` | **Implemented.** Infix `xfx`/`xfy`/`yfx`, prefix `fx`/`fy`, postfix `xf`/`yf`. Priority 0 removes. Name = atom or list of atoms. `current_op/3` is not implemented; ops are process-global. |
| External fact sources | **Implemented.** `javascript_wam_fact_sources([source(P/2, file(Path))])` (alias `js_fact_sources/1`) emits `CallFactStream` and a Node `fs` reader for TSV/CSV and JSONL. First-arg index when A1 is bound. Same lightweight file-backed model as Lua. **LMDB / CSR are out of scope.** Inline facts (no option) are unchanged. |
| Conformance harness adapter | See `INTEGRATION_PATCH.md` (coordinator applies `conformance_target(javascript)`). |

## How to run

```bash
mkdir -p output/advanced
# Dedicated probe + local 48-query suite (does not edit the shared harness):
swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl

# File-backed P/2 fact sources (CSV/TSV/JSONL):
swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl

# Tier-2 lowered / mixed emit-mode suite:
swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl

# Opt-in profiler (stderr only; stdout unchanged). 1 = table, json = JSON:
# UW_PROFILE=1 node output/js_wam_profile_probes/js/generated_program.js fib/2 20

# After INTEGRATION_PATCH.md is applied:
CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt \
  tests/test_wam_cross_target_conformance.pl

# Peerhailer CLI parser (A2) — compile, 17-test corpus, differential vs oracle:
bash examples/cli_args/wamjs/build.sh
node --test examples/cli_args/wamjs/cliArgs.wamjs.test.mjs
bash examples/cli_args/wamjs/run_differential_wamjs.sh
```

Residual ISO corners not covered: bagof/setof of *unbound* free vars (two
solutions that leave the same witness unbound) is grouped by copied
variable name rather than `@=`; `^/2` as a standalone metacall just
runs the RHS; `format` does not implement `~f` / `~r` / column
positioning; assoc is a list-of-pairs, not SWI's AVL tree;
`numbervars` does not letter-render `'$VAR'(N)` on `write/1`;
`op/3` is process-global (no module-local ops) and `current_op/3` is
not implemented. `writeq` atom quoting is a pragmatic SWI
subset: unquoted for `[]`, `!`, `;`, `{}`, `^[a-z][A-Za-z0-9_]*$`, and
ISO graphic atoms (`#$&*+-./:<=>?@^~\`) except the lone atom `.`;
quoted otherwise, escaping `\\` and `\'` only (no `\n`/`\t` escapes).
Brace terms `{a}` are written as `'{}'(a)` / `{}(a)`, not SWI `{a}`.
`write/1` list spacing is unchanged (`", "`). Fact-source cells stay
atoms even when the host file looks like a quoted string.

## Document status

Initial JS WAM bring-up + builtin port from Lua, ISO bagof/3 and setof/3,
first-argument indexing, the Tier-2 lowered emitter, ISO/library builtin
breadth (sort, lists, atom/string, format, assoc), the G-W2 runtime term
parser, G-W4 file-backed fact sources (TSV/CSV/JSONL; LMDB/CSR out of
scope), the G-W3 term-meta family (`term_variables/2`,
`numbervars/3`, `=@=/2`, `\=@=/2`), then G-W2 `op/3` (dynamic Pratt
table: infix + prefix + postfix), then a distinct string term tag
(`V.String`; string-producing builtins + standard order), then
`string_length/2` and `writeq/1` / recursive `~q` quoting, then
compiled `"foo"` literals as `V.String` (double-quoted WAM spelling;
classifier still returns `atom(_)`), then opt-in interpreter profiling
(`UW_PROFILE=1` / `json`, stderr-only table or JSON; lowered = call counts),
then the peerhailer CLI argparser through the interpreter (A2:
`examples/cli_args/wamjs/`, 17/17 corpus + 5067-line differential vs the
JS oracle, 0 divergences).
Source-verified against SWI-Prolog as the oracle (2026-09-01).
