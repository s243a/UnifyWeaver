<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P8 Integration Patch (for INT-0)

**Task:** G-P8 — add a streaming/generator emit mode to the TypeScript pattern
target so a suitable predicate compiles to TS that reads stdin incrementally
(line-by-line, Node's built-in `readline`, no npm dependency), applies the
predicate, and streams results to stdout — instead of only the batch form.
**Worktree:** `agent-af15c767b897ce8da`
**Shared-file rule:** this agent did **NOT** edit `clojurescript_target.pl`,
`clojure_target.pl`, any `wam_*` file, `core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, `glue/js_glue.pl`,
`vanilla_js_target.pl`, or `annotated_js_target.pl`.

## Files changed (all inside the allowed set — no central wiring required)

| File | Change |
|------|--------|
| `src/unifyweaver/targets/typescript_target.pl` | Added a streaming emit path: a new first clause of `compile_predicate_to_typescript/3` fires only when a streaming option is present, plus a `STREAMING / GENERATOR EMIT MODE` section (`ts_streaming_option/2`, `compile_streaming_typescript/3`, `ts_streaming_filter/3`, `ts_streaming_transform/3`, and the two module templates). Reuses the target's existing `ts_guard_condition/3`, `ts_output_goal_last/3`, `ts_expr/3` and `clause_body_analysis`'s `normalize_goals/2`, `clause_guard_output_split/4`, `goal_output_var/2`. No public export added (dispatch is via `compile_predicate/3`). |
| `tests/core/test_typescript_target.pl` | Added 10 `stream_*` tests (module `test_typescript_target_core`): structural recognition of the filter/transform templates in both modes, both clojure-style aliases, the batch-unchanged guarantee, the multi-clause fall-back, and two node-execution tests (`condition(node_available)`) that pipe stdin under `node --experimental-strip-types` and assert the streamed output equals SWI's solution set over the same input. Added a stdin-piping helper `ts_write_run_stdin/3`. |

**No central wiring is required.** All edited files are in the allowed set.

## Mode / options added

Primary spelling is `mode/1`; the `*_mode(true)` / `*_input(true)` aliases match
the option names `clojure_target.pl` already uses, so one pipeline spec can
target either family with the same options:

| Option | Mode |
|---|---|
| `mode(generator)` or `generator_mode(true)` | generator |
| `mode(pipeline)`  or `pipeline_input(true)` | pipeline |

## How the stdin loop + emit works

Emitted program (transform shown; filter is analogous):

```ts
import { createInterface } from "node:readline";
export function fooTransform(x: number): number[] {
  if (!(<guards>)) return [];        // guard failure drops the record
  return [<expr>];
}
const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (trimmed.length === 0) return;
  const x = Number(trimmed);
  for (const result of fooTransform(x)) console.log(String(result));
});
```

- **Incremental:** `readline` emits one `line` event per record as it arrives —
  no buffering of the whole input.
- **Transform** returns an array of 0+ results, so the *same* shape expresses
  generator (mapcat/flatMap) and pipeline (keep/drop) semantics; a guard failure
  yields `[]` (record dropped).
- **Filter** (arity 1) compiles to a `booleanTest(x)`; generator mode emits the
  numeric value, pipeline mode passes the original input line through.

## What shapes qualify vs fall back to batch

Single-clause predicates only:
- **Filter** `pred(X) :- Guard1, Guard2, ...` — 0+ comparison guards over `X`.
- **Transform** `pred(X, Y) :- [Guards,] Y is Expr` (or `Y = Expr`) — guards plus
  exactly one output goal binding the head's output arg.

Anything else — multi-clause, non-comparison body goals, non-var head args,
arity ≠ 1/2, multiple output goals — makes `compile_streaming_typescript/3`
**fail**, so `compile_predicate_to_typescript/3` backtracks to the normal batch
clauses. Numeric records are the target (input parsed with `Number()`).

## Behavior preservation

The streaming clause is guarded by `ts_streaming_option/2`, which fails when no
streaming option is present. A predicate compiled **without** a streaming option
never enters the streaming path and is byte-for-byte identical to before
(verified by `stream_default_compile_unchanged` and the whole pre-existing
suite). A predicate compiled **with** a streaming option but of a non-qualifying
shape also falls back to batch (verified by
`stream_multiclause_falls_back_to_batch`).

## Inheritance to annotated_js / vanilla_js (flows automatically)

`vanilla_js_target:compile_predicate_to_vanilla_js/3` and
`annotated_js_target`'s compile path both delegate to
`typescript_target:compile_predicate_to_typescript/3` (threading `Options`) and
then rewrite, so streaming flows to both by inheritance. To keep the emitted TS
rewrite-safe under the existing vanilla type-strip and annotated JSDoc rewrite,
the templates deliberately avoid three constructs those rewrites do not yet
handle (see next section):
- a **named** `import { createInterface } from "node:readline"` (not `import * as`),
- an **array** return type `number[]` (not a `number | null` union),
- **no** inline type annotation on the `rl.on("line", (line) => …)` callback param.

Verified by running the vanilla `.js` and annotated `.js` streaming output on
stock `node` — both stream results matching SWI.

## Notes for INT-0 — latent annotated_js / vanilla_js rewrite gaps (NOT blockers)

Surfaced while making the streaming output flow by inheritance; worked around in
the TS templates, but worth fixing centrally at some point (outside this task's
edit scope — those files are owned elsewhere):
1. `vanilla_js_type_strip/2` does not strip a **union return type** such as
   `: number | null` (rule 6 in `vanilla_js_target.pl` only matches a single
   type token before `=>|[,);{=]`).
2. `annotated_js`'s JSDoc rewrite **mangles** an `import * as X from "…"` line
   (produced e.g. `import */** @type … */ ();`) and does **not** strip an inline
   arrow-callback param annotation like `(line: string)`.

## Suggested punchlist update (this agent did not edit the punchlist to avoid
parallel-edit conflicts)

Mark **G-P8** ✅ done in `docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md` §3:
TypeScript now has a streaming/generator emit mode (`mode(generator)` /
`mode(pipeline)` + clojure-style aliases), single-clause filter/transform shapes,
node-verified against SWI, flowing to annotated_js/vanilla_js by inheritance.
No `docs/BINDING_MATRIX.md` row is needed (streaming is an emit mode, not a
binding).

## Acceptance (verbatim)

```
$ swipl -q -g test_typescript_target_core -t halt tests/core/test_typescript_target.pl
[TypeScript Target] Initialized with bindings
.................................................        (49 tests, exit 0, no failures)

$ swipl -q -g test_annotated_js_target -t halt tests/core/test_annotated_js_target.pl
(exit 0, no failures)

$ swipl -q -g test_vanilla_js_target -t halt tests/core/test_vanilla_js_target.pl
(exit 0, no failures)
```

Streaming demo (TypeScript path) — filter `big(X) :- X > 5` in pipeline mode:

```
$ printf '3\n7\n10\n2\n6\n' | node --experimental-strip-types big.ts
7
10
6
$ swipl -q -g "forall((member(X,[3,7,10,2,6]),X>5),(write(X),nl))" -t halt
7
10
6
```

Transform `posdoub(X,Y) :- X > 0, Y is X*2` in generator mode:

```
$ printf -- '-3\n4\n0\n5\n8\n' | node --experimental-strip-types posdoub.ts
8
10
16
$ swipl -q -g "forall((member(X,[-3,4,0,5,8]),X>0,Y is X*2),(write(Y),nl))" -t halt
8
10
16
```
