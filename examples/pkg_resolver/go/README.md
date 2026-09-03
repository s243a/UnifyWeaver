# uw-resolve on Go (`wam_go`)

Compile the frozen P0.5 package resolver (`../resolver.pl`) through
the Go WAM backend and drive it with a JSON shim. The shim converts
catalogs/requests/env **in** and selections/explanations **out**. It
contains no resolver logic (no candidate order, no constraint
arithmetic, no layer walk).

## Build / run

From the repo root (needs `swipl` 9.x, Go 1.22+, Node 18+ for the
differential generator):

```bash
bash examples/pkg_resolver/go/build.sh
bash examples/pkg_resolver/go/run_corpus_go.sh
bash examples/pkg_resolver/go/run_differential_go.sh
bash examples/pkg_resolver/go/run_scale_go.sh   # B3 5k catalog
```

`build.sh` is regenerable: it loads `resolver.pl` into `user`, calls
`write_wam_go_project/3` with `prefer_wam(true)`, then `go build`s
`cmd/uwresolve`. Generated WAM sources (`lib.go`, `runtime.go`,
`state.go`, …) are committed so a Go-only rebuild works; re-run
`build.sh` after template or `wam_go_target.pl` edits.

The JSONL driver (`uwresolve`) reads cases on stdin and writes one
result object per line. `--corpus FILE` compares against each row's
`expected` field (SWI dump from `dump_corpus.pl`). `--scale-probe DIR`
times `resolve_layered` on `DIR/rich.jsonl` + `DIR/probe.json`.

## Fixes this program forced

| # | Symptom | Cause | Fix | Probe |
|---|---------|-------|-----|-------|
| 1 | `go build` failed: `unknown field Val` on `switch_on_structure` | Structure-index cases reused the constant-table `{Val, Label}` shape | Emit `{Functor, Label}`; runtime matches `functor/arity` | `tests/test_wam_go_switch_on_structure.pl` |
| 2 | `resolve_pending([], Acc, Acc)` never matched | `append([],[],L)` yielded a zero-length `*List`; `GetConstant []` compared interned `[]` by pointer | `valueEquals` treats empty `*List` ≡ interned `[]`; list-producing builtins emit the atom when empty | `tests/test_wam_go_empty_list_constant.pl` |
| 3 | `sort(Acc, Selection)` kept one `Name-Ver` pair | `compareValues` ranked atoms/numbers and treated every compound as equal; ISO `sort/2` unique-collapses equals | Recursive `compareTerms` (list/structure order) for `sort`/`msort`/`@<`/`compare/3` | `tests/test_wam_go_sort_compounds.pl` |
| 4 | `removal_orphans*` returned `[]`; findall names were `null` | `executeAggregate` shallow-copied `sub.deref(template)`; nested Unbounds lived only in the clone's Bindings | `freezeTerm` deep-copies after deref; collect/bag use `listFromItems` | `tests/test_wam_go_findall_freeze.pl` |
| 5 | `call/1` failed (no label); cuts inside `call` would have been wrong | `call/1` is not `is_builtin_pred`; Go emitted `Call{"call/1"}` | Translate to `BuiltinCall call/1`; opaque cut floor at metacall entry | `tests/test_wam_go_cut_semantics.pl` p09/p10 |
| 6 | `bagof`/`setof` compiled to a missing `call bagof/3` | Go did not pass `inline_bagof_setof(true)` | Opt in at `go_compile_predicate_to_wam/3`; empty bagof/setof fail; setof sorts | cut-semantics p14/p15 |
| 7 | bagof/setof probes left `L` as `__R200` | 4-arg `begin_aggregate bagof, Y, X, ''` was dropped as `// TODO`, so `EndAggregate` was a no-op | Accept the 4th witness-register argument (empty witness list uses the 3-arg runtime) | cut-semantics p14/p15 |
| 8 | Cut suite lost extra solutions / p29 printed `1` not `ok` | Y slots are global; Proceed popped a Call Y-save that backtrack did not restore; `!/0` used the caller's `EnvFrame.CutB0` | Call pushes B0+Y; Execute rebases B0 only (LCO); Proceed pops; CPs snapshot/restore both stacks; `!/0` truncates to `PendingB0` | `tests/test_wam_go_cut_semantics.pl` (35/35) |
| 9 | B3 `resolve_layered` unified Acc with `[]` | `allocVarId` started at 1000; shim minted output vars at `Idx: 10000+i`; after ~9000 cells, `NextVarId` aliased `Bindings[10002]` with `Selection` | skip 10000–10999 in `allocVarId`; shim uses `vm.allocVarId()` | `tests/test_wam_go_varid_collision.pl` |

## Cut / choice-point barriers

`tests/test_wam_go_cut_semantics.pl` ports the 35-probe JS corpus
(SWI oracle, `prefer_wam(true)`). Go has no `emit_mode`; JS
lowering-refusal tests are N/A.

§9 (`docs/WAM_BACKEND_CONVENTIONS.md`): `!` is a barrier at WAM B0
(`PendingB0`), never a stack wipe. Call pushes the caller's B0 and Y
registers; Execute rebases B0 without pushing (LCO reuses the Call
slot); Proceed pops both. Every choice point snapshots `PendingB0`,
`CutB0Stack`, and `YSaves` and restores them on backtrack so a first
success Proceed does not extra-pop on retry. `!/0` truncates to
`PendingB0` (not `peekEnvFrame().CutB0`), so a no-Allocate neck-cut
cannot steal the caller's alternatives.

Aggregates still run in a `Clone()`. `call/1` is an opaque scope (cut
floor = CP height at metacall entry). Residual: leftover choice points
from a *user* goal inside `call/1` are not resumed as extra solutions
of the metacall (first-solution nested run). p09/p10 do not require
that path.

**Cut audit:** 35/35 probes vs SWI, 0 refused-loudly.

## Benchmarks (B1–B3)

Measured on this Cloud Agent VM (SWI 9.0.4, Go 1.22.2). Dump corpus
has 39 cases (plunit `test_resolver.pl` is 38 tests; the extra dump
row is `alias_request_edge`).

| Bench | What | Number |
|-------|------|--------|
| B1 | 39-scenario corpus, Go binary, one process (`run_corpus_go.sh`) | **0.024s wall, 39/39 vs SWI** |
| B1 | `go build` of `uwresolve` (not in B1 wall; clean rebuild) | **0.168s** |
| B2 | Differential Go leg (`run_differential_go.sh`, seed `0xa5b6c7d8`) | **8.869s, 2400 cases, 0 divergences, 0 crashes** |
| B2 | SWI oracle leg, same machine | **1.331s** |
| B3 | 5k catalog `resolve_layered` (`run_scale_go.sh`, seed `0xc0ffee01`) | Go **load 0.060s / resolve 4.604s**, 10-package selection `p1-v(1,0,0)`, `p10`, `p12`, `p2`, `p3`, `p30`, `p4`, `p5`, `p6`, `p7` |

SWI term reference on the **same** `go/.scale_out` catalog + `p30` probe
(not `run_scale_demo.sh`, which also builds the store path): **load
0.428s / resolve 0.008s**, same 10 packages.

## Residuals

- **A2** (X↔Y aliasing on the flat `vm.Regs`, `X_n→n+99` / `Y_n→n+199`
  so X101 ≡ Y1): still structural. Call Y-save is a partial mitigation
  for Allocate-less Y clobber; the numeric alias is unchanged. The P0.5
  resolver did not hit a no-`Allocate` fact with >99 X placeholders.
- **A3**: `BuiltinExecute` covers known builtins. A builtin missed by
  translation-time classification still fails silently. `call/1` is
  now classified. `member/2` is `BuiltinCall` here.
- **call/1 extra solutions**: nested user goals are first-solution;
  opaque `!` is correct.
- **bagof witness grouping** is not implemented (empty witness list is
  OK for cut-semantics p14/p15).
- **Var-id skip window** 10000–10999 is a shim-friendly hole, not a
  general heap allocator. Programs that allocate >9000 unbound cells
  still wrap past the window; the shim now uses `vm.allocVarId()` so
  output vars never sit in that hole.
- P0.5 resolver residuals (unchanged, frozen source): Provides/virtual,
  Debian epoch/tilde, write paths, incremental stores, pkg CLI
  (concurrent), per-file SFS layers.
- Do not create `cljs/` or `rust/` — concurrent rounds.
