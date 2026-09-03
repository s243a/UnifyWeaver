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

## Cut / choice-point barriers

`tests/test_wam_go_cut_semantics.pl` ports the 35-probe JS corpus
(SWI oracle, `prefer_wam(true)`). Go has no `emit_mode`; JS
lowering-refusal tests are N/A.

Go already had M17 `GetLevel`/`Cut` (`ite_use_y_level(true)`),
`PendingB0` rebased by Call and Execute, and `EnvFrame.CutB0` so `!/0`
is a barrier not a stack wipe. Aggregates run in a `Clone()` so an
inner `!` cannot destroy the caller's CPs. `call/1` is now an opaque
scope (cut floor = CP height at metacall entry). Residual: leftover
choice points from a *user* goal inside `call/1` are not resumed as
extra solutions of the metacall (first-solution nested run). p09/p10
do not require that path.

## Benchmarks (B1–B3)

Measured on this Cloud Agent VM (SWI 9.0.4, Go 1.22). Fill-in after
the timed runs in this PR:

| Bench | What | Number |
|-------|------|--------|
| B1 | 39-scenario corpus, Go binary, one process (`run_corpus_go.sh`) | *pending* |
| B1 | `go build` of `uwresolve` (not in B1 wall) | *pending* |
| B2 | Differential Go leg wall + case count (`run_differential_go.sh`) | *pending* |
| B2 | SWI oracle leg, same machine | *pending* |
| B3 | 5k catalog `resolve_layered` (`run_scale_go.sh`, seed `0xc0ffee01`) load / resolve | *pending* |

SWI 5k reference: `bash examples/pkg_resolver/run_scale_demo.sh`.

## Residuals

- **A2** (X↔Y aliasing on the flat `vm.Regs`, `X_n→n+99` / `Y_n→n+199`
  so X101 ≡ Y1): still structural. This program did not hit it
  (no no-`Allocate` fact with >99 X placeholders). See
  [`docs/WAM_GO_STATUS.md`](../../../docs/WAM_GO_STATUS.md).
- **A3**: `BuiltinExecute` covers known builtins. A builtin missed by
  translation-time classification still fails silently. `call/1` is
  now classified. `member/2` is `BuiltinCall` here.
- **call/1 extra solutions**: nested user goals are first-solution;
  opaque `!` is correct.
- P0.5 resolver residuals (unchanged, frozen source): Provides/virtual,
  Debian epoch/tilde, write paths, incremental stores, pkg CLI
  (concurrent), per-file SFS layers.
- Do not create `cljs/` or `rust/` — concurrent rounds.
