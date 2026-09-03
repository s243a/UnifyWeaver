# WAM Go Target — Status

Living summary of the hybrid WAM-Go backend
(`wam_go_target.pl` + `wam_go_lowered_emitter.pl`). Distinct from the
**non-WAM** direct Go stream/dataflow compiler documented in
[`GO_TARGET.md`](GO_TARGET.md) (`go_target.pl`), which remains Go's
**default** product path.

Companion docs:

- [`design/WAM_GO_PARITY_AUDIT.md`](design/WAM_GO_PARITY_AUDIT.md) — parity vs Rust/Haskell.
- [`GO_TARGET.md`](GO_TARGET.md) — non-WAM sibling compiler.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Strong Go runtime backend.** Broad builtin/IO/aggregate surface with
a full FFI graph-kernel set, reachable when a workload opts into the
WAM pipeline via `prefer_wam(true)`.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_go_target.pl` | ~4.3k |
| `src/unifyweaver/targets/wam_go_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~23 files |

## What's shipped

**All 7 FFI kernels.** Full shared-detector set via
`go_foreign_lowering` / FFI dispatch. (`go_supported_shared_kernel/1`
lists 5; weighted/A* are separate arms — the effective set is all 7.)

**Fact sources.** TSV and LMDB atom-fact paths. The LMDB source
carries the full `lmdb_materialisation(eager|lazy|cached|auto)` +
`lmdb_l2_capacity(N|auto)` tier set (LMDB-GO), matching F#: eager
materialises once at construction, lazy spawns the helper per lookup,
cached memoises through an L1/L2 pair. `auto` defers to the shared cost
model in `core/cost_model.pl`. Default is `cached`.

**ISO three-form errors.** Full adoption (ISO-GO): `catch/3` + `throw/1`
via panic/recover, `error(Formal, Context)` constructors, `is_iso`/
`is_lax`, ISO/lax forms of the six arithmetic comparisons, `succ_iso`/
`succ_lax`, per-predicate key rewrite, and `wam_go_iso_audit/3`.
Rewriting is opt-in: without an `iso_errors` option the WAM text keeps
its plain keys. `catch/3` is deterministic — it commits to the goal's
first solution.

**Effective-distance benchmark.** Scale-300 row populated (BENCH-GO):
query_ms 898 / total_ms 898, 5-rep median, output an exact multiset
match against the reference. See
[`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md).

**Compiled runtime parser.** `runtime_parser(compiled)` compiles the
portable `prolog_term_parser` and the target-agnostic wrappers into the
project, so a generated program can call `read_term_from_atom/2`
directly (PARSE-GO). Default stays `none`. Proven end-to-end by
`tests/test_wam_go_parser_smoke.pl` — 22 inputs including operator
precedence and associativity.

**Dual lowering.** WAM instruction VM plus the lowered emitter.

**Broad surface.** Builtins, IO, and aggregate coverage tracked in the
parity audit against Rust/Haskell.

**Conformance.** Registered `conformance_target(go)` and green — but
**requires `prefer_wam(true)`**, because the default Go strategy is the
non-WAM dataflow/stream compiler, not the WAM pipeline. Stays opt-in
(needs a `go` per-program build), not default CI.

## Gaps (relative to Rust / Haskell / F#)

- **Default product path is non-WAM** (`go_target.pl`); the WAM route
  is opt-in.
- **Effective-distance row is interpreter-bound** (~898 ms query vs
  Rust's 17 ms): the benchmark runs `category_ancestor/4` through the
  shared bytecode loop rather than the FFI kernel path.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Go's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | Go has `sub_atom/5` (`state.go.mustache:2792`) but no `sub_string/5` |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **verified (structural); Call Y-save is a partial mitigation; X101≡Y1 not hit by uw-resolve** | encoding `X_n→n+99 / Y_n→n+199`, so **X101 ≡ index 200 ≡ Y1**. Call snapshots Y 200..299 and Proceed restores; Execute does **not** push (LCO). Choice points snapshot the Y-save stack. The numeric alias is unchanged. |
| A3 | `Execute` of a builtin doesn't return to the continuation | **handled for known builtins; `call/1` now classified** | `BuiltinExecute` takes Proceed's return path **including** `popCallFrame`. Residual: a missed classifier entry still silently fails. `call/1` is `wam_go_direct_builtin` → `BuiltinCall`. `member/2` in uw-resolve is `BuiltinCall`. |
| A4 | String fidelity | **rung 0** | `value.go.mustache` has Integer/Float/Atom/Compound/Structure/List/Ref/Unbound — no string type; D37's double-quoted literals intern as atoms |

Pattern lane: `go_target.pl` compiles facts from `clause(Head, true)`
(`:6790-6798`) — the G-A3-8 execute-at-compile-time hazard is absent.
The G-A3 machinery has no analogue; presume those gaps present until
`examples/cli_args/` is attempted through the Go pattern lane.

## Cut and choice-point barriers (2026-09)

Pinned by `tests/test_wam_go_cut_semantics.pl` (35 probes, SWI oracle,
`prefer_wam(true)` only — Go has no `emit_mode`). **35/35 vs SWI, 0
refused-loudly.** The JS audit found 12 divergences of this class (`!`
wiping ALL choice points). This backend now matches §9:

- Call = `pushCallFrame` (Y-save + push B0, then `PendingB0 = len(CPs)`).
- Execute = `enterExecute` (rebase `PendingB0` **without** pushing).
- Proceed / `BuiltinExecute` = `popCallFrame`.
- `!/0` truncates to `PendingB0`, not `EnvFrame.CutB0` (no-Allocate
  neck-cut).
- Every choice point snapshots `PendingB0`, `CutB0Stack`, and `YSaves`.

uw-resolve plus the 35-probe corpus also forced:

- **`call/1` opaque scope.** Previously a missing label (silent fail).
  Now a builtin whose `!` truncates only to the metacall entry height.
  Residual: nested *user* goals inside `call/1` are first-solution
  (leftover CPs are not resumed as extra metacall solutions). p09/p10
  do not need that path.
- **`inline_bagof_setof(true)`.** bagof/setof compile to
  `BeginAggregate` instead of `call bagof/3`. Empty bagof/setof fail;
  setof sorts.
- Aggregates still run in `Clone()`; an inner `!` cannot destroy the
  caller's CPs. `freezeTerm` copies collected templates out of the clone
  (findall nested Unbounds).

ITE condition / `\+` / `once` / `forall` remain the M17 soft-cut
rewrite. Probe count and any refused-loudly shapes are reported by the
suite (currently 35 compile).

## Whole-program exercise: uw-resolve (`examples/pkg_resolver/go/`)

P0.5 resolver compiled through `wam_go` (`prefer_wam(true)`). JSON shim
is term↔JSON IO only. Corpus **39/39** vs SWI; seeded differential
**2400/0**. Additional runtime bugs the program forced (not in the A2
table): empty-list `GetConstant` vs `*List`; `sort/2` unique-collapsing
compounds; `switch_on_structure` emission using `Val` instead of
`Functor`; 4-arg `begin_aggregate`; `allocVarId` aliasing driver Idx
10000–10999 (B3 `sort/2` unifying Acc with `[]`). **B3** on the 5k
catalog (`0xc0ffee01`): Go load **0.060s** / resolve **4.604s**, same
10-package selection as SWI (load **0.428s** / resolve **0.008s**). See
`examples/pkg_resolver/go/README.md`.

## Path forward

1. Decide whether the WAM route should become a first-class Go product
   path or stay the kernel-benchmarking arm.
2. Route the effective-distance benchmark through the registered
   foreign kernels — the single biggest lever on the Go perf row.
3. Consider registering `conformance_target(go)` as default CI rather
   than opt-in, now that the runtime has been hardened by the parser
   bring-up.

## Document status

Fleet-aligned snapshot; source-verified against `wam_go_target.pl`,
the parity audit, and the conformance harness (`prefer_wam(true)`
requirement confirmed) on 2026-07-11. Refreshed 2026-08-06 after all four Go
gap cards (LMDB-GO, ISO-GO, BENCH-GO, PARSE-GO) landed. Update the parity audit first,
then refresh here. 2026-09-01: added the whole-program (A2) deficiency
audit — X→Y aliasing and the `BuiltinExecute` mitigation verified by
source reading; see [`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
2026-09-03: uw-resolve on Go + 35-probe cut audit (35/35, 0 refused).
A2 still structural (Call Y-save is a partial mitigation). A3 residual
remains for unclassified builtins; `call/1` now classified. Cut suite
is `prefer_wam(true)` only. B3 5k `resolve_layered` matches SWI after
`allocVarId` skips Idx 10000–10999 (Go resolve 4.604s vs SWI 0.008s).
