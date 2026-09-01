# WAM Haskell Target — Status

Living summary of the hybrid WAM-Haskell backend. Distinct from the
**non-WAM** direct Haskell compiler documented in
[`HASKELL_TARGET.md`](HASKELL_TARGET.md) (`haskell_target.pl`).

Companion docs:

- [`SESSION_HASKELL_PARITY_SUMMARY.md`](SESSION_HASKELL_PARITY_SUMMARY.md) —
  17-PR parity campaign (ISO, CSR, bidirectional, parser, lowered).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md) —
  cross-target matrix.
- [`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md) — fleet roadmap.
- Design set under `docs/design/WAM_HASKELL_*` (perf, lowered, mode
  analysis, intra-query, LMDB lessons, Elixir backport).
- Handoff: [`handoff/wam_haskell_enwiki_benchmark_handoff.md`](handoff/wam_haskell_enwiki_benchmark_handoff.md).

## Role

**Scale + fusion.** Cheap materialisation (LMDB FactSource), GHC
list-comprehension fusion on recursive numeric/list work, and
`parMap rdeepseq` for multi-core fanout.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_haskell_target.pl` | ~7.0k |
| `src/unifyweaver/targets/wam_haskell_lowered_emitter.pl` | ~1.3k |
| Dedicated tests (`tests/test_wam_haskell*`, `tests/core/test_wam_haskell*`) | ~14 files |

## What's shipped

**Dual lowering.** WAM instruction interpreter + per-predicate
lowered emitter (clause-1 of multi-clause; Phase I specialized
instructions; inline get/put/set/cut optimizations — PR #2509).

**Shared graph kernels.** All seven detector kinds
(`transitive_closure2`, `category_ancestor`, `transitive_distance3`,
`transitive_parent_distance4`, `transitive_step_parent_distance5`,
`weighted_shortest_path3`, `astar_shortest_path4`) plus
**bidirectional_ancestor** template with CSR auto-resolution
(PRs #2514–#2518).

**Materialisation.** LMDB FactSource on the safe key/value API
(raw-pointer path abandoned). Emit path is driven by `use_lmdb(true)`
plus `lmdb_cache_mode` tiers (raw / memoize≈sharded / L1 / L2 /
two_level) in `lmdb_fact_source.hs.mustache` — **not** by the
`lmdb_materialisation(eager|lazy|cached)` option, which is resolved
but currently unused by `generate_lmdb_wiring`. Edge-store + auto
resolvers (PR #2519). Compile-time `atom_intern_id` table.

**Parallelism.** `parMap rdeepseq` on fork-eligible paths; nested
spark explosion guarded. Negation race-to-true available.

**ISO substrate.** `WamException`, `throw/1`, `catch/3`, `is_iso/2`,
ISO comparison variants, succ variants (PRs #2510, #2526) with smoke
coverage in `test_wam_haskell_iso_smoke.pl`. Cross-target ISO status
doc historically said Haskell was “mostly missing” — that is
**stale** relative to the substrate; C++/Elixir remain the documented
*reference* adopters in `WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`.

**Runtime parser.** Opt-in `runtime_parser(compiled)`; default off.
E2E generation ~60s for ~50 parser predicates (PR #2522).

**Classic conformance.** Registered (`CONFORMANCE_TARGETS=haskell`);
green. Cabal compile cost makes it opt-in rather than default CI.

## Performance notes

- Scale-300 effective-distance (FFI): ~32 ms query / 75 ms total on
  4 cores; ~107 / 193 ms single-core
  ([`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md)).
- Enwiki-ish handoff also reports larger-scale timings
  ([`handoff/wam_haskell_enwiki_benchmark_handoff.md`](handoff/wam_haskell_enwiki_benchmark_handoff.md)).
- Workload fit: pure recursive numeric aggregation within ~1–2× of
  Rust when fusion fires; parallel fanout is the differentiator.

## Known issues / gaps

- Generated `Main.hs` always pulls LMDB types even when unused —
  blocks GHC without the lmdb package.
- `lmdb_materialisation(...)` option does not yet drive codegen;
  wire emit to it or drop the dead resolver.
- Scan/segregation LMDB modes not yet.
- Classic conformance builds are heavy (cabal per program).
- Runtime parser E2E is slow when enabled.
- Align shared ISO status table with the shipped substrate.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Haskell's audit — the least
favourable in the mature band:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | no `sub_string/5` (or `sub_atom/5`) in the step dispatch |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **verified (structural)** | register encoding is `X→+100 / Y→+200` (`wam_haskell_target.pl:6806-6807`), and `getReg`/`putReg` treat **any id ≥ 200 as a Y register in the topmost env frame** (`:5395-5416`). A large ground fact's X100+ placeholders therefore write into the *caller's* frame — the exact `default_registry/1` clobber the JS runtime hit. Worse: with no frame on the stack, `updateTopEnv` **silently drops the write** (`:5413`), so the fact's own head match can misbehave |
| A3 | `Execute` of a builtin doesn't return to the continuation | **verified partial** | only `isIsoMetaBuiltin` (`catch/3`, `throw/1`) routes through `BuiltinCall` with the CP-return conversion (`:2231-2235`); any other unlabelled name → `Nothing` = silent goal failure (`:2238-2245`, ST path `:3497-3516`) |
| A4 | String fidelity | **rung 0** | `data Value` has no string constructor (`:5721-5729`); D37's double-quoted literals intern as atoms |

Pattern lane — **one verified hazard**: `haskell_target.pl`'s dispatch
falls back to `compile_facts_to_haskell` for any predicate without a
recognized type (`haskell_target.pl:104-114`), and that path enumerates
"facts" by **executing the predicate** —
`findall(..., (functor(Goal,Pred,Arity), call(Goal), ...))`
(`:127-132`). This is exactly typescript's G-A3-8, which hung the
compiler at 1.5 GB on an ordinary rule predicate before it was guarded;
the same guard (facts path only when every clause body is `true`, loud
refusal otherwise) is needed here. The rest of the G-A3 machinery has
no analogue in the pattern lane.

## Path forward

1. Elixir-style cost gates (`forkMinCost` / runtime probe) for
   parallel sparks.
2. Keep LMDB on the safe API; deepen cached mode sizing for enwiki.
3. Align ISO status documentation once three-form rewrite matches
   C++/Elixir/F#/Python.
4. Optional: lighter conformance driver (prebuilt cabal package).

## Document status

Snapshot aligned with `SESSION_HASKELL_PARITY_SUMMARY.md` and the
hybrid comparison branch. Update this file when a materialisation,
kernel, or ISO milestone lands. 2026-09-01: added the whole-program
(A2) deficiency audit — Y-aliasing into the caller's env frame, silent
`Execute` failure, `sub_string/5`, and the pattern lane's
execute-at-compile-time fact fallback, all verified by source reading;
see [`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
