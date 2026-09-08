<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# WAM Rust genrec: general deterministic-recursion recognizer (sibling-gap + detection scaffolding)

Status: **banked ON** (2026-09-07). Branch `claude/wam-rust-genrec-aodas5`.

This is the first stage of the GENERAL deterministic-recursion recognizer for
the Rust WAM lowered tier — the compositional `deterministic_recursion_class/2`
pipeline of
[`DETERMINISTIC_RECURSION_TAXONOMY.md`](../proposals/DETERMINISTIC_RECURSION_TAXONOMY.md)
§11 — replacing the need for one-off hand-written region recognizers. It closes
the taxonomy's ranked gap #3 (the "sibling gap": `filter_satisfies`,
`key_pkg_rows`, `tree_lookup`) with a single structural classifier, and
scaffolds detection for committed-choice (`close_moving`) and mutual recursion.

## 1. Architecture: where classification lives vs emission

Target-agnostic **classification** (pipeline stages 1–6) is a new shared core
module, `src/unifyweaver/core/deterministic_recursion.pl`, reusable by every
hybrid WAM backend. Only **emission** (stage 7) is per-target, in
`wam_rust_target.pl`. This is the same "shared front-end, per-target back-end"
split as `recursive_kernel_detection.pl`.

`deterministic_recursion_class(+Module, +Name/Arity, -Class)` — never fails,
always binds a class term or `decline(Reason)`:

| Stage | What it does | Where |
|---|---|---|
| 1. Call-graph + SCC | reachability-based SCC (Tarjan-equivalent for tiny graphs); size≥2 ⇒ mutual | `dr_scc/3`, `dr_reaches/3` |
| 2. Meta-call preconditions | any `call/N` reachable in the SCC ⇒ `decline(meta_call_unresolved)` (the 3 preconditions unmet this round) | `dr_scc_has_metacall/2` |
| 3. Per-clause arity + position | self/SCC-call count per clause; tail vs non-tail | `dr_count_self_calls`, `dr_scc_calls_tail` |
| 4. Commit/aggregate boundary | `->`/once/cut detection; committed-tail shape | `dr_has_commit`, `dr_committed_tail_shape` |
| 5. At-most-one shape proof | structural: []/[_|_] or compare trichotomy + committed body + variable-sharing checks | the `dr_*_shape` recognizers |
| 6. Termination class | structural decrease (implicit in the list/tree shapes accepted this round) | shape recognizers |
| 7. **Emission (Rust)** | map a class to a native dispatch method + `lowered_call` arm | `rust_genrec_arms/3`, `state.rs.mustache` |

Class vocabulary: `tail_loop(M:PI, Shape)`, `committed(M:PI, Shape)`,
`mutual(SCC, tail_shaped(Bool), Members)`, `decline(Reason)`. Shapes:
`list_filter(Guard, In, ConPos, Out)`, `list_map_index(EF/EA, Row, In, Idx, Out)`,
`bst_descent(TF/TA, KPos, VPos, In, Key, Out)`.

The recognizers are **structural** (term inspection + `==` variable-sharing
checks), never `=@=` against a frozen literal, so the same classifier fires for
ANY predicate of the shape. `tests/test_deterministic_recursion.pl` proves this
on hermetic fixtures with unrelated names (12 tests, all green).

## 2. What it lowers (and what it declines)

Classification over `resolver.pl` (unmodified):

| Predicate | Class | Emitted? |
|---|---|---|
| `filter_satisfies/3` | `tail_loop(list_filter(satisfies/2,1,2,3))` | **YES** → `region_filter_satisfies_dispatch` |
| `key_pkg_rows/3` | `tail_loop(list_map_index(package/2,pkg_row,1,2,3))` | **YES** → `region_key_pkg_rows_dispatch` |
| `tree_lookup/3` | `tail_loop(bst_descent(t/4,2,3,1,2,3))` | **YES** → `region_tree_lookup_dispatch` |
| `key_dep_rows/3` | `tail_loop(list_map_index(depends/4,dep_row,1,2,3))` | region 3a wins (genrec skips) |
| `matching_versions/4`, `matching_deps/4` | `committed(committed_tail(1))` | regions 1/2 win |
| `close_moving/3`, `dep_breaks/5` | `committed(committed_tail(1))` | detected; emission deferred (region 5 covers dep_breaks) |
| `segs_lt/2`+`segs_lt_1/2` | `mutual(tail_shaped(true))` | detected; already fused by region 2 |
| `lookup_held/3`+`item_ver/3` | `mutual(tail_shaped(false))` | detected; emission deferred (cold, non-tail cross-call) |
| `topo_all/7`+`topo_one/7` | `mutual(tail_shaped(false))` | detected; emission deferred (non-tail, explicit stack) |

The three sibling-gap predicates are lowered by the GENERAL classifier — none is
covered by any hand-written region. Regions 1–5 are untouched and win on overlap
(genrec skips their keys; its emittable shapes do not overlap any region shape).

Every native method keeps the region G-1..G-5 discipline: P2 minimal three-scalar
snapshot, no choice point pushed, CP-depth asserted, and DECLINE to the
interpreter on any off-shape input (open tail, bound output, unbound constraint,
non-integer index, unground key, an un-orderable `compare/3` operand). Declining
is always sound. `region_std_order` reproduces `compare/3` standard order of
terms for the resolver's key types (atoms; `-(N,V)` over atoms/integers) and
declines any other (e.g. deb-version list args) — so deb-keyed dep lookups fall
back to the interpreter, byte-identically.

## 3. Gates (both configs, both lanes)

Default ON; disable with `genrec(false)` or `UW_GENREC_OFF=1`.

| Gate | genrec ON | genrec OFF |
|---|---|---|
| term corpus (B1) | 51 / 51 | 51 / 51 |
| term differential (B2) | 2600 / 0 / 0 | 2600 / 0 / 0 |
| store corpus | 51 / 51 | 51 / 51 |
| store differential | 503 / 0 | 503 / 0 |
| cargo lib (term crate) | 220 passed | — |
| classifier plunit | 12 passed | — |

Output verified **byte-identical** OFF vs ON: the full 2600-case term-lane output
AND the scale-5000 output (where all three predicates fire) are `cmp`-identical
between sha-distinct binaries (ON `90f3b424`, OFF `585a45b5`).

## 4. Real A/B (B3 scale benchmark)

The sibling-gap predicates fire on the INDEXED resolve path (`resolve_layered`
on a ≥64-package catalog: `key_pkg_rows` builds the pkg index, `tree_lookup` +
`filter_satisfies` serve every indexed version lookup). Firing was confirmed
directly with one-shot markers. The store lane's `_store` adapter bypasses
resolver.pl's index, so they are inert there (neutral, byte-identical); the B2
term differential uses sub-threshold catalogs (also neutral). The B3 term-lane
scale benchmark is where they are hot.

Interleaved, drift-cancelling, 20 rounds, sha-distinct verified binaries,
timing the resolve only:

| Benchmark | ON median | OFF median | Δ | ranges |
|---|---|---|---|---|
| scale-1000 (2 preds) | 106.3 ms | 129.8 ms | **−18.09%** | non-overlapping |
| scale-5000 (2 preds) | 386.5 ms | 507.4 ms | **−23.84%** | non-overlapping |
| scale-5000 (all 3) | 388.6 ms | 494.4 ms | **−21.38%** | non-overlapping |

Confirmed win on top of regions 1–5 → banked **ON**.

## 5. Stress tests

`state.rs.mustache` genrec test module (21 tests): filter keep/drop + any + empty
+ decline (open tail / bound output / unbound constraint) + no-CP/registers;
key_pkg_rows monotone index + non-zero start + empty + decline (non-package /
non-integer index / bound output) + no-CP/registers; tree_lookup hit (atom +
compound keys) + miss + empty + decline (bound output / unground key) +
no-CP/registers. `tests/test_deterministic_recursion.pl` (12 tests): every
taxonomy shape recognized structurally on unrelated-name fixtures, plus SCC
membership, self-recursion, and the two decline reasons.

## 6. Files changed

- `src/unifyweaver/core/deterministic_recursion.pl` — NEW target-agnostic classifier (stages 1–6).
- `src/unifyweaver/targets/wam_rust_target.pl` — genrec wiring (stage 7): `rust_genrec_arms/3`, `rust_genrec_shape_method/2`, `rust_genrec_enabled/1`, region-claim exclusion.
- `templates/targets/rust_wam/state.rs.mustache` — 3 native dispatch methods, `region_std_order`/`region_order_rank`, 21 tests.
- `tests/test_deterministic_recursion.pl` — NEW hermetic classifier tests.
- `examples/pkg_resolver/rust/uw_resolve_wam/**` — regenerated term crate (committed).

## 7. Follow-ups

- **Mutual-recursion EMISSION** (flagged as deserving a deeper look): detection is
  done; a stack-safe cyclic emission is deferred. Tail-shaped SCCs
  (`segs_lt`/`segs_lt_1`) already fuse via region 2's plain-host-call precedent;
  non-tail SCCs (`topo_all`/`topo_one`, `lookup_held`/`item_ver`) need an explicit
  stack (taxonomy §5/§8b) and were deliberately NOT half-built.
- **`close_moving` committed-choice emission**: detected as the same class as
  region 5's `dep_breaks`; deferred because it is on the cold
  `safe_upgrade`/`upgrade_set` path and needs native `first_broken`/`pick_repair`/
  `candidates_high_first` copies (region-5-scale effort, cold ROI).
- **Meta-call / `process_all/4`**: currently `decline(meta_call_unresolved)`;
  needs the 3-precondition meta-call layer (taxonomy §6).
- **Aggregate-then-iterate emitter** (taxonomy §7): not started.
- **Subsuming regions 1–5**: the general classifier already recognizes
  `key_dep_rows` (identical `list_map_index` shape) and classifies
  `matching_versions`/`matching_deps` as `committed` — a later cleanup could route
  the region predicates through genrec once genrec has a `list_map_index(dep_row)`
  emitter and a committed emitter, retiring the per-predicate `=@=` recognizers.
