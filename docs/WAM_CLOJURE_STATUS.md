# WAM Clojure Target — Status

Living summary of the hybrid WAM-Clojure backend
(`wam_clojure_target.pl` + `wam_clojure_lowered_emitter.pl`). Distinct
from the **non-WAM** direct Clojure compiler (`clojure_target.pl`).
Design proposals live under `docs/**/WAM_CLOJURE_*`.

Companion docs:

- `WAM_CLOJURE_*` proposal docs (design/proposals).
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**LMDB-on-the-JVM niche.** First-class LMDB JNI data tier with cache
policies; a JVM WAM route distinct from Scala's, where the lowered
emitter is actually larger than the target module.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_clojure_target.pl` | ~0.9k |
| `src/unifyweaver/targets/wam_clojure_lowered_emitter.pl` | ~1.5k |
| Dedicated tests | ~5 files (~147 plunit cases) |

Note the lowered emitter is **larger** than the target module — most
of the codegen weight is in the lowered path.

## What's shipped

**LMDB JNI tier.** Production-grade JNI loader (delay-wrapped) with
cache policies: `memoize` / `shared` / `two_level`. LMDB is the
dominant theme in the source.

**Foreign category handlers.** Foreign handlers for
`category_parent` / `category_ancestor` — **not** the shared-7 FFI
kernel set.

**Deterministic-prefix lowering.** Dual WAM-instr + lowered emitter;
T4 lowering **strips `switch_on_constant` prefixes** rather than
emitting a switch table (it is not "no switch handling").

**First-argument indexing in the runtime (2026-09).** The whole
standard-WAM switch family that `wam_target.pl` emits is now executed
rather than skipped: `switch_on_constant[_fallthrough]`,
`switch_on_structure`, `switch_on_term`, their `_a2` (second-argument)
forms, and the dedicated `try` / `retry` / `trust` dispatch chains that
carry a group of more than one clause. Entries are resolved once, at
load time, into a map from the dispatch key to the target address
(`resolve-instruction`); dispatch is a hash lookup, and a hit jumps
straight at the clause body, pushing no choice point. Entry lists are
read **first-match-wins**, which is what keeps the indexed answer order
identical to the unindexed one — `wam_target` does emit repeated keys
(`req/2:default req/2:L_..._2`), and a last-wins map silently drops the
earlier clause. An unbound dispatch register, and any key the table does
not mention, fall through to the full clause chain.

Pinned by `tests/core/test_clojurescript_wam_indexing.pl` (five probes,
compiled and run under nbb).

## Gaps (relative to Rust / Haskell / F#)

- **No shared-7 FFI graph kernels** — only the two foreign category
  handlers.
- **No conformance registration** — no `conformance_target(clojure)`.
- **Sequential-only tests**; lowered emitter covers the deterministic
  prefix, not non-deterministic prefixes.
- **No runtime-parser capability entry.**
- **No emitted switch tables** for T4 — the lowered T4 path still strips
  the switch prefix and tries clauses in order inline. The interpreted
  path (everything the resolver uses) now dispatches on the table.
- **Interpretation cost, not indexing, is what bounds this lane at
  scale.** P3 + G1 `index_threshold(64)`: one `resolve_layered` on the 5k
  catalog is **17.6 s / 2.68 M instructions** (index build 2.65 M, query
  27 k) at ~6.6 µs/instr under nbb. Pre-P3 was ~39–42 s / 7.56 M. The
  index cuts the *search* ~276×; the build is interpreted too and is now
  99% of the census. Details in `examples/pkg_resolver/cljs/README.md`.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Clojure's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | `sub_atom` appears in the builtin table but `sub_string/5` nowhere |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **suspected (low)** | registers are string-named and env frames are per-`Allocate` maps (`runtime.clj.mustache:2846-2855`), so the numeric X→Y aliasing form is absent; the frameless-Y-write form was not fully audited |
| A3 | `Execute` of a builtin doesn't return to the continuation | **partially closed (P3)** | `:execute` now succeed-states after `variant/2`, `maplist/2-4`, and `predsort/3`. Other unresolved names still backtrack; `UW_WAM_WARN_UNKNOWN=1` prints `[wam_clojure] execute of unresolved goal NAME failed` (nbb sink: `js/console.error`) |
| A4 | String fidelity | **rung 0** | no string term tag; D37's double-quoted literals intern as atoms |

Compounding A3/B-H2: `wam_clojure` has **no conformance arm**
(CONF-CLOJURE still open in
[`WAM_FLEET_GAP_TASKS.md`](WAM_FLEET_GAP_TASKS.md)) — with Lua, one of
only two WAM backends whose emitted output the shared harness never
executes.

Pattern lane: `clojure_target.pl` compiles facts from
`clause(Head, true)` — the G-A3-8 execute-at-compile-time hazard is
absent; the G-A3 machinery has no analogue there.

## Path forward

1. Extend the lowered emitter to non-deterministic prefixes and emit
   real switch tables (the interpreted path now has them; T4 does not).
1b. Cut the per-instruction interpretation cost — compile each
   instruction into a closure at load time and fuse straight-line runs,
   so a WAM instruction costs one interpreted call rather than a fetch,
   a `case`, several helper calls and a `:pc` update.
2. Add parallelism gates.
3. Register a conformance adapter.
4. Broaden foreign handlers toward the shared-7 kernel set if graph
   perf becomes a goal.

## P3 port (2026-09)

uw-resolve P3 on this lane (mirrors D61 Go): `maplist/2,3,4` + `predsort/3`
meta-call user predicates with isolated env/unify/CP stacks (D61 hygiene
on a persistent-map state); shim speaks `deb/3`, alternatives, provides
3/4-ary, `catalog/10`, blocked `providers`/`alternatives`. Last-slash
arity parse so `///2` is `put_structure`, not `:raw` — that was the 54
G1-index divergences. Gates: corpus **51/51**, differential **2600/0**,
D49+D54 probes green, CLI **142/153** vs frozen expected (live SWI and
the JS CLI agree with cljs on the 11 `install-plan` misses).

## Document status

Fleet-aligned snapshot; source-verified line counts (lowered > target),
the LMDB-JNI + cache-policy surface, the foreign category handlers, the
`switch_on_constant` prefix-stripping T4 behavior, and the absence of
conformance registration (2026-07-11). 2026-09-01: added the
whole-program (A2) deficiency audit; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md). 2026-09-05: P3 port + G1
`///2` parse; B3 census split recorded above.
