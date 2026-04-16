# Shared Purity Certificate Module: Implementation Plan

> **Phased work breakdown** for implementing the module specified in
> `PURITY_CERTIFICATE_SPECIFICATION.md`. Each phase produces something
> testable and commits cleanly as its own PR. Skip ahead by phase when
> picking up this work.

## Overall shape

Five phases, ordered to minimize risk and keep every step
independently shippable:

|Phase|Scope|Risk|Demonstrable outcome|
|---|---|---|---|
|P1|Create `core/purity_certificate.pl` with certificate shape + blacklist producer|Low|Existing `is_pure_goal` callers still work; new API returns the same verdicts wrapped in certificates|
|P2|Absorb `advanced/purity_analysis.pl` as whitelist strategy|Low|Tail-recursion pipeline unchanged; whitelist verdicts now flow through the same certificate API|
|P3|Kernel registry producer|Low|Every detected kernel has an explicit `purity_cert(pure, certified(kernel_registry), 1.0, [kernel_owned])`|
|P4|Wire into Haskell WAM Phase 4.1 `ParTryMeElse` emission|Med|Annotated predicates emit `ParTryMeElse`; unannotated emit `TryMeElse`; tests confirm equivalence under `+RTS -N1`|
|P5|JSON serialization + documentation|Low|`purity_cert/1` can cross the Prolog↔C# boundary when that integration becomes real|

Phase 4 is the first one that touches an emitter; Phases 1–3 are pure
refactors with strong test equivalence guarantees.

Each phase below details: what changes, where, how it's tested, what's
deferred.

---

## Phase P1: Create `core/purity_certificate.pl`

Extract the certificate shape and wire the existing blacklist
(`clause_body_analysis.pl`'s `is_impure_builtin/1`) in as the first
producer.

### Changes

- New file: `src/unifyweaver/core/purity_certificate.pl`
  - Defines `purity_cert/4` compound term (verdict, proof, confidence,
    reasons).
  - Public API per spec §2: `analyze_goal_purity/2`,
    `analyze_predicate_purity/2`, `analyze_clause_purity/2`,
    `merge_certificates/2`.
  - Convenience predicates: `is_certified_pure/1`,
    `is_certified_impure/2`, `purity_confidence/2`.
  - Registration API: `register_purity_producer/2`.
  - `user_annotation_analyzer/2` — reads `:- parallel/1` /
    `:- order_independent/1` / `:- parallel_safe/1` directives.
  - `blacklist_analyzer/2` — wraps `clause_body_analysis:is_impure_builtin/1`.

- `clause_body_analysis.pl` — **no deletions in P1**:
  - `is_pure_goal/1`, `is_order_independent/2`, `classify_parallelism/3`
    all continue to exist. They become thin wrappers that call the new
    API internally. Callers unaware of the change see identical
    behavior.

### Tests

- `tests/core/test_purity_certificate.pl` — new file.
  - Every existing `:- parallel/1` and `:- order_independent/1`
    annotation in the codebase produces `purity_cert(pure, declared,
    1.0, [declared_by_user])`.
  - Every predicate the old `is_pure_goal/1` considered pure now
    certifies `pure` via the new API.
  - Every predicate the old logic considered impure certifies
    `impure(_)` with the correct reason atoms (`io_ops`,
    `database_mods`, `global_state`).
  - Known-unknown builtin (e.g., a synthetic `my_custom_op/2` with no
    declaration) certifies `unknown`.
  - Determinism: 1000 repeated calls return bit-identical certificates.
  - Termination: an artificial deeply-nested goal (`conj(conj(...))`
    nested 10k deep) returns in under 100ms.

- `tests/core/test_clause_body_analysis.pl` — existing tests unchanged.
  All pass. This is the equivalence gate: if anything regresses, P1
  is not shippable.

### Deferred

- Kernel registry producer (Phase P3).
- Whitelist absorbed from `advanced/purity_analysis.pl` (Phase P2).
- Haskell emission integration (Phase P4).

### Risk

Low. The certificate shape is purely additive; `clause_body_analysis.pl`'s
old API remains as a thin wrapper. The only way this breaks is if the
new wrapper computes a verdict differently from the old code — which
the tests will catch.

---

## Phase P2: Absorb the whitelist strategy

Migrate `advanced/purity_analysis.pl`'s whitelist (`pure_builtin/1`)
into the new module as an alternative strategy. The tail-recursion
transformation switches to consuming certificates instead of booleans.

### Changes

- `purity_certificate.pl` — add `whitelist_analyzer/2`:
  - Consults a `pure_builtin/1` clause table identical to the old
    `advanced/purity_analysis.pl` module.
  - Registered at priority 40 (below blacklist — whitelist is
    strictly safer but more conservative; callers that want
    whitelist-only semantics filter on `Proof = analyzed(whitelist)`).

- `advanced/purity_analysis.pl` — thin wrapper:
  - `is_pure_goal/1` and `is_pure_body/1` become wrappers that call
    `analyze_goal_purity/2` and filter for whitelist strategy.
  - File stays for back-compat; can be deprecated in a later pass.

- `core/recursion/linear_to_tail.pl` (or wherever tail-recursion
  transformation lives) — update to use the new API directly.

### Tests

- `tests/core/test_purity_certificate.pl` — add:
  - Whitelist-only consumer gets `pure` only for builtins in the
    strict whitelist; falls back to `unknown` for others even when the
    blacklist would say pure.
  - Tail-recursion transformation test fixtures produce the same
    transformed output as before the migration.

- `tests/core/advanced/test_purity_analysis.pl` — existing tests pass
  unchanged.

- `tests/core/test_linear_to_tail.pl` — existing tests pass. This is
  the equivalence gate for Phase P2.

### Deferred

- Removing `advanced/purity_analysis.pl` entirely. We keep the wrapper
  to avoid touching every caller in one PR. A follow-up can delete it
  once confident.

### Risk

Low. Whitelist semantics are strictly narrower than blacklist; any
verdict the whitelist returns is also returned by the blacklist.
Tail-recursion transformation either gets the same verdict (safe) or a
more conservative one (also safe — just fewer transformations).

---

## Phase P3: Kernel registry producer

Every detected recursive kernel gets an explicit purity certificate.

### Changes

- `purity_certificate.pl` — add `kernel_analyzer/2`:
  - Given a predicate indicator, checks
    `recursive_kernel_detection:kernel_detector/2` membership.
  - If present, returns `purity_cert(pure, certified(kernel_registry),
    1.0, [kernel_owned])`.
  - Registered at priority 90 (trusts registry over blacklist — kernels
    are hand-coded and audited).

- `recursive_kernel_detection.pl` — no changes. The analyzer reads the
  existing registry; no new metadata needed.

- Documentation update: `recursive_kernel_detection.pl` header comment
  notes that kernels are trusted pure and surfaces through the
  certificate module.

### Tests

- Every kernel in the current registry (7 kernels as of this writing —
  `category_ancestor`, `transitive_closure2`, `transitive_distance3`,
  `weighted_shortest_path3`, `transitive_parent_distance4`,
  `astar_shortest_path4`, `transitive_step_parent_distance5`)
  certifies `purity_cert(pure, certified(kernel_registry), 1.0,
  [kernel_owned])`.
- A predicate not in the registry does not pick up this certificate.
- The kernel certificate takes priority over the blacklist — even if a
  kernel's Prolog shape looks impure (e.g., uses a builtin the
  blacklist doesn't recognize), the registry says pure.

### Deferred

- Validating that a kernel's Prolog shape *actually is* pure. The
  registry is trust-by-construction. A future auditing pass could
  cross-check against the blacklist; for now, reviewer discipline is
  the enforcement mechanism.

### Risk

Low. Adding a producer is additive. The only new failure mode is a
reviewer adding an impure kernel; the audit proposal above addresses
that as a follow-up.

---

## Phase P4: Wire into Haskell WAM Phase 4.1 emission — **DELIVERED 2026-04-15**

First real consumer. When Phase 4.1 emits `ParTryMeElse` vs
`TryMeElse`, it consults the certificate module.

**Status: delivered**, combined with intra-query Phase 4.1 on the same
branch — emission was wired certificate-driven from the start rather
than shipping a simpler annotation check first. Confidence threshold
0.85 matches the spec. Kill switch `intra_query_parallel(false)`
lands alongside the emission path.

**Precondition:** Phase 4.1 of the intra-query parallelism
implementation plan (WAM instruction additions). P4 of this plan
depends on that.

### Changes

- `src/unifyweaver/targets/wam_haskell_target.pl` — emission decision:
  ```prolog
  (   analyze_predicate_purity(Pred/Arity, purity_cert(pure, _, Conf, _)),
      Conf >= 0.85,
      \+ option(intra_query_parallel(false), Options)
  ->  emit_par_try_me_else(...)
  ;   emit_try_me_else(...)
  ).
  ```

- The `intra_query_parallel(false)` kill switch from
  `WAM_HASKELL_INTRA_QUERY_SPEC.md` §3.3 lands here — it's a single
  option check wrapped around the emit call.

### Tests

- `tests/test_wam_haskell_target.pl`:
  - Predicate with `:- parallel(P/N).`: generated code contains
    `ParTryMeElse`.
  - Predicate without annotation: generated code contains `TryMeElse`
    (default-safe).
  - Predicate with impure body (e.g., calls `format/2`): generated
    code contains `TryMeElse` even if annotated (the blacklist
    overrides the annotation — verdict merge returns `impure`).
  - With `intra_query_parallel(false)`: all predicates emit
    `TryMeElse` regardless of annotation or verdict.

- `tests/test_wam_haskell_intra_query_integration.pl` — new:
  - An annotated parallelizable predicate generates code that runs
    identically under `+RTS -N1` to the unannotated version
    (sequential semantic equivalence).

### Deferred

- Actually forking at runtime (that's Phase 4.2 of the intra-query
  plan, not this plan's phase 4).
- Work-estimation threshold wiring — the certificate provides verdict
  + confidence; threshold logic is a separate concern in the
  intra-query plan (§4.5).

### Risk

Medium. First time the certificate module drives code generation.
Risk areas: certificate cache invalidation (if a predicate is
redefined between analysis and emission), verdict merging when both
annotation and blacklist produce opinions, and edge cases around
user-defined predicates calling each other.

---

## Phase P5: JSON serialization + cross-process story

Enables certificates to cross language boundaries. Not urgent, but
the right time is while the certificate shape is still fresh and
producer/consumer contracts are recent.

### Changes

- `purity_certificate.pl` — add:
  - `cert_to_json(+Cert, +Metadata, -JsonTerm)` — serializes to a dict
    in the format specified in spec §3.2.
  - `cert_from_json(+JsonTerm, -Cert)` — deserializes. Rejects
    malformed input with informative error terms.

- `docs/design/PURITY_CERTIFICATE_SPECIFICATION.md` — update §3.2 with
  any refinements discovered during implementation.

### Tests

- Round-trip: `cert → json → cert` produces the original certificate
  bit-identical.
- Malformed JSON rejected: missing `verdict`, invalid confidence range,
  unknown proof kind.
- Cross-version forward-compat: unknown metadata fields are tolerated
  (accepted but not consumed). Enables future producer-versioning
  without breaking existing consumers.

### Deferred

- Actually connecting to the C# engine. That requires the C# side to
  start producing verdicts, which is its own project (probably Phase
  4.6 of the intra-query plan).

### Risk

Low. Serialization is mechanical; tests catch drift.

---

## Cross-cutting concerns

### Backward compatibility

Every phase's non-test changes are behavior-preserving refactors with
one exception: Phase P4 introduces a new emission path (`ParTryMeElse`).
That path only activates when:

- The predicate is annotated OR certified pure.
- `intra_query_parallel(false)` is not set.
- Phase 4.1+ of the intra-query plan is merged.

Until all three hold, output is bit-identical to pre-certificate code
generation.

### Test infrastructure

Each phase ships with tests that:
- Exercise the new functionality directly.
- Confirm the affected existing functionality is unchanged (equivalence
  gate).

Phase P4 needs runtime equivalence testing — actually running the
generated Haskell with `+RTS -N1` and comparing output with the
pre-certificate version for a battery of predicates.

### Documentation

Each phase's merge updates:
- `docs/design/PURITY_CERTIFICATE_SPECIFICATION.md` with any
  refinements from implementation.
- `docs/design/PURITY_CERTIFICATE_IMPLEMENTATION_PLAN.md` marking the
  phase complete.
- The first time a target consumes certificates (Phase P4),
  the target's documentation updates too
  (`docs/vision/HASKELL_TARGET_ROADMAP.md`).

### Target-scope reminder

This plan's consumers are demonstrated on Haskell WAM (Phase P4), but
the module is **target-agnostic**. Rust WAM, LLVM, WAT, C#, and other
targets can consume certificates without changes to the module. When
those targets need verdicts, they call the same `analyze_predicate_purity/2`
API and decide — per their own cost model — whether to act on the
result. Phase P4 is *first consumer*, not *only consumer*.

## When to start

Ship P1–P3 as soon as the plan is approved. They're low-risk refactors
that pay back by removing the scattered-analyzer problem immediately.

P4 waits for Phase 4.1 of the intra-query parallelism plan — the
`Par*` instructions don't exist yet, so there's nothing for the
certificate module to drive.

P5 (JSON) can ship anytime after P1 but is lowest priority until a
cross-process consumer materializes.

## Related documents

- `docs/design/PURITY_CERTIFICATE_PROPOSAL.md` — motivation
- `docs/design/PURITY_CERTIFICATE_SPECIFICATION.md` — mechanism
- `docs/design/WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md` — Phase 4.1 is P4's precondition
- `docs/design/WAM_HASKELL_INTRA_QUERY_SPEC.md` §3 — consumer contract
