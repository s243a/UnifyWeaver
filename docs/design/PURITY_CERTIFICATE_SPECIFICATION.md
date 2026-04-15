# Shared Purity Certificate Module: Specification

> **Scope:** This specifies the data shape, public API, producer/consumer
> contract, and serialization format for the shared purity certificate
> module proposed in `PURITY_CERTIFICATE_PROPOSAL.md`. The certificate
> is target-agnostic — same shape consumed by Haskell WAM, Rust WAM,
> C# query engine, and any future backend.
>
> See `PURITY_CERTIFICATE_IMPLEMENTATION_PLAN.md` for the phased rollout.

## 1. Certificate shape

```prolog
%% purity_cert(+Verdict, +Proof, +Confidence, +Reasons)
%
% A single certificate captures one analysis result about one
% predicate (or one goal). The shape is a compound term so it can
% be pattern-matched directly and serialized without schema drift.

purity_cert(
    Verdict,       % pure | impure(ReasonList) | unknown
    Proof,         % declared | analyzed(Strategy) | certified(Source) | inferred
    Confidence,    % float in [0.0, 1.0]
    Reasons        % list of atom — e.g. [io_ops, retract, cuts_escape, ...]
).
```

### 1.1 Verdict

|Value|Meaning|
|---|---|
|`pure`|No side effects; clauses are order-independent.|
|`impure(ReasonList)`|Side effects detected; `ReasonList` enumerates culprits.|
|`unknown`|Analysis could not reach a conclusion (e.g., transitive call to a predicate without its own certificate).|

`pure` is the only verdict that justifies emitting parallel instructions
or reordering clauses. `impure` is always a reject. `unknown` is
treated as `impure` by conservative consumers but may be upgraded by
a future producer (e.g., a user annotation arriving later).

### 1.2 Proof provenance

|Value|Meaning|Typical confidence|
|---|---|---|
|`declared`|User asserted via `:- parallel(P/N).` or `:- order_independent(P/N).`|1.0 (trust the user)|
|`analyzed(Strategy)`|Static analysis — `Strategy` names the analyzer (`blacklist`, `whitelist`, `clause_walk`, …)|0.85–0.95|
|`certified(Source)`|External certifier — `Source` is an atom like `kernel_registry`, `csharp_demand_analysis`, `user_review`|0.90–1.0|
|`inferred`|Weak heuristic (e.g., "all recursive calls hit certified predicates")|0.50–0.75|

### 1.3 Confidence

Float in `[0.0, 1.0]`. Advisory — tells a consumer how much to trust the
verdict when multiple certificates disagree or when a consumer weighs
verdict against cost (e.g., a Rust target may require confidence ≥ 0.9
before paying the `Clone` cost, while Haskell may act on ≥ 0.5).

Producer convention: exactly what the producer's own process guarantees.
A user `:- parallel/1` annotation is 1.0. A blacklist analyzer on a
predicate with no transitive user-predicate calls is 0.9. A
kernel-registry certification is 1.0 (handwritten and reviewed).

### 1.4 Reasons

List of atoms identifying either the **culprits** (for `impure`) or
the **evidence** (for `pure`).

**For `impure`:**
- `io_ops` — `write`, `read`, `format`, `get_char`, etc.
- `database_mods` — `assert*`, `retract*`.
- `global_state` — `nb_setval`, `b_setval`.
- `cuts_escape` — `!` whose barrier is outside the predicate.
- `calls_impure_pred(Pred/Arity)` — transitive call to an impure predicate.
- `unknown_builtin(Name/Arity)` — builtin not in any whitelist/blacklist.
- `mutable_state` — for targets that track this (e.g., aliased refs).

**For `pure`:**
- `blacklist_clean` — no impure builtins detected.
- `whitelist_only` — all builtins in the strict whitelist.
- `disjoint_bindings` — for `goals_are_independent` derivations.
- `declared_by_user` — from `:- parallel/1` or `:- order_independent/1`.
- `kernel_owned` — FFI-owned predicate; trusted by construction.

A `Reasons` list is never empty for `impure`. It may be empty `[]` for
`pure` if `Proof = declared` (the user's word is the entire evidence).

## 2. Public API

### 2.1 Producer API

```prolog
%% analyze_goal_purity(+Goal, -Cert)
%  Certify a single goal. Looks up the goal's functor in the
%  registered analyzers (blacklist, whitelist, user annotations,
%  kernel registry). Returns the highest-confidence verdict.
analyze_goal_purity(Goal, Cert).

%% analyze_predicate_purity(+PredIndicator, -Cert)
%  Certify a predicate by analyzing its clauses. PredIndicator is
%  `Module:Name/Arity` or bare `Name/Arity` (user:-prefixed).
%
%  Resolution order:
%    1. User annotation (:- parallel/1 or :- order_independent/1) → declared
%    2. Kernel registry membership → certified(kernel_registry)
%    3. Clause-body analysis over every clause → analyzed(clause_walk)
%    4. Otherwise → unknown
analyze_predicate_purity(PredIndicator, Cert).

%% analyze_clause_purity(+ClauseTerm, -Cert)
%  Certify a single clause (Head-Body). Rarely called directly; used
%  internally by analyze_predicate_purity.
analyze_clause_purity(Head-Body, Cert).
```

### 2.2 Registration API

```prolog
%% register_purity_producer(+ProducerSpec, +Priority)
%  Add a new producer to the analysis chain. Priority is checked in
%  descending order; first verdict wins.
%
%  ProducerSpec = purity_producer(Name, Analyzer, SupportedConstructs)
%  Analyzer is a pred:(+Input, -Cert) meeting the API.
register_purity_producer(purity_producer(user_annotations, user_annotation_analyzer, [predicate]), 100).
register_purity_producer(purity_producer(kernel_registry,  kernel_analyzer,          [predicate]), 90).
register_purity_producer(purity_producer(blacklist,        blacklist_analyzer,       [goal, clause, predicate]), 50).
register_purity_producer(purity_producer(whitelist,        whitelist_analyzer,       [goal, clause, predicate]), 40).
```

Producers are registered at startup via directives; consumers don't
see them directly.

### 2.3 Consumer API

```prolog
%% Convenience predicates for common questions.
is_certified_pure(+PredIndicator)       :- ...  % Cert = purity_cert(pure, _, _, _).
is_certified_impure(+PredIndicator, -Reasons). % Cert = purity_cert(impure(Reasons), _, _, _).
purity_confidence(+PredIndicator, -Conf).      % Float in [0,1] regardless of verdict.
```

### 2.4 Merge API (for multi-producer scenarios)

```prolog
%% merge_certificates(+CertList, -MergedCert)
%  Combine multiple certificates for the same predicate. Resolution:
%    - Any impure → impure (short-circuit, conservative).
%    - All pure → pure with max(Confidence) and merged Reasons.
%    - Mixed pure + unknown → pure if at least one producer has
%      Confidence ≥ 0.9, else unknown.
merge_certificates(Certs, Merged).
```

Used when the C# engine contributes a verdict *and* the Prolog analyzer
has one, or when `:- parallel/1` coexists with a blacklist warning.

## 3. Serialization format

For cross-process communication (Prolog ↔ C# query engine, future
persistence to disk, test fixtures). Two formats:

### 3.1 Prolog term (native)

```prolog
purity_cert(pure, declared, 1.0, [declared_by_user])
purity_cert(impure([io_ops, database_mods]), analyzed(blacklist), 0.95, [])
purity_cert(unknown, inferred, 0.3, [unknown_builtin(my_op/2)])
```

Use when the consumer is another Prolog module.

### 3.2 JSON (interop)

```json
{
  "verdict": "pure",
  "proof": {"kind": "declared"},
  "confidence": 1.0,
  "reasons": ["declared_by_user"],
  "subject": "category_ancestor/4",
  "producer": "user_annotations",
  "producer_version": "1.0.0"
}
```

Use when crossing a language boundary. `producer` and `producer_version`
are metadata not in the core term but useful for debugging
provenance in a multi-producer pipeline.

The `impure` verdict wraps its reason list into the top-level
`reasons` field:

```json
{
  "verdict": "impure",
  "proof": {"kind": "analyzed", "strategy": "blacklist"},
  "confidence": 0.95,
  "reasons": ["io_ops", "database_mods"],
  "subject": "log_and_dispatch/2"
}
```

## 4. Producer contracts

Each producer must meet:

- **Determinism.** Given the same input, return the same certificate.
  No reliance on assert order, clock time, or GHC optimization level.
- **Termination.** No infinite loops even on deeply nested goals.
  Cycles in call graphs must be detected and terminated with `unknown`.
- **Conservative defaults.** Unknown builtins, unknown functors, and
  unanalyzable structures produce `unknown` — never `pure`.
- **No global state mutation.** The analysis itself must be pure —
  otherwise the analyzer is a side-effect source, destroying its own
  verdict's meaning.

## 5. Consumer contracts

Each consumer must meet:

- **Treat `unknown` as `impure` by default.** Only override with explicit
  user policy (e.g., a debug flag).
- **Respect `Confidence` when the action is costly.** A Rust target
  may refuse to fork on confidence < 0.9 even if verdict is `pure`.
  The certificate says "safe"; the target decides "worthwhile."
- **Record verdicts consumed.** When the consumer makes a compilation
  decision (emit Par* vs Try*), log the verdict that drove it — for
  debugging and for the `intra_query_parallel(false)` future kill
  switch (see intra-query spec §3.3).

## 6. Relationship to existing modules

### 6.1 `clause_body_analysis.pl`

**Role: principal donor.** Current predicates map to the new module as:

|Old API|New API|Migration|
|---|---|---|
|`is_pure_goal(G)`|`analyze_goal_purity(G, purity_cert(pure, _, _, _))`|wrapper in old module|
|`is_order_independent(P, declared)`|`analyze_predicate_purity(P, purity_cert(pure, declared, _, _))`|wrapper|
|`is_order_independent(P, proven(R))`|`analyze_predicate_purity(P, purity_cert(pure, analyzed(_), _, R))`|wrapper|
|`classify_parallelism/3`|Stays — now internally calls `analyze_predicate_purity`|refactor|
|`partition_parallel_goals/5`|Stays — uses `analyze_goal_purity` per-goal|refactor|

`is_impure_builtin/1`'s clauses move into `blacklist_analyzer`'s
definition.

### 6.2 `advanced/purity_analysis.pl`

**Role: whitelist strategy.** `pure_builtin/1` becomes the body of
`whitelist_analyzer`. Callers get the same semantics via
`analyze_goal_purity` with a strict-mode consumer preference:

```prolog
% Consumer picks whitelist-only by filtering on Proof.
analyze_goal_purity(G, Cert),
Cert = purity_cert(pure, analyzed(whitelist), _, _).
```

The old `is_pure_goal/1` in that module becomes a thin wrapper for
back-compat.

### 6.3 `recursive_kernel_detection.pl`

**Role: kernel producer.** Every kernel registered via `kernel_detector/2`
(and its companions `kernel_native_kind/2`, etc.) gains an implicit
certificate:

```prolog
purity_cert(pure, certified(kernel_registry), 1.0, [kernel_owned])
```

The certificate exists because FFI-owned kernels are hand-coded native
implementations audited for purity by whoever wrote them. Making this
explicit lets consumers (e.g., Haskell WAM `ParTryMeElse` emission)
treat kernels and user-annotated predicates uniformly.

### 6.4 `csharp_target.pl`

**Role: consumer only in Phase 1. Potential producer later.**
`safe_recursive_clause_for_need/2` (lines 1990–2006) is **structural
need-closure safety**, not purity — it stays where it is. The C# target
may start consuming certificates in Phase 5+ for query-plan
reordering. If the C# engine eventually produces side-effect verdicts
(from LINQ analysis or Roslyn data-flow), those become a new producer
at priority ≥ 80 (trusted external certification).

## 7. What stays the same

- **User annotations** keep their existing syntax. `:- parallel(P/N).`
  and `:- order_independent(P/N).` are parsed and turned into
  declared-verdict certificates, not replaced.
- **Existing callers of `is_pure_goal/1`** continue to work — the old
  predicate becomes a thin wrapper returning a boolean.
- **Performance.** The new module is no slower than the old per-call;
  the first call to `analyze_predicate_purity/2` memoizes the verdict
  so repeated queries (e.g., during WAM emission) are constant-time
  lookups.

## 8. Target-scope applicability

The certificate shape is **target-agnostic**. Verdicts apply equally to:

- **Haskell WAM (immutable).** First consumer. Uses verdicts to emit
  `ParTryMeElse` / `ParRetryMeElse` / `ParTrustMe` where purity allows.
- **Rust WAM (mutable).** Consumes the same verdicts. Fork cost is
  higher (needs `Clone` bounds or `Arc` wrapping), so Rust may apply
  stricter confidence thresholds or defer to user opt-in via
  `:- parallel/1` only. The certificate is the oracle; Rust's fork
  primitive decides how to act.
- **C# query engine.** Verdict consumers for query-plan reordering
  (pure goals hoist past aggregates), join ordering, and eventual
  parallel materialization. The existing structural need-closure check
  stays separate — a joint caller combines both verdicts when
  relevant.
- **Other targets (Go, AWK, LLVM, WAT, Bash).** Any target that needs
  to know "can I reorder this" or "is this side-effect-free" plugs in
  as a consumer. The module makes no commitments about the target's
  memory model.

**What the certificate does not include:** fork primitives, state-copy
strategies, runtime spark management. Those are target concerns. A
Rust target may decide *never* to fork even on a 1.0-confidence pure
verdict because `Clone` is too expensive for the workload; that's a
target policy, not a certificate concern.

## 9. Open questions

1. **How do we handle `meta_predicate/1`?** E.g., `findall/3` is
   pure iff its Goal argument is pure. Current blacklists don't handle
   this. Options: (a) special-case meta-calls in the analyzer; (b) punt
   to `unknown` and require user annotation; (c) introduce a
   "contingent purity" verdict. MVP picks (b); (c) is a future
   refinement.

2. **Caching policy for `analyze_predicate_purity`.** Memoize forever?
   Invalidate on `assertz`? MVP: memoize per-module-load, invalidate
   on `retractall` of the target predicate.

3. **Should `certified(Source)` include a version?** For forward-compat
   when a certifier's semantics change. MVP: no; version lives only in
   JSON serialization for cross-process correlation.

## Related documents

- `docs/design/PURITY_CERTIFICATE_PROPOSAL.md` — motivation
- `docs/design/PURITY_CERTIFICATE_IMPLEMENTATION_PLAN.md` — rollout
- `docs/design/WAM_HASKELL_INTRA_QUERY_SPEC.md` §3 — first consumer
- `src/unifyweaver/core/clause_body_analysis.pl` — existing donor
- `src/unifyweaver/core/advanced/purity_analysis.pl` — whitelist source
