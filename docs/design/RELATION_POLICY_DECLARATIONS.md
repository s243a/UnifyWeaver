# Relation Policy Declarations

**Status**: draft. Cross-cutting (all targets, all fact-source
backends). Companion to `WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md`, which
will be the first concrete consumer once Phase 2 lands.

## TL;DR

A user-facing Prolog directive `:- relation_policy(Pred/Arity,
OptList).` declares storage- and semantics-level contracts about
a predicate independent of the backend that physically holds its
facts. Policies cover **key** (what determines row identity),
**order** (deterministic clause iteration), **unique** (assertion
about row distinctness), **on_duplicate** (what to do when
uniqueness is violated), and **determinism** (compiler hint about
mode behaviour).

Per-source overrides in `*_fact_sources` specs let the user soften
or sharpen the predicate-level policy for a specific backend
without breaking the canonical contract elsewhere.

Rollout is phased so the parser + registry lands first (no
behaviour change), then per-backend enforcement, then compiler
optimisations that exploit `unique` and `determinism` for CP-skip
and dispatch shortcuts.

## Why declarations

Today every backend makes implicit decisions:

| Implicit today | Where it bites |
|---|---|
| In-source clause order | Cut-based first-solution patterns silently break when facts move to LMDB and the load order changes |
| Duplicate keys silently accepted | Data quality bugs surface as "extra solutions" in user queries, hard to trace |
| Determinism inferred per call site | Compiler can't CP-skip even when the user knows the predicate is functional |
| No way to validate a fact source against expectations | Wrong-schema LMDB file loads silently, queries return garbage |

Making the contract explicit:

1. **Optimisation**: `unique(true)` lets the compiler mark calls as
   deterministic, skipping CP creation in the hot path. Real perf
   win for table-driven hot predicates.
2. **Validation**: `on_duplicate(throw)` at load time catches data
   issues before the user sees garbage from queries.
3. **Semantics**: order matters for cut and first-solution
   patterns; making it explicit beats "whatever the storage backend
   happens to do."
4. **Composability**: declared once at the predicate, applies to
   all backends. Per-source overrides handle the "trust this LMDB,
   not that TSV" case.
5. **Portability**: same Prolog source compiles deterministically
   to any target that supports the policy primitives.

## Surface API

### Predicate-level declaration

```prolog
:- relation_policy(edge/2,
    [ key([arg(1), arg(2)]),
      order([arg(1), asc(arg(2))]),
      unique(true),
      on_duplicate(throw),
      determinism(semidet)
    ]).
```

All options are independent and may be omitted; defaults below.

### Per-source override

```prolog
cpp_fact_sources([
    source(edge/2, lmdb('graph.mdb',
        [ on_duplicate(warn),       % softer than throw
          order(natural)            % use LMDB sort
        ]))
]).
```

Source-level options merge with predicate-level, with source-level
winning on **softening** changes (`throw` → `warn` is OK,
`warn` → `throw` is OK at the source level too). The compiler
emits a warning when a source-level option would invalidate a
compile-time optimisation (e.g. downgrading `unique(true)` after
the compiler has already CP-skipped call sites).

## Option semantics

### key(KeySpec)

What constitutes row identity for uniqueness and dedup checks.

| Value | Meaning |
|---|---|
| `arg(N)` | Single column N |
| `[arg(N), arg(M), ...]` | Composite key over listed columns |
| `all` | Entire row is the key (full-row distinctness) |

Default: `all`.

### order(OrderSpec)

Deterministic clause iteration order.

| Value | Meaning |
|---|---|
| `natural` | Backend-native — LMDB sort, MMA insertion, TSV file order, in-source clause order |
| `[arg(N), ...]` | Lexicographic ascending by listed columns (multi-key fallback for ties) |
| `[asc(arg(N)), desc(arg(M)), ...]` | Explicit direction per column |
| `insertion` | Preserve original ingestion order — requires backend support, may not be available everywhere |

Default: `natural`.

Note: `order` only affects the order of *successful* solutions when
backtracking; it does not affect whether unification succeeds at
all. Programs that don't depend on order are unaffected.

### unique(true | false)

Assertion that the key (per `key(...)`) is unique across all rows.
Drives:

- Compile-time CP-skip for fully-bound calls on the key columns.
- Load-time validation via `on_duplicate`.
- Downstream optimisation (deterministic dispatch).

Default: `false` (no assumption).

A `unique(true)` declaration is a **promise from the user**. The
compiler may take it at face value for optimisation; the loader
checks it at ingest via the `on_duplicate` policy.

### on_duplicate(Policy)

What to do when uniqueness is violated. Only meaningful with
`unique(true)`.

| Policy | Behaviour |
|---|---|
| `throw` | Load-time hard error. Halts ingest. Strict default. |
| `warn` | Log warning, keep all rows. Effectively degrades to `unique(false)` for this load but records the violation. |
| `overwrite` | Last write wins. Backend-dependent (`MDB_OVERWRITE` for LMDB; replay-in-memory for in-source / TSV). |
| `first_wins` | Keep earliest occurrence by load order, drop subsequent matches. |
| `keep_all` | Silently keep all rows. Equivalent to declaring `unique(false)`. |
| `fallback(Policy)` | Try the named policy as a fallback (chains: `fallback(warn)` means "try the previous policy first, fall back to warn"). For declarative readability of cascading rules. |

Default: `throw` if `unique(true)`, else `keep_all`.

### determinism(Mode)

Compiler hint about call-time mode behaviour.

| Mode | Meaning |
|---|---|
| `det` | Always succeeds exactly once. Strongest hint; enables aggressive CP-skip. |
| `semidet` | Succeeds at most once. CP-skip OK on bound-key calls. |
| `nondet` | May succeed any number of times. No optimisation. |
| `multi` | Always succeeds at least once (and may succeed more times). |

Default: `nondet` (no assumption).

A wrong `determinism` declaration is a soundness bug — the
compiler may emit code that depends on the claim. Like
`unique(true)`, this is a user promise the compiler may not be
able to check at compile time.

### cardinality(Hint)

Rough size hint for query planning. Not currently used by any
backend but reserved so we don't have to renegotiate the syntax
later.

| Value | Meaning |
|---|---|
| `unknown` | Default |
| `small` | < 100 rows. Stays in memory, no special handling. |
| `medium` | < 1M rows. Eager-load fine. |
| `large` | >= 1M rows. Prefer lazy probing if backend supports it. |
| `Number` | Exact estimate (or known count). |

Default: `unknown`.

## Defaults summary

When `relation_policy` is not declared for `Pred/Arity`, the
implicit defaults are:

```prolog
relation_policy(_/_, [
    key(all),
    order(natural),
    unique(false),
    on_duplicate(keep_all),
    determinism(nondet),
    cardinality(unknown)
]).
```

This guarantees existing code without declarations behaves
identically to today. No silent semantic change.

## Backend enforcement matrix

Which backend honours which option in which phase. (✓ supported,
~ partial, ✗ deferred.)

| Option | Inline facts | LMDB v1 | LMDB v1.5+ | TSV | MMA v3 |
|---|---|---|---|---|---|
| `key(...)` | ✓ compile-time | ✓ load-time | ✓ load-time + cursor | ✓ load-time | ✓ load-time |
| `order(natural)` | ✓ source order | ✓ LMDB sort | ✓ LMDB sort | ✓ file order | ✓ insertion order |
| `order([...])` | ✓ keysort at compile | ✓ sort post-load | ~ partial | ✓ sort post-load | ✓ sort post-load |
| `unique(true)` + `on_duplicate(throw)` | ✓ compile-time check | ✓ load-time check | ✓ load-time check | ✓ load-time check | ✓ load-time check |
| `unique(true)` + `on_duplicate(overwrite)` | ✗ (already compiled) | ✓ MDB_OVERWRITE | ✓ MDB_OVERWRITE | ✓ dedupe at load | ✓ dedupe at load |
| `on_duplicate(warn)` | ✓ all backends | ✓ all backends | ✓ all backends | ✓ all backends | ✓ all backends |
| `determinism(det/semidet)` | ✓ CP-skip in compiler | ✓ propagated to compiler | ✓ propagated to compiler | ✓ propagated to compiler | ✓ propagated to compiler |
| `cardinality(...)` | ✗ reserved | ✗ reserved | ~ planning hint | ✗ reserved | ✗ reserved |

The compiler must error (not silently ignore) when a target lacks
support for an option the user explicitly set — silent ignoring is
the worst outcome because the user gets a correct-looking compile
and a wrong-behaving program.

## Compiler integration

The `relation_policy/2` directives are read at the same point we
read `:- dynamic` and `:- discontiguous` — early, before clause
compilation. Stored in a module-scoped registry keyed by
`Pred/Arity`:

```prolog
:- dynamic relation_policy_db/3.   % ModulePred, OptionKey, OptionValue

relation_policy_db(edge/2, key, [arg(1), arg(2)]).
relation_policy_db(edge/2, order, [arg(1), asc(arg(2))]).
relation_policy_db(edge/2, unique, true).
relation_policy_db(edge/2, on_duplicate, throw).
```

Each target's codegen consults the registry via a single helper:

```prolog
get_relation_policy(Pred/Arity, OptionKey, Value)
get_relation_policy(Pred/Arity, OptionKey, Value, Default)
```

The helper handles per-source overrides: if the codegen call site
passes a source-spec OptList, that overrides the registry; if the
codegen call site passes nothing, the registry value (or default)
is returned.

Per-target codegen consults the registry when emitting:

- **WAM step** for `unique(true)` calls (skip CP if all positional
  args are bound on entry).
- **Fact-source registration** (pass policy to backend loader).
- **Sort emission** (insert `keysort` / `predsort` call at the
  right point in init).
- **Validation calls** (insert `on_duplicate` check after load).

## Rollout

### Phase 1 — parser + registry + propagation

Lands the `relation_policy/2` directive in the front-end, stores
policies in the registry, exposes `get_relation_policy/3`,
propagates to every target's codegen. **No enforcement.**
Existing code is unaffected because every default matches today's
implicit behaviour.

Test surface: `:- relation_policy(...)` parses; querying the
registry returns the declared value; targets that ignore the
registry continue working unchanged.

### Phase 2 — per-backend enforcement

Target-by-target, starting with LMDB v1 (which is the immediate
consumer). Each backend adds load-time validation, sort emission,
and dedup handling per the matrix above. Lands as small PRs, one
per backend per option, so regressions are isolatable.

Suggested order:

1. **LMDB v1**: `key`, `unique`, `on_duplicate`, `order` (natural
   only).
2. **TSV** (Haskell, Elixir, R, etc.): same four.
3. **Inline facts**: `unique` + `on_duplicate` compile-time check.
4. **MMA v3**: same four as LMDB (when MMA lands).

### Phase 3 — compiler optimisations

Once `determinism` and `unique` are reliably propagated, the
compiler can:

- Skip CP creation for `det` / `semidet` calls with bound keys.
- Generate deterministic-dispatch shortcuts for `unique(true)`
  predicates (single hash lookup, no backtracking machinery).
- Plan join order using `cardinality` hints.

This phase is purely optimisation — no behaviour change. Can land
opportunistically as benchmarks justify it.

## Interaction with cut and first-solution

`order(natural)` (the default) makes clause order
**backend-dependent**, which means cut and `once/1` semantics
shift when a predicate moves between backends. This is
intentional — declaring `order([...])` is how a user opts out of
that fragility.

The compiler emits a warning when:

- A predicate is called under cut and has `order(natural)` and is
  backed by a fact source (i.e. order may shift between runs).
- A `unique(true) on_duplicate(warn)` predicate is called in a
  context requiring `det` mode (warn + duplicates → silent
  multi-solution → likely user surprise).

Warnings are not errors — the user may know what they're doing.
But the compiler shouldn't pretend the issue isn't there.

## Future extensions

- `index(Spec)` — declare indexes the runtime should build at
  load time (analogous to `:- dynamic foo/2.` indexing today, but
  explicit).
- `partition_by(Column)` — for very large fact sources, declare
  how the data should be sharded across files / processes.
- `valid_from(Timestamp) / valid_until(Timestamp)` — temporal
  validity for fact-source files that age out.
- `derived_from(Pred/Arity, Goal)` — express that one predicate
  is materialised from another, letting the compiler verify
  consistency.
- `protocol(json | msgpack | tagged | utf8)` — encoding hint for
  backends that support multiple wire formats (Haskell tagged,
  R TAB-separated, future binary).

Each is independently shippable; none belong in v1.

## Open questions

1. **Module scope.** Are policies per-module or global? Current
   sketch is per-module, matching `:- dynamic`. Cross-module
   imports would need explicit re-declaration. Reasonable?
2. **Conflict resolution.** If two modules declare conflicting
   policies for the same predicate (imported separately into a
   third), what wins? Strict-first-declaration vs last-wins vs
   error.
3. **Querying the policy at runtime.** Should there be a
   `current_relation_policy/3` predicate that introspects the
   active policy? Useful for libraries that want to honour user
   intent.
4. **Versioning.** When the policy syntax evolves, how do we
   handle code that uses an older form? Implicit version field in
   the OptList, or strict-parser-with-error?

## Cross-references

- C++ LMDB design (first concrete consumer):
  `docs/design/WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md`
- LMDB-resident interning (v2 target schema for C++ LMDB):
  `docs/design/WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`
- Rust crate option pattern (analogous codegen-option mechanism):
  `docs/design/WAM_RUST_LMDB_CRATE_DECISION.md`
- C# memory-mapped array precedent (future MMA backend consumer):
  `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs:403, 606`
