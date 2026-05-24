# WAM F# Parity Audit

This note compares the F# hybrid WAM target against the cross-target
reference baselines, using Haskell and Rust for WAM runtime/builtin
behavior, C++/Elixir/Python for ISO-error parity, and Haskell/Rust for
LMDB lazy-access parity. It also records readiness against the
in-flight reverse-index/CSR design.

Companion docs:

- [`../WAM_FSHARP_TARGET.md`](../WAM_FSHARP_TARGET.md) - the user-facing
  F# target reference.
- [`WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md) -
  shared ISO-error contract; the table here mirrors the rows in that
  status doc.
- [`WAM_LMDB_LAZY_PHILOSOPHY.md`](WAM_LMDB_LAZY_PHILOSOPHY.md) and the
  matching `_SPECIFICATION.md` / `_IMPLEMENTATION_PLAN.md` - LMDB
  `eager`/`lazy`/`cached` taxonomy. F# is not currently named in the
  phased rollout.
- [`WAM_REVERSE_INDEX_ARTIFACTS.md`](WAM_REVERSE_INDEX_ARTIFACTS.md) -
  CSR reverse-index design; Phase C ships a Rust reader first, with C#
  to follow once the format and policy are stable.
- [`WAM_GO_PARITY_AUDIT.md`](WAM_GO_PARITY_AUDIT.md),
  [`WAM_PYTHON_PARITY_AUDIT.md`](WAM_PYTHON_PARITY_AUDIT.md),
  [`WAM_LUA_PARITY_AUDIT.md`](WAM_LUA_PARITY_AUDIT.md) - sibling
  per-target audits this one is modelled on.

## Verified Runtime Surface

The F# WAM runtime in
`src/unifyweaver/targets/fsharp_runtime/WamRuntime.fs`, plus the
binding-level types in
`src/unifyweaver/bindings/fsharp_wam_bindings.pl` and the codegen in
`src/unifyweaver/targets/wam_fsharp_target.pl` /
`src/unifyweaver/targets/wam_fsharp_lowered_emitter.pl`, supplies:

| Area | F# support | Rust/Haskell baseline | Status |
| --- | --- | --- | --- |
| Choice points and backtracking | `TryMeElse`/`RetryMeElse`/`TrustMe`, `Try`/`Retry`/`Trust` for indexed dispatch (issue #2400), `MemberRetry`/`SelectRetry`/`FactRetry`/`HopsRetry`/`FFIStreamRetry` builtin CPs | Choice point stack with normal/builtin/fact-stream resume states | Present |
| Direct fact dispatch | `ForeignFacts` / `FfiFacts` / `FfiWeightedFacts` in-memory maps on `WamContext` | `call_indexed_atom_fact2`, fact-source registry | Present for in-memory facts only - no external fact source |
| Aggregates | `BeginAggregate`/`EndAggregate` + `MergeStrategy` DU (`MergeSum`/`Count`/`Bag`/`Set`/`Findall`/`Sequential`); accumulator materializes on backtrack | `findall/3`, `aggregate_all/3` count/sum/min/max/set families | Present |
| Structural builtins | `member/2`, `memberchk/2`, `append/3`, `length/2`, `reverse/2`, `last/2`, `nth0/3`, `nth1/3`, `delete/3`, `select/3`, `sort/2`, `msort/2`, `compare/3` | Same baseline expanded by Go/Clojure/C++ | Present |
| Type builtins | `atom/1`, `atomic/1`, `compound/1`, `integer/1`, `number/1`, `float/1`, `var/1`, `nonvar/1`, `is_list/1` | Same baseline | Present |
| Comparison builtins | `==/2`, `\==/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2`, `@</2`, `@=</2`, `@>/2`, `@>=/2`, `compare/3` via Prolog-standard `compareValue` (Var < Number < Atom < Compound) | Same baseline | Present |
| Unification builtin | `=/2`, `\=/2` | Same | Present |
| Term inspection | `functor/3`, `arg/3` (both modes); specialized `Arg` opcode for literal-N case | Same | Present |
| Univ | `=../2` compose and decompose; specialized `PutStructureDyn` opcode for runtime-parsed functor | Same | Present |
| Copying | `copy_term/2` with fresh variables and preserved sharing | Same | Present |
| Control | `true/0`, `fail/0`, `!/0` (full B0 cut barrier per Aït-Kaci), `\+/1`, `CutIte`, `runNegationParallel` racing isolated clause bodies | Same baseline; race-to-true also in Haskell/Go | Present |
| Arithmetic | `is/2` (full evaluator: `+`, `-`, `*`, `/`, `//`, `mod`, `rem`, `**`, `abs`, `min`, `max`, `gcd`, `truncate`, `round`, `ceiling`, `floor`, `sqrt`, `sin`, `cos`, `tan`, `pi`, `e`, bitwise) | Same | Present **without ISO/lax three-form split** - see "ISO Error Readiness" below |
| Atom / text conversion | `atom_codes/2`, `atom_chars/2`, `atom_length/2`, `atom_concat/3`, `atom_string/2`, `atom_number/2`, `number_codes/2`, `number_chars/2`, `char_code/2`, `upcase_atom/2`, `downcase_atom/2`, `sub_atom/5`, `string_codes/2`, `string_chars/2` | Same baseline | Present |
| IO | `write/1`, `display/1`, `nl/0` | Same | Present |
| Database (limited) | `assert/1`, `assertz/1`, `asserta/1`, `retract/1` via `WcLoweredPredicates` mutation - works for facts, not clauses with bodies | C++/Python/Haskell have a richer dynamic-database story | Partial - documented in target doc as "use Python for dynamic-database semantics" |
| Sets | `BuildEmptySet`, `SetInsert`, `NotMemberSet` for graph-kernel visited sets | Same shape across targets | Present |
| Runtime parser | `runtime_parser(compiled(prolog_term_parser))` compiles `prolog_term_parser.pl` into WAM and exposes `read_term_from_atom/2,3`, `parse_term_from_atom/3,4`, `parse_term_from_codes/3,4` | Per `WAM_RUNTIME_PARSER_STATUS.md`: F# is fully covered (compiled mode) | Present |

## ISO Error Readiness

F# is now a **partial ISO-error adopter**. The substrate
(catch/throw + error constructors + throw_iso_error), the config
loader/rewrite/audit plumbing, and `is_iso/2` + `is_lax/2` shipped
together. The arithmetic-comparison and `succ/2` ISO/lax variants
are follow-up work.

| Component | F# status | Notes |
| --- | --- | --- |
| Prolog `catch/3` / `throw/1` substrate | Present | `WamException` exception type plus a `WsCatchers : CatcherFrame list` side stack on `WamState`. `dispatchCall` has a top-level try/with that prints "Uncaught Prolog throw" and returns `None`. The WAM compiler emits `catch/3` and `throw/1` as `Call`/`Execute` meta-calls (not `BuiltinCall`); special-case dispatch in the F# step `Call`/`Execute` arms routes them through `BuiltinCall` so the ISO arms fire. |
| ISO error constructors | Present | `makeInstantiationError`, `makeTypeError`, `makeDomainError`, `makeEvaluationError`, `makePredIndicator` in `fsharp_wam_bindings.pl`. |
| `throw_iso_error` helper | Present | Wraps the inner error term in `error(ErrorTerm, _)` (with a fresh unbound `Context`) and raises `WamException`. |
| `is_iso/2` / `is_lax/2` | Present | `is/2` and `is_lax/2` share the lax body; `is_iso/2` does three-step classification (unbound -> instantiation_error, zero divide -> evaluation_error(zero_divisor), otherwise -> type_error(evaluable, Name/Arity)). |
| ISO/lax arithmetic compares | **Missing** | Six comparison variants (`>`, `<`, `>=`, `=<`, `=:=`, `=\\=`) still need ISO/lax three-form dispatch. Follow-up PR. |
| `succ/2` and ISO/lax variants | **Missing** | F# does not currently expose `succ/2` at all. Follow-up PR. |
| Lax IEEE-754 float divide | Partial | F# `is/2` already returns `nan`/`inf`/`-inf` for float zero division (CLR default). Integer divide-by-zero in lax mode fails silently because `evalArith` returns `None` (no exception escapes), matching the documented lax contract. |
| Per-predicate ISO config loader | Present | `iso_errors_config(File)`, inline `iso_errors(Default)` / `iso_errors(PI, Mode)`, file-vs-inline precedence per spec. Copied from Python target; future extraction into `src/unifyweaver/core/iso_errors.pl` is appropriate now that F# is the third adopter. |
| Per-predicate default rewrite | Present | `iso_errors_rewrite_text/4` walks WAM text, rewriting `is/2` -> `is_iso/2` / `is_lax/2` according to the predicate's resolved mode. Wired into `compile_predicates_to_fsharp` so generated F# uses the ISO-mode key. |
| ISO audit predicate | Present | `wam_fsharp_iso_audit/3` reports per-call-site resolution following the shared audit shape. `wam_fsharp_iso_audit_report/1` pretty-prints. |

### Remaining F# ISO Work

Steps 1-5 and 7-8 of the original minimum-useful-adoption list (now
matched against `WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` § "What Counts
As Adoption") shipped together. The remaining items:

- **Arithmetic-compare ISO/lax variants**: `>_iso/2`, `<_iso/2`,
  `>=_iso/2`, `=<_iso/2`, `=:=_iso/2`, `=\\=_iso/2` plus matching
  `*_lax/2` aliases. Each needs a step branch alongside the existing
  lax branch, an entry in both `iso_errors_default_to_iso/2` and
  `iso_errors_default_to_lax/2`, and a regression test case.
- **`succ/2` family**: F# does not currently support `succ/2` at
  all. Adding it adds three keys (default, `_iso`, `_lax`) per the
  three-form pattern.
- **Shared helper extraction**: F# is now the third adopter (after
  Elixir and Python -- C++ is its own reference). Extracting the
  `iso_errors_*` predicates into `src/unifyweaver/core/iso_errors.pl`
  is appropriate next, as called out in
  `WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` § "Remaining Work".

## LMDB Fact-Source Readiness

F# is **not yet** named in `WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md`. The
current state across all WAM targets (from the plan's §1 table) is:

| Target | `eager` | `lazy` | `cached` | Scan | Segregation |
| --- | :-: | :-: | :-: | :-: | :-: |
| Haskell | shipped | degenerate (cap 0) | shipped | not yet | not yet |
| Rust | shipped | planned R7 | planned R8 | planned R10 | planned R9 |
| C# | shipped + planner | partial | partial | partial | partial |
| Go | shipped | not yet | not yet | not yet | not yet |
| Elixir | shipped + generator-mode | not yet | not yet | not yet | not yet |
| Python | shipped + `yield from` | not yet | not yet | not yet | not yet |
| **F#** | **none** | **none** | **none** | **none** | **none** |

The `eager` baseline is missing too because F# has no `WcFactSource`
shape at all - the `WamContext` exposes only in-memory
`Map<string, Map<string, string list>>` for `WcForeignFacts` and
`Map<string, Map<int, int list>>` for `WcFfiFacts`. There is no
fact-source trait, no LMDB template, and no .NET LMDB binding
selected. The closest precedent in the codebase is the C# ingest at
`src/unifyweaver/runtime/csharp/lmdb_ingest/` which uses the
[LightningDB](https://www.nuget.org/packages/LightningDB) NuGet
package (0.21+) - F# can consume the same package since it shares the
CLR with C#.

### Minimum Useful F# LMDB Adoption

The path mirrors the Rust R7 plan but is independent of it:

1. **Pick the binding.** LightningDB 0.21+ is the obvious choice
   (already used by `lmdb_ingest`); a `FSharp.LMDB` wrapper is not
   needed and would be premature.
2. **Add a `LookupSource` interface to `WamRuntime.fs`** with one
   method along the shape `member _.Lookup : int -> seq<int>` (or
   `seq<Value>` once interning vs raw int IDs is decided). This is
   the equivalent of the Haskell `FactSource` typeclass and the Rust
   `LookupSource` trait that R7 introduces.
3. **Add an `LmdbCursorLookup` implementation** reading the Phase 1
   resident layout (`int32_le` keys) from
   `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`. One shared cursor
   protected by a mutex is the simplest starting shape.
4. **Add `WcLookupSources : Map<string, LookupSource>`** to
   `WamContext` and a `dispatchLookup` path that prefers the
   registered lookup over `WcFfiFacts` when present.
5. **Codegen option `lmdb_materialisation(eager|lazy|cached)`** in
   `wam_fsharp_target.pl`. Initial PR can omit the `auto` token
   (resolver wiring) and the `cached` decorator; those land in a
   follow-up.
6. **Mustache template `templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache`**
   carrying the LMDB-zero-allocation pattern used by the Rust
   template at `templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache`.
7. **Tests** under `tests/test_wam_fsharp_target.pl` for codegen
   shape; a generated-project smoke test under `tests/core/` for
   end-to-end behavior against a small fixture LMDB.

`cached` mode (step beyond the minimum) follows the Haskell
`resident_cursor` pattern: a `CachedLookup` decorator wraps any
`LookupSource` with a sharded LRU. .NET's
`System.Collections.Concurrent.ConcurrentDictionary` plus a
`LinkedList<int>` for LRU ordering is the simplest portable choice;
`Microsoft.Extensions.Caching.Memory` is heavier than needed.

The plan's §9 "out-of-scope" list does not currently include F#; this
audit suggests adding F# explicitly so future readers know whether to
expect it in a given phase.

## CSR Reverse-Index Readiness

F# is **not yet** named in `WAM_REVERSE_INDEX_ARTIFACTS.md`. The
implementation plan's Phase C calls for "a small Rust reader,
co-located with the Rust LMDB sink/build path, with direct-index or
binary-search lookup. C# binding can follow once the format and policy
are stable." F# is a natural follower of the C# binding step since
both share the CLR and can consume the same `.csr.idx` / `.csr.val` /
`.csr.meta` files.

For F#, the minimum useful unit once Phase C has stabilized the
format is:

1. **Add a CSR-reader module** to the F# runtime tree (likely
   `src/unifyweaver/targets/fsharp_runtime/CsrReader.fs`) implementing
   the `io_policy(buffered_pread)` and `io_policy(buffered_pread_drop)`
   paths. `direct_io` can stay out-of-scope on the first pass.
2. **Wire the `reverse_index(csr(...))` option** into
   `wam_fsharp_target.pl`'s option parsing, including
   `id_encoding(int32_le)` and `phase(planning_only|cache_warmup|runtime_available)`.
3. **Phase enforcement**: reject `phase(runtime_available)` at codegen
   time until an F# runtime reverse-lookup API exists - matches the
   guard already specified in §9 Phase A of the artifacts doc.
4. **Tests** verifying identical descendant sets for sampled parents
   against the Rust-built CSR fixture used by the existing
   `tests/test_benchmark_reverse_csr_lookup.py`.

CSR work is gated on LMDB work because the cost-model resolver
(`WAM_REVERSE_INDEX_ARTIFACTS.md` §5) needs the parent-edge
`eager`/`lazy`/`cached` choice as input. Sequencing: F# LMDB first,
then F# CSR.

## Other Notable Gaps

These are smaller than the three above but worth tracking:

- **`succ/2`** is in the Clojure/Elixir baseline but not F#. Would
  land naturally with the ISO `succ_iso/2` / `succ_lax/2` work
  above.
- **`assertz/1` / `retract/1` for clauses with bodies** - currently
  documented as "use Python WAM target if you need this". A real fix
  would require a heap-resident clause store rather than
  `WcLoweredPredicates` mutation. Not gating other work.
- **`docs/WAM_FSHARP_TARGET.md` "Control" line** incorrectly lists
  `throw/1` and `catch/3` as supported. Should be corrected when ISO
  substrate work begins, or sooner as a docs-only fix.

## Recommended Follow-Up Order

1. **Docs accuracy**: drop `throw/1` and `catch/3` from the
   "Control" line in `docs/WAM_FSHARP_TARGET.md`, or mark them
   "planned (see WAM_FSHARP_PARITY_AUDIT.md)". One-line PR.
2. **ISO substrate Phase 1**: `WamException` + catch/throw + ISO
   error constructors + `throw_iso_error`. Self-contained and
   unblocks every later ISO-mode builtin. Matches the C++/Elixir
   "minimum useful adoption unit".
3. **ISO substrate Phase 2**: per-predicate config loader + default
   rewrite + `is_iso/2` / `is_lax/2` + comparison aliases +
   `wam_fsharp_iso_audit/3`. Lands once Phase 1 is in.
4. **LMDB Phase 1 (eager + LookupSource interface)**: introduces
   the trait shape without committing to `lazy`/`cached`. Lets later
   phases land independently.
5. **LMDB Phase 2 (lazy)**: `LmdbCursorLookup` + the
   `lmdb_materialisation(lazy)` codegen option.
6. **LMDB Phase 3 (cached)**: `CachedLookup` decorator + auto
   resolver.
7. **`succ/2`** small builtin: bundle with the ISO `succ_iso/2` /
   `succ_lax/2` PR.
8. **CSR reader**: gated on LMDB work and on Rust CSR Phase C
   stabilising the format.

## Verification Commands

Use these checks after touching F# WAM parity:

```sh
swipl -q -g run_tests -t halt tests/test_wam_fsharp_target.pl
swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_runtime_smoke.pl
swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_dotnet_smoke.pl
swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_parser_smoke.pl
swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_lowered_smoke.pl
swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_lowered_parser_smoke.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_fsharp_target), halt"
```

The dotnet smoke tests need the .NET 8 SDK on `PATH` and should be
run under `LANG=C.UTF-8` (the test files contain em-dashes that the
default POSIX locale rejects).
