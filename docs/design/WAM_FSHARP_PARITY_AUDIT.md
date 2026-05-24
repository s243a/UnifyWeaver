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
| Type builtins | `atom/1`, `atomic/1`, `compound/1`, `integer/1`, `number/1`, `float/1`, `var/1`, `nonvar/1` | Same baseline; Lua/Python/Go also have `is_list/1` | Present **except `is_list/1`** (not currently wired) |
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

F# is **not yet** an ISO-error adopter. Per
`WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`, the reference consumers are
C++ and Elixir (both fully shipped) and Python (substrate + arithmetic
variants shipped, remaining concrete builtins outstanding). F# has
none of the components below.

| Component | F# status | Notes |
| --- | --- | --- |
| Prolog `catch/3` / `throw/1` substrate | **Missing** | `WamRuntime.fs` has no `WamException` type and no catcher frames on `WsStack` / `WsCPs`. `docs/WAM_FSHARP_TARGET.md` currently lists `throw/1` and `catch/3` under "Control" - **that line is inaccurate** and should be removed or marked "planned" when this audit lands. The `throw` calls in `wam_fsharp_target.pl` and `wam_fsharp_lowered_emitter.pl` are codegen-time Prolog `throw/1`, not WAM-runtime catch/throw. |
| ISO error constructors | **Missing** | No runtime builders for `instantiation_error`, `type_error/2`, `domain_error/2`, `evaluation_error/1` in `WamRuntime.fs`. |
| `throw_iso_error` helper | **Missing** | Depends on the substrate above. |
| `is_iso/2` / `is_lax/2` | **Missing** | Current `is/2` is lax by construction (arithmetic failures fail silently / return `nan`/`inf` via `Double.NaN` and `Double.PositiveInfinity`, matching the F# CLR float semantics). That is a viable starting point for `*_lax` aliases once the three-form split exists. |
| ISO/lax arithmetic compares | **Missing** | Six comparison variants would need ISO/lax three-form dispatch. |
| `succ/2` and ISO/lax variants | **Missing** | F# does not currently expose `succ/2`. |
| Lax IEEE-754 float divide | Partial | F# `is/2` already returns `nan`/`inf`/`-inf` for float zero division (CLR default). Integer zero division throws `DivideByZeroException` which propagates as a runtime crash rather than failing silently; that needs adjustment for lax mode. |
| Per-predicate ISO config loader | **Missing** | No `iso_errors_config(File)`, no inline `iso_errors(Default)` / `iso_errors(PI, Mode)` option parsing in `wam_fsharp_target.pl`. |
| Per-predicate default rewrite | **Missing** | Text-level rewrite from `is/2` to `is_iso/2`/`is_lax/2` would feed both the interpreter and lowered emitter; neither is wired today. |
| ISO audit predicate | **Missing** | No `wam_fsharp_iso_audit/3`. |

### Minimum Useful F# ISO Adoption

Per the shared contract in `WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` §
"What Counts As Adoption", the minimum useful unit for F# is:

1. Add a `WamException` exception type (carries the ISO error term).
2. Add catcher-frame plumbing on `WsCPs` so `step` can unwind on
   `throw/1` and resume at a matching `catch/3` handler. Pattern:
   either a new `ChoicePoint` variant with `CpCatcher: Value option`,
   or a sibling stack on `WamState` for catcher frames. The Elixir
   target's side-stack pattern is the closest reference.
3. Add ISO error constructors (`instantiation_error/0`,
   `type_error/2`, `domain_error/2`, `evaluation_error/1`) and a
   `throw_iso_error` helper in `WamRuntime.fs`.
4. Per-predicate config/override parsing in `wam_fsharp_target.pl`,
   mirroring the option shape already shipped by C++/Elixir/Python.
5. Default-key rewrite for `is/2`; `is_iso/2` and `is_lax/2`
   implementations that survive the rewrite.
6. Six ISO/lax arithmetic-compare aliases (`=:=`, `=\=`, `<`, `>`,
   `=<`, `>=`).
7. `wam_fsharp_iso_audit/3` reporting builtin call sites using the
   shared audit shape.
8. Tests under `tests/core/test_wam_fsharp_iso_*.pl` matching the
   shape of the existing Python ISO E2E coverage.

Steps 1-3 form a single self-contained PR; the rest can land in
follow-ups once the substrate exists.

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

- **`is_list/1`** is in the cross-target baseline (Lua, Python, Go)
  but not currently wired in F#. Trivial addition - one match clause
  in `WamRuntime.fs` `step` plus a Prolog dispatch entry.
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
7. **`is_list/1` and `succ/2`** small builtins: bundle with whichever
   ISO PR they fit alongside.
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
