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
| ISO/lax arithmetic compares | Present | All six variants (`>`, `<`, `>=`, `=<`, `=:=`, `=\\=`) ship with `_iso/2` and `_lax/2` aliases. The lax aliases share the existing lax body via OR-pattern dispatch; the ISO variants classify failures as `instantiation_error` (unbound side) or `type_error(evaluable, X/N)`. |
| `succ/2` and ISO/lax variants | Present | `succ/2` (bidirectional successor with lax silent-fail) plus `succ_iso/2` (raises `instantiation_error` / `type_error(integer, _)` / `type_error(not_less_than_zero, X)` / `domain_error(not_less_than_zero, Y)` per Aït-Kaci spec §6) and `succ_lax/2`. |
| Lax IEEE-754 float divide | Partial | F# `is/2` already returns `nan`/`inf`/`-inf` for float zero division (CLR default). Integer divide-by-zero in lax mode fails silently because `evalArith` returns `None` (no exception escapes), matching the documented lax contract. |
| Per-predicate ISO config loader | Present | `iso_errors_config(File)`, inline `iso_errors(Default)` / `iso_errors(PI, Mode)`, file-vs-inline precedence per spec. Copied from Python target; future extraction into `src/unifyweaver/core/iso_errors.pl` is appropriate now that F# is the third adopter. |
| Per-predicate default rewrite | Present | `iso_errors_rewrite_text/4` walks WAM text, rewriting `is/2` -> `is_iso/2` / `is_lax/2` according to the predicate's resolved mode. Wired into `compile_predicates_to_fsharp` so generated F# uses the ISO-mode key. |
| ISO audit predicate | Present | `wam_fsharp_iso_audit/3` reports per-call-site resolution following the shared audit shape. `wam_fsharp_iso_audit_report/1` pretty-prints. |

### Remaining F# ISO Work

The arithmetic-compare sweep and `succ/2` family shipped in a
follow-up PR alongside the substrate work. F# now matches the full
"minimum useful adoption unit" in
`WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` § "What Counts As Adoption"
plus the C++/Elixir/Python reference key tables (minus `read/*` /
`read_term*/*`, which are runtime-parser-dependent and live on the
F# runtime-parser side).

Remaining cross-cutting work:

- _None at this layer._  The shared helper extraction landed once F#
  became the third adopter -- `src/unifyweaver/core/iso_errors.pl`
  now hosts the config loader, mode resolver, multi-module warning,
  item-level rewrite, and audit-walker primitives.  Python, Elixir,
  and F# all `use_module` from there and only keep their key-table
  assertions, `iso_errors_rewrite_text` parser, and target-specific
  audit wrapper.  Shared-module tests live in
  `tests/test_iso_errors.pl`.

## LMDB Fact-Source Readiness

F# now has all three materialisation modes from
`WAM_LMDB_LAZY_SPECIFICATION.md`:

| Target | `eager` | `lazy` | `cached` | Scan | Segregation |
| --- | :-: | :-: | :-: | :-: | :-: |
| Haskell | shipped | degenerate (cap 0) | shipped | not yet | not yet |
| Rust | shipped | planned R7 | planned R8 | planned R10 | planned R9 |
| C# | shipped + planner | partial | partial | partial | partial |
| Go | shipped | not yet | not yet | not yet | not yet |
| Elixir | shipped + generator-mode | not yet | not yet | not yet | not yet |
| Python | shipped + `yield from` | not yet | not yet | not yet | not yet |
| **F#** | **shipped** | **shipped** | **shipped** | not yet | not yet |

### What shipped

- **LightningDB 0.21** NuGet package (same CLR binding as
  `src/unifyweaver/runtime/csharp/lmdb_ingest/`).
- **`ILookupSource`** interface in `WamTypes.fs` with
  `member _.Lookup : int -> int list`.
- **`EagerLookupSource`** wraps a pre-loaded `Map<int, int list>`
  (Phase 1 eager materialisation; zero-cost unwrap via `.Data`).
- **`LmdbCursorLookup`** opens a per-call ReadTransaction + cursor
  (Phase 2 lazy mode; no startup cost, per-lookup cursor overhead).
- **`CachedLookupSource`** decorates any `ILookupSource` with a
  `ConcurrentDictionary<int, int list>` memo cache (flat unbounded).
- **`TwoLevelCachedLookupSource`** — per-thread L1 array (4096
  slots, collision-overwrite) + shared L2 `ConcurrentDictionary`
  (default 512k entries ≈ 8 MB). L2 grows lazily; cap prevents
  unbounded growth. For full enwiki (~2M category nodes), pass
  `maxL2Entries = 2_000_000` (~80 MB) for heavy-batch workloads.
- **`DictLookupSource`** wraps `Dictionary<int, int list>` for O(1)
  eager access without the cost of immutable Map.
- **`resolveFactLookup`** returns `int -> int list` dispatching
  through `ILookupSource.Lookup`; kernel codegen uses this so the
  DFS never materialises the full relation into a Map.
- **`lmdb_path(Path)`** codegen option: includes `LmdbFactSource.fs`
  + `LightningDB` NuGet reference in the generated project.
- **`lmdb_materialisation(eager|lazy|cached)`** codegen option
  (default `cached`); logged at project-generation time.
- Template: `templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache`.
- E2E tests: `tests/core/test_wam_fsharp_lmdb_smoke.pl` (18
  assertions against a synthetic Phase 1 LMDB fixture).

### Remaining LMDB work

- **`lmdb_materialisation(auto)`**: **Shipped** as
  `resolve_auto_lmdb_materialisation_fs/2` in `wam_fsharp_target.pl`.
  The unconditional `cached` default remains correct for
  effective-distance workloads; `auto` exists for workloads that may
  prefer eager (full-graph scan with high demand ratio) — rare in
  practice. Earlier drafts of this audit listed `auto` as future; that
  wording is obsolete.
- **Scan-mode** and **workload-segregation** contract: Rust R9/R10;
  out of scope for F# until Rust ships the reference.
- **L2 cache sizing**: defaults to `auto` (runtime memory-based
  formula matching Haskell). Also supports named sizes (`tiny`,
  `small`, `medium`, `large`), corpus presets (`enwiki`,
  `simplewiki`, `dev`), byte budgets (`80mb`), and explicit entry
  counts. Codegen option: `lmdb_l2_capacity(auto|small|enwiki|...)`.
  See `lmdb_fact_source.fs.mustache` resolveL2Capacity for the full
  resolution table.

## CSR Reverse-Index Readiness

**Status**: Implemented (Phase 1). The F# `CsrLookupSource` reads
the `unifyweaver.reverse_csr.v1` binary format (`.csr.idx` /
`.csr.val` / `.csr.meta`) and implements `ILookupSource` for reverse
child-edge lookup (parent -> children).

### What's done

1. **`CsrReader.fs` template** (`templates/targets/fsharp_wam/csr_reader.fs.mustache`):
   `CsrLookupSource` loads the index into memory and reads the values
   file via positioned seek+read (thread-safe via lock). Validates the
   JSON manifest format and id_encoding. Implements `IDisposable`.

2. **`csr_path(Path)` codegen option** in `wam_fsharp_target.pl`:
   conditional `.fsproj` inclusion of `CsrReader.fs` (compiled after
   `WamTypes.fs`, before `WamRuntime.fs`). No extra NuGet dependencies
   (uses `System.IO` + `System.Text.Json` from .NET 8 SDK).

3. **Composable with TwoLevelCachedLookupSource**: wrapping
   `CsrLookupSource` in the two-level cache gives the same L1/L2
   caching benefits as LMDB cached mode.

4. **Tests**:
   - `tests/core/test_wam_fsharp_csr_smoke.pl` -- E2E smoke (50 parents, correctness)
   - `tests/core/test_wam_fsharp_csr_bench.pl` -- CSR vs LMDB benchmark (500 parents)

### Benchmark (500 parents x 6 children = 3000 edges)

| Mode | median_ms | vs LMDB cursor |
|---|---:|---|
| CSR raw | 0.55 | 2.7x faster |
| CSR cached (L1/L2) | 0.06 | 24.5x faster |
| LMDB cursor | 1.47 | baseline |

### What's next

- `io_policy(buffered_pread_drop)` -- `posix_fadvise(DONTNEED)` after reads
- `io_policy(direct_io)` -- `O_DIRECT` for page-cache isolation
- `index_backend(lmdb_offset)` -- LMDB offset index for sparse IDs
- Phase enforcement (`planning_only` / `cache_warmup` / `runtime_available`)
- `reverse_index(csr(...))` declarative option parsing
- Integration with effective-distance kernel (descendant-path exploration)

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
8. **CSR reader**: done. `CsrLookupSource` reads
   `unifyweaver.reverse_csr.v1` format, wired via `csr_path(Path)`.
   Next: `io_policy` variants and phase enforcement.

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

# CSR tests (need python3 + lmdb + .NET 8 SDK):
swipl -g main tests/core/test_wam_fsharp_csr_smoke.pl
swipl -g main tests/core/test_wam_fsharp_csr_bench.pl
```

The dotnet smoke tests need the .NET 8 SDK on `PATH` and should be
run under `LANG=C.UTF-8` (the test files contain em-dashes that the
default POSIX locale rejects). The CSR tests additionally need
`python3` with the `lmdb` package.
