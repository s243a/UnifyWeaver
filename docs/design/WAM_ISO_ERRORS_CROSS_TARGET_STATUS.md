# WAM ISO Errors - Cross-Target Status

This note generalizes the C++ ISO-error design into the reusable target
contract, and records how much of that contract has shipped so far.

Companion docs:

- `WAM_CPP_ISO_ERRORS_PHILOSOPHY.md` - why the three-form design was chosen.
- `WAM_CPP_ISO_ERRORS_SPECIFICATION.md` - concrete C++ implementation shape.
- `WAM_ELIXIR_GAPS_SPECIFICATION.md` - Elixir parity inventory.
- `RUNTIME_PARSER_TRANSPILATION_PHILOSOPHY.md` - a separate cross-target
  runtime-parser effort; related because both are target-runtime libraries, but
  independent of ISO-error dispatch.

## Shared Contract

Targets that adopt ISO runtime errors should expose three callable forms for
each audited builtin:

| Form | Example | Meaning |
|---|---|---|
| Default | `is/2` | Rewritten by the generator according to the enclosing predicate's ISO mode. |
| Explicit ISO | `is_iso/2` | Always throws structured ISO errors. |
| Explicit lax | `is_lax/2` | Always preserves lax/fail-style behavior. |

The generator resolves default call sites at compile time, per predicate. A
runtime-wide `iso_errors` flag is intentionally avoided: it adds hot-path
branches, makes per-predicate migration awkward, and makes explicit lax/ISO
overrides harder to reason about.

The shared config shape is a Prolog facts file:

```prolog
iso_errors_default(true).
iso_errors_override(legacy_lookup/3, false).
iso_errors_override(experimental:my_pred/2, true).
```

Inline options may override the file:

```prolog
write_wam_TARGET_project(Preds, [
    iso_errors_config('config/iso_errors.pl'),
    iso_errors(false),
    iso_errors(my_pred/2, true)
], Dir).
```

Bare `Name/Arity` overrides match predicates in any module. A
`Module:Name/Arity` override is module-scoped. Targets should warn when a bare
override matches multiple modules, because that is usually a migration
footgun.

## Current Implementation Status

Survey columns: shipped means code and tests exist in the target today.

| Component | C++ | Elixir | Other WAM targets |
|---|---|---|---|
| Prolog config loader (`iso_errors_config/1`, inline overrides) | shipped | shipped | Python and F# plumbing shipped; other targets not adopted |
| Bare-PI multi-module warning | shipped | shipped | Python and F# plumbing shipped; other targets not adopted |
| Per-predicate default rewrite | shipped | shipped | Python and F# plumbing shipped; other targets not adopted |
| Text-path rewrite coverage (`builtin_call`, `put_structure`, `call`, `execute`) | shipped | shipped | target-specific |
| Audit predicate and report | `wam_cpp_iso_audit/3` | `wam_elixir_iso_audit/3` | `wam_python_iso_audit/3`, `wam_fsharp_iso_audit/3`; others not adopted |
| `catch/3` + `throw/1` substrate | shipped | shipped | Python and F# shipped; others mostly missing/partial |
| Error constructors and `throw_iso_error` helper | shipped | shipped | Python and F# shipped; others not adopted |
| `is_iso/2` / `is_lax/2` | shipped | shipped | Python and F# shipped; others not adopted |
| ISO/lax arithmetic compares | shipped | shipped | Python and F# shipped; others not adopted |
| `succ_iso/2` / `succ_lax/2` | shipped | shipped | Python and F# shipped; others not adopted |
| Lax IEEE-754 float divide behavior | shipped | shipped | Python shipped; F# partial (float div by zero returns nan/inf via CLR; integer div by zero fails silently); others not adopted |

The C++ and Elixir targets are therefore the current reference consumers. C++
was the first implementation; Elixir proves the design is not C++-specific.
Python adopted the catch/throw substrate, ISO error constructors,
`throw_iso_error`, per-predicate config/rewrite/audit plumbing, arithmetic
assignment variants, arithmetic comparison variants, and successor variants. F#
now matches that adoption surface (substrate + config/rewrite/audit +
`is_iso/2` / `is_lax/2` + six arithmetic-compare ISO/lax variants +
`succ/2` family). Neither Python nor F# should be described as fully
ISO-error compatible until remaining concrete builtins also adopt three-form
keys. R, Lua, Rust, and the remaining targets are still mostly missing
ISO three-form adoption. **Haskell** now has a catch/throw + `is_iso`
substrate and smoke tests (`test_wam_haskell_iso_smoke.pl`, PRs
#2510/#2526) but is not yet listed as a reference adopter alongside
C++/Elixir/F#/Python — treat it as partial until the shared status
table is refreshed end-to-end.

## What Counts As Adoption

A target should not claim this feature after adding only `is_iso/2`. The
minimum useful adoption unit is:

1. Runtime support for Prolog `catch/3` and `throw/1`.
2. Constructors for `error(ErrorType, Context)` terms.
3. The shared config shape, including inline overrides.
4. Per-predicate rewriting of default builtin keys.
5. Explicit `_iso` and `_lax` keys that survive mode flips.
6. Tests showing ISO-mode default calls throw, lax-mode defaults fail or use
   lax behavior, and explicit overrides bypass the rewrite.
7. An audit predicate or equivalent report that shows reviewers what each call
   site resolves to.

The audit predicate name can remain target-specific until at least three
targets need it. At that point, the shared pieces are good candidates for a
common Prolog helper module:

- config loading and merge precedence;
- predicate-indicator matching and bare-PI warnings;
- default/ISO/lax key tables;
- audit record formatting.

## Relationship To Runtime Parser Transpilation

ISO errors and runtime parser transpilation solve different problems.

ISO errors are about how existing runtime operations report malformed
arguments. The key design choice is compile-time dispatch among default,
explicit ISO, and explicit lax builtins.

Runtime parser transpilation is about letting generated programs parse Prolog
source text at runtime. The R target is the reference consumer because it has
`read/2`, `read_term_from_atom/2,3`, reverse `term_to_atom/2`, and CLI term
parsing. R currently keeps its native inline parser as the hot path, while the
portable parser in `src/unifyweaver/core/prolog_term_parser.pl` acts as the
cross-target semantic source and compile/runtime guard.

The two efforts may meet when parser-dependent builtins need ISO-style error
reporting, but neither depends on the other.

## Remaining Work

- Shared ISO config/audit helpers were extracted into
  `src/unifyweaver/core/iso_errors.pl` once F# became the third adopter.
  Python, Elixir, and F# all `use_module` from there now and only keep
  the target-specific parts (key-table assertions, `iso_errors_rewrite_text`
  variants, target audit wrapper).  C++ has its own implementation in C++
  source and is intentionally separate.  Tests in `tests/test_iso_errors.pl`.
- Decide which target should be the next adopter based on real `catch/3` /
  arithmetic-error users, not just target popularity.
- Keep C++ and Elixir docs in sync with the shipped state; older parity plans
  should be treated as history once their PR sequence has landed.
- Expand the builtin table only after each builtin is audited for ISO term
  shape, lax behavior, and happy-path cost.
