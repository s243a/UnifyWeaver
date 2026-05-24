# WAM Runtime Parser — Cross-Target Status & Plan

This is a cross-target status doc for the **runtime Prolog-term parser**:
the facility that lets generated WAM target code call `read_term_from_atom/2`,
`read_term_from_atom/3`, `term_to_atom/2` (reverse mode), `read/1,2`,
`read_term/1,2` and friends — i.e. parse Prolog source text into runtime
terms at execution time.

Targets diverge in **how** they provide this facility (and whether they
provide it at all).  This doc explains the three approaches, the
per-target current state, the bugs found during the audit, and what the
sensible next steps are.

The canonical capability table lives in
[`src/unifyweaver/targets/wam_runtime_parser_capability.pl`](../src/unifyweaver/targets/wam_runtime_parser_capability.pl)
(predicates `wam_target_runtime_parser/3` +
`target_supports_runtime_parser_mode/2`).  When this doc and that file
disagree, the file wins; please update both.

## The three parser modes

```
runtime_parser(off)         no source-term parser; calls reject at codegen
runtime_parser(native)      host-language parser; usually canonical-only
runtime_parser(compiled)    bundle the portable WAM-compiled parser
runtime_parser(auto)        per-target default (the entry below)
```

### `off` — no parser available

User code can't call `read_term_from_atom`, `term_to_atom(_, +)`, or
`read/1,2`.  Doing so is **rejected at codegen time** with a
`permission_error(use, runtime_parser, ...)` exception, so the failure
mode is loud rather than silent.  This is the safe default for any
target that doesn't have one of the two real implementations wired up.

### `native` — host-language parser

The host language ships its own term parser, hand-written in the target
language.  Pros: no parser library to bundle (small generated code),
fast.  Cons: usually doesn't support operator notation — `1+2` has to be
written as `+(1, 2)` in canonical form.  Used for things like LMDB fact
keys, atom-codes round-trips, and `term_to_atom/2`'s reverse mode where
the input is well-typed.

### `compiled` — bundled portable parser

The portable Prolog-term parser
([`src/unifyweaver/core/prolog_term_parser.pl`](../src/unifyweaver/core/prolog_term_parser.pl))
is compiled to WAM and the WAM-compiled predicates get prepended to the
target project's predicate list (along with `read_term_from_atom/2,3`
and friends from
[`src/unifyweaver/core/cpp_runtime_parser_wrappers.pl`](../src/unifyweaver/core/cpp_runtime_parser_wrappers.pl)).
The user predicate calls the wrapper, the wrapper calls
`parse_term_from_atom/3`, which is a normal WAM predicate that runs on
the target's interpreter.

Pros: full operator notation (everything `read_term/1` on a normal
Prolog system handles), single source of truth, every target that
bundles it inherits parser improvements for free.  Cons: the generated
project ships an extra ~45 predicates of parser code (kilobytes to
megabytes of generated source, depending on how the target encodes
each WAM instruction).

## Per-target current state

| target | `none` | `native` | `compiled` | default | end-to-end tested |
| --- | :---: | :---: | :---: | --- | --- |
| **F#**       | ✅ | — | ✅ | `none` | yes — `test_wam_fsharp_parser_smoke.pl` 42/42 |
| **Python**   | ✅ | — | ✅ | `none` * | yes — `test_wam_python_target.pl` `wam_python_runtime_parser_mode` block |
| **C++**      | ✅ | ✅ | ✅ | `native(parse_term)` | partial — codegen tests in `test_wam_cpp_generator.pl`, one `'42'` parse verified end-to-end during the audit |
| **R**        | ✅ | ✅ | ✅ | `native(parse_term)` | codegen tests; runtime end-to-end via the R smoke harness |
| **Elixir**   | ✅ | — | — | `none` | n/a — `runtime_parser(compiled)` correctly throws `domain_error` (since Elixir isn't in the capability table); guarded by `test_runtime_parser_compiled_request_errors` |
| **Rust**     | ✅ | — | — | `none` | n/a (no parser-mode wiring) |
| **Haskell**  | ✅ | — | — | `none` | n/a (no parser-mode wiring) |
| **Go**       | ✅ | — | — | `none` | n/a (no parser-mode wiring) |
| **Clojure**  | ✅ | — | — | `none` | n/a (no parser-mode wiring) |
| **Lua**      | ✅ | — | — | `none` | n/a (no parser-mode wiring) |

`*` Python doesn't have an explicit `target_runtime_parser_default/2`
fact, so `runtime_parser(auto)` falls through to `none`.  In practice
the codegen pulls the parser when the user requests `compiled`.

## Audit findings (cross-target backport of the F# fixes)

During the F# `MemberRetry`/`dispatchCall WsCP`/`putReg` perf cycle
the F# target accumulated a substantial set of runtime-parser fixes.
Auditing the other targets for the same bug classes turned up:

| target | finding | PR |
| --- | --- | --- |
| **F#** | 5 bugs across `member/2` backtracking, `==/2` list encoding, `findall` result-reg seeding, `dispatchCall` WsCP propagation, quoted-atom rendering | merged across #2415, #2419, #2422, #2423, #2424–#2425, #2428 |
| **Rust** | `rust_val_literal` re-emitted quoted-numeric atoms verbatim (`Value::Atom("'42'")` instead of `Value::Atom("42")`) — same class as F# #2422 | #2431 |
| **Python** | `_constant_term` re-parsed `Atom("42").name` through `_parse_constant`, silently promoting it to `Int(42)` at `put_constant` time; `wam_lines_to_python` used a naive whitespace split that broke any atom token containing a space (`':- p'`) | #2433 |
| **C++** | audit clean — uses `wam_text_to_items/2` + `wam_classify_constant_token/2` which already honour the quoted-atom convention.  One end-to-end `'42'` parse verified before the ~12-min-per-case compile time made `:- p` / `\+ foo` impractical to also verify. | none |
| **R, Elixir, Lua, Clojure, Haskell, Go, Rust** (compiled path) | not applicable — these targets don't bundle the compiled parser at all | none |

The bug classes are roughly:

1. **Quoted-atom rendering** — a source-level atom whose name parses as
   a number (`'42'`, `'-3.14'`) loses its atom-ness somewhere in the
   pipeline.  Symptom: `read_term_from_atom('42', T)` fails because the
   input never becomes an Atom that the parser can process.

2. **Naive whitespace tokenisation** — a target's WAM-text reader uses
   `split_string` on whitespace instead of `wam_text_parser:wam_tokenize_line/2`,
   so atoms containing spaces (`':- p'`, `'a b c'`) break.

3. **WAM-runtime invariants** — F#-specific issues (`MemberRetry` for
   non-deterministic `member/2` through a cut, immutable-Map snapshot
   semantics on backtrack, etc.) that don't have analogs in mutable-
   state targets (Python, Rust, etc. use direct trail unbinding rather
   than persistent maps).

## Approach divergence — is this intentional?

Yes, mostly.  The three modes exist for different workloads:

- **`off` / `none`** is right when the target predicates never read
  external Prolog source.  Pure compute workloads (graph reachability,
  recursive-kernel rewrites, materialised view rebuilds) don't need a
  parser and pay zero overhead.

- **`native`** is right when the target needs canonical-form parsing
  only — e.g. an LMDB fact source whose keys are well-typed
  `Compound("edge", [...])` strings.  C++ uses this for
  `LmdbFactSource`; R uses it similarly.  The hand-written native
  parser is fast and stays inside the host language's idioms.

- **`compiled`** is right when the target needs *operator-aware* parsing
  — e.g. interpreting user-supplied Prolog source like `1+2` or
  `p :- q ; r`.  This is the heavyweight mode; you ship ~45 extra
  predicates but you get every operator + variable-binding behaviour
  the portable parser supports.

The divergence in **which targets implement which modes** is partly
historical (parser-mode wiring landed first in the targets that needed
it for early benchmark workloads) and partly principled (a target with
no hand-written runtime parser shouldn't advertise `native` until
someone writes one).

## Plan forward

Three threads, roughly in order of impact:

### 1. Elixir parser bundling (if a user appears)

Elixir is not in the capability table for any parser mode, so
`runtime_parser(compiled)` and `runtime_parser(native)` both
correctly throw `domain_error` at codegen.  Regression-protected
by `test_runtime_parser_compiled_request_errors` in
`tests/test_wam_elixir_target.pl` (added alongside the existing
`test_runtime_parser_native_request_errors`).

(An earlier version of this doc described Elixir as having an
"advertised but stub" compiled mode -- that was inaccurate; the
capability resolver rejects before reaching the unreachable
`elixir_runtime_parser_mode_literal(compiled(_), _)` clause.  The
clause itself is dead forward-compat code; left in place in case a
later implementation needs it.)

If someone wants Elixir to actually support `runtime_parser(compiled)`,
the work is: add Elixir to the capability table, port Python's
`expand_python_runtime_parser_predicates/3` to prepend the parser
predicates to the project list, and add atom-codes / number-codes /
WAM-text plumbing to the Elixir runtime equivalent to Python's.
Significant work; worth it if there's a real user, not otherwise.

### 2. Add `compiled` to one more target with a real use case

Rust is the next natural candidate — it has the largest gap between
"target has WAM but no parser" and "target needs to read user terms"
(LMDB fact sources, anything reading config-file Prolog).  The work
mirrors what's already done for Python: append `prolog_term_parser`
predicates, write a Python-style `_execute_read_term_from_atom`
wrapper in Rust, expose it as a WAM builtin.

Pre-work that benefits everyone: factor out the existing C++ and
Python wrapper logic into a shared template so the Rust port (and any
future port) is mostly translation rather than re-design.

### 3. Lower-priority: cross-target test for the compiled path

The Python target now has end-to-end tests in
`tests/test_wam_python_target.pl` for bare-integer atoms,
prefix-directive atoms, and prefix-NaF atoms.  The C++ target has
end-to-end tests for parser-label presence in the generated source but
nothing that actually builds + runs.  Building C++ projects with the
~45-predicate parser library takes ~12 minutes per case unoptimised,
which is the blocker.

A cheap improvement: pre-compile the C++ runtime once into a static
library at fixture setup, then each per-case build only links the
generated-program against it.  Order-of-magnitude faster, and unlocks
end-to-end coverage of `:- p` / `\+ foo` / numeric quoted atoms for
C++ — bringing it to parity with Python's coverage.

### Not on the plan

- **Flipping the F# default to `compiled`.**  The parser now works
  correctly under F# but `compiled` ships ~45 extra predicates per
  project.  Keep it opt-in.
- **Adding `native` parsers to the targets without one** (F#, Python,
  Rust, etc.).  Each one is a from-scratch hand-written term-parser
  port and there's no current user need.  Use `compiled` instead.

## Related docs

- [`docs/WAM_FSHARP_TARGET.md`](WAM_FSHARP_TARGET.md) — the F# WAM target's full usage guide, including the parser-related runtime invariants.
- [`src/unifyweaver/targets/wam_runtime_parser_capability.pl`](../src/unifyweaver/targets/wam_runtime_parser_capability.pl) — the canonical capability + default table.
- [`src/unifyweaver/core/prolog_term_parser.pl`](../src/unifyweaver/core/prolog_term_parser.pl) — the portable parser source itself.
- [`src/unifyweaver/core/cpp_runtime_parser_wrappers.pl`](../src/unifyweaver/core/cpp_runtime_parser_wrappers.pl) — the `read_term_from_atom/2,3` wrappers used by both C++ and the other `compiled`-mode targets.
