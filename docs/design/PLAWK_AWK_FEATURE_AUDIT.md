<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk — AWK feature audit & roadmap

**Status**: living checklist (audited 2026-07). plawk is a *specialised* awk-like
front-end for the hybrid WAM/LLVM target, not a full AWK clone — it adds things
POSIX AWK has no notion of (multi-pass caches, `over query(Goal)` readers,
generator blocks, durable LMDB stores, a Prolog foreign-call surface). This doc
tracks how much of the *basic* AWK surface exists, so the DSL stays familiar to
AWK users, and prioritises the gaps. Legend: ✅ done · ◐ partial · ⏳ surface
only (runtime pending) · ❌ missing.

## Patterns

| Feature | Status | Notes |
|---|---|---|
| `BEGIN` / `END` | ✅ | incl. constant `print` (BEGIN/END literal print) |
| `/regex/` | ◐ | leading `^` prefix + literal-contains; general ERE metachars limited |
| `$N == "v"`, `$3 > 100` | ✅ | field-equality + numeric field guards |
| `&& \|\| !` combinators | ✅ | awk precedence, parens, single-block lowering |
| `~` / `!~` match | ✅ | POSIX ERE |
| expression pattern (bare `NR==1`) | ◐ | comparison guards yes; arbitrary expr no |
| range pattern (`/a/,/b/`) | ❌ | |

## Actions & control flow

| Feature | Status | Notes |
|---|---|---|
| `print` (fields, literals, `NR`/`NF`, `length`/`substr`/`index`/`tolower`/`toupper`, arithmetic) | ✅ | constant fields (`print 1`, `print "x"`) landed |
| `printf` | ◐ | subset `%%`,`%s`,`%d`,`%i`,`%ld`; no `%f`/`%c`/`%x`/width/precision |
| var assignment, `+=`, `++`, `//` | ✅ | indexed native scalar slots |
| `if` / `else` (chains) | ✅ | |
| `for (k in arr)` | ✅ | assoc for-in (rule body + END) |
| `while (VAR CMP int)` | ⏳ | surface parses; runtime pending — `PLAWK_CONTROL_FLOW_PLAN.md` |
| `do { } while (VAR CMP int)` | ⏳ | surface parses; shares the loop runtime |
| `next` | ✅ | structural (guarded clause per rule) |
| `break` / `continue` | ◐ | present in some loop contexts |
| `if` with a plain (non-accumulator) body | ❌ | `{ if (c) { print $1 } }` is exit 3 — the scalar `if` lowering assumes branch bodies update scalars; blocks regex-in-`if` too |
| regex in `if` (`if ($0 ~ /re/)`) | ◐ | condition parses (`~` ok); blocked by the guarded-print body above, not the regex |
| brace-less `if`/loop body | ❌ | `if (c) print` (no braces) doesn't parse |
| field assignment (`$2 = expr`) | ❌ | rebuilding `$0` from mutated fields not wired |
| C-style `for (;;)` | ❌ | |
| `exit [n]` | ❌ | |
| `delete arr[k]` | ❌ | |
| `getline` | ❌ | the multi-pass / `over` readers cover much of its use |

## Functions

| Feature | Status | Notes |
|---|---|---|
| user functions (`function f(a) { return … }`) | ✅ | compile to Prolog clauses; work in accumulator / typed-field (`BINFMT`) contexts **and text mode** — `print f($1)` auto-coerces the field (awk semantics). |
| string: `length` `substr` `index` `tolower` `toupper` | ✅ | native, allocation-free |
| string: `split` `sub` `gsub` `match` `sprintf` | ❌ | |
| numeric: `int` | ✅ | `sin`/`cos`/`sqrt`/`rand`/… ❌ (f64 machinery exists) |
| Prolog foreign calls (`pred($1)`, `float(pred(…))`) | ✅ | plawk-specific, beyond AWK |

## Variables & operators

| Feature | Status | Notes |
|---|---|---|
| `$0` `$N` `NR` `NF` `FS` `OFS` | ✅ | |
| `FNR` `FILENAME` `ARGV` `ARGC` `RS` `ORS` `SUBSEP` `RSTART` `RLENGTH` | ❌ | single newline-delimited record model |
| arithmetic `+ - * / % //` | ✅ | i64, awk precedence, safe div/mod |
| comparison, `~`/`!~` | ✅ | |
| ternary `?:` | ❌ | does not parse |
| string concatenation (juxtaposition `$1 $2`) | ❌ | does not parse |
| exponentiation `^` / `**` | ❌ | |

## Arrays

| Feature | Status | Notes |
|---|---|---|
| assoc arrays `arr[k]` | ✅ | native assoc tables |
| `k in arr` (membership / for-in) | ✅ | |
| multi-dim `arr[i,j]` (SUBSEP) | ❌ | |
| `delete` | ❌ | |

## plawk-specific surface (beyond AWK)

Multi-pass `pass { }` blocks · `cache(...)` (file / LMDB, namespaces, multi-table)
· `over TABLE` / `records of` / `rows of` / `over query(Goal)` readers · reader
guards · generator blocks (`gen { emit … } as name`, input iterators) ·
`@prolog` blocks · `compile(...)` / `dyncall` eval surface · binary records
(`BINFMT`) / DCG readers.

## Prioritised gaps (recommended order)

1. **`while` / `do-while` loop runtime** — the surfaces just landed; wire the
   loop (mutable scalar state to a fixed point). **Plan:**
   `PLAWK_CONTROL_FLOW_PLAN.md` — bracket each loop with a memory slot (SSA →
   mem → loop → SSA) so the existing forward-phi scalar machinery is untouched.
   The most-requested basic control structure still missing at runtime.
2. **User-function call in text/print context** — `print f($1)` returns `0`: a
   text field is passed to the foreign call as an *atom*, so the synthesised
   `f(X,R) :- R is X*2` fails `is`. Works in `BINFMT`/typed mode.
   **Decision: auto-coerce (awk semantics).** A field (a string) used in a
   numeric context coerces to a number, matching AWK. Implementation options
   (a design call for the PR, not the surface): coerce in the WAM arithmetic
   builtins (an atom operand of `is`/`>`/`*`/… parses to a number — most
   awk-faithful, but the WAM engine is shared, so scope/gate it to avoid
   changing Prolog `is/2` for non-plawk callers), or coerce the field arg at the
   plawk call boundary when the callee uses it numerically (needs light body
   inference). No explicit `num($1)` is required of the user.
   *Future direction — a **performance** lever, not just safety:* **optional
   type annotations** on function inputs (e.g. `function f(num x)` / a `:- ftype`
   directive). Because plawk compiles to **typed WAM**, a declared arg type would
   (a) **skip the runtime coercion** — the value is already the right type, so
   the hot path pays nothing (auto-coerce parses `atom → number` on every call),
   and (b) enable **compile-time type checking** (a mismatched call is an error).
   This is the same principle as explicit `emit` in generators
   (`PLAWK_GENERATOR_BLOCKS.md` §2.1): **static type knowledge lets the compiler
   elide coercion and serialisation.** Given the per-call coercion cost, this
   ranks higher than a pure nice-to-have — it is the typed-fast path over the
   dynamic-correct default. Layered *on* auto-coerce (which stays the annotation-
   free default), not instead of it; whether/when to build it is open.
2b. **`if` with a plain guarded body** — `{ if (c) { print $1 } }` doesn't
   compile (the scalar `if` lowering assumes branch bodies update scalars); this
   is what actually blocks regex-in-`if`. Fix the `if`-body lowering.
3. **`printf` format coverage** — add `%f`/`%g` (f64 exists), `%c`, `%x`, and
   width/precision; the current subset is thin for real formatting.
4. **`exit [n]`** — common and cheap; a flagged early-terminate of the record
   loop + END.
5. **String concatenation & ternary `?:`** — the two most-missed *expression*
   forms; both are parser + expression-lowering work.
6. **`delete arr[k]`** — rounds out the assoc-array story (paired with the
   existing for-in / `in`).
7. **`split` / `sub` / `gsub` / `match` / `sprintf`** — the string-builtin
   family; larger, lower priority for the DSL's data-pipeline focus.

Deferred by design (the DSL's model diverges here): `RS`/multi-char record
separators, `getline` (the `over`/multi-pass readers subsume most uses),
multi-file `ARGV`/`FILENAME`/`FNR`, C-style `for(;;)` / `do-while` (the `while`
runtime + for-in cover the loop needs).
