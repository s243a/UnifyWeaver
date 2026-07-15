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
| `print` (fields, literals, `NR`/`NF`, `length`/`substr`/`index`/`tolower`/`toupper`, arithmetic, **concatenation**) | ✅ | constant fields (`print 1`, `print "x"`) + juxtaposition concat (`print $1 $2`) landed |
| `printf` | ◐ | subset `%%`,`%s`,`%d`,`%i`,`%ld`; no `%f`/`%c`/`%x`/width/precision |
| var assignment, `+=`, `++`, `//` | ✅ | indexed native scalar slots |
| `if` / `else` (chains) | ✅ | field/pattern guards (`$1 > 2`, `$0 ~ /re/`) **and scalar-variable conditions** (`if (i > 2)`, `if (i < n && j > 0)`) in rule bodies, loops, **and `END`** (`END { if (n > 1) print … ; else print … }`, over final slot values) |
| `for (k in arr)` | ✅ | assoc for-in (rule body + END) |
| `while (COND)` | ✅ | runtime landed (loop-header phis); condition is scalar comparisons (`VAR CMP int`/`VAR`) combined with `&&`/`||` — `PLAWK_CONTROL_FLOW_PLAN.md` PR 2–3. `break`/`continue` deferred (PR 3b) |
| `do { } while (COND)` | ✅ | runtime landed; body runs at least once; same general condition |
| `next` | ✅ | structural (guarded clause per rule) |
| `break` (rule-level stream break) | ✅ | non-standard awk extension; stops the record stream |
| `break` / `continue` (loop-local) | ◐ | **`while` loops: landed** (`break` leaves the loop, `continue` re-tests — SSA merge phis at the loop exit/head). `do-while` break/continue and nested-loop break: not yet (`PLAWK_CONTROL_FLOW_PLAN.md` §3b) |
| `if` with a plain (non-accumulator) body | ✅ | `{ if (c) { print $1 } }` compiles and runs (fixed by the while-runtime body-print enablers); if/else with plain bodies too |
| regex in `if` (`if ($0 ~ /re/)`) | ✅ | `if ($0 ~ /re/) { … }` and `!~` compile and run — guards a plain body |
| brace-less `if`/loop body | ✅ | `if (c) print`, `while (c) x++`, `do stmt while (c)`, braceless else-if chains — a body is a braced block or one statement |
| field assignment (`$2 = expr`) | ❌ | rebuilding `$0` from mutated fields not wired |
| C-style `for (;;)` | ❌ | |
| `exit [n]` | ❌ | |
| `delete arr[k]` | ❌ | |
| `getline` | ❌ | the multi-pass / `over` readers cover much of its use |

## Functions

| Feature | Status | Notes |
|---|---|---|
| user functions (`function f(a) { return … }`) | ✅ | compile to Prolog clauses; work in accumulator / typed-field (`BINFMT`) contexts **and text mode** — `print f($1)` auto-coerces the field (awk semantics). Optional numeric arg typing (`function f(num x)`) skips the coercion (typed-fast path). |
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
| string concatenation (juxtaposition `$1 $2`) | ◐ | **`print` context landed** — `print $1 $2`, `print "x" $1 "-" $2`, in rule bodies and `END`; arithmetic binds tighter, comma still splits. Assignment concat (`x = $1 $2`) needs string-valued scalars — separate |
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

1. **`while` / `do-while` loop runtime — LANDED (PR 2).** The loop iterates its
   mutable scalar state via **loop-header phis** (reusing `foreach_loop`'s head
   phis — no memory slots needed after all; see `PLAWK_CONTROL_FLOW_PLAN.md`).
   Body: `set` / `inc` / `+=` over i64 + `print`. Enablers landed with it:
   **bare scalar-var print** (`print i`) and a **body-printing scalar chain with
   no `END`**. **PR 3 (general condition) LANDED**: scalar comparisons
   `VAR CMP (int | VAR)` combined with `&&` / `||`. **PR 3b (`while`
   break/continue) LANDED**: SSA merge phis at the loop exit (`break`) and head
   (`continue`); scalar `if` conditions and END scalar-`if` landed alongside.
   Remaining: `do-while` break/continue, and nested / multi-pass loops (PR 4).
2. **User-function call in text/print context — LANDED (auto-coerce).**
   `print f($1)` used to return `0`: a text field reached the synthesised
   `f(X,R) :- R is X*2` as an *atom*, failing `is`. Now the synthesised clause
   coerces each untyped param inline — `(number(H) -> V = H ; atom_number(H, V))`
   — so a text field parses to a number (awk semantics) while a typed i64 arg
   (`BINFMT`) passes straight through. No engine change, no `num($1)` required.
   *Typed-fast path — LANDED (optional arg typing).* An optional numeric type
   annotation, `function f(num x)` (also `int` / `float`), declares the param
   already-numeric, so the synthesised clause **skips the coercion goal** — the
   head var *is* the value var. Because plawk compiles to **typed WAM**, a
   declared arg carries in its native representation and the hot path pays
   nothing (auto-coerce otherwise runs `number/1` every call). This is the same
   principle as explicit `emit` (`PLAWK_GENERATOR_BLOCKS.md` §2.1,
   `UNIFYWEAVER_LANGUAGE_PRINCIPLES.md` Principle 2): **static type knowledge
   elides coercion.** It is layered *on* auto-coerce (the annotation-free
   default), not instead of it; an unannotated param still auto-coerces. In text
   mode a typed param is a contract that the arg is numeric — a bare text field
   passed to it is the caller's responsibility. *Still open (follow-on):*
   compile-time **type checking** of call sites (a mismatched call → error) and
   distinguishing `int` vs `float` for representation; all three keywords
   currently mean "numeric, skip coercion".
2b. **`if` with a plain guarded body + regex-in-`if` + braceless bodies —
   LANDED.** `{ if (c) { print $1 } }`, `if ($0 ~ /re/) { … }` / `!~`, and
   if/else with plain bodies all compile and run (the guarded-print gap was
   fixed by the while-runtime body-print enablers — a plain body no longer has
   to update a scalar). Braceless bodies also landed: `if (c) print`,
   `while (c) x++`, `do stmt while (c)`, and braceless else-if chains — a
   control-flow body is a braced block *or* a single statement.
3. **`printf` format coverage** — add `%f`/`%g` (f64 exists), `%c`, `%x`, and
   width/precision; the current subset is thin for real formatting.
4. **`exit [n]`** — common and cheap; a flagged early-terminate of the record
   loop + END.
5. **Ternary `?:`** (string concatenation in `print` context **landed** —
   `print $1 $2`, rule bodies + `END`; the remaining concat work is assignment
   into a string-valued scalar). Ternary is the next most-missed *expression*
   form; parser + expression-lowering work.
6. **`delete arr[k]`** — rounds out the assoc-array story (paired with the
   existing for-in / `in`).
7. **`split` / `sub` / `gsub` / `match` / `sprintf`** — the string-builtin
   family; larger, lower priority for the DSL's data-pipeline focus.

Deferred by design (the DSL's model diverges here): `RS`/multi-char record
separators, `getline` (the `over`/multi-pass readers subsume most uses),
multi-file `ARGV`/`FILENAME`/`FNR`, C-style `for(;;)` / `do-while` (the `while`
runtime + for-in cover the loop needs).
