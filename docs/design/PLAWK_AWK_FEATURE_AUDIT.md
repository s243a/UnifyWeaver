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
| `printf` | ✅ | standard `%[flags][width][.precision][length]conv`: integers `d`/`i`/`x`/`X`/`o`/`u` + `c` (code point), floats `f`/`g`/`e`/`F`/`G`/`E`, strings `%s`; flags `-+ 0#`, width, precision. Field-slice `%s` takes width but not precision (non-terminated buffer); `%c` needs a numeric arg |
| var assignment, `+=`, `++`, `//` | ✅ | indexed native scalar slots |
| `if` / `else` (chains) | ✅ | field/pattern guards (`$1 > 2`, `$0 ~ /re/`), scalar-variable conditions (`if (i > 2)`, `if (i < n && j > 0)`) in rule bodies, loops, **and `END`**, **and string-equality guards on a string scalar** (`if (s == "text")` / `!=`, lowered as interned atom-id comparison — single `==`/`!=`, not combined with `&&`/`||`) |
| `for (k in arr)` | ✅ | assoc for-in (rule body + END) |
| `while (COND)` | ✅ | runtime landed (loop-header phis); condition is scalar comparisons (`VAR CMP int`/`VAR`) combined with `&&`/`||` — `PLAWK_CONTROL_FLOW_PLAN.md` PR 2–3. `break`/`continue` deferred (PR 3b) |
| `do { } while (COND)` | ✅ | runtime landed; body runs at least once; same general condition |
| `next` | ✅ | structural (guarded clause per rule) |
| `break` (rule-level stream break) | ✅ | non-standard awk extension; stops the record stream |
| `break` / `continue` (loop-local) | ✅ | `while` **and** `do-while`: `break` leaves the loop, `continue` re-tests (SSA merge phis at the loop exit / head / body-done). Works nested (inner break targets the innermost loop) — `PLAWK_CONTROL_FLOW_PLAN.md` §3b |
| `if` with a plain (non-accumulator) body | ✅ | `{ if (c) { print $1 } }` compiles and runs (fixed by the while-runtime body-print enablers); if/else with plain bodies too |
| regex in `if` (`if ($0 ~ /re/)`) | ✅ | `if ($0 ~ /re/) { … }` and `!~` compile and run — guards a plain body |
| brace-less `if`/loop body | ✅ | `if (c) print`, `while (c) x++`, `do stmt while (c)`, braceless else-if chains — a body is a braced block or one statement |
| field assignment (`$2 = expr`) | ❌ | rebuilding `$0` from mutated fields not wired |
| C-style `for (;;)` | ❌ | |
| `delete arr[k]` | ◐ | `delete arr[$k]` removes the entry keyed by a field (parity with `arr[$k]++`); backward-shift deletion in the runtime (later colliding keys stay reachable), missing key is a no-op. String-literal / var keys are a follow-on |
| `exit [n]` | ✅ | stops the record loop, runs END, returns N (default 0); `exit` in a rule body, an `if`/`else` branch, or a loop (propagates past the loop) — scalar state at the exit point flows into END |
| `delete arr[k]` | ❌ | |
| `getline` | ❌ | the multi-pass / `over` readers cover much of its use |

## Functions

| Feature | Status | Notes |
|---|---|---|
| user functions (`function f(a) { return … }`) | ✅ | compile to Prolog clauses; work in accumulator / typed-field (`BINFMT`) contexts **and text mode** — `print f($1)` auto-coerces the field (awk semantics). Optional numeric arg typing (`function f(num x)`) skips the coercion (typed-fast path). |
| string: `length` `substr` `index` `tolower` `toupper` | ✅ | native, allocation-free |
| string: `split` `sub` `gsub` `match` `sprintf` | ◐ | **`sprintf` landed** — `x = sprintf("fmt", args)` formats into a string scalar, reusing the printf format engine (same conversions/flags/width/precision) + `snprintf`→intern. Numeric convs need a numeric arg (`$1+0`/`NR`), as printf. `split`/`sub`/`gsub`/`match` still ❌ |
| numeric: `int` | ✅ | `sin`/`cos`/`sqrt`/`rand`/… ❌ (f64 machinery exists) |
| Prolog foreign calls (`pred($1)`, `float(pred(…))`) | ✅ | plawk-specific, beyond AWK |

## Variables & operators

| Feature | Status | Notes |
|---|---|---|
| `$0` `$N` `NR` `NF` `FS` `OFS` | ✅ | |
| `FNR` `FILENAME` `ARGV` `ARGC` `RS` `ORS` `SUBSEP` `RSTART` `RLENGTH` | ❌ | single newline-delimited record model |
| arithmetic `+ - * / % //` | ✅ | i64, awk precedence, safe div/mod |
| comparison, `~`/`!~` | ✅ | |
| ternary `?:` | ✅ | `COND ? A : B` in print / printf args **and scalar assignment** (`x = $1 > $2 ? $1 : $2`) — numeric comparison condition, numeric branches (fields, `NR`/`NF`, int literals, i64 arithmetic); lowered to an LLVM `select`. Scalar-var operands, string branches, and `&&`/`||` conditions are follow-ons |
| string concatenation (juxtaposition `$1 $2`) | ✅ | `print` context **and** assignment: `print $1 $2`; `x = $1 $2` / `x = "id:" $1` build a **string-valued scalar** (an interned atom id in an i64 slot, resolved to text on read/print), **incl. accumulation `x = x $1`** (a string-scalar read as a concat operand — resolved to text and re-interned). Arithmetic binds tighter, comma still splits |
| exponentiation `^` / `**` | ❌ | |

## Arrays

| Feature | Status | Notes |
|---|---|---|
| assoc arrays `arr[k]` | ✅ | native assoc tables |
| `k in arr` (membership / for-in) | ✅ | |
| multi-dim `arr[i,j]` (SUBSEP) | ❌ | |
| `delete` | ◐ | `delete arr[$k]` (field key) removes an entry; backward-shift runtime delete. String-literal / var keys pending |

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
   **do-while break/continue and nested loops also LANDED** (`while` inside
   `while`/`do-while`, inner break targets the innermost loop). Remaining: loops
   inside a multi-pass `pass { }` block (PR 4).
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
3. **`printf` format coverage — LANDED.** The rewriter now parses the standard
   conversion prefix `%[flags][width][.precision][length]<conv>` and rewrites to
   a C printf spec driven by each argument's inferred kind: integer args (i64)
   take `d`/`i`/`x`/`X`/`o`/`u` (with the `l` length modifier) and `c` (code
   point → character); float args (f64) take `f`/`g`/`e`/`F`/`G`/`E`; string
   args take `%s` with flags/width/precision. Flags `-+ 0#`, width, and
   precision are honoured. A record-field `%s` (a non-null-terminated slice)
   takes flags/width and rides `.*` for its length, so a user precision on a
   field is a clean compile error (buffer safety) — a follow-on. `%c` needs a
   numeric arg (a string arg's first-char form is a follow-on). Tests:
   `tests/test_plawk_printf.pl`. *Still open (follow-on):* `%c` on a string
   (first char), precision on a field slice, `*`-width (width from an arg), and
   `printf` in a `BEGIN`/`END` block (a separate driver gap).
4. **`exit [n]` — LANDED.** A rule-level `exit` / `exit N` stops the record loop,
   runs END, and returns N (default 0). It reuses the rule-level stream-break
   path (`break_close_stream` → END → `ret`), adding an exit code stored in
   `@plawk_exit_code` and read at the final `ret`. `exit` is modelled as a
   terminal control (`terminal_exit`, like `terminal_break`) at rule-body top
   level; inside an `if`/`else` branch or a loop it flows via the `branch_exit`
   exit + `plawk_branch_to_done_ir` (branch to `break_close_stream`). Unlike
   `break`, an `exit` inside a loop is NOT consumed by the loop — it always ends
   the whole program (propagates past any enclosing loop). Scalar state at the
   exit point merges into END through the break-close phi. Tests:
   `tests/test_plawk_exit.pl`. *Still open (follow-on):* `exit` inside a `BEGIN`
   or `END` block, and non-constant exit codes (`exit code_var`).
5. **Ternary `?:` — LANDED (print / printf, numeric).** `COND ? A : B` in a
   print field or printf argument: the condition is a numeric comparison
   `L <op> R` and both branches are numeric (fields, `NR`/`NF`, integer
   literals, i64 arithmetic). Lowered to an LLVM `select` — both branches are
   evaluated (no side effects in an i64 expression), so it composes straight-line
   anywhere an i64 value is used (`plawk_i64_expr_ir(ternary(...))`), and the
   `%s`/`%d`/arith paths are unaffected. Tests: `tests/test_plawk_ternary.pl`.
   *Still open (follow-on):* assignment-context ternary (`x = c ? a : b`, needs
   scalar-var operands + the assignment RHS grammar), string-valued branches,
   and boolean-combination conditions (`a && b ? ...`).
   (String concatenation landed in both contexts — `print $1 $2` earlier, and
   **assignment concat `x = $1 $2` via string-valued scalars** now: a string
   scalar's slot holds an interned atom id (an i64, so it reuses the whole
   SSA/phi scalar machinery), the RHS is built into a buffer and interned at
   assignment, and a read/print resolves the id to text — id 0 is the unset
   sentinel, printed as empty. Numeric and string scalars coexist in one
   program. Tests: `tests/test_plawk_strscalar.pl`. *Still open (follow-on):*
   a scalar-var concat operand (`x = x $1` accumulation), string scalars in an
   `if`/loop body, and string comparison/guards on a string scalar.)
6. **`delete arr[k]` — LANDED (field key).** `delete arr[$k]` removes the entry
   keyed by field k, matching the counted inc `arr[$k]++`. The runtime primitive
   `@wam_assoc_i64_delete` does **backward-shift deletion** on the linear-probing
   table, so later keys that collided into the same cluster stay reachable and
   `get`/`inc`/`set` need no change (the no-gap invariant they rely on is
   preserved); a missing key is a no-op. Wired through the assoc for-in driver
   (build → delete → for-in report). Tests: `tests/test_plawk_delete.pl` and
   `tests/core/test_wam_llvm_assoc_i64_runtime.pl` (backward-shift unit test).
   *Still open (follow-on):* string-literal / variable keys (`delete arr["k"]`,
   `delete arr[k]`), and `delete` in the scalar/mixed sequence-walker path.
7. **`split` / `sub` / `gsub` / `match` / `sprintf`** — the string-builtin
   family; larger, lower priority for the DSL's data-pipeline focus.

Deferred by design (the DSL's model diverges here): `RS`/multi-char record
separators, `getline` (the `over`/multi-pass readers subsume most uses),
multi-file `ARGV`/`FILENAME`/`FNR`, C-style `for(;;)` / `do-while` (the `while`
runtime + for-in cover the loop needs).
