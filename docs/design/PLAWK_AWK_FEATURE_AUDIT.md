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
| expression pattern (bare `NR==1`) | ◐ | comparison guards yes; arbitrary expr no. **`NR` is a special in `if`/`while` condition guards** (`if (NR == 2)`, `if (NR >= 2 && NR <= 3)`) — the record counter, not a phantom scalar slot (a prior bug read it as an uninitialised var → always 0). Fixes the record-gated idioms (`if (NR == 1) { first = $1 }`, conditional accumulation). **`NF` is a guard special too** (`if (NF > 3)`, `if (NF >= 2 && NR == 1)`) — the field count of the current record, computed from `%line` via the field-count runtime with the active `FS` threaded into the condition lowering; works in `if`/`while`/`do-while` guards. `NF` in an END condition is a clean not-yet (no current record) |
| range pattern (`/a/,/b/`) | ◐ | `/start/,/end/ { … }` fires for records from a /start/ match through a /end/ match (inclusive), via a per-rule i1 latch global; re-arms for later ranges, runs to EOF if unterminated. Regex endpoints (v1); general-pattern endpoints (`NR==1,NR==3`) are a follow-on |

## Actions & control flow

| Feature | Status | Notes |
|---|---|---|
| `print` (fields, literals, `NR`/`NF`, `length`/`substr`/`index`/`tolower`/`toupper`, arithmetic, **concatenation**) | ✅ | constant fields (`print 1`, `print "x"`) + juxtaposition concat (`print $1 $2`) landed. **Per-record print from an assoc-mutating rule (no `END` needed)**: `{ c[$1]++; print $1, c[$1] }` (a running count) and `{ split($0,a,","); print a[1] }` (print an element) compile — previously an assoc program had to be END-tethered (a rule that mutated a table *and* printed per-record was rejected unless the program also had an `END { print arr[…] }`). Body-print fields: `$N` (N>0), `$0` (the whole record), `arr[N]` (integer element — split/positional tables keyed by position; str values resolve to text, i64 counters print numerically), `arr[$k]` (field-keyed lookup), a **string literal** (its bytes ride the assoc rule chain's global channel; printed via `%s`, so a `%` in the literal is verbatim), and scalar reads — in a comma-separated print list **or glued by juxtaposition-concat** (`print "x=" c[$1]` → `x=5`; the concat parts print adjacently with no separator, reusing the per-field emitters) |
| `printf` | ✅ | standard `%[flags][width][.precision][length]conv`: integers `d`/`i`/`x`/`X`/`o`/`u` + `c` (code point), floats `f`/`g`/`e`/`F`/`G`/`E`, strings `%s`; flags `-+ 0#`, width, precision. Field-slice `%s` takes width but not precision (non-terminated buffer); `%c` needs a numeric arg |
| var assignment, `+=`, `++`, `//` | ✅ | indexed native scalar slots |
| `if` / `else` (chains) | ✅ | field/pattern guards (`$1 > 2`, `$0 ~ /re/`), scalar-variable conditions (`if (i > 2)`, `if (i < n && j > 0)`) in rule bodies, loops, **and `END`**, **and string guards on a string scalar** (`if (s == "text")` — `==`/`!=` via interned atom-id compare, `<`/`<=`/`>`/`>=` via `strcmp` — a single comparison, not combined with `&&`/`||`). **POSIX strnum duality is partially landed**: a scalar copied from a field (`x = $1`) that is only printed or compared against another such strnum or a string literal is typed `scalar_strnum` and compares **numerically or lexically by its runtime content** (`x=$1;y=$2; if (x>y)` gives `10 9`→numeric, `10 9x`→lexical, via `@wam_strnum_cmp`). Comparison against an **integer literal** works too (`if (x > 3)`, `if (n == 5)` via `@wam_strnum_cmp_int` — `"abc" > 3` lexically true, `"5x" == 5` false), and a strnum read in **pure arithmetic** is coerced to a number (`y = x + 1` — i64 via the same field parser as before, so byte-identical; f64 via `strtod`), letting a field copy be used in both arithmetic and a string/number comparison. A plain copy `z = x` **propagates** strnum-ness (transitively through chains). Names read in a string concat / call / ternary / index still stay on the plain i64 path via a greatest-fixpoint read-use gate — no regressions. **Split-array elements compare too**: in an `END` for-in filter `for (k in a) { if (a[k] CMP …) … }`, a str-valued (split / `as assoc(str)`) element compares against **an integer** via `@wam_strnum_cmp_int` on its text (numeric when it looks like a number, else lexical) instead of a raw i64 icmp on its atom id — which had silently compared registry positions (a bug); and against **a string literal** — `==`/`!=` via canonical atom-id `icmp eq/ne` (the literal interned from a module constant), the **ordering** ops (`a[k] < "x"` etc.) via `strcmp` on the element text. A **float-literal** RHS (`a[k] < 3.5`) compares via `@wam_strnum_cmp_double` on a str element, or widens an i64 counter to `double` + `fcmp`. Counter tables (`arr[k]++`, a genuine i64) keep the plain icmp for integer RHS. **Non-field strnum sources landed**: a scalar from a command-line argument (`x = ARGV[N]`) or a getline read (`getline v < "file"`, `s = getline v < "file"`) is a strnum too — `if (x > 5)` dispatches through `@wam_strnum_cmp_int` on its text (previously it compared the interned atom id — every value looked "big", a bug). A candidate whose reads are unsupported (e.g. used in a concat) deactivates to a plain string scalar (its prior behaviour), so activation never regresses. Remaining: general rule-body `arr[N]` element reads (outside the compilable surface — only the for-in contexts read elements); nested-block field copies (blocked on a pre-existing nested-assignment control-flow limitation). See `PLAWK_STRNUM_DUALITY.md` for the representation and phased plan |
| `for (k in arr)` | ✅ | assoc for-in (rule body + END) |
| `while (COND)` | ✅ | runtime landed (loop-header phis); condition is scalar comparisons (`VAR CMP int`/`VAR`) combined with `&&`/`||` — `PLAWK_CONTROL_FLOW_PLAN.md` PR 2–3. `break`/`continue` deferred (PR 3b) |
| `do { } while (COND)` | ✅ | runtime landed; body runs at least once; same general condition |
| `next` | ✅ | structural (guarded clause per rule) |
| `break` (rule-level stream break) | ✅ | non-standard awk extension; stops the record stream |
| `break` / `continue` (loop-local) | ✅ | `while` **and** `do-while`: `break` leaves the loop, `continue` re-tests (SSA merge phis at the loop exit / head / body-done). Works nested (inner break targets the innermost loop) — `PLAWK_CONTROL_FLOW_PLAN.md` §3b |
| `if` with a plain (non-accumulator) body | ✅ | `{ if (c) { print $1 } }` compiles and runs (fixed by the while-runtime body-print enablers); if/else with plain bodies too |
| regex in `if` (`if ($0 ~ /re/)`) | ✅ | `if ($0 ~ /re/) { … }` and `!~` compile and run — guards a plain body |
| brace-less `if`/loop body | ✅ | `if (c) print`, `while (c) x++`, `do stmt while (c)`, braceless else-if chains — a body is a braced block or one statement |
| field assignment (`$2 = expr`) | ✅ | `Pattern { $N = expr; …; print $0 }` splits the record once into an editable field buffer (`%WamFieldBuf` — slices into the record text, no interning), mutates the slot in place (`@wam_fields_set`), and `print $0` joins the fields with OFS once (`@wam_fields_join`). O(record), not O(assignments × record), and zero interning on this path. RHS is a string/integer literal or another field `$M` (read from the current buffer); chained assignments and a pattern guard work; setting a field past the end pads with empties. **v1 scope:** an explicit `FS` (single char **or a multi-char regex** — the buffer splits via `@wam_fields_new_re`), a single rule with no `END`, and the print-the-record idiom. Follow-ons: default (space) `FS`, in-body field reads after assignment, `$0 = expr`, multi-rule/`END` programs, concat/arithmetic RHS. See `PLAWK_FIELD_BUFFER.md` for the field-buffer representation and its scaling note (flat array now; map / hashtable for wide or sparse field spaces) |
| C-style `for (INIT; COND; UPDATE)` | ◐ | **landed for the three-part form.** A parser normalisation pass (`plawk_normalise_c_for`) desugars `for (INIT; COND; UPDATE) BODY` to `INIT; while (COND) { BODY; UPDATE }`, reusing the while runtime with no codegen changes. INIT/UPDATE are simple statements — an assignment (`i = 0`), an increment (`i++`), or a compound-assign (`i += 3`); COND is the general `while` condition (`i < n`, `&&`/`||`). `break` exits the loop, and loops nest (`for (i…) for (j…)`). Tried after `for (k in arr)` (which commits on `in`); the `;` separators distinguish them. Follow-ons: **`continue`** targeting the for (it would skip the appended UPDATE and infinite-loop, so a for whose own body has a `continue` is left un-desugared and **cleanly rejected** by codegen — a `continue` inside a nested loop in the body is fine); empty parts (`for (;;)`, `for (; c; )`) — v1 requires all three |
| `delete arr[k]` | ◐ | `delete arr[$k]` removes the entry keyed by a field (parity with `arr[$k]++`); backward-shift deletion in the runtime (later colliding keys stay reachable), missing key is a no-op. String-literal / var keys are a follow-on |
| `exit [n]` | ✅ | stops the record loop, runs END, returns N (default 0); `exit` in a rule body, an `if`/`else` branch, or a loop (propagates past the loop) — scalar state at the exit point flows into END |
| `delete arr[k]` | ❌ | |
| `getline` | ◐ | **`getline var < "file"` landed** (phase 1): reads the next line of a file into a scalar, returning 1/0/-1. `status = getline var < "file"` captures the return (dual-slot: `var` string, `status` i64); a bare `getline var < "file"` discards it. Handles are keyed by **filename** in a process-wide registry (`@wam_getline_file`), so every site reading the same file shares one advancing handle (as in awk); EOF preserves `var`. The **canonical in-condition idiom** works: `while ((getline var < "file") > 0) BODY` (normalised at parse to a priming read + a `while (status > 0)` loop whose body re-reads — reusing the phase-1 machinery, no new codegen). Follow-ons: plain `getline` / `getline < file` into `$0` (updates `NF`), `cmd | getline`, `getline` in BEGIN/END, and `var` as a strnum source. The multi-pass / `over` readers still cover the bulk-scan use |

## Functions

| Feature | Status | Notes |
|---|---|---|
| user functions (`function f(a) { return … }`) | ✅ | compile to Prolog clauses; work in accumulator / typed-field (`BINFMT`) contexts **and text mode** — `print f($1)` auto-coerces the field (awk semantics). Optional numeric arg typing (`function f(num x)`) skips the coercion (typed-fast path). |
| string: `length` `substr` `index` `tolower` `toupper` | ✅ | native, allocation-free |
| string: `split` `sub` `gsub` `match` `sprintf` | ◐ | **`sprintf` + `split` landed.** `sprintf`: `x = sprintf("fmt", args)` → string scalar (printf engine + `snprintf`→intern). `split`: `split($N, arr, "sep")` populates a positional string array `arr` (keys 1..n) via `@wam_str_split_into` (clears + repopulates; empty pieces kept), read via for-in (`for (k in arr) print k, arr[k]`). the separator may be a **single char (literal byte) or a multi-char POSIX ERE regex** — a per-split-site pattern/cache (`@wam_str_split_into_re`), independent of `FS`. Follow-ons: a return count, a default-FS separator, and `arr[N]` constant-index read. **`sub`/`gsub`/`match` landed.** `match(SRC, /re/)` (SRC = `$0` or `$N`) returns the 1-based match position (0 if none) and sets **`RSTART`/`RLENGTH`** (`RLENGTH` = -1 on a miss) via `@wam_regex_match` — an i64 expression usable in `print`. `sub`/`gsub`: `Pattern { sub/gsub(/re/, "repl"); … ; print $0 }` rewrites `$0` (first match / all matches) via `@wam_regex_gsub`, with an unescaped `&` in the replacement expanding to the matched text and a per-site compiled ERE; a string-literal pattern works too, and chained substitutions compose. **Scope:** `match` works in `print` **and scalar-assignment** contexts (`n = match($0, /re/)`, `x = RSTART`); and `RSTART`/`RLENGTH` read as i64 specials in `if`/`while` comparison guards (`if (RSTART > 0)`, `if (RLENGTH >= 3)` — the special is loaded, not a phantom slot). `sub`/`gsub` target `$0` (the stream-editor idiom), a **string-scalar variable** (`gsub(/re/, "repl", s)` — substitutes over the slot's current interned value and re-interns), **or a field `$N`** (`gsub(/re/, "repl", $2)` — rewrites the field via the field buffer and rebuilds `$0` with OFS; composes with `$N = expr` assignments in the same record). **Count capture** landed too: `n = gsub(/re/, "repl", var)` (and `sub`) is a dual-slot write — `var` is substituted in place (a string scalar) and `n` receives the substitution count (an i64 slot, loaded from the shared `@plawk_gsub_count`), usable in later guards (`if (n > 1)`). All compose with chaining, `&`, `END`, and string-equality guards. Follow-ons: `match` as a bare guard condition (`if (match(...))` — use `if ($0 ~ /re/)` or capture-then-test), and count capture into a field target (`n = gsub(..., $2)`) |
| numeric: `int` `sqrt` `sin` `cos` `exp` `log` `atan2` `rand` `srand` | ✅ | `int` (truncation) plus the **libm math builtins**: `sqrt`/`sin`/`cos`/`exp`/`log` (unary), `atan2` (binary), and **`rand`** (zero-arg). Each parses to `math_call(Fn, Args)` and lowers through the double expression path — the argument is evaluated in f64 (an integer field promotes via `sitofp`, so `sqrt($1)` works), the libm function is called, and the result prints via `%g` (`sqrt(16)` → `4`, `sqrt(2)` → `1.41421`). `log` is the natural logarithm. `rand()` returns a double in `[0, 1)` (libm `drand48`). Arguments may be full arithmetic expressions (`sqrt($1 * 4)`, `sqrt(2^4)`), and a call composes inside larger arithmetic (`2 * sqrt(9)`, `rand() * 100`) and `printf`. **`srand(N)` / `srand()`** are rule-body seeding statements (`@srand48`): `srand(N)` makes the sequence reproducible (drand48 is POSIX-specified, so `srand(1)` → `rand()` = `0.0416303`), `srand()` seeds from the wall clock. A name is only a builtin when directly followed by `(`, so `sine`/`randy` stay identifiers. Same supported surface as exponentiation (print / printf / arithmetic value positions); guards and double-scalar-reread are the same pre-existing limitations. Follow-ons: `srand` in a `BEGIN` block (BEGIN hosts no non-print action yet), a variable/expression seed (`srand($1)`), and `srand`'s awk return value (the previous seed) |
| Prolog foreign calls (`pred($1)`, `float(pred(…))`) | ✅ | plawk-specific, beyond AWK |

## Variables & operators

| Feature | Status | Notes |
|---|---|---|
| `$0` `$N` `NR` `NF` `FS` `OFS` | ✅ | `FS` accepts a **multi-char / regex** value (a POSIX ERE, awk semantics): a length-≥2 `BEGIN { FS = "…" }` compiles to a reserved sentinel separator byte that the field runtime dispatches to `@wam_fs_regex_field_slice_value` / `_count_value` (lazily `regcomp`ed, one FS per program). Because the numeric / `length` / `eq` / `cmp` field projectors all delegate to the core slice, the regex FS reaches `$N` reads, `NF`, guards, and concat uniformly; `$0` and EOF are unaffected. A single-char `FS` stays a literal byte (awk treats a one-char FS literally). **Field assignment splits on a regex FS too** (via `@wam_fields_new_re`), and `split()` takes its own multi-char/regex separator. **`OFS` is now multi-char / empty**: it is a list of bytes (default `[32]`), emitted between a print's fields as one `putchar` per byte, so `BEGIN{OFS=", "}` prints `a, b` and `OFS=""` prints fields adjacent; it composes with `ORS`. (The field-assignment `$N=expr; print $0` rebuild path also honours a multi-char / empty OFS now: it joins via `@wam_fields_join_str`, which takes a separator pointer + length — the OFS is emitted as a private `@.plawk_field_ofs` constant.) Remaining: regex FS inside the multipass / dyncall record-view drivers (which read a non-`%line` record). |
| `ARGV[N]` `ARGC` | ◐ | **command-line arguments landed** (rule bodies): the plawk driver is a native binary, so `ARGV`/`ARGC` read straight from the process `argv` via `/proc/self/cmdline` (`@wam_argc`, `@wam_argv_get`) — `ARGV[0]` the program, `ARGV[1]` the first argument (the input path; `-` = stdin), `ARGV[2..]` extras, matching awk. `ARGC` is an i64 usable anywhere an i64 special is (`print ARGC`, `if (ARGC > 1)`, `n = ARGC - 1`, `c = ARGC`, concat). `ARGV[N]` (a non-negative integer-literal index) is a string scalar: `print ARGV[N]`, `x = ARGV[N]` (then prints / string-compares); an out-of-range index is `""`. Follow-ons: `ARGV`/`ARGC` in **BEGIN**/**END** blocks (their print/action pipelines are separate — BEGIN print is string-literal-only), a direct `ARGV[N]` comparison operand (`if (ARGV[2]=="x")` — use assign-then-compare), `ARGV[N]` in juxtaposition-concat (parses as an assoc read), and a variable/expression index. Linux-specific (`/proc/self/cmdline`) |
| `FNR` `FILENAME` | ◐ | **landed as single-input record-identity specials.** plawk streams one input, so there is no per-file counter reset: **`FNR` is an exact alias for `NR`** (the record number) — it reads anywhere `NR` does (print, arithmetic, `if`/`while` guards, `END`), because it parses to `special('NR')` and reuses the NR codegen unchanged. **`FILENAME` is the input path** — the native driver takes the input as `ARGV[1]` (`-` denotes stdin), so `FILENAME` parses to `argv_at(1)` and reuses the `ARGV[N]` string machinery: a string scalar for `print FILENAME`, `x = FILENAME`, and string-equality via assign-then-compare (`f = FILENAME; if (f == "path")`). A trailing identifier-boundary keeps `FILENAMEX` / `FNRX` ordinary identifiers. Follow-ons: `FILENAME` in an `END` block (same limitation as `ARGV[N]` in `END`); a direct `FILENAME`/`ARGV[N]` comparison operand |
| `ORS` | ✅ | **output record separator, any string.** `BEGIN { ORS = "…" }` sets what is printed after each `print` (default newline) — single-char, multi-char (`" | "`), or empty. It is emitted through a runtime pointer global `@plawk_ors_ptr` (statically initialised to the ORS constant) that every print terminator loads and prints via `printf("%s", ptr)`, so the ORS may be any length and any content — **a literal `%` is data, not a format directive** — with no change to the terminator emitters and no per-emitter length. Per-record **and** `END` prints honour it, and it composes with (multi-char) `OFS` (`OFS` separates a print's fields, `ORS` terminates the record — `BEGIN{OFS="-";ORS=" | "} {print $1,$2}` → `a-b | c-d | `) |
| `RS` | ✅ | **input record separator: literal char, POSIX-ERE regex (gawk), or paragraph mode.** `BEGIN { RS = "…" }` sets what the readers split on. A one-character `RS` is a literal byte (`";"`); a **nonempty multi-character `RS` is a POSIX extended regular expression** (gawk's regex-RS extension), so variable-width separators, alternation, groups, and character classes work — `BEGIN{RS="[[:digit:]]+"}` over `foo12bar345baz` yields `foo`/`bar`/`baz`. The runtime dispatches on `@wam_rs_mode` (0 literal / 1 paragraph / 2 regex): a regex `RS` is `regcomp`ed once at startup, and each reader scans the buffered record with `regexec`, trims the leftmost **nonempty** match, and replays the bytes after it. The exact matched separator is exposed through read-only **`RT`** (including in `END`; cleared on an unterminated record). Both readers (`@wam_stream_read_line_transient_value` and its persistent `getline` sibling) share the behaviour, and it composes with a regex `FS`. The trailing-CR strip (CRLF) is gated on `RS` being exactly a one-byte newline, so a custom `RS` keeps `\r` as data. **An empty `RS` is paragraph mode:** each reader skips leading blank lines, treats two adjacent newlines as the boundary, collapses blank-line runs, and strips a lone trailing newline at EOF — a paragraph may span lines, and the default whitespace FS splits `$N` across the embedded newlines. An invalid ERE fails at runtime (exit 2, focused diagnostic). Caveat: POSIX `regexec` has no partial-match interface, so a match that begins but stays incomplete at a read-buffer boundary follows the documented once-per-block scan rather than once-per-byte |
| `SUBSEP` | ◐ | **multi-dimensional subscripts landed** for the two-field key. `arr[$i,$j]` joins the two field values with SUBSEP into one string key (awk `arr[i SUBSEP j]`): the write counter `arr[$i,$j]++`, the element read `arr[$i,$j]`, and for-in over the joined keys. The join runs in the runtime helper `@wam_intern_subsep_key2` (slice both fields → join with the SUBSEP bytes → intern to one atom id — the same key both write and read build), keyed off `@wam_subsep_ptr`/`@wam_subsep_len` (default the `0x1C` byte, awk's SUBSEP). `BEGIN { SUBSEP = "…" }` overrides it (any string, incl. empty = no separator) via the RS-style begin-globals + begin-instructions seam. Follow-ons — **clean rejection, never miscompiled**: three-plus subscripts (`arr[$i,$j,$k]`), non-field subscripts (`arr[$i,"x"]`), and multi-dim on the other write shapes (`=`, `+=`, `delete`) / binfmt records / `(i,j) in arr` membership |
| `ENVIRON["NAME"]` | ◐ | **read-only env lookup landed** (getenv via `@wam_environ_get`): `x = ENVIRON["NAME"]` (a string scalar — so it prints and string-compares) and `print ENVIRON["NAME"]`; an unset variable is `""` (awk semantics). v1 takes a string-literal key. Follow-ons: a non-literal key, `for (k in ENVIRON)`, and ENVIRON in juxtaposition-concat |
| `RSTART` `RLENGTH` | ✅ | set by `match(SRC, /re/)`, readable as i64 specials in `print` (backed by `@plawk_rstart`/`@plawk_rlength`) |
| arithmetic `+ - * / % //` | ✅ | i64, awk precedence, safe div/mod |
| comparison, `~`/`!~` | ✅ | |
| ternary `?:` | ✅ | `COND ? A : B` in print / printf args **and scalar assignment** (`x = $1 > $2 ? $1 : $2`) — numeric comparison condition, numeric branches (fields, `NR`/`NF`, int literals, i64 arithmetic); lowered to an LLVM `select`. Scalar-var operands, string branches, and `&&`/`||` conditions are follow-ons |
| string concatenation (juxtaposition `$1 $2`) | ✅ | `print` context **and** assignment: `print $1 $2`; `x = $1 $2` / `x = "id:" $1` build a **string-valued scalar** (an interned atom id in an i64 slot, resolved to text on read/print), **incl. accumulation `x = x $1`** (a string-scalar read as a concat operand — resolved to text and re-interned). Arithmetic binds tighter, comma still splits |
| exponentiation `^` / `**` | ◐ | **landed for expression / output contexts.** `^` and `**` are the same operator — a new precedence level that binds tighter than `* / %` and is **right-associative** (`2^3^2` == `2^(3^2)` == 512). As in awk it is **always computed in floating point** (via libm `@pow`), so the result keeps its fraction and prints through the double `%g` path: `2^10` → `1024`, `4^0.5` → `2`, `2^0.5` → `1.41421`. Operands may be integers, fields, or floats (`$1^2`, `2.5^2`), and it composes inside larger arithmetic (`$1^2 + 1`, `10 - 2^3`), juxtaposition concat (`print "r=" 2^3`), and `printf` `%g`/`%e`/`%f`. Follow-ons — each a **pre-existing limitation the operator inherits, not one it introduces**: pow in an `if`/`while` guard (guards accept no arithmetic operand at all — `$1*2 > 5` is rejected identically); pow fed to a `%d`/`%i` conversion (no double→i64 truncation); pow assigned to a scalar then reread (same as `x = 2.5; print x`, a double-scalar-shape limit); a unary-minus exponent (`2^-1`, since unary minus is not a factor in the grammar). Rejections are clean parse/surface errors, never a miscompile |

## Arrays

| Feature | Status | Notes |
|---|---|---|
| assoc arrays `arr[k]` | ✅ | native assoc tables |
| `k in arr` (membership / for-in) | ✅ | |
| multi-dim `arr[i,j]` (SUBSEP) | ◐ | `arr[$i,$j]` (two field subscripts joined by SUBSEP): counter `arr[$i,$j]++`, read `arr[$i,$j]`, for-in over joined keys; `BEGIN{SUBSEP="…"}` override. Three-plus / non-field subscripts a follow-on — see the `SUBSEP` row |
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
multi-file `FILENAME`/`FNR` (`ARGV[N]`/`ARGC` themselves are landed for rule
bodies — see the table), C-style `for(;;)` / `do-while` (the `while`
runtime + for-in cover the loop needs).
