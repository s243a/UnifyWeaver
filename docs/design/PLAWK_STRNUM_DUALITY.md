<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk strnum: string/number scalar duality

**Status**: **design note + steps 1–4 landed (scoped).**
- **Step 1** (runtime): `@wam_looks_numeric` and `@wam_strnum_cmp` are
  implemented in `src/unifyweaver/targets/wam_llvm_target.pl` and unit-tested in
  `tests/core/test_wam_llvm_assoc_i64_runtime.pl` (the recogniser over
  numeric/non-numeric/blank-padded/empty inputs, and the comparison over the
  POSIX kind table incl. the `10 9` vs `10 9x` divergence).
- **Step 2** (origin analysis): `plawk_scalar_strnum_names/2` in
  `examples/plawk/codegen/plawk_native_codegen.pl` decides which names are
  strnum by provenance (assigned only from a field copy, never from a literal /
  arithmetic / concat / string builtin).
- **Step 3** (activation, scoped): the type inferencer now **produces**
  `scalar_strnum` (precedence: double > strnum > string > counter — the sets are
  disjoint), a strnum assignment **interns the field's raw bytes** into the slot
  (retaining the text), `print` and END-print **resolve the id to text**, and a
  **strnum-vs-strnum** numeric comparison in an `if`/`while` guard dispatches to
  `@wam_strnum_cmp` (the `10 9` vs `10 9x` fix); a strnum-vs-string-literal
  comparison is handled by the existing string-guard clauses (lexical, which is
  POSIX-correct). All wired in `plawk_native_codegen.pl`; tested in
  `tests/test_plawk_strnum.pl`.

  **Step 3b** extends comparison to an **integer literal** (`if (x > 3)`,
  `if (n == 5)`) via a second runtime primitive `@wam_strnum_cmp_int` (numeric
  when the field looks numeric, else lexical against the integer formatted as a
  decimal string — so `"abc" > 3` is lexically true and `"5x" == 5` is false).
  A dedicated `plawk_while_cond_build` clause dispatches it.

  **Step 3c** adds arithmetic **coercion**: a strnum read in a pure arithmetic
  expression (`y = x + 1`, `c += x`) is coerced to a number — to i64 via the
  *same* field parser the pre-strnum path used (so the result is byte-identical,
  incl. the not-a-number → 0 default), or to double via `strtod` in an f64
  context. The read tag is `ssa_strnum`; the i64/f64 expression leaves coerce it.
  This lets a field copy used in **both** arithmetic and a string/number
  comparison (`x = $1; y = x + 1; if (x == "42") …`) be a full strnum.

- **Step 4** (source propagation): a plain copy `z = x` now **propagates**
  strnum-ness — `z` becomes a strnum holding `x`'s atom id (awk copies the dual
  type), transitively through chains (`w = z = x`). The origin analysis grows the
  set through copy edges, then a greatest-fixpoint (`plawk_strnum_stabilize/4`)
  drops any name whose reads are unsupported **or** whose only source is a copy
  from a now-dropped name, so copy chains collapse correctly. (The design doc's
  other step-4 sources — `getline`, `ARGV`/`ENVIRON`, and `split()` elements —
  are **blocked**: `getline`/`ARGV`/`ENVIRON` are not implemented in plawk, and
  `split()` writes to an assoc-array, a separate subsystem from scalar slots.
  Nested-block field copies are also excluded: plawk does not yet propagate a
  value assigned inside an `if`/loop body to a later statement — a pre-existing
  control-flow limitation, independent of strnum — so activating strnum there
  would ride on an unsupported shape.)

  **Scope gate (honest scoping, absorbs step 5 for these forms):** a name is a
  `scalar_strnum` **only if** it has a valid source (a field copy, or a copy from
  another strnum) **and every read of it is a supported position** — a bare
  `print` field, a comparison against another strnum var / string literal /
  integer literal, or a **pure arithmetic expression** (coerced). A read in a
  string concat, a function/`dyncall` argument, a ternary, an index, an assoc
  key, etc. deactivates the name (it stays a plain i64 counter), so activation
  **never regresses** those programs. The analysis is a greatest-fixpoint
  because a var-vs-var comparison is only supported when both sides are strnum
  and a copy target only stays strnum while its source does; the set is shrunk
  until stable. The generic numeric-compare lowering additionally **fails
  closed** if a strnum operand ever reaches it in an unsupported form, so no path
  emits a raw `icmp` on an atom id.

**Not yet built**: comparison against a **non-integer / float literal** or a
**non-strnum var**; strnum reads inside **calls / ternary / index / concat**;
non-field sources that don't yet exist (`getline`, `ARGV`/`ENVIRON`) or live in
another subsystem (`split()` array elements); nested-block field copies (blocked
on the nested-assignment control-flow limitation). §4–§5 describe model aspects
that remain partial.

**Representation note**: the shipped `scalar_strnum` slot uses the §4b
interned-id representation (an i64 atom id, same width as a string scalar) rather
than the §4a wide `%WamStrnum` cell. This keeps the SSA phi model unchanged — a
strnum slot phis a single i64 like every other slot — and the looks-numeric
decision is recomputed at each comparison. The §4a cell stays the documented
option if per-comparison recomputation ever shows up in profiles.

## 1. What strnum is (POSIX awk)

In awk a value is not simply a string or a number — some values carry a **dual
type** the spec calls *numeric string* (strnum). A strnum is a string that also
remembers it *originated from input* and *looks like a number*; it is compared
**numerically** when both sides are numeric-looking, and **lexically**
otherwise. The values that are strnums:

- fields `$1..$NF` (and `$0`),
- the results of `split()`,
- `getline` inputs,
- `FS`-split pieces,
- `ARGV` / `ENVIRON` elements,
- command-line `var=value` assignments and `-v` assignments.

String **literals** in the program text are *never* strnums — `"10"` is always
a string. Values produced by arithmetic are always numbers. So the type of a
value depends on **where it came from**, and its comparison behaviour depends on
**its runtime content**:

```awk
$1 == 10        # $1 is a strnum: numeric compare if $1 looks numeric,
                #   else string compare against "10"
$1 == "10"      # "10" is a string literal, but $1 is a strnum, so the
                #   comparison is numeric iff $1 looks numeric  (POSIX: a
                #   strnum vs string comparison is numeric only when the
                #   strnum is numeric AND ... — see §5 for the exact table)
"10" == "10"    # both string literals -> string compare
x = $1; x == 10 # x inherits $1's strnum-ness through a plain copy
```

The classic surprises this produces:

```awk
echo "10 9" | awk '{ print ($1 > $2) }'    # 1  -> numeric (both look numeric)
echo "10 9x" | awk '{ print ($1 > $2) }'   # 0  -> string ("10" < "9x")
```

The "looks like a number" test is a full numeric-prefix recogniser applied to
the *entire* trimmed field (leading/trailing blanks allowed): optional sign,
digits, optional fraction, optional exponent, or `inf`/`nan` spellings — with
nothing left over.

## 2. plawk's current model — three disjoint static slot kinds

plawk types every scalar **statically, at compile time**, into one of three
disjoint slot kinds (`plawk_scalar_typed_slot/4` in
`examples/plawk/codegen/plawk_native_codegen.pl`):

| slot kind | LLVM repr | inferred when |
|---|---|---|
| `scalar_counter(N)` | `i64` | default (counters, integer arithmetic) |
| `scalar_double(N)`  | `double` (f64) | fixpoint: assigned a float leaf or reads a double |
| `scalar_string(N)`  | `i64` holding an **interned atom id** | assigned a string RHS (`x = $1 $2`, `x = "…"`, `sub/gsub`) |

Fields are read **per use site** as *either* an i64 (`field_i64`, via the
numeric field slice) *or* an interned string, and the choice is made
**statically** by the surrounding expression's shape. Comparisons likewise pick
their instruction at compile time:

- numeric guards → `icmp`/`fcmp` on the i64/double slot value;
- string-equality guards (`if (s == "text")`) → compare **atom ids** (interning
  is canonical, so equal strings share an id — see `plawk_if_cond_ir` ==/!=);
- string-ordering guards (`if (s < "text")`) → resolve the id to text and
  `strcmp`.

The pivotal property: **which comparison is emitted is fixed at compile time**
from the *statically inferred* slot type and the *syntactic* RHS. There is no
representation in which one value carries both a string identity and a numeric
interpretation and picks between them from its **runtime content**. That is
exactly what strnum requires.

### Where this already bites (and where it doesn't)

Today's model handles the common cases because plawk leans on syntactic
context:

- `$1 + 0`, `$1 * $2` → numeric field read (`field_i64`), correct.
- `x = $1 $2`, `if (name == "lit")` → string path, correct.
- `s = $1 ""` (the concat-with-empty idiom) → forces the string interpretation
  explicitly; this is the workaround the `gsub`-into-scalar tests use.

What it *cannot* express is a single value that compares numerically **or**
lexically depending on its bytes at runtime — the `10 9` vs `10 9x` example. A
bare `if ($1 == $2)` today resolves each side by its static slot type, not by
whether the fields look numeric, so it can silently pick the wrong comparison
for adversarial input. plawk's honest-scoping stance means such programs should
either compile to the POSIX-correct runtime dispatch (this project) or be
guarded/rejected — not silently mis-compare.

## 3. Why this is a type-model project, not a feature

strnum is cross-cutting because it changes the **representation of a value**,
which nearly every consumer touches:

1. **A value needs to carry both faces.** A strnum must hold (a) its string
   identity (interned atom id or slice) and (b) a decision about whether it is
   numeric, plus (c) its numeric value when it is. That is wider than any of the
   three current single-word slots.
2. **Provenance must be tracked, not just type.** "Is this a strnum?" is a
   property of *origin* (input/field/split/getline) that survives plain copies
   (`x = $1`) but is destroyed by arithmetic (`x = $1 + 0` → number) and never
   attaches to literals. The static type lattice has no notion of provenance.
3. **Comparison becomes runtime dispatch.** `==`, `!=`, `<`, `<=`, `>`, `>=`
   between two operands must, in the general case, branch at runtime on each
   operand's strnum-ness and numeric-ness to choose numeric vs lexical — a
   decision the current compile-time emitter makes once, statically.
4. **It touches every value source at once.** Fields, `split`, `getline`,
   `ARGV`/`ENVIRON`, command-line assignments all become strnum-producing;
   partial adoption (fields only) creates an inconsistent surface.
5. **It interacts with the SSA phi model.** Slots are joined by phi nodes; a
   strnum slot must phi a *wider* value (or a stable tagged representation)
   across blocks without splitting into parallel i64/double/string phis.

## 4. Proposed representation

Two workable representations, in increasing invasiveness:

### 4a. Tagged strnum cell (recommended)

Introduce a runtime value cell that a **new fourth slot kind**
`scalar_strnum(N)` carries:

```llvm
%WamStrnum = type { i64, double, i8 }   ; atom-id, numeric value, flags
; flags bit0 = came-from-input (is a strnum candidate)
; flags bit1 = looks-numeric (the recogniser said yes)
```

- Field reads, `split`, `getline` etc. produce a `%WamStrnum`: intern (or slice)
  the text into the id, run the numeric recogniser once to set `flags`/`numeric`.
- A plain copy propagates the cell unchanged (strnum-ness survives).
- Arithmetic reads `.numeric` (or coerces the string) and yields a plain number
  — strnum-ness lost, as POSIX requires.
- String context reads `.id`.
- Comparison calls a single runtime primitive
  `@wam_strnum_cmp(%WamStrnum a, %WamStrnum b) -> i32` that implements the POSIX
  table (§5) — numeric when both effective-numeric, else `strcmp` of the
  resolved texts.

This keeps the phi model simple (one cell type per strnum slot) and confines
the POSIX comparison logic to one primitive. Cost: strnum slots are wider and
every field read runs the recogniser once (cheap, O(field length), and only on
the strnum path).

### 4b. Lazy / by-need recogniser

Keep the interned-id representation and compute looks-numeric **on demand** at
each comparison via `@wam_looks_numeric(id) -> i1` + `@wam_atom_to_double`. No
wider slot; provenance tracked by a **compile-time** "this slot is strnum" bit
(from origin analysis) rather than a runtime flag. Cheaper representation, but
recomputes the recogniser at every comparison and still needs the origin
analysis of §3.2 — so it saves representation width, not the hard part.

Recommendation: **4a**. The wider cell is the honest model and localises the
POSIX rules; 4b optimises the easy part and leaves the cross-cutting provenance
work untouched.

## 5. The comparison table

`@wam_strnum_cmp` (and the compile-time emitter that calls it) must implement:

| left | right | comparison |
|---|---|---|
| number | number | numeric |
| number | strnum (numeric) | numeric |
| number | strnum (non-numeric) | numeric side coerced to string, **lexical** |
| number | string literal | **lexical** (number → string) |
| strnum (numeric) | strnum (numeric) | numeric |
| strnum (numeric) | strnum (non-numeric) | lexical |
| strnum | string literal | lexical |
| string literal | string literal | lexical |

(Right column symmetric.) The single rule: **the comparison is numeric iff both
operands are numeric-typed** — where a strnum counts as numeric-typed only when
its content looks numeric, a real number always counts as numeric, and a string
literal never does. Everything else is lexical on the resolved texts.

## 6. Phased implementation plan

1. **Recogniser + primitive (runtime-only, unit-tested).** Add
   `@wam_looks_numeric` and `@wam_strnum_cmp` to `wam_llvm_target.pl`; unit-test
   in `tests/core/test_wam_llvm_assoc_i64_runtime.pl` against the POSIX table and
   the `10 9` / `10 9x` cases. No codegen change yet.
2. **`scalar_strnum` slot kind + origin analysis.** Add the fourth slot type and
   a `plawk_scalar_strnum_names/2` provenance pass: a name is strnum if assigned
   directly from a field / `split` element / `getline` and never subsequently
   from a literal or arithmetic. Type inference precedence: arithmetic-double >
   strnum > string > counter. Phi the `%WamStrnum` cell across blocks.
3. **Comparison dispatch.** Route guard/`while` comparisons whose operands are
   strnum (or number-vs-strnum) through `@wam_strnum_cmp`; keep the existing
   fast paths for statically-known number-vs-number and literal-vs-literal.
4. **Extend value sources.** Make `split`, `getline`, `ARGV`/`ENVIRON`, and
   command-line assignments strnum-producing, so the surface is consistent.
5. **Audit + honest-scoping gate.** Until a source is converted, keep the
   current static behaviour and *reject* (clean exit 3) the specific
   bare-comparison shapes that would mis-compare, rather than silently choosing
   wrong — matching the campaign's honest-scoping value.

## 7. Scope boundaries / deferrable

- **Uninitialised variables** are strnums of value `""`/`0` in POSIX; plawk's
  zero-initialised slots already approximate this and can keep the current
  behaviour under strnum (an unset strnum is numeric-0).
- **Locale-sensitive numeric parsing** (decimal comma, thousands separators) is
  out of scope; the recogniser uses the C locale.
- **`CONVFMT`/`OFS` formatting** of a numeric strnum printed as a string is an
  orthogonal follow-on (number→string formatting), not part of the comparison
  model.
- A **fields-only** first cut (strnum for `$N` and `split`, deferring
  `getline`/`ARGV`) is a legitimate intermediate milestone as long as step 5's
  gate rejects the not-yet-converted sources rather than mis-comparing them.
