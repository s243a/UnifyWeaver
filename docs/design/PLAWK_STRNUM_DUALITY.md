<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk strnum: string/number scalar duality

**Status**: **forward-looking design note.** None of §3–§5 is implemented.
This captures what POSIX "strnum" is, where plawk's current static type model
diverges from it, why closing the gap is a *type-model* project rather than a
single feature, and a phased plan for getting there without destabilising the
three-way slot model that ships today.

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
