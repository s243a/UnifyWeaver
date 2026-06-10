# WAM backend conventions (cons cells + operator functors)

This is a checklist of the **WAM-bytecode conventions** a target backend's
runtime has to honour to run UnifyWeaver's generated code correctly. It
exists because the *same handful of bugs* showed up, independently, in
every WAM backend as it was brought up — and the cross-target conformance
harness keeps re-discovering them one program at a time. If you are adding
or debugging a WAM runtime (Scala, Elixir, WAT, Haskell, Python, Go, and
the not-yet-conformant Rust/C/C++/Lua/…), read this first.

For *how a backend is wired into the conformance harness* (the driver /
invocation contract), see
[`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md). This
document is about the layer below that: making the bytecode evaluate to the
right answer in the first place.

The oracle for "the right answer" is the hand-specified expected-results
table in `tests/wam_conformance_fixtures.pl` (standard Prolog semantics).
Run your backend against it (`CONFORMANCE_TARGETS=<target>`) — the six
programs there (`member`, `append`, `reverse`, `fib`, `ack`, `builtins`)
are deliberately chosen to exercise every convention below.

---

## TL;DR — the six things that bite

| # | Convention | Symptom if you get it wrong | Conformance program that catches it |
|---|---|---|---|
| 1 | Cons cells have **two spellings** (`put_list` *and* `put_structure [|]/2`) | recursive list predicates mis-traverse the tail | `member`, `reverse` |
| 2 | A functor string is `name/arity` and **the name may contain `/`** (and `\`, e.g. `=\=`) | `//` and `/` arithmetic silently break; `=\=` no-ops in C string literals | `builtins` (`cbi_arith`, `cbi_cmp`) |
| 3 | Nested terms are built **outer-first** via placeholder vars that a later `put_*` must **bind** | nested structures / list tails stay unbound | `builtins`, `member`, `reverse` |
| 4 | **`deref` before every type test** (`is_var`, `is_list`, …) | a *bound* variable is mistaken for unbound → write-mode corruption | `member`, `reverse` |
| 5 | `is/2` must produce an **integer** for integral results (if unify is type-strict) | `R is N+1` fails when `R` is a ground integer | `fib`, `ack` |
| 6 | **Never drop or throw on an unhandled instruction** — emit a real no-op so PC/label alignment is preserved | indexing hints (`switch_on_term*`, `switch_on_constant_fallthrough`) vanish from the code vector, shifting every later label by one; backtracking skips `retry_me_else`/`trust_me` and loops or mis-clauses | `fib`, `ack`, `append`, `reverse` |

---

## 1. Cons cells have two spellings — alias them

The compiler does **not** spell every list cell the same way. The *outer*
cell of a freshly built list uses `put_list`; every *inner tail* cell uses
`put_structure [|]/2`; and the empty list is the **atom `[]`**. Compiling
`member(a, [a,b,c])` shows all three:

```
    put_list A2
    set_constant a            % head of outer cell
    set_variable X4           % tail placeholder (see §3)
    put_structure [|]/2, X4   % inner cell  ← NOT put_list
    set_constant b
    set_variable X5
    put_structure [|]/2, X5
    set_constant c
    set_constant []           % empty list = the atom []
```

So in your runtime a "list cell" is **any** of:

- the dedicated list representation produced by `put_list` / `get_list`;
- a 2-argument compound/structure whose functor name is `[|]` *or* `.`
  (i.e. `[|]/2` or `./2` — Prolog systems disagree on the spelling, so
  accept both, plus a bare `.`);

and the empty list is the atom `[]` (some backends also have a native
empty-list value — alias that too).

**What to make uniform:**

- `get_list` (read mode) must succeed on a `[|]/2`/`./2` structure, not
  only on your native list type, exposing `head = arg0`, `tail = arg1`.
- unification must treat a native list cell and a `[|]/2`/`./2` structure
  cell as equal when head and tail unify.
- term-decomposition used by `=..`, `functor/3`, printing, and **arithmetic
  evaluation** must apply the same aliasing.

**Precedents** (search these if you want a worked example):

- Elixir — `step_get_structure_matches?/2` aliases `./2`↔`[|]/2`; the
  runtime also aliases native `[]` with WAM `"[]"`.
- Haskell — `intern_struct_functor/2` folds every cons spelling onto a
  single `atomDot` id, so identity comparison just works.
- WAT — `get_list` also accepts a tag-3 `[|]/2` compound (`$cons_op1`).
- Go — `consHeadTail` + a `GetList` fallback for cons-functor structures.

---

## 2. Functor strings are `name/arity` — and the name can contain `/`

WAM instructions carry the functor as a single `name/arity` string. The
trap: the **name itself** can contain `/`, because Prolog's arithmetic
operators include `/` and `//`:

| Source | Functor string | Correct (name, arity) |
|---|---|---|
| `X + Y` | `+/2` | (`+`, 2) |
| `X mod Y` | `mod/2` | (`mod`, 2) |
| `X / Y` (float divide) | `//2` | (`/`, 2) |
| `X // Y` (integer divide) | `///2` | (`//`, 2) |

The naive parse — *split on `/`, expect exactly two parts* — turns `///2`
into four parts and `//2` into three, and silently falls back to arity 0
or an empty name. The 0-arity compound then carries no argument cells, and
`is/2` evaluates `17 // 5` to nothing → `cbi_arith` fails.

**Rule:** the arity is the final `/<digits>` segment; the name is
everything before it. Parse from the right (take the last `/`-separated
component as arity, `join` the rest back as the name) — never assume the
name is `/`-free.

**Precedents:** Haskell `bareArithOp`, WAT `functor_arity_of`, Python
`python_functor_arity/2`, Go `parseFunctorName`/`parseFunctorArity`. All
strip only a trailing `/<digits>`.

---

## 3. Nested terms are built outer-first — bind the placeholder

The compiler emits nested terms (and multi-element lists) **outermost
first**, dropping an unbound *placeholder* variable into the enclosing
argument and filling it in later. `R is A + B + C`, i.e.
`+(+(A,B),C)`, compiles to:

```
    put_structure +/2, A2     % outer +(_, _)
    set_variable X108         % outer arg0 = placeholder X108
    set_value X202            % outer arg1 = C
    put_structure +/2, X108   % inner +(_, _) built INTO X108
    set_value X200            % inner arg0 = A
    set_value X201            % inner arg1 = B
    is/2
```

`set_variable X108` puts one fresh variable in **two** places: the outer
structure's `arg0` *and* register `X108`. The later
`put_structure +/2, X108` must make the inner structure visible **through
the outer arg too** — i.e. it must **bind the placeholder**, not merely
overwrite register `X108`. If `put_structure`/`put_list` only overwrites
the register, the outer `arg0` stays pointing at the still-unbound
placeholder, and anything that walks the outer term (here, arithmetic
evaluation) gives up at depth ≥ 2. The identical shape produces list
**tail** cells (`set_variable` tail placeholder, then
`put_structure [|]/2` into it).

**Rule:** when `put_structure`/`put_list` writes to a register that
currently holds an unbound variable, **bind** that variable to the new
cell (and trail the binding so it is undone on backtracking). Then the
embedded copy resolves to the new term.

**Precedents:** Haskell — `addToBuilder`/finalize binds the embedded tail
placeholder; Go — `PutStructure` binds an unbound placeholder it
overwrites, via the trailed `bindUnbound`.

---

## 4. `deref` before every type test

A bound variable is still *typed* as a variable — binding lives in a trail
/ side table / heap cell, not in the value's Go/Haskell/… type. So a type
predicate that inspects the value directly (`isUnbound`, `is_list`,
`is_atom`, …) lies about a value that has since been bound.

The classic failure: `get_list` checks `isUnbound(reg)` **before**
dereferencing. After §3's fix a list tail is a *bound* variable; the
un-dereferenced check reports it as unbound, so `get_list` takes the
**write-mode** branch and fabricates a fresh list cell — wrongly
succeeding `member(z, [a,b,c])` and looping `reverse`.

**Rule:** `deref` first, *then* test the type. Every read-mode instruction
(`get_*`, `unify_*`) and every builtin that pattern-matches an argument
must deref before deciding "is this an unbound var / a list / a struct?".

**Precedent:** Go — `GetList` now derefs before the `isUnbound` check.

---

## 5. `is/2` result typing

`is/2` evaluates the right-hand side and unifies the result with the left.
If your arithmetic is computed in a single numeric type (e.g. all
`float64`) but your unifier is **type-strict** (an integer value never
unifies with a float value), then `R is N + 1` fails whenever `R` is
already bound to a ground **integer** — which is exactly the shape of
`ack(0,5,6)` (`R is N + 1`) and `fib`'s `R is R1 + R2`.

**Rule:** when the evaluated result is integral, produce your integer
value type (not a float). Mirror standard Prolog: integer-valued
expressions yield integers, genuinely fractional ones yield floats.

**Precedents:** Go wraps integral `is/2` results as `Integer`; Python uses
the same int-vs-float heuristic.

---

## 6. Unhandled instructions must no-op, not vanish

A backend that lowers the WAM listing one line at a time computes label
PCs on the assumption that **every instruction line occupies exactly one
slot**. If you encounter an instruction you do not implement and either
*drop it* (emit a comment / nothing) or *throw*, you break that
assumption:

- **Throwing** stops the whole backend (C used to `throw` on
  `switch_on_term_a2`).
- **Dropping** is worse — it silently corrupts. The dropped line still
  counted as a PC when labels were computed, so every label after it now
  points **one instruction too far**. The classic failure: a clause-chain
  label lands *past* its `retry_me_else`/`trust_me`, so the choice point's
  alternative PC is never updated — execution loops on the same clause
  (hangs / hits a step limit) or falls into the wrong clause.

This bit two backends. Rust dropped `switch_on_constant_fallthrough` and
`switch_on_term` (comments in the `vec!`); C threw on `switch_on_term_a2`.

**Rule:** emit a real **no-op** instruction (one slot) for anything you do
not translate. First-argument indexing (`switch_on_*`) is **only an
optimisation** — skipping it and letting the `try_me_else` clause chain run
is always correct, and the no-op keeps every later label aligned. Make the
*fallback* for unknown instructions a no-op, so the next unimplemented
indexing variant degrades gracefully instead of corrupting.

**Precedents:** Rust emits `Instruction::NoOp`; C emits `INSTR_NOOP`. Both
route any unrecognised `switch_on_*` (and the generic unknown-instruction
fallback) through it.

> Note: `switch_on_term` dispatches on the term's *tag*, so it must also
> obey §1 — a cons cell that happens to be a `[|]/2` **structure** has to be
> routed to the *list* clause, not the (empty) structure table.

---

## New-backend checklist

Before declaring a WAM backend conformant, confirm each of these against a
*two-or-more-element, recursive* list program and a *depth-≥2* arithmetic
expression (the conformance fixtures do both):

- [ ] `get_list` / list unification accepts `put_list` cells **and**
      `[|]/2`/`./2` structures, and the empty list `[]`. (§1)
- [ ] functor arity is parsed as the trailing `/<digits>`; `//` (`///2`)
      and `/` (`//2`) evaluate correctly. (§2)
- [ ] `put_structure`/`put_list` into a register holding an unbound
      placeholder **binds** the placeholder (trailed). Test
      `R is A+B+C` and a 3-element list. (§3)
- [ ] every type test derefs first; `member(z, [a,b,c])` is **false** and
      `reverse([a,b,c],[a,b,c])` terminates as **false**. (§4)
- [ ] `is/2` of an integral expression unifies with a ground integer. (§5)
- [ ] unimplemented instructions emit a **no-op** (not a comment / throw),
      so label PCs stay aligned; `switch_on_term` routes a `[|]/2` structure
      to the list clause. Test a *depth-≥2* recursion (`fib(10,55)`,
      `append`). (§6)
- [ ] operator functors are escaped for the host string syntax (`=\=` must
      survive as `=\\=` in a C/Java/… literal). (§2)
- [ ] `CONFORMANCE_TARGETS=<target>` is green with no `ct_xfail` entries.

The first backend that hits a *new* class of divergence should add a row
to the table in `WAM_CROSS_TARGET_CONFORMANCE.md` and, if it is a general
convention rather than a one-off, a section here.
