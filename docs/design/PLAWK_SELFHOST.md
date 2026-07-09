<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk self-host (JIT roadmap item 5, milestone 6)

The capstone of the eval bootstrap: a Prolog→`.wamo` compiler that itself
runs **as a loaded `.wamo` object**, so source text becomes a callable
grammar entirely inside the running binary. This is the design pass —
scoping *what* to build and *in what testable stages* — not an
implementation PR.

## What "self-host" means here (and what it does not)

Milestone 5 landed the eval loop at the runtime layer: `@wam_object_eval`
runs a compiler object on a source atom, takes the `.wamo` bytes it returns,
loads them with `@wam_object_load_bytes`, and hands back a fresh
`{vm, entry_pc}` — all in one process. It was verified with a **stand-in**
compiler that echoes its input (the "source" was already valid `.wamo`).
Milestone 6 replaces the stand-in with a compiler that actually **turns
Prolog source into `.wamo` bytes**.

The word "self-host" is doing careful work here. It does **not** mean
compiling the existing ~22 000-line `wam_target.pl` / `wam_llvm_target.pl`
to a `.wamo`. That compiler leans on the full SWI-Prolog library, module
system, DCG expansion, and host I/O — far outside the loadable subset, and
not a useful bootstrap target. It means:

> **A minimal, self-contained Prolog→`.wamo` compiler, written entirely in
> the loadable subset, that compiles a defined subset of Prolog source into
> a valid `.wamo` object, demonstrated end to end through the existing
> `@wam_object_eval` pipeline.**

"Self-host" in the bootstrap sense: the compiler is *the same kind of
artifact it produces* — a `.wamo` object — and could, in principle, compile
a source copy of itself once its accepted subset covers its own
constructs. That fixpoint is the north star; the deliverable is the staged
climb toward it, each stage a runnable, tested increment.

## Why this is now within reach

Three facts, established by earlier milestones, collapse what looked like
the hardest milestone into "write a small compiler in the subset we already
have":

1. **`.wamo` is a text format.** `wamo_serialize/8`
   (`wam_llvm_target.pl:21836`) emits a whitespace-separated token stream —
   magic, version, counts, length-prefixed strings, and
   `<tag> <op1> <op2> <relocid>` instruction rows. There is **no binary
   encoding step**. Emitting a `.wamo` is therefore *string assembly*, which
   milestone 4 already proved a loaded grammar can do (build a byte string
   via `number_codes`/`atom_concat`/`atom_codes`, return it as an Atom,
   host reads it via `@wam_object_call_bytes`). The compiler's back end is a
   string builder, not a bit-packer.

2. **The reader produces terms.** `read_term_from_atom/2` (milestone 3b) is
   a full recursive-descent parser with operator precedence, variables,
   floats, quoted atoms, and control operators — a whole clause parses
   (`foo(X) :- bar(X), baz(X)` → `:-(foo(X), ','(bar(X),baz(X)))` with `X`
   shared head↔body). The compiler's front end already exists as a runtime
   primitive.

3. **Term inspection and list processing are loadable builtins.**
   `functor/3` (id 26), `arg/3` (id 27), `=../2` (id 28), `copy_term/2`
   (29), plus the whole `atom_codes`/`number_codes`/`atom_concat`/
   `atomic_list_concat`/`char_code`/`sub_atom` family and list builtins
   (`append`, `length`, `reverse`, `nth0/1`, `msort`, `keysort`, …) all
   lower to `builtin_call <id>` and run in loaded objects. The compiler's
   middle — walk a term, build an instruction list, render it — is expressible.

So milestone 6 is **not** more runtime plumbing. The runtime-primitive layer
is complete (reader, term builtins, dynamic store, catch/throw, byte output,
eval pipeline). Milestone 6 is *authoring a grammar* in that layer, plus
closing a couple of small subset gaps the compiler grammar happens to hit.

## The compiler grammar's shape

A minimal single-clause compiler, front to back:

```
compile(Src, Wamo) :-
    read_term_from_atom(Src, Clause),      % front end (milestone 3b)
    clause_to_instrs(Clause, Instrs, Meta),% middle: codegen
    serialize_wamo(Instrs, Meta, Wamo).    % back end: string assembly
```

- **Front end** — `read_term_from_atom/2`, done. Accepts the clause shapes
  the reader covers.
- **Middle (codegen)** — the new work. Walk `Clause` (a head, optionally a
  `:-`/2 with a body), allocate argument/temporary registers, and emit a
  list of instruction terms `enc(Tag, Op1, Op2, Reloc)`. Needs term
  inspection (`functor/3`, `=..`, unification), a register/label counter
  (an accumulator threaded through recursion, or the dynamic store), and the
  instruction/operand encoding conventions the loader expects.
- **Back end (serializer)** — a loadable-subset re-implementation of
  `wamo_serialize/8`: assemble the header (`WAMO\n2\n<entry>\n<NE>\n`), the
  named-entry table, the four length-prefixed string tables (atoms,
  functors) and count/PC/instruction/meta-row sections, as one Atom. Pure
  string building — `number_codes`/`atom_concat`/`atomic_list_concat`.

## Loadability of the pieces the grammar needs

Almost everything the compiler grammar needs is already loadable. The audit:

| Compiler need | Mechanism | Loadable? |
|---|---|---|
| Parse source → term | `read_term_from_atom/2` (id 174) | **yes** (M3b) |
| Inspect term shape | `functor/3` (26), `=../2` (28) | **yes** |
| Decompose args | `arg/3` (27) **with a variable index** | **yes** (see gap below) |
| Build instruction/string lists | `append`/`length`/`reverse`/`nth0` (55/31/54/51) | **yes** |
| Number → text | `number_codes/2` (43), `atom_number/2` (42) | **yes** |
| Assemble byte string | `atom_codes` (36), `atom_concat` (40), `atomic_list_concat/2,3` (73/74), `char_code` (37) | **yes** |
| Formatted assembly (optional) | `format/2` (32) into a code buffer | **yes** |
| Return bytes to host | Atom output → `@wam_object_call_bytes` | **yes** (M4) |
| Symbol tables / counters | dynamic store `assertz`/`retractall` (175/177) or threaded accumulator | **yes** (M3b-db) |
| Ordered dedup for tables | `sort/2` (81), `msort/2` (30), `keysort/2` (80) | **yes** |
| Recursion over its own clauses | `call`/`execute`, `try_me_else` chains, indexing nop | **yes** (M1/M2) |
| Error handling on bad source | `catch/3` / `throw/1` | **yes** (M3c) |

### The one real subset gap: `arg/3` with a **constant** index

A grammar written as `arg(1, T, A), arg(2, T, B)` — literal indices — is how
one naturally decomposes a fixed-arity compound. The tier-2 compiler
specialises a **constant-index** `arg/3` into a dedicated `arg` opcode that
is **not** in the `.wamo` loadable subset, so such a clause will not load.
Two ways around it, in preference order:

1. **Coding constraint on the bootstrap compiler (zero new work):** decompose
   with a **variable** index — `arg(N, T, A)` where `N` is bound at runtime
   (e.g. a loop counter over `1..Arity` from `functor/3`). That stays a
   plain `builtin_call arg/3` (id 27), which *is* loadable. Or use `=../2`
   to get the whole arg list at once and walk it with list builtins. Either
   avoids the specialised opcode entirely. **This is the recommended path for
   milestone 6** — the compiler grammar simply never writes a constant-index
   `arg/3`.
2. **Lift the `arg` opcode into the loadable subset** (a small, independent
   PR — one of the deferred items). Cleaner long-term, but not required to
   ship milestone 6 if we accept constraint (1).

### Instruction & operand encoding — the detail to pin down

`wamo_serialize/8` emits each instruction as `enc(Tag, Op1, Op2, Reloc)`.
The compiler grammar must reproduce:

- **Tag numbers** for each WAM opcode it emits (e.g. `get_constant`,
  `put_value`, `call`, `proceed`, `builtin_call`). These come from the same
  encoding `wam_instruction_to_llvm_literal` / `wamo_enc` uses.
- **Operand packing** — the reference `p(R) :- R = 42` object encodes its
  body-`=` as `0 42 65536 0` (tag 0, op1 `42`, op2 `65536` = `0x10000`) then
  `20 0 0 0` (proceed). The `0x10000` is a **register-encoding convention**
  packed into op2; the compiler must emit the same packing. Nailing the
  register/argument encoding is the single most error-prone part of codegen
  and the first thing Stage A locks down against a golden object.
- **Reloc classes** (atom/functor/float table indices). The minimal compiler
  can start with only the reloc classes its subset needs (integer/atom
  constants), extending as the subset grows.

The safest way to fix these conventions is **differential**: for each source
program the bootstrap compiler should handle, compile it *with the host*
`write_wam_object/3` to get a golden `.wamo`, and require the grammar to emit
a byte-identical (or load-equivalent) object. That turns "did I get the
encoding right?" into a diff.

## Staged plan

Each stage is an independently testable PR; each leaves a runnable artifact.
The spine is **build the back end first** (deterministic, diffable against
golden objects), then the front-to-back path over a widening source subset.

### Stage A — `.wamo` serializer in the loadable subset — *LANDED*

Port `wamo_serialize/8` to a loadable grammar. Input is a **fixed instruction
list** given as ground `enc(Tag,Op1,Op2,Reloc)` terms (no codegen yet); output
is the `.wamo` byte string. The serializer runs *as a loaded object* (the
eval-pipeline compiler entry `wamoserz(Src, Wamo)`, `Src` ignored) and emits a
valid `.wamo` for a 42-returning program built only from loadable builtins
(`atom_codes`/`number_codes`/`append`/`length`) — genuine string assembly, not
a string literal. Two checks (`tests/test_wam_object.pl`): the serializer's
output is **byte-identical to the host writer's golden `.wamo`** for the same
program (differential, run in SWI — locks the operand encoding and section
layout); and the object it emits, run through `@wam_object_eval`, **loads and
runs to `42`** end to end (the eval loop closes on a grammar-emitted object).

**Deliverable — done:** a loaded grammar emits a valid `.wamo` from a
hand-supplied instruction list; emitted object loads and runs.

**Two loaded-runtime bugs Stage A surfaced — both now FIXED (each its own PR):**

1. **Register-file ceiling (correctness bug, affected AOT too) — FIXED.** The
   register file was `[64 x %Value]`, partitioned A1–16 / X1–32 / **Y1–16**
   (48–63). A clause holding >16 permanent variables assigned Y17+ → array
   index 64+, writing **past the register array** into the adjacent `%WamState`
   fields (the stack pointer) → memory corruption / segfault. Confirmed: a
   clause with 16+ call-spanning variables crashes; ≤12 is fine. Fixed by
   enlarging the file to `[128 x %Value]` and widening the Y window to Y48
   (formula unchanged, so Y1–16 keep identical slots), growing the
   `allocate`/`deallocate` snapshot to match, and adding a compile-time guard
   (`wam_too_many_permanent_vars`) so any future overflow is a clean error, not
   corruption. The Stage A serializer's small-clause style is no longer forced,
   but remains good practice.

2. **`get_structure` did not compare the functor — FIXED.** The real bug behind
   the "multi-way functor dispatch anomaly" was simpler and more fundamental
   than first thought: `get_structure f/N` entered read mode for **any**
   Compound (`tag == 3`) without comparing the functor name or arity against the
   expected `f/N`. So `get_structure atom/1` on `ins(enc(...))` *succeeded* and
   read `ins`'s arg as if it were `atom`'s. Multi-clause first-argument dispatch
   therefore only worked by accident — the wrong clauses' **bodies** had to fail
   (`atom_codes`/`number_codes` of a compound fails). A body that did not
   cleanly fail (e.g. `length/2` on a compound) ran the wrong clause and
   returned garbage; a first clause whose body *succeeded* on the wrong data
   returned the wrong answer outright. This affected AOT and loaded objects
   alike. Fixed by comparing arity, then functor (via `@wam_functor_eq` —
   pointer-fast with a `strcmp` fallback for reader/`=..`-built compounds),
   before entering read mode; a mismatch now fails as it should.

**Consequence for later stages:** a tagged token/AST union walked by
first-argument functor dispatch — the natural codegen representation for Stages
B–D — now dispatches correctly in loaded objects. The serializer here still uses
list `[]`/`[|]` + `enc/4` dispatch, but that is no longer a requirement.

*Aside:* the byte-return path (`atom_codes` interning) strips **trailing**
newlines from an atom, so the loaded serializer's object is 1 byte shorter than
the golden (final `\n` after the last token dropped). This is **load-equivalent**
(the `.wamo` token parser does not need the trailing newline) — hence the
byte-identity check runs in SWI and the loaded check asserts load-and-run. Worth
noting because a milestone-4 byte output that legitimately ends in `\n` would
lose it; a candidate fix if that ever matters.

### Stage B — minimal codegen for one clause shape — *LANDED*

`cgcompile/2` is a real compiler grammar (the eval-pipeline `compile(Src,Wamo)`
entry): it parses `Src` with the runtime reader (`read_term_from_atom/2`), walks
the clause to an instruction list (`clause_to_instrs/3`), and hands it to the
Stage A serializer (`wz_serialize/8`). The loadable subset: a one-argument
clause whose body binds the head variable to an integer, either directly
(`p(R) :- R = 42`) or by evaluating a ground arithmetic expression
(`p(R) :- R is 6*7`). Both lower to `[get_constant(V,A1), proceed]` — the golden
shape from Stage A, parameterized by the value `V` (`= N` binds `N`; `is Expr`
evaluates `Expr` in the grammar). `body_int/2` dispatches on the body's functor
(`=/2` vs `is/2`) — the exact tagged-dispatch shape the get_structure functor
check made correct. No constant-index `arg/3` (the known subset gap): `functor/3`
gives the predicate name and `body_int/2` the value.

**Deliverable — done:** source text → `.wamo` → run, end to end through
`@wam_object_eval`. Verified (`tests/test_wam_object.pl:selfhost_codegen_stage_b`):
both `p1(R) :- R = 42` and `p1(R) :- R is 6*7` compile from source text inside a
loaded compiler object and run to `42`. Codegen logic also checked byte-identical
to the host writer's golden `.wamo`. This is the first source→bytecode compile:
reader + codegen + serializer composed into one loadable object.

### Stage C — multi-goal bodies and predicate calls — *predicate calls LANDED*

Extend codegen to conjunctions (`,`/2), calls to other predicates
(`call`/`execute` + the meta-call table from M2), and multi-argument heads
with register allocation. Test: a two-clause source program where one
clause calls the other.

**Predicate calls — LANDED.** `cgcprog/2` compiles a **multi-clause** program
(source parses in one reader call to a *list of clauses*,
`[(main0(R):-helper(R)), helper(42)]`) into a **multi-predicate `.wamo`**. Each
clause gets a label (its index); a single-goal predicate-call body compiles to
`[allocate, deallocate, execute(CalleeLabel)]` (a tail call), with the callee
resolved through a name/arity→label map built in a first pass. `execute`
references the label **directly** (tag 19, op1 = label index), so no meta-call
table is needed (`NM = 0`). The label→PC table is exactly the PC list the Stage
A serializer already accepts — so the serializer was unchanged; all the new work
is codegen (label assignment, PC computation, call resolution). Facts `P(Int)`
and the Stage B body forms (`=`/`is`) still lower to
`[get_constant(V,A1), proceed]`. Verified end to end
(`tests/test_wam_object.pl:selfhost_codegen_stage_c`): `main0` tail-calls
`helper` → `42`; the codegen is byte-identical to the host writer's golden for
the same program.

**Conjunction + register allocation — LANDED.** `cgconj/2` compiles a clause
whose body is a conjunction of unification goals (`X = Y`, `X = Int`) with a
real (simple) register allocator: `numbervars/3` binds each source variable to
`'$VAR'(N)`, mapped to permanent register Y(N+1); an initialized-Y set is
threaded through so a variable's **first** occurrence emits
`put_variable`/`get_variable` and **later** occurrences `put_value`. Head
arguments are saved with `get_variable`; each `=` goal stages A1/A2
(`put_variable`/`put_value`/`put_constant`) then `builtin_call =/2` (id 24).
This is the WAM register-allocation core. Verified byte-identical to the host
writer for `pconj(R) :- Y = 42, R = Y` (Y a temporary shared across both goals),
and end to end (`tests/test_wam_object.pl:selfhost_codegen_stage_c_conjunction`):
compiles from source and runs to `42`.

**Runtime arithmetic — LANDED.** `cgarith/2` extends the conjunction compiler
with `Var is BinExpr` goals. A binary expression `op(A,B)` builds a compound on
the heap: `put_structure op/2` into A2 (op1 = **functor-table index**, reloc 2),
then `set_value` (a variable operand) / `set_constant` (an integer operand) per
arg, then `builtin_call is/2` (id 0). The functor names used across the clause
are collected into the object's **functor table** (`NF > 0`), emitted by a
functor-aware serializer (`wzf_serialize`); `put_structure`'s op1 references a
functor by index, which the loader relocates to the object's own functor
pointer. Reuses the cgconj register allocator. Verified byte-identical to the
host writer for `ca(R) :- X is 6*7, R = X` — which combines conjunction, a
shared temporary, arithmetic, and unification — and end to end
(`tests/test_wam_object.pl:selfhost_codegen_stage_c_arithmetic`): both
constant-operand (`6*7`) and variable-operand (`R is A+B` after `A=40, B=2`)
forms compile from source and run to `42`.

**Non-tail predicate calls — LANDED (the unified compiler).** `cgfull/2` merges
every Stage C piece into one multi-clause compiler: labels + PC table, per-clause
register allocation, conjunction, runtime arithmetic + functor table, and now
non-tail predicate-call goals. A call goal `p(Args)` stages A1.. with the args
(`operand_instr`) then `call(CalleeLabel, arity)` (tag 18) — a *non-tail* call:
`cp` = the next PC, so execution resumes after the call when the callee
proceeds, and a permanent (any variable spanning the call, made a Y-register by
the allocator) carries the result forward. Each clause is `copy_term`'d +
`numbervars`'d so its variables are clause-local; facts emit head + `proceed`
(no env). Verified byte-identical to the host writer for
`[(mnt(R):-helper(A), R=A), helper(42)]`, and end to end
(`tests/test_wam_object.pl:selfhost_codegen_stage_c_nontail_call`): a passthrough
call (`mnt` → `helper` → 42) and a compute call whose callee does arithmetic
(`main0` calls `add1(41,V)`, `add1(X,Y):-Y is X+1`, → 42) both run.

*Runtime fix this surfaced:* the transient **arena** (put_structure / `=..` /
reader workspace) was 1 MiB and grows monotonically within a top-level call;
`wam_arena_alloc` returns null past the cap and callers do not check, so an
allocation-heavy grammar — a compiler assembling instruction/code lists via
`append` — silently exhausted it and segfaulted. Enlarged to 16 MiB (headroom
for compiler workloads). This is why cgfull can now compile a program mixing
calls *and* arithmetic (which crossed 1 MiB) rather than only one or the other.

**Remaining:** nested arithmetic expressions (recursive `put_structure`) and
last-call optimization (a tail call as `execute` rather than `call`+`proceed`).
Both are codegen refinements; the mechanisms are all in place.

**Deliverable — met:** the compiler handles the clause shapes a small
hand-written grammar uses (multi-clause, conjunction, register allocation,
unification, arithmetic, predicate calls) — "compile a grammar at runtime from
source" is now real for non-trivial grammars. Stage D widens the accepted subset
toward the compiler compiling itself.

### Stage D — widen toward the compiler's own subset (the fixpoint)

Iteratively add the constructs the *bootstrap compiler's own source* uses
(list building, term inspection, the control operators it emits) until the
compiler grammar can compile a source copy of itself and the re-compiled
object behaves identically. This is the self-host fixpoint. It is a campaign,
not a single PR — each construct added is a small, tested increment.

**Multi-clause predicates — LANDED (the Stage D opener).** The biggest
structural gap between `cgfull` and real Prolog was one-clause-per-predicate.
Now consecutive clauses with the same name/arity group into a predicate
(`group_clauses`); the predicate owns the entry label, clauses 2..k get
**alternative labels** (laid out after all entry labels, so the label→PC table
is simply `EntryPCs ++ AltPCs`), and a multi-clause predicate compiles to a
`try_me_else(Alt1)` / `retry_me_else(Alt_k+1)` / `trust_me` chain (tags
22/23/24) around the per-clause code. A failing head match backtracks to the
next clause — **backtracking dispatch and recursion now compile from source**.
Verified end to end
(`tests/test_wam_object.pl:selfhost_codegen_stage_d_multiclause`): a two-clause
dispatch whose caller needs the *second* clause (→ 42), and **recursive
factorial** (base clause + recursive clause with arithmetic and a self-call by
label; `fact(3)` → 6), both compiled inside a loaded compiler object.

**Two more runtime bugs this stage surfaced — both FIXED:**

1. **Choice-point saved-register block not widened with the register file.**
   The register-file fix (PR #3510) grew the file to `[128 x %Value]` and the
   env `y_save` to 48 permanents, but the `%ChoicePoint` register block stayed
   `[64 x %Value]` — so `try_me_else` saved and `backtrack` restored only regs
   0–63. A **failed clause body** writes its own Y registers and never reaches
   `deallocate` (the env snapshot is bypassed); backtrack restored Y1–Y16
   (48–63) from the CP but left Y17–Y48 (64–95) holding the failed clause's
   junk — corrupting the backtracked-to alternative's permanents. The
   multi-clause compiler was the first backtracking-heavy client with clauses
   big enough (the new `cgfull/2` entry holds 17 permanents) to hit it.
   Fixed: CP block `[128 x %Value]`, save/restore copies 2048 bytes (the
   deliberate 512-byte *iterator* restores are untouched).

2. **`copy_term/2` aliased instead of copying.** The runtime
   `@wam_copy_term_value` was documented as naive: a `Ref` fell into the
   atomic default and was returned **unchanged**, so the "copy" shared the
   source heap cells — `numbervars`/unification on the copy bound the
   *original* through them — and unbound sharing was not preserved
   (`p(X,X)` copied to `p(_A,_B)`). The compiler grammar hit this squarely:
   clauses parsed from one source string share variables by name, and
   `numbervars` on clause 1's "copy" contaminated clause 3, making two
   different variables compile to the same Y register. Fixed with a
   var-mapped deep copy (the `@wam_dyn_copy_var` technique):
   `@wam_deref_keep_var` keys each distinct unbound cell by heap address,
   each maps to exactly one fresh heap cell — sharing preserved, zero
   aliasing.

**Remaining for the fixpoint:** first-arg **indexed** dispatch is not needed
(chains suffice, correctness-first), but the compiler's own source also uses
list patterns in heads (`[C|Cs]`), cut, `==`/`=:=` guards, if-then-else — the
next Stage D increments, in whatever order the fixpoint attempt surfaces them.

**Lists + atom table — LANDED.** The list-walking shape — a base clause on
`[]` and a recursive clause destructuring `[H|T]` — now compiles from source:
`[(main0(R):-suml([10,20,12],0,R)), suml([],A,A), (suml([H|T],A,R):- A1 is
A+H, suml(T,A1,R))]` → `42`. New pieces, learned from the host golden:
- **Atom table** (`NA > 0`): a var-safe walk over every clause term collects
  `[]` and atom constants (plus the cons functor `[|]`); atom constants carry
  the atom-table *index* with reloc class 1, relocated at load to interned
  atom ids that unify with reader-built atoms. NB: in SWI `atom([])` is
  *false* (nil is a special object), so nil has dedicated clauses in each
  argument compiler; in the loaded runtime nil *is* the atom `"[]"` and the
  same clauses match it.
- **Head list patterns**: `[H|T]` → `get_list Ai` (4) + per element
  `unify_variable` (5, first occurrence) / `unify_value` (6, later) /
  `unify_constant` (7, integers with tag<<16, atoms with reloc 1).
- **Repeated head variables**: second occurrence now emits `get_value` (2) —
  previously a second `get_variable` silently *overwrote* the first binding
  instead of unifying (`suml([],A,A)` would have been wrong): a latent
  codegen bug this increment fixed.
- **List literals in call args**: built top-down like the host — `put_list
  TARGET` (12), `set_*` for the head, `set_variable Xtemp` (13) for the tail,
  then `put_structure cons/2` into the temp (write mode binds through). X
  temps start at reg 16 and live only within the build.

**Comparison guards + if-then-else — LANDED.** Comparison goals (`>`, `<`,
`>=`, `=<`, `=:=`, `=\=`, `==`, `\==`) stage A1/A2 and emit `builtin_call`
with the comparison id. `( Cond -> Then ; Else )` compiles to the host's ITE
shape: `try_me_else(ElseLabel)` pushes the guard CP; Cond runs; `cut_ite`
(tag 31, soft cut) pops the guard CP on success; Then runs and
`jump(JoinLabel)` (tag 32, label operand) skips the else; at ElseLabel a
`trust_me` pops the CP (reached by backtracking when Cond fails, which also
undoes Cond's bindings) and Else runs to the join. This forced codegen to
become **PC- and label-aware**: else/join labels are allocated mid-clause
from the same counter as clause-chain alternatives, each recorded as a
Label-PC pair, keysorted at the end (`keysort` + `pairs_values`, both
loadable) into the positional label→PC table. Init-set rule: Then continues
from Cond's set; Else restarts from the pre-ITE set (backtracking undid
Cond's register writes); after the ITE the set is the intersection of the
branch out-sets — variables introduced inside one branch are branch-local
(the compiler's own helpers bind head-initialized outputs in branches, which
this covers). Verified end to end
(`tests/test_wam_object.pl:selfhost_codegen_stage_d_guards`): max-of-two via
ITE through the THEN branch and through the ELSE branch (both → 42), and a
plain comparison guard in a conjunction (`X > 10` → 42).

**General structure patterns — LANDED.** Arbitrary compound terms now compile
in heads and call arguments — `pt(X,Y)`, nested `f(g(X))`, pair sugar `A-B`,
and the compiler's own instruction shape `enc(T,O1,O2,Rl)`. Head side:
`get_structure F/N Ai` + one `unify_*` per arg, with compound/list **children
deferred**: `unify_variable Xtemp` saves the child during the parent's unify
sequence, and after the parent completes a `get_structure`/`get_list` into
the temp reads it (the host's canonical nesting shape). Build side is the
mirror: `put_structure F/N` + `set_*` per arg, children deferred via
`set_variable Xtemp` and built after the parent with write-mode
`put_structure` binding through the fresh cell. One X-temp counter (from reg
16) threads through each head or build. The table walk now collects **every
data functor** (subsuming the old is-operator collection; control/goal
functors are skipped). Verified end to end
(`tests/test_wam_object.pl:selfhost_codegen_stage_d_structures`): write-mode
build via a fact head (`mk(pt(40,2))` called with unbound), read-mode
destructuring, nesting, pairs, and `enc/4` — all → `42`.

*Runtime bug this surfaced — the `get_structure` fix's sibling:* **`get_list`
read mode accepted ANY compound** — no cons-functor check, no arity check —
so a `[H|T]` clause head wrongly matched e.g. `foo(A,B)`, reading its args as
list head/tail. Unexercised until arbitrary compounds started reaching
list-matching clauses (the grammar's own `walk_term([H|T],…)` clause matched
every compound, silently corrupting the functor table). Fixed like #3511:
verify `@wam_functor_is_cons` (byte compare, so reader-built lists match) and
arity 2 before pushing the UnifyCtx — the fifth latent runtime bug the
self-host campaign has found and fixed.

**Builtin goals — LANDED.** A whitelist of ~37 builtins the compiler's own
source uses now compiles as goals: term inspection (`functor/3`, `arg/3`,
`=../2`, `copy_term/2`), text (`atom_codes`, `number_codes`, `atom_concat`,
`char_code`, `term_to_atom`, `read_term_from_atom`, …), lists (`length`,
`append`, `reverse`, `nth0/1`, `msort`, `sort`, `keysort`, `pairs_*`,
`memberchk`, `between`), type checks (`atom`, `integer`, `var`, `compound`,
…), `numbervars/3` and `\=`. Each stages its args like a call (`c_operand`
per arg) and emits `builtin_call(Id, Arity)`. Notably the grammar emits
`builtin_call arg/3` even for a constant index — the host compiler's
specialised-`arg` subset gap simply never arises in self-compiled code.
`=/2` upgraded to full-term operands (either side may be a structure or list
literal), and the table walk collects data atoms into the atom table.
Verified end to end (`selfhost_codegen_stage_d_builtins`): functor+arg,
`=..` construct + structure unify, atom_codes+length over a data atom, and a
type-check ITE condition — all → `42`.

**Runtime fixes this surfaced (nos. 6–7 of the campaign):**

1. **The loaded reader lacked `=..`, `=\=`, `\==`** in its operator table
   (`@wam_infix_op`) — `T =.. L` would not even parse in a loaded object.
   Added as xfx 700 length-3 symbol operators (`.` was already a symbol
   char, so `=..` tokenizes as one run).
2. **`=../2` compose mode had never worked** ("most benchmarks only exercise
   decompose"). Three distinct defects, found by printf-instrumenting the
   generated IR: (a) the list walk **never dereferenced** — compiled list
   spines link conses through Refs (`set_variable` tail + write-mode
   `put_structure`), so the walk bailed at the first tag-5 tail; (b) the
   functor element was converted by `inttoptr` of its payload — but atoms
   are id-based since M100, so the built compound carried a garbage functor
   pointer (the decompose/functor/3 fix had missed compose); (c) the result
   was bound with `wam_set_reg` — overwriting register A1 only, never
   binding **through** the Ref shared with the caller's permanent register
   (the M9 aggregate-result bug, resurfaced in univ), so the constructed
   term vanished for later goals. All three fixed; construct mode now works
   in AOT and loaded objects alike.

**THE FIXPOINT — first slice LANDED.** The loaded bootstrap compiler compiled
the source text of **its own Stage A serializer** (the `wz_*` chain, restated
cut-free in the accepted subset — 18 clauses, ~1.4 KB), and the
doubly-compiled serializer then serialized the golden `ea(R):-R=42` program
**byte-exactly**: the compiled program checksums its own output (byte sum +
length = 2263) and matches the Stage A implementation computed in SWI. The
compiler has compiled its own back end — the first self-application.
Two small subset gaps closed on the way: `sum_list/2` whitelisted, and
**var-tail list literals** (`[32|Cs]` in a call argument) supported in the
build path (the tail variable is staged directly as the final `set_*` slot).

*Runtime finding (campaign no. 8):* the 16 MiB arena was still too small —
accumulator-style grammars (`append` onto a growing list per emitted byte)
allocate **quadratically** in their output size, and the fixpoint compile
segfaulted at exactly the 16 MiB boundary (a `%Value` store to the null
`wam_arena_alloc` result, pinned by gdb). First mitigated by bumping to
256 MiB virtual; then fixed for real with the **chained arena** (below).

**Chained arena LANDED** (the honest fix for finding no. 8). The bump arena
now links a new block on exhaustion instead of returning null: each block
carries a 32-byte header (previous-block pointer, data capacity, virtual
base offset), the allocation slow path mallocs `max(cap × 2, requested)`
and chains it, and blocks **never move** — `%Value`s hold raw pointers into
them, so live terms stay valid across growth. Marks became virtual offsets
(`base + pos`), globally monotonic across links, so the graph kernels'
mark/rewind pairs keep working even when the arena grew in between: rewind
pops (frees) every block newer than the mark and restores the bump position
in the survivor. `@wam_arena_reset` (per-query cleanup) rewinds to virtual
offset 0, handing growth blocks back to malloc while keeping the initial
block mapped. The initial block dropped back to 16 MiB — the default query
never chains; the self-hosted compiler's quadratic appends now hit growth,
not a segfault, with no practical ceiling. Covered by a dedicated native
driver suite (`test_wam_llvm_arena_chain_runtime.pl`): a 64-byte first
block forced through two links with mark checks at every step, cross-block
rewinds, block-stability sentinels, first-block reuse after reset, destroy
+ ensure re-init; plus a ~100 MB growth-stress loop through a 1 KiB initial
block. The fixpoint compile itself (needs > 16 MiB before linearisation)
exercised chaining in production on every suite run.

**Difference-list serializer LANDED** (the compile-*time* fix). The `wz_*`
emitters flipped from forward accumulators (`append(A0, Cs, B)` — copy the
ENTIRE output so far, per emitted token) to difference lists: `A0` is the
open list being built, `A1` its tail after the item, and each emission
appends only its own codes (`append(Cs, [10|A1], A0)`). The threading
predicates (`wz_header`/`wz_body`/`wz_funcs`/…) are direction-agnostic and
did not change; only the leaf emitters and the top wrappers (which now
close the tail with `[]`) know the representation. The fixpoint source
restated its serializer the same way — the doubly-compiled serializer is
linear too, and compiling it exercises open-tail call operands
(`[10|A1]`, `[32|Mid]`) and `append/3` with partial/unbound tails in the
loaded subset. Measured on the eval host (loaded `cgfull` compiling scaled
copies of the fixpoint source):

| source | quadratic (before) | difference lists (after) |
|---|---|---|
| 1.4 KB | 0.17 s / 246 MB | 0.01 s / 10 MB |
| 2.9 KB | 4.5 s / 944 MB | 0.01 s / 10 MB |
| 5.9 KB | 20.2 s / 3.7 GB | 0.02 s / 17 MB |
| 11.9 KB | (est. ~15 GB — untested) | 0.04 s / 35 MB |

The compile-time budget for the full self-compile is closed: an 11.9 KB
source — larger than cgfull's own restated source will be — compiles
loaded in 40 ms.

*Runtime finding (campaign no. 9):* the difference-list style immediately
exposed a variable-identity bug in **two** runtime paths, both the same
shape: using the FULL deref on a value that can be an unbound variable
collapses Ref-to-unbound-cell into the shared Unbound sentinel `{tag 6,
payload 0}` — no cell address — so "binding" one side to it silently
severs the link (the same collapse `wam_strict_eq` had already documented
for `==`). (a) `get_value` var-var: a head like `wz_funcs([], A, A)`
called with two unbound arguments left them UNLINKED — the unification
was a no-op that still succeeded, and every later binding through one
side was invisible through the other; the serialized output truncated at
exactly that knot. (b) `builtin_append`'s result-tail seed:
`append(Cs, T, L)` with `T` an unbound BARE variable seeded the built
list's tail with the collapsed sentinel instead of `Ref{cell of T}`.
Fixed via `@wam_deref_keep_var` (variables keep their cell identity):
`get_value` now classifies each side as variable-with-cell / naked
register variable / bound, links var-var pairs toward an existing heap
cell (same-cell guard against the M139 self-reference hazard; a fresh
shared cell if both sides are naked register variables), and binds
var-bound through `wam_bind_reg`; the append seed uses the keep-var
deref. Regression test `get_value_var_var_and_append_var_seed` drives
both paths through a loaded object.

**GEN 3 LANDED — the compiler compiles a compiler.** The next slice after
the serializer: the loaded `cgfull` (gen 1) compiled a mini-COMPILER
(gen 2, `fixpoint_compiler_source/1`), and the doubly-compiled compiler
compiled TWO golden programs byte-exactly (combined checksum 4676,
verified against the Stage A serializers computed in SWI). Where the
serializer slice started from a hard-coded instruction list, gen 2 starts
from **source text**: it runs the runtime reader as a compiled goal
(`read_term_from_atom/2`), decomposes the clause with `=../2`, makes a
**dispatching** codegen decision — `( integer(V) -> ` int `get_constant`
`; ` atom `get_constant` with an **atom-table row emitted from the
compiled program** (reloc class 1) `)` — assembles the entry-name codes,
and serializes with the difference-list wz chain. Front-end ground proven
for the capstone: the embedded program sources are quoted atoms with
spaces and operators (`'ea(R2) :- R2 = 42'`) that survive collection into
gen 2's atom table, relocation at load, and re-parsing by the loaded
reader — two compile generations deep. One representational constraint
surfaced: control functors (`:-`, `,`, …) are excluded from the data
tables by design, so a compiler in the subset decomposes clauses with
`=../2` instead of matching a `(H :- B)` pattern literal.

*Diagnosability note (the round's real lesson):* the first attempt wrote
`R is S1 + L1 + S2 + L2` in gen 2 — a NESTED arithmetic expression, the
known deferred gap. In SWI that fails cleanly; **loaded, the codegen
failure exploded into catastrophic backtracking** through the compile's
stale choice points (loaded objects run unindexed, cut-free clause
chains) — a silent multi-minute hang, not an error.

**Both follow-ups LANDED** the next round:

- **Nested arithmetic.** The `is`-expression is just a term, and cgfull
  already had the machinery: `f_goal`'s `is`-clause now stages the
  expression with `c_operand` (`build_struct` + X-temp deferral), so
  `(X + Y) * (X - 1) + 100` compiles; the operators land in the functor
  table automatically because `walk_term` skips `is/2` itself but walks
  its argument tree. gen 2's `main0` writes the nested
  `R is S1 + L1 + S2 + L2` again — the flattening workaround is gone.
  Test `selfhost_codegen_stage_d_nested_arith` (→ 114). The Stage C
  `cgarith` chain keeps its historical flat `expr_build` untouched.
- **Fail-fast compile diagnostic.** Catch-all clauses at the end of
  `f_goal`, `c_operand`, and `head_arg_instrs` `throw/1` a diagnostic
  term (`cg_unsupported_goal(P, A)` / `..._operand(T)` /
  `..._head_arg(T)`) instead of failing — and `throw` is loadable (call
  sentinel), so the loaded compiler aborts immediately rather than
  backtracking for minutes. Test
  `selfhost_codegen_fail_fast_on_unsupported`: a `findall/3` source
  makes the eval host exit nonzero at once (this test would time the
  suite out on the old behavior).

**THE MIDDLE, first slice LANDED — the compiler compiles its own codegen.**
The loaded cgfull compiled `fixpoint_middle_source/1` (mid2): a cut-free
restatement of the single-clause codegen — `copy_term` + `numbervars`
(both loadable builtins) to make variables matchable, conjunction
splitting via `=..`/`==` (control functors cannot be pattern literals),
functor-table collection in first-occurrence order (head functor first,
then the is-expression operators — matching `collect_tables`), a head
walk with variable-index `arg/3` emitting `get_variable`/`get_value` by
first occurrence, and `L is E` compiled the way `f_goal` does it
post-nested-lift (operand → A1, `put_structure op/2` + `set_*` → A2,
builtin `is/2`). The init-set threads as a plain N-list with `memberchk`;
every dispatch is an ITE. The doubly-compiled codegen compiled
`sum3(A, B, R2) :- T is A + B, R2 is T + 1` **byte-identically to the
production cgfull middle** (`cgfull_term/2`, the reader split off so SWI
can compute the golden bytes) — real register allocation, two compile
generations deep, checksum 8755. Test `selfhost_codegen_stage_d_middle`.

Getting there surfaced three more items:

- *Runtime finding (campaign no. 10):* an **uncaught throw behaved as a
  plain failure** — `wam_throw` set halted and returned false, but
  `@backtrack` resumed into live choice points and **cleared halted
  unconditionally** (it must, for re-satisfaction of completed queries),
  re-executing over the half-unwound state: with enough choice-point
  structure the corrupted terms spun forever inside the `append` builtin
  (gdb-pinned: an inlined cons-building loop bump-allocating endlessly).
  This silently defeated the fail-fast diagnostics in exactly the
  compiles they were added for. Fixed with an explicit abort flag
  (`@wam_uncaught`): set on the uncaught path, checked at `@backtrack`
  entry (refuse resumption), cleared per top-level query by
  `@wam_catch_reset`. Regression `selfhost_uncaught_throw_aborts` (hung
  for minutes before; must exit nonzero immediately).
- *Reader lift:* **quoted functor applications** — `'$VAR'(N)` parsed as
  the atom `$VAR` and then failed on the paren. The quoted-atom lexer
  now peeks for an immediately-following `(` and routes into the shared
  argument parser (functor id phi: quoted or unquoted name span).
- *Compiler convention:* cgfull uses `'$VAR'(N)` as its numbervars
  marker, so a **source-level** `'$VAR'(Pattern)` (exactly what mid2's
  clause heads need to match numbervarred terms) was misread as a
  marker — after numbervars the pattern is `'$VAR'('$VAR'(0))` and
  `Y is 48 + N` crashed. The marker clauses in `head_arg_instrs`,
  `u_arg`, `c_operand`, and `s_arg` now guard on `integer(N)`; a
  `'$VAR'` structure with a non-integer argument compiles as an ordinary
  structure pattern (`get_structure '$VAR'/1` + inner marker), which is
  precisely the semantics a self-compiled middle needs. The residual
  ambiguity — a literal `'$VAR'(3)` in user source is still read as a
  marker — is inherent to the numbervars encoding and documented here.

**THE FRONT LANDED — the compiler compiles its clause-grouping and chains.**
The loaded cgfull compiled `fixpoint_front_source/1`: the middle plus
cgfull's whole front restated cut-free — facts vs rules (`mhb` tags facts
with body `true`), the generic table walk mirroring
`collect_tables`/`walk_term` (var/nil/integer/atom/compound dispatch with
the control-functor skip list), `group_clauses`/`take_same`,
`group_labels` + label lookup, and the `try_me_else`/`retry_me_else`/
`trust_me` chain builders with PC threading and Label-PC pair collection
(`keysort` + `pairs_values` close the PC table exactly like cgfull), plus
integer/atom head constants, predicate-call goals, and `=/2` goals. The
doubly-compiled compiler compiled a **multi-predicate program** — a rule
with arithmetic, a rule calling it, and a two-clause fact predicate with
a try/trust chain — **byte-identically to the production cgfull**
(checksum 13679). Test `selfhost_codegen_stage_d_front`.

*Runtime finding (campaign no. 11):* compiling the front source tripped
the **reader variable-dictionary cap**. The var-dict was a fixed
128-name block; past it, `wam_var_ref` fell back to a FRESH heap cell
per occurrence — repeated occurrences of the same variable name in later
clauses silently stopped sharing. The self-hosted compiler thus
**miscompiled its own serializer**: in `wzr`, the second occurrence of
the codes list variable became a fresh unbound, `append` copied nothing,
and every emitted instruction lost its opcode digits (the object stayed
structurally plausible — headers, tables, and PC rows intact — which
made the corruption look like a serializer bug rather than a reader
bug; the emitted-object diff of two near-identical sources pinned it:
`put_value Y54` in the good compile vs `put_variable Y55` in the bad
one). Fixed by making the dict growable (`[count][cap][pairs…]`,
doubling; entries are plain ids/addresses so relocating the block is
safe). Regression `reader_var_dict_grows_past_128` (130 names, repeated
variable in the last clause).

**THE WALKERS LANDED (loaded, minus deferral) — ITE codegen, guards,
builtins self-compiled.** The loaded cgfull compiled
`fixpoint_walkers_source/1`: the front plus the last codegen walkers —
**PC/label-threaded goal compilation with full ITE codegen** (`mite`
mirrors `f_goal`'s ITE clause: else/join labels, `cut_ite`, `jump`,
Label-PC pairs, init-set intersection), comparison guards (`mcmp`), the
builtin whitelist dispatch (`mbi`), general operands with list-literal
builds, and structure/list patterns with X-temp deferral on both sides.
The doubly-compiled compiler compiled a golden with an if-then-else +
comparison + atom operands, builtin calls with list-literal arguments,
and structure head patterns with repeated variables **byte-identically
to the production cgfull** (checksum 22412; test
`selfhost_codegen_stage_d_walkers`). The walkers' **complete** logic —
including the X-temp deferral cases (nested head pattern
`w(f(g(Z)), Z)`, nested expression build `(A + B) * 2`) — is proven
byte-exact by interpreting the same source as Prolog in SWI against
`cgfull_term/2` (test `selfhost_walkers_logic_in_swi`, checksum 33858).

*Runtime finding (campaign no. 12) — ROOT CAUSE ESTABLISHED, fix in
progress.* The deferral paths crash LOADED with a heap oob read at
exactly `heap_top`. A printf-instrumented `-O0` trace of the minimal
pair pinned the mechanism end to end:

1. A goal fails AFTER an earlier call in the same derivation succeeded
   via a NON-LAST clause of its chain (the walkers' `muarg` succeeded
   through its `'$VAR'` clause, leaving the chain's try_me_else CP
   live — correct Prolog re-satisfaction semantics).
2. The failure cascades backward INTO that stale CP (trace: `BT
   pc=529` — the alternative points at the predicate's next clause),
   restoring registers and rewinding the heap to the call-time marks.
3. The next clause RE-RUNS on the same term — and because the dispatch
   clauses OVERLAP (`compound('$VAR'(0))` is true, so the
   compound-defer clause also matches the numbervarred-variable term),
   execution diverges down a different compile path instead of failing
   through, and the divergent execution reads a register Ref above the
   rewound heap top (`TME ht=87` followed by a read of cell 87).

Two layers of fix, one landed conceptually and one still open:

- **Subset-level (validated on the repro):** make dispatch clauses
  MUTUALLY EXCLUSIVE — the compound clauses of
  `muarg`/`msarg`/`mopd`/`mharg` gain `functor(C, F, _), F \== '$VAR'`
  (and cons-exclusion) guards, so stale-CP re-entry fails cleanly and
  the cascade passes through to the right alternative. On the minimal
  pair this converts the crash into clean failure. SWI validates the
  guarded walker logic byte-exact. A residual loaded-vs-SWI divergence
  remains on the full deferral golden (clean failure loaded, success
  in SWI; a second crash signature on the nested-build golden), so the
  guarded walkers source is NOT yet landed — next round continues from
  the committed trace methodology.
- **Runtime-level (open):** even with divergence-free re-entry, the
  trace shows the re-executed path can observe a register Ref above the
  rewound heap top — the re-entry state reconstruction (CP register
  snapshot + heap mark + trail unwind + untrailed arena-slot writes) is
  not fully consistent somewhere. Pinning that inconsistency — with
  the same instrumented-trace technique, now proven effective — is the
  next round's opening move, and matters beyond the self-host: ANY
  loaded program that backtracks into a completed call's chain CP is
  exposed.

The architectural observation for the design record: the loaded
subset's "deterministic first solution" philosophy leaves chain CPs of
completed calls live, so real failures re-enter completed calls.
That is correct Prolog — SWI does the same and survives because its
re-entry state is consistent — so the philosophy stands; the runtime's
re-entry consistency is what must be fixed. First-argument indexing
(deliberately NOPed in the loader) would mask most instances but is
not the correctness fix.

**Remaining toward the full fixpoint:** every compiler stage is now
self-compiled — serializer, single-clause middle, grouping/label/chain
front, generic table walk, and the last walkers (ITE codegen, guards,
builtins; deferral logic SWI-proven). Left: fix runtime finding no. 12
(the ITE/deferral interaction above) so the deferral paths run loaded,
then stitch the slices into one `compile(SelfSource)` whose accepted
subset covers its own source — the capstone.

**Deliverable:** the demonstrable self-host — the compiler compiles itself,
and `compile(SelfSource)` yields a working compiler object.

## Risks and open questions

- **Operand encoding is the crux.** The register/argument packing (the
  `0x10000`-in-op2 convention) and the exact tag numbers must match what the
  loader expects. Mitigation: differential testing against golden objects
  from the host `write_wam_object/3`; Stage A exists precisely to lock this
  down before any codegen.
- **Register allocation in the subset.** The host compiler's allocator is
  elaborate. The bootstrap compiler needs only a *correct, unoptimised*
  allocator (sequential temporaries; correctness over register reuse) — the
  loaded-objects-run-unindexed philosophy (M1) applies to allocation too:
  correctness first, speed later.
- **Counter/symbol-table strategy.** Threaded accumulators keep the compiler
  purely functional and side-effect-free (easier to reason about, no store
  state to reset); the dynamic store is more convenient but introduces
  process-global state that must be cleared between compiles. **Recommend
  threaded accumulators** for the bootstrap compiler unless a construct
  forces otherwise.
- **`arg/3` constant-index gap** — resolved by the coding constraint above;
  optionally close it for real with the deferred opcode lift.
- **Reader coverage vs. the compiler's own source.** The self-host fixpoint
  (Stage D) requires the reader to parse every construct the compiler source
  uses. The reader is "complete for the canonical + operator surface" but has
  no escape handling in quoted atoms and other edges — Stage D will surface
  the exact reader gaps, each a small extension.
- **Object size / compiler-object load cost.** Even the minimal compiler is a
  larger `.wamo` than the grammars it compiles; `@wam_object_load_cached`
  (M5) already memoises it so the cost is paid once (the `DYNCACHE` role).
  Pay-for-what-you-use holds: it loads only when an `eval`/`compile` surface
  is compiled in.
- **Scope discipline.** The temptation is to grow the bootstrap compiler
  toward the full host compiler. It should stay minimal — its job is to
  *demonstrate the loop closes*, not to replace the AOT path. The full
  compiler stays AOT; the bootstrap compiler is the JIT proof.

## Relationship to the deferred small items

Several deferred items (from the dynamic-DB / eval work) are natural
companions but **not prerequisites** for Stage A/B:

- `arg/3` constant-index subset lift — closes the one real gap cleanly
  (otherwise handled by coding constraint).
- Cut (`!`) and cross-goal backtracking in rule bodies — only needed if the
  bootstrap compiler emits/uses them; the minimal subset avoids both.
- `call/N` partial application in the store — only if the compiler builds
  goals dynamically rather than emitting `call`/`execute` directly.

The recommended entry point is **Stage A** (the serializer): it is the
highest-risk, most-diffable piece, unblocks everything after it, and needs no
new runtime work — only a grammar and a golden-object test harness.
