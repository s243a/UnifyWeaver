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
