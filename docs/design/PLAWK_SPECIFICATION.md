<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 John William Creighton (s243a)
-->

# plawk — Specification

*Project: UnifyWeaver — Hybrid WAM/LLVM Target*
*Author: John Creighton*
*Status: Early Design / Prototype Phase*

> **Companion docs:** [Philosophy](PLAWK_PHILOSOPHY.md) ·
> [Implementation Plan](PLAWK_IMPLEMENTATION_PLAN.md) ·
> [Submodule README](../../examples/plawk/README.md)

---

## 1. Core execution model

The DSL execution model is a deterministic loop over an input stream:

```prolog
:- mode process_all(+Reader, +Handler, +State0, -StateN).
% Intended determinism: det (succeeds once, no choicepoints).

process_all(Reader, Handler, State0, StateN) :-
    call(Reader, Item, State0, State1),
    (   Item == end_of_file
    ->  StateN = State1                       % end of stream
    ;   call(Handler, Item, State1, State2, Continue),
        (   Continue == yes
        ->  process_all(Reader, Handler, State2, StateN)
        ;   StateN = State2                   % break
        )
    ).
```

**Invariants:**

- `Reader` is `det` in mode `(+State0, -Item, -State1)`: it always succeeds and
  yields either the next item or the atom `end_of_file`.
- `Handler` is `det` in mode `(+Item, +State1, -State2, -Continue)`: succeeds
  exactly once, returning updated state and a continuation flag.
- `Continue` is `yes` (proceed) or `no` (break).
- `State` is a single opaque term (record/struct) threaded through every call.

> **Design change from the original draft.** The original `process_all/4` made
> the Reader **semidet** and used `( call(Reader,…) -> … ; end )`, so *any*
> Reader failure — including a mid-stream parse error — looked like clean EOF.
> This spec makes the Reader **det** and signals end-of-stream with an explicit
> `Item = end_of_file` (mirroring Prolog's own `read/1`). Genuine errors can then
> surface as `Item = error(Reason)` instead of being silently swallowed. The
> tail-recursive call is in last-call position so a determinism-preserving
> backend can apply TCO (`musttail`) and compile the loop without stack growth.

---

## 2. State representation

State is a single Prolog term treated as a record:

```prolog
% state/N functor — fields may evolve during design
state(
    InputStreams,   % map/array of open input stream handles
    OutputStreams,  % map/array of open output stream handles
    Counter,        % current record number (NR equivalent)
    UserFields      % extensible slot for user-defined accumulators
).
```

- A single `state/N` functor keeps every predicate arity fixed; adding a field
  does not change `process_all/4` or any handler signature.
- State update is functional: `StateOut = state(IS, OS, Counter1, UF1)`. No
  destructive mutation at the Prolog level; the LLVM backend may map these to
  mutable struct fields.
- `nb_setval/2` and similar global side effects are reserved for integration
  points (caches, foreign state) and must not appear in handler bodies.

---

## 3. Item representation

An **Item** is the unit of data produced by the Reader for one iteration:

```prolog
record(text, Fields)          % Fields = [F1, F2, ..., FN]
record(binary, Type, Payload) % a typed record parsed from a binary stream
record(term, Term)            % a full Prolog term read by the runtime parser
end_of_file                   % sentinel: stream exhausted
error(Reason)                 % sentinel: reader-level failure
```

- `$N` lowers to `item_field(N, Item, Value)`.
- `NF` lowers to `item_field_count(Item, N)`.
- `NR` is held in `State`'s counter field, not in `Item`.

---

## 4. Reader predicate

```prolog
:- mode reader(+Stream, +StateIn, -Item, -StateOut).
% Intended determinism: det (yields end_of_file rather than failing at EOF).
```

Reader variants:

- **Text line reader** — reads one line from `Stream`, splits on `FS` (from
  `State`), returns `record(text, Fields)`.
- **Binary record reader** — reads a fixed-size or framed record, returns
  `record(binary, Type, Payload)`.
- **Term reader (runtime parser)** — reads one Prolog term, returns
  `record(term, Term)`; the compiled equivalent of `read/1`.
- **DCG-based reader** — applies a DCG grammar to the stream.

> **Implementation note (gap).** The LLVM target currently has **no streaming
> input builtin** — only whole-file `read_file_to_atom/2`. The text line reader
> therefore requires a new buffered builtin (`stream_open`/`read_line`/
> `stream_close`). See Implementation Plan § Codebase reconciliation for the
> sketch. A Phase-0 shortcut is to slurp the whole input and `split_string/4` on
> newlines, iterating in Prolog (correct, but not unbounded-stream-safe).

---

## 5. Writer predicate

```prolog
:- mode writer(+Stream, +Item, +StateIn, -StateOut).
% Intended determinism: det.

:- mode select_writer(+Item, -Writer).
% Intended determinism: det (exactly one matching clause per Item shape).

select_writer(record(text, _),       write_text_record).
select_writer(record(binary, T, _),  write_binary_record(T)).
select_writer(record(term, _),       write_term_record).
```

Inside a handler, a `print`/explicit write compiles to:

```prolog
select_writer(Item, Writer),
call(Writer, OutStream, Item, StateIn, StateOut).
```

---

## 6. Handler predicate

```prolog
:- mode handler(+Item, +StateIn, -StateOut, -Continue).
% Intended determinism: det.
```

The handler is the compiled form of the DSL `{ ... }` block. For:

```awk
$1 == "ERROR" { count++; print $0 }
```

the compiler generates:

```prolog
handler(Item, StateIn, StateOut, Continue) :-
    item_field(1, Item, F1),
    (   F1 == "ERROR"
    ->  StateIn  = state(IS, OS, NR, state_user(Count0, Rest)),
        Count1   is Count0 + 1,
        StateTmp = state(IS, OS, NR, state_user(Count1, Rest)),
        select_writer(Item, Writer),
        call(Writer, OS, Item, StateTmp, StateOut),
        Continue = yes
    ;   StateOut = StateIn,
        Continue = yes
    ).
```

Multiple pattern-action blocks compile to a sequence of guards within one
handler, or to a chain of guarded handler clauses (see §7 for why the clause
chain is preferred for `next`).

---

## 7. Break and next

- **`break`** → `Continue = no`.
- **`next`** → skip the remaining pattern-action pairs for this record, proceed
  to the next iteration.

> **Refinement from the original draft.** The original treated `next` as
> "`Continue = yes` with a skip-writers flag," which is vague. Two cleaner
> options: (a) make `Continue` a three-valued tag (`continue | next | break`);
> or (b) compile each pattern-action pair into its own guarded clause so `next`
> is simply "do not fall through to the remaining clauses." Option (b) is
> preferred — it keeps each clause deterministic and maps to structural control
> flow rather than a runtime flag.

---

## 8. Mode and determinism declarations

Every predicate the DSL exposes carries a mode declaration, and is *written* to
satisfy its intended determinism:

```prolog
:- mode process_all(+Reader, +Handler, +State, -State).
:- mode handler(+Item, +State, -State, -Continue).
:- mode reader(+Stream, +State, -Item, -State).
:- mode writer(+Stream, +Item, +State, -State).
:- mode select_writer(+Item, -Writer).
:- mode item_field(+N, +Item, -Value).
```

These serve two purposes:

1. **Codegen (today).** `:- mode` is read by `demand_analysis:read_mode_declaration/3`,
   feeds `binding_state_analysis`, and lets the WAM/LLVM pipeline select
   deterministic builtin variants and index on input positions. This is an
   existing, load-bearing path.
2. **Documentation + future determinism checking.** Intended determinism is
   recorded in comments (`% Intended determinism: det`) and verified in Phase 0
   with SWI's determinism tooling. A machine-readable `:- det` directive that
   *drives* choicepoint elision is **not yet** part of UnifyWeaver and is a
   deliberate later step (see Implementation Plan).

---

## 9. AWK syntactic sugar (Phase 2)

| AWK Syntax | Prolog Core |
|---|---|
| `$N` | `item_field(N, Item, Value)` |
| `NF` | `item_field_count(Item, N)` |
| `NR` | field access on `State` counter |
| `FS = ":"` | set in `State` / `BEGIN` clause |
| `{ count++ }` | inline arithmetic on a `State` field |
| `{ print $0 }` | `select_writer(Item, W), call(W, ...)` |
| `{ next }` | guarded-clause fall-through (see §7) |
| `{ break }` (non-std) | `Continue = no` |
| `BEGIN { ... }` | initialization predicate before `process_all/4` |
| `END { ... }` | finalization predicate after `process_all/4` |
| `/regex/ { ... }` | pattern guard using a regex match primitive |

---

## 10. File descriptors and stream model (Phase 4)

Bash-like stream capabilities are modelled as handles stored in `State`:

- `State`'s `InputStreams`/`OutputStreams` are key→handle maps.
- Descriptors 0/1/2 are pre-populated.
- `open_stream(PathOrFD, Mode, Key, StateIn, StateOut)` adds a handle.
- Named pipes: `mkfifo_stream(Path, Key, StateIn, StateOut)`.
- Subprocess IPC: `spawn_process(Cmd, StdinKey, StdoutKey, StateIn, StateOut)`.

> **Grounding.** Much of this layer **already exists as LLVM builtins** in
> `wam_llvm_target.pl`: `shell/1,2`, `mkfifo/2`, `kill/2`, plus a full filesystem
> surface (`copy_file`, `rename_file`, `directory_files`, `realpath`, `symlink`,
> `path_join`, `working_directory`, `read_file_to_atom`, `write_atom_to_file`, …)
> and the POSIX id/limit/priority syscalls. Phase 4 is therefore largely
> *wiring State-held handles to existing builtins*, plus the streaming reader
> from §4 — not building an I/O layer from scratch. It should also reuse the
> existing `src/unifyweaver/glue/{shell,pipe,native,streaming}_glue.pl` rather
> than introduce a parallel mechanism.

At the DSL surface, bash-like redirection (`>`, `<`, `2>`, `|`) desugars into
these primitives.

---

## 11. Runtime parser

The runtime parser is a Prolog term reader compiled to LLVM — the compiled
equivalent of `read/1` / `read_term/3`.

> **Grounding.** UnifyWeaver already has a portable parser
> (`src/unifyweaver/core/prolog_term_parser.pl`) compiled to WAM, exposed through
> a capability table (`wam_runtime_parser_capability.pl`) with three modes:
> `off` / `native` / `compiled` (see `docs/WAM_RUNTIME_PARSER_STATUS.md`). The
> new work is **adding LLVM to that capability matrix**, not writing a term
> reader. Because `compiled` mode is heavyweight (it bundles ~45 parser
> predicates), the DSL defaults to `off`/`native` and only pulls `compiled` when
> `read_term`/JIT is actually used (Phase 5).

DCG grammars can be loaded at runtime and compiled through the WAM/LLVM path,
enabling runtime grammar extension (Phase 5).

---

## Appendix A: Key predicate summary

| Predicate | Mode | Intended Det | Role |
|---|---|---|---|
| `process_all/4` | `(+,+,+,-)` | det | Main deterministic loop |
| `reader/4` | `(+,+,-,-)` | det (yields `end_of_file`) | Produce next Item |
| `handler/4` | `(+,+,-,-)` | det | Compiled `{}` body |
| `writer/4` | `(+,+,+,-)` | det | Emit Item to output stream |
| `select_writer/2` | `(+,-)` | det | Type-dispatch to a Writer |
| `item_field/3` | `(+,+,-)` | det | `$N` field access |
| `item_field_count/2` | `(+,-)` | det | `NF` field count |
| `open_stream/5` | `(+,+,+,+,-)` | det | Open file/FD as named stream |
| `spawn_process/5` | `(+,+,+,+,-)` | det | Spawn subprocess with pipe FDs |
