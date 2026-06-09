<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 John William Creighton (s243a)
-->

# plawk — Philosophy

*Project: UnifyWeaver — Hybrid WAM/LLVM Target*
*Author: John Creighton*
*Status: Early Design / Prototype Phase*

> **Companion docs:** [Specification](PLAWK_SPECIFICATION.md) ·
> [Implementation Plan](PLAWK_IMPLEMENTATION_PLAN.md) ·
> [Submodule README](../../examples/plawk/README.md)

---

## 1. Core motivation

Unix shell tools — particularly `awk` and `bash` — have endured for decades
because they offer a simple, composable model: read records from a stream,
apply a pattern-action rule, write results. This model is powerful precisely
because it is minimal.

The goal of this DSL is to preserve that ergonomic simplicity while replacing
the interpreted, string-centric execution model with something rigorous:
compiled Prolog, executed via UnifyWeaver's hybrid WAM/LLVM backend, operating
on binary/typed data structures.

The guiding principle is: **familiar surface, principled interior**.

From the outside, a user writes something that looks and feels like `awk`. Under
the hood, the program is a deterministic Prolog predicate, compiled through
UnifyWeaver to native LLVM code, using binary record structures rather than text
strings.

### 1.1 The performance argument, stated precisely

awk cannot compete with Rust/Haskell/F#/Go for real data-processing work (e.g.
the graph algorithms UnifyWeaver already compiles) for **three** reasons, not one:

1. **Per-field string cost.** Every field is a heap string; `$3 + 1` reparses
   text to a number on every access.
2. **String-keyed arrays.** awk's associative arrays are string-keyed hash
   tables.
3. **Inter-stage serialization.** A pipeline `awk | sort | awk` re-serializes
   every record to text at each `|`.

The WAM addresses (1) and (2) for free: integers stay tagged integers, compound
terms are structure-shared, and keys are terms rather than strings. The LLVM
target's existing machinery (`musttail` TCO, BFS worklist, memo tables) already
lowers the hot patterns to unboxed, iterative native code.

**(3) is the real differentiator** — larger than "binary vs string." Because the
DSL and UnifyWeaver's graph algorithms compile to the *same* engine, a pipeline
like *parse records → build graph → reachability → emit* runs in **one native
binary with no text boundary between stages**. A Unix pipeline cannot do this; a
hand-written Rust program can, but you would write it by hand. The niche is:
awk-level ergonomics, no serialization boundaries, and the ability to drop into
graph reachability mid-stream.

**Honest scope.** This will not beat hand-tuned Rust on raw single-pass
throughput in the near term — boxed WAM cells and residual choicepoints cost
real time. Closing that gap depends on the determinism story below actually
landing in codegen (see the Specification and the reconciliation findings in the
Implementation Plan).

---

## 2. Philosophy of determinism

Standard Prolog is relational and supports backtracking. This is powerful for
reasoning but is an obstacle when compiling to efficient, LLVM-native control
flow. The DSL adopts a **determinism-first stance**:

- Each handler predicate has a declared primary *mode* (`+` inputs, `-` outputs).
- Predicates intended to succeed exactly once are written to do so and leave no
  choicepoints.
- The compiler uses mode information to generate straighter, less-backtracking
  code where possible.
- Backtracking is not removed from Prolog; it is *contained and annotated*.

This mirrors the relationship between Haskell's `IO` monad and pure functions:
backtracking is possible, but it is the exception and must be made explicit.

> **Grounding (see Implementation Plan § Codebase reconciliation).** Mode
> declarations are **already consumed** by UnifyWeaver's codegen: `:- mode`
> flows through `demand_analysis` and `binding_state_analysis` into the WAM
> pipeline, which picks deterministic builtin variants and indexes on input
> positions. A `:- det` *declaration*, by contrast, is **not** consumed today;
> determinism is currently achieved structurally (cut, if-then-else, switch
> indexing, `musttail`). The DSL therefore leans on **modes + structural
> determinism** first, and treats a `det`-directive-driven choicepoint-elision
> pass as a later, measurement-justified addition.

---

## 3. Philosophy of stream abstraction

`awk` hardcodes its stream model: one record per line, fields split by `FS`,
output to stdout. This DSL abstracts that model into three decoupled roles:

- A **Reader** abstracts "how to obtain the next item from an input stream." It
  may read lines, binary records, parsed Prolog terms, or data from a DCG grammar.
- A **Writer** abstracts "how to emit a result." It may write to stdout, a named
  pipe, a binary file, or another stream held in `State`.
- The **Handler** (`{}` body) dispatches between writers or passes control to a
  next stage.

A program written in the DSL is therefore not tied to text I/O. The same
pattern-action logic can process binary network packets, structured log records,
or Prolog terms.

---

## 4. Philosophy of staged development

Complexity is introduced in layers, each stabilized before the next is added:

1. **Core Prolog layer** — `process_all/4`, `reader`, `handler`, `writer`,
   `item_field/3`, mode declarations; validated in plain SWI-Prolog.
2. **UnifyWeaver compilation** — transpile the core to LLVM via the hybrid WAM
   target.
3. **AWK syntactic sugar** — a front-end parser emits Prolog core from awk-like
   `pattern { body }` syntax.
4. **Bash compatibility** — file descriptors, redirection, subprocesses, named
   pipes (much of which already exists as LLVM builtins; see Specification §10).

The DSL syntax is sugar; **the Prolog core is the specification.**

---

## 5. Relationship to the existing AWK target

The name **`plawk`** = "**Prolog awk**": an awk-like surface compiled through
Prolog → WAM → LLVM. It also disambiguates from the existing **AWK target**
(`docs/AWK_TARGET_STATUS.md`, `src/unifyweaver/targets/awk_target.pl`). That
target compiles Prolog *into* awk scripts — the opposite data-flow direction
from `plawk`, which compiles an awk-like surface *down to* Prolog→LLVM. They are
complementary, not competing: one emits awk for portability, the other consumes
an awk-like surface for native performance.
