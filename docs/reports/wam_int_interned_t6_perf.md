# T6 first-argument indexing for the int-interned WAM targets

The atom-keyed targets (Rust/C++/F#/Go) compare the first-argument discriminator
as a **string**, so a string switch/map is an obvious win. The remaining T5
targets — **Lua, Haskell, Scala, LLVM** — are **int-interned**: an atom becomes
an integer id at codegen, so the T5 dispatch is an *integer*-equality chain. The
open question (the "lost to the compiler" risk) was whether the host compiler
already turns that chain into a jump table, making an explicit T6 switch
redundant.

This report records the triage micro-benchmarks that answered it, and the
verdict per target. Each measurement is the average dispatch cost over **all N
keys plus one miss** (the T5 chain's worst case is the last clause / the miss).

## Method

For each target, two functions are generated for a predicate with **N** distinct
single-atom clauses and timed in a tight loop:

- **T5** — the linear discriminator chain (`v == Atom n` / `t5a1.id == n` / …).
- **T6** — a native indexed dispatch on the interned id (host `switch` /
  `case` / `match` / hash table).

Tools: `lua` (5.4), `ghc -O2`, `scalac` + `java`, `clang -O2` / `opt`.

## Results (ns/op, lower is better)

| Target  | N=8        | N=64        | N=256        | T6 form |
|---------|-----------:|------------:|-------------:|---------|
| **Lua**     | 1.7× | 8.2×  | 29.7× | hash table of per-clause closures (built once) |
| **Haskell** | 1.4× | 2.5×  | 4.5×  | `case` on the interned id (GHC jump table) |
| **Scala**   | 1.3× | 3.1×  | ≫ (alloc/JIT-inflated) | `match` on the id → JVM `tableswitch` |
| **LLVM**    |  —   |  —    |  —    | **declined** — see below |

(Lua: 102.7→59.6, 482→58.8, 1822→61.3 ns. Haskell: 10.4→7.6, 24.9→9.9, 85.3→19.0.
Scala: 9.08→7.12, 24.7→8.02, 6619→18.0 — the case-class `==` cascade both
allocates an `Atom(n)` per comparison and grows linearly, and the giant 256-way
method is past HotSpot's inlining/compile thresholds; the `tableswitch` is O(1).)

## Verdict

- **Lua, Haskell, Scala — implemented** (gated T6, default `t6_min_clauses=8`).
  All three show a real, growing win; none is "lost to the compiler":
  - **Lua** is interpreted, so there is no optimiser to recover the table — the
    linear chain stays linear. Biggest win.
  - **Haskell**'s guard chain is only partly optimised; GHC compiles a dense
    `case` on `Int#` to a jump table, so the explicit `case` still wins.
  - **Scala**'s `==` cascade allocates per comparison and is not rewritten to a
    `tableswitch`; the explicit `match` is.
- **LLVM — declined.** At `-O2`, clang/LLVM's SimplifyCFG already converts an
  int-equality if-chain into a `switch`: the if-chain and an explicit switch
  compile to **identical** assembly. An explicit T6 switch in the generated IR
  would be redundant — the genuine "lost to the compiler" case the matrix warned
  about. (If the LLVM target ever emits unoptimised IR for the lowered path, T6
  would help there; under the standard optimised pipeline it does not.)

So **3 of the 4 int-interned targets keep a gated T6; LLVM declines on
benchmark evidence.** Combined with the atom-keyed set (Rust/C++/F#/Go), every
T5 target now either has a gated T6 or has a benchmark-backed reason not to.
