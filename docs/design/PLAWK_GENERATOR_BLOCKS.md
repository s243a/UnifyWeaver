<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk generator blocks — `gen { … } as name` (design)

**Status**: design sketch, not yet implemented. This is the **producer dual**
of the `over query(Goal)` reader (`PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md`,
phase 6). Where the query reader lets a **Prolog goal drive a plawk pass**
(Prolog → plawk records), a generator block lets a **plawk `{}` block be called
by a Prolog goal** (plawk → Prolog solutions). The two are complementary and
compose — a generator block feeds a query pass.

## 1. Motivation

The query reader consumes solutions; nothing yet lets plawk *produce* a
relation for Prolog to consume. A generator block closes that loop:

```awk
gen { emit 1; emit 2; emit 3 } as small          # defines small/1

pass over query(small(X)) { print $1 }            # 1 / 2 / 3
```

`gen { … emit E … } as name` runs a `{}` block that emits values; naming it
exposes a predicate `name(X)` reachable from any goal — including a query
reader's — so a producer written in plawk and a consumer written as a query
compose cleanly.

### Why a block, not a lazy function

AWK users read `{}` as *streaming work* and functions as *eager
call-and-return*. Making a **function** lazy would violate that intuition;
a `{}` block that emits is already the AWK mental model for producing a stream.
So streaming is driven from **block syntax**, not from function laziness. This
keeps plawk's surface honest: `function f(x) { return … }` stays eager, and the
new streaming construct is visibly a block.

## 2. Surface

```
gen { BODY } as name                     # pure producer (no input)
gen over SOURCE as v { BODY } as name     # optional input iterator (see §4)
```

- **`BODY`** is an action block that may call `emit E` (one value) to
  contribute a solution. Ordinary plawk actions (assignments, arithmetic,
  scalars) are allowed; the block runs to completion.
- **`as name`** binds the generator to a predicate `name/1` (or `name/k` for a
  tuple `emit`, §5). The name lives in the same predicate universe `dyncall` /
  a query goal reach.
- **`emit E`** is the producer counterpart of `print` — instead of writing a
  record to stdout, it contributes `E` to the relation's solution set.

### 2.1 Explicit `emit` vs implicit yield via the field separator

A generator has to signal "here is one solution." Two options:

- **Explicit `emit E`** — a dedicated statement, one call per solution.
- **Implicit via FS** — the block's `print` output *is* the stream: each printed
  record becomes a solution, split into a tuple by `FS`/`RS` (maximally AWK, no
  new keyword).

**Decision: explicit `emit` is the default; implicit-FS is opt-in sugar.** The
deciding factor is **typing**. A generator feeds Prolog goals, which consume
*ground typed terms* — and the query reader (the consumer dual) now carries
per-column integer/string kinds end to end (the tagged materialisation of
`PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md` PR 6). If a generator yielded via
`print` + `FS`, every value would round-trip through text and arrive as a
string, throwing away exactly that typing — the consumer would have to re-parse.
`emit E` keeps the term's type (an integer stays an integer, an atom an atom),
mirroring how the reader materialises columns. Three more reasons:

- **Emission ≠ stdout.** Overloading `print` to *also* mean "produce a solution"
  conflates two jobs — a gen block may legitimately want to write to stdout
  (debug/log) without that becoming a solution. `emit` keeps them separate.
- **Arity is explicit.** `emit (A, B)` says arity 2 (`name/2`); an FS-split line
  makes arity depend on the runtime field count, which is fragile.
- **No separator/buffering ambiguity.** `emit` is one solution per call; there
  is no "was that a partial record?" question, and `FS`/`RS` keep their meaning
  as **input** parsing, not emission.

**Implicit-FS as opt-in.** The FS idea is genuinely convenient for the "each
line of text → a record" case, so keep it — as an explicit opt-in, not the
default. A gen block that emits whole text lines can request FS-splitting into a
tuple (spelling TBD, e.g. `gen ... as name split FS`), with the understanding
that those fields are text-typed. That preserves the AWK ergonomics for the
text case without making the default lossy. (Symmetrically, splitting is really
a *consumer* concern — a query reader could offer `split by FS` on a text
column — so this may land on the reader side instead; noted, not yet decided.)

## 3. Semantics: materialise, don't stream

A generator block **does not** retain a live choicepoint across `emit`s. That
would be the "residual choicepoint in the hot loop" the determinism model
(`PLAWK_MULTIPASS_CACHE.md` §1) forbids — it would pin the trail/heap and break
constant-memory streaming. Instead the block **runs to completion, collecting
each `emit E` into a list**, and the generated predicate exposes that list as an
ordinary non-deterministic relation:

```
name(X) :- member(X, Collected).
```

This is the **bounded-multiplicity principle**
(`UNIFYWEAVER_LANGUAGE_PRINCIPLES.md`, Principle 1) run in the **producer**
direction: the block's multiplicity is collapsed to a materialised set at
definition time, and consumers (the `findall` inside a query pass) iterate it
deterministically. It is the exact mirror of query-reader PR 2 — where the query
reader **reads** a `findall` list into a table, a generator block **writes** an
emitted list into one and wraps it as a callable relation. The materialisation
runtime is shared.

## 4. The input iterator is optional

A generator can be a **pure producer** (no input) or **optionally** consume an
input iterable, transforming it:

```
gen { emit 1; emit 2 } as small                          # no input
gen over nums as n { emit n * n } as squares             # input: a table/array
gen over query(edge(A)) as a { emit a } as edge_ids       # input: another goal
```

- **Optional by default.** The bare `gen { … } as name` form takes no input and
  is the common case (a fixed or computed set). The `gen over SOURCE as v { … }`
  form is opt-in.
- **What a SOURCE can be.** Any already-materialisable iterable: an assoc/array
  table, a DB column (phase 5 store), or another relation via `over query(…)`
  (so generators chain). Each drives `BODY` once per element, with `v` bound to
  the element; the block may `emit` zero or more values per element (a
  filter/flat-map). Because both the input set and the emitted set are
  materialised, no lazy pipeline is created — each stage collapses at its
  boundary.
- **Iterating an iterable is a distinct, orthogonal producer.** One could also
  imagine `for v in SOURCE` as a bare loop; the point of folding it into `gen`
  is that the block form reads as AWK *and* yields a callable relation. The
  input iterator is the bridge, not the primary feature — hence optional.

## 5. Modes and arity

- **First cut: all-output, arity 1.** `emit E` where `E` is an integer;
  `name/1` is a pure generator. This maps directly onto the query reader's
  per-column integer materialisation (PR 3) run backwards.
- **Tuple `emit`.** `emit (A, B)` → `name/2`, materialised per column exactly
  as the query reader consumes columns (PR 3). Symmetry keeps the runtime one
  mechanism used in both directions.
- **Input binding (modes).** With an input iterator, `v` is bound (an input) and
  `emit` produces outputs — the generated `name(X)` is still a pure generator to
  its Prolog caller. Passing *arguments* into `name` (partial application /
  input modes on the generated predicate) connects to the query reader's
  "inputs bound from the record" follow-on and the §3.10 `X in domain` producer;
  it is a later refinement.

## 6. Composition with the query reader

Generators and query readers are producer/consumer duals over the same
boundary, so they nest:

```
gen { emit 10; emit 20; emit 30 } as weights

pass over query(weights(W)) { if ($1 > 10) print $1 }     # 20 / 30
```

The generator materialises its set at definition; the query pass runs `findall`
over `weights/1` and iterates — both collapses are bounded-multiplicity, so the
composed program stays deterministic (§1 intact) with snapshot semantics at each
boundary.

## 7. Runtime (reuses the query-reader machinery)

- **Collection.** The `gen` block compiles to a function that runs `BODY`,
  routing each `emit E` into a growing list (the same cons-cell construction the
  `findall` path already builds). With an input iterator, the block is wrapped
  in the existing table/goal iteration (query-reader PR 2/3 or `over TABLE`).
- **Exposure.** The collected list backs `name(X) :- member(X, Collected)` —
  a non-deterministic relation over a fixed set. `member/2` (or an equivalent
  compiled fact table) is the enumeration primitive; a query pass's `findall`
  then re-collects it, mirroring PR 2.
- **Durability (optional).** A generator's collected set is a materialised
  table; a durable generator could back it with a cache / LMDB store (phase 5),
  so the set is computed once and reused across runs — the producer counterpart
  of the query reader's `use STORE` sources.

## 8. Open questions

- **When the block runs.** Once at program start (definition time), or lazily on
  first call, or per call when it takes input arguments. First cut: once at
  start, before any pass that consumes it.
- **`emit` value types.** Integers first (matching PR 2/3 columns); atoms /
  strings via the `posarray_str` path (shared with the query reader's string-
  column follow-on); tuples for arity > 1.
- **Ordering & duplicates.** `member/2` over the collected list preserves emit
  order and multiplicity — consistent with the query reader's ordered,
  multiplicity-preserving iteration. Whether a generator may be declared a *set*
  (dedup on emit) is a later option.
- **Interaction with writes.** A generator that reads a table an ordinary pass
  writes wants the same snapshot boundary the query reader documents; the
  materialise-at-definition rule gives it for free.

## 9. Sequencing

Best built **after** the query reader's remaining PRs (mixed passes, string
columns, snapshot test): it shares their materialise-then-iterate runtime and
the per-column mechanism, so it is largely a matter of running that machinery in
the producer direction plus the `emit` collection. A rough PR shape:

1. **Surface + AST** — parse `gen { BODY } as name` (and the optional
   `gen over SOURCE as v { BODY }`) to a `gen_block` term; `emit` action; a
   clean not-yet error until the runtime lands.
2. **Collection runtime** — compile the block to an emit-collecting function;
   expose `name/1` as `member` over the collected list; verify a pure arity-1
   generator feeds a query pass.
3. **Optional input iterator** — `gen over SOURCE as v { … }` for a table /
   `over query` source (flat-map / filter).
4. **Tuples + durability** — `emit (A, B)` → `name/2`; optional cache/LMDB-backed
   collected set.
