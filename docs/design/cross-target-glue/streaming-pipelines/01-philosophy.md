# Streaming Data Pipelines — Philosophy

Application-level design on top of the cross-target glue system
(phases 1–5 in the sibling docs). Covers the pattern where a
predicate in one target produces a stream of records that a
predicate in another target consumes — typical shape for
preprocessing pipelines, ETL, and cross-language data handoffs.

## Motivating use case

The immediate driver: parsing a MediaWiki SQL dump (gzipped,
multi-GB) into LMDB for the WAM Haskell scaling benchmarks.

```
[gzipped MySQL dump] → [Rust parser: stream of (child, parent)] → [Python: LMDB ingest]
```

The parser is byte-level state-machine work (MySQL INSERT VALUES
tuples with `\'` / `\\` escape rules) — not something Prolog is
well-suited to author directly. But the *composition* of parser
→ filter → ingest is exactly what UnifyWeaver describes well.

## Three claims

### Claim 1: The Prolog is invariant across transport phases

The same high-level description should work whether the stream is
text over a pipe, binary over a pipe, or an in-process iterator
API. In Prolog terms:

```prolog
:- declare_target(parse_category_links/3, rust, [streaming(true)]).
:- declare_target(ingest_to_lmdb/3, python, [streaming(true)]).

process_dump(DumpPath, LmdbPath) :-
    forall(
        parse_category_links(DumpPath, Child, Parent),
        ingest_to_lmdb(LmdbPath, Child, Parent)
    ).
```

This same code compiles under every transport in the evolution
below. Only the generated glue changes.

### Claim 2: Streaming is a fold-over-side-effects, not an aggregation

`aggregate_all(Reducer, Gen, Result)` materializes all solutions
of `Gen` before reducing — fine for small sets, fatal at 20 GB of
input. `forall(Gen, Action)` is the right pattern: for each
solution of `Gen`, run `Action` for side effects, retain nothing.

The generator backtracks through rows as the stream advances. The
transpiler maps this to a `for row in iter { ... }` loop, not to a
`collect().into_iter()` followed by a reduce.

The purity story: the generator is pure over a stream
(order-independence up to the stream order), and the action is
the side-effectful sink. The certificate system already
distinguishes these — we just lean on the existing analysis.

### Claim 3: Fast tokenization stays in the target's native library

Byte-level tokenization, gzip decoding, string escape handling —
these are one-time-written primitives in the target language (Rust
for speed, Python for ergonomics). They're declared as leaf
predicates, *not* transpiled.

Prolog contributes at the layer above: filtering, column
destructuring, joining streams, handing off to another target.

The rule: if a predicate's body would be a byte-level state
machine, it's a leaf primitive. If it's compositional logic over
typed records, it's transpiled.

This is the same rule the WAM Haskell target already follows for
LMDB: `lmdbRawEdgeLookup` is a hand-written Haskell function, not
transpiled Prolog. The `EdgeLookup = Int -> [Int]` type is the
contract; everything above it is composed in Prolog.

## Three-phase transport evolution

Same Prolog, different glue. Each phase is a separate glue
template; upgrading is a flag flip, not a refactor.

### Phase 1 — Text IO over pipe

Simplest. The Rust producer `println!`s `"{child}\t{parent}\n"`
per tuple; the Python consumer reads stdin line by line and
parses.

- Throughput: ~50 MB/s realistic (stringify + parse dominate)
- Complexity: minimal; uses existing `pipe_glue.pl` templates
- Cost per tuple: ~20-40 bytes of text + parse overhead

**Appropriate for: one-off preprocessing steps** where total
runtime is minutes, not seconds. Enwiki dump parse falls here —
runs once per dump release.

### Phase 1.5 — Binary IO over pipe

Same pipe, no stringification. Rust writes packed little-endian
bytes (e.g., `[child:i32][parent:i32]` = 8 bytes per tuple);
Python reads with `struct.unpack` or a NumPy `fromfile`.

- Throughput: ~200-500 MB/s (pipe-IO bound)
- Complexity: frame-type declaration (`streaming(true, binary(i32_pair))`)
- Cost per tuple: 8 bytes, one buffered read

**Appropriate for: repeated streaming workloads** where the cost
of Phase 1 becomes noticeable. This is the LMDB CBOR→raw lesson
applied to pipes: the speed-up is mostly eliminating per-record
text parsing, not reducing IPC cost.

### Phase 2 — API / in-process object stream

Shared address space. The Rust producer is loaded as a pyo3
module (or equivalent for the consumer target); the Prolog-
declared generator becomes a Python iterator. Types are passed
as language-native objects, not bytes.

- Throughput: function-call bound, typically 1-10 ns/tuple
- Complexity: pyo3/maturin build integration, shared allocator,
  lifetime rules for streamed handles
- Cost per tuple: one function call + a few field copies

**Appropriate for: in-process composition** where the producer
and consumer are tightly coupled, or when you want to pass
handles (open LMDB transactions, memory-mapped regions) rather
than their serialized shadows.

This is the phase that unlocks genuinely new composition
patterns — not just faster data flow but passing *capabilities*
across language lines.

## Target choice is (mostly) orthogonal in Phase 1

A consequence of Claim 1 (Prolog invariant across transports) and
Claim 3 (fast tokenization stays native): in Phase 1, *any target
that can read bytes and write text lines* works. The declaration
is decoupled from the mechanism.

This is worth saying explicitly because it suggests a staged
target-selection strategy.

### FFI-capable targets (preferred for long-term)

Languages that emit a shared library with a C ABI are the
eventual Phase 2 candidates:

| Target | FFI story | Python bindings |
|--------|-----------|-----------------|
| C | Native; the original FFI language | `ctypes` / `cffi` trivial |
| Rust | `extern "C"` + `pyo3` for Python-specific ergonomics | Excellent (`maturin`) |
| Haskell | `foreign export ccall`; needs RTS init | Workable but awkward |
| Go | `cgo`; similar RTS-init concerns to Haskell | Possible but rough |

If we start Phase 1 with one of these, the Prolog stays
unchanged when we later flip to Phase 2 — we're just adding a
different glue template for the same target. No re-implementation.

### Non-FFI targets (fine for Phase 1, dead-ends for Phase 2)

Languages that are interpreted or whose native runtime doesn't
support embedding as a shared library:

| Target | Phase 1 fit | Phase 2 path |
|--------|-------------|--------------|
| AWK | Excellent for MySQL INSERT parsing (sed/awk combos work today) | Requires re-port to an FFI-capable target |
| Python | Works but defeats the purpose (why shell out to itself?) | — |
| Shell scripts | Works for simple pipelines | Requires re-port |

AWK is a legitimately good Phase 1 choice for the MySQL dump
parser: `mysqldump-to-csv` already demonstrates that an
awk-based parser handles the escape rules correctly, and the
existing `shell_glue.pl` infrastructure would wire it up with
almost no new code. The catch: if Phase 2 performance ever
becomes interesting, the parser has to be rewritten in a native
target.

### Practical rule

Pick the target for Phase 1 based on who's authoring the
primitive tokenizer:

- **If we're writing the tokenizer from scratch**: use a
  FFI-capable target (Rust or Haskell) so the Phase 2 upgrade
  is a glue-template addition, not a re-port.
- **If we can reuse an existing tool** (e.g., `mysqldump-to-csv`
  exists as sed/awk): wrap it as an AWK or shell target in
  Phase 1; accept that Phase 2 would need a re-implementation if
  we ever pursue it.

For the enwiki parser specifically: Haskell is attractive because
`attoparsec` gives us high-speed byte parsing out of the box and
the codebase already has a working Haskell toolchain. Rust is
attractive because `pyo3`/`maturin` is the most polished Phase 2
path. AWK is attractive because it requires essentially zero new
infrastructure and the data volumes are fine for Phase 1.

Any of the three is defensible. The Prolog layer doesn't care.

## What the glue layer does (the invariant)

Across all three phases, the glue is responsible for:

1. Reading the `declare_target` annotations on producer and
   consumer predicates
2. Compiling the producer into a target-appropriate artifact
   (binary for Phase 1/1.5, shared library for Phase 2)
3. Generating a call-site wrapper in the consumer's target that
   knows how to invoke the producer and iterate its output
4. Wiring the data format appropriate to the phase (text lines,
   packed binary, native objects)

What it does *not* do:

- Interpret the predicate bodies. Those are handed to the
  individual targets' codegen.
- Impose a schema language beyond what the types declare.
  `streaming(true, binary(i32_pair))` is sufficient; we don't
  need protobuf IDL for this pattern.

## Relationship to existing work

- `native_glue.pl` already handles Phase 1 for Go and Rust
  binaries (pipe-compatible `main()`, cargo build orchestration).
  Streaming data pipelines are a specific *use* of this — this
  doc describes the pattern, `native_glue.pl` provides the
  mechanism.
- `rpyc_glue.pl` / `janus_glue.pl` are steps toward Phase 2 via
  RPC/embedded interpreter. True in-process pyo3-style binding
  is not yet built.
- `pipe_glue.pl` templates handle text-over-pipe today. Binary
  framing (Phase 1.5) is a natural addition: same template,
  different framer.

## Design principles

1. **Phase 1 is the default** — text IO works everywhere, needs
   no extra build steps, and is correct-by-construction. Most
   preprocessing workloads stop here.
2. **Upgrade by flag, not refactor** — changing transport should
   never require rewriting the Prolog. If the Phase 1 predicates
   need to be rewritten to become Phase 2 predicates, the
   abstraction has failed.
3. **Leaf primitives over transpiled parsers** — byte-level work
   belongs in hand-written target code, not transpiled Prolog.
   Prolog shines at composition.
4. **Streaming defaults to no-state** — `forall` is the right
   primitive. Aggregation (`aggregate_all`, `sum`, etc.) is the
   opt-in case, not the default.
5. **Preprocessing doesn't need to be fast** — these pipelines
   run once per dataset update. Phase 1 is usually enough;
   don't over-engineer.

## Open questions

- How should cross-target atom interning work? If the Rust side
  parses strings but the Python side needs interned Int32 IDs,
  where does the intern table live and how is it shared across
  the pipe? (Candidate: the consumer owns the intern table, the
  producer emits strings, the consumer interns on ingest.)
- Backpressure. In Phase 1 the OS pipe buffer handles it for
  free. In Phase 2 we need explicit flow control.
- Error propagation. A parse error in the Rust producer should
  surface as a sensible exception in the Python consumer, not a
  broken pipe EPIPE.
- Observability. How do we instrument tuple counts, bytes/sec,
  progress for long-running preprocessing jobs?

## Companion documents (planned)

- `02-specification.md` — `declare_target` option grammar for
  streaming, exact contracts per phase, glue-template surface
- `03-implementation-plan.md` — phased rollout starting with
  binary framing on existing `pipe_glue.pl`, enwiki parser as
  the reference case
- `04-use-cases.md` — MySQL dump parser, CSV bulk load, log
  tailing, any other streaming preprocessors
