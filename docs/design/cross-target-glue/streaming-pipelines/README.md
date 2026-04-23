# Streaming Data Pipelines

Design documents for a cross-target streaming pattern on top of
the existing cross-target glue infrastructure. Motivating use
case: parsing a MediaWiki SQL dump into LMDB for the WAM Haskell
scaling benchmarks.

## Status: design (pre-implementation)

This directory captures the design; none of it is implemented
yet. The existing infrastructure in `../` (cross-target-glue
Phases 1–5) provides the foundation — this is a specific
application pattern extending from there.

## Documents

| Document | Purpose |
|----------|---------|
| [01-philosophy.md](01-philosophy.md) | Three-phase transport evolution (text → binary → API); leaf-primitive rule; target-choice guidance |
| [02-specification.md](02-specification.md) | `declare_target` / `declare_connection` option grammar, TSV framing, multi-table handling, auto-selection, error semantics |
| [03-implementation-plan.md](03-implementation-plan.md) | Phased rollout starting with Phase S1 (text IO); concrete code snippets; deliverable checklists |
| [04-use-cases.md](04-use-cases.md) | MySQL dump parser in Rust + Haskell + AWK (multi-target demo); profiling matrix; future use cases |

## Key design principles

1. **The Prolog is invariant across transport phases.** Same
   `declare_target` + same `forall` composition works under
   text pipes, binary pipes, or API calls — only the glue
   template changes. Upgrading transport is a flag flip, not a
   rewrite.

2. **Streaming is `forall`, not `aggregate`.** Fold-over-side-
   effects, no materialization. `aggregate_all` is the opt-in
   stateful case.

3. **Fast tokenization stays in native code.** Leaf primitives
   (byte-level state machines, gzip decoders) are hand-written
   in the target language and bound via `leaf(true)` +
   `native_crate(X)`. Only the compositional layer is
   transpiled. Same pattern as WAM Haskell's `EdgeLookup`.

4. **TSV is the text framing.** Matches existing
   `pipe_glue.pl` templates and existing benchmark data shape
   (`category_parent.tsv`, `article_category.tsv`). No new
   format invented.

5. **Target choice in Phase 1 is (mostly) free.** Any target
   that can emit text lines works. FFI-capable targets (Rust,
   Haskell, C) preserve the upgrade path to Phase 2; AWK is
   fine for Phase 1 but a dead-end for higher-performance
   transports.

6. **Multi-table dumps**: separate predicates per table is the
   default; tagged-TSV single-pass is the performance upgrade;
   multi-stream output is deferred.

7. **Virtual files** unify multi-table streams with on-disk
   TSV benchmarks. A virtual file is a named, typed data stream
   that the glue may back with a real file, a named pipe, an
   in-memory buffer, or a fan-out tee — transparently to user
   code. Existing TSV benchmarks in `data/benchmark/` already
   fit this shape; streaming producers become the live case of
   the same abstraction. (See 02-specification.md §4.5.)

## Three phases of transport evolution

| Phase | Transport | Throughput | Complexity | Use case |
|-------|-----------|-----------:|-----------:|----------|
| S1 | Text pipe (TSV) | ~50 MB/s | Minimal | One-time preprocessing |
| S1.5 | Binary pipe (packed) | ~200-500 MB/s | + framing template | Repeated streaming |
| S2 | In-process API (pyo3) | Function-call bound | + pyo3/maturin build | Tight-loop composition |

## Relationship to existing infrastructure

What's already built:

- `target_mapping.pl` — `declare_target/3`, `declare_location/2`,
  `declare_connection/3`
- `target_registry.pl` — target metadata, default transports per
  location pair
- `native_glue.pl` — Rust/Go binary compilation, pipe-compatible
  `main()` templates, cargo management
- `pipe_glue.pl` — TSV reader/writer templates, pipeline
  orchestration
- `janus_glue.pl` — Prolog↔Python in-process (Phase 2 precedent)
- `rpyc_glue.pl` — Python RPC (remote Phase 2 precedent)

What this design adds:

- `streaming(true, FrameSpec)` option semantics
- TSV as the default Phase 1 framing; binary framers for Phase 1.5
- `leaf(true)` + `native_crate(X)` for hand-written parsers
- Multi-table handling (separate predicates / tagged TSV /
  multi-stream)
- Auto-selection extension to cover streaming-specific
  transports
- Optional observability hooks for profiling

## Motivating use case recap

Parse `enwiki-latest-categorylinks.sql.gz` (~3 GB compressed,
multi-GB uncompressed) into LMDB for use by WAM Haskell
benchmarks at million-edge scale — the projected IntMap/LMDB
crossover zone from the scaling-insights doc.

Prolog declaration shape (Phase 1, Rust target):

```prolog
:- declare_target(parse_category_subcats/3, rust,
                  [streaming(true, tsv),
                   leaf(true),
                   native_crate(mysql_stream)]).
:- declare_target(ingest_to_lmdb/3, python,
                  [streaming(true, tsv)]).

process_dump(DumpPath, LmdbPath) :-
    forall(
        parse_category_subcats(DumpPath, Child, Parent),
        ingest_to_lmdb(LmdbPath, Child, Parent)
    ).
```

The same declaration shape works with Haskell or AWK producers
(see 04-use-cases.md) — demonstrating that the target choice is
a one-word change, not a rewrite. The multi-target demo also
serves as the profiling matrix (cross-language performance
comparison on a fixed workload).

## Open design questions (tracked in 01-philosophy.md §Open questions)

- Cross-target atom interning: which side owns the intern table?
- Backpressure beyond the OS pipe buffer.
- Error propagation across process boundaries.
- Observability hooks: metrics emission side channel.

## Next steps toward implementation

See `03-implementation-plan.md` for the full phased plan.
Shortest path to a validated demo:

1. Leaf Rust crate for MySQL INSERT tokenization (S1.1)
2. Extend `native_glue.pl` with `generate_rust_streaming_main/3`
   (S1.2, small addition to existing templates)
3. Example project: `examples/streaming/enwiki_category_ingest.pl`
4. Integration test on simplewiki (fast, already present locally)
   before committing to enwiki download
5. Profiling run → fill in `RESULTS.md`
