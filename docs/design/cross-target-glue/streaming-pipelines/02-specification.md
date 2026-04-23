# Streaming Data Pipelines — Specification

Extends the existing cross-target glue `declare_target` /
`declare_connection` grammar (see sibling `02-specification.md`
in the parent directory) with streaming-specific options and
contracts.

## 1. Declaration grammar

### 1.1 Target-side streaming options

Applies to producer predicates. Extends `declare_target/3`:

```prolog
:- declare_target(Pred/Arity, Target, Options).

% Streaming-specific options:
streaming(true)                    % Default. Text-mode line pipe.
streaming(true, text)              % Explicit text mode.
streaming(true, binary(FrameSpec)) % Packed binary frames.
streaming(true, api(ModuleSpec))   % In-process object stream.
```

`FrameSpec` describes how a tuple is laid out on the wire for
binary framing:

```prolog
FrameSpec ::= i32_pair           % (Int32, Int32), 8 bytes, little-endian
            | i64_pair           % (Int64, Int64)
            | record(Fields)     % Named fields; see section 1.3
            | packed(TypeList)   % Raw sequence of types, no framing
```

`ModuleSpec` describes the in-process integration point:

```prolog
ModuleSpec ::= pyo3              % Rust → Python via pyo3/maturin
            |  janus             % Prolog ↔ Python via SWI's Janus (existing)
            |  jython            % JVM ↔ Python via Jython (existing glue)
            |  ffi(Header)       % Raw C ABI with a named header
```

### 1.2 Connection-side streaming options

Applies to producer/consumer pairs. Extends
`declare_connection/3`:

```prolog
:- declare_connection(Producer/ProdArity, Consumer/ConArity, Options).

% Streaming-specific options:
streaming(auto)                  % Let the glue pick (default).
streaming(Mode)                  % Force a specific mode.
buffer_size(Bytes)               % Override OS pipe buffer.
batch(N)                         % Batch N records per transport unit.
on_error(Policy)                 % continue | abort | retry(N)
```

`Policy` defaults to `abort` — a parse error in the producer
terminates the pipeline with a propagated exception in the
consumer.

### 1.3 Record types (for typed streams)

A `record(Fields)` frame spec is a list of named, typed fields:

```prolog
record([child:int32, parent:int32])
record([ts:int64, level:string(max(32)), msg:string(bounded)])
```

Typed records are the basis for schema validation across the
transport boundary. At Phase 1.5, `record(Fields)` generates a
packer/unpacker pair on each side; at Phase 2, the fields
become named attributes on the transported object.

## 2. Auto-selection algorithm

When `streaming(auto)` is specified (or defaulted), the glue
resolves the transport using this priority:

1. **Explicit override** — `transport(T)` in connection options
   wins outright.
2. **In-process candidate check** — if producer and consumer
   both resolve to `location(in_process)` and a shared runtime
   exists (Janus for Prolog↔Python, Jython for JVM↔Python,
   pyo3 if both targets expose it), pick the API transport.
3. **Local-process fallback** — if locations are
   `local_process` / `local_process`, pick `pipe` with the
   smallest framing compatible with the declared types
   (`binary` if types are fixed-size, `text` otherwise).
4. **Remote fallback** — `socket` or `http` per the existing
   remote-transport rules in the parent spec.

This mirrors the existing `auto_select_transport/2` pattern in
`mindmap_glue.pl`, extended to consider streaming-specific
framing preferences.

## 3. Leaf primitive contracts

Leaf primitives are hand-written target-native predicates that
the transpiler doesn't inspect. The streaming system needs a few
well-known shapes:

### 3.1 Stream producer

A predicate declared with `streaming(true)` in a target like
Rust/Haskell/AWK MUST, as its native implementation:

- Read its input arguments (file paths, handles, seeds, etc.)
- Emit zero-or-more records in the declared frame format
- Flush output and exit cleanly when input is exhausted
- Exit with a non-zero code on parse error (Phase 1/1.5) or
  raise an exception (Phase 2)

### 3.2 Stream consumer

A predicate receiving a stream MUST:

- Initialize any per-stream state before the first record
- Process records in order (no reordering within a stream)
- Accept end-of-stream as a normal termination signal
- Flush and commit any batched state on end-of-stream

### 3.3 Primitive boundary rule

A predicate is a leaf primitive (not transpiled) when any of
these hold:

- Its body contains byte-level IO (`read_byte/2`, `write_bytes/2`)
- It calls a target-specific library not in the transpiler's
  known-primitives list
- It has a `:- pragma native` directive attached

Otherwise it's transpilable composition — operators, filters,
joins, aggregations — and the target codegen handles it.

## 4. Type marshaling per phase

### 4.1 Phase 1 — Text (TSV)

TSV is the text framing across all Phase 1 streaming
implementations. This matches existing UnifyWeaver conventions
(`data/benchmark/*/category_parent.tsv`, `article_category.tsv`)
and the existing `pipe_glue.pl` TSV reader/writer templates.

- `int32`, `int64`: decimal `"%d"`
- `string`: UTF-8, with TAB and NEWLINE escaped as `\t` / `\n`
  (two-byte sequences) per the existing TSV convention
- Field separator: `\t`
- Record terminator: `\n`
- Record boundary: one record per line

Rationale: TSV reuses `generate_tsv_writer/3` and
`generate_tsv_reader/3` from `pipe_glue.pl` without modification.
CSV is rejected as a framing because quoting rules for strings
containing commas are painful to implement consistently across
targets — `pipe_glue.pl` already pays the cost once for TSV and
gets it right.

### 4.2 Phase 1.5 — Binary

- Integers: little-endian fixed-width (`i32`, `i64`)
- Strings: length-prefixed (`u32` length, then UTF-8 bytes)
- Record terminator: none (framing is implicit in fixed shape)
- Record boundary: consumer reads the frame size per record type

### 4.3 Phase 2 — API

- Types pass as native objects of the consumer language
- Rust `i32` → Python `int`, Rust `String` → Python `str`, etc.
- pyo3 handles marshaling; the glue template generates the
  `#[pyclass]` / `#[pyfunction]` scaffolding
- Record boundary: iterator protocol (`__next__` returning a
  typed tuple or `StopIteration`)

## 4.4 Multi-table streams

Real dumps often contain multiple logical tables. The MediaWiki
`categorylinks` SQL dump is one file with two row types
distinguished by a `cl_type` column (`'subcat'` for
category→category edges, `'page'` for article→category edges).
A single parser pass is much cheaper than two passes over a
3 GB gzipped file.

Three ways to handle this, in ascending order of complexity:

### 4.4.1 Separate predicates (simplest — recommended default)

```prolog
:- declare_target(parse_category_subcats/3, rust, [streaming(true, tsv)]).
:- declare_target(parse_article_categories/3, rust, [streaming(true, tsv)]).
```

Two parser passes, each emits one row type. Simplest semantics,
but 2× the IO. Appropriate when passes are rare (one-time
preprocessing) and the data volume is modest.

### 4.4.2 Tagged TSV (multi-type in one stream)

First TSV column is a discriminator tag; remaining columns
depend on the tag:

```
subcat	12345	Physics
subcat	67890	Chemistry
page	Albert_Einstein	Physicists
page	Marie_Curie	Chemists
```

```prolog
:- declare_target(parse_categorylinks/5, rust,
                  [streaming(true, tsv),
                   tagged([subcat/2, page/2])]).
```

The `tagged/1` option declares the row-type registry: `subcat`
rows have arity 2, `page` rows have arity 2, etc. The consumer
dispatches:

```prolog
forall(
    parse_categorylinks(DumpPath, Tag, F1, F2, _),
    (   Tag == subcat -> ingest_subcat(Db, F1, F2)
    ;   Tag == page   -> ingest_page(Db, F1, F2)
    )
).
```

One pass, one stream. Consumer complexity shifted into Prolog
dispatch. Recommended when IO cost dominates.

### 4.4.3 Multiple output streams

The parser opens numbered file descriptors (fd=3 for subcats,
fd=4 for page links) and writes each row type to its own
stream. Consumer reads from two pipes in parallel.

```prolog
:- declare_target(parse_categorylinks/4, rust,
                  [streaming(true, tsv),
                   outputs([subcats, pages])]).

process(DumpPath, Db) :-
    parse_categorylinks(DumpPath, SubcatStream, PageStream, _),
    par_forall([
        forall(read_tsv(SubcatStream, C, P), ingest_subcat(Db, C, P)),
        forall(read_tsv(PageStream,   A, K), ingest_page(Db, A, K))
    ]).
```

True parallel consumption, no consumer dispatch. Most complex
glue (two pipe endpoints per consumer process). Deferred; not
needed for the initial implementation.

### 4.4.4 Recommendation

- **Initial impl**: 4.4.1 (separate predicates). Clearest
  semantics, least glue complexity. Acceptable for enwiki since
  one-time preprocessing runs on the order of minutes.
- **Upgrade to 4.4.2** (tagged TSV) when IO cost becomes the
  bottleneck. Consumer-side dispatch is a small Prolog change,
  no target-codegen changes.
- **4.4.3** (multi-stream) only if per-row-type parallelism
  matters enough to justify the glue complexity.

## 4.5 Virtual files (observability-first abstraction)

### 4.5.1 What virtual files buy over Unix named pipes

At Phase 1, `mkfifo` + shell redirection already gets you
most of the transport topology. Virtual files only justify
their ceremony when they add something Unix doesn't — and the
primary thing they add is **uniform observability**. If the
glue owns the abstraction, `observe(counters(...))` (spec §7)
works without per-target instrumentation hooks.

Concrete delta versus Unix primitives:

| Capability | Unix `mkfifo` / `tee` | Virtual files |
|------------|----------------------|--------------|
| Transport between local processes | ✓ (pipe) | ✓ (same pipe) |
| Fan-out to N consumers | ✓ (`tee`) | ✓ (glue inserts tee) |
| Typed schema on the wire | ✗ (opaque bytes) | ✓ (`schema([...])`) |
| Counter/throughput instrumentation | Manual per-consumer | ✓ (glue wraps transport) |
| Switch disk-file ↔ pipe without code change | ✗ (requires rewrite) | ✓ (`backing(Auto/Disk/Pipe)`) |
| Validation on shape mismatch | ✗ (runtime errors) | ✓ (at declare-time) |
| Cross-target-agnostic reference | Path string only | Named + typed |

The first three rows are table stakes. The last four are where
the abstraction earns its keep. In a single-producer /
single-consumer pipeline with no instrumentation, virtual files
are equivalent to `mkfifo` plus a variable name; the ceremony
doesn't pay off.

### 4.5.2 Declaration shape

The cleanest way to express multi-table streaming AND preserve
composability with existing on-disk TSV data is to promote the
notion of a "file" to a first-class virtual concept:

```prolog
%% Declare virtual files with schema and format
:- virtual_file('subcats.tsv',
                format(tsv),
                schema([child:int32, parent:string])).
:- virtual_file('pages.tsv',
                format(tsv),
                schema([article:string, category:string])).

%% Producer: emits to multiple virtual files via row-type routing
:- declare_target(parse_categorylinks/1, rust,
                  [streaming(true),
                   leaf(true),
                   native_crate(mysql_stream),
                   produces([
                     'subcats.tsv' = filter(cl_type = "subcat"),
                     'pages.tsv'   = filter(cl_type = "page")
                   ])]).

%% Consumer: reads from a virtual file
:- declare_target(ingest_subcats_to_lmdb/2, python,
                  [streaming(true),
                   reads('subcats.tsv')]).

%% Composition references virtual files by name
process_dump(DumpPath, LmdbPath) :-
    run_producer(parse_categorylinks, [source(DumpPath)]),
    run_consumer(ingest_subcats_to_lmdb, [sink(LmdbPath)]).
```

### 4.5.3 What a virtual file is

A named, typed data stream that can be backed by:

- **A real file on disk** — existing benchmark TSVs like
  `data/benchmark/10k/category_parent.tsv` work as virtual
  files referring to the on-disk artifact
- **A named pipe (FIFO)** — when producer and consumer are
  separate processes, no intermediate disk
- **An in-memory buffer** — when both are in-process
- **A tee / fan-out** — multiple consumers reading the same
  virtual file get the producer's output multiplexed

The *consumer* doesn't know and doesn't need to know which
backing the glue chose. Same Prolog, same call site.

### 4.5.4 Glue responsibilities for virtual files

When the compiler encounters a pipeline that references virtual
files, it picks the backing based on pipeline topology:

1. **1-producer 1-consumer, both local processes** → named pipe
2. **1-producer N-consumers** → named pipe with tee, or
   producer writes to disk for fan-out
3. **Producer now, consumer later** (async) → disk file as
   materialized intermediate
4. **Both in-process** → in-memory ring buffer
5. **Cross-host** → network-backed (HTTP/gRPC) per existing
   transport rules

The choice is transparent to the user code. A flag can force
a specific backing for testing (`virtual_file(..., backing(disk))`).

### 4.5.5 Why this unifies multi-table handling

Sections 4.4.1–4.4.3 become special cases of virtual-file
topology:

- **4.4.1 separate predicates** = two producers writing to two
  virtual files independently
- **4.4.2 tagged TSV** = one producer writing a single
  virtual file with discriminator column
- **4.4.3 multi-stream output** = one producer writing to N
  virtual files via `produces([...])`

The spec doesn't need to pick one — the glue picks based on
declarations, and the user describes *what* they want, not *how*
it's transported.

### 4.5.6 Why this matches existing conventions

The benchmark data shape already encodes this mental model:

```
data/benchmark/10k/
├── category_parent.tsv     ← virtual file, disk-backed
├── article_category.tsv    ← virtual file, disk-backed
└── root_categories.tsv     ← virtual file, disk-backed
```

These are named, typed TSV files. A streaming producer that
writes to `'subcats.tsv'` (virtual) produces output
semantically identical to a run that wrote to
`data/benchmark/enwiki/subcats.tsv` (disk). The consumer code
is unchanged either way.

Virtual files make the streaming pipeline composable with the
existing on-disk TSV benchmarks: the same consumer predicate
works against a file written yesterday or a stream flowing live
from a parser subprocess.

### 4.5.7 Transpilation consequence

When the Prolog references `'subcats.tsv'` and the glue chooses
a named-pipe backing, the generated code on each side uses the
pipe as if it were a file:

```rust
// Rust producer: glue-generated
let subcats_fd = std::env::var("UW_VFILE_SUBCATS").unwrap();
let mut subcats = File::create(subcats_fd)?;  // pipe looks like a file
for row in parse_dump(&dump_path) {
    if row.cl_type == "subcat" {
        writeln!(subcats, "{}\t{}", row.child, row.parent)?;
    }
}
```

```python
# Python consumer: glue-generated
subcats_fd = os.environ['UW_VFILE_SUBCATS']
with open(subcats_fd, 'r') as subcats:
    for line in subcats:
        child, parent = line.rstrip('\n').split('\t')
        ingest_subcats_to_lmdb(env, int(child), parent)
```

The glue lays down the named pipe (`mkfifo`), populates the
environment variables, orchestrates the processes. Neither
producer nor consumer knows it's not a real file.

### 4.5.8 Deferred — first implementation may skip this

Virtual files are a significant feature beyond Phase S1. For
the initial enwiki demo, 4.4.1 (separate predicates calling the
parser twice, once per row type) is simpler and correct. The
virtual-file abstraction is the right *eventual* design but not
a blocker. Adding it is a natural follow-up after S1.5 lands —
at that point the glue layer has enough framing machinery that
the virtual-file layer becomes a thin wrapper.

## 5. Error semantics

| Phase | Producer error | Transport error | Consumer error |
|-------|---------------|-----------------|----------------|
| 1 | Non-zero exit | `BrokenPipeError` / `EPIPE` | `ValueError` on parse |
| 1.5 | Non-zero exit | Same as Phase 1 | `struct.error` on unpack |
| 2 | Native exception | N/A (shared process) | Native exception |

Under Phase 1/1.5, the glue wraps the consumer side to convert
EPIPE into a normal end-of-stream when the producer exits cleanly,
but re-raises when the producer exit code is non-zero.

Under Phase 2, exceptions propagate across the FFI boundary with
whatever fidelity the bindings support (pyo3 has decent
`PyErr`↔`anyhow::Error` story).

## 6. Backpressure

- **Phase 1/1.5**: OS pipe buffer + `write` blocking handles it
  transparently. No explicit flow control needed.
- **Phase 2**: the iterator protocol is pull-based by nature —
  the consumer pulls records on demand, so the producer can't
  race ahead. For push-based APIs, explicit bounded channels
  (`crossbeam::channel::bounded`) required; out of scope for
  initial implementation.

## 7. Observability hooks (optional)

The streaming options may include observability:

```prolog
streaming(true, text, [observe(counters(tuples, bytes))])
```

When specified, the glue wraps the transport to emit metrics
(tuples/sec, bytes/sec, total count, elapsed) to a side channel
(stderr by default, configurable). Implementation is a later
phase; this doc reserves the option shape.

## 8. Concrete touchpoints in existing code

These are the specific locations where streaming would plug in.

### 8.1 Producer template (existing `native_glue.pl`)

The existing `generate_rust_tsv_main/2` already emits a
pipe-compatible `main()` that's very close to what we need for
Phase 1 streaming. From `src/unifyweaver/glue/native_glue.pl:401`:

```prolog
generate_rust_tsv_main(Logic, Code) :-
    format(atom(Code), '
use std::io::{self, BufRead, Write};

fn process(fields: &[&str]) -> Option<Vec<String>> {
~w
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        match line {
            Ok(text) => {
                let fields: Vec<&str> = text.split(\'\\t\').collect();
                if let Some(result) = process(&fields) {
                    writeln!(stdout, "{}", result.join("\\t")).unwrap();
                }
            }
            ...
        }
    }
}
', [Logic]).
```

This is **already Phase 1 transport**. What's missing is:
- A `generate_rust_binary_pipe_main/2` sibling for Phase 1.5
  (packed byte I/O instead of TSV)
- A `streaming_source(SourceType)` option so the template can
  emit from-file input instead of stdin (for the enwiki dump,
  stdin is the gzipped file — we want an explicit file path so
  the producer can open and unzip it itself)

### 8.2 Transport auto-selection (existing `mindmap_glue.pl`)

From `src/unifyweaver/mindmap/glue/mindmap_glue.pl:192`:

```prolog
select_transport(_Predicate, Options, Transport) :-
    member(transport(Transport), Options),  % explicit override
    !.
select_transport(Predicate, _Options, Transport) :-
    auto_select_transport(Predicate, Transport).

auto_select_transport(_Predicate, janus) :-
    mindmap_janus_available,                % in-process candidate
    check_mindmap_packages(true),
    !.
auto_select_transport(_Predicate, pipe) :-  % fallback
    !.
```

This is the pattern for the streaming auto-selection in Section 2
— generalize the janus-availability check to cover pyo3 (for
Rust→Python) and jython (for JVM→Python) as in-process
candidates before falling back to `pipe`.

### 8.3 Declaration entry point (existing `target_mapping.pl`)

From `src/unifyweaver/core/target_mapping.pl:67`:

```prolog
declare_target(Pred/Arity, Target, Options) :-
    ...
```

No changes needed to the declaration API — `streaming(true)` and
related options just pass through `Options`. The glue reads them
during code generation.

## 9. What's in scope for first implementation

Minimum viable streaming glue:

- [x] Existing: `native_glue.pl` handles Rust/Go binary
      compilation + pipe orchestration
- [ ] New: `streaming(true, text)` as the default, explicit
      form for existing pipe behavior (semantic preservation)
- [ ] New: `streaming(true, binary(i32_pair))` frame
      generator for the enwiki use case
- [ ] New: `pred_is_streaming_leaf/2` helper that tells the
      codegen not to transpile a predicate body
- [ ] New: `auto_select_transport` extension for streaming
      connections

Out of scope for first impl:

- Phase 2 pyo3 integration (research branch)
- `record(Fields)` typed frame generation
- Observability hooks
- Cross-host streaming (HTTP/gRPC transport) — possible later,
  but first validate locally-piped case
