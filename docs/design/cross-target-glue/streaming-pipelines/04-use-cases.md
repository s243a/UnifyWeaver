# Streaming Data Pipelines — Use Cases

Concrete examples grounding the philosophy and specification
docs. Each use case includes the Prolog declarations, the leaf
primitive(s), and the expected data flow.

## 1. MySQL dump parser — the primary demo

Parse `enwiki-latest-categorylinks.sql.gz` into LMDB for use by
the WAM Haskell target's category-hierarchy benchmarks.

### 1.1 The shared Prolog pipeline

The goal is that this declaration works unchanged regardless of
which target implements the parser:

```prolog
%% examples/streaming/enwiki_category_ingest.pl
:- use_module(library(cross_target_glue/target_mapping)).

:- declare_target(ingest_to_lmdb/3, python,
                  [streaming(true, text)]).

process_dump(DumpPath, LmdbPath) :-
    forall(
        parse_category_links(DumpPath, Child, Parent),
        ingest_to_lmdb(LmdbPath, Child, Parent)
    ).
```

The consumer (`ingest_to_lmdb`) is fixed as a Python predicate
because Python has the best LMDB ergonomics. The *producer*
(`parse_category_links`) is what we implement in multiple
targets as a demo of reusability.

### 1.2 Rust implementation (primary, Phase 2 path)

```prolog
:- declare_target(parse_category_links/3, rust,
                  [streaming(true, text),
                   source(file),
                   leaf(true),
                   native_crate(mysql_stream)]).
```

Leaf primitive at `src/unifyweaver/runtime/rust/mysql_stream/`:

```rust
// Approximately 150 lines hand-written Rust:
//   - GzDecoder wrapping BufReader
//   - State machine: SkipPreamble → SeekInsert → InValues → InTuple
//                    → InField → InString → EscapeChar → ...
//   - Type dispatch: quoted string | unquoted int | NULL | backtick ident
//   - Yields (i32, String) tuples for cl_type='subcat' rows

pub fn iter_category_links(path: &str) -> impl Iterator<Item = (i32, String)> {
    let file = File::open(path).expect("open dump");
    let gz = GzDecoder::new(file);
    MysqlInsertIter::new(BufReader::with_capacity(1 << 20, gz))
        .filter_map(|t| match (t.get(0), t.get(1), t.get(6)) {
            (Some(Int(from)), Some(Str(to)), Some(Str(typ)))
              if typ == "subcat" => Some((*from as i32, to.clone())),
            _ => None,
        })
}
```

**Why Rust here**: cleanest Phase 2 path via pyo3 + maturin, no
runtime initialization cost (unlike Haskell/Go), the broadest
deployment story.

### 1.3 Haskell implementation (second demo)

```prolog
:- declare_target(parse_category_links/3, haskell,
                  [streaming(true, text),
                   source(file),
                   leaf(true),
                   cabal_package('mysql-stream')]).
```

Leaf primitive uses `attoparsec` on `ByteString`:

```haskell
-- src/unifyweaver/runtime/haskell/mysql-stream/src/MysqlStream.hs
module MysqlStream (iterCategoryLinks) where

import Data.Attoparsec.ByteString.Lazy as AL
import qualified Data.ByteString.Lazy as BL
import Codec.Compression.GZip (decompress)

iterCategoryLinks :: FilePath -> IO [(Int, ByteString)]
iterCategoryLinks path = do
    gz <- BL.readFile path
    let decoded = decompress gz
    case AL.parse mysqlDumpParser decoded of
        AL.Done _ tuples -> return (filter subcatOnly tuples)
        AL.Fail _ _ err  -> error $ "parse failed: " ++ err
  where
    subcatOnly (_, _, t) = t == "subcat"
    ...
```

**Why Haskell here**: existing toolchain in the repo,
`attoparsec` is arguably the most readable byte-level parser
library in any language, and it exercises the Haskell target's
`declare_target` path end-to-end. Phase 2 path goes through
`foreign export ccall` — workable but doc as "research" status.

### 1.4 AWK implementation (third demo — simple-tools showcase)

```prolog
:- declare_target(parse_category_links/3, awk,
                  [streaming(true, text),
                   source(stdin),
                   leaf(true),
                   script('parse_mysql_inserts.awk')]).
```

Leaf AWK script:

```awk
# src/unifyweaver/runtime/awk/parse_mysql_inserts.awk
# Reads zcat output on stdin, writes TSV (child, parent) to stdout
# for cl_type='subcat' rows.

BEGIN { FS = ""; RS = ";"; }

/^INSERT INTO `categorylinks`/ {
    # VALUES (...),(...),...  — each group is one row
    while (match($0, /\(([0-9]+),'([^']*)',[^)]*,'subcat'\)/, arr)) {
        print arr[1] "\t" arr[2]
        $0 = substr($0, RSTART + RLENGTH)
    }
}
```

Orchestrated as:

```bash
zcat enwiki-latest-categorylinks.sql.gz \
  | awk -f parse_mysql_inserts.awk \
  | python3 ingest.py ./enwiki_lmdb/
```

**Why AWK here**: demonstrates that the declaration-side of
UnifyWeaver is target-agnostic — dropping in AWK as the producer
is a one-line change in the `declare_target`. The shell pipeline
is an existing capability. Phase 2 is N/A for AWK; this implementation
is a dead-end for the speed axis but a live demo of minimum-infrastructure
integration. Caveat: the AWK regex above elides the full escape
rules — realistic for simplewiki (clean data), not for enwiki
(malformed historical rows exist). Production use would need
error-tolerance.

### 1.5 Cross-implementation comparison table

| Implementation | Lines of native code | Phase 2 path | Infrastructure needed |
|----------------|---------------------|--------------|----------------------|
| Rust | ~150 | pyo3/maturin (clean) | cargo, flate2 crate |
| Haskell | ~80 (attoparsec) | foreign export ccall (awkward) | cabal, attoparsec, zlib |
| AWK | ~15 | N/A (dead-end) | gawk only |

All three are equivalent at the Prolog-declaration level — the
only change is the `Target` argument to `declare_target/3`. This
is the generality demonstration.

### 1.6 Profiling matrix across configurations

The three-language demo doubles as a performance measurement
exercise — same workload (enwiki category ingestion), same
Prolog declarations, different configurations. This is the
natural experiment UnifyWeaver uniquely enables.

**Axes:**

| Axis | Values |
|------|--------|
| Parser target | rust, haskell, awk |
| Transport phase | S1 (text) / S1.5 (binary) / S2 (pyo3 API) |
| Consumer | python (LMDB ingest, fixed) |

**Cell coverage** (✓ feasible, ✗ structurally impossible):

|              | S1 text | S1.5 binary | S2 pyo3 API |
|--------------|---------|-------------|-------------|
| Rust parser  | ✓ | ✓ | ✓ |
| Haskell parser | ✓ | ✓ | ✗ (no pyo3 for Haskell; would need `foreign export ccall` + custom FFI) |
| AWK parser   | ✓ | ✗ (AWK can emit binary with `printf "\x.."` but it's ugly) | ✗ |

That's five solid cells + two "requires extra work" cells — a
real cross-product.

**Metrics to capture per cell:**

- **Throughput** (MB/s of gzipped input, tuples/sec emitted)
- **CPU breakdown** (parser % / IPC % / consumer %) via
  `/usr/bin/time -v` per process in the pipeline
- **Peak RSS** per process
- **Startup overhead** (time to first record)
- **Elapsed wall time** end-to-end

**Expected rough ordering** (hypothesis to test, not a prediction):

- `(Rust, S2)` fastest — no IPC, native function calls
- `(Rust, S1.5)` within 2-3x — pipe-IO-bound, packed bytes
- `(Rust, S1)` within 5-10x — text parsing overhead
- `(Haskell, S1)` similar to `(Rust, S1)` if attoparsec is as
  fast as expected — interesting if not
- `(AWK, S1)` probably 2-3x slower than Rust S1 on CPU-bound
  parsing, but surprisingly competitive because the whole
  thing is bytecode-tight

The *interesting* findings are likely the counterintuitive
ones — where does the ordering break? What does pipe buffering
do at different record sizes? Does GC in the Haskell producer
dominate or hide?

**Fixed consumer constraint**: keeping the Python LMDB ingest
side constant across all cells isolates the producer/transport
variable. A separate matrix could vary the consumer (Python
LMDB / Rust LMDB / Haskell LMDB) with a fixed producer.

### 1.7 Profiling infrastructure

The streaming spec (section 7) reserves an `observe(counters(...))`
option; implementing it for this demo gives us the metrics
above without per-parser instrumentation:

```prolog
:- declare_connection(
     parse_category_links/3,
     ingest_to_lmdb/3,
     [streaming(true, Mode),
      observe(counters(tuples, bytes), interval(1s))]).
```

The glue wraps the transport with a counter emitter to stderr
or a sidecar file. Works uniformly across Rust/Haskell/AWK
because it's at the transport layer, not the parser layer.

### 1.8 Reporting format

A CSV or markdown table like:

| parser | transport | mb_per_sec | tuples_per_sec | parser_cpu_% | consumer_cpu_% | peak_rss_mb | startup_ms |
|--------|-----------|-----------:|---------------:|-------------:|---------------:|------------:|-----------:|
| rust | text | ... | ... | ... | ... | ... | ... |
| rust | binary | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | |

Checked into `docs/design/cross-target-glue/streaming-pipelines/
RESULTS.md` once the demo lands. The per-cell measurements
become the story of "what we gained from each configuration."

## 2. CSV bulk loader (future)

Generalized case: ingest any tab/comma-separated data into a
store. The producer is a leaf primitive in the chosen target,
the consumer is target-agnostic:

```prolog
:- declare_target(parse_csv/4, awk,
                  [streaming(true, text),
                   source(file),
                   leaf(true),
                   delimiter(',')]).
:- declare_target(ingest_to_store/4, python,
                  [streaming(true, text)]).

bulk_load(Path, StorePath, Schema) :-
    forall(
        parse_csv(Path, Schema, Row),
        ingest_to_store(StorePath, Schema, Row, _)
    ).
```

Shows the pattern reused for any column-oriented source.

## 3. Log tailer (future — Phase 2 candidate)

Real-time streaming, not batch. Producer is a Rust binary
tailing a file; consumer is a Python predicate filtering and
alerting:

```prolog
:- declare_target(tail_log/2, rust,
                  [streaming(true, text),
                   source(file_follow),
                   leaf(true)]).
:- declare_target(filter_and_alert/2, python,
                  [streaming(true, text)]).

watch_logs(LogPath) :-
    forall(tail_log(LogPath, Line),
           filter_and_alert(Line, _)).
```

This case motivates Phase 2 more than preprocessing does: the
per-record overhead of text IO is observable in a
many-small-records workload like log tailing.

## 4. What the multi-target demo proves

The MySQL parser implemented in three targets (Rust, Haskell,
AWK) against the same Prolog declarations is the cleanest
demonstration that:

1. **The `declare_target` API is the real interface.** Changing
   targets is a one-word change, not a rewrite.
2. **Leaf primitives are a legitimate boundary.** Not every
   predicate needs to be transpiled — some are native libraries
   the user brings. This is the same pattern WAM Haskell uses
   for LMDB's `EdgeLookup`.
3. **Phase-1 IO transport is target-agnostic.** Any leaf
   primitive that emits text lines composes with any consumer
   that reads them.
4. **Phase 2 is a long-term upgrade path, not a prerequisite.**
   Text pipes work across all reasonable targets today. API-based
   streaming is a per-target-pair investment that pays off only
   for tight inner loops.

For documentation purposes, we'd ship all three implementations
as `examples/streaming/mysql-dump-{rust,haskell,awk}/`, each with
its own README showing the identical Prolog call site. The
comparison table in 1.5 becomes the reference "which should I
use?" guide.
