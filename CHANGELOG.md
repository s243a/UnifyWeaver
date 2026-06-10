# Changelog

All notable changes to UnifyWeaver will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **WAM LLVM target (M139): permanent-variable corruption via the
  put-instruction bind-through.** The pre-existing bug noted in the
  M138 test comments: top-level fresh-variable goals mis-executed —
  `var(X), X = done` failed, and `var(X)` followed by `nonvar(X)`
  BOTH succeeded on the same unbound variable. Root cause:
  `put_variable Yn, Ai` places the same `Ref{addr}` into both
  registers, and the later `put_value`/`put_variable` over `Ai`
  called `@wam_bind_through_if_unbound_ref(old_Ai, val)`, which
  either (a) wrote the Ref back into its own cell — a
  self-referential cell that derefs to itself, so `var/1` saw the
  variable as bound and every later unification corrupted — or
  (b) bound the live unbound variable previously occupying `Ai` to
  the incoming value, silently aliasing two unrelated variables
  (e.g. in `var(X), R0 is 1, X = x` the `put_variable` staging `R0`
  bound `X` to `R0`, so `X = x` then failed against `1`). Fixed
  three ways: a self-reference guard in
  `@wam_bind_through_if_unbound_ref` (never store a Ref into its own
  cell), `put_value` is now a pure register copy per standard WAM,
  and `put_variable` no longer binds `Ai`'s previous occupant. The
  remaining bind-through callers (`put_constant`, `put_structure`,
  `put_list`, `get_list`) are unchanged but now protected by the
  self-reference guard. Five regression tests in the new
  `--- M139 put_value Y-reg aliasing ---` section.
- **WAM LLVM target (M138): `@value_equals` call-site audit** — the
  follow-up flagged in the M137 (#2929) review. Three bound-bound
  compare sites used shallow equality where WAM semantics require
  unification, and one equality site inherited the M137 nested-Ref bug:
  - `get_value` now routes its bound-bound case through
    `@wam_unify_value` — `p(X, X)` heads called with unifiable
    compounds like `p(f(A), f(B))` previously failed instead of
    binding `A = B`.
  - `unify_value` (read mode) both-bound case likewise unifies; it
    also never dereffed either side, so Ref-wrapped values compared
    by heap address.
  - `arg/3` with bound A3 now unifies with the extracted argument
    (ISO behavior); the argument slot may hold a Ref the old compare
    saw as an address. `arg(1, h(f(a)), f(a))` previously failed.
  - `sort/2` dedup now uses `@wam_strict_eq` (dedup is `==/2`
    equality — it must not bind, so unification would be wrong):
    duplicate compounds with Ref-stored args survived dedup
    (`sort([f(g(a)), f(g(a))], L)` kept both).
  - `@wam_strict_eq` now derefs via a new `@wam_deref_keep_var`
    helper that stops at the last Ref when the target cell is
    unbound, so variable identity is the cell address: `X == X` is
    true, `X == Y` (distinct fresh vars) is false, and sort/2 keeps
    `f(X)` / `f(Y)` distinct without binding them. Previously every
    unbound var collapsed to the shared `{tag 6, payload 0}`
    sentinel and compared equal.
- **WAM LLVM target: undefined `@wam_dispatch_meta_call` when every
  predicate is fully lowered.** With `emit_mode(functions)` and all
  predicates natively lowered, no bytecode records exist, so the merged
  WAM section — which carries the `@wam_dispatch_meta_call` definition —
  was skipped entirely. The interpreter runtime's `do_call`/`do_execute`
  cases reference it unconditionally, so clang rejected the module
  (`use of undefined value`). Both empty-bytecode short-circuits now
  still emit the dispatch function with empty tables. Fixes the
  previously-failing `test_wam_llvm_lowered_ite_exec`.
- **WAM Haskell target: ITE bodies declined by the lowered emitter
  under `ite_use_y_level` bytecode.** The WAM compiler emits if-then-else
  as `get_level Yn` + `cut Yn`, but the Haskell lowered emitter's
  `supported/1` whitelist and `struct_item/3` only knew the legacy
  `cut_ite` marker, so every ITE predicate fell back to the interpreter
  (`lowered=0`) and the gated exec test's harness referenced lowered
  functions that were never generated. The whitelist now accepts
  `cut(_)`/`get_level(_)`, `struct_item` keeps `cut(Yn)` bare so the
  shared structurer folds it as the commit (its `is_commit/1` already
  accepted that form), and `get_level` is a no-op in the lowered
  if/else. Fixes the previously-failing
  `test_wam_haskell_lowered_ite_exec`.
- **WAM Haskell target: generated-file writes crash under POSIX/ASCII
  locales.** `write_hs_file/2` (and the T4/T5/ITE exec tests' harness
  writers) opened output streams without `encoding(utf8)`; the
  generated modules embed UTF-8 from the runtime templates, so
  `format/3` raised `Encoding cannot represent character` in CI-style
  containers. All writers now open with `encoding(utf8)`, matching the
  LLVM target's writer.
- **WAM Scala target: LMDB cursor-scan (`streamAll`) reflection.**
  `LmdbFactSource.scanCursorAll`/`cursorSeekAndCollectDupSort` looked up
  the lmdbjava cursor method via `getMethod("`val`")` — the Scala
  keyword-escaping backticks were embedded in the *reflective name
  string*, so the lookup (for a method actually named `val`) threw
  `NoSuchMethodException` at runtime. The earlier LMDB tests only used
  ground-key lookups (`Dbi.get`), never the cursor scan, so this never
  fired. Surfaced by running a graph kernel over an LMDB edge relation
  (which enumerates the whole relation via `streamAll`); fixed to
  `getMethod("val")`.

### Added
- **WAM Scala target: LMDB-backed graph kernels** verified end-to-end —
  `kernel_dispatch(true)` composes with an `lmdb(...)` edge fact source so
  a native kernel reads its adjacency directly from LMDB. New gated
  `lmdb_backed_kernel` test in `tests/test_wam_scala_lmdb_runtime_smoke.pl`
  (transitive closure over an LMDB-stored chain), which also covers the
  `streamAll` cursor-scan path the key-lookup tests miss.
- **Cross-target audit: WAM first-argument-indexing instruction handlers**
  (`docs/WAM_SWITCH_INDEXING_CROSS_TARGET.md`). After fixing the Scala
  `switch_on_*_a2` / `_fallthrough` gap, audited all WAM targets for the
  same gap and — crucially — quantified the *actual* correctness impact:
  first-arg indexing is an optimization, so a target that **drops** an
  unhandled switch instruction (Python/Go/Rust/R/Lua) stays correct (just
  unoptimised; confirmed by running `member/2` on the Python target),
  while only targets whose catch-all emits a **harmful** instruction
  (haskell → `Proceed`, elixir main path → `:fail`, wat → `allocate`,
  jvm → `ldc`) actually mis-execute indexed predicates. The doc gives the
  coverage matrix, the drop-vs-harmful classification with file:line
  evidence, and a prioritised fix plan (harmful-catch-all targets first).
- **WAM Scala target: arity-N LMDB fact sources.** The `lmdb(...)`
  fact-source backend, previously arity-2 only, now supports any arity
  ≥ 2: the LMDB key holds arg1 and the value holds args 2..N tab-joined,
  which `LmdbFactSource` splits back into registers 2..N (arity-2 is the
  no-tab degenerate case, unchanged). The codegen handler clause and the
  runtime drop the arity-2 restriction. `tests/test_wam_scala_lmdb_runtime_smoke.pl`
  gains a gated arity-3 end-to-end test (seed `k → a<TAB>b` → query the
  triple).

### Fixed
- **WAM Scala target: LMDB fact source now actually works (Phase S8).**
  The arity-2 `lmdb(...)` fact-source adaptor and its runtime test had
  never been able to run; validating them end-to-end (with `lmdbjava`
  0.9.0) surfaced and fixed several latent bugs:
  - `write_runtime_source/3` placed `WamRuntime` in the program's
    `package` but `GeneratedProgram` imports it from `runtime_package` —
    so whenever the two differ, including the **default options**
    (`…core` vs `…runtime`), the generated project failed to compile.
    All prior tests happened to pass the same package for both, hiding it.
    Now placed in `runtime_package`.
  - The LMDB runtime test's gate (`getenv('SCALA_LMDB_TESTS', "1")`) could
    never be true (`getenv/2` yields an atom, never the string `"1"`), so
    the test was permanently skipped. Gate now keys on the variable's
    presence.
  - `LmdbFactSource` reflection was written against a different lmdbjava
    API: fixed `Env.Builder.open(File, EnvFlags…)` + `setMapSize`/
    `setMaxReaders`, `Dbi.get(Txn, key)` (was `Txn.get`), `Cursor.get`,
    generic-erasure parameter types (`Object`, not `ByteBuffer`), and
    default/unnamed-DB handling; added a configurable `mapSize` (default
    1 GiB). The end-to-end LMDB smoke test (seed → read → query) now
    passes.
  - Documented the JDK 16+ module flags lmdbjava requires
    (`--add-opens java.base/java.nio` and
    `--add-exports java.base/sun.nio.ch`).

### Added
- **WAM Scala target: execution-mode benchmark** — new
  `tests/benchmarks/wam_scala_mode_bench.pl` compares the interpreter,
  lowered (`emit_mode(functions)`), and kernel (`kernel_dispatch(true)`)
  modes on a transitive-closure workload via the generated program's
  `--bench` inner-loop mode. Results + interpretation in
  `benchmarks/wam_scala_mode_bench.md`: the native graph kernel runs deep
  reachability ~4× faster at chain depth 100 and ~9× at depth 300 (the win
  grows with depth), with fixed setup overhead on trivial/shallow queries;
  the lowered emitter is roughly neutral for recursion-heavy predicates
  (its benefit is largest for single-clause inline-deterministic ones).
- **WAM Scala target: hot-path graph kernels (opt-in `kernel_dispatch(true)`)** —
  brings Scala onto the Rust/Haskell/Elixir/Go native-kernel route. The
  shared recursive-kernel detector runs over the predicates; a matching
  predicate is replaced by a synthesized Scala `ForeignHandler` that does
  the traversal natively, bypassing the WAM step loop. The handler builds
  its adjacency map from the kernel's edge relation via a new
  `WamRuntime.collectBinarySolutions/2` enumerator (works for WAM facts
  and fact sources alike). Kernel kinds implemented: `transitive_closure2`,
  `transitive_distance3` (BFS shortest-path distance),
  `transitive_parent_distance4` (target + immediate predecessor + distance),
  `transitive_step_parent_distance5` (target + first hop + parent +
  distance), `category_ancestor` (depth-bounded ancestor search with a
  visited list, `max_depth` from config; parses the visited list via a new
  `WamRuntime.wamListToVector/2` helper), and `weighted_shortest_path3`
  (Dijkstra over a ternary weighted edge relation via a new
  `WamRuntime.collectTernarySolutions/2` enumerator; binds the shortest
  total weight as a float), and `astar_shortest_path4` (goal-directed A*
  over the ternary weighted edges with a heuristic oracle
  (`direct_dist_pred`) and Minkowski dimensionality `f = g^D + h^D`; binds
  the shortest distance as a float). **All seven recognised kernel kinds
  are now implemented**, bringing Scala to full graph-kernel parity with
  the Rust/Haskell/Elixir/Go targets.
  `tests/test_wam_scala_kernels.pl` has structural tests plus gated runtime
  tests asserting kernel-mode and interpreter-mode results are identical
  and correct.
- **WAM Scala target: per-predicate lowered emitter** — brings the Scala
  hybrid WAM to parity with the Haskell/Rust/C++/F#/Go/Clojure targets,
  all of which already shipped a `wam_*_lowered_emitter.pl`.
  - New `src/unifyweaver/targets/wam_scala_lowered_emitter.pl` emits a
    native Scala `lowered_<pred>_<arity>(s, program): Boolean` function
    per lowerable predicate (deterministic clause 1; simple register ops
    inlined, structure/unify ops via new `lo*` `WamRuntime` helpers,
    deterministic builtins via `loBuiltin`).
  - New `emit_mode(Mode)` option on `write_wam_scala_project/3`
    (`interpreter` default / `functions` / `mixed([P/A,...])`), plus a
    global `user:wam_scala_emit_mode/1` hook. The generated
    `loweredEntries` map + `runEntry` try the fast path and fall back to a
    fresh interpreter run on a clause-1 miss, so results are identical to
    the pure interpreter for any boolean query.
  - New `tests/test_wam_scala_lowered_emitter.pl` — structural tests
    (always run) plus gated runtime parity tests that compile the same
    predicates in both modes and assert identical, correct results.
  - Fixed first-argument indexing instructions that were silently
    degrading to `Raw(...)` stubs and breaking the interpreter for some
    predicate shapes, now handled (matching the F#/C/C++/R targets):
    `switch_on_constant_fallthrough` (mixed fact+rule predicates such as
    the factorial/Ackermann/Fibonacci base cases) reuses the
    `SwitchOnConstant` instruction; `switch_on_term_a2` /
    `switch_on_constant_a2` (+`_fallthrough`) second-argument indexing
    (e.g. `member/2`) dispatch on the correct register via a new `reg`
    field on the `SwitchOnConstant` / `SwitchOnTerm` runtime instructions
    (defaulting to 1, so A1-indexed codegen is unchanged).
- **Client-Server Phase 8: Service Tracing** - OpenTelemetry-compatible distributed tracing
  - `tracing(Bool)` option to enable distributed tracing
  - Trace exporters: `otlp`, `jaeger`, `zipkin`, `datadog`, `console`, `none`
  - Trace propagation formats: `w3c`, `b3`, `b3_multi`, `jaeger`, `xray`, `datadog`
  - `trace_sampling(Rate)` for sampling rate (0.0-1.0)
  - `trace_service_name(Name)` for custom service name in traces
  - `trace_propagation(Format)` for context propagation format
  - `trace_attributes(List)` for default span attributes
  - `trace_batch_size(N)` and `trace_export_interval(Ms)` for batch export
  - SpanContext with W3C traceparent header generation/parsing
  - Span management with SpanKind (Server/Client/Producer/Consumer/Internal)
  - SpanEvent support for span events with timestamps
  - Tracer with sampling decision, context extraction/injection, batch export
  - SpanExporter interface with OTLP, Jaeger, Zipkin, Console implementations
  - Helper predicates: `is_tracing_enabled/1`, `get_trace_sampling/2`, `get_trace_exporter/2`, `get_trace_service_name/2`, `get_trace_propagation/2`, `get_trace_attributes/2`
  - Tracing service compilation for Python, Go, and Rust targets
  - 20 integration tests in `tests/integration/test_service_tracing.sh`
  - Documentation updated in `docs/CLIENT_SERVER_DESIGN.md`

- **Client-Server Phase 7: Service Discovery** - Automatic service registration and health checks
  - `discovery_enabled(Bool)` option to enable service discovery
  - Discovery backends: `consul`, `etcd`, `dns`, `kubernetes`, `zookeeper`, `eureka`
  - `health_check(Config)` for health check configuration: `http(Path, IntervalMs)`, `tcp(Port, IntervalMs)`
  - `discovery_ttl(Seconds)` for service TTL in heartbeat
  - `discovery_tags(List)` for service filtering tags
  - ServiceRegistry interface with ConsulRegistry and LocalRegistry implementations
  - HealthChecker with HTTP/TCP health check support
  - ServiceInstance with metadata, health status, and last heartbeat
  - Automatic heartbeat mechanism with TTL-based renewal
  - Graceful deregistration on shutdown
  - Helper predicates: `is_discovery_enabled/1`, `get_health_check_config/2`, `get_discovery_ttl/2`, `get_discovery_backend/2`, `get_discovery_tags/2`
  - Discovery service compilation for Python, Go, and Rust targets
  - 20 integration tests in `tests/integration/test_service_discovery.sh`
  - Documentation updated in `docs/CLIENT_SERVER_DESIGN.md`

- **Client-Server Phase 6: Distributed Services** - Cluster-aware services with sharding and replication
  - Sharding strategies: `hash`, `range`, `consistent_hash`, `geographic`
  - Consistency levels: `eventual`, `strong`, `quorum`, `read_your_writes`, `causal`
  - Partition key routing with `partition_key(Field)` option
  - Replication factor with `replication(N)` option
  - Cluster configuration with `cluster([node(Id, Host, Port)])` option
  - ConsistentHashRing implementation with virtual nodes for all targets
  - ShardRouter with configurable sharding strategies
  - ReplicationManager with write/read quorum support
  - Distributed service validation in `service_validation.pl`
  - Helper predicates: `get_replication_factor/2`, `get_consistency_level/2`, `get_sharding_strategy/2`, `get_partition_key/2`, `get_cluster_config/2`, `is_distributed_service/1`
  - Distributed service compilation for Python, Go, and Rust targets
  - Thread-safe implementations: Python threading.Lock, Go sync.RWMutex/atomic, Rust RwLock/AtomicU64
  - 24 integration tests in `tests/integration/test_distributed_services.sh`
  - Documentation updated in `docs/CLIENT_SERVER_DESIGN.md`

- **Client-Server Phase 5: Polyglot Services** - Cross-language service calls
  - `polyglot(true)` option for cross-language services
  - `target_language(Lang)` option (python, go, rust, javascript, csharp)
  - `depends_on([dep(Name, Lang, Transport)])` for service dependencies
  - ServiceClient class/struct for HTTP-based cross-language calls
  - ServiceRegistry for local and remote service management
  - Automatic endpoint extraction from transport configurations
  - Polyglot service validation in `service_validation.pl`
  - Helper predicates: `get_target_language/2`, `get_service_dependencies/2`, `get_service_endpoint/2`, `is_polyglot_service/1`, `is_valid_target_language/1`, `is_valid_service_dependency/1`
  - Polyglot service compilation for Python, Go, and Rust targets
  - Thread-safe implementations with HTTP clients: Python urllib.request, Go net/http, Rust reqwest
  - 22 integration tests in `tests/integration/test_polyglot_services.sh`
  - Documentation updated in `docs/CLIENT_SERVER_DESIGN.md`

- **Client-Server Phase 4: Service Mesh** - Load balancing, circuit breakers, and retry with backoff
  - Load balancing strategies: `round_robin`, `random`, `least_connections`, `weighted`, `ip_hash`
  - Circuit breaker pattern with `threshold`, `timeout`, `half_open_requests`, `success_threshold`
  - Retry with backoff: `fixed`, `linear`, `exponential` strategies
  - Retry options: `delay(Ms)`, `max_delay(Ms)`, `jitter(Bool)`
  - Service discovery configuration: `static`, `dns`, `consul`, `etcd`
  - Backend pool configuration for load balancing targets
  - Service mesh validation in `service_validation.pl`
  - Helper predicates: `get_load_balance_strategy/2`, `get_circuit_breaker_config/2`, `get_retry_config/2`, `has_load_balancing/1`, `has_circuit_breaker/1`, `has_retry/1`, `is_service_mesh_service/1`
  - Service mesh compilation for Python, Go, and Rust targets
  - Thread-safe implementations: Python threading.Lock, Go atomic operations, Rust RwLock/AtomicI32
  - 61 integration tests in `tests/integration/test_service_mesh.sh`
  - Documentation updated in `docs/CLIENT_SERVER_DESIGN.md`

- **Client-Server Phase 3: Network Services** - TCP and HTTP transport support
  - TCP transport for network socket communication (`transport(tcp(Host, Port))`)
  - HTTP/REST transport for web API endpoints (`transport(http(Endpoint))`)
  - JSONL protocol for TCP services (streaming JSON lines)
  - JSON protocol for HTTP services (REST API style)
  - Transport categorization: `is_network_service/1` predicate
  - Full REST method support: GET, POST, PUT, DELETE, PATCH
  - Stateful and stateless service variants for both TCP and HTTP
  - Service compilation for Python, Go, and Rust targets
  - Client generation for all three targets
  - 26 integration tests in `tests/integration/test_network_services.sh`
  - Documentation updated in `docs/CLIENT_SERVER_DESIGN.md`

- **Enhanced Pipeline Chaining** - Complex data flow patterns across all targets (#296-#300)
  - `fan_out(Stages)` — Broadcast records to stages (sequential execution)
  - `parallel(Stages)` — Execute stages concurrently using target-native parallelism
  - `merge` — Combine results from fan_out or parallel stages
  - `route_by(Pred, Routes)` — Conditional routing based on predicate
  - `filter_by(Pred)` — Filter records by predicate condition
  - `batch(N)` — Collect N records into batches for bulk processing
  - `unbatch` — Flatten batches back to individual records
  - Supported targets: Python, Go, C#, Rust, PowerShell, AWK, Bash, IronPython
  - `docs/ENHANCED_PIPELINE_CHAINING.md` — Unified documentation
  - Integration tests for all targets

- **Parallel Stage Execution** - True concurrent processing for performance-critical workloads
  - `parallel(Stages)` stage type for concurrent stage execution
  - `parallel(Stages, Options)` with options support:
    - `ordered(true)` — Preserve stage definition order in results (default: completion order)
  - Target-native parallelism mechanisms:
    - Python: `ThreadPoolExecutor`
    - Go: Goroutines with `sync.WaitGroup`
    - C#: `Task.WhenAll`
    - Rust: `std::thread` by default, rayon with `parallel_mode(rayon)` option
    - PowerShell: Runspace pools
    - Bash: Background processes with `wait`
    - IronPython: .NET `Task.Factory.StartNew` with `ConcurrentBag<T>`
    - AWK: Sequential by default, GNU Parallel with `parallel_mode(gnu_parallel)` option
  - Validation support: empty parallel detection, single-stage parallel warning, invalid option detection
  - Clear distinction from `fan_out` (sequential) vs `parallel` (concurrent)

- **Pipeline Validation** - Compile-time validation for enhanced pipeline stages
  - `src/unifyweaver/core/pipeline_validation.pl` — Validation module
  - Error detection: empty pipeline, invalid stages, empty fan_out, empty parallel, empty routes, invalid route format, invalid batch size
  - Warning detection: fan_out/parallel without merge, merge without fan_out/parallel
  - Options: `validate(Bool)` to enable/disable, `strict(Bool)` to treat warnings as errors
  - Integrated into all enhanced pipeline compilation predicates
  - `tests/integration/test_pipeline_validation.sh` — Integration tests
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Aggregation Stages** - Deduplication and grouping at pipeline level
  - Deduplication stages:
    - `unique(Field)` — Keep first occurrence of each unique field value
    - `first(Field)` — Alias for unique, keep first occurrence
    - `last(Field)` — Keep last occurrence of each unique field value
  - Grouping stage:
    - `group_by(Field, Aggregations)` — Group records by field with aggregations
    - Built-in aggregations: `count`, `sum(F)`, `avg(F)`, `min(F)`, `max(F)`, `first(F)`, `last(F)`, `collect(F)`
  - Sequential processing:
    - `reduce(Pred, Init)` — Fold all records into single result with custom reducer
    - `scan(Pred, Init)` — Like reduce but emits intermediate results
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_aggregation_stages.sh` — Integration tests (12 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Sorting Stages** - Ordering records at pipeline level
  - Field-based ordering:
    - `order_by(Field)` — Sort by field ascending
    - `order_by(Field, Dir)` — Sort by field with direction (asc/desc)
    - `order_by(FieldSpecs)` — Sort by multiple fields with individual directions
  - Custom comparator:
    - `sort_by(ComparePred)` — Sort using user-defined comparison function
  - Key distinction: `order_by` is declarative (fields), `sort_by` is programmatic (comparator)
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_sorting_stages.sh` — Integration tests (12 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Error Handling Stages** - Resilient data processing with error recovery
  - Try-catch pattern:
    - `try_catch(Stage, Handler)` — Execute stage, route failures to handler
  - Retry logic:
    - `retry(Stage, N)` — Retry stage up to N times on failure
    - `retry(Stage, N, Options)` — Retry with delay and backoff options
    - Options: `delay(Ms)`, `backoff(linear)`, `backoff(exponential)`
  - Global error handling:
    - `on_error(Handler)` — Global error handler for the pipeline
  - Nested error handling: `try_catch(retry(...), fallback)` for complex recovery
  - Error records: Failed retries emit `{_error, _record, _retries}` for downstream handling
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_error_handling_stages.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Timeout Stage** - Time-limited stage execution
  - `timeout(Stage, Ms)` — Execute stage with time limit, emit error record on timeout
  - `timeout(Stage, Ms, Fallback)` — Execute stage with time limit, use fallback on timeout
  - Timeout record: `{_timeout, _record, _limit_ms}` for downstream handling
  - Combines with other error handling: `try_catch(timeout(...), handler)`
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_timeout_stage.sh` — Integration tests (12 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Rate Limiting Stages** - Throughput control for pipeline processing
  - `rate_limit(N, Per)` — Limit throughput to N records per time unit
    - Time units: `second`, `minute`, `hour`, `ms(X)`
    - Uses interval-based timing for precise rate control
  - `throttle(Ms)` — Add fixed delay of Ms milliseconds between records
  - Combines with other stages: `try_catch(rate_limit(...), handler)`, `timeout(rate_limit(...), ms)`
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_rate_limiting_stages.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Buffer and Zip Stages** - Record batching and stream combination
  - `buffer(N)` — Collect N records into batches for bulk processing
    - Flushes remaining records at stream end
  - `debounce(Ms)` — Emit record only after Ms quiet period (no new records)
    - Useful for smoothing bursty traffic
  - `zip(Stages)` — Run multiple stages on same input, combine outputs record-by-record
    - Enables parallel enrichment from multiple sources
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_buffer_zip_stages.sh` — Integration tests (18 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Window/Sampling/Partition Stages** - Stream windowing and data reduction
  - Window stages:
    - `window(N)` — Collect records into non-overlapping windows of size N
    - `sliding_window(N, Step)` — Create overlapping windows with step size
  - Sampling stages:
    - `sample(N)` — Randomly sample N records using reservoir sampling
    - `take_every(N)` — Take every Nth record (deterministic sampling)
  - Partition stage:
    - `partition(Pred)` — Split stream into matches and non-matches based on predicate
  - Take/Skip stages:
    - `take(N)` — Take first N records
    - `skip(N)` — Skip first N records
    - `take_while(Pred)` — Take records while predicate is true
    - `skip_while(Pred)` — Skip records while predicate is true
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_window_sampling_stages.sh` — Integration tests (32 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Distinct/Dedup Stages** - Duplicate removal at pipeline level
  - Global deduplication:
    - `distinct` — Remove all duplicate records, keeping first occurrence
    - `distinct_by(Field)` — Remove duplicates based on specific field value
  - Consecutive deduplication:
    - `dedup` — Remove consecutive duplicate records only
    - `dedup_by(Field)` — Remove consecutive duplicates based on specific field
  - Key differences:
    - `distinct` uses hash set (memory: O(n) for seen records)
    - `dedup` only compares adjacent records (memory: O(1))
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_distinct_dedup_stages.sh` — Integration tests (22 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Interleave/Concat Stages** - Stream combination at pipeline level
  - Round-robin interleaving:
    - `interleave(Stages)` — Alternate records from multiple stage outputs in round-robin fashion
    - Takes one record from each stream in turn until all exhausted
  - Sequential concatenation:
    - `concat(Stages)` — Concatenate multiple stage outputs sequentially
    - Yields all records from first stage, then second, etc.
  - Use cases:
    - `interleave` — Merge multiple data sources with fair ordering
    - `concat` — Combine results from different transformations
  - Composable with other stages: `distinct`, `filter_by`, `window`, `parallel`, etc.
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_interleave_concat_stages.sh` — Integration tests (18 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Merge Sorted Stage** - Efficient k-way merge for pre-sorted streams
  - Merge pre-sorted streams:
    - `merge_sorted(Stages, Field)` — Merge streams sorted by field (ascending)
    - `merge_sorted(Stages, Field, Dir)` — Merge with direction (asc/desc)
  - Efficient k-way merge algorithm:
    - Python: Heap-based merge using `heapq`
    - Go: Index-tracking merge with type comparison
    - Rust: Iterator-based merge with `serde_json::Value` comparison
  - Use cases:
    - Merging time-series data from multiple sources
    - Combining sorted partitions for final output
    - Efficient merge phase in external sort
  - Assumes input streams are already sorted by the specified field
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_merge_sorted_stage.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Tap Stage** - Observe stream without modification for side effects
  - Side-effect observation:
    - `tap(Pred)` — Execute side effect predicate for each record without modifying stream
    - `tap(Pred/Arity)` — Explicit arity specification supported
  - Use cases:
    - Logging pipeline progress
    - Collecting metrics and telemetry
    - Debugging intermediate values
    - Audit trail generation
  - Error isolation: Side effect errors don't interrupt the pipeline
    - Python: Exception handling with pass
    - Go: defer/recover pattern
    - Rust: std::panic::catch_unwind
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_tap_stage.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Flatten Stage** - Flatten nested collections into individual records
  - Collection flattening:
    - `flatten` — Flatten nested lists/arrays into individual records
    - `flatten(Field)` — Flatten a specific field within each record, expanding arrays
  - Behavior:
    - Simple flatten: Records containing `__items__` arrays are expanded
    - Field flatten: Records where field contains an array become multiple records
  - Use cases:
    - Expanding nested JSON arrays
    - Normalizing denormalized data
    - Processing hierarchical structures
    - Exploding array fields for analysis
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_flatten_stage.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Debounce Stage** - Rate-limit noisy streams by emitting only after silence
  - Debounce variants:
    - `debounce(Ms)` — Emit record only after Ms milliseconds of silence
    - `debounce(Ms, Field)` — Use specified timestamp field for timing
  - Behavior:
    - Groups records by time windows
    - Emits only the last record when silence period is reached
    - Useful for suppressing rapid successive updates
  - Use cases:
    - Rate-limiting sensor data
    - Suppressing duplicate events
    - Coalescing rapid updates
    - Smoothing noisy time-series data
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_debounce_stage.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Branch Stage** - Conditional routing within pipeline
  - Branch syntax:
    - `branch(Cond, TrueStage, FalseStage)` — Route records based on condition
    - `branch(Cond/Arity, TrueStage, FalseStage)` — With explicit arity
  - Behavior:
    - Records matching condition go through TrueStage
    - Records not matching go through FalseStage
    - Results from both branches are combined in output
    - Supports nested branches and complex sub-stages
  - Use cases:
    - A/B processing paths
    - Conditional transformations
    - Error vs success routing
    - Type-based record handling
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_branch_stage.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **Pipeline Tee Stage** - Fork stream to side destination while passing through
  - Tee syntax:
    - `tee(Stage)` — Run Stage as side effect, discard results, pass original through
  - Behavior:
    - Like Unix `tee` - fork stream to side destination
    - Side stage receives full stream (not per-record like tap)
    - Side stage results are discarded
    - Original records pass through unchanged
    - Side effect errors don't interrupt main pipeline
  - Comparison with tap:
    - `tap(Pred)` — Per-record side effect function
    - `tee(Stage)` — Full pipeline stage as side effect
  - Use cases:
    - Writing to log files while continuing processing
    - Sending copies to monitoring systems
    - Archiving data streams
    - Audit trails and metrics collection
  - Supported targets: Python, Go, Rust
  - `tests/integration/test_tee_stage.sh` — Integration tests (16 tests)
  - Documentation in `docs/ENHANCED_PIPELINE_CHAINING.md`

- **XML Data Source Playbook** - A new playbook for processing XML data.
  - `playbooks/xml_data_source_playbook.md` - The playbook itself.
  - `playbooks/examples_library/xml_examples.md` - The implementation of the playbook.
  - `docs/development/testing/playbooks/xml_data_source_playbook__reference.md` - The reference document for reviewers.
  - Updated `docs/EXTENDED_README.md` to include the new playbook.

- **Client-Server Architecture Phase 1: In-Process Services** - Foundation for service-oriented pipeline composition
  - Service Definition DSL:
    - `service(Name, HandlerSpec)` — Define a stateless service with operations
    - `service(Name, Options, HandlerSpec)` — Define service with options (stateful, transport, timeout)
  - Service Operations:
    - `receive(Var)` — Bind incoming request to variable
    - `respond(Value)` — Send response to caller
    - `respond_error(Error)` — Send error response
    - `state_get(Key, Value)` — Get state value (stateful services)
    - `state_put(Key, Value)` — Set state value (stateful services)
    - `call_service(Name, Request, Response)` — Call another service
    - `transform(In, Out, Goal)` — Transform data with predicate
    - `branch(Cond, TrueOps, FalseOps)` — Conditional execution
    - `route_by(Field, Routes)` — Route by field value
  - Service Options:
    - `stateful(Bool)` — Enable/disable state management
    - `transport(Type)` — Transport type (in_process, unix_socket, tcp, http)
    - `protocol(Format)` — Wire format (jsonl, json, messagepack, protobuf)
    - `timeout(Ms)` — Request timeout in milliseconds
    - `max_concurrent(N)` — Maximum concurrent requests
    - `on_error(Handler)` — Error handler predicate
  - Pipeline Integration:
    - `call_service(Name, RequestExpr, ResponseVar)` — Pipeline stage for service calls
    - `call_service(Name, Request, Response, Options)` — With options (timeout, retry, fallback)
    - Call service options: `timeout(Ms)`, `retry(N)`, `retry_delay(Ms)`, `fallback(Value)`
  - Multi-Target Compilation:
    - Python: Service classes with `_services` registry
    - Go: Service interfaces with struct implementations
    - Rust: Service trait with lazy_static registration
  - Validation:
    - `src/unifyweaver/core/service_validation.pl` — Service definition validation
    - Extended `src/unifyweaver/core/pipeline_validation.pl` — call_service stage validation
  - `tests/integration/test_in_process_services.sh` — Integration tests (13 tests)
  - Documentation in `docs/CLIENT_SERVER_DESIGN.md`

- **Client-Server Architecture Phase 2: Cross-Process Services** - Unix socket transport for inter-process communication
  - Unix Socket Server:
    - `transport(unix_socket(Path))` — Service option for Unix socket transport
    - Multi-threaded connection handling
    - JSONL request/response protocol with `_id`, `_payload`, `_status` fields
    - Graceful shutdown with signal handling (SIGINT, SIGTERM)
    - Configurable timeout per connection
  - Unix Socket Client:
    - Connection pooling with automatic reconnect
    - Request/response correlation via `_id`
    - Error propagation with structured error responses
    - Convenience functions for one-shot calls
  - Service Helpers:
    - `get_service_transport/2` — Extract transport from service definition
    - `get_service_protocol/2` — Extract protocol from service definition
    - `get_service_timeout/2` — Extract timeout from service definition
    - `is_cross_process_service/1` — Check if service uses Unix sockets
    - `is_network_service/1` — Check if service uses network transport
  - Multi-Target Support:
    - Python: `socket.AF_UNIX`, threading, JSONL via `json` module
    - Go: `net.Listen("unix", ...)`, goroutines, `encoding/json`
    - Rust: `std::os::unix::net::UnixListener`, threads, `serde_json`
  - Stateful Services:
    - Thread-safe state with locks (Python: `threading.Lock`, Go: `sync.Mutex`, Rust: `RwLock`)
    - State persists across connections for stateful services
  - `tests/integration/test_unix_socket_services.sh` — Integration tests (18 tests)
  - Documentation in `docs/CLIENT_SERVER_DESIGN.md`

## [0.1] - 2025-11-15

### 🎉 Milestone Release: Initial Vision Achieved

This release represents the completion of UnifyWeaver's initial design vision: a multi-target Prolog compiler with comprehensive data source integration, cross-platform support, and production-ready tooling.

### Added

#### C# Compilation Target
- **LINQ-style query compilation** - Prolog predicates compile to C# with LINQ expressions
- **Automated testing infrastructure** - C# query target tests with .NET integration
- **Multi-target architecture** - Template-based system supports Bash and C# targets

#### LLM Workflow System
- **Declarative agent orchestration** - Prolog-based workflow definitions for AI agents
- **Comprehensive documentation** - Playbook authoring guides and best practices
- **Example library** - Production-ready workflow templates

#### Platform Compatibility Enhancements
- **Platform detection** - Automatic detection of Windows/WSL/Linux/macOS/Docker
- **Smart workarounds** - Platform-specific adaptations for known limitations
- **Emoji rendering** - Platform-aware Unicode/BMP/ASCII emoji support

#### Testing Infrastructure
- **Complete integration tests** - Full ETL pipeline validation across platforms
- **Cross-platform test plans** - Linux and PowerShell test environments
- **Automated test discovery** - Pattern-based test runner generation

### Fixed
- **Firewall configuration** - Corrected file_read_patterns, cache_dirs, and python_modules in data_sources_demo.pl
- **PowerShell integration** - Platform detection workaround for SQLite test cumulative state issue
- **Test reliability** - Improved cross-platform test stability

### Changed
- **Version updates** - All examples and documentation updated to v0.1
- **POST_RELEASE_TODO** - Updated to target v0.2 improvements

### Known Limitations
- PowerShell SQLite integration test skipped due to cumulative state issue (works in isolation and on Linux)
- C# test cleanup permission errors on Dropbox/WSL (low priority, doesn't affect functionality)

See `POST_RELEASE_TODO.md` for planned post-v0.1 improvements.

## [0.0.2] - 2025-10-14

### Added

#### Data Source Plugin System
- **Complete data source architecture** - Plugin-based system for external data integration
  - 4 production-ready data source plugins: CSV, Python, HTTP, JSON
  - Self-registering plugin architecture following established patterns
  - Template-based bash code generation with comprehensive error handling
  - Configuration validation with detailed error messages

#### New Data Source Plugins

- **CSV/TSV Source** (`src/unifyweaver/sources/csv_source.pl`)
  - Auto-header detection from first row with intelligent column mapping
  - Manual column specification for headerless files
  - Quote handling (strip/preserve/escape) for embedded delimiters
  - TSV support via configurable delimiter
  - Skip lines configuration for comments/metadata

- **Python Embedded Source** (`src/unifyweaver/sources/python_source.pl`)
  - Heredoc `/dev/fd/3` pattern for secure Python code execution
  - SQLite integration with automatic query generation and connection management
  - Inline Python code support with comprehensive error handling
  - External Python file loading and execution
  - Configurable timeout and multiple input modes (stdin, args, none)

- **HTTP Source** (`src/unifyweaver/sources/http_source.pl`)
  - curl and wget support with intelligent tool selection
  - Response caching with configurable TTL and cache management
  - Custom headers, POST data, and query parameter support
  - Cache invalidation and inspection utilities
  - Network timeout management and error handling

- **JSON Source** (`src/unifyweaver/sources/json_source.pl`)
  - jq integration with flexible filter expressions
  - File and stdin input modes for pipeline compatibility
  - Multiple output formats (TSV, JSON, raw, CSV) with @csv/@tsv support
  - Custom filter chaining and composition
  - Flag management (raw, compact, null input)

#### Enhanced Security Framework

- **Multi-Service Firewall** - Extended `src/unifyweaver/core/firewall.pl`
  - Multi-service validation (python3, curl, wget, jq, awk)
  - Network access control with host pattern matching and wildcards
  - Python import restriction system with regex parsing
  - File access patterns for read/write operations
  - Cache directory validation with pattern matching
  - Backward-compatible extensions to existing firewall predicates

- **New Security Policy Terms**
  ```prolog
  - network_access(allowed|denied) - Control HTTP source access
  - network_hosts([pattern1, pattern2, ...]) - Whitelist host patterns  
  - python_modules([module1, module2, ...]) - Restrict Python imports
  - file_read_patterns/file_write_patterns - Control file access
  - cache_dirs([dir1, dir2, ...]) - Restrict cache locations
  ```

#### Comprehensive Testing Suite

- **Unit Tests** - Individual plugin validation
  - `tests/core/test_csv_source.pl` - CSV parsing, headers, TSV support
  - `tests/core/test_python_source.pl` - Heredoc pattern, SQLite, file integration
  - `tests/core/test_firewall_enhanced.pl` - Enhanced security validation

- **Integration Tests** - `tests/core/test_data_sources_integration.pl`
  - Cross-plugin pipeline testing (CSV → Python, HTTP → JSON)
  - Multi-source firewall validation
  - Real-world ETL scenario testing (GitHub API → SQLite)
  - Complete system integration verification

#### Production Examples and Documentation

- **Complete Demo** - `examples/data_sources_demo.pl`
  - Real ETL pipeline: JSONPlaceholder API → JSON parsing → SQLite storage
  - CSV user data processing with auto-header detection
  - Multi-service firewall configuration showcase
  - Interactive and command-line execution modes

- **Implementation Plan** - `docs/DATA_SOURCES_IMPLEMENTATION_PLAN.md`
  - Complete architectural design and implementation timeline
  - Technical specifications and usage examples
  - Real-world use cases and best practices

### Technical Implementation

#### Architecture Features
- **Plugin Registration System** - Self-registering plugins with `:- initialization`
- **Template Integration** - Seamless integration with UnifyWeaver's template system
- **Configuration Validation** - Comprehensive validation with detailed error messages
- **Security-First Design** - Enhanced firewall with deny-by-default approach

#### Code Quality
- **2,000+ lines** of production-ready, thoroughly tested code
- **Comprehensive error handling** throughout all components
- **Proper documentation** with extensive inline comments
- **Follows UnifyWeaver conventions** and coding standards

### Files Added

```
docs/DATA_SOURCES_IMPLEMENTATION_PLAN.md    | 514 +++++++++++++++++
examples/data_sources_demo.pl               | 199 +++++++
src/unifyweaver/sources/csv_source.pl       | 300 ++++++++++
src/unifyweaver/sources/http_source.pl      | 344 +++++++++++
src/unifyweaver/sources/json_source.pl      | 285 ++++++++++
src/unifyweaver/sources/python_source.pl    | 310 ++++++++++
tests/core/test_csv_source.pl               | 108 ++++
tests/core/test_python_source.pl            | 124 ++++
tests/core/test_firewall_enhanced.pl        | 156 +++++
tests/core/test_data_sources_integration.pl | 201 +++++++
```

### Enhanced Existing Files

```
src/unifyweaver/core/firewall.pl            | 250 +++++++++++++++
```

### Compatibility

- **100% backward compatible** - No breaking changes to existing functionality
- **Additive enhancements only** - All existing predicates and behavior preserved
- **Self-contained plugins** - No impact on existing components
- **Safe firewall extensions** - New validation predicates don't affect existing rules

### Contributors
- John William Creighton (@s243a) - Project architecture and design guidance
- Cline (Claude 3.5 Sonnet) - Implementation of complete data source plugin system

## [0.0.1-alpha] - 2025-10-12

### Added
- **Pattern exclusion system** - `forbid_linear_recursion/1` to force alternative compilation strategies
  - Manual forbidding: `forbid_linear_recursion(pred/arity)`
  - Automatic forbidding for ordered constraints (`unordered=false`)
  - Check if forbidden: `is_forbidden_linear_recursion/1`
  - Clear forbid: `clear_linear_recursion_forbid/1`
  - Enables graph recursion with fold helpers for predicates like fibonacci
- **Educational materials workflow** - Support for Chapter 4 tutorial
  - `test_runner_inference.pl` accepts `output_dir(Dir)` option
  - Fixed module imports in `recursive_compiler.pl` for education library alias

### Changed
- **Linear recursion pattern** - Now handles 1+ independent recursive calls (not just single calls)
  - Fibonacci now compiles as linear recursion with aggregation (not tree recursion)
  - Added independence checks: arguments must be pre-computed, no inter-call data flow
  - Added structural pattern detection to distinguish tree recursion (pattern matching) from linear with aggregation (computed scalars)
  - Tests: `fibonacci/2` now uses linear recursion, `tree_sum/2` correctly identified as tree recursion

### Fixed
- **Fibonacci test isolation** - Removed from tree_recursion tests (belongs in linear_recursion)
- **Test runner inference** - Excluded `parse_tree` helper from being tested directly
- **Integration tests** - Fixed module context for test predicates (use `user:` prefix)
- **Pattern detection** - Tree recursion properly distinguished from linear with multiple calls

### Documentation
- Added `RECURSION_PATTERN_THEORY.md` - Comprehensive theory document explaining pattern distinctions
  - Detailed `forbid_linear_recursion` system documentation
  - Graph recursion with fold helper pattern
- Updated `ADVANCED_RECURSION.md` - Reflects new linear recursion behavior with examples
  - Pattern exclusion documentation
- Updated `README.md` - Corrected fibonacci classification and added pattern exclusion feature

## [1.0.0] - 2025-10-11

### Added

#### Core Features
- **Stream-based compilation** - Memory-efficient compilation using bash pipes and streams
- **Template system** - Mustache-style template rendering with file loading and caching
- **Constraint analysis** - Automatic detection of unique and ordering constraints
- **BFS optimization** - Transitive closures automatically optimized to breadth-first search
- **Cycle detection** - Proper handling of cyclic graphs without infinite loops
- **Control plane** - Firewall and preferences system for policy enforcement

#### Advanced Recursion Patterns
- **Tail recursion optimization** - Converts tail-recursive predicates to iterative bash loops
  - Accumulator pattern detection
  - Iterative loop generation
  - Tests: `count_items/3`, `sum_list/3`

- **Linear recursion** - Memoized compilation for single-recursive-call patterns
  - Automatic memoization with associative arrays
  - Pattern detection for exactly one recursive call per clause
  - Tests: `factorial/2`, `length/2`

- **Tree recursion** - Handles multiple recursive calls (2+)
  - List-based tree representation: `[value, [left], [right]]` or `[]`
  - Fibonacci-like pattern recognition
  - Binary tree operations (sum, height, count)
  - Bracket-depth tracking parser for nested structures
  - Tests: `fibonacci/2`, `tree_sum/2`

- **Mutual recursion** - Handles predicates calling each other cyclically
  - SCC (Strongly Connected Components) detection via Tarjan's algorithm
  - Shared memoization tables across predicate groups
  - Call graph analysis
  - Tests: `is_even/1`, `is_odd/1`

#### Module Structure
- `template_system.pl` - Template rendering engine
- `stream_compiler.pl` - Non-recursive predicate compilation
- `recursive_compiler.pl` - Basic recursion analysis and compilation
- `constraint_analyzer.pl` - Constraint detection and optimization
- `firewall.pl` - Policy enforcement for backend usage
- `preferences.pl` - Layered configuration management
- `advanced/` directory with specialized recursion compilers:
  - `advanced_recursive_compiler.pl` - Pattern orchestration
  - `call_graph.pl` - Predicate dependency graphs
  - `scc_detection.pl` - Tarjan's SCC algorithm
  - `pattern_matchers.pl` - Pattern detection utilities
  - `tail_recursion.pl` - Tail recursion compiler
  - `linear_recursion.pl` - Linear recursion compiler
  - `tree_recursion.pl` - Tree recursion compiler
  - `mutual_recursion.pl` - Mutual recursion compiler

#### Testing Infrastructure
- Comprehensive test suite with 28+ tests
- Auto-discovery test system
- Test environment with `init_testing.sh` and PowerShell support
- Individual test predicates: `test_stream`, `test_recursive`, `test_advanced`, `test_constraints`
- Bash test runners for generated scripts

#### Documentation
- README.md with comprehensive examples
- TESTING.md for test environment setup
- ADVANCED_RECURSION.md for recursion pattern details
- PROJECT_STATUS.md for roadmap tracking
- Implementation summaries in `context/` directory

### Fixed
- Module import paths in `stream_compiler.pl` (library → local paths)
- Singleton variable warning in `compile_facts_debug/4`
- Linear pattern matcher now correctly counts recursive calls
- Tree parser handles nested bracket structures properly

### Technical Details

#### Pattern Detection Priority
1. Tail recursion (simplest optimization)
2. Linear recursion (single recursive call)
3. Tree recursion (multiple recursive calls)
4. Mutual recursion (multi-predicate cycles)
5. Basic recursion (fallback with BFS)

#### Code Generation Features
- Associative arrays for O(1) lookups
- Work queues for BFS traversal
- Duplicate detection with process-specific temp files
- Stream functions for composition
- Memoization tables with automatic caching
- Iterative loops for tail recursion

#### Compilation Options
- `unique(true)` - Enables early exit optimization for single-result predicates
- `unordered(true)` - Enables sort-based deduplication
- Runtime options override declarative constraints

### Known Limitations
- Tree recursion uses simple list-based representation only
- Parser has limitations with deeply nested structures (addressed with bracket-depth tracking)
- Memoization disabled by default for tree recursion in v1.0
- Divide-and-conquer patterns not yet supported
- Bash 4.0+ required for associative arrays

### Contributors
- John William Creighton (@s243a) - Core development
- Gemini (via gemini-cli) - Constraint awareness features
- Claude (via Claude Code) - Advanced recursion system, test infrastructure

---

## Release Notes

### v1.0.0 Highlights

This is the initial stable release of UnifyWeaver, featuring:

**Complete Advanced Recursion Support:**
- 4 recursion pattern compilers (tail, linear, tree, mutual)
- Automatic pattern detection and optimization
- Comprehensive test coverage

**Production-Ready Features:**
- Stream-based compilation for memory efficiency
- Template system with file loading
- Constraint-aware code generation
- Policy enforcement via control plane

**Developer Experience:**
- Auto-discovery test environment
- Cross-platform support (Linux, WSL, Windows)
- Extensive documentation
- 28+ passing tests

### Upgrade Notes

This is the first release, so no upgrade path is needed.

### Future Roadmap

**v1.1:** Improved pattern detection and memoization defaults
**v1.2:** Better tree parsing and more complex tree structures
**v1.3:** C# backend with native type support
**v2.0:** Dynamic sources plugin system (AWK, SQL integration)

---

[1.0.0]: https://github.com/s243a/UnifyWeaver/releases/tag/v1.0.0
