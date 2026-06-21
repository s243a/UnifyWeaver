# T9 fact-table inline (Rust) — benchmark

Quantifies the T9 fact-table inline path (static row table + first-arg hash
index + choice-point enumeration) against the default T4 path (the ground facts
compiled into the shared WAM instruction table and enumerated by the
interpreter). Parallels `wam_rust_t7_speedup_benchmark.md`.

All measurements on this container; release builds; `edge(Key, Value)` fact sets
with distinct string keys (`k0..kN-1`). Reproduce: `output/bench_t9` /
`output/bench_t4` generators in the session, and the committed
`tests/test_wam_rust_fact_table_throughput.pl` (T9 capstone, cargo-gated).

## Headline

For ground-fact predicates, T9 is the clear win, but **the decisive axes are code
size, compile time, and correctness — not raw per-lookup latency**:

| Axis | T4 (WAM shared table) | T9 (fact table) |
|------|----------------------|-----------------|
| WAM instructions emitted (per fact) | ~4 (one giant `vec!`) | **0** (data table) |
| Release compile, 500 facts | **≈14 min** (pathological `vec!` opt) | seconds |
| Release compile, 2000 facts | did **not finish** in 10 min | seconds |
| Correctness, many keys | **drops some keys** (see below) | all keys correct |
| Point-lookup latency | n/a (unsound baseline) | **~8 µs/query** |

## Code size (instructions emitted)

T9 emits **zero** interpreter instructions for a pure fact predicate — the rows
are a `static` data table built once via `OnceLock`. T4 lowers every clause into
the shared instruction `Vec<Instruction>`:

| facts | T9 `lib.rs` lines | T9 WAM instrs | T4 `lib.rs` lines | T4 WAM instrs |
|------:|------------------:|--------------:|------------------:|--------------:|
|    64 |               140 |         **0** |               447 |           257 |
|   256 |               332 |         **0** |             1 599 |         1 025 |
|  1024 |             1 100 |         **0** |             6 207 |         4 097 |

T4 grows at ~4 instructions/fact, all inside a single `vec![ ... ]` literal.

## Compile time

The T4 instruction `vec!` is the bottleneck: `rustc` optimises a multi-thousand
element array literal pathologically slowly in release.

- **500 facts**: the T4 release `cargo test` took **867 s** wall (≈14.5 min,
  dominated by codegen); the equivalent T9 build+run finished in **~1 s**.
- **2000 facts**: the T4 release build **did not finish within a 10-minute
  budget** and was abandoned; T9 built and ran in seconds.

This alone makes T9 the only practical option for large fact sets in release.

## Runtime (T9, correct)

Measured through the generated code, release:

- **Point lookup** (committed throughput test): 200 unique keys, 100 000
  point lookups via `edge_2` + a `backtrack()` exhaustion check —
  **8.045 µs/query**, every key returns its single correct row, no dropped keys,
  no spurious extra solution.
- **`findall` point lookup** (create vm + clone the tiny shared program +
  `findall/3` of one row, 1 000 000 queries): **16.3 µs/query** — the extra cost
  over the direct call is the aggregate frame + per-call program clone.
- **Bulk `findall`** (50 rows/key, atom keys, correct): **~550 µs/query ≈
  11 µs/row**.

Note: the per-call program clone in the generated `pub fn` wrappers is paid by
both paths; T9's is negligible because its shared program is ~0 instructions,
whereas T4 clones its multi-thousand-instruction program per call — another
structural T9 advantage that the runtime numbers understate (T4 could not be
timed as a sound baseline; see below).

## Correctness — why T4 is not a sound baseline

At the time of this benchmark the default T4 shared-table path **silently dropped
some keys**. With 500 distinct keys, `findall(X, edge(k5, X), L)` returned
`L = []` while `k0`, `k1`, `k250`, `k499` resolved correctly — a first-argument
`switch_on_constant` indexing defect: the runtime binary-searched Atom keys
assuming a lexically sorted table, but the table is emitted in clause order
(`k0,k1,..,k9,k10,..`, which is not lexically sorted), so it mis-navigated and
dropped keys whose clause order differed from lexical order. **Fixed** (the
`SwitchOnConstant`/`SwitchOnConstantPc` handlers now linear-scan by value
equality; regression: `tests/test_wam_rust_switch_on_constant_keydrop_exec.pl`).
At the time, this meant a like-for-like T4 speed ratio was not meaningful, so the
comparison here is correctness + code size + compile time.

The committed throughput test asserts every one of 200 keys resolves to its
exact row (and that no extra solution remains), so it also guards against this
class of regression on the T9 path.

## Conclusion

T9 fact-table inline turns a ground-fact predicate from ~4 interpreter
instructions/fact (in a release-hostile `vec!`) into a zero-instruction static
table with an O(1) first-arg index. The win is: **compiles in seconds instead of
minutes-to-never, emits no interpreter code, and is correct at scale**, at a
point-lookup latency of ~8 µs/query. Recommended for any sizeable ground-fact
predicate; gated behind `fact_table_inline(true)` so default output is unchanged.

The T4 `switch_on_constant` key-dropping defect surfaced here has since been
fixed (linear scan by value equality), so the WAM path is now correct for
many-key fact sets too.
