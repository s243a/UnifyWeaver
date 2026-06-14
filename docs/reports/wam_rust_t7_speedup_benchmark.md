# T7 parallel aggregates — wall-clock speedup (end-to-end)

**Result:** the fully-compiled route-1 path delivers a real wall-clock speedup
through the actual generated Rust — not just matching results. This closes the
original requirement ("make sure it's actually a performance gain, since Rust
already has parallel capabilities").

## Setup

A user aggregate compiled with `parallel_aggregates(true)` (so it goes through
the real injection → `__par_enum`/`__par_body` helpers → native `par_collect`):

```prolog
bench_fib(0,0). bench_fib(1,1).
bench_fib(N,F) :- N>1, N1 is N-1, N2 is N-2, bench_fib(N1,F1), bench_fib(N2,F2), F is F1+F2.
bench_body(X,F) :- N is BASE + (X mod 3), bench_fib(N,F).   % expensive recursive per-branch
bench_collect(L) :- findall(F, (bench_fact(X), bench_body(X,F)), L).   % 32 facts
```

Measured in the generated project (`--release`) by timing `seq_collect` vs
`par_collect` over the same generated predicate functions, asserting equal
results. Machine: 4 cores. Test: `tests/test_wam_rust_parallel_speedup.pl`.

## Numbers

| Per-branch work | sequential | parallel | speedup |
|---|---:|---:|---:|
| `BASE=20` (fib 20–22), committed test | 14.46 s | 4.26 s | **3.39×** |
| `BASE=22` (fib 22–24), heavier | 38.68 s | 17.88 s | **2.16×** |

Both: n=32, parallel result-set == sequential. 4 cores → ideal 4×; the gap is the
sequential enumerator pass, per-branch machine clone, and thread/merge overhead
(more visible at the heavier per-branch size, consistent with the substrate's
2–3.7× range in `wam_rust_t7_parallel_perf.md`).

## Why this is the meaningful measurement

It runs through the **whole compiled path** a user gets — cost gate → split →
transform → generated helper WAM functions → native `par_collect` over cloned
machines — not a synthetic microbenchmark. The cost gate is what makes this safe
in general: cheap-bodied aggregates stay sequential (no fork regression), only
expensive/recursive ones fan out, which is exactly the case measured here.

## Caveats

- The committed test is heavy (~20 s run + a `--release` build); it is
  `cargo`-gated and asserts a speedup only with ≥ 2 cores.
- Absolute times reflect the interpreted WAM (naive fib is deliberately costly to
  create expensive branches); the **ratio** is the point.
