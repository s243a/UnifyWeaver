# Go WAM lowered dispatch: T5 cascade vs T6 string switch

Benchmark backing the gated **T6 first-argument indexing** back-end for the Go
WAM target (`wam_go_lowered_emitter.pl`). Companion to
`docs/reports/wam_rust_dispatch_alloc_perf.md` (Rust/C++).

## What is measured

A predicate `shade/1` with **N** distinct single-atom facts is lowered two ways:

- **T5** — the linear `if valueEquals(t5a1, atomK) { … }` cascade (forced with
  `t6_min_clauses(999999)`).
- **T6** — a single type-assertion to `*Atom` then a native Go
  `switch t6atom.Name { case "…": … }` (the default, gate fires at ≥ 8 atoms).

The benchmark loops over **all N keys plus one miss**, so each measurement is the
average dispatch cost across every clause position (the cascade's worst case is
the last clause / the miss; its average is ~N/2 comparisons).

`go test -bench`, `go1.24.7 linux/amd64`, Intel Xeon @ 2.10 GHz, `-benchtime=1s`.

## Results (ns/op, lower is better)

| N (clauses) | T5 cascade | T6 string switch | speedup |
|------------:|-----------:|-----------------:|--------:|
|           8 |      22.08 |            4.594 |   4.81× |
|          64 |      150.9 |            4.760 |  31.70× |
|         256 |      513.0 |            8.721 |  58.82× |

## Reading the result

- The T5 cascade is **O(N)** — each clause is a `valueEquals` call (an interface
  type-assertion plus a comparison), so cost grows linearly: 22 → 151 → 513 ns.
- The T6 switch is **O(1)** — one type-assertion, then a Go string `switch` the
  compiler lowers to a length/hash jump: ~4.6–8.7 ns, essentially flat.
- Unlike an *integer* equality chain (which Go and most backends would
  switch-convert anyway — the "lost to the compiler" case), Go does **not**
  rewrite the `valueEquals` interface-call chain, so the explicit switch is a
  real win. Go atoms are `&Atom{Name string}` interned by name, i.e. the
  discriminator is string-keyed — Go is an **atom-keyed** target (like
  Rust/C++/F#), not int-interned.
- The win already pays off at the default gate of 8 clauses (4.8×) and is large
  by 64 (32×); below the gate the cascade is fine and the compiler would just
  flatten a small switch back to comparisons, so T6 stays gated.
