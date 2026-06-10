# WAM Scala execution-mode benchmark: interpreter vs lowered vs kernel

Quantifies the speedup from the two acceleration paths added to the WAM
Scala hybrid target — the per-predicate **lowered emitter**
(`emit_mode(functions)`) and the native **graph kernels**
(`kernel_dispatch(true)`) — over the baseline step-loop **interpreter**.

Harness: [`tests/benchmarks/wam_scala_mode_bench.pl`](../tests/benchmarks/wam_scala_mode_bench.pl).
Each mode generates a self-contained SBT project, compiles it once with
`scalac`, then runs the query under the generated program's `--bench N`
inner-loop mode (JVM startup amortised; reported elapsed is dominated by
per-query runtime cost). Per-iteration time = elapsed / N.

```
swipl -g main -t halt tests/benchmarks/wam_scala_mode_bench.pl -- <N> <inner>
```

## Workload

Transitive closure over a chain `c0 → c1 → … → cN`:

```prolog
tc(X, Y) :- edge(X, Y).
tc(X, Y) :- edge(X, Z), tc(Z, Y).
```

Two query shapes:

- **deep** — `tc(c0, cN)`: reachability across the whole chain. The
  graph-algorithm shape the kernels target; the recursive clause is the
  hot path.
- **base** — `tc(c0, c1)`: a single direct edge — clause 1, which the
  lowered fast path handles without falling back to the interpreter.

## Results

Measured with Scala 3.3.4 / OpenJDK 21. Per-iteration seconds; speedup is
relative to the interpreter (higher is faster).

### N = 100 (inner = 3000)

| query | interp | lowered | kernel |
|---|---|---|---|
| deep `tc(c0,c100)` | 0.000409 (1.00×) | 0.000454 (0.90×) | 0.000106 (**3.85×**) |
| base `tc(c0,c1)`   | 0.000030 (1.00×) | 0.000031 (0.98×) | 0.000086 (0.35×) |

### N = 300 (inner = 2000)

| query | interp | lowered | kernel |
|---|---|---|---|
| deep `tc(c0,c300)` | 0.001818 (1.00×) | 0.001829 (0.99×) | 0.000194 (**9.39×**) |
| base `tc(c0,c1)`   | 0.000041 (1.00×) | 0.000038 (1.06×) | 0.000133 (0.31×) |

## Interpretation

1. **Kernels are the real lever for graph traversal, and the win grows
   with depth.** On the deep query the native BFS handler is **3.85× at
   N=100 and 9.39× at N=300** — the interpreter pays an O(depth) cost with
   a high per-WAM-step constant (dispatch, unification, choice points,
   trail), while the kernel does a native BFS with a low constant. Deeper
   chains amplify the gap. This matches the cross-target story in
   [`wam_effective_distance_cross_target.md`](wam_effective_distance_cross_target.md):
   kernel-based lowering produces the dramatic wins.

2. **Kernels have a fixed setup cost that loses on trivial queries.** On
   the base query the kernel is *slower* (0.31–0.35×): it always builds
   the full adjacency map and computes every node reachable from the
   source, then filters — overkill for a one-hop lookup. Kernels pay off
   on deep / whole-relation traversals, not shallow probes. (`kernel_dispatch`
   is opt-in for exactly this reason.)

3. **The lowered emitter is roughly neutral on recursion-heavy
   predicates.** For `tc/2` the recursive clause is not lowered (only
   clause 1 is; the recursion falls back to the interpreter), so
   `emit_mode(functions)` lands within a few percent of the interpreter —
   sometimes marginally slower on the deep query (the failed clause-1
   fast-path attempt before the interpreter fallback), sometimes marginally
   faster on the base query (clause 1 handled inline, no fallback). This is
   consistent with the roadmap's assessment that *"WAM-instruction lowering
   alone is not enough … per-predicate native fast-path emitters close part
   of the gap. Useful, but bounded."* The lowered emitter's benefit is
   largest for **single-clause, inline-deterministic** predicates
   (arithmetic, unification, fixed goal sequences) that run entirely in the
   emitted function with no interpreter fallback — a shape `tc/2` does not
   exercise.

## Takeaways

- For graph algorithms, reach for `kernel_dispatch(true)` — multiple-× and
  growing with scale.
- `emit_mode(functions)` is a low-risk, behaviour-preserving option whose
  upside depends on predicate shape; it does not regress correctness
  (interpreter fallback) and helps inline-deterministic predicates.
- Both are opt-in; the default interpreter remains the most general path.

Numbers are indicative single-run measurements on one machine; rerun the
harness in your environment for absolute figures. The *shape* (kernel win
growing with depth; kernel overhead on shallow queries; lowered ≈ neutral
for recursion) is the durable result.
