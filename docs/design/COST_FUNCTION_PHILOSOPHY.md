# Cost Function — Philosophy

## What this is

A design-space discussion for the cost functions used by the
cache-warming tree builder (see
`docs/design/SCAN_STRATEGY_SPECIFICATION.md`). The spec doc is
the formal contract; this doc is the *why* — the theoretical
landscape, the trade-offs between variants, and the convergence
spectrum.

Cost functions score nodes for "relevance to the query
endpoints". They feed the bounded-heap eviction policy during
warm-build: the K nodes with the best scores survive into the
snapshot cache. Different mathematical forms encode different
*models* of how relevance disperses through the graph.

## Why approximation suffices

Cache warming is a **ranking problem, not a correctness problem**.

We use the cost function to pick the top ~10% of nodes by score.
The cache is initialised with those nodes. As long as two
ranking schemes agree on which nodes land in the top 10%, they
are *equivalent for our purpose*, even if they disagree on
relative ordering inside the top 10% or absolute score values
in the long tail.

This buys a substantial concession: we don't need to converge to
an exact solution. We don't need PDE-level accuracy. We don't
need fixed-point iteration to ε. A reasonable approximation —
often a single sweep — gives the same top-K membership as the
exact answer, at a fraction of the cost.

This is *not* a license to be sloppy in general. It's a
workload-specific observation: for cache warming, exact
convergence buys precision in the long tail, which is exactly
what we discard.

## Terminology note

The math we're doing is well-studied under several names:

| field | what it covers |
| --- | --- |
| Iterative numerical methods | general framework for fixed-point iteration; Jacobi, Gauss-Seidel, power iteration |
| Graph signal processing | modern term for propagating scalar/vector fields over graph topology |
| Graph diffusion | physics-flavoured name for the same propagation; analogous to heat-flow on graphs |
| Markov processes / random walks | probabilistic dual; stationary distributions of certain walks correspond to flux measures |
| Personalized PageRank (PPR) | specific algorithm for computing stationary distribution biased by a seed set |

A note on what this *isn't*: **geometric algebra** (Clifford
algebras — multivectors, geometric products, exterior algebra)
is a related-but-different mathematical framework. It's the
right vocabulary for some problems (rotors, oriented subspaces,
unified scalar+vector+bivector ops), and unsurprisingly comes up
in adjacent UnifyWeaver work like the density explorer. For
cost functions on graphs we're closer to graph diffusion and
iterative solvers; the variants discussed below are scalar
fields propagated along graph edges, not multivectors.

## The framework

Given a graph G = (V, E) and an endpoint set E ⊆ V (seeds and/or
roots), we want a score s: V → ℝ representing each node's
relevance to the endpoints.

Different cost functions are different choices of:

1. **What does "flux" mean?** Probabilistic mass? Geometric
   attenuation? Distance? Embedding similarity?
2. **How does it propagate?** Multiplicative decay per hop?
   Inverse-radial spread? Additive path probability?
3. **How is it aggregated** when a node has multiple paths from
   the endpoints? Sum (true flux)? Max (greedy)? Min (shortest)?
   Weighted average?
4. **When does iteration stop?** Single pass? N rounds? Local
   ε-convergence? Global ε-convergence?

The variants below pick combinations of these choices.

## Variants

### Hop distance

```
s(n) = min over paths from endpoints of (length of path to n)
```

The simplest variant. No "flux" interpretation — just shortest-
path distance.

**Computational profile**: one BFS sweep from the endpoint set,
O(V + E). Stops naturally; no convergence to worry about.

**Per node**: one Int (the hop count).

**Strength**: dirt cheap, no per-graph tuning, always defined.

**Weakness**: very coarse. A node reachable by 1 path at hop 3
ties with a node reachable by 1000 paths at hop 3. Information
about path multiplicity is discarded entirely. Needs a
tiebreaker for ranking (we use `node_id` lexicographically).

**Aggregation**: only `min` makes sense.

### Exponential decay flux

```
flux(n) = Σ over paths p from endpoints to n of (decay/branching)^|p|
```

Multiplicative per-hop attenuation, with branching factor in the
denominator so high-fanout nodes contribute less per path. This
is what `SCAN_STRATEGY_SPECIFICATION.md`'s spec doc describes.

**Computational profile**: one iteration is O(V + E). Number of
iterations to convergence is on the order of graph diameter or
the user's chosen `iterations(N)`.

**Per node**: one Double for accumulated flux.

**Strength**: closed-form, easy to bound, intuitive ("each hop
loses some flux, branching shares it among neighbours").

**Weakness**: when paths converge at a node (multiple endpoints
reach it), the multiplicative form can lose fine structure —
geometric mean rather than true sum.

**Aggregation**: `sum` for exact flux; `max` for greedy
approximation (use the best single path).

### Power-law / inverse-radial flux

```
flux contribution per hop = 1 / r^N
where r is per-hop characteristic distance and N is branching
```

Models each node as a source emitting flux in N-dimensional
space; intensity at radial distance r falls as 1/r^N, like
Gauss's law. After H hops with branching N at each, total
decay is r^(−N·H).

In log space:
```
log flux = −N · log r  per hop  ⇒  contributions sum across walk
```

**Computational profile**: same as exponential, but log-space
computation avoids underflow at long walks and turns
multiplications into additions.

**Per node**: one Double (log flux).

**Strength**: physically motivated (inverse-square / inverse-N
intuition is well-grounded). Computationally clean in log space.

**Weakness**: penalty on high-branching is sharper than
exponential — may over-weight dead-end paths. Choice between
exponential and power-law is empirical, depends on which
matches the graph's actual flux behaviour.

**Aggregation**: `sum` (log-sum-exp for true sum), or `max` in
log space (simple max of log fluxes).

### Additive path-probability flux (PPR-style)

```
flux(n) = Σ over walks w from endpoints of P(walk reaches n)
        = harmonic measure at n with endpoints as boundary
        = Green's function G(source, n) of the graph Laplacian
```

Three equivalent definitions — all referring to the same object.
See *Unifying framework: Green's function on the graph Laplacian*
below for the underlying theory. This is what proper personalised
PageRank converges to; it's also the steady-state potential
distribution of a linear resistive network with the endpoints as
fixed-voltage boundaries.

**Computational profile**: iterates to ε-convergence. Many
rounds — typically until the largest per-node delta is below a
threshold. O((V + E) · iterations).

**Per node**: one Double (probability) + tracked deltas for the
ε-convergence test.

**Strength**: most rigorous; probability is *additive* by
definition of measure, so multi-path contributions combine
correctly without geometric-mean distortion.

**Weakness**: most expensive. ε-convergence may need many rounds
for graphs with long diameters.

**Aggregation**: `sum` (intrinsic to the definition).

### Semantic similarity

```
s(n) = cosine_similarity(embedding(n), query_embedding)
```

No propagation. Each node has an embedding vector; relevance is
the dot product with a query embedding. Used when the workload
provides a query as a vector (e.g. natural-language query
embedded via an LM, or a "show me things like X" embedding of
node X).

**Computational profile**: O(V · D) where D is embedding
dimension. One-shot — no iteration, no propagation.

**Per node**: D Doubles (the embedding).

**Strength**: captures latent similarity not visible in graph
structure. Works on disconnected components.

**Weakness**: requires embeddings, which require either an
existing table or a (potentially expensive) embedding model.

**Aggregation**: `none` (not a propagation model).

## Unifying framework: Green's function on the graph Laplacian

The propagating cost functions — exponential decay flux,
power-law flux, additive path-probability flux — all have a
common ancestor: they're approximations to the **Green's function
of the graph Laplacian** with the endpoints as boundary conditions.

### The setup

Treat the graph as a linear resistive network:

- Each edge `(u, v) ∈ E` has unit conductance (or weighted, if
  edge weights are available).
- The source set `S ⊆ V` (seeds) injects current at fixed voltage
  (or fixed current).
- The sink set `T ⊆ V` (roots) is grounded at `V = 0`.
- Node potentials `V: V → ℝ` satisfy Kirchhoff's current law at
  every interior node:

```
Σ_{w ∼ v} (V_v − V_w) = b_v
```

In matrix form: `L V = b`, where `L = D − A` is the graph
Laplacian (`D` = diagonal degree matrix, `A` = adjacency),
and `b` is the boundary-condition / current-injection vector.

### What the Green's function gives us

`V_n = G(source, n)` is the **harmonic measure at `n`** given the
boundary conditions. In probabilistic terms:

> `V_n` = probability that a simple random walker starting at
> `n` reaches the source before hitting any sink.

(Equivalent dual: voltage at `n` when unit current is injected at
the source and extracted at the sinks.)

This is *the* exact cost function for "how strongly is node `n`
connected to the source under the given boundary conditions." Every
other propagating variant we've considered is some approximation
to it.

### How the variants map to the Green's function

| variant | relationship to Green's function |
| --- | --- |
| `hop_distance` | takes only shortest hitting time; throws away path multiplicity (very coarse) |
| Exponential decay flux | multiplicative per-hop attenuation; works when paths don't strongly converge but loses fine structure at confluences |
| Power-law / Gauss-radial flux | log-space approximation with branching-aware scaling; still approximate when paths combine |
| Additive PPR-style flux | random-walk estimator of the Green's function; converges to exact `V_n` in the limit |
| `semantic_similarity` | not a propagation model at all; orthogonal axis |

The Green's function is *the* mathematically rigorous answer. The
cheaper variants are choices about which information to throw away
in exchange for less computation. For cache-warming, the throwaway
choices are usually fine — what we need is the top-K ranking, not
the exact value of `V_n`.

### Why this framing matters

Three practical consequences:

1. **Convergence semantics**. We can talk about "how close are we
   to the Green's function" rather than "how converged is our
   iteration". The two are related but the former gives us a
   target to measure approximation quality against.

2. **Local-exact methods become natural**. Solve `L V = b` on a
   subgraph around the source/sink set, with Dirichlet boundary
   conditions on the subgraph's boundary. This is exactly what
   push-based PPR (Andersen-Chung-Lang) does — it bounds the
   work to a local region while remaining mathematically faithful
   to the global Green's function within that region.

3. **Connection to effective resistance**. `V_source − V_sink`
   divided by injected current is the effective resistance
   between source and sink. This is a well-studied quantity with
   known efficient computation methods (e.g. spectral sparsifiers,
   Johnson-Lindenstrauss sketches). Future cost-function work can
   draw from this literature when more rigorous flux is needed.

### Practical iteration: Gauss-Seidel and friends

The simplest iterative solver for `L V = b` is Gauss-Seidel:

```
V_n ← (1 / deg(n)) · ( b_n + Σ_{w ∼ n} V_w )
```

One sweep through the nodes updates every `V_n` to the average of
its current neighbours plus its boundary contribution. Repeated
sweeps converge to the exact `V`. **One sweep = "additive
aggregation with one iteration"**, which is the recommended
default for cache-warming.

So `aggregation(sum) + iterations(1)` for the additive variant is
not just an approximation — it's one Gauss-Seidel sweep. Increasing
`iterations(N)` moves us along the spectrum toward the exact
Green's function.

## Convergence spectrum

For propagating cost functions (everything except hop_distance
and semantic_similarity), we have a spectrum from cheap to
expensive:

| mode | description | cost | when to use |
| --- | --- | --- | --- |
| **One-pass approximation** | single sweep; aggregate by max or first-fit. No iteration. | O(V+E) | cache warming; cost-function-as-ranking |
| **Bounded iterative** | N fixed iterations of propagation; aggregate per iteration. `iterations(N)` knob. | O((V+E)·N) | when ranking accuracy matters more than worst-case cost |
| **Local exact** | iterate to ε-convergence but only within a local region around endpoints. Push-based PPR (Andersen-Chung-Lang), FORA-style. | bounded by local work | when a single query needs ranking accuracy in a focused region; rest of graph irrelevant |
| **Global exact** | iterate to global ε-convergence (standard PageRank). | O((V+E)·iters_to_ε) | when the cost function output is a research artefact, not a cache-warming aid |

For cache-warming, **one-pass approximation is usually right**.
Local exact becomes interesting when a workload has a small set
of high-stakes endpoints whose ranking accuracy materially
affects which 10% of nodes get warmed — but we don't have such a
workload yet. The design preserves the option without
implementing it.

## Aggregation rules

When a node is reached by multiple paths during propagation,
different cost functions combine the contributions differently:

| rule | semantics | use case |
| --- | --- | --- |
| `min` | shortest / earliest reaches | hop_distance (only sensible value) |
| `max` | best single path's contribution | greedy approximation of flux |
| `sum` | additive accumulation | exact flux, PPR |
| `weighted_average` | depth-weighted blend | hybrid schemes |
| `none` | no propagation | semantic_similarity |

The `aggregation` knob is cost-function-specific. Each registry
entry declares which aggregation rules it accepts; combinations
that don't make sense (e.g. `aggregation(sum)` on hop_distance)
throw at validation.

**Default per cost function**:

- `hop_distance` → `min` (only sensible value)
- `flux` (any form) → `max` (greedy approximation; cheap)
- `additive_flux` → `sum` (intrinsic)
- `semantic_similarity` → `none`

Workload authors who want the expensive-but-exact form of flux
opt in via `aggregation(sum)` plus `iterations(N)` greater than 1.

## Decision guide

A rough heuristic for picking a cost function. Concrete numbers
should come from measurement (see `WAM_PERF_OPTIMIZATION_LOG.md`
Phase L appendices).

| workload property | recommended cost function | aggregation | iterations |
| --- | --- | --- | --- |
| Generic graph, no per-graph data | `hop_distance` | `min` | n/a |
| Wikipedia-shape (asymmetric branching) | `flux` (exp or power) | `max` | 1 |
| Embeddings available | `semantic_similarity` | `none` | n/a |
| Hot region, ranking accuracy critical | (future) local PPR | `sum` | local-exact |
| Research / publication artefact | global PPR | `sum` | ε-converge |

When in doubt: start with `hop_distance`. It's the safest
default — works on any graph, costs almost nothing, gives a
useful (if coarse) ranking. The cost-model crossover formula
(`docs/design/CACHE_COST_MODEL_PHILOSOPHY.md`) tells you whether
you can afford a more expensive cost function for your scale.

## Implementation status

| variant | aggregation supported | status |
| --- | --- | --- |
| `hop_distance` | `min` | real (Phase 1 / P1) |
| `flux` | `max`, `sum` (panic stub for now) | spec-only; implementation deferred to P3+ |
| `semantic_similarity` | `none` (panic stub) | spec-only; requires embeddings |
| additive PPR / local-exact PPR | `sum` | filed as future research direction |

Adding a new variant: extend the registry in
`core/cost_function.pl`, add the param schema, write the Haskell
template constructor, add unit + codegen tests. The plumbing is
pluggable; the math is what the new variant brings.

## Cross-references

- `docs/design/SCAN_STRATEGY_SPECIFICATION.md` — formal
  cost-function slot, option list, integration with warm-build
- `docs/design/SCAN_STRATEGY_PHILOSOPHY.md` — why we do
  cache-warming at all
- `docs/design/CACHE_COST_MODEL_PHILOSOPHY.md` — the seek-vs-scan
  crossover that gates which variants are affordable at runtime
- `docs/design/SCAN_STRATEGY_IMPLEMENTATION_PLAN.md` — P1 (slot
  + hop_distance), P3+ (warm-build + flux)
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` — Phase L / Phase M
  measurements that inform the decision guide
- `src/unifyweaver/core/cost_function.pl` — the registry
- `templates/targets/haskell_wam/cost_function.hs.mustache` —
  Haskell-side concrete constructors
