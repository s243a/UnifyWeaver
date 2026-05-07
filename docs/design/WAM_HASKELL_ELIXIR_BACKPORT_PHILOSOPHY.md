# WAM Haskell Backport Philosophy From Elixir Large-Scale Benchmarks

## Thesis

The Elixir large-scale benchmark result does not show that BEAM is generally
faster than compiled Haskell. It shows that graph workloads are dominated by
planning shape: which seeds enter the recursive kernel, which edges remain in
the candidate graph, and whether fact access is paid one small lookup at a time
or amortized through preprocessing and cache population.

The goal is to port the useful planning lessons back to Haskell while
preserving Haskell's existing strengths: compiled code, native FFI kernels,
bounded LMDB caches, and explicit control over cache modes.

## Lessons To Port

### Gate Before Execution

The biggest Elixir correction was not an LMDB micro-optimization. It was
skipping seeds that cannot reach the bound root before calling the recursive
kernel. Haskell already computes a root demand set for the category workload,
but benchmark behavior suggests the seed loop still pays substantial per-seed
cost. The Haskell fast path should make the demand decision before constructing
WAM state, entering `executeForeign`, or collecting solutions.

For a bound-root query over `category_parent(Child, Parent)`, the structural
demand set is all nodes that can reach the selected root. This is a
semantics-preserving filter: if a seed is outside the demand set, it cannot
produce a path to the root under the original graph relation.

### Preserve Paths When The Program Observes Paths

Folded kernels are only correct when the caller observes an aggregate, not the
individual paths. For example, an aggregate-only effective-distance query may
fold hop weights directly. A query that asks for paths, hop rows, witnesses, or
duplicate route counts needs a path-preserving predicate or kernel mode.

The compiler should maintain two surfaces:

- A path-preserving surface that enumerates the same solutions as the Prolog
  predicate.
- A folded aggregate surface that is selected only when the enclosing query
  consumes the recursive predicate through an aggregate that does not expose
  individual paths.

The folded version must still implement the same duplicate-path semantics,
visited behavior, depth bound, and base/recursive case order that the
path-preserving predicate would have produced.

### Cache Population Is A Planning Problem

Haskell's LMDB cache tiers are valuable, but a cache is only as good as the
access plan that fills it. Elixir's point cache barely moved the 50k result;
the useful shift came from demand filtering plus a shape where the hot graph is
tiny. The first Haskell backport should therefore focus on structural demand
gating and cache instrumentation.

For larger workloads, all targets should distinguish:

- Structural cache warming: safe, derived from a bound root or other complete
  structural constraint.
- Semantic cache warming: heuristic or domain-guided, derived from embeddings
  or similarity to endpoints, and therefore explicit in the source or query
  metadata.

Semantic top-K is a second-stage, cross-target idea. It can be very useful for
prepopulating LMDB cache pages with one large read, but it is not automatically
semantics-preserving. It should be treated as a planning hint unless the
program declares a completeness contract, and it should not be framed as a
Haskell-only optimization.

## Scale Boundary

On a PC, structural demand filtering and bounded LMDB caches are appropriate
default tools for Haskell and other local targets. Semantic top-K filtering over
precomputed embeddings is plausible when the embedding index fits locally and
the top-K is moderate, but it belongs to the next cross-target planning layer.

Cluster-scale planning enters when the candidate set is too large to rank or
merge locally. The natural distributed shape is map-side scoring over shards,
local top-K selection, then a global merge of sorted top-K streams. That is
Hadoop-style in spirit, but it should remain an optional backend strategy, not a
requirement for desktop-scale WAM targets.
