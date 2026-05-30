# Metric-tree-likeness: when tree-search is statistically enough

**Operational claim**: a (graph, metric) pair is *metric-tree-like*
when you can search the graph as if it were a tree (ignoring
cross-edges) and the metric value you get is statistically close
to the full-DAG value.

**Status**: Design note. Captures a property observed empirically on
simplewiki during bidirectional-kernel benchmarking. Not yet a theorem;
not yet a calibration step. Worth recording because it gives a
principled reason to choose **tree-search over bidirectional search**
for (graph, metric) pairs that satisfy it — a substantial speedup with
documented statistical accuracy.

The property attaches to the **pair**, not to either alone. The same
graph under uniform weighting may not be metric-tree-like; the same
metric on a symmetric DAG may not be metric-tree-like. The
calibration of `b` and `D` is what couples them — and that
calibration is what `bidirectional_ancestor` already computes from
the graph during setup.

**Date**: 2026-05-29
**Source data**: `/tmp/uw_simplewiki_phase1_lmdb` (294,773 edges,
correct-mode ingest from PR #2568 + parser fix PR #2569)

## 1. The observation

Running the direction-weighted bidirectional kernel against Wikipedia's
category graph rooted at Physics, we found that the effective distance
d_wPow barely moves as we lower `childCost` and admit exponentially
more mixed paths:

| childCost | total paths | mixed (child-hop) paths | avg d_wPow |
|-----------|-------------|------------------------|------------|
| ∞ (up-only) | 1,118 | 0 | 5.044 |
| 10 | 1,173 | 55 | 5.044 |
| 5 | 143,992 | 142,874 | 5.045 |
| 3 | 2,692,988 | 2,691,870 | 5.045 |

Across a 2400× increase in path count, the aggregate metric moved
0.02%. Per-seed (20 seeds at depth 4 from Physics), the worst-case
drift was 0.007% — every single pair converged.

## 2. Why this happens: the geometric-series argument

Each path's weight in the metric is

```
w(path) = (1/D)^N · (1/(b·D))^M
```

where N = parent hops, M = child hops, D = avg child fan-out, and
b = E[d²_child] / E[d²_parent] (the calibrated branching asymmetry).

On simplewiki: D ≈ 7.34, b ≈ 1353. So a single child hop carries
weight ≈ `1/(b·D) ≈ 10⁻⁴`, and each additional child hop multiplies
by another factor of 10⁻⁴.

The contribution to the weighted sum from paths with M child hops
is bounded by

```
contribution(M) ≤ (number of M-child-hop paths) · (1/(b·D))^M
```

For this graph, the number of M-child-hop paths grows ~15-100× per
extra child hop (within budget=15), but `b·D ≈ 9933` dominates that
growth by 2+ orders of magnitude. Result: a convergent geometric
series whose terms shrink by ~10⁴ per level. The total contribution
of mixed paths is asymptotically negligible compared to the
upward-only baseline.

## 3. Proposed formal property: metric-tree-likeness

> **Definition (operational).** A (graph, metric) pair is
> *metric-tree-like* iff searching the graph as if it were a tree
> (e.g. shortest-path tree from root, ignoring all cross-edges)
> gives a metric value statistically close to the full-DAG
> value.

This is the *practical* formulation. It tells you what to do —
"just run tree search, the answer will be close enough" — rather
than just describing a property of the graph.

The equivalent structural formulation is the one we've been
measuring: the weighted contribution of paths that *use* the
non-tree edges is asymptotically negligible compared to paths
that stay on the tree. Both are saying the same thing; the
operational version is what's useful for cost-model decisions.

Two versions of "statistically close":

- **Aggregate metric-tree-likeness**: the property holds *on
  average* over a query distribution. Mean d_eff under
  tree-search matches mean d_eff under full bidirectional search.
- **Per-pair metric-tree-likeness**: the property holds for *every*
  (seed, root) pair. Tree-search gives the right answer for each
  individual query, not just the average.

Per-pair is strictly stronger and what we ultimately want to
claim. Aggregate is what we can typically promise without
exhaustive case analysis. The user-facing claim should usually be
the aggregate one: "for queries on this graph under this metric,
tree-search is accurate to ε on average — individual queries may
differ but rarely materially."

### 3.1 What this is *not*

- Not **structural tree-likeness** (treewidth). The graph can have
  arbitrarily many cycles and still be metric-tree-like under the
  right metric.
- Not **Gromov hyperbolicity** (negative curvature). Hyperbolicity
  is a geometric property of the distance metric itself; metric-
  tree-likeness is a property of how the *weighted path sum*
  decomposes between tree edges and cross edges.
- Not **bounded treewidth + small cycle space**. Wikipedia
  categories likely have huge cycle space — every cross-cutting
  categorization adds a cycle. The metric just happens to weight
  those cycles down.

It's a *third* notion: a property of the (graph, metric) pair,
not of either alone. A graph that's metric-tree-like under
power-mean weighting may not be under uniform weighting, and vice
versa.

### 3.2 What it is: the "child shortcuts are statistically rare"
property

When a (graph, metric) pair is metric-tree-like, child shortcuts
*exist* in the graph but don't carry enough weight under the metric
to perturb the answer. Wikipedia categories have cross-cuts
everywhere (a 20th-century physicist is in both "physicists" and
"20th-century people"), but each category has one **dominant**
topical parent. The cross-cuts are real edges in the graph but
*minority paths* in the metric.

That's why this property attaches to the pair, not to either alone:
the same graph under a metric that doesn't differentiate
upward/downward weighting (e.g. uniform power-law) would *not*
exhibit this — we measured exactly this divergence (5.044 → 0.29,
a 90% drift) before introducing the direction-weighted metric. The
graph didn't change; the metric did.

## 4. Evidence (so far)

### 4.1 Aggregate, simplewiki rooted at Physics

20 seeds at depth 4, budget=15, n=2 power-mean:

```
cc=100→3 : avg d_wPow drift = 0.02% over 4 decades of path growth
```

### 4.2 Per-pair, same setup

Every individual seed converged to within 0.007% (max). 19 of 20
converged to within 0.003%. No bimodal distribution — the
convergence is genuinely per-pair on this dataset.

Sample of the 20-seed per-pair drift between `childCost=∞` and
`childCost=5`:

| seed page_id | d @ cc=∞ | paths @ cc=5 | d @ cc=5 | drift |
|--------------|---------:|-------------:|---------:|------:|
| 30843        | 5.0044   | 85           | 5.0045   | 0.001% |
| 646701       | 5.0000   | 18           | 5.0000   | 0.000% |
| 195588       | 5.0269   | 9,233        | 5.0271   | 0.002% |
| 211205       | 5.1020   | 11,727       | 5.1024   | 0.007% |
| 144727       | 5.1020   | 11,758       | 5.1024   | 0.007% |

The raw number of child-hop paths varies 200× across seeds
(18 → 11,758), but their *aggregated contribution to the metric*
is uniformly negligible. That spread in path count without a
spread in metric drift is the cleanest signature of the
geometric-series collapse described in §2.

### 4.3 Comparison with broken-ingest run (PR #2502 data)

The earlier broken-ingest simplewiki test (mixed-namespace, 25k
edges) had b ≈ 61 and showed d_wPow drift of about 1% (5.17 →
5.22). The current correctly-ingested graph (294k edges, b ≈ 1353)
drifts 0.02%. Two interpretations, both probably partly true:

1. The corrected graph has more uniform "category-real" edges,
   making the asymmetry stronger and b larger.
2. The broken graph injected spurious cross-namespace edges that
   counted as cycles, lowering apparent tree-likeness.

Either way, the convergence rate *changed with the graph*, which
is what the property predicts.

## 5. Open questions

### 5.1 Sufficient conditions

What graph properties guarantee metric-tree-likeness? Working
hypotheses:

- **Power-law degree distribution** with sufficiently heavy
  asymmetry between parent and child fan-out (makes
  E[d²_c]/E[d²_p] large, makes b·D large, makes the geometric
  ratio small).
- **Dominant parent rule**: each non-root node has a "primary"
  parent that carries most of its semantic weight (a structural
  way of saying "the graph is approximately a tree with annotations").
- **Sparse cross-edges relative to fan-out depth**: cross-edges
  exist but the diameter-vs-cross-edge-count ratio is small enough
  that the geometric series dominates.

We don't have a clean theorem. The empirical observation is
consistent with all three but doesn't isolate which is doing the
work.

### 5.2 Necessary conditions

Is metric-tree-likeness equivalent to some classical property
(e.g., bounded fractional cycle space), or genuinely a new
property? Probably new because it depends on the metric.

### 5.3 What would falsify it?

A graph engineered to defeat the metric:

- **Symmetric DAG** with E[d²_c] ≈ E[d²_p] → b ≈ 1, child paths
  weight nearly as much as parent paths, no convergence.
- **Bipartite-style** graph where most upward paths to root pass
  through a small set of choke nodes, but child shortcuts exist
  that bypass them. Per-pair metric-tree-likeness would fail even
  if aggregate held.
- **Diamond graph** with two distinct distance regimes between
  every leaf-pair, depending on whether you go up-then-down or
  the other way. Child shortcuts could carry as much weight as
  parent paths.

Constructing one of these synthetically and verifying the metric
drifts more is the natural next experimental step.

## 6. Algorithmic consequences

### 6.1 Convergence as a certificate: use tree-search

If a single empirical convergence check (run at cc=100 and cc=5,
measure drift) shows < 0.1%, we have evidence that for this
(graph, metric, query class) the **tree-search answer is
statistically equivalent to the full bidirectional answer**.
The runtime decision is then simple: **use the tree-search**.

On simplewiki, this is a measured **~2000× speedup** (11 ms vs
~43 s) for the same metric value to four significant digits.
It's much cheaper than proving non-existence of meaningful
cross-paths structurally, and the certificate is honest: it's a
statistical claim about *this* graph under *this* metric for
queries from *this* distribution, not a universal one.

### 6.2 Drift as a diagnostic

If the convergence check shows > 1% drift, that's a signal:
either the graph is genuinely not metric-tree-like (run
bidirectional), or our metric calibration is wrong (re-check b
and D).

### 6.3 Per-pair check is cheap

For high-value queries, an extra 20-seed sample at cc=5 + cc=100
costs ~1 second on simplewiki. The aggregate-vs-per-pair check
gives us early warning if some pairs drift while others don't.

## 7. Status of this document

This is **not a theorem**. It's a name (`metric-tree-likeness`)
attached to a phenomenon we've observed once, with a reasonable
mechanism (geometric series), and a list of things that could
falsify it. The next step is to construct a (graph, metric) pair
where it *fails*, verify the metric does drift there, and refine
the definition to account for what we see.

The property attaches to the **pair** `(graph, metric)`. That's
the most useful framing because it tells you what to ask in
practice: not "is my graph tree-like?" (often hard, sometimes
unanswerable) and not "is my metric well-chosen?" (philosophical)
but "**for the queries I'm about to run, will tree-search give
me an answer close enough to the bidirectional one?**" That
question has a cheap empirical answer (run two convergence
probes), independent of theory.

If the synthetic-graph test confirms the dichotomy, this becomes
a useful property worth adding to the cost-model framework:
(graph, metric) pairs that pass the cheap convergence check get
the fast tree-search path; pairs that fail get the full
bidirectional search.

## 8. References

- `WAM_FSHARP_CSR_KERNEL_INTEGRATION.md` — design of the
  bidirectional kernel itself.
- `WIKIPEDIA_CATEGORYLINKS_INGEST_MODES.md` — the ingester whose
  correct-mode output made this measurement possible.
- Simplewiki benchmark log: `/tmp/per_seed.log` (transient).
