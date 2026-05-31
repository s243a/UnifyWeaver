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

In retrospect this measurement already satisfies §3's homogeneity
precondition without our noticing: rooting at `Category:Physics`
restricts the BFS reach to the topical subgraph (see §4.5), so
the A* pruning effectively excludes admin hubs. We weren't aware
of the inhomogeneity issue at the time, but the experimental setup
sidestepped it. §4.5 derives the consequence — calibration must be
performed on the *traversed* subgraph, not the global one.

## 2. Why this happens: the geometric-series argument

### 2.0 Notation

Several variant symbols denote the branching asymmetry at different
stages of refinement. They are not interchangeable; the distinction
matters for §4–§5.

| Symbol | Definition | Where it's computed | Example value (simplewiki) |
|---|---|---|---|
| `D` | `E[d_child]` — average child fan-out | first moment of the child-degree distribution | 7.34 |
| `b` (early) | `E[d²_child] / E[d²_parent]` — raw second-moment ratio | scan child/parent degree distributions globally | 1353 |
| `b_eff` | `(E[d²_c]/E[d_c]) / (E[d²_p]/E[d_p])` — friendship-paradox-corrected branching asymmetry | same scan, with first-moment correction | 589 (global) / 9.59 (topical, §4.5) |
| `BranchRatio` | `b_eff × routing_correction` — the composed scalar the kernel passes to the metric | adds the empirical routing factor | 226 (global) / 3.68 (topical, currently — see §5.4) |
| `b'` | per-child-hop empirical path-count growth | measured by running the kernel at varying `cc` and reading path-count ratios | ~11 |

`b` (early-formula) was used in §2 below before the first-moment
correction; subsequent sections use `b_eff` exclusively. **The
operative quantity for convergence is `b' < b_eff · D`, not the
raw `b·D`.** §4.4 and §5.5 explain why the early `b·D ≈ 9933`
number, while not wrong, is misleading once we account for
inhomogeneity.

### 2.1 The geometric-series argument

Each path's weight in the metric is

```
w(path) = (1/D)^N · (1/(b·D))^M
```

where N = parent hops, M = child hops, D = avg child fan-out, and
b = E[d²_child] / E[d²_parent] (the early calibrated branching
asymmetry — see §2.0; honest topical calibration uses `b_eff`
instead, with a different numerical value).

On simplewiki under the original global calibration:
D ≈ 7.34, b ≈ 1353. So a single child hop carries weight
≈ `1/(b·D) ≈ 10⁻⁴`, and each additional child hop multiplies by
another factor of 10⁻⁴.

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

> **Forward reference.** The `b·D ≈ 9933` figure here is the
> *global* calibration. §4.4 shows this overestimates the
> traversed branching by ~50× because the global moment scan
> includes admin hubs the search never touches; §4.5 derives the
> honest topical recalibration `b_eff ≈ 9.59` giving `b_eff·D ≈ 70`.
> Both numbers satisfy the convergence inequality `b·D > b'` (with
> empirical `b' ≈ 11` per §4.4), which is why the property holds
> under either calibration — but §5.5 explains why the topical
> number is the principled one for any quantitative use of `(b, D)`.

## 3. Proposed formal property: metric-tree-likeness

> **Definition (operational).** A (graph, metric) pair is
> *metric-tree-like* iff searching the graph as if it were a tree
> (e.g. shortest-path tree from root, ignoring all cross-edges)
> gives a metric value statistically close to the full-DAG
> value.

> **Homogeneity precondition (added on the basis of §4.5).** The
> definition above assumes the graph is *statistically
> homogeneous* — every region the query distribution can reach
> has the same characteristic degree distribution and the same
> calibrated `b_eff`. For inhomogeneous graphs the property must
> be re-stated as: **there exists a decomposition into
> homogeneous subgraphs such that each subgraph is metric-tree-
> like under its own calibration**; the query's reachable set
> determines which subgraph's calibration applies. See §4.5 for
> the topical-core example on Wikipedia.

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

### 3.1 Why the property is principled: `D` does double duty

The convergence inequality `b·D > b'` (path-count growth `b'`
empirically matches `b_eff` once topical calibration is honest)
has substantial slack on real graphs — by 6×+ on simplewiki's
topical core. That slack might look like a tuning artifact ("of
course it works, the inequality has tons of room"), but it isn't.
The slack has a principled source: `D = E[d_child]` is doing
*two distinct jobs* simultaneously, both grounded in a single
measured graph property.

1. `D` is the **per-parent-hop weight** in `w(path) = (1/D)^N (1/(b·D))^M`.
   It enters the metric formula directly.
2. `D` is the **convergence slack**. Once `b` is calibrated to
   match empirical path growth (`b ≈ b'`), the convergence
   ratio reduces to `b'/(b·D) ≈ 1/D`. So `D` is what buys the
   safety margin in the inequality.

Both roles come from the same single measurement
(`E[d_child]`). `D` is not a free parameter that happens to
absorb errors — it's a structurally-motivated quantity that
inherits its double duty from the construction of the metric
itself. That's what makes metric-tree-likeness an honest
property rather than a tuning success.

The detailed analysis — including the consequence that the
routing correction *consumes* part of `D`'s slack without
naming itself as a slack consumer — is in §5.5.1.

### 3.2 What this is *not*

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

### 3.3 What it is: the "child shortcuts are statistically rare"
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

### 4.4 Calibration vs empirical: the inhomogeneity gap

Running the full calibration probe on the correct-ingest simplewiki
gives concrete numbers for each correction term:

| Quantity | Value | Interpretation |
|---|---|---|
| `E[d²_c] / E[d²_p]` (second-moment ratio) | 1353 | Raw distributional asymmetry |
| `× E[d_p] / E[d_c]` (first-moment correction) | 0.436 | Friendship-paradox normalisation |
| `b_eff` (global, traversal-effective) | **589** | Calibrated asymmetry, global graph |
| `× routing_correction` (avg_min_dist / avg_path_hops) | 0.384 | Hub-correlation pruning |
| Final `BranchRatio` (global) | 226 | Composed calibration |

And the *empirical* per-child-hop branching ratio (measured directly
by running the kernel at multiple childCost values and reading the
path-count growth):

| childCost transition | extra child hops allowed | ratio of path counts |
|---|---|---|
| 100 → 10 | 0 → 1 | 1.05 (essentially none) |
| 10 → 7  | 1 → 2 | **16.19** |
| 7  → 5  | 2 → 3 | **7.58**  |

Geometric mean across the meaningful transitions: **~11×** per added
child-hop level. The global calibration says 226 (or 1353 without
any corrections); the search actually experiences ~11.

This is a **20–120× gap between calibration and reality** that no
amount of correction term tuning fully closes — because it isn't a
correction-factor problem. It's a *graph-decomposition* problem.

### 4.5 Homogeneity evidence: the topical core

The gap closes essentially completely when calibration is restricted
to the **topical subgraph** — the descendants of a topical root like
`Category:Articles` (page_id 137597 on simplewiki, analogous to
`Category:Main topic classifications` on enwiki).

| Calibration scope | b_eff | Admin hubs in scope | vs empirical (~11) |
|---|---|---|---|
| Global (whole simplewiki) | 589 | 20/20 | 53× too high |
| `Category:Contents` (includes admin subroot) | 589 | 20/20 | 53× too high |
| `Category:Articles` (topical core) | **9.59** | 3/20 (semi-topical only) | **within 13%** |
| `Category:Physics` (topical subtree) | 9.51 | 3/20 (same three) | within 14% |

Two things this shows:

1. **The topical core is statistically homogeneous.** `Category:Articles`
   and `Category:Physics` give essentially the same b_eff (9.59 vs
   9.51, agreement to <1%). They reach different sets of nodes but
   measure the *same statistical regime*. If the topical core were
   itself inhomogeneous, these two scopes would disagree materially.
   They don't.

2. **The 20–120× calibration error was driven entirely by admin
   hubs.** Once we BFS from a topical root, the maintenance
   categories (`CatAutoTOC generates no TOC`, `Hidden categories`,
   `Navseasoncats…`, etc.) are not reached and don't enter the
   moment calculations. Three semi-topical categories remain
   (`Births by year`, `Deaths by year`, `Songs by artist`), and
   they're genuinely topical meta-indexes — their fan-out reflects
   real categorical structure, not maintenance noise.

The implication: **Wikipedia's category graph is inhomogeneous —
two qualitatively different statistical regimes mixed together**.
The topical regime (γ ≈ 2.4–2.5, characteristic fan-outs 1–100,
tree-like organisation) and the administrative regime (list-shaped
categories with fan-outs in the thousands, no internal hierarchy).
Each regime can be metric-tree-like under its own calibration;
together they are not metric-tree-like under any single calibration.

The routing correction (the 0.384 factor in §4.4) was effectively
*band-aiding* this inhomogeneity — compensating for hub categories
that the search wasn't actually traversing. **With proper subgraph
scoping the routing correction becomes unnecessary**: the topical
b_eff = 9.59 already matches empirical without it.

## 5. Open questions

### 5.1 Sufficient conditions

The operative quantity is the convergence inequality `b·D > b'`
from §5.5.1 (path-count growth bounded by calibrated branching).
The conditions below are *sufficient* graph-structural conditions
under which that inequality is expected to hold; they are not
themselves the property. Working hypotheses:

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
work — and §5.5.1's "as long as `b·D > b'`" framing suggests any
graph property that yields the inequality (whatever its
structural origin) would be enough.

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

### 5.4 Routing correction under topical scoping — redundant?

The kernel currently composes `BranchRatio = b_eff × routing_correction`
where `routing_correction = avg_min_dist / avg_path_hops` (≈ 0.384
on simplewiki rooted at Physics). The routing correction was
introduced as a heuristic compensation when calibration was global
and `b_eff` was wildly inflated by admin hubs (589 instead of
~10).

With topical-root scoping (§4.5) `b_eff` already matches the
empirical per-child-hop path-count growth directly:

| Calibration | b_eff | × routing | empirical growth | match |
|---|---|---|---|---|
| Global | 589 | 226 | ~11 | 20× too high |
| Topical (Articles) | 9.59 | 3.68 | ~11 | b_eff alone matches; ×routing overcorrects |

Three observations make this a real open question rather than a
settled answer:

1. **The routing-correction probe is already topical.** The
   kernel's probe uses `gcr` = min-dist from root via children,
   which returns infinity for nodes unreachable from root. Admin
   hubs that aren't descendants of the root are already pruned by
   A*. So the existing 0.384 was measured on the topical-Physics
   subgraph — it isn't a global artifact, it's a real
   path-vs-shortest-path divergence.

2. **`b_eff` is structural, routing correction is process.**
   `b_eff = (E[d²_c]/E[d_c]) / (E[d²_p]/E[d_p])` is a property of
   the degree distribution alone. Routing correction is a property
   of how the search wanders. In principle they measure different
   things and should compose. In practice they appear to be
   measuring overlapping effects once admin pollution is removed.

3. **Convergence holds both ways at this budget.** With topical
   `b·D = 9.59 × 7.34 ≈ 70` and empirical path-growth ≈ 11, the
   ratio `path_growth / (b·D) ≈ 0.157` ≪ 1 — comfortable
   convergence. With `× routing` factored in we get
   `b·D ≈ 27` and ratio ≈ 0.407 — also converges but tighter. The
   `< 0.1%` aggregate-drift criterion hasn't been re-measured
   under topical calibration; both forms may still pass, or only
   one may.

**Tentative recommendation:** under topical-root calibration, drop
the routing correction and use `BranchRatio = b_eff(topical)`
directly. The routing correction appears to have been a
band-aid for inhomogeneity that topical scoping addresses at the
source. **This needs verification** by re-running the drift probe
on simplewiki rooted at `Category:Articles` with and without the
routing-correction factor, and checking which composition gives
the aggregate-drift below the 0.1% threshold (task #14).

### 5.5 Convergence robustness — a feature for certification, a trap for cost modelling

The convergence condition `b·D > path_growth` is a strict
inequality with substantial slack on the simplewiki graph. With
`path_growth ≈ 11`, any `b·D > 11` produces a convergent series,
and the metric becomes increasingly insensitive to the exact
value of `b·D` as it grows past the boundary.

Working through the actual values:

| Calibration | b·D | per-child weight | path_growth / (b·D) |
|---|---|---|---|
| Boundary | 11 | 9.1% | 1.00 (no convergence) |
| Topical, with routing | 27 | 3.7% | 0.407 |
| Topical, no routing | 70 | 1.4% | 0.157 |
| Global, with routing | 226 | 0.44% | 0.049 |

So per-child penalties anywhere in the band roughly **0.4% – 8%**
produce a convergent metric on this graph. That's nearly two
orders of magnitude of valid penalty values. This explains why
three different calibration approaches with very different `b`
numbers all yielded aggregate drift `< 0.1%` in earlier
probes — the metric isn't sensitive in the interior of the
convergent regime.

This robustness cuts two ways depending on what you're using
`(b, D)` for:

1. **Tree-likeness certification (§6.1).** Only needs
   `b·D > path_growth`. Wide tolerance. Even an order-of-
   magnitude calibration error is harmless as long as you stay
   on the convergent side of the inequality. This is why the
   broken-ingest comparison (§4.3) still showed convergence
   despite `b = 61` — `61·D` was still > 11.

2. **Cost modelling and scheduling.** Anywhere `(b, D)` shows
   up as a *quantitative* cost factor — predicted runtime,
   memory estimates, scheduler weights — needs the *actual*
   per-hop behaviour. Here a calibration that overshoots by
   10× gives 10×-wrong predictions even though the certificate
   still passes.

For the certificate, both routing-corrected and uncorrected
forms work. For honest cost modelling, the principled topical
`b_eff` (no routing) is the right choice because it matches
reality. The routing correction is interesting as a
*separate* measurement — "how far do paths wander vs the
shortest path" — but it shouldn't be folded into `b` if `b` is
being used as a per-hop branching prediction.

The same robustness eats uncertainty in `path_growth` itself:
on simplewiki the measured per-child-hop growth ratios were
7.58 and 16.19 at successive budget transitions (with the
first transition essentially flat because budget was too
tight). The geometric mean ~11 is itself only an order-of-
magnitude estimate. The convergence holds because `b·D ≫
path_growth` across most plausible measurements.

The corollary is uncomfortable: a metric-tree-likeness
certificate that passes by a wide margin tells you *almost
nothing* about whether your calibration is correct. The
certificate is necessary for using the cheap tree-search path,
but it's not sufficient for trusting `(b, D)` for any other
purpose. **Calibration honesty needs a separate check** — the
b_eff vs empirical-path-growth comparison in §4.4 / §4.5 is
that check.

#### 5.5.1 Where the slack actually lives

The robustness band has a clean structural interpretation once
`b` is calibrated honestly:

- **`b` is constrained**, not slack. Honest topical calibration
  forces `b ≈ path_growth` (the §4.5 result: 9.59 vs 11). Any
  free parameter in `b` is a calibration error.
- **`D` is the slack.** With `b ≈ path_growth`, the convergence
  ratio reduces to `path_growth / (b·D) ≈ 1/D`. So `D` directly
  buys the safety margin in the inequality.
- **The routing correction borrows from `D`'s slack.** Multiplying
  `b` by `routing < 1` effectively reduces the convergence
  margin by `1/routing`. On simplewiki that drops the effective
  slack factor from `D ≈ 7.3` to `D · routing ≈ 3`. Still
  positive, still convergent — but the slack has been spent on
  a heuristic that doesn't name itself as a slack consumer.

The reason this distinction matters: **`D` is independently
principled** while the routing correction isn't. `D = E[d_child]`
is a graph-structural property with a clear interpretation
(average child fanout, spectral-dimension proxy, branching
factor in the tree approximation). Critically, `D` is doing two
jobs at once:

1. It is the per-parent-hop weight in `w(path) = (1/D)^N (1/(b·D))^M`.
2. It is the convergence slack once `b` matches reality.

Both jobs come from the same measured graph property. That's
what makes it principled — `D` isn't a free parameter that
happens to absorb errors; it's a single measurement doing
double duty by construction of the metric.

(§5.6 reaches a related conclusion from a different angle: the
weight formula itself is a path-count normaliser, so the
*structure* of `(1/D)^N · (1/(b·D))^M` already encodes the
counting-vs-information balance the property needs. §5.5.1 is
the calibration-side of that story; §5.6 is the formula-side.)

The routing correction doesn't have that double role. It's a
single-purpose empirical adjustment for path-wander. It happens
to leave convergence intact because `D`'s slack absorbs it, not
because the algebra demands it. Folding it into `b` for the
metric obscures both the calibration of `b` (which is now off
by `1/routing`) and the slack in `D` (which is now silently
reduced). Keeping it as a separate diagnostic — "this graph
wanders 2.6× longer than min-dist" — preserves the
interpretability of `b`, `D`, and the convergence ratio
individually.

### 5.6 Weights as path-count normalisers — a cleaner restatement

This section is the formula-side companion to §5.5.1's
calibration-side analysis. §5.5.1 explains why `D`'s slack is
principled; §5.6 explains why the *form* of the weight formula
encodes path-count normalisation by construction.

The metric weight `w(path) = (1/D)^N · (1/(b·D))^M` is often
read as "give parent paths weight 1/D and child paths weight
1/(b·D)" — as if expressing a *belief* that child paths matter
less. A more accurate reading: **the weights cancel
path-count growth so the metric isn't dominated by combinatorial
abundance**.

The number of paths reaching root with `N` parent hops and `M`
child hops grows like:

```
#paths(N, M)  ≈  D^N · (b·D)^M
```

(parent fan-in times per-child-hop path growth, per §2). The
weight `w = (1/D)^N · (1/(b·D))^M` is then precisely

```
w  ∝  1 / #paths(N, M)
```

That is: **each path-shape contributes equally** before the
`(h+1)^(-n)` length factor sorts paths by how short they are.
The weight isn't downweighting children because children are
suspicious; it's downweighting them in proportion to how many
there are, so multiplicity doesn't drown out the
information-bearing factor `(h+1)^(-n)`.

This reframes the property cleanly. The graph can be tree-shaped,
small-world, or ultra-small-world at the *geometric* level. What
matters for the metric is whether the calibrated `b'` (the
empirical per-child-hop path growth, which equals `b · D` in our
notation when properly calibrated) makes the M ≥ 1 contributions
to `d_wPow` small enough to ignore.

#### 5.6.1 Operational restatement of the property

A graph is **ε-metric-tree-like** under the weighted-power-mean
metric iff its empirically calibrated `b'` produces weights
`(1/b')^M` small enough that the sum over M ≥ 1 paths
contributes less than `ε` to `d_wPow`:

```
sum_{M ≥ 1}  #paths(M) · (1/b')^M · (length_factor)^M  <  ε · d_wPow(M=0)
```

This formulation has three nice properties:

1. **Geometric structure is not required.** A small-world graph
   with the right `b'` can still be metric-tree-like. The
   graph's geometric regime determines `b'`; only `b'` enters
   the property.
2. **Calibration honesty is the whole game.** If `b' << path_growth`,
   the property silently fails because the weights don't actually
   cancel path-count growth. If `b' >> path_growth`, the property
   trivially holds but the metric is wasting precision on
   nonexistent paths. Both regimes were observed (§4.4–§4.5).
3. **The certificate (§6.1) tests this directly.** Aggregate
   drift `< ε` at increased child budget *is* the operational
   check that the sum over M ≥ 1 contributions is below ε. It
   doesn't matter whether the graph is geometrically tree-shaped;
   what matters is whether the calibration is honest enough to
   make those contributions actually vanish.

#### 5.6.2 Decoupling geometry from metric

The §4.5 result decomposes cleanly under this view, **provided we
respect the homogeneity precondition from §3**. The Cohen-Havlin
ultra-small-world theorem requires statistical homogeneity (one
degree distribution throughout, uniform mixing). The global
Wikipedia graph violates both — so even though the *combined*
degree distribution fits `γ ≈ 2.41 < 3`, we cannot invoke the
ultra-small-world conclusion globally.

What the global graph actually is: a **mixture of regimes**, not a
single regime with a heavy tail.

- **Topical regime alone:** modest degree distribution,
  assortative-within-type connectivity. Routing within topical is
  probably tree-like or normal small-world — to be measured
  (task #15).
- **Admin regime alone:** heavily skewed distribution (list-shaped
  categories with thousands of children). Within-admin routing
  is probably ultra-small through admin-to-admin hub edges.
- **Cross-regime edges are sparse.** A random topical node has
  low probability of linking into an admin hub, so admin hubs do
  *not* compress topical-to-topical distance — even though they
  would for a truly homogeneous scale-free graph at this `γ`.

The global degree fit `γ = 2.41` (KS = 0.048) is statistically
valid — the *combined* distribution really is consistent with a
power law — but the fit is an artifact of the mixture: two
distributions superimpose to *look* uniform without the routing
geometry inheriting the homogeneous-graph behaviour. **The
power-law fit is necessary but not sufficient for ultra-small-world
geometry; homogeneity is the missing ingredient**, and it's
exactly what's absent.

This also explains why global `b' ≈ 589` is so misleading: it's
averaging path-growth rates from the mixture, where admin-
flavoured paths inflate the second moment but the actual search
rarely traverses them. The topical recalibration `b' ≈ 9.59` is
measuring path growth in the regime the search *actually walks*.

So the corrected decomposition:

- **Global graph:** heavy-tailed *but inhomogeneous*. No single
  geometric label applies; `b'` measured globally is dominated by
  paths that aren't traversed in practice.
- **Topical core (Articles):** statistically homogeneous (Articles
  ≈ Physics b_eff, §4.5). Geometric regime to be measured (tree-
  like or small-world, task #15). `b' ≈ 9.59` reflects the
  actually-traversed regime, calibration is honest, property
  holds.

The metric-tree-likeness property doesn't require the topical core
to be *geometrically* tree-shaped — only that its calibration
honestly captures whatever regime it is in. The task #15
measurement settles the geometric question; the §6.1 certificate
settles the metric question; they're independent answers to
independent questions.

This is a sharper statement of what §3 was reaching for: the
property is fundamentally about calibration-induced weight
balance, evaluated on a *statistically homogeneous* subgraph the
query distribution actually traverses, not about global graph
shape.

## 6. Algorithmic consequences

### 6.1 Convergence as a certificate: use tree-search

If a single empirical convergence check (run at cc=100 and cc=5,
measure drift) shows < 0.1%, we have evidence that for this
(graph, metric, query class) the **tree-search answer is
statistically equivalent to the full bidirectional answer**.
The runtime decision is then simple: **use the tree-search**.

**The certificate must be obtained at the production budget B.**
The convergence rate depends on B (higher budget admits more
M-child-hop paths before saturation; see §2.1's geometric-series
argument), so a certificate obtained at B=15 doesn't license
tree-search at B=50 without re-checking. A safe protocol: pick
the largest budget you expect to use in production, certify at
that budget, then deploy tree-search for any query at budget ≤
B_certified.

On simplewiki, this is a measured **~2000× speedup** (11 ms vs
~43 s) for the same metric value to four significant digits.
It's much cheaper than proving non-existence of meaningful
cross-paths structurally, and the certificate is honest: it's a
statistical claim about *this* graph under *this* metric for
queries from *this* distribution *at this budget*, not a universal
one.

### 6.2 Drift as a diagnostic

If the convergence check shows > 1% drift, that's a signal:
either the graph is genuinely not metric-tree-like (run
bidirectional), or our metric calibration is wrong (re-check b
and D).

### 6.3 Per-pair check is cheap

For high-value queries, an extra 20-seed sample at cc=5 + cc=100
costs ~1 second on simplewiki. The aggregate-vs-per-pair check
gives us early warning if some pairs drift while others don't.

## 7. Data-prep consequences of the inhomogeneity finding

§6 follows directly from the property: if a (graph, metric) pair
is metric-tree-like, you can use tree-search. The items below are
different — they are *production recommendations* that follow from
§4.5's inhomogeneity finding, not from the property itself. They
depend on the §4.5 evidence holding up (which it does on
simplewiki) and on the §5.4 open question about the routing
correction (which is *not yet settled*; see task #14).

> **Status:** these recommendations are **tentative** until §5.4
> is settled by task #14. The §4.5 evidence is empirical and
> robust; the §5.4 claim that routing correction can be dropped
> is supported by argument but not yet by a re-run of the drift
> probe under topical calibration. Treat §7.1's "no
> routing-correction band-aid" as conditional on task #14
> confirming the drift criterion still holds without that factor.

### 7.1 Data preparation: build the LMDB from a topical root

The inhomogeneity result (§4.5) has a direct production
consequence: **the LMDB should be built from a topical root, not
from the full category dump**.

Recipe:

1. Identify the topical root for the target wiki. On
   simplewiki: `Category:Articles` (page_id 137597). On
   enwiki: `Category:Main_topic_classifications` (page_id
   7345184). Both verified against the live Wikipedia API on
   2026-05-31.
2. During ingest, BFS from the topical root via the
   `cl_type = 'subcat'` edges; mark every reached page_id as
   in-scope.
3. Emit only in-scope edges into `category_parent` /
   `category_child`. Out-of-scope ancestors (administrative
   hubs, maintenance trees, hidden-category bins) are
   silently filtered.

The result is an LMDB that contains the "topical core" only.
Calibration on this LMDB will give `b_eff` matching the empirical
per-hop branching out of the box (no topical-subgraph scoping
needed at runtime, no manual hub blacklist). The size reduction
on simplewiki is ~75% (~80k in-scope nodes vs ~92k full), which
is also cache-friendly. **Whether the routing-correction factor
can be dropped from `BranchRatio` is task #14**; if it cannot,
the same LMDB still works, the routing correction just stays
applied at calibration time.

Implementation: add `--filter-root <page_id>` to the
`mysql_stream_lmdb` ingester (see the categorylinks 3-mode
ingester for the natural extension point). The flag is optional;
omitted, the full graph is ingested as today.

### 7.2 Calibration recipe (summary)

Putting §4 and §7.1 together:

1. **Build** the LMDB with `--filter-root <topical_root_id>`
   (§7.1).
2. **Calibrate** at setup time: scan `category_child` and
   `category_parent` for degree moments, compute
   `b_eff = (E[d²_c]/E[d_c]) / (E[d²_p]/E[d_p])`. Whether to
   drop the routing correction factor is contingent on task #14
   (see §5.4).
3. **Certificate**: run the 20-pair drift probe at the production
   budget (§6.1). If `ε_agg < 0.1%`, deploy tree-search.
4. **Per-query**: tree search using `(b_eff, D, parentCost)`.

The whole pipeline is one-time setup cost + per-query
near-tree-traversal cost. The inhomogeneity decomposition lives
entirely at step 1 (ingest), the calibration honesty lives at
step 2, the certificate lives at step 3, and runtime stays
fast.

## 8. Status of this document

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

## 9. References

- `WAM_FSHARP_CSR_KERNEL_INTEGRATION.md` — design of the
  bidirectional kernel itself.
- `WIKIPEDIA_CATEGORYLINKS_INGEST_MODES.md` — the ingester whose
  correct-mode output made this measurement possible.
