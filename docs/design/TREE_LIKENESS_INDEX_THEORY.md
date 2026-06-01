# Tree-likeness index: theoretical foundations

**Companion to** [`TREE_LIKENESS_INDEX.md`](TREE_LIKENESS_INDEX.md), which
documents the empirical observation, the calibration recipe, and the
algorithmic consequences. This document attempts to formalise the
theoretical content: what we can prove, what we conjecture, what is
borrowed from existing literature.

**Status**: Work in progress. Most central claims are conjectures
rather than theorems. Established results from graph theory are
cited; novel claims are clearly labelled. The design note's §4
documents the empirical evidence supporting the conjectures here.

**Scope**: We formalise the *tree-likeness index* `TLI(G, μ, Q, B)`
as a statistical quantity computed from a directionally-weighted
metric on a directed graph. The central conjecture (§3.1) gives an
operational sufficient condition for low TLI in terms of two
calibrated graph quantities `b_eff` and `D`. The non-trivial
structural feature that makes the index interesting is *directional
asymmetry* (parent-direction branching vs child-direction shortcut
frequency), not acyclicity or any global structural label.

## 0. Setup and notation

### 0.1 The graph

Let $G = (V, E)$ be a directed graph with a designated root
$r \in V$. We do **not** require $G$ to be strictly acyclic: rare
directed cycles are tolerated by visited-set pruning in traversal,
and contribute negligibly to path counts when present below ~1%
of vertices (which is the case for Wikipedia category graphs).

What we *do* require is a parent/child decomposition: for each
non-root $v$, define
$$
\mathrm{parents}(v) \subseteq V, \quad
\mathrm{children}(v) \subseteq V
$$
as the sets of nodes connected to $v$ by an in-edge or out-edge
respectively, in the categorisation interpretation. The
distinction between "parent direction" and "child direction" is
load-bearing — it is what gives the metric below its directional
character.

The relevant structural feature is **multi-parent connectivity**:
typically $|\mathrm{parents}(v)| > 1$ for most non-root nodes,
creating abundant undirected cycles (alternate ancestor paths
through different parent chains). These undirected cycles are
the *subject* of the index, not a perturbation to it.

### 0.2 Paths

A *path* from $v$ to $r$ in $G$ is a sequence
$v = v_0 \to v_1 \to \dots \to v_k = r$ where each step is either a
**parent hop** ($v_{i+1} \in \mathrm{parents}(v_i)$) or a
**child hop** ($v_{i+1} \in \mathrm{children}(v_i)$). We
decompose:

- $N(p)$ = number of parent hops in path $p$
- $M(p)$ = number of child hops in path $p$
- $h(p) = N(p) + M(p)$ = total hop count

We assume the traversal uses a visited-set so paths are simple
(no node repeated).

### 0.3 The directionally-weighted metric

Following the design note's §2.0 notation, let:

- $D := \mathbb{E}[d_c]$, the average child fan-out
- $b := \mathbb{E}[d_c^2] / \mathbb{E}[d_p^2]$, the raw
  second-moment ratio (the "early-formula" calibration)
- $b_{\text{eff}} := \frac{\mathbb{E}[d_c^2]/\mathbb{E}[d_c]}{\mathbb{E}[d_p^2]/\mathbb{E}[d_p]}$, the
  friendship-paradox-corrected branching asymmetry
- $b'$ := the empirical per-child-hop path-count growth (defined
  in §0.4 below)

The path weight is
$$
w(p) = D^{-N(p)} \cdot (b \cdot D)^{-M(p)}
$$
and the **directionally-weighted power-mean metric** with exponent
$n$ is
$$
d_{\text{wPow}}(v; B, cc) = \left(
\frac{\sum_{p \in \mathcal{P}(v; B, cc)} w(p) \cdot (h(p)+1)^{-n}}
     {\sum_{p \in \mathcal{P}(v; B, cc)} w(p)}
\right)^{-1/n}
$$
where $\mathcal{P}(v; B, cc)$ is the set of paths from $v$ to $r$
with cost at most $B$ when each parent hop costs $1$ and each
child hop costs $cc$. We typically use $n = 2$ in experiments.

### 0.4 Path-count growth

For a node $v$ reachable to $r$ within budget $B$, define
$$
\#\text{paths}(v; N, M; B) :=
\bigl|\{p : v \to r,\ p \text{ has } N \text{ parent hops}
              \text{ and } M \text{ child hops, cost} \le B\}\bigr|
$$
The **empirical per-child-hop path-count growth** is
$$
b'(G, Q, B) := \lim_{M \to \infty}
\frac{\mathbb{E}_{v \sim Q}\bigl[\#\text{paths}(v; \cdot, M+1; B)\bigr]}
     {\mathbb{E}_{v \sim Q}\bigl[\#\text{paths}(v; \cdot, M; B)\bigr]}
$$
where the dot in $\#\text{paths}(v; \cdot, M; B)$ denotes summation
over $N$, and the limit is taken over $M$ values for which both
quantities are non-zero. In practice we estimate $b'$ from a
finite range of $M$ by running the bidirectional kernel at
varying $cc$ and reading the ratio of total path counts (design
note §4.4).

### 0.5 The tree-likeness index

The **tree-likeness index** of the tuple $(G, \mu, Q, B)$ where
$\mu$ is the directionally-weighted metric is
$$
\mathrm{TLI}(G, \mu, Q, B) :=
\frac{\bigl|\mathbb{E}_{v \sim Q}\bigl[d_{\text{wPow}}(v; B, \infty)\bigr] -
            \mathbb{E}_{v \sim Q}\bigl[d_{\text{wPow}}(v; B, 0^+)\bigr]\bigr|}
     {\mathbb{E}_{v \sim Q}\bigl[d_{\text{wPow}}(v; B, 0^+)\bigr]}
$$
i.e. the relative drift of the mean metric value as the child-step
cost ranges from infinity (tree-search, $M = 0$ only) to
$0^+$ (full DAG search, all $M$ admitted within budget).

In practice we estimate this with a finite contrast between two
$cc$ values, typically $cc = 100$ (effectively tree-search at
$B = 15$) and $cc = 5$ (admits up to 3 child hops); see design
note §4.1.

## 1. Background: established results we build on

### 1.1 Random-graph distances (Erdős–Rényi / Chung–Lu)

**Theorem (Chung & Lu, 2002).** For a random graph $G(n, p)$ with
expected average degree $\langle k \rangle = pn$ satisfying
$np \to \infty$, the average shortest-path length between random
node pairs satisfies
$$
L \sim \frac{\log n}{\log \langle k \rangle}
$$
as $n \to \infty$.

Source: F. Chung and L. Lu, "The average distances in random
graphs with given expected degrees", *PNAS* 99(25): 15879–15882
(2002).

This is the baseline for "what branching alone achieves" in a
homogeneous graph — relevant to §5.3 of the design note's
geometric-regime comparison.

### 1.2 The small-world property (Watts–Strogatz)

**Definition (Watts & Strogatz, 1998).** A graph is *small-world*
if it simultaneously satisfies:

1. $L = \Theta(\log n)$ (logarithmic average shortest-path length)
2. $C \gg C_{\text{random}}$ (clustering coefficient significantly
   above that of a random graph of the same size and density)

Source: D. J. Watts and S. H. Strogatz, "Collective dynamics of
'small-world' networks", *Nature* 393: 440–442 (1998).

### 1.3 Ultra-small-world for scale-free graphs (Cohen–Havlin)

**Theorem (Cohen & Havlin, 2003).** For a scale-free network with
degree distribution $P(k) \propto k^{-\gamma}$:

- If $\gamma > 3$: $L \sim \log n / \log \langle k \rangle$
- If $\gamma = 3$: $L \sim \log n / \log \log n$
- If $2 < \gamma < 3$: $L \sim \log \log n$ (ultra-small-world)
- If $\gamma \le 2$: $L = O(1)$ (degenerate; degree diverges
  with $n$)

The result requires *statistical homogeneity* of the graph: the
degree distribution holds uniformly across $V$, with no
sub-populations having qualitatively different scaling.

Source: R. Cohen and S. Havlin, "Scale-free networks are
ultrasmall", *Physical Review Letters* 90(5): 058701 (2003).

**Relevance to TLI.** The Cohen–Havlin theorem governs the
*geometric* regime of the graph (how distances scale with $n$).
It does not directly bound TLI, but it constrains the family of
graphs on which TLI can be low under any single global
calibration (see §5.6.2 of the design note).

### 1.4 Power-law fitting methodology (Clauset–Shalizi–Newman)

**Method (Clauset, Shalizi & Newman, 2009).** Given empirical data
$\{d_1, \dots, d_n\}$ believed to follow a power-law tail
$P(k) \propto k^{-\gamma}$ for $k \ge d_{\min}$:

1. For each candidate $d_{\min}$, compute the continuous MLE
$$
\hat\gamma(d_{\min}) = 1 + n_{\ge d_{\min}} \biggm/
\sum_{d_i \ge d_{\min}} \log(d_i / d_{\min})
$$
2. Compute the Kolmogorov-Smirnov distance between the empirical
   tail CDF and the fitted power-law tail CDF
3. Select the $d_{\min}$ that minimises KS distance

Source: A. Clauset, C. R. Shalizi, and M. E. J. Newman,
"Power-law distributions in empirical data", *SIAM Review*
51(4): 661–703 (2009).

**Application here.** Design note §4.4 reports
$\gamma = 2.41 \pm 0.05$ and $d_{\min} = 29$ with KS distance
$0.048$ on the simplewiki child-degree distribution (global). The
fit is statistically valid, but §5.6.2 explains why the
homogeneity precondition is what blocks invoking Cohen-Havlin
ultra-small-world consequences from this fit.

### 1.5 Friendship paradox / size-biased degree (Feld)

**Lemma (Feld, 1991).** For a graph with finite degree
distribution $P(k)$, the expected degree of a node reached by
following a random edge is
$$
\frac{\mathbb{E}[k^2]}{\mathbb{E}[k]} \ge \mathbb{E}[k]
$$
with equality iff degrees are constant.

Source: S. L. Feld, "Why your friends have more friends than you
do", *American Journal of Sociology* 96(6): 1464–1477 (1991).

**Relevance to TLI.** The first-moment correction in our
definition of $b_{\text{eff}}$ versus the raw second-moment ratio
$b$ is precisely this size-biased correction applied separately
to the child and parent edge populations. Design note §4.4
labels this the "degree-bias correction"; this lemma is the
underlying reason it has the form it does.

### 1.6 Tree pair distance

**Lemma (folklore).** In a balanced $D$-ary tree of depth $L$ with
$n = (D^{L+1} - 1)/(D - 1)$ nodes:

- Depth-to-root: $L = \log_D(n(D - 1) + 1) \approx \log_D n$
- Average pair distance: for any two distinct nodes $u, v$,
$$
\mathbb{E}_{u,v}[d(u,v)] \le 2L \approx 2 \log_D n
$$
with equality approached as $D \to \infty$ (large branching makes
LCAs shallow).

**Proof sketch.** The distance between $u$ and $v$ is
$\mathrm{depth}(u) + \mathrm{depth}(v) - 2 \cdot \mathrm{depth}(\mathrm{LCA}(u,v))$.
For uniform-random $u, v$ in the leaves of a balanced tree, the
probability that the LCA is at depth $\ge \ell$ is approximately
$D^{-\ell}$, so the expected LCA depth is $O(1/(D-1))$, vanishing
for large $D$. The pair distance is therefore close to twice the
depth $L$. $\square$

**Relevance to TLI.** A tree-shaped graph is the structural
limit where TLI is trivially $0$ — there are no cross-edge paths
to admit, so the metric is path-invariant in $cc$. The interest
of TLI is what happens *between* this limit and a graph with no
asymmetry at all (§3 conjecture below).

## 2. What we can prove

We state three results that follow from the definitions in §0 and
mild homogeneity assumptions. They are *not* the central
conjecture (§3.1); they are building blocks supporting it.

### 2.1 Path-count growth lemma (conditional on homogeneity)

**Lemma 2.1.** Suppose $G$ is *statistically homogeneous* in the
following sense: there exist constants $D = \mathbb{E}[d_c]$ and
$b' \ge 1$ such that, uniformly over all
$v \in V$ reachable to $r$,
$$
\#\text{paths}(v; N, M; B) \approx D^N \cdot (b')^M
$$
as $N, M$ range over admissible hop counts within budget $B$,
where the approximation holds up to a multiplicative
$(1 + o(1))$ factor.

Then for any query distribution $Q$ supported on the reachable
set,
$$
\mathbb{E}_{v \sim Q}\bigl[\#\text{paths}(v; N, M; B)\bigr] \approx D^N \cdot (b')^M
$$
with the same asymptotic factor.

**Proof.** Direct by linearity of expectation, applied to the
log-path-count under the homogeneity assumption. $\square$

**Remark.** The homogeneity assumption here is restrictive. For
inhomogeneous graphs (Wikipedia, globally), the path-count
function has different growth rates in different regions of $V$,
and the lemma fails. This is what design note §4.5 documents and
§5.6.2 reframes as the homogeneity precondition. For the
homogeneous topical core (`Category:Articles` reach), the lemma
holds empirically with $D \approx 7.34$ and $b' \approx 11$.

### 2.2 Weights normalise path counts

**Proposition 2.2.** Under the homogeneity assumption of Lemma 2.1
with calibration $b \cdot D = b'$,
$$
w(p) \cdot \#\text{paths}\bigl(\cdot;\ N(p), M(p);\ B\bigr) \approx 1
$$
for all paths $p$, i.e. the weight formula is precisely a
path-count normaliser.

**Proof.** Direct substitution:
$$
w(p) = D^{-N(p)} (b \cdot D)^{-M(p)}
     = D^{-N(p)} (b')^{-M(p)}
$$
By Lemma 2.1,
$\#\text{paths}(N, M) \approx D^N (b')^M$, so the product is
$1$ up to the $(1+o(1))$ factor. $\square$

This is the formal statement of design note §5.6's "weights as
path-count normalisers" reframing.

### 2.3 Convergence theorem (geometric series bound)

**Theorem 2.3.** Suppose Lemma 2.1 holds with growth constants
$D$ and $b'$, and suppose the metric is calibrated with
$b \cdot D > b'$. Define the convergence ratio
$$
r := \frac{b'}{b \cdot D} < 1
$$
Then the contribution to the numerator of $d_{\text{wPow}}$ from
paths with $M \ge 1$ child hops, weighted by the length factor
$(h+1)^{-n}$, is bounded by
$$
\sum_{M \ge 1} \mathrm{contribution}(M) \le
\frac{r}{1 - r} \cdot d_{\text{wPow}}(M = 0)
$$
where $d_{\text{wPow}}(M = 0)$ denotes the contribution from
parent-only paths.

**Proof.** By Lemma 2.1 and Proposition 2.2, the *weighted*
contribution at level $(N, M)$ is approximately
$$
w \cdot \#\text{paths}(N, M) \cdot (h+1)^{-n}
  \approx (h+1)^{-n}
$$
i.e. the abundance and weight cancel. The contribution at level
$M$ summed over $N$ is therefore proportional to
$D^N \cdot (b')^M \cdot D^{-N} \cdot (b \cdot D)^{-M} = (b')^M / (b \cdot D)^M = r^M$
times the length factor. Summing the geometric series in $M$
gives the stated bound. $\square$

**Corollary 2.3.1** (informal version of the convergence
conjecture). Under the assumptions of Theorem 2.3, the
contribution of child-using paths to $d_{\text{wPow}}$ is bounded
by $r/(1-r)$ times the parent-only contribution. When $r$ is
small (typical: $r \approx 0.157$ on simplewiki topical core), the
contribution is small. **This is a partial proof of Conjecture
3.1 under the additional homogeneity assumption.**

## 3. Conjectures

### 3.1 The central conjecture: convergence as a sufficient condition

**Conjecture (Main).** There exists a monotone-increasing function
$\psi : [0, 1) \to \mathbb{R}_{\ge 0}$ with $\psi(0) = 0$ and
$\psi(r) \to \infty$ as $r \to 1^-$, such that for any $(G, \mu, Q, B)$
satisfying:

(a) *Homogeneity precondition*: the path-count growth in §0.4 is
well-defined and uniform across the support of $Q$, yielding
constants $D(G, Q)$, $b'(G, Q, B)$, $b_{\text{eff}}(G, Q)$.

(b) *Convergence condition*:
$b_{\text{eff}}(G, Q) \cdot D(G, Q) > b'(G, Q, B)$.

Then
$$
\mathrm{TLI}(G, \mu, Q, B) \le \psi\!\left(\frac{b'}{b_{\text{eff}} \cdot D}\right)
$$

In words: under the homogeneity precondition, the TLI is bounded
by a function of the convergence ratio $r = b'/(b_{\text{eff}} \cdot D)$.

**Partial support.** Theorem 2.3 establishes the bound with
$\psi(r) = r/(1-r)$ under additionally assuming exact path-count
factorisation (Lemma 2.1) and exact calibration $b \cdot D = b'$.
In practice $b_{\text{eff}}$ and $b'$ agree only approximately
(within ~15% on simplewiki, §4.5 of the design note); the
conjecture extends Theorem 2.3 to the empirical regime where
this approximation is not exact.

### 3.2 Topical scoping is sufficient for homogeneity on Wikipedia categories

**Conjecture 3.2.** For Wikipedia category graphs of the form
"all descendants of a single top-level topical root" (e.g.
`Category:Articles` on simplewiki, `Category:Main_topic_classifications`
on enwiki), the homogeneity precondition of Conjecture 3.1 holds
to a degree sufficient for low TLI in practice.

**Partial support.** Design note §4.5 measures
$b_{\text{eff}}$ on `Category:Articles` (9.59) and on
`Category:Physics` (9.51), finding ~1% agreement. This is
evidence that the topical core is *statistically self-similar*
across different sub-trees — a necessary condition for
homogeneity.

**Caveat.** The conjecture has been tested on simplewiki only.
The enwiki topical-root verification (page_id 7345184 confirmed,
LMDB construction pending) is task #14 in the design note.

### 3.3 The convergence inequality is the operative condition

**Conjecture 3.3.** For graphs satisfying the homogeneity
precondition, $b_{\text{eff}} \cdot D > b'$ is the operative
condition for low TLI — i.e. the *graph-structural* properties
sometimes invoked as sufficient (power-law tail, dominant parent
rule, sparse cross-edges; see design note §5.1) act only by
producing this inequality. Any graph property that yields
$b_{\text{eff}} \cdot D > b'$ produces low TLI; conversely, no
graph property that fails to produce the inequality can produce
low TLI.

This is the "decoupling geometry from metric" claim of design
note §5.6.2 stated as a falsifiable conjecture.

### 3.4 Symmetric DAGs have high TLI under any directional metric

**Conjecture 3.4.** Let $G$ be a directed graph where, in
calibration,
$\mathbb{E}[d_c^2] \approx \mathbb{E}[d_p^2]$ (symmetric DAG, no
parent/child distinction). Then for *any* directionally-weighted
metric of the form
$w(p) = c_1^{N(p)} c_2^{M(p)}$ with constants $c_1, c_2$ derived
from $G$'s degree distribution, $\mathrm{TLI}(G, \mu, Q, B)$
is bounded *below* (not above) by a function of $B$ growing as
$B \to \infty$.

In words: symmetric DAGs cannot have low TLI under any
calibration of the directional metric. The directional asymmetry
is *necessary*, not just sufficient.

This conjecture motivates the synthetic-graph falsification task
(task #10 in the design note).

## 4. Empirical status

For each conjecture, we summarise the empirical evidence
recorded in the design note. See `TREE_LIKENESS_INDEX.md` §4
for full details.

### 4.1 Conjecture 3.1 (main)

- Aggregate TLI: ~0.02% on simplewiki rooted at Physics, $n = 20$
  pairs, budget $B = 15$, child-step costs $cc = 100$ vs $cc = 5$
  (design note §4.1).
- Per-pair TLI: ~0.007% worst case across all 20 pairs (design
  note §4.2).
- Convergence ratio: $r \approx 0.157$ with topical calibration
  ($b_{\text{eff}} = 9.59$, $D = 7.34$, $b' \approx 11$).
- Theorem 2.3 bound: $\psi(0.157) = 0.157/(1 - 0.157) \approx 0.186$,
  i.e. ~18.6% drift bound. Empirical: ~0.02%. The bound is loose;
  see §5.1 below.

### 4.2 Conjecture 3.2 (topical homogeneity)

- `Category:Articles` $b_{\text{eff}} = 9.59$ (79,797 reachable
  nodes).
- `Category:Physics` $b_{\text{eff}} = 9.51$ (79,199 reachable
  nodes).
- Agreement: ~1% (design note §4.5).
- Global (not topical) $b_{\text{eff}} = 589$, an order-of-
  magnitude discrepancy — direct evidence that homogeneity fails
  globally.

### 4.3 Conjecture 3.3 (b·D > b' is operative)

The robustness band documented in design note §5.5 shows three
calibration regimes (global with routing, topical with routing,
topical without routing) all yielding TLI < 0.1% despite
producing $b_{\text{eff}} \cdot D$ values that range from 27 to
1659. The common factor is that all three exceed $b' \approx 11$.
This is suggestive evidence but does not establish necessity.

### 4.4 Conjecture 3.4 (symmetric DAG falsification)

Pending: synthetic-graph experiment (task #10). The Cohen-Havlin
theorem (§1.3) implies that scale-free graphs in the
$2 < \gamma < 3$ regime are *not* asymptotically symmetric, so a
falsification requires explicit construction.

## 5. Open theoretical questions

### 5.1 Tightness of the convergence bound

Theorem 2.3 establishes $\mathrm{TLI} \le r/(1 - r)$ under the
exact-factorisation assumption. On simplewiki this bounds TLI by
18.6%, while empirically TLI is ~0.02% — a factor of ~1000
gap. The bound is loose because:

- It assumes worst-case length factors $(h+1)^{-n}$ for child
  paths, but in practice child paths are *longer*, so the length
  factor *additionally* suppresses them.
- It does not exploit the structure of the power-mean
  aggregation (the Lipschitz constants are smaller than the
  worst-case sum).

**Open question.** Is there a tighter version of Theorem 2.3 that
uses the length-factor suppression to give a bound matching the
empirical ratio? Conjecturally, $\psi(r) \sim r^{n+1}$ or similar
for power-mean exponent $n$, but this needs proof.

### 5.2 Quantitative homogeneity ↔ calibration error

Conjecture 3.2 asserts that topical scoping is "sufficient for
homogeneity in practice". Made precise: how much deviation from
exact homogeneity can be tolerated before the convergence bound
of Theorem 2.3 fails to apply? Specifically, if the path-count
factorisation in Lemma 2.1 holds with multiplicative error
$(1 + \delta)$, what is the corresponding error in the TLI
bound?

A quantitative answer here would convert Conjecture 3.2 from
"this seems to work" to a calibration-error budget.

### 5.3 Scale-free regime under topical scoping

Cohen–Havlin (§1.3) implies that scale-free graphs with
$2 < \gamma < 3$ have ultra-small-world geometry. Wikipedia's
*global* degree distribution fits $\gamma \approx 2.41$, which is
in this range. The conjecture is that *topical* sub-graphs do
not satisfy the precondition of Cohen-Havlin (because they are
not homogeneously scale-free), so they remain in the small-world
or tree-like geometric regime.

**Open question.** Empirically measure the topical sub-graph's
degree distribution and compute its power-law exponent (if it
fits one). Test whether $\gamma_{\text{topical}} > 3$, which
would put the topical sub-graph in the *small-world* but not
ultra-small-world regime, consistent with the observed TLI.

(Task #15 in the design note will partly answer this by
measuring average pair distance on the topical sub-graph.)

### 5.4 Extension to undirected graphs

The current theory assumes a directed graph with a clear
parent/child decomposition. What happens when:

- The graph is undirected, or
- The graph is directed but symmetric (every edge has a reverse
  edge of equal weight)?

In these settings, the directional asymmetry $b > 1$ vanishes,
and Theorem 2.3 predicts $\mathrm{TLI} \to 1$. This is consistent
with intuition: an undirected graph has no way for tree-search
(which depends on a parent direction) to be a meaningful
approximation. But the formal statement is conditional on the
parent/child decomposition existing in the first place.

**Open question.** Is there a generalisation of TLI to
undirected graphs that uses some other asymmetry (e.g.,
spectral, expansion-based) in place of directional?

### 5.5 Connection to spectral expansion

The convergence ratio $r = b'/(b \cdot D)$ has a spectral
interpretation we have not yet developed:

- $D$ is the average degree, related to the largest eigenvalue
  $\lambda_1$ of the adjacency matrix.
- $b$ involves second-moment ratios, related to the variance of
  the degree distribution.
- $b'$ is a path-count growth rate, related to higher-order
  expansion properties.

**Open question.** Is there a relationship between $r$ and the
spectral gap $\lambda_1 - \lambda_2$, or the second-largest
eigenvalue magnitude? A graph with a large spectral gap is an
expander — fast-mixing, with quickly-growing neighbourhood
sizes. Is rapid path-count growth (large $b'$) equivalent to
poor spectral gap, and is this why ultra-small-world graphs
(Cohen-Havlin) defeat TLI even under topical scoping?

## 6. References

- Chung, F. and Lu, L. (2002). *The average distances in random
  graphs with given expected degrees*. PNAS 99(25): 15879–15882.
- Watts, D. J. and Strogatz, S. H. (1998). *Collective dynamics
  of "small-world" networks*. Nature 393: 440–442.
- Cohen, R. and Havlin, S. (2003). *Scale-free networks are
  ultrasmall*. Physical Review Letters 90(5): 058701.
- Clauset, A., Shalizi, C. R., and Newman, M. E. J. (2009).
  *Power-law distributions in empirical data*. SIAM Review
  51(4): 661–703.
- Feld, S. L. (1991). *Why your friends have more friends than
  you do*. American Journal of Sociology 96(6): 1464–1477.
- [`TREE_LIKENESS_INDEX.md`](TREE_LIKENESS_INDEX.md) (companion
  design note: empirical observations, calibration, algorithmic
  consequences).
