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
$$\mathrm{parents}(v) \subseteq V, \quad
\mathrm{children}(v) \subseteq V$$
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
- $b_{\text{eff}} := \dfrac{\mathbb{E}[d_c^2]/\mathbb{E}[d_c]}{\mathbb{E}[d_p^2]/\mathbb{E}[d_p]}$, the
  **friendship-paradox-corrected branching asymmetry** (the
  calibrated quantity the kernel actually uses; "size-biased
  child branching" divided by "size-biased parent branching"). See
  §1.5 for the Feld correction.
- $b'$ := the empirical per-child-hop path-count growth (defined
  in §0.4 below)
- $b := \mathbb{E}[d_c^2] / \mathbb{E}[d_p^2]$, the raw
  second-moment ratio — this is the *early-formula* calibration
  used in the design note's pre-correction analysis. It is not
  used in the kernel or in this document beyond historical
  context; $b_{\text{eff}}$ is the canonical asymmetry quantity.

The path weight is
$$w(p) = D^{-N(p)} \cdot (b_{\text{eff}} \cdot D)^{-M(p)}$$
and the **directionally-weighted power-mean metric** with exponent
$n$ is
$$d_{\text{wPow}}(v; B, cc) = \left(
\frac{\sum_{p \in \mathcal{P}(v; B, cc)} w(p) \cdot (h(p)+1)^{-n}}
     {\sum_{p \in \mathcal{P}(v; B, cc)} w(p)}
\right)^{-1/n}$$
where $\mathcal{P}(v; B, cc)$ is the set of paths from $v$ to $r$
with cost at most $B$ when each parent hop costs $1$ and each
child hop costs $cc$. We typically use $n = 2$ in experiments.

**Compatibility with the early formula.** The original design
note uses the early $b$ in some prose. The substitution
$b \mapsto b_{\text{eff}}$ changes the numerical value of
$b \cdot D$ but not the *form* of the weight formula. All
theorems and conjectures below are stated in terms of
$b_{\text{eff}}$.

### 0.4 Path-count growth

For a node $v$ reachable to $r$ within budget $B$, define
$$\mathrm{paths}(v; N, M; B) :=
\bigl|\{p : v \to r,\ p \text{ has } N \text{ parent hops}
              \text{ and } M \text{ child hops, cost} \le B\}\bigr|$$
At any fixed budget $B$ and child-step cost $cc > 0$, the maximum
admissible $M$ is finite ($\lfloor B/cc \rfloor$), so a literal
$M \to \infty$ limit does not exist. We instead define $b'$ as
the **asymptotic per-child-hop path-count growth rate** under
the homogeneity assumption: there exists a constant $b' \ge 1$
and a node-dependent constant $C(v) > 0$ such that, for $M$ in
the range admissible at $(B, cc)$,
$$\mathbb{E}_{v \sim Q}\bigl[\mathrm{paths}(v; \cdot, M; B)\bigr]
\approx C \cdot (b')^M$$
where $\mathrm{paths}(v; \cdot, M; B)$ denotes summation over $N$
and the approximation holds up to a multiplicative
$(1 + o(1))$ factor uniformly in $M$.

This makes $b'$ a property of the *graph* (and query
distribution), not of the specific $(B, cc)$ probe used to
estimate it. Different $(B, cc)$ pairs that admit overlapping
$M$ ranges should yield consistent $b'$ estimates under
homogeneity; the design note's §4.4 measurements at $cc \in
\{100, 10, 5, 3\}$ are such an overlapping family.

**Estimation in practice.** At fixed $B$, the maximum admissible
$M$ is finite, so $b'$ cannot be extracted from a single
$(B, cc)$ measurement. Instead, we run the kernel at a sequence
of decreasing $cc$ values that admit successively more child
hops. For two adjacent values $cc_1 > cc_2$ with
$M_{\max}(B, cc_1) = m_1$ and $M_{\max}(B, cc_2) = m_2 > m_1$,
under homogeneity
$$\frac{T(B, cc_2)}{T(B, cc_1)}
\;\approx\;
\frac{\sum_{M=0}^{m_2} (b')^M}{\sum_{M=0}^{m_1} (b')^M}$$

where $T(B, cc)$ denotes the total path count at budget $B$ and
child-step cost $cc$ — summed over $N$ and $M$ within budget.
so taking the geometric mean over transitions where
$m_{k+1} - m_k = 1$ recovers $b'$ directly. For larger
$M$ increments, the appropriate root corrects for the number
of admitted child-hop levels.

In practice (design note §4.4), at $B = 15$ we use $cc \in
\{100, 10, 5, 3\}$ with corresponding $M_{\max} \in \{0, 1, 3, 5\}$.
The transition ratios are $1.05$, $122.7$, $18.7$. The
$cc = 10 \to 5$ transition admits *two additional* $M$ levels
($M = 2$ and $M = 3$), so the per-child-hop growth recovers as
$b' \approx (122.7)^{1/2} \approx 11.1$ — the exponent $1/2$ is
$1/(\Delta M_{\max})$, the inverse of the number of new
child-hop levels admitted at the transition. The
$cc = 100 \to 10$ ratio is essentially flat (1.05) because that
transition only admits $M = 1$ paths, which are few. The
geometric mean across non-degenerate transitions gives
$b' \approx 11$ on simplewiki topical core.

### 0.5 The tree-likeness index

The **tree-likeness index** of the tuple $(G, \mu, Q, B)$ where
$\mu$ is the directionally-weighted metric is
$$\mathrm{TLI}(G, \mu, Q, B) :=
\frac{\bigl|\mathbb{E}_{v \sim Q}\bigl[d_{\text{wPow}}(v; B, \infty)\bigr] -
            \mathbb{E}_{v \sim Q}\bigl[d_{\text{wPow}}(v; B, 0^+)\bigr]\bigr|}
     {\mathbb{E}_{v \sim Q}\bigl[d_{\text{wPow}}(v; B, 0^+)\bigr]}$$
i.e. the relative drift of the mean metric value as the child-step
cost ranges from infinity (tree-search, $M = 0$ only) to
$0^+$ (full DAG search, all $M$ admitted within budget).

**Denominator choice.** We use
$d_{\text{wPow}}(v; B, 0^+)$ — the full-DAG metric value — as the
reference. The interpretation: tree-search is an *approximation*
to the full-DAG value, and TLI is the relative approximation
error of that approximation. Using $d_{\text{wPow}}(v; B, \infty)$
as the denominator would give the inverse error; the absolute
value in the numerator makes the index sign-agnostic
regardless. (For the empirical drifts observed on simplewiki,
the difference between the two normalisations is
~0.02% × O(TLI) and immaterial.)

**Absolute value.** The convergence direction is not guaranteed
to be monotonic in $cc$ for all $(G, \mu, Q, B)$ — admitting
more paths can either decrease or increase $d_{\text{wPow}}$
depending on whether the new paths have above- or below-average
$(h+1)^{-n}$ values. The absolute value handles both
directions; empirically on simplewiki the drift is positive (the
full-DAG value is slightly higher).

In practice we estimate this with a finite contrast between two
$cc$ values, typically $cc = 100$ (effectively tree-search at
$B = 15$) and $cc = 5$ (admits up to 3 child hops); see design
note §4.1.

### 0.6 Statistical homogeneity

A precondition many of the §2 results and §3 conjectures rest on.

**Definition 0.6 (Statistical homogeneity).** A tuple
$(G, Q, B)$ is *statistically homogeneous* if there exist
constants $D, b' \ge 1$ such that, uniformly over $v$ in the
support of the marginal distribution $Q$:

(H1) $\mathbb{E}\bigl[|\mathrm{children}(v)|\bigr] = D$
(within multiplicative factor $1 + o(1)$);

(H2) The path-count growth in §0.4 has the same $b'$ at every
$v$ (i.e. $C(v)$ varies but the exponential base does not);

(H3) The calibrated branching asymmetry $b_{\text{eff}}$
computed from local degree distributions agrees with the global
$b_{\text{eff}}$ within multiplicative factor $1 + o(1)$.

In words: every region the query distribution can reach
exhibits the same statistical fingerprint $(D, b_{\text{eff}}, b')$
within tolerance.

**Inhomogeneity decomposition.** A tuple is *piecewise
homogeneous* if $\mathrm{supp}(Q)$ decomposes as
$\bigsqcup_i V_i$ where each $(G|_{V_i}, Q|_{V_i}, B)$ is
homogeneous. In this case the theorems below apply
*piecewise* — each subgraph has its own $(D_i, b'_i,
b_{\text{eff},i})$.

**Important caveat.** The piecewise framing only applies when $Q$
is restricted to a single component $V_i$ at calibration time
(equivalently, the calibration sees only the subgraph
$G|_{V_i}$). If $Q$ mixes nodes from multiple components — i.e.
the query distribution genuinely spans different statistical
regimes — Lemma 2.1's factorisation fails (different parts of
$\mathrm{supp}(Q)$ produce different $(D, b')$ values), and
neither Theorem 2.3 nor Corollary 2.3.1 applies. The
recommended workflow is to *constrain* $Q$ to a single
homogeneous component via the topical-root LMDB construction
(design note §7.1), not to apply theorems piecewise to a query
that crosses components.

This is the setting design note §4.5 documents on Wikipedia: the
global graph is piecewise homogeneous (topical regime +
administrative regime) but not globally homogeneous, and the
production recipe constrains $Q$ to the topical component
specifically because cross-component queries break the theory.

## 1. Background: established results we build on

### 1.1 Random-graph distances (Chung–Lu)

**Theorem (Chung & Lu, 2002).** For a random graph drawn from the
given-expected-degree model with weights
$\mathbf{w} = (w_1, \dots, w_n)$ — where edge $(i,j)$ is present
with probability $w_i w_j / \sum_k w_k$ — and average degree
$\langle k \rangle = \mathbb{E}[w] > 1$, the average shortest-path
length between random node pairs satisfies
$$L \sim \frac{\log n}{\log \tilde{d}}$$
as $n \to \infty$, where $\tilde{d} = \mathbb{E}[w^2] / \mathbb{E}[w]$
is the *expected size-biased degree* (the average degree of a
random endpoint of a random edge).

The Erdős–Rényi $G(n, p)$ result with $\tilde{d} = \langle k \rangle$
is a special case (constant-weight model).

Source: F. Chung and L. Lu, "The average distances in random
graphs with given expected degrees", *PNAS* 99(25): 15879–15882
(2002).

**Relevance to TLI.** For graphs with heavy-tailed degree
distribution (like Wikipedia categories), $\tilde{d} \gg
\langle k \rangle$ because the second moment is dominated by
hubs. The Chung-Lu theorem then predicts substantially shorter
distances than the constant-degree estimate would suggest. This
size-biased correction is the same one entering our
$b_{\text{eff}}$ definition (§1.5).

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

**Relevance to TLI: only the distance criterion is operative.**
The clustering coefficient $C$ does not appear in the theorems
or conjectures below; only the $L = \Theta(\log n)$ condition is
used (to classify regimes in §5.3). Strictly, Erdős-Rényi and
Chung-Lu random graphs are *not* small-world by Watts-Strogatz's
full definition (they have $C \to 0$), but they share the
distance scaling. The operative concept in this document is
"graphs with $L = \Theta(\log n)$" — call this the
*logarithmic-diameter regime*. It includes both Watts-Strogatz
small-worlds *and* random graphs, and is the regime we compare
against tree-like and ultra-small-world graphs in §5.3.

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
$$\hat\gamma(d_{\min}) = 1 + n_{\ge d_{\min}} \biggm/
\sum_{d_i \ge d_{\min}} \log(d_i / d_{\min})$$
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
$$\tilde{d} := \frac{\mathbb{E}[k^2]}{\mathbb{E}[k]} \ge \mathbb{E}[k]$$
with equality iff degrees are constant. The quantity $\tilde{d}$
is the *size-biased mean* or **Chung-Lu effective degree** (cf.
§1.1).

Source: S. L. Feld, "Why your friends have more friends than you
do", *American Journal of Sociology* 96(6): 1464–1477 (1991).

**Relevance to TLI: $b_{\text{eff}}$ is the ratio of Chung-Lu
effective degrees.** Applying the size-biased mean separately to
the child and parent edge populations gives
$$\tilde{d}_c = \frac{\mathbb{E}[d_c^2]}{\mathbb{E}[d_c]}, \qquad
\tilde{d}_p = \frac{\mathbb{E}[d_p^2]}{\mathbb{E}[d_p]}$$
and our calibrated branching asymmetry is exactly the ratio
$$b_{\text{eff}}
= \frac{\mathbb{E}[d_c^2]/\mathbb{E}[d_c]}{\mathbb{E}[d_p^2]/\mathbb{E}[d_p]}
= \frac{\tilde{d}_c}{\tilde{d}_p}$$
i.e. the Chung-Lu effective degree in the child direction divided
by the Chung-Lu effective degree in the parent direction. **Low
TLI thus has a clean interpretation: the child-direction
Chung-Lu effective degree dominates the parent-direction one by
a factor large enough that the convergence inequality
$b_{\text{eff}} \cdot D > b'$ holds.**

The raw second-moment ratio $b = \mathbb{E}[d_c^2]/\mathbb{E}[d_p^2]$
omits the first-moment factors and is only approximately equal
to $b_{\text{eff}}$ when $\mathbb{E}[d_c] \approx \mathbb{E}[d_p]$;
on simplewiki the two differ by the factor 0.436 (design note
§4.4), the same first-moment correction.

### 1.6 Tree pair distance

**Lemma (folklore).** In a balanced $D$-ary tree of depth $L$ with
$n = (D^{L+1} - 1)/(D - 1)$ nodes:

- Depth-to-root: $L = \log_D(n(D - 1) + 1) \approx \log_D n$
- Average pair distance: for uniform-random distinct nodes $u, v$,
$$\mathbb{E}_{u,v}[d(u,v)] \approx 2L - \frac{2D}{D-1} \approx 2 \log_D n$$
with the constant offset $2D/(D-1)$ converging to $2$ as
$D \to \infty$ (large branching makes LCAs shallow).

**Proof sketch.** The distance between $u$ and $v$ is
$\mathrm{depth}(u) + \mathrm{depth}(v) - 2 \cdot \mathrm{depth}(\mathrm{LCA}(u,v))$.
For uniform-random $u, v$ in the leaves of a balanced tree, the
probability that the LCA is at depth $\ge \ell$ is approximately
$D^{-\ell}$ (both leaves must descend from the same depth-$\ell$
ancestor). The expected LCA depth is therefore
$\sum_{\ell \ge 0} D^{-\ell} \cdot \mathbf{1}[\ell \le L] \approx \sum_{\ell \ge 0} D^{-\ell}
= D/(D-1)$, and pair distance is
$2L - 2 \cdot D/(D-1)$. $\square$

**Relevance to TLI: regime-comparison context only.** This lemma
appears in §5.3 to provide a benchmark for "tree-shaped graph
average pair distance", which we compare against
small-world and ultra-small-world distance scalings. The lemma
is *not* used to argue $\mathrm{TLI} = 0$ for trees — that
follows trivially from the definition (a tree has no cross-edge
paths, so $\mathrm{paths}(v; N, M; B) = 0$ for all $M \ge 1$,
hence $d_{\text{wPow}}$ does not depend on $cc$ and TLI = 0
identically).

## 2. What we can prove

We state three results that follow from the definitions in §0 and
mild homogeneity assumptions. They are *not* the central
conjecture (§3.1); they are building blocks supporting it.

### 2.1 Path-count growth lemma (conditional on homogeneity)

**Lemma 2.1.** Suppose $(G, Q, B)$ is statistically homogeneous in
the sense of Definition 0.6. Then for any node $v$ in the support
of $Q$,
$$\mathrm{paths}(v; N, M; B) \approx D^N \cdot (b')^M$$
as $N, M$ range over admissible hop counts within budget $B$,
where the approximation holds up to a multiplicative
$(1 + o(1))$ factor. Consequently,
$$\mathbb{E}_{v \sim Q}\bigl[\mathrm{paths}(v; N, M; B)\bigr] \approx D^N \cdot (b')^M$$
with the same asymptotic factor.

**Proof.** Condition (H2) of Definition 0.6 fixes the asymptotic
behaviour in $M$ at every $v$; condition (H1) fixes the
asymptotic behaviour in $N$ (since the number of distinct
$N$-step parent paths from $v$ scales as $D^N$ when each step
expands by average factor $D$). Combining, and taking
expectation by linearity, gives the stated form. $\square$

**Remark.** The homogeneity assumption here is restrictive. For
globally inhomogeneous graphs (Wikipedia at full scope), the
path-count function has different growth rates in different
regions of $V$, and the lemma fails. This is what design note
§4.5 documents and §5.6.2 reframes as the homogeneity
precondition. For the homogeneous topical core
(`Category:Articles` reach), the lemma holds empirically with
$D \approx 7.34$ and $b' \approx 11$.

### 2.2 Weights normalise path counts

**Proposition 2.2.** Under Definition 0.6 homogeneity with
calibration $b_{\text{eff}} \cdot D = b'$,
$$w(p) \cdot \mathrm{paths}(v;\ N(p), M(p);\ B) \approx 1$$
for all paths $p$ from $v$ to $r$, i.e. the weight formula is
precisely a path-count normaliser.

**Proof.** Direct substitution:
$$w(p) = D^{-N(p)} (b_{\text{eff}} \cdot D)^{-M(p)}
     = D^{-N(p)} (b')^{-M(p)}$$
By Lemma 2.1,
$\mathrm{paths}(v;\ N, M;\ B) \approx D^N (b')^M$, so the
product is $1$ up to the $(1+o(1))$ factor. $\square$

This is the formal statement of design note §5.6's "weights as
path-count normalisers" reframing.

**Caveat — calibration is approximate.** Proposition 2.2 assumes
exact $b_{\text{eff}} \cdot D = b'$. Empirically (design note
§4.5), simplewiki topical calibration gives
$b_{\text{eff}} \approx 9.59$ and $b' \approx 11$, so the
weight-as-normaliser correspondence is only approximate (within
~15%). The next theorem operates in the realistic regime where
$b_{\text{eff}} \cdot D$ exceeds $b'$ by some margin but does
not equal it exactly.

### 2.3 Convergence theorem (geometric series bound)

**Theorem 2.3.** Suppose $(G, Q, B)$ is homogeneous (Definition
0.6) with growth constants $D$ and $b'$, and the metric is
calibrated with $b_{\text{eff}} \cdot D > b'$. Define the
**convergence ratio**
$$r := \frac{b'}{b_{\text{eff}} \cdot D} \in [0, 1)$$
Let
$$S_M := \sum_{p:\ M(p) = M} w(p) \cdot (h(p)+1)^{-n}, \qquad
W_M := \sum_{p:\ M(p) = M} w(p)$$
be the M-level *numerator* and *denominator* contribution sums
to $d_{\text{wPow}}^{-n}$. Then both ratios are bounded by the
same geometric series:
$$\frac{\sum_{M \ge 1} S_M}{S_0} \le \frac{r}{1 - r}, \qquad
\frac{\sum_{M \ge 1} W_M}{W_0} \le \frac{r}{1 - r}$$

**Proof.** By Lemma 2.1, the number of paths at level $(N, M)$
is $D^N (b')^M$. Each carries weight
$$w(p) = D^{-N} \cdot (b_{\text{eff}} \cdot D)^{-M}$$
Multiplying path count by weight, the $D^{\pm N}$ factors cancel:
$$D^N (b')^M \cdot D^{-N} (b_{\text{eff}} \cdot D)^{-M}
\;=\; \frac{(b')^M}{(b_{\text{eff}} \cdot D)^M}
\;=\; \left(\frac{b'}{b_{\text{eff}} \cdot D}\right)^M
\;=\; r^M$$
i.e. the weighted path count at level $(N, M)$ is $r^M$ for every
$N$, with the length factor $(h+1)^{-n}$ multiplying for the
numerator and $1$ for the denominator.

For the numerator,
$$S_M \approx \sum_N r^M \cdot (N+M+1)^{-n} = r^M \cdot L_M$$
where $L_M := \sum_{N \in [d_{\min},\ B - M \cdot cc]} (N+M+1)^{-n}$.

**Claim: $L_M \le L_0$ for all $M \ge 0$.** For each $N$ in the
admissible range at level $M$ (which is a subset of the level-0
range $[d_{\min}, B]$ truncated from the top by $M \cdot cc$),
$(N+M+1)^{-n} \le (N+1)^{-n}$ pointwise since $x \mapsto x^{-n}$
is decreasing. So $L_M$ is term-by-term dominated and has fewer
or equal terms compared to $L_0$, giving $L_M \le L_0$. Hence
$S_M / S_0 \le r^M$ and the geometric sum gives
$\sum_{M \ge 1} S_M / S_0 \le r/(1-r)$.

For the denominator,
$$W_M \approx \sum_N r^M = r^M \cdot R_M$$
where $R_M$ is the size of the admissible $N$-range at level $M$.

**Claim: $R_M \le R_0$ for all $M \ge 0$.** The level-$M$ range
$[d_{\min}, B - M \cdot cc]$ is a (possibly empty) subset of the
level-0 range $[d_{\min}, B]$ obtained by truncating from the
top, so $R_M \le R_0$. Hence $W_M / W_0 \le r^M$, hence
$\sum_{M \ge 1} W_M / W_0 \le r/(1-r)$. $\square$

**Caveats and tightness.**

1. **The bound is on *contribution sums*, not directly on TLI.**
   Since $d_{\text{wPow}} = (\text{numerator} / \text{denominator})^{-1/n}$
   is a *ratio*, and both numerator and denominator drift by
   $r/(1-r)$, the actual TLI is much smaller than this individual
   bound would suggest — the drifts partially cancel. A first-order
   estimate: $\mathrm{TLI} \lesssim |S_{\text{drift}} - W_{\text{drift}}|/n$,
   typically a fraction of $r/(1-r)$ rather than the full value.
   §5.1 discusses this in more depth and conjectures a tighter
   $O(r^2/(1-r))$ bound following GPT's review observation.

2. **The proof bounds $L_M \le L_0$.** A more careful analysis
   would track $L_M$ as a function of $M$ explicitly. For the
   simplewiki budget $B = 15$ with $cc = 5$ and $d_{\min} = 4$,
   $L_M$ decreases roughly as $\zeta(n) - \sum_{k = 1}^{M} k^{-n}$,
   contributing a sub-geometric factor that further tightens the
   bound. §5.1 leaves this as an open question for a sharper
   $\psi$.

3. **The bound does not require $b_{\text{eff}} \cdot D = b'$
   exactly.** As long as $b_{\text{eff}} \cdot D > b'$, $r < 1$
   and the geometric series converges. The case $r \to 1$ is
   the boundary where convergence fails.

**Corollary 2.3.1.** Under the assumptions of Theorem 2.3, the
M ≥ 1 contributions to both the numerator and denominator of
$d_{\text{wPow}}$ are bounded by $r/(1-r)$ times the M = 0
contributions. When $r$ is small (typical: $r \approx 0.157$ on
simplewiki topical core), the *individual* contribution drifts
are small. The drift in $d_{\text{wPow}}$ itself is smaller still
because the numerator and denominator drift partially cancel —
empirically by ~1000× (theorem bound: 18.6%; empirical TLI:
~0.02%). **This is a partial proof of Conjecture 3.1 (the main
convergence conjecture) under the homogeneity assumption.**

## 3. Conjectures

Ordered by logical dependency: the central convergence conjecture
(3.1) presupposes statistical homogeneity (Def 0.6); two
strengthenings refine it (3.2 sharpens the sufficient condition,
3.3 asserts necessity); two specific instances ground it (3.4 the
Wikipedia topical case, 3.5 the falsification target); and one
production-engineering conjecture (3.6) is the last open question
from the design note.

### 3.1 The central conjecture: convergence as a sufficient condition

**Conjecture 3.1 (Main).** There exists a monotone-increasing
function $\psi : [0, 1) \to \mathbb{R}_{\ge 0}$ with $\psi(0) = 0$
and $\psi(r) \to \infty$ as $r \to 1^-$, such that for any
$(G, \mu, Q, B)$ satisfying:

(a) *Homogeneity precondition*: Definition 0.6 holds, yielding
constants $D(G, Q)$, $b'(G, Q, B)$, $b_{\text{eff}}(G, Q)$.

(b) *Convergence condition*:
$b_{\text{eff}}(G, Q) \cdot D(G, Q) > b'(G, Q, B)$.

Then
$$\mathrm{TLI}(G, \mu, Q, B) \le \psi\!\left(\frac{b'}{b_{\text{eff}} \cdot D}\right)$$

In words: under homogeneity, the TLI is bounded by a function
of the convergence ratio $r = b'/(b_{\text{eff}} \cdot D)$.

**Partial support.** Theorem 2.3 establishes the contribution-sum
bound $r/(1-r)$ under additionally assuming exact path-count
factorisation (Lemma 2.1) and exact calibration
$b_{\text{eff}} \cdot D = b'$. The translation to a TLI bound
is not direct because $d_{\text{wPow}}$ is a ratio; §5.1
conjectures the true $\psi$ is closer to $r^2/(1-r)$.

### 3.2 The convergence inequality is the operative condition (necessity)

**Conjecture 3.2.** For graphs satisfying the homogeneity
precondition, $b_{\text{eff}} \cdot D > b'$ is *both* necessary
and sufficient for low TLI — i.e. the graph-structural properties
sometimes invoked as sufficient (power-law tail, dominant parent
rule, sparse cross-edges; design note §5.1) act only by producing
this inequality. Any graph property that yields
$b_{\text{eff}} \cdot D > b'$ produces low TLI; any tuple with
$b_{\text{eff}} \cdot D \le b'$ has TLI bounded below.

**Partial support: sufficiency direction.** Design note §5.5
documents the robustness band: three calibration regimes (global
with routing, topical with routing, topical without routing)
yield TLI < 0.1% despite producing $b_{\text{eff}} \cdot D$
values from 27 to 1659. The common factor is all three exceed
$b' \approx 11$.

**Open: necessity direction.** We do not have clean empirical
evidence for the necessity claim (i.e. that $b_{\text{eff}} \cdot D
\le b'$ *forces* high TLI). The closest evidence is design note
§4.3's broken-ingest comparison ($b \approx 61 \cdot D = 448$,
TLI ≈ 1%) vs the correct-ingest version ($b \approx 9933$, TLI ≈
0.02%), which suggests *some* correlation between $b \cdot D$ and
TLI but doesn't reach the boundary $b \cdot D \le b'$. Conjecture
3.4 (synthetic graph at the boundary) is the natural empirical
test.

This is the "decoupling geometry from metric" claim of design
note §5.6.2 stated as a falsifiable conjecture.

### 3.3 Sharpening of the bound (refinement of Conjecture 3.1)

This conjecture *refines* Conjecture 3.1 by proposing a specific
form for the function $\psi$ — it is not an independent claim
but a guess at the leading-order asymptotic.

**Conjecture 3.3.** The function $\psi$ in Conjecture 3.1
satisfies $\psi(r) \le C(n, B, d_{\min}) \cdot r^{\alpha} / (1 - r)$
for some constant $C$ depending on $n$, the budget $B$, and the
minimum distance $d_{\min}$ from the support of $Q$ to the
root, with exponent $\alpha \ge 1$.

**Motivation.** Theorem 2.3 bounds the *individual*
contribution-sum drifts by $r/(1-r)$, but $d_{\text{wPow}}$ is
the ratio of these sums and the drifts partially cancel. The
question is *how much* they cancel — i.e. what $\alpha$ is.

**Status.** Mixed:

1. **First-order Taylor analysis at simplewiki parameters
   ($B = 15, cc = 5, n = 2$) shows substantial-but-not-complete
   cancellation.** See §5.1 for the verified numerical table:
   $L_1/L_0 \approx 0.631$ and $R_1/R_0 \approx 0.583$, so the
   $r^1$ coefficient is $\approx 0.048$ — small (5% of the
   worst-case) but not zero. Leading-order TLI prediction
   $\approx 0.44\%$, a 40× reduction from the contribution-sum
   bound of 18.6%.
2. **The empirical TLI is still ~22× smaller** than this
   first-order prediction (0.02% vs 0.44%). Source unclear —
   candidates include budget-truncated $M$ ranges (the $M = 3$
   level is empty at simplewiki parameters), higher-order
   Taylor terms with their own cancellation, or measurement
   variance in the aggregate over only 20 seed pairs.

A definitive answer requires either a careful expansion to
higher order in $r$, or empirical Monte Carlo at varying $r$ to
fit $\alpha$ directly. The conjecture as stated is *consistent
with* but not *fully derived from* the empirical observations:
the 40× drop from contribution-sum bound to first-order
prediction is genuine evidence of substantial cancellation,
but the remaining 22× gap is unexplained.

### 3.4 Topical scoping is sufficient for homogeneity on Wikipedia

**Conjecture 3.4.** For Wikipedia category graphs of the form
"all descendants of a single top-level topical root" (e.g.
`Category:Articles` on simplewiki,
`Category:Main_topic_classifications` on enwiki), the
homogeneity precondition of Definition 0.6 holds to a degree
sufficient for low TLI in practice (within tolerance for $\psi$
of Conjecture 3.1).

**Partial support.** Design note §4.5 measures
$b_{\text{eff}}$ on `Category:Articles` (9.59) and on
`Category:Physics` (9.51), finding ~1% agreement. This is
evidence that the topical core is *statistically self-similar*
across different sub-trees — a necessary condition for
homogeneity (H3 of Definition 0.6).

**Caveat.** The conjecture has been tested on simplewiki only.
The enwiki topical-root verification (page_id 7345184 confirmed
via Wikipedia API, LMDB construction pending) is task #14 in the
design note.

### 3.5 Symmetric DAGs have high TLI under any directional metric

**What TLI measures, precisely.** Before stating the conjecture,
worth emphasising what TLI actually captures: it is the
contribution of *shorter-via-child paths* to the metric value.
In a multi-parent DAG, the shortest $v \to r$ path can include
child hops — if some descendant of $v$ has a shorter parent
chain to root, then the route
$v \to \mathrm{child} \to \cdots \to r$ via that descendant is
shorter than any pure-parent route from $v$. Tree-search misses
these. Low TLI means: *either* such shorter-via-child paths
don't exist, *or* the metric weighting crushes them to
negligibility. The simplewiki TLI of 0.02% means one of these
holds in practice; symmetric DAGs (below) are constructed so
that neither does.

**Conjecture 3.5 (Falsification target).** Let $G$ be a directed
graph where, in calibration,
$\mathbb{E}[d_c^2] \approx \mathbb{E}[d_p^2]$ (symmetric DAG, no
parent/child distinction). Then for *any* directionally-weighted
metric of the form
$w(p) = c_1^{N(p)} c_2^{M(p)}$ with constants $c_1, c_2$ derived
from $G$'s degree distribution, $\mathrm{TLI}(G, \mu, Q, B)$ is
bounded *below* (not above) by a function of $B$ growing as
$B \to \infty$.

In words: symmetric DAGs cannot have low TLI under any
calibration of the directional metric. The directional asymmetry
is *necessary*, not just sufficient.

This conjecture motivates the synthetic-graph falsification task
(task #10 in the design note) and is the natural test of the
necessity direction of Conjecture 3.2.

### 3.6 Routing-correction redundancy under topical calibration

**Conjecture 3.6.** Under the topical-root LMDB construction
recipe (design note §7.1), the *routing correction* factor
$\rho := \mathrm{avg\_min\_dist} / \mathrm{avg\_path\_hops}$
becomes unnecessary as a separate calibration term — i.e. the
kernel-internal composition
$\mathrm{BranchRatio} = b_{\text{eff}} \cdot \rho$ should be
replaced by $\mathrm{BranchRatio} = b_{\text{eff}}$ when
calibrating on a homogeneous topical subgraph.

**Motivation.** The routing correction was introduced as a
heuristic compensation when $b_{\text{eff}}$ was inflated by
admin hubs in global calibration. Under topical scoping
$b_{\text{eff}}$ already matches $b'$ to within ~15% (design
note §4.5); applying $\rho \approx 0.384$ on top *deflates*
$b_{\text{eff}}$ to roughly $1/3$ of its honest value, which the
robustness band absorbs but is not principled. See design note
§5.4 for the discussion.

**Directionality clarification.** A stronger statement than
"unnecessary": $\rho < 1$ deflates $b_{\text{eff}}$, which
*decreases* $\mathrm{BranchRatio}$, *increases* the convergence
ratio $r = b'/(\mathrm{BranchRatio} \cdot D)$, and *loosens* the
Theorem 2.3 bound. Dropping the routing correction therefore
gives a *strictly tighter* theoretical bound on TLI, not just a
simpler formula. The conjecture is more precisely "the routing
correction was actively miscalibrating in the wrong direction,
and dropping it gives a better-justified bound while leaving
empirical TLI unchanged."

**Status.** *Tentative.* The drift probe under the alternate
composition has not been re-run on topical calibration. Task #14
in the design note is to settle this empirically by measuring
TLI under both compositions and verifying both remain below the
0.1% certificate threshold.

**Production consequence.** If confirmed, the data-prep recipe
(design note §7.1) and calibration recipe (§7.2) become
unconditionally simpler: drop the routing correction term, use
$b_{\text{eff}}$ directly. The theoretical bound becomes tighter
in the bargain. If refuted (i.e. dropping routing correction
substantially increases TLI), the routing correction stays but
its theoretical role becomes "calibration-error band-aid for an
unmodelled effect" rather than "principled factor."

#### 3.6.1 Bayesian restatement of Conjecture 3.6

The directionality observation from Conjecture 3.6 has a cleaner
statement in Bayesian language. This sub-section is an
*interpretive restatement* of the conjecture, not an
independent claim — the formal content is in Conjecture 3.6
above, and what follows is just a way of seeing it.

**The basic intuition.** Each path from $v$ to $r$ is a piece of
evidence about "the true distance" (whatever that means under
the metric). If we *expect* many branches per step, any
individual path is one of many similar paths and individually
carries little information. If we expect few branches, each path
is one of few candidates and carries more information per path.

The metric encodes this directly:
$w(p) = D^{-N(p)} \cdot (b_{\text{eff}} \cdot D)^{-M(p)}$. Larger
$b_{\text{eff}} \cdot D$ (more expected branching per child
hop) yields *smaller* per-path weight — fewer information per
path because many sibling paths exist.

**Without $\rho$ (the no-correction metric).** The weight
$w(p) \propto 1/\mathrm{paths}(v;\ N, M;\ B)$ under exact
calibration. This is the *maximum-entropy prior over path
shapes*: among all priors with the constraint "give each
$(N, M)$ stratum equal total mass after accounting for path
counts," the uniform-per-shape prior maximises entropy. (More
precisely: the prior is uniform over the discrete set of
$(N, M)$ shape classes, with no further information injected
about which paths within a shape are more likely.)

**With $\rho < 1$.** The weight becomes
$w(p) \propto 1/(\mathrm{paths}(v;\ N, M;\ B) \cdot \rho^M)$.
The extra factor $1/\rho^M > 1$ *upweights* M-hop child paths
by $(1/\rho)^M$ per child hop. Read as a prior shift: $\rho$
deflates the expected per-child-hop branching from
$b_{\text{eff}} \cdot D$ to $b_{\text{eff}} \cdot \rho \cdot D$,
i.e. claims "actually there is less branching than the moment
scan suggests, so each individual child path should be trusted
more."

**The Bayesian critique — qualified.** The TLI measurement
itself is the data on which the prior would be updated.
Empirically (design note §4), child paths contribute negligibly
to $d_{\text{wPow}}$ — TLI ≈ 0.02% on simplewiki Physics-rooted.
The data shows child paths are uninformative; $\rho$'s prior
shift claims they are *more* informative. So the prior shift
contradicts the data **directionally**.

A symmetric observation applies, however, to the no-$\rho$
metric: Proposition 2.2 requires *exact* calibration
$b_{\text{eff}} \cdot D = b'$, and empirically the agreement is
only within ~15% (design note §4.5). So the no-$\rho$ weight is
itself only an approximate path-count normaliser. The
distinction between the two cases:

- **Without $\rho$:** the calibration error of ~15% is small and
  data-driven — it reflects the residual gap between the
  moment-scan estimate of $b_{\text{eff}}$ and the empirical
  $b'$. The direction of the residual error is uncontrolled
  but bounded.
- **With $\rho$:** the prior shift is order-unity (factor 2.6×
  per child hop on simplewiki) and in a *specifically
  identifiable* direction — toward "child paths informative" —
  that the data refutes.

So both compositions have a prior-vs-data mismatch, but $\rho$'s
mismatch is *much larger* and *systematically miscalibrated*,
while the no-$\rho$ mismatch is *small* and *unconstrained but
bounded*.

**Theorem 2.3 bound direction.** Multiplying $b_{\text{eff}}$
by $\rho < 1$ deflates $b_{\text{eff}} \cdot D$, which
*increases* the convergence ratio $r$ from $r_0 = b'/(b_{\text{eff}} \cdot D)$
to $r_\rho = r_0 / \rho > r_0$. The Theorem 2.3 bound $r/(1-r)$
loosens correspondingly. So the prior shift not only contradicts
the data but also degrades the theoretical bound.

**Restatement of Conjecture 3.6 in Bayesian language.** Under
homogeneity (Definition 0.6), the routing correction $\rho$
encodes a directional prior shift toward "child paths
informative" with order-unity magnitude, while empirical TLI
measurements refute this shift. Dropping $\rho$ replaces this
miscalibrated prior with the only-approximately-correct (but
small-error) path-count-normaliser prior, and tightens the
Theorem 2.3 bound in the process. Outside homogeneity, $\rho$
is one possible heuristic compensation among many — but the
principled fix is topical scoping at ingest, not a per-step
weighting change.

**Task #14 design implication.** Beyond measuring TLI under both
compositions, the experiment should also report *per-pair*
$d_{\text{wPow}}$ values. A finding of "aggregate TLI passes
under both, per-pair distributions are indistinguishable" is the
strongest possible confirmation that $\rho$ was operationally
benign but theoretically obscuring. A finding of "per-pair
values differ materially even though aggregate passes" would
indicate $\rho$ is encoding *some* real structure (not
necessarily what it claims) and a deeper investigation is
warranted.

## 4. Empirical status

For each conjecture, we summarise the empirical evidence
recorded in the design note. See `TREE_LIKENESS_INDEX.md` §4
for full details.

### 4.1 Conjecture 3.1 (main convergence)

- Aggregate TLI: ~0.02% on simplewiki rooted at Physics, $n = 20$
  pairs, budget $B = 15$, child-step costs $cc = 100$ vs $cc = 5$
  (design note §4.1).
- Per-pair TLI: ~0.007% worst case across all 20 pairs (design
  note §4.2).
- Convergence ratio: $r \approx 0.157$ with topical calibration
  ($b_{\text{eff}} = 9.59$, $D = 7.34$, $b' \approx 11$).
- Theorem 2.3 contribution-sum bound:
  $r/(1 - r) \approx 0.186$, i.e. ~18.6%. Empirical TLI: ~0.02%.
  Gap factor ~1000×; see §5.1 below.

### 4.2 Conjecture 3.2 (b·D > b' is operative — sufficiency)

The robustness band documented in design note §5.5 shows three
calibration regimes (global with routing, topical with routing,
topical without routing) all yielding TLI < 0.1% despite
producing $b_{\text{eff}} \cdot D$ values that range from 27 to
1659. The common factor is all three exceed $b' \approx 11$.
This supports *sufficiency* but does not establish *necessity*.

Note on the "1659" figure: this is the global calibration's
$b_{\text{eff}} \cdot D = 589 \cdot 0.384 \cdot D = 226 \cdot D \approx 1659$,
where the routing correction $\rho = 0.384$ is folded in. The
original design note pre-correction had this as $b_{\text{eff}}
\cdot D \approx 9933$ using the early-formula $b$, and the
relationship between these calibrations is documented in design
note §2.0's notation table.

### 4.3 Conjecture 3.3 ($\psi$ sharpening)

**Mixed evidence.** Recomputed first-order Taylor analysis at
simplewiki parameters (§5.1, verified table) shows the $r^1$
coefficient is ~0.048 — small (5% of the worst-case bound) but
not zero. The predicted first-order TLI is ~0.44%, reducing the
naive contribution-sum bound (18.6%) by ~40× through partial
num/denom cancellation. The empirical TLI of 0.02% is still
~22× below this — consistent with $r^2$-like scaling but not
purely $r^2$ either (which would predict ~0.025%, off by
another factor of two from observed).

Honest status: the cancellation is real and substantial, but
not asymptotically clean at $r = 0.157$. Higher-order Taylor
terms or budget-truncation effects must account for the
remaining gap. Pending: either an analytic expansion of
$d_{\text{wPow}}(B, cc)$ to second order in $r$, or Monte Carlo
evaluation at varying $r$ to fit $\alpha$ empirically.

### 4.4 Conjecture 3.4 (topical homogeneity)

- `Category:Articles` $b_{\text{eff}} = 9.59$ (79,797 reachable
  nodes).
- `Category:Physics` $b_{\text{eff}} = 9.51$ (79,199 reachable
  nodes).
- Agreement: ~1% (design note §4.5).
- Global (not topical) $b_{\text{eff}} = 589$, a 60× discrepancy
  — direct evidence that homogeneity fails globally.

### 4.5 Conjecture 3.5 (symmetric DAG falsification)

Pending: synthetic-graph experiment (task #10). The Cohen-Havlin
theorem (§1.3) implies that scale-free graphs in the
$2 < \gamma < 3$ regime are *not* asymptotically symmetric, so a
falsification requires explicit construction.

### 4.6 Conjecture 3.6 (routing-correction redundancy)

Pending: design note task #14. The kernel template's b_eff
formula update is staged locally; the experiment is to compare
TLI under `BranchRatio = b_eff` (no routing) vs
`BranchRatio = b_eff · ρ` (with routing) on simplewiki topical
calibration, and verify both pass the 0.1% TLI threshold.

## 5. Open theoretical questions

### 5.1 Tightness of the convergence bound

Theorem 2.3 establishes
$\sum_{M \ge 1} S_M / S_0 \le r/(1-r)$ and similarly for $W$ —
i.e. a bound on *individual* contribution-sum drifts. On
simplewiki this gives $r/(1 - r) \approx 0.186$, while
empirically TLI is ~0.02% — a factor of ~1000 gap.

Since $d_{\text{wPow}} = (N/W)^{-1/n}$ is a *ratio*, the first-
order TLI estimate is
$$\mathrm{TLI} \approx \frac{1}{n}
\bigl|\Delta\log N - \Delta\log W\bigr|$$
where $\Delta\log N \approx \sum_{k \ge 1} r^k L_k / L_0$ and
$\Delta\log W \approx \sum_{k \ge 1} r^k R_k / R_0$.

**Numerical check at simplewiki parameters.** For $B = 15$,
$cc = 5$, $n = 2$, $d_{\min} = 4$:

| Level $M$ | $N$ range | $L_M = \sum_N (N+M+1)^{-2}$ | $L_M / L_0$ | $R_M$ | $R_M / R_0$ | $L_M/L_0 - R_M/R_0$ |
|---|---|---|---|---|---|---|
| 0 | $[4, 15]$ | $\approx 0.1607$ | 1.000 | 12 | 1.000 | 0 |
| 1 | $[4, 10]$ | $\approx 0.1014$ | 0.631 | 7 | 0.583 | **+0.048** |
| 2 | $[4, 5]$  | $\approx 0.0360$ | 0.224 | 2 | 0.167 | +0.057 |
| 3 | $\varnothing$ | 0 | 0 | 0 | 0 | 0 |

(The $M = 3$ level requires $N \le B - 3 \cdot cc = 0$, which
is below $d_{\min} = 4$, so the range is empty.)

**Partial first-order cancellation.** The $r^1$ coefficient of
$\alpha_S - \alpha_W$ is the $M=1$ difference, $\approx 0.048$.
This is small but *non-zero*: a 47% reduction from the
worst-case contribution-sum bound (where the coefficient would
be 1), but not the full cancellation that pure $r^2$ scaling
would require.

**Predicted first-order TLI on simplewiki:**
$$\mathrm{TLI}_{\text{1st-order}} \approx \frac{1}{n} \left|
r \cdot 0.048 + r^2 \cdot 0.057 \right|
\approx \frac{1}{2} \left| 0.157 \cdot 0.048 + 0.0246 \cdot 0.057 \right|
\approx \frac{1}{2} \cdot 0.00889
\approx 0.44\%$$

So:
- Contribution-sum bound (Theorem 2.3): $r/(1-r) \approx 18.6\%$
- First-order with partial cancellation: $\approx 0.44\%$
- Empirical TLI: $\approx 0.02\%$
- **Remaining gap: ~22× (better than the original ~1000× but not negligible)**

**Status of Conjecture 3.3.** *Mixed evidence.* The $r^1$
coefficient is small (~5% of the worst-case), consistent with
substantial-but-incomplete first-order cancellation. This is
*partially consistent* with $\psi(r) \sim r^2/(1-r)$ scaling
(which predicts $\approx 0.025\%$ from this $r$) but the
predicted ~0.44% sits between the $r^1$ and $r^2$ scaling
regimes. The remaining ~22× gap to empirical likely involves:

- **Budget-truncated higher-$M$ levels.** The $L_3, R_3$ being
  zero (rather than continuing geometrically) eats some
  predicted contribution.
- **Higher-order Taylor terms.** Beyond first order in $r$, the
  ratio $N/W$ has corrections from $(1+\alpha_S)(1+\alpha_W)^{-1}$
  expansion that may further cancel.
- **Length-factor structure.** The $(h+1)^{-n}$ factor for $n=2$
  more aggressively suppresses longer paths than the geometric
  bound assumes.

**Open question.** What is the actual scaling of $\psi(r)$ as a
function of $r$ and the budget $B$? Resolving this requires
either (i) a careful Taylor expansion accounting for the budget-
truncated $L_M, R_M$ sums, or (ii) Monte Carlo evaluation at
varying $r$ to fit the exponent empirically. Conjecture 3.3 as
stated is *consistent with* the empirical gap but not yet
derived from theory.

### 5.2 Quantitative homogeneity ↔ calibration error

Conjecture 3.4 asserts that topical scoping is "sufficient for
homogeneity in practice". Made precise: how much deviation from
exact homogeneity can be tolerated before the convergence bound
of Theorem 2.3 fails to apply? Specifically, if the path-count
factorisation in Lemma 2.1 holds with multiplicative error
$(1 + \delta)$, what is the corresponding error in the TLI
bound?

A quantitative answer here would convert Conjecture 3.4 from
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

The convergence ratio $r = b'/(b_{\text{eff}} \cdot D)$ has a
plausible but unconfirmed spectral interpretation:

- $D$ is the average degree, related to the largest eigenvalue
  $\lambda_1$ of the adjacency matrix.
- $b_{\text{eff}}$ involves second-moment ratios, related to the
  variance of the degree distribution.
- $b'$ is a path-count growth rate, related to higher-order
  expansion properties.

**Sharpening observation.** The connection cannot be a direct
equivalence between $r$ and the spectral gap. Counter-example:
a $D$-regular *directed* graph (every node has $D$ children and
$D$ parents, $D \ge 2$) has a *large* spectral gap (the
adjacency matrix is dominated by $\lambda_1 = D$ with a
substantial gap to $\lambda_2$), yet has $b_{\text{eff}} = 1$
(no directional asymmetry) and $b' \approx D$ (every child hop
multiplies paths by ~$D$), giving $r = D/(1 \cdot D) = 1$. So
the convergence fails for a graph with excellent spectral
expansion.

This rules out "$r$ = function of spectral gap" but leaves open:
$r$ may be related to a *directional* spectral quantity (e.g.
the singular value spectrum of the parent/child adjacency
asymmetry), not the symmetric spectrum.

**Open question.** Is there a relationship between $r$ and the
spectral gap $\lambda_1 - \lambda_2$, or the second-largest
eigenvalue magnitude? A graph with a large spectral gap is an
expander — fast-mixing, with quickly-growing neighbourhood
sizes. Is rapid path-count growth (large $b'$) equivalent to
poor spectral gap, and is this why ultra-small-world graphs
(Cohen-Havlin) defeat TLI even under topical scoping?

### 5.6 Convergence robustness under $\rho$ miscalibration

This sub-section asks two related but distinct questions:

(i) **When does $\rho$ break convergence outright?** i.e. when
does $\rho \le r$ so that the routing-correction-adjusted
inequality $b_{\text{eff}} \cdot \rho \cdot D > b'$ fails.

(ii) **When should we drop $\rho$ even if it doesn't break
convergence?** This is the production question — under
homogeneity (Definition 0.6) the principled answer is
*always* (per Conjecture 3.6), but the magnitude of the
miscalibration depends on graph regime.

The two questions have different answers across regimes, so the
sub-section addresses both. Notation in this section:
$K_p := \mathbb{E}[d_p]$ is the mean parent in-degree.

**The tree-like limit.** For an idealised tree-like graph
(every node has exactly one parent, $D$ children):

- $b' \approx D$ (each child hop multiplies paths by ~$D$)
- $b_{\text{eff}} = D \cdot \mathbb{E}[d_p] / \mathbb{E}[d_c] \cdot \ldots$
  — under "single parent, $D$ children" this evaluates to
  $b_{\text{eff}} = D / K_p$ where $K_p = 1$, so $b_{\text{eff}} = D$
- $r = b' / (b_{\text{eff}} \cdot D) = D / D^2 = 1/D$

So $r \to 1/D$ in the tree limit, and the buffer condition for
$\rho$ becomes $\rho > 1/D$ — easily satisfied for any moderate
$D$ (simplewiki has $D = 7.3$, $1/D = 0.137$, observed
$\rho = 0.384$ clears this).

**Regime sketch.** As $\gamma$ (the child-degree power-law
exponent under homogeneity) decreases from $\infty$ (tame) to
$2$ (degenerate), $b'$ and $b_{\text{eff}}$ both shift but at
different rates:

| Regime | $\gamma$ | $b'$ scaling | $b_{\text{eff}}$ scaling | $r$ behaviour | Buffer for $\rho$ |
|---|---|---|---|---|---|
| Tree-like / tame | $> 3$ | $\approx D$ | $\approx D / K_p$ | $\approx K_p / D$ | wide |
| Wikipedia topical | $\sim 3$ (TBD, task #15) | $b' \approx 1.5 \cdot D$ (empirical) | $\approx 1.3 \cdot D$ (empirical) | $\approx 0.15$ | moderate ($\rho = 0.384$ is ~2.5× buffer) |
| Wikipedia global (not homogeneous) | $\approx 2.4$ globally | hub-mediated, unstable | inflated by hubs | n/a — Definition 0.6 violated | n/a |
| Ultra-small-world (homogeneous, $2 < \gamma < 3$) | $2 < \gamma < 3$ | diverges with $N$ (hub-mediated) | diverges with $N$ (size-biased) | race; can $\to 1$ | shrinks with $N$ |
| Degenerate | $\le 2$ | diverges fastest | diverges | $\to 1$ | none |

(Simplewiki numerics: $b' \approx 11$, $D \approx 7.34$, so
$b'/D \approx 1.5$, not the rough "~10" I had in an earlier
draft.)

**Hand-waving heuristic.** In the ultra-small-world regime, hubs
dominate path counts on both sides of the convergence
inequality, but unequally. $b'$ grows faster than
$b_{\text{eff}}$ because the path count is mediated by hub
traversal (path count grows exponentially in the number of
hub-mediating hops), while $b_{\text{eff}}$ is a size-biased
moment ratio (polynomial in degree statistics). The race
favours $b'$, pushing $r$ upward.

Below some critical $\gamma_*$ (somewhere between $2$ and $3$,
depending on $N$ and budget $B$), the inequality
$b_{\text{eff}} \cdot D > b'$ fails *even before* considering
$\rho$, and the property cannot hold at all. Note that hub
divergence is bounded for fixed budget $B$ (only paths within
$B$ count), so this is an $N \to \infty$ statement; for finite
$N$ and modest $B$ the divergence is finite.

**The empirical Wikipedia case.** Global Wikipedia has
$\gamma \approx 2.41$, which would place it in the ultra-small-
world danger zone *if* homogeneous — but the global graph is
*not* homogeneous (Definition 0.6 violated, see design note
§4.5). The topical core likely has $\gamma > 3$ (less hub-driven
after admin removal), placing it in the tree-like regime where
the buffer is wide. Task #15 (random-pair BFS distance
measurement) will partially confirm this by locating the topical
core in the $\{tree-like, small-world, ultra-small\}$
classification.

**Open question.** What is $\gamma_*$? For a homogeneous graph
with degree exponent $\gamma$, finite size $N$, and finite
budget $B$, derive the $(b', b_{\text{eff}}, D)$ scaling and
identify the $\gamma$ at which $r \to 1$ (convergence breaks
outright) and the $\gamma$ at which $\rho > r$ becomes
restrictive (the buffer for any specific $\rho$ value disappears).
This would characterise the *region of applicability* of TLI as
a property of graph-distribution-tail behaviour, sharpening
Conjecture 3.6 into a quantitative statement about which graph
regimes can safely drop the routing correction without
re-checking.

### 5.7 Weighting as expected-average over admissible paths

The weight formula
$w(p) = D^{-N(p)} \cdot (b_{\text{eff}} \cdot D)^{-M(p)}$
isn't an arbitrary parameter choice — it's the unique exponential
weighting that gives $d_{\text{wPow}}$ the interpretation of an
**expected average over paths within the length budget**.

**The intuition.** When we explore a node, we expect $D$ new
parent-direction paths and $b_{\text{eff}} \cdot D$ effective
new child-direction paths on average per step. If we want each
"explored path" to contribute its share to the average (not be
double-counted by virtue of having many siblings), each parent
hop should be weighted by $1/D$ and each child hop by
$1/(b_{\text{eff}} \cdot D)$. The product
$D^{-N} (b_{\text{eff}} \cdot D)^{-M}$ then reflects the
*expected number of paths arriving at the given $(N, M)$ shape* —
and weighting by its reciprocal makes the sum effectively an
expectation rather than a count.

**Formal consequence (under exact calibration $b_{\text{eff}} \cdot D = b'$).**
Under Lemma 2.1, the number of paths at level $(N, M)$ is
$\approx D^N (b')^M$. Multiplying by the weight gives
$D^N (b')^M \cdot D^{-N} (b')^{-M} = 1$, so each $(N, M)$
shape contributes a total weight $\approx 1$ to the metric sum.
The double sum
$\sum_{N, M} 1 \cdot (h+1)^{-n} = \sum_{h \ge h_{\min}} (\text{shape count at } h) \cdot (h+1)^{-n}$
makes $d_{\text{wPow}}$ the *expected* $(h+1)^{-n}$ over the
distribution that puts equal probability on each accessible
$(N, M)$ shape.

In short: the metric is approximately the *expected
short-distance metric over all paths within the length budget*,
where the implicit probability distribution treats each
path-shape class as a priori equally likely (and within-shape
paths are uniform). Per §3.6.1, this is the maximum-entropy
prior over shape classes.

**Why weighting is *necessary* on effectively-infinite graphs.**
On graphs too large to fully enumerate, the unweighted average
of $(h+1)^{-n}$ over paths would be dominated by the exponential
profusion of deep paths, regardless of their information content.
Concretely on an effectively-infinite tree:

- Number of paths at depth $h$: $\sim D^h$ (exponential in $h$).
- Unweighted "average" of $(h+1)^{-n}$: dominated by largest $h$,
  collapses toward $0$ as the graph extends.
- Weighted by $D^{-h}$: per-depth contribution $(h+1)^{-n}$,
  total $\sum_h (h+1)^{-n} = \zeta(n) - $ (finite head correction)
  — well-defined.

The weight $D^{-N} (b_{\text{eff}} \cdot D)^{-M}$ is *the unique
exponential weighting* (up to constants) that cancels the
exponential growth and leaves polynomial-decay length factor in
control. Alternative weightings either diverge or collapse.

**Alternative graph regimes.** Some graphs have sub-exponential
path-count growth (e.g. bottleneck topologies, or trees where
branching decreases with depth). On these, the unweighted
average is well-defined and the exponential weighting is
over-correction. These graphs are rare in practice; most
natural graphs of interest (Wikipedia categories, citation
networks, dependency graphs, scale-free in general) have
exponential path-count growth, where the weighting is
*structurally required* for the metric to be defined.

### 5.8 Convergence on effectively-infinite graphs

We distinguish three categories of graphs for the purposes of
TLI applicability:

| Category | Example | Computational reach |
|---|---|---|
| **Materially finite, fully reachable** | Simplewiki topical (~80k nodes) | All paths within reasonable $B$ enumerable |
| **Effectively infinite** | Enwiki (10M+ pages), boolean-equivalent expression graphs, code-refactoring DAGs | Finite but you never reach the bottom; bounded neighbourhood is all you ever observe |
| **Materially infinite** | Pure mathematical infinite trees | Doesn't exist as data; only in formal proofs |

**Most of our theory targets the middle category.** The
convergence condition $b_{\text{eff}} \cdot D > b'$ (Theorem 2.3)
is what makes the metric well-defined on effectively-infinite
graphs:

- **Without convergence:** the metric value depends on where you
  truncate. Increasing $B$ keeps changing the answer, so there
  is no principled budget at which to certify.
- **With convergence:** the metric *saturates* to a $B$-independent
  value at some "natural depth" of the graph. Increasing $B$
  beyond that adds path-count growth, but the weighting crushes
  the new contributions. There exists a finite $B^*$ such that
  $d_{\text{wPow}}(B) \approx d_{\text{wPow}}(B^*)$ for all
  $B \ge B^*$ within practical precision.

This is the actual practical promise: the convergence condition
isn't about mathematical infinity. It's about graphs large enough
you'll never see the bottom but still need a stable answer.

**Algorithmically-generated graphs as a test case.** Boolean-
equivalent expression graphs are a clean example of an
effectively-infinite, lazy-generated graph: nodes are expressions
equivalent (under some chosen equivalence) to a fixed formula,
edges are single applications of a transformation rule (De
Morgan, distributivity, etc.). Any non-trivial starting
expression has more boolean-equivalent forms than can ever be
materialised, so you only ever explore a bounded neighbourhood
on demand. The kernel can be pointed at such a generator
without modification — the convergence condition is what makes
the resulting metric well-defined.

We note this as a natural test case for Conjecture 3.5
(symmetric DAG falsification): boolean transformations are
near-bidirectional, so $b_{\text{eff}}$ on such graphs should be
close to $1$, putting them firmly in the "convergence fails"
regime. A direct measurement would either confirm or refute
Conjecture 3.5 cleanly, without Wikipedia-specific topology
surprises.

(Detailed exploration deferred — this section names the
direction; implementation and experiments remain future work.)

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
