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
$$
w(p) = D^{-N(p)} \cdot (b_{\text{eff}} \cdot D)^{-M(p)}
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

**Compatibility with the early formula.** The original design
note uses the early $b$ in some prose. The substitution
$b \mapsto b_{\text{eff}}$ changes the numerical value of
$b \cdot D$ but not the *form* of the weight formula. All
theorems and conjectures below are stated in terms of
$b_{\text{eff}}$.

### 0.4 Path-count growth

For a node $v$ reachable to $r$ within budget $B$, define
$$
\#\text{paths}(v; N, M; B) :=
\bigl|\{p : v \to r,\ p \text{ has } N \text{ parent hops}
              \text{ and } M \text{ child hops, cost} \le B\}\bigr|
$$
At any fixed budget $B$ and child-step cost $cc > 0$, the maximum
admissible $M$ is finite ($\lfloor B/cc \rfloor$), so a literal
$M \to \infty$ limit does not exist. We instead define $b'$ as
the **asymptotic per-child-hop path-count growth rate** under
the homogeneity assumption: there exists a constant $b' \ge 1$
and a node-dependent constant $C(v) > 0$ such that, for $M$ in
the range admissible at $(B, cc)$,
$$
\mathbb{E}_{v \sim Q}\bigl[\#\text{paths}(v; \cdot, M; B)\bigr]
\approx C \cdot (b')^M
$$
where $\#\text{paths}(v; \cdot, M; B)$ denotes summation over $N$
and the approximation holds up to a multiplicative
$(1 + o(1))$ factor uniformly in $M$.

This makes $b'$ a property of the *graph* (and query
distribution), not of the specific $(B, cc)$ probe used to
estimate it. Different $(B, cc)$ pairs that admit overlapping
$M$ ranges should yield consistent $b'$ estimates under
homogeneity; the design note's §4.4 measurements at $cc \in
\{100, 10, 5, 3\}$ are such an overlapping family.

**Equivalent operational definition.** Equivalently,
$$
b' = \lim_{cc \to 0^+}
\left(\frac{\text{total } \#\text{paths at } (B, cc)}
           {\text{total } \#\text{paths at } (B, cc = \infty)}\right)^{1/M_{\max}(B, cc)}
$$
where $M_{\max}(B, cc) = \lfloor B/cc \rfloor$. This makes the
$cc \to 0^+$ regime explicit: $b'$ is the per-child-hop multiplier
seen as we admit increasing numbers of child hops within fixed
budget.

In practice (design note §4.4) we estimate $b'$ as the geometric
mean of successive total-path-count ratios at $cc$ transitions
$100 \to 10 \to 5 \to 3$, yielding $b' \approx 11$ on simplewiki
topical core.

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

### 0.6 Statistical homogeneity

A precondition many of the §2 results and §3 conjectures rest on.

**Definition 0.6 (Statistical homogeneity).** A tuple
$(G, Q, B)$ is *statistically homogeneous* if there exist
constants $D, b' \ge 1$ such that, uniformly over $v$ in the
support of the marginal distribution $Q$:

(H1) $\mathbb{E}\bigl[\#\mathrm{children}(v)\bigr] = D$
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
b_{\text{eff},i})$. This is the setting design note §4.5
documents on Wikipedia: the global graph is piecewise
homogeneous (topical regime + administrative regime) but not
globally homogeneous.

## 1. Background: established results we build on

### 1.1 Random-graph distances (Chung–Lu)

**Theorem (Chung & Lu, 2002).** For a random graph drawn from the
given-expected-degree model with weights
$\mathbf{w} = (w_1, \dots, w_n)$ — where edge $(i,j)$ is present
with probability $w_i w_j / \sum_k w_k$ — and average degree
$\langle k \rangle = \mathbb{E}[w] > 1$, the average shortest-path
length between random node pairs satisfies
$$
L \sim \frac{\log n}{\log \tilde{d}}
$$
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
used (to classify regimes in §5.3). A *random* small-world
(Erdős-Rényi or Chung-Lu) has $C \to 0$ but still has the
distance scaling. The Watts-Strogatz definition is included here
for terminology completeness; the operative concept is "graphs
with $L = \Theta(\log n)$", which includes both Watts-Strogatz
and random-graph small-worlds.

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
- Average pair distance: for uniform-random distinct nodes $u, v$,
$$
\mathbb{E}_{u,v}[d(u,v)] \approx 2L - \frac{2D}{D-1} \approx 2 \log_D n
$$
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
paths, so $\#\text{paths}(v; N, M; B) = 0$ for all $M \ge 1$,
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
$$
\#\text{paths}(v; N, M; B) \approx D^N \cdot (b')^M
$$
as $N, M$ range over admissible hop counts within budget $B$,
where the approximation holds up to a multiplicative
$(1 + o(1))$ factor. Consequently,
$$
\mathbb{E}_{v \sim Q}\bigl[\#\text{paths}(v; N, M; B)\bigr] \approx D^N \cdot (b')^M
$$
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
$$
w(p) \cdot \#\text{paths}\bigl(\cdot;\ N(p), M(p);\ B\bigr) \approx 1
$$
for all paths $p$, i.e. the weight formula is precisely a
path-count normaliser.

**Proof.** Direct substitution:
$$
w(p) = D^{-N(p)} (b_{\text{eff}} \cdot D)^{-M(p)}
     = D^{-N(p)} (b')^{-M(p)}
$$
By Lemma 2.1,
$\#\text{paths}(N, M) \approx D^N (b')^M$, so the product is
$1$ up to the $(1+o(1))$ factor. $\square$

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
$$
r := \frac{b'}{b_{\text{eff}} \cdot D} \in [0, 1)
$$
Let
$$
S_M := \sum_{p:\ M(p) = M} w(p) \cdot (h(p)+1)^{-n}, \qquad
W_M := \sum_{p:\ M(p) = M} w(p)
$$
be the M-level *numerator* and *denominator* contribution sums
to $d_{\text{wPow}}^{-n}$. Then both ratios are bounded by the
same geometric series:
$$
\frac{\sum_{M \ge 1} S_M}{S_0} \le \frac{r}{1 - r}, \qquad
\frac{\sum_{M \ge 1} W_M}{W_0} \le \frac{r}{1 - r}
$$

**Proof.** By Lemma 2.1 and Proposition 2.2 the weighted
contribution per path is $\approx (h+1)^{-n}$ for numerator and
$\approx 1$ for denominator. The number of paths at level
$(N, M)$ is $D^N (b')^M$, each carrying weight
$D^{-N} (b_{\text{eff}} \cdot D)^{-M} = D^{-N} (r \cdot D / D)^{-M} \cdot D^{-M}$.
Multiplying:

For the numerator,
$$S_M \approx \sum_N r^M \cdot (N+M+1)^{-n} = r^M \cdot L_M$$
where $L_M = \sum_{N \in [\text{admissible range}]} (N+M+1)^{-n}$.
Since the length factor $(N+M+1)^{-n}$ decreases in both $N$
and $M$, $L_M \le L_0$ for all $M \ge 0$, so
$S_M / S_0 \le r^M$ and the geometric sum gives the bound.

For the denominator,
$$W_M \approx \sum_N r^M = r^M \cdot |\text{admissible } N \text{ range at level } M|$$
The range size shrinks weakly in $M$ (the budget loses capacity
$cc$ per child hop), so $W_M / W_0 \le r^M$ similarly. $\square$

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
$$
\mathrm{TLI}(G, \mu, Q, B) \le \psi\!\left(\frac{b'}{b_{\text{eff}} \cdot D}\right)
$$

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

### 3.3 Sharpening of the bound: $\psi(r) \sim r^2 / (1 - r)$

**Conjecture 3.3.** The function $\psi$ in Conjecture 3.1
satisfies $\psi(r) \le C \cdot r^2 / (1 - r)$ for some constant
$C$ depending only on $n$ (the power-mean exponent) and
$\mathrm{depth}_{\min}$ (the minimum distance from the support
of $Q$ to the root).

**Motivation.** Theorem 2.3 bounds the *individual*
contribution-sum drifts (numerator and denominator) by $r/(1-r)$.
But $d_{\text{wPow}}$ is the ratio of these sums; if both drift
by the same amount, the ratio drift is second-order in $r$. The
empirical gap (theorem bound $r/(1-r) \approx 18.6\%$ vs
empirical TLI ~$0.02\%$, a factor of ~1000) is consistent with
this conjectured tighter scaling.

**Status.** Open. Even the constant $C$ is unknown; a careful
calculation distinguishing the numerator and denominator
contribution sums via their different length-factor dependences
should resolve it.

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

**Status.** *Tentative.* The drift probe under the alternate
composition has not been re-run on topical calibration. Task #14
in the design note is to settle this empirically by measuring
TLI under both compositions and verifying both remain below the
0.1% certificate threshold.

**Production consequence.** If confirmed, the data-prep recipe
(design note §7.1) and calibration recipe (§7.2) become
unconditionally simpler: drop the routing correction term, use
$b_{\text{eff}}$ directly. If refuted, the routing correction
stays but its theoretical role becomes "calibration-error
band-aid" rather than "principled factor."

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

### 4.3 Conjecture 3.3 ($\psi \sim r^2/(1-r)$ tightening)

Pending: a careful theoretical analysis distinguishing the
numerator and denominator contribution sums via their different
length-factor dependences. The empirical 1000× gap between the
contribution-sum bound and the actual TLI is consistent with the
conjectured $r^2$ scaling.

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

The gap arises because $d_{\text{wPow}} = (N/W)^{-1/n}$ is a
*ratio*. To first order in $r$,
$$
\frac{\mathrm{TLI}}{1} \approx \frac{1}{n}
\bigl|\Delta\log N - \Delta\log W\bigr|
$$
where $\Delta\log N \approx \sum r^k L_k / L_0$ and
$\Delta\log W \approx \sum r^k R_k / R_0$ with $L_M$ the
length-factor sum at level $M$ and $R_M$ the admissible-$N$
range at level $M$. Both $\Delta\log N$ and $\Delta\log W$ are
bounded by $r/(1-r)$ but their *difference* is much smaller —
both are dominated by the same $r/(1-r)$ leading term, and the
correction depends on the *ratio* $L_M/R_M$ vs $L_0/R_0$.

**Open question (Conjecture 3.3).** Is $\psi(r) = O(r^2/(1-r))$
the correct scaling, with the constant depending on $n$,
$d_{\min}$, and the budget $B$? A careful Taylor expansion of
the ratio $N/W$ in $r$ should resolve this. The empirical 1000×
gap is consistent with $r^2$ scaling: $r^2/(1-r) \approx 0.029$
for $r = 0.157$, only ~1.5× off from the empirical 0.02% (and
closer once the constant $C$ is known).

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
