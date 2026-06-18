# Graph Functional Semirings — distributional functionals without the distribution

*Theory note for the WAM-Rust graph-search target. Generalizes the boundary
distribution cache (see `WAM_RUST_BOUNDARY_DISTRIBUTION_*`) from "cache the path-length
histogram" to "propagate the **functionals** of that histogram directly, never forming
the histogram." Forward-looking: it scopes the next increments rather than describing
shipped code. Status of what *is* shipped lives in the boundary spec/plan.*

---

## 1. The thesis

The boundary cache exists to answer aggregate questions about the set of paths from a
node to the root: how many, how long, how short, what is the effective distance. The
natural object behind all of these is the **path-length histogram**

```
H_v[L] = number of paths of length L from v to the root   (bounded/visited semantics)
```

equivalently its generating function `H_v(z) = Σ_L H_v[L] · z^L`. That histogram is a
*high-dimensional* object — one bucket per achievable length, up to the budget. But the
quantities a query actually wants are **cheap functionals** of it:

| functional | in terms of `H` / `H(z)` |
|---|---|
| mass (path count) | `M = Σ_L H[L] = H(1)` |
| raw moments | `m_k = Σ_L L^k · H[L]` (so `m₁ = H'(1)`, …) |
| support floor / ceiling | `min = ` lowest nonzero `L`, `max = ` highest (≤ budget) |
| effective-distance weight | `WeightSum_N = Σ_{L>0} H[L] · L^{-N}`, `d_eff = WeightSum_N^{-1/N}` |

The central observation: **each of these functionals satisfies its own recurrence over
the graph and can be propagated node-to-node — and spliced at a boundary — without ever
materializing `H`.** We compute *with* the distribution without computing the
distribution. The histogram is the implicit object; the scalars are what move.

This is the same move as the **kernel trick** in machine learning (§6): there the
feature map `φ(x)` is never formed because every computation factors through inner
products `⟨φ(x), φ(y)⟩`; here the histogram `H_v` is never formed because every query
factors through functionals that have their own closed propagation law.

## 2. Why the functionals propagate: the composition law

Two structural facts about paths drive everything.

**Concatenation adds lengths ⇒ histograms convolve.** A path `a → c` through a cut node
`b` is a prefix `a → b` followed by a suffix `b → c`, with lengths adding. So

```
H_{a→c} = H_{a→b} ⊛ H_{b→c}        (truncated convolution; the splice)
H_{a→c}(z) = H_{a→b}(z) · H_{b→c}(z)
```

**Alternatives union ⇒ histograms add.** A node with several parents reaches the root
by the union of the per-parent path sets, so `H_v = Σ_{p∈parents(v)} (shift_by_one ∘
H_p)`.

> **Elementarily (the inner-product picture).** The histogram is a *measure* (weighting
> function) `μ` over length `λ`, and every functional is a **linear pairing** — an inner
> product `⟨f, μ⟩ = Σ_λ f(λ)·μ(λ)`. Mass is `⟨1, μ⟩`, the raw moments are `m_k = ⟨λ^k, μ⟩`,
> `WeightSum` is `⟨λ^{-N}, μ⟩`, and a weighted mean is `⟨f, μ⟩ / ⟨1, μ⟩` (divide by the
> total weight). This is the same content as the homomorphism below, without the algebra:
> - **Branching is disjointness made linear.** Paths *partition by their first edge* (a
>   `v→p₁` path and a `v→p₂` path are distinct even if they later reconverge), so `μ_v` is a
>   **disjoint union** `Σ_p S·μ_p`. "Traversal through one parent doesn't affect the other"
>   is exactly that disjointness, and linearity of the pairing distributes over it:
>   `⟨f, μ_v⟩ = Σ_p ⟨f, S·μ_p⟩` — that *is* the `⊕`-additivity.
> - **The `+1` edge lives on the test function.** Shifting the measure is the adjoint shift
>   of `f`: `⟨f, S·μ⟩ = Σ_λ f(λ)μ(λ−1) = ⟨f(·+1), μ⟩`. For `f = λ^k`, `(λ+1)^k` expands
>   binomially → the binomial moment law. The edge is just `f ↦ f(·+1)`, not a separate
>   mechanism.
> - **Why carry raw moments, not the mean.** The numerator `⟨f, μ⟩` and the normalizer
>   `⟨1, μ⟩` are each linear (carry-able through the disjoint union); their *ratio* (the
>   mean) is not — so carry the pairings, divide at the read-out (the normative rule).
> - **Why `WeightSum` is the exception.** It is a fine linear pairing, so branching (`⊕`)
>   is no problem — it adds over parents. It fails only at a **cut**, where the joint
>   measure is a *convolution* and `⟨f, ν∗μ⟩ = Σ_{a,b} f(a+b)ν(a)μ(b)` factors into single-
>   side pairings **iff `f(a+b)` separates**: `(a+b)^k` does (binomial), `(a+b)^{-N}` does
>   not. One inner-product identity explains both why moments splice and why `d_eff` won't.

A functional `F` can therefore be propagated *on its own value* exactly when it respects
both operations — i.e. `F` is a homomorphism from the histogram algebra `(⊛, +)` into
some small algebra `(⊗, ⊕)`. Working them out:

- **mass** `M = H(1)`. Convolution multiplies, union adds:
  `M_{a→c} = M_{a→b} · M_{b→c}`, `M_{v} = Σ_p M_p`. → the **counting semiring** `(+, ×)`.
- **raw-moment jet** `(M, m₁, …, m_K)`. Under *union* every raw moment is linear
  (`m_k(H₁+H₂) = m_k(H₁) + m_k(H₂)`). Under *convolution* the **binomial convolution**
  (because `(a+b)^k` expands binomially):
  ```
  m_k(A ⊛ B) = Σ_{j=0..k} C(k,j) · m_j(A) · m_{k-j}(B)
  ```
  — for `K = 2` this is `M_ac = M_ab·M_bc`, `m₁_ac = m₁_ab·M_bc + M_ab·m₁_bc`,
  `m₂_ac = m₂_ab·M_bc + 2·m₁_ab·m₁_bc + M_ab·m₂_bc`. So `⊗` is a triangular
  (Vandermonde / "jet") multiply, `⊕` is componentwise addition, and the order-`K` jet
  is a `K+1` vector spliced in `O(K²)`. This is a commutative semiring.

> **Precondition: no path-length budget (the dual of the `g_B` caveat).** The binomial
> moment law is exact **only for *untruncated* convolution**. A moment is a scalar that
> has already summed over *all* lengths — unlike a histogram you cannot reach into it and
> zero the buckets past a budget, so a length cap is simply not representable in moment
> space. The moment jet therefore propagates the moments of the **full, unbounded**
> distribution, or nothing exact. This is the mirror image of `g_B`, which needs a
> *fixed* budget; the moment jet needs *no* budget. The condition is automatic in the
> **acyclic ancestor space** (§4): there every path length is finite and bounded by the
> DAG height, so no cap is ever needed and the jet is exact for free. A budget only
> re-enters for *cyclic* graphs (to keep counts finite) — and there the moment shortcut
> breaks and you fall back to the bucket-truncatable histogram. The governing property
> is **direction of convergence**, not merely acyclicity (§4): the budget-free jet is at
> home in the *convergent ancestor direction*, while the *divergent descendant
> direction* wants a budget even in a pure DAG.
- **support interval** `(min, max)`. Convolution adds the extremes, union takes the
  extremes: `min_{ac} = min_ab + min_bc`, `min_v = min_p (1 + min_p)`; symmetrically for
  `max` with `max`. → the **tropical** semirings min-plus and max-plus.

Each functional is thus an instance of one algebraic-path-problem recurrence

```
f(root) = one;     f(v) = ⊕_{p ∈ parents(v)} ( edge ⊗ f(p) )
```

over a semiring `(S, ⊕, ⊗)`. An **n-tuple** of functionals is just the **product
semiring** `S₁ × … × Sₖ`, propagated componentwise — so "arbitrary vectors of scalar
difference equations on the graph" is exactly *pick a product semiring and run the one
recurrence*. Exactness is automatic: `⊗` distributing over `⊕` is the defining semiring
axiom, and that is precisely what makes the splice exact.

> **NORMATIVE RULE — the single most important implementation invariant.** Carry the
> **raw moments** `(M, m₁, m₂, …)`; **never** `(count, mean, variance)`. Means/variances
> add under *concatenation* but **not** under *union* (mixing two distributions is not
> summing two independent variables), so they are not a semiring element — an implementor
> who writes the natural mean/variance accumulator gets a non-semiring that is **silently
> wrong at every node with more than one parent**. Raw moments are linear under union and
> Leibniz under convolution — clean for both. `mean = m₁/M`, `var = m₂/M − mean²` are
> **nonlinear read-outs at the end**, computed once, never carried through the recurrence.

> **Odd moments and cancellation (why raw, again).** Because path lengths are
> non-negative, every *raw* moment is a sum of non-negative terms `Lᵏ·H[L]`, and the
> binomial convolution combines only non-negative quantities (`C(k,j) ≥ 0`, all
> `m ≥ 0`) — so **propagation never cancels, at any order, odd or even.** Cancellation
> appears only in the **central read-out** (`μ₃ = m₃′ − 3μ·m₂′ + 2μ³` subtracts
> comparable terms), and only when the central moment is *small* — a near-symmetric
> node, where `μ₃ ≈ 0` is a small difference of large raw moments. That is benign: you
> only *need* `μ₃` accurate when it is **large** (strong skew), which is the
> non-cancelling regime; when it is small the skew correction it feeds is negligible, so
> the lost precision does not move the reconstruction. And the **CDF gate is the final
> backstop** — a numerically degraded skew/kurtosis reconstruction misses the Kolmogorov
> certificate and is rejected back to the Gaussian or the histogram, exactly as a *model*
> mismatch would, so correctness never rests on the conditioning of the conversion. (A
> shifted-origin moment conditions the read-out better but reintroduces signs and breaks
> the non-negative, linear propagation, so it stays a read-out-time option, not a change
> to what is carried.)

### Raw moments vs cumulants — an open design fork (the additivity question)

Raw moments are *a* valid carry, not the forced one. The higher raw moments are **not
additive** — they combine by the `O(K²)` binomial convolution above — but there is a
representation in which concatenation collapses to plain *addition*: the **cumulants**
`κ_k`. Under an independent concatenation `A ⊛ B` every cumulant adds:

```
κ₁(A⊛B) = κ₁(A) + κ₁(B)      (means add)
κ₂(A⊛B) = κ₂(A) + κ₂(B)      (variances add — i.e. σ adds *in quadrature*)
κ_k(A⊛B) = κ_k(A) + κ_k(B)   (all orders)
```

— an `O(K)` additive splice that also sidesteps the central-read-out cancellation of the
previous note (the large `m₃ ~ μ³` term never forms). So for **concatenation-heavy**
regions (deep, thin ancestor spines) cumulants are strictly the better carry.

**The catch is the *other* operation.** "Variance adds in quadrature" is a property of an
*independent sum* — i.e. of `⊗` / concatenation. It does **not** hold under `⊕` / union: a
node with several parents is a **mixture** of its parents' suffix distributions, not an
independent sum, and a mixture's variance carries the law-of-total-variance *between*-
component term `Σ wᵢ(μᵢ − μ̄)²` on top of the within-component variances. So cumulants are
**not `⊕`-linear** (only `κ₁` is) — they break exactly at the branching, reconvergent,
root-near nodes the cache most targets. Raw moments *are* `⊕`-linear, which is why they
are the safe carry for the branching DAG.

The right framing (the "the higher moments aren't additive, but estimators still combine
them" point): the closed combination rule exists for *both* operations in *both*
representations — raw moments are `⊕`-linear + `⊗`-binomial, cumulants are `⊗`-additive +
`⊕`-nonlinear — so the choice is only **which makes both cheap in the region that
dominates.** Resolution: **raw moments for the branching solve**, converting to cumulants
at the read-out (Edgeworth, §7, is a cumulant expansion anyway); **cumulants for
chain-dominated spines** where `⊗` rules and both the quadrature additivity and the
cancellation-freedom pay. This is also the same chain-vs-branch split that governs
reconstruction (§7): a concatenation-heavy node tends Gaussian (CLT — cheap moment-jet
reconstruction works), a branching-heavy node is a genuine mixture (possibly multimodal —
GMM/histogram territory). `§8.1` freezes the `Elem` type into shipped code, so this trade
should be **named and measured there, not defaulted silently.**

**[named, implemented]** Both carries now exist as concrete types so the choice is explicit, not
a silent default: the raw-moment `MomentJet` (`⊕`-linear `union`, `O(K²)`-binomial `convolve`) and
the `CumulantJet` `(mass, κ₁..κ₄)` (`O(K)`-additive `convolve` — `mass ×`, `κ +` — with `union`
round-tripping through moments since a mixture's cumulants are not additive). They are exact
inverses (`from_moment`/`to_moment`), validated agreeing on both operations by
`cumulant_jet_additive_splice_matches_moment_jet`. So a `⊗`-heavy spine can carry cumulants for the
cheap additive splice and a `⊕`-heavy region carries raw moments, converting at the boundary —
the *measurement* of where that crossover sits (which dominates a given cache region) is the
remaining open step, but the two representations it would choose between are now both shipped and
interchangeable.

**[measured]** The chain-vs-branch split that governs the *carry* also governs the deeper
question — *when is the moment/cumulant summary a faithful stand-in for the histogram, without
ever forming the histogram?* — and `moment_reconstruction_faithful_on_chains_not_on_branches`
measures it. Two synthetic graphs propagate the jet (never the histogram) and the reconstruction
is scored against the independently-computed true histogram:

| node shape | true distribution | `to_normal_repr` CDF error | excess kurtosis (self-diagnostic) |
|---|---|---|---|
| `⊗`-heavy chain (24 length-`{1,2}` diamonds) | shifted binomial → Gaussian (CLT) | **0.0019** | `−0.08` (reads Gaussian) |
| `⊕`-heavy branch (two depths, 6 vs 41) | bimodal mixture | **0.48** | `−2.0` (reads strongly non-Gaussian) |

(The chain's `0.0019` is *below* the Berry–Esseen CLT bound `≈ 0.097` for `n=24` — not a
contradiction: the chain is **symmetric** (`γ₁ = 0`), so the leading `O(n^−½)` Edgeworth term
vanishes and only the `O(n^−1)` correction remains. Berry 1941 / Esseen 1942.)

The answers this pins down (and the honest limits):

- **The moments are exact; only the *reconstruction* approximates.** Mean/variance/skew/kurtosis
  are computed exactly by propagation — `κ₂ = 24·0.25` on the chain confirms cumulant additivity.
  Faithfulness is entirely a property of the reconstruction step (moments → distribution shape).
- **You know it is faithful from *structure*, not the histogram.** A `⊗`-heavy node is a sum of
  many independent stages → Gaussian by the CLT, so the moment-Normal is near-exact (0.002); a
  `⊕`-heavy node is a *mixture*, possibly multimodal, where it fails (0.48). The graph shape is the
  prior — read off the topology, not the histogram.
- **The carried higher moments *self-diagnose*, via their *ratio*.** Not a single threshold but the
  **Pearson `(β₁, β₂)` moment-ratio diagram** (`β₁ = skew²`, `β₂ = excess kurtosis`): every
  distribution obeys the universal bound `β₂ ≥ β₁ − 2`, with equality attained **only** by two-point
  (Bernoulli) distributions (the bimodal branch sits there: skew `0`, excess kurtosis `−2.0 = 0 −
  2`). So the slack `d = β₂ − β₁ + 2 ≥ 0` is a histogram-free **multimodality detector**, and
  `MomentJet::reconstruction_class` reads it: `d < 0.7` → **Multimodal**; mild `|skew|,|kurtosis|` →
  **Gaussian**; otherwise **GramCharlier**. Crucially **some skew is fine** — a skewed binomial
  (`skew 0.31`) classifies `GramCharlier`, reconstructed by the skew/kurtosis corrections — so it
  is the *kurtosis and the ratio*, not skew, that flag genuine non-normality. (The Gram–Charlier A
  expansion is itself a valid non-negative density only over a *bounded* `(γ₁, γ₂)` region, roughly
  `|γ₁| ≲ 2` — Jondeau & Rockinger 2001; outside it the §7 CDF gate rejects the negative-density
  result, so the pre-screen stays safe.)
- **The `0.7` threshold is a conservative heuristic, not the rigorous boundary** — and this is the
  interesting subtlety. `0.7` has **zero false positives** (anything below it is provably near the
  two-mode extremum), but `d` does *not* cleanly separate multimodal from unimodal: the
  **platykurtic** band `0.7 ≲ d ≲ 2` holds *both*. The **uniform** is unimodal/amodal yet has
  `d ≈ 0.80` (excess kurtosis `−1.2`); four moments cannot tell a flat-but-unimodal shape from a
  mild bimodal. `0.7` sits just below the uniform's `0.80`, so platykurtic-unimodals route to
  GramCharlier — *raising it to ≈1.5 would wrongly call the uniform a mixture.* (Published
  unimodality floors — Sharma & Bhandari 2015; Klaassen & van Es 2023, `d ≥ 189/125 ≈ 1.512` — are
  for *strictly* unimodal densities and exclude the flat uniform, which is why we keep `0.7`.) The
  platykurtic band is precisely the **genuinely ambiguous middle** the moments can't resolve — the
  case for the §7 CDF gate or a Monte-Carlo / GMM fit, not a sharper threshold.
- **Multimodal is still *closed-form* — a mixture, not "give up and store the histogram."** The
  flag means *not a single mode*, so reconstruct with a **mixture / GMM** (`HistRepr::DiscGmm` /
  `Mixture`). And the same **Pearson** framework behind the `(β₁,β₂)` diagram is Pearson's 1894
  **method of moments** for a 2-Gaussian mixture: in the *symmetric* case (equal weights, equal
  component variance) the mixture is fit from the *same four moments* in closed form — two modes at
  `μ ± δ` with `δ = σ·(−γ₂/2)^¼`, which is **exact for all component variances `s ≥ 0`** (from
  `γ₂ = −2(δ/σ)⁴`), not just the `s→0` point-mass limit. On the bimodal branch that yields modes at
  exactly `{6, 41}` (the true spikes) from the same moments the single Gaussian botched at error
  `0.48`. So the ladder is Gaussian → Gram–Charlier → **mixture (closed form)** → exact histogram,
  and only the last is non-parametric. **Honest gap:** four moments under-determine a *general*
  mixture — Pearson's *asymmetric* 2-Gaussian case already reduces to a **9th-degree (nonic)
  polynomial** in the separation and typically needs more than four moments; fitting that needs the
  histogram, or a **Monte-Carlo goodness-of-fit / EM** step (sample paths,
  build the empirical distribution, fit/test), which is embarrassingly parallel and a natural
  **GPU** workload (a *future* direction). The §7 reconstruction stays **CDF-gated**
  (histogram-validated) as the final word; `reconstruction_class` is the cheap pre-screen that
  decides *which* parametric family to attempt (single mode vs mixture), not whether to keep a
  closed form at all.
- **Cumulants vs moments is *orthogonal* to all of this.** They carry the same information
  (`κ_k ⟺ m_k`), so the reconstruction is identical; the §3 fork is purely the *splice cost /
  numerical-stability* axis (additive cumulants for `⊗`-heavy spines), not a representation-quality
  axis. Faithfulness is a §7 reconstruction question, the same for both carries.

### The one that does *not* factor: `WeightSum`

**The unifying principle: point-evaluation of the GF.** Treat `H(z) = Σ_L H[L] z^L` as a
formal power series in `R[[z]]` (the *probability* generating function is the normalised
`H(z)/M`). *Evaluation at a point*, and its derivatives, is a **ring homomorphism**
`R[[z]] → R`: `M = H(1)`, `m₁ = H'(1)`, and so on. Convolution is multiplication in
`R[[z]]`, and a ring homomorphism sends products to products — so **every point-evaluation
functional splices multiplicatively.** That is the one structural reason the mass and the
moment jet propagate; they are not separate calculations.

`WeightSum_N = Σ_L H[L]·L^{-N}` is the exception precisely because it is **not** a
point-evaluation of `H` or its derivatives — it is a Mellin / negative-moment functional
(a pairing against `L^{-N}`), so it is not a ring homomorphism. Concretely it is
`⊕`-linear (it adds under union) but **not** `⊗`-multiplicative:
`Σ_{a+b=n} … (a+b)^{-N}` does not separate into `f(prefix)·g(suffix)` because `(a+b)^{-N}`
is not a product of a-only and b-only terms. So there is no scalar concatenation law for
the effective-distance weight. The existing `g_B` pre-weighted basis is the partial fix:
fix the budget and pre-weight, `g_B[a] = Σ_b H_B[b]·(a+b)^{-N}`, giving a *vector indexed
by prefix length* (not a scalar), valid only at that fixed `N` and budget. In kernel-trick
terms (§6) this is a kernel that does **not** factor through a finite feature — you either
carry the whole histogram, or accept the cheap **bracket** of §5 instead.

## 3. The `PathSemiring` framework

**[Implemented — increment 2d.]** With two concrete instances in hand (the moment jet and
the tropical interval) and the cyclic star settled (§4, increment 2a), the trait is now
real, not a sketch:

```rust
pub trait PathSemiring: Copy {
    fn zero() -> Self;            // ⊕-identity: unreachable / no path
    fn one()  -> Self;            // ⊗-identity: the root / empty path
    fn add(self, other: Self) -> Self;   // ⊕ — combine a node's parents
    fn step(self) -> Self;               // ⊗ by one edge (shift_one); keeps `zero` inert
}
// suffix_value::<S>(node, root, parents, memo, on_stack) — the one generic recurrence;
// suffix_moment_jet = suffix_value::<MomentJet>, suffix_interval wraps suffix_value::<Interval>.
```

`MomentJet` and `Interval` both implement it, so the two recurrences collapse to one
generic `suffix_value`. Adding a payload is now: implement four methods. How the review's
three trait gaps were resolved:

- **(1) Star / closure for cycles** — *not* a trait method, deliberately. The closure
  exists only for *closed* payloads: min-plus is closed (its star is the separate
  `min_distance_closure` / `weighted_distance_closure`), while counting/moments diverge on
  a cycle. So `star` is a per-payload free function/method, not a trait method some impls could
  not honour. (`suffix_value` itself is the acyclic recurrence.) The **element** stars are now
  implemented and characterized per §8 increment 1.5: `MomentJet::star` (closed form iff
  `mass < 1`, else `None`) and `Interval::star` (`None` for any positive-length loop, the max-plus
  factor diverging) — each `a* = one ⊕ a⊗a*`, the building block for splicing a condensed SCC.
- **(2) ⊕/⊗ asymmetry** — `step` (⊗ by an edge) is exact only untruncated; a budget would
  break `step`/⊗ but never `add`/⊕. The recurrence never truncates (acyclic, budget-free),
  so it is exact; the asymmetry is documented on the trait and the budgeted case lives in
  the histogram path. The `zero`-inert-under-`step` law (so unreachable parents contribute
  nothing) is enforced by `path_semiring_laws_and_generic_equivalence`.
- **(3) Read-out / decode** — left payload-specific for now (`MomentJet` → mean/variance/
  skew; `Interval` → min/max), since a common typed read-out has no clean shared codomain;
  a generic `project` is deferred until a consumer needs it.

The band selection, shared-memo sweep, eviction, and persistence skeleton remain
payload-agnostic — only the per-node element type changes.

| payload | semiring | cost | answers | exact for |
|---|---|---|---|---|
| histogram | convolution `(⊛, +)` | O(budget) | every linear functional (mass, moments, `WeightSum`, CDF) | everything |
| moment jet `(M,m₁,…,m_K)` | truncated power series | `K+1` scalars (3 for mean/var) | count, mean, variance, skew/kurtosis → CLT/Edgeworth distribution (§7); **needs unbounded length** | mass + first `K` moments |
| interval `(min,max)` | min-plus × max-plus | 2 scalars | shortest + longest; brackets `d_eff` (with `mass`, §5) | both endpoints |
| shortest scalar `min` | min-plus | 1 scalar | shortest distance (A* heuristic / landmark) | shortest only |

## 4. The domain: a node's ancestor space

The recurrence is an **up-propagation** — it walks `parents`, toward the root — so its
support is exactly the **ancestor space (up-closure) of the query node**, not the whole
graph. This is the right and load-bearing domain:

- **Exactness needs only the reachable set to be acyclic**, and the reachable set *is*
  the up-closure. A taxonomic / `is-a` relation is a partial order, so every node's
  ancestor space is a DAG even when the relation reconverges (diamonds). We never needed
  a globally acyclic graph — only acyclic ancestor spaces. *(Scope: this assumes a single
  bottom-up sweep over the **data** relation's parent graph. Under tabled/SLG resolution
  the relevant graph is the subgoal-dependency graph, whose SCCs can be cyclic even over
  acyclic data — verified not the case in this codebase, which has no tabling/SLG machinery
  — so the argument is sound here but should not be transplanted to a tabled evaluator
  without re-checking.)*
- **Ancestor spaces share their root-near core.** Different query nodes' up-closures
  overlap heavily near the root; that shared upper sub-DAG is computed once and spliced
  into many nodes — which is *why* root-near boundary caching pays, and what the
  shared-memo sweep already exploits.
- **Cyclic up-sets are the general-graph case.** For arbitrary-graph distance queries
  the up-closure can contain a cycle; then the clean DAG solve is replaced by the
  **closed-semiring** version (`a* = ⊕ aⁱ` must converge — min-plus on nonnegative
  weights does, counting does not without truncation), which in this codebase is the
  existing **budget + visited-guard** truncation. Exact on poset/taxonomic data;
  truncation-approximate on general graphs. **The DFS+memo recurrences are themselves
  unsound on cycles** — the `on_stack` guard makes a node's value depend on the current
  stack, and that context-dependent value is then memoised (a node first reached inside a
  cycle can be cached as wrongly `None`/unreachable). The closed-semiring solve is
  context-free; for min-plus it is `min_distance_closure` (**[2a, implemented]** a BFS
  fixpoint from the root over the reversed graph, `a* = 0`, O(V+E), cycle-correct).
  `min_distance_closure_is_cycle_correct_where_dfs_poisons` exhibits exactly the cyclic
  case where the DFS memo poisons a node and the closure does not.

### Convergence, not just acyclicity: why unbounded length is meaningful here

The deeper reason the ancestor direction tolerates an unbounded path length — the
precondition the moment jet needs (§2) — is that it **converges to a unique sink, the
root**. Every path funnels inward and *completes* at the root, so the path set is the
canonical, finite, *total* set of the node's derivations ("all the ways `v` is-a … is-a
root"). Completeness is well-defined and length is bounded by the DAG height for free, so
relaxing the budget is natural, not a patch — and the complete statistics *are* the
meaning.

The **descendant direction is the opposite geometry**: paths *diverge* toward many leaves
with **no unique sink**, so "all paths without a length cap" is not a canonical statistic
— it is dominated by long, indirect routes and explodes with fan-out. There a budget is a
**semantic filter** (which paths count as meaningful: the short, direct ones), and it is
needed *even in a pure DAG*. So the precondition for the budget-free moment jet is
properly the **convergent ancestor direction**, of which "acyclic up-closure" is the
taxonomic instance — not acyclicity per se.

| direction / structure | unique sink? | unbounded length is… | payload |
|---|---|---|---|
| ancestor → root (convergent) | yes (root) | finite **and** meaningful | **moment jet exact** (+ histogram) |
| descendant (divergent, even acyclic) | no | finite but indirect-path-dominated | budget + histogram (jet breaks under the cap) |
| cyclic (either way) | — | infinite | budget + histogram (finiteness necessity) |

This is also why the moment-jet idea sits naturally on the existing machinery: the
boundary cache already stores **suffixes toward the root** — the convergent direction — so
budget-free moment propagation rides what exists, while descendant / general-direction
search intrinsically carries the budget and stays on the histogram.

This staging — **convergent ancestor space first, divergent/cyclic closure second** —
orders the implementation (§8).

## 5. A certified bracket on the effective distance from `(min, max, mass)`

Let `W = WeightSum_N = Σ_{L>0} H[L]·L^{-N}` and `M = mass`. The **normalised** effective
distance is the power mean of the path lengths with exponent `−N`,

```
pm = (W / M)^{-1/N} = (E[L^{-N}])^{-1/N}
```

and by the power-mean inequality `min_L ≤ pm ≤ max_L` **always** — so the tropical
interval `(min, max)` brackets the *normalised* metric exactly, for two integers, exact at
the endpoints and sound between.

The **raw** effective distance the kernel reports is the *un-normalised* `d_eff = W^{-1/N}`
(no division by `M`) — i.e. the power mean scaled by the path count:

```
d_eff = M^{-1/N} · pm      ⟹      d_eff ∈ M^{-1/N} · [min, max].
```

So bracketing the raw `d_eff` needs the **mass** component too — which is exactly why the
payload carries `(min, max, mass)`: the two tropical scalars bound the *shape*, the count
sets the *scale*. (`(min, max, mass)` alone, no `m₁`/`m₂` needed for the bracket;
validated in `boundary_cache::tests::interval_and_mass_bracket_d_eff`.) It is the cheap
surrogate for the `WeightSum` functional that §2 showed cannot be carried as a scalar.

### 5a. Composite caret distance — a between-nodes *upper* bound

The to-root distance cache (increment 2) answers "how far is `v` from the root". The same
two cached scalars give an **upper bound** on the distance **between two nodes**. A path
`u → v` can always go **up to a shared ancestor (a *bridge*) and back down** — a `∧`/caret
path `u ↑ B ↓ v` of length `d(u→B) + d(v→B)`. The **root is a universal bridge**, so

```
d_undirected(u, v)   ≤   d(u→root) + d(v→root)        (composite caret, root bridge)
```

is free from the cache (`caret_distance_upper`) — it is the length of a real `∧`-path. A
**lower bridge** (a common ancestor nearer `u, v`, ultimately the **lowest common
ancestor**) gives a *tighter* caret; `caret_distance_lca` computes the exact `∧`-distance
`min_B (d(u→B) + d(v→B))` by a joint upward BFS. The caret **equals** the true shortest-path
*length* on a *tree* (the cophenetic / tree distance — a scalar functional, not a route; see
§5b) and is a **certified upper bound** on a DAG (a non-ancestor route can be shorter).

**No matching lower bound from the cache (the correction).** It is tempting to add
`|d(u→root) − d(v→root)| ≤ d(u,v)` as a lower bound (the ALT landmark heuristic), but
**that is false in general.** The reverse triangle inequality needs a *metric* (symmetric
distances); the cache stores the *directed* up-distance, and the caret distance is
*undirected*. On a DAG with a shortcut the undirected distance can be far smaller than
`|d_u − d_v|` — e.g. a chain `4→3→2→1→0` (so `d(4→root)=4`) plus `5→0` and an edge `5—4`
gives `d(5→root)=1` while `4,5` are *adjacent* (`d=1`), yet `|4−1| = 3 > 1`
(`alt_lower_bound_is_directed_only`). The symmetric bound holds **only on a tree** (there
the directed up-distance *is* the undirected metric). The valid cache lower bound is the
*directed* one — `max(0, d(u→root) − d(v→root)) ≤ d(u→v)`, because `u→v→root` is a walk to
root (`directed_distance_lower`) — the admissible A* heuristic for the **directed** query,
a bound on a *different* quantity than the undirected caret. So on a DAG there is a
certified upper bound (undirected caret) and a directed lower bound, but **not** a single
two-sided bracket; the bracket is a tree-only special case.

Validated by `caret_distance_on_a_tree_equals_true_distance`,
`caret_distance_on_a_dag_is_an_upper_bound`, and `alt_lower_bound_is_directed_only`. This
is the natural *between-nodes* use of the to-root cache — the general companion to
increment 2's *to-root* query.

### 5b. Two caret measures: auto-LCA (shortest-path length) vs designated bridge

There are **two** ways to pick the bridge, and they answer different questions:

- **Auto-LCA** — `caret_distance_lca` minimises over bridges, so the bridge is *implicit*
  (the lowest common ancestor). But minimising over `B` means it **collapses to the
  (undirected) shortest-path *length*** (on a tree, exactly the tree distance). So it carries
  **no information beyond distance** — you don't pick a bridge, but you also learn nothing the
  shortest distance wouldn't tell you. `caret_distance_budgeted(u, v, budget)` is this measure
  *scoped*: the budget admits only bridges within a radius (the support upper bound is the
  natural value), but within scope it is still the auto-minimising shortest distance.
- **Designated bridge** — `caret_through_bridge(u, v, B) = d(u→B) + d(v→B)` *fixes* the
  bridge to a chosen reference node `B` (defined when `B` is an ancestor of both). It
  measures relatedness **as seen from a chosen level** — "through the physics category", or
  through a node higher up. You pick the bridge, and in exchange it keeps information the
  shortest path discards.

> **A caret distance is a *functional*, not a path (value vs. route).** Saying the auto-LCA
> caret "is the shortest path" is loose in an important way: it is the shortest-path
> **length**, a scalar **functional** of the path-length distribution — its support floor
> `min{L : H_{u↕v}[L] > 0}`, the min-plus / tropical read-out (exactly the `min` of the
> interval payload). It does **not** single out a route. The same minimal value is realized
> by a whole **sub-distribution of shortest `∧`-paths** — several shortest `u→B` paths ×
> several shortest `v→B` paths, summed over any *tying* bridges `B`. Their **multiplicity**
> (the *number* of shortest paths) is a **different** functional of the same distribution —
> the histogram's count at the floor, `H[floor]` — that the distance value says nothing
> about. This is the note's central thesis on the distance side: the shortest distance, the
> shortest-path count, the mean length, the moments are all functionals of *one* path-length
> distribution; the caret reads the **floor**, nothing more. (Carrying *(distance, #shortest
> paths)* together is the min-plus semiring **with multiplicities** — a clean `PathSemiring`
> instance not yet built: the bare interval gives the floor but not its count, and the moment
> jet gives mass/moments but not the count *at* the floor.)

> **Fixing the bridge does not fix the length — two nested collapses, neither uniform.** The
> caret `min_B (d(u→B) + d(v→B))` is a `min` of a `min`, and it is worth being explicit that
> *neither* collapses the path population to one length. (i) **Inner** (`d(u→B)`): even for a
> *single* bridge `B`, parent **branching** gives routes `u → B` of *different* lengths, so
> `H_{u→B}[L] = #{u→B paths of length L}` has support across many `L`; `d(u→B)` reads only the
> **floor** of that multi-length distribution. (Example: `u`'s parents `a, b` with `a→B` 1 hop
> and `b→…→B` 2 hops give `u→B` lengths `{2, 3}` to the *same* bridge; the min reads `2`.)
> (ii) **Outer** (`min_B`): selects *which* bridge, nothing about length. So the `∧`-path
> lengths through `B` form a genuinely multi-length set
> `{a + c : a ∈ support(H_{u→B}), c ∈ support(H_{v→B})}`, and the caret is **one scalar
> summarising that whole population** — the floor of it. This is the fuller version of the
> "value vs. route" point above (which only noted ties *at* the floor): the population to a
> fixed bridge already spans lengths, and the `min` discards everything but its floor — which
> is exactly why the **soft** `d_eff` read (which *weights* those different-length routes
> instead of dropping them) carries strictly more than the bare shortest.

> **Hard vs. soft shortest distance — the admitted paths need not be one length.** The
> floor functional above is the **hard** minimum (tropical / min-plus): *only* floor-length
> paths get nonzero weight, so the value is `min L` exactly and the multiplicity `H[floor]`
> is a separate read-out. But the framework's *original* distance functional is **soft** —
> the effective distance `d_eff = (Σ_L H[L]·L^{-N})^{-1/N}` (§5) — a weighted sum that admits
> paths of **all** lengths, weighting shorter ones more (`L^{-N}`). For finite `N` it is a
> power mean in `[floor, ceiling]` (the §5 bracket), reaching the floor only as `N → ∞`. So
> "shortest distance" is genuinely two functionals: the **hard** floor (one length, with a
> separate count) or the **soft** `d_eff` (every length contributes a nonzero weight, short
> ones dominating). The two even meet at the multiplicity — the large-`N` expansion is
> `d_eff ≈ floor · H[floor]^{-1/N}`, so the soft functional's *leading term* is the floor and
> its *first correction* encodes the **number** of shortest paths. A single soft functional
> thus carries both the floor and (asymptotically) the multiplicity; the hard min carries
> only the floor. This is why a caret built from `d_eff` rather than the hard min is, exactly
> as it sounds, a weighted average over `∧`-paths of *different* lengths — not a single route.

**The distance between the two measures.** With the LCA below the chosen bridge `B`, on a
tree `d(u→B) = d(u→LCA) + d(LCA→B)` (and likewise for `v`), so

```
through(B)  −  lca   =   2 · d(LCA → B)
```

— the designated-bridge caret is the **shortest-path caret plus twice the lift of the
reference above the true LCA** (`caret_through_bridge_vs_lca_and_the_gap`). That gap is
exactly the signal the auto-LCA throws away: two topic pairs with the *same* shortest path
get *different* through-`B` distances when their LCAs sit at different depths below `B`. So
`through(B)` decomposes as **intrinsic distance** (the shortest path) + **2 × (how far the
pair's meeting point sits below your reference level)**. Choosing `B` = the physics category
yields "relatedness within physics"; raising `B` yields relatedness at a coarser level — the
multi-level reading the budgeted measure cannot give (it only scopes, never reframes). On a
DAG both are upper-bound approximations, as the caret itself is.

### 5c. Computing the caret — per-query search, landmark precompute, nested cuts

There is a clean computational spectrum, and it mirrors §5b: the **auto-LCA** measure resists
caching, the **designated-bridge** measure is cacheable.

- **Per-query joint search (what is built).** `caret_distance_lca` / `_budgeted` /
  `category_caret_distance` do a joint upward BFS from `u` and `v`, intersect the ancestor
  sets, take the min-sum. The bridge is found *dynamically* — nothing is stored. So there is
  **no per-bridge distribution** at query time (no explosion), but also **no reuse**. This is
  the right tool when the bridge varies per pair (auto-LCA) or for one-off queries.
- **Why the auto-LCA caret cannot be cached like the to-root distance.** To get "precompute
  once, O(1) per pair" you would treat the bridge as a **boundary cut** (nothing above it)
  and enumerate **downward** from it. But the auto-LCA bridge **varies per pair**, so covering
  all pairs needs one downward distribution **per possible bridge** — the all-pairs blow-up.
  That is the precise reason the to-root distance is a precompute and the auto-LCA caret is a
  search.
- **Designated bridges are a handful of landmarks.** Fix the bridge and the blow-up vanishes:
  one downward BFS from `physics` gives `d(·→physics)` for the whole subtree once, and then
  `caret_through_bridge(·, ·, physics)` is O(1) per pair. "A node higher up" is just another
  field (`d(·→natural_sciences)`, …) — `K` levels = `O(K·V)` storage, not `O(V²)`. The
  multi-level bridges **are** the landmarks, and there are only a few. This is the
  boundary-cache *reuse*, finally applied to the caret — and the *computational* reason
  (beyond §5b's informational one) to prefer the designated bridge: it is the cacheable one.

> **Theory grounding (3f, now built).** The designated-bridge landmark scheme is a
> semantically-scoped instance of **2-hop cover / hub labeling** (Cohen, Halperin, Kaplan &
> Zwick 2002; Abraham, Delling, Goldberg & Werneck 2012; *pruned landmark labeling*, Akiba,
> Iwata & Yoshida 2013). Each bridge's field `d(·→B)` is one **column of the all-pairs distance
> matrix in the min-plus (tropical) semiring** `(min, +)`, and the caret read
> `min_B (d(u→B) + d(v→B))` is a **min-plus inner product** of `u`'s and `v`'s label rows — the
> tropical analogue of `Σ_B x_u[B]·x_v[B]`. So "carry the functional, not the distribution" is
> here *literally* the tropical-algebra strategy: the cached label is a min-plus vector, the
> query a min-plus dot product, and no path-length distribution between the pair is ever formed.
> The `O(K·V)` storage is **tight** for an O(1)-lookup landmark scheme (Tretyakov et al. 2011):
> no asymptotically smaller compact representation supports constant-time distance reads. And
> the read is the true LCA caret **only if the optimal common ancestor is one of the chosen
> bridges**; with a sparser set it is a valid **upper bound**, larger by `2·d(LCA→nearest
> bridge)` (§5b) — the standard landmark-distance over-estimate (Storandt 2022). The field
> construction is the **reversed-BFS** identity: child edges are the exact reversal of parent
> edges, so `src→B` up-paths and `B→src` down-paths are in length-preserving bijection and their
> minima coincide (the Contraction-Hierarchies / bidirectional-search argument; holds on any
> directed graph, diamonds included). (`bridge_distance_fields`, `caret_through_bridge_cached`,
> `caret_min_over_cached_bridges`; live `build_caret_landmarks`.)
- **Nested cuts (the convolution refinement).** A bridge is a **cut**, and distributions
  *factor* there: `H_{u→root} = H_{u→B} ⊛ H_{B→root}` (counting) / `d(u→root) = d(u→B) +
  d(B→root)` (min-plus). So one can cache the **suffix above** `B` and treat `B` as a *fresh
  root* for the subtree below — computing each node's below-`B` part once and reusing the
  shared above-`B` suffix across all the levels that nest above it. This is the boundary cache
  generalised to a **hierarchy of cuts** (the route-planning *hub-labeling / highway-
  hierarchies* structure). It pays off for **deep** hierarchies (many nesting levels, large
  shared upper part); for the *few*-level case the flat `K`-landmark precompute above is the
  better cost/complexity point. **Caveat:** the factorisation needs `B` to be a *proper cut*
  (a **dominator** — every `u→root` path crosses `B`): automatic on a **tree**, but on a
  **DAG** (real Wikipedia, with cycles) only at dominator nodes, so maintaining the cut
  property is part of the "complexity" — the same "proper boundary cut" precondition the
  boundary spec already requires. *Deferred — correct, but only worth it at depth.*

> **Two separate questions: which bridge to *pick* vs. is it a *dominator*.** These are
> easily conflated (I did). (1) **Bridge selection** — *which* nodes to designate as bridges
> — is a soft, practical choice: good bridges are **convergence hubs**, the "union-type"
> categories where the hierarchy funnels (`Physics` = the union of mechanics, EM, thermo, …).
> They are **common ancestors of large descendant sets**, so one downward field serves *many*
> pairs (max reuse), and they are meaningful semantic boundaries. The caret through a
> designated bridge needs only that `B` be a **common ancestor** of `u` and `v` — *not* a
> dominator — so it is well-defined (an upper bound) regardless of cross-lineages. (2) Being
> a **dominator** (a *proper cut*: every `u→root` path crosses `B`) is the *stronger*
> condition needed **only** for the exact convolution factorisation of the nested-cut
> optimisation. A convergence hub that is *also* the sole gateway to root is both — but real
> Wikipedia's **cross-listings** (`Mathematical_physics` under both Math and Physics) and
> **cycles** make strict dominators *rare*. Hence: the designated-bridge caret (needs only
> "common ancestor") is **robust** — pick the union hubs and go — while the nested-cut
> factorisation (needs "dominator") is **fragile**, which is the deeper reason it is deferred.

> **Defining "downward convergence" *cheaply* — and *non-circularly*.** Bridge selection (1)
> asks us to score nodes by how much the hierarchy funnels through them. The honest constraint
> is sharper than "make it fast": the score must **not call the very upward distance work the
> bridges exist to amortize**. The natural definition — *descendant-cone size*, `|desc(B)|` —
> needs reachability (a global per-node traversal); and ranking candidate bridges by
> `caret_through_bridge` is the same circularity wearing a hat (it runs `up_distance_to` at
> every candidate). Both are rejected: they spend exactly what hubs were meant to save. What is
> left must be **structural and local**, readable *before* any query.
>
> - **The true signal is a cone-size *step*, not a large cone.** A hub is where the descendant
>   cone drops off sharply as you descend through it: large at `B`, but each child just below
>   carries only a slice. That dropoff *is* the right characterization — and its exact form
>   (cone size at every node) is the reachability we are refusing. So we take its **cheap
>   shadows**, in two rungs.
> - **Rung 1 — local fan-in (free, cycle-robust).** `fanin[B] = #{v : B ∈ parents[v]}`, the
>   in-degree in the child→parent graph. It is the *first derivative* of the dropoff: it counts
>   *how many* cones merge at `B` in one hop without summing their sizes. One pass over the
>   `parents` map we already hold — on the boundary-sweep path it is just `children[B].len()`,
>   already materialized, so the score costs **nothing extra**. It reads no distance, so it
>   cannot be circular, and it is well-defined even with cycles. Its blind spot: *branchiness
>   only* — two giant subtrees merging is `fanin = 2` yet a huge step.
>   (`convergence_fanin`, `hubs_by_fanin`.)
> - **Rung 2 — additive descendant weight (one pass, magnitude-aware).** `w(B) = 1 + Σ_{c ∈
>   children(B)} w(c)` in reverse-topological order recovers the *magnitude* fan-in misses. It
>   **over-counts diamonds** (a descendant on two paths is counted twice) — but that over-count
>   is precisely the price of dodging distinct-set reachability, and it is a single **O(V+E)**
>   sweep, not per-node BFS. The dropoff is then `jump(B) = w(B) − max_c w(c)` — a leaf jumps
>   `1`, a hub merging several heavy subtrees jumps large. Needs a DAG (a reverse-topo order);
>   `None` on a cycle — so rung 1 is the cyclic-graph fallback. (`descendant_weight`,
>   `convergence_jump`.)
> - **Rung 3 — parent reconvergence (the ancestor-side definition).** The two rungs above read
>   the *descendant* cone (down from `B`); the sharpest hub definition reads the *ancestor* cone
>   (up from `B`). The signature: ascending one level from `B` loses **far fewer distinct
>   ancestors than the parent branching factor `b = |parents(B)|` predicts**, because the
>   parents' upward cones *overlap* — the lineage re-merges above. The deficit (expected-from-`b`
>   minus actual) *is* the convergence; `b` large with a big deficit is a hub, `b` large with
>   none is just a fan. The exact deficit is reachability again — and note the additive
>   ancestor-weight is no help, it *is* the disjoint/branching-factor expectation, blind to
>   overlap, so a separate overlap-sensitive probe is required. The cheap one is **local**:
>   overlapping parents first share **grandparents**, so probe each parent's bounded `up_hops`-
>   deep up-cone and measure their overlap. `up_hops = 1` is "do the parents share grandparents"
>   (a 2-hop neighbourhood, no walk to root); `up_hops → ∞` is the exact deficit. `up_hops` is
>   the cost/sensitivity knob, and the depth bound makes it cycle-safe. (`parent_reconvergence`,
>   returning the *duplicate-mass fraction* `overlap/total_mass = o/(Σsᵢ)` in `[0,1]` — **not**
>   a Jaccard `o/(Σsᵢ−o)`; for two identical cones it returns `1/2`, intentionally, to avoid
>   Jaccard's denominator instability near `o ≈ Σsᵢ`.)
> - **Rung 4 — ancestor sketch + small-world lift (height-agnostic, baseline-corrected).** The
>   fixed-`up_hops` probe of rung 3 has a fatal flaw: it only sees a crossover *within* `k`
>   hops, but the **crossover height is unknown and varies per node** — too small misses deep
>   hubs, too large walks to root (the reachability we refuse). Worse, in a **small-world**
>   graph the up-cones cover most of the graph within a few hops, so *raw* overlap (at any `k`,
>   even exact) approaches 1 for **every** pair — it measures "are we in a small world," not "is
>   `B` a funnel." Two fixes, both O(k): (a) **height-agnosticism** — summarize each node's
>   *whole* lineage to root once, as a fixed-size **KMV/MinHash ancestor sketch**
>   `sig(B) = bottom-k( {B} ∪ ⋃_p sig(p) )`, one root→leaf pass; overlap (`sketch_jaccard`) is
>   then read at *any* depth with no knob (error ∝ 1/√k, not a depth cutoff). (The sketch is the
>   bottom-`k`/KMV lineage: MinHash for Jaccard, Broder 1997; the one-pass bottom-`k` Jaccard
>   estimator, Cohen & Kaplan 2007; the `(k−1)/ĥ_k` distinct-count read, Bar-Yossef et al. 2002 /
>   Beyer et al. 2007.) (b) **baseline
>   correction** — a real hub reconverges *more than chance*: against the configuration-model
>   null `E|A∩B| ≈ |A|·|B|/N`, the signal is `lift = observed |A∩B| / E|A∩B|` (`sketch_overlap_
>   lift`), `>1` an excess funnel, `≈1` just small-world background. The sketch yields `|A|`,
>   `|B|` *and* `|A∩B|` from the same reads, so height-agnostic detection and the small-world
>   correction share one precompute. This is the §6 kernel trick literally applied: the ancestor
>   *set* is the never-materialized feature map, the sketch its inner-product handle. (`None` on
>   a cycle — SCC-condense first, since a cycle's nodes share their entire up-cone.)
> - **The min-over-hubs caret is then quantized-LCA.** With hubs *cheaply* pre-selected (by
>   fan-in / jump / reconvergence / lift, **no distances**), `caret_min_over_hubs(u, v, hubs) = minᵦ
>   caret_through_bridge(u, v, B)` picks the hub giving the least distance. The only distance
>   work runs over the *already-chosen small* hub set — bounded by hub count, not by ranking the
>   whole graph — so selection stays free and only the final min-pick costs anything. With
>   **every** node a hub it equals `caret_distance_lca` exactly (the unquantized shortest-path
>   caret); with a sparser hub set it is that caret **quantized up to the nearest hub level**,
>   larger by the gap `2·d(LCA→nearest hub)` of §5b. Tightness (low, dense hubs → small gap)
>   trades against reuse (high, sparse hubs → one field serves more pairs) — and *that* knob,
>   unlike the cone size, is chosen with arithmetic we already paid for.

### 5d. Two regimes: the per-pair mixing boundary (primary) vs the global hub measure (deferred)

The rungs of §5c quietly answered the *global* question — "which nodes are good bridges for
*any* pair." But that conflates two problems, and the **per-pair** one (the original
`d_eff`-style query, "distance between *these two* nodes") is both primary and *easier*.

**Per-pair: search only the mixing boundary.** For a fixed pair the relevant bridges live in
the **common-ancestor space** `CA(u,v) = anc(u) ∩ anc(v)`, which is upward-closed (once the two
lineages mix, everything above is common). The minimum caret is achieved on its **lower
boundary** — the lowest common ancestors, i.e. a node that is "mixed" (both lineages reach it)
yet has **at least one child still in a single lineage**. Every node above the boundary only
adds `2` per level (§5b), so it can never win the `min`. This gives an *exact, precompute-free*
algorithm that **does not climb to the root**: expand the joint up-BFS from `u` and `v` in
lockstep by radius `r`, and stop once the best matched sum `≤ r+1` (any *unmatched* common
ancestor has far-side depth `≥ r+1`, hence sum `≥ r+1`, so it cannot beat the best). The search
radius is bounded by `max(d(u→LCA*), d(v→LCA*))` — in the **balanced** case `≈ caret/2`, but for
an **asymmetric** pair (one node *is* the LCA) the near side speculatively climbs *above* the
LCA up to `≈ caret` before the stop fires (it cannot know it is the LCA until the far side
arrives). In every case it stays near the common-ancestor space rather than the height to root —
on a tall stem with a low fork it touches a handful of nodes where the full-cone
`caret_distance_lca` touches the whole stem — but the **worst-case** node-visit count is still
`O(V+E)` for graphs with wide upward frontiers. (`caret_distance_lca_boundary[_counted]`.) This is the honest framing of §5c: the
global hub set is an *approximate, reusable stand-in* for this boundary, justified only when
**batching many pairs** amortizes its precompute; for a one-off pair, just search the boundary.
The live runtime now does exactly that: `WamState::category_caret_distance` is the
boundary-restricted lockstep search over the real edge accessor (replacing the earlier
full-cone joint BFS), with `category_caret_distance_counted` exposing the visit count — it
equals the full caret everywhere and, on a tall stem with a low fork, touches three nodes
where the full cone walked the whole stem.

**Global hub measure — deferred, and the following is a *conjecture*, not a result.** The
global question ("rank all nodes as generic bridges") is harder, because — as the §5c rungs
keep running into — *some* fan-in is near-universal (any multi-parent node reconverges
*somewhere*), so raw merge counts do not discriminate. The missing ingredient is the **semantic
diversity** of what merges: a node is a good *generic* bridge only if its parents (or the child
populations it joins) span genuinely *different* regions, so it sits on the boundary for *many
diverse* pairs rather than for near-duplicates. A speculative way to score that with category
**embeddings**: stack a node's parent vectors into a matrix `M` and take its singular values
`σ₁ ≥ σ₂ ≥ …`. The *product* of the top few, `∏σᵢ = √det(MMᵀ)`, is the determinantal-diversity
(DPP / Gram-volume) measure used elsewhere for "diverse subset" scoring — the **volume** the
parents span in semantic space. But the raw volume conflates *magnitude* (parent count, vector
norms) with *spread*, so the better diversity score is the **geometric mean** of the top few,
`(∏₁ᵏ σᵢ)^{1/k}` — the volume *normalized per dimension*, i.e. the average semantic spread per
effective axis, **decoupled from count**. That cleanly separates the two factors a good hub
wants: the geometric mean is the pure **diversity** term, and the **parent count `p`** is the
separate **magnitude** term — a combined score would multiply them (`p · geomean`), rather than
let a high count masquerade as diversity. A natural truncation rank `k` is the **size-biased
mean parent count** `E[p²]/E[p]` (the effective branching seen along a random edge), `≈ 4` for
Wikipedia — so "geometric mean of the top-4 singular values, times parent count" is the first
guess. **Caveat:** this is an *ad-hoc proposal*; the truncation rank, parents-vs-child-centroids,
and the count/diversity weighting are all unvalidated, and it presumes a meaningful embedding.
Recorded as a future direction, not a recommendation.

**Known limitation of the rung-4 lift null (deep DAGs).** The configuration-model null
`E|A∩B| ≈ |A|·|B|/N` assumes *independent* ancestor membership, which a strongly hierarchical
DAG violates: ancestor-set sizes grow as `Θ(branching^depth)`, so for deep nodes `|A|·|B|/N` can
**exceed** the actual intersection, driving `lift < 1` (or undefined) even for genuine hubs. The
null is therefore calibrated only for shallow/sparse hierarchies; for deep, high-branching DAGs
the **absolute** lift values are unreliable, though the **ranking** of hubs against each other
stays usable (the bias is roughly monotone in depth). The Gene Ontology semantic-similarity
literature avoids this with an **information-content** null instead — `IC(t) = −log₂ P(node
annotated under t)`, with similarity read from the IC of the LCA (Resnik 1995; Lin 1998) — a
depth-aware baseline that does not inflate. For calibrated absolute scores, an IC-style null is
the principled replacement; for bridge *selection* (a ranking), the current lift suffices.

**This IC null is now implemented** (`information_content`, `resnik_similarity`,
`lin_similarity`). It rests on a **descendant sketch** — `descendant_minhash`, the exact
downward mirror of the rung-4 `ancestor_minhash`: one reverse-topological pass gives each node a
fixed-`k` KMV sketch of its descendant cone, which (being a *set*) **dedups by construction**, so
`sketch_card` estimates the *distinct* cone size `|desc(t)|` that `descendant_weight` over-counts.
Then `IC(t) = −log₂(|desc(t)|/N)`, `resnik = IC(MICA)` (the most informative common ancestor —
max `IC` over the common ancestors, which for *exact* IC is a lowest one since `IC` is
non-increasing upward; note the MICA can be non-unique, but we return the IC *value*, so ties are
immaterial, and on *saturated* sketches the estimated-max node may not be the exact MICA),
and `lin = 2·IC(MICA)/(IC(u)+IC(v)) ∈ [0,1]`, **undefined (→ `None`) when both nodes are the
root** (`IC = 0`, a `0/0` ratio). The cost is split: the **sketch + IC** are the `O(V·k)`
precompute / `O(k)` read; `resnik`/`lin` then add a **per-query `O(V+E)` ancestor BFS** to find
the MICA (it is *not* an `O(k)` read — no DAG library does sub-`O(V+E)` MICA without all-pairs
precompute). Unlike the configuration-model lift it uses *actual* cone frequencies, so it stays
calibrated on deep DAGs — the principled absolute-score companion to the lift's ranking signal.
(Hub *selection* from these scores is still the open global problem; this only fixes the
calibration of the relatedness read-out.)

> **`μ`-weighted IC — "partial admission" (the first membership-weighted read-out).** `IC` counts
> every descendant as `1`; on a non-taxonomic graph the **associative leak** bloats a domain node's
> cone with out-of-domain descendants, so its `|desc|` looks huge and the raw `IC` mistakes it for
> *generality* (big cone → low `IC`). Fix it by **admitting each node with weight `μ ∈ [0,1]`**
> instead of 0/1 (the fuzzy-membership `μ` of the measurement note): `IC_μ(t) = −log₂(Σ_{d∈desc(t)}
> μ_d / Σ_all μ)` (`information_content_weighted`, with `descendant_mu_mass` the distinct weighted
> mass; `μ ≡ 1` recovers `IC`). Low-`μ` leaked descendants barely count, so the effective cone
> shrinks to its in-domain size and the node's `IC` is *raised* to its true specificity — on a leaky
> test graph a physics node's IC goes `0.49 → 1.28`, closing the false generality gap to a clean
> sibling. The admission weight need not be linear in `μ`: `fuzzy_admission` is an **S-curve**
> (logistic) transfer `w(μ) = 1/(1+e^{−k(μ−c)})` — default `c=0.55, k≈4.39` fits the anchors
> `w(0.8)=0.75`, `w(0.3)=0.25` — for a nearly-full/nearly-zero admission with a tunable knee.
> *Scaling caveat:* `descendant_mu_mass` is exact (per-node BFS, `O(V·(V+E))`); the large-graph form
> wants a **`μ`-weighted MinHash** sketch (the weighted analogue of `descendant_minhash`), future
> work. And this is exactly where membership *matters* (per §-aside): a read-out over the *global*
> cone, not the per-pair caret, which is membership-robust.
>
> **Novel-contribution flag.** Resnik-style IC over a frequency null (Resnik 1995; Seco 2004's
> intrinsic descendant-count form) is established; *graded* (fuzzy `μ ∈ [0,1]`) **partial admission**
> into that descendant-mass null — as the calibration fix for the associative leak on a non-taxonomic
> DAG — is, to our knowledge, original here and not a restatement of a published estimator. It is a
> strict generalization: `μ ≡ 1` reduces *exactly* to the sketch IC (regression-tested in
> `mu_weighted_ic_reduces_to_sketch_ic_at_mu_one`). The default S-curve steepness is the *exact*
> anchor fit `k = 4·ln 3 ≈ 4.3944` (symmetric anchors `0.25` either side of `c`); the logistic only
> *asymptotes* to `0`/`1`, so if exact endpoints are wanted (fully-out nodes contributing *exactly*
> zero) Zadeh's piecewise-quadratic S-function is the drop-in alternative.

### 5e. A gentle primer on information-content similarity (for the reader learning this)

*§5d is written for someone who already has the vocabulary. This subsection builds the same idea
from scratch, with worked numbers — skip it if §5d read easily. Running example: the balanced
tree the test uses — root `0`; `1,2 → 0`; `3,4 → 1`; `5,6 → 2`, so seven nodes total.*

**One idea: rarity is information.** Imagine someone tells you "this article is filed under
category `t`." How *informative* is that? If `t` is the root (every article is under it), you
learned nothing — it was certain. If `t` is a tiny, specific leaf category, you learned a lot —
that was surprising. So a node's information is its **rarity**: let `p(t) = |desc(t)| / N` be the
fraction of all nodes that fall under `t` (its descendant cone over the total). The root has
`p = 1`; a leaf has `p = 1/N`. This descendant-fraction definition is **intrinsic** IC — it reads
the rarity off the graph structure alone, needing no external corpus of annotation frequencies;
Seco, Veale & Hayes (2004) introduced it and showed it matches corpus-based IC closely (≈ 0.84 vs
0.79 correlation with human similarity benchmarks), which is why we use it here.

**Why `−log₂`.** We want "information" to be `0` for the certain thing (`p=1`) and to *grow* as
things get rarer (`p → 0`), and we want it to *add up* for independent facts. The function with
those properties is `IC(t) = −log₂ p(t)` — the number of **bits of surprise**. Worked on the
example (`N = 7`):

| node | cone `desc(t)` | `\|desc\|` | `p = \|desc\|/7` | `IC = −log₂ p` |
|------|----------------|-----------|------------------|----------------|
| `0` (root) | all seven | 7 | 1.00 | **0.00** |
| `1` (internal) | `{1,3,4}` | 3 | 0.43 | **1.22** |
| `3` (leaf) | `{3}` | 1 | 0.14 | **2.81** |

So depth/specificity shows up as higher IC, automatically — no hand-tuned "level" number, just
the cone fraction.

**Resnik similarity: how related are `u` and `v`? Look at the deepest category that holds both.**
The common ancestors of `u` and `v` are the categories containing *both*. The **most informative**
one — smallest cone, highest IC — is their *most specific shared category*, the `MICA`. Resnik
says: `sim(u,v) = IC(MICA)`. Intuition: if the deepest thing that contains both *quantum
electrodynamics* and *quantum chromodynamics* is the very specific *quantum field theory*, they
are closely related; if the only thing containing both *QED* and *medieval poetry* is the root
("everything"), they are unrelated (IC = 0). On the example: `Resnik(3,4)` — their deepest shared
category is `1`, so `= 1.22`; `Resnik(3,5)` — they share only the root, so `= 0`. (Why the MICA is
always a *lowest* common ancestor: cones only grow as you go up, so `IC` only *falls* as you go
up — the maximum IC is at the bottom of the shared region, the merge frontier of §5d.)

**Lin similarity: normalize so "identical" scores 1.** Raw Resnik isn't on a fixed scale — a deep
tree gives big IC numbers, a shallow one small. Lin divides by how specific the two items
themselves are: `sim(u,v) = 2·IC(MICA) / (IC(u) + IC(v))`. If `u = v` (and `IC(u) > 0`) the MICA
*is* `u`, so it is `2·IC(u)/2·IC(u) = 1`; if they share only the root, `IC(MICA)=0` so it is `0`.
On the example, `Lin(3,4) = 2(1.22)/(2.81+2.81) = 0.43`. Now every pair sits in `[0,1]`, comparable
across graphs. **One exception:** when both nodes are the root, `IC(u)=IC(v)=0`, the denominator is
`0`, and Lin is undefined (`0/0`) — the implementation returns `None` there (while Resnik returns
`0`). So the "identical → 1" identity holds for every node *except* the root. (`information_content`
is an `O(k)` read, but calling `resnik`/`lin` is *not* `O(k)` — each runs a per-query `O(V+E)`
upward BFS to find the MICA; cache the ancestor sets for repeated queries on the same graph.)

**FaITH similarity: the Jiang–Conrath-faithful sibling.** Lin isn't the only way to normalize
Resnik. The *Jiang–Conrath distance* (Jiang & Conrath 1997) `JC(u,v) = IC(u) + IC(v) − 2·IC(MICA)` measures *how far
apart* `u` and `v` are: it is the IC you'd have to "spend" climbing from each down to the MICA —
`0` for identical nodes, large when they meet only high up. **FaITH** (Pirró & Euzenat 2010) turns
that distance into a bounded similarity, `sim(u,v) = IC(MICA) / (IC(u) + IC(v) − IC(MICA))`, which
rearranges to the clean form `FaITH = 1 / (1 + JC/IC(MICA))` — a distance-to-similarity map scaled
by how informative the shared category is. It sits in `[0,1]`, is `1` for identical non-root nodes
and `0` when only the root is shared (`FaITH(3,4) = 1.22/(2.81+2.81−1.22) = 0.28` on the example —
note it ranks the *same* pairs as Lin but on a different curve, being harsher on weak overlap). A
small honesty correction to a tempting claim: FaITH is sometimes said to "avoid the undefined-at-
root case," but it does **not** — at root-root all three ICs are `0`, the denominator
`IC(u)+IC(v)−IC(MICA)` (which is `≥ max(IC(u),IC(v))`) is `0`, and it returns `None` exactly as
Lin does. Its real merit is the JC-faithfulness, not dodging that corner. (`faith_similarity`.)

**Why we needed the descendant *sketch* (and not the additive weight).** Every formula above needs
`|desc(t)|`, the **distinct** count of nodes under `t`. Computing that exactly for all `t` is
reachability — the global blow-up we keep refusing. The cheap one-pass additive `descendant_weight`
(rung 2) is no good *here*: it counts a node reachable by two paths **twice**, so it inflates
`|desc|`, distorts `p`, and would corrupt the IC. The fix is a **set**: `descendant_minhash` keeps
a fixed-`k` sample of the cone, and a set automatically counts each member once — so its size
estimate is the *distinct* `|desc|` we need. Same `O(V·k)` precompute / `O(k)` read **for the
sketch and `information_content`** as the rung-4 ancestor sketch, just pointed downward (the
`resnik`/`lin` MICA search on top adds the per-query `O(V+E)` BFS noted above). (And it is the §6
kernel-trick move once more: the cone is the big object we never materialize; the sketch is the
small handle we read it through.)

**Where this sits.** This gives a *calibrated relatedness read-out* between two nodes that does not
inflate on deep graphs — the honest replacement for the rung-4 lift's absolute value. What it does
**not** yet answer is the *global* question — *which* nodes make good generic bridges to
precompute — which stays open (§5d). Picking good bridges is selection; scoring how related two
nodes are is a read-out; this increment is the read-out.

**Live path.** As with the caret, the read-outs are wired into the runtime:
`WamState::build_descendant_sketches` precomputes and caches the descendant sketches once (like
`build_boundary_distances`/`build_boundary_jets`), and `category_resnik` / `category_lin` /
`category_faith` answer per-pair queries against them — eager-edge only (the sketch needs the
in-memory parent map), `None` until the sketches are built.

## 6. Aside: the kernel-trick analogy

*(A mnemonic, not load-bearing — the mechanics above stand on their own; skip if you only
want the implementation contract.)*

In kernel methods a feature map `φ: X → H` (often infinite-dimensional) is **never
materialized**, because every algorithm is written to touch only inner products
`K(x, y) = ⟨φ(x), φ(y)⟩`. The structure of `K` (bilinearity, positive-definiteness)
guarantees the implicit computation is exact. The win is purely *implicitness*: you
compute in a huge space while only ever handling small quantities.

The correspondence here is tight:

| kernel methods | graph functional semirings |
|---|---|
| feature map `φ(x)` (big / ∞-dim) | path-length histogram `H_v(z)` (budget-dim power series) |
| "never form `φ`" | "never form the histogram" |
| inner product `K(x,y)=⟨φ(x),φ(y)⟩` | functional `F(H_v)` propagated by its own law |
| bilinearity / Mercer PSD makes `K` factor | semiring homomorphism makes `F` factor through `(⊕,⊗)` |
| representer theorem: solution in `span` of data | splice: query value determined by boundary values |
| a kernel that *is* an inner product (admissible) | a functional that *is* a homomorphism (mass, moments, min/max) |
| a similarity that is **not** PSD (no RKHS) | `WeightSum_N`, which is **not** `⊗`-multiplicative (§2) — no scalar law |

Two honest limits on the analogy. (a) It is an analogy *by implicitness*, not a literal
RKHS — the structure exploited is a commuting diagram / semiring homomorphism, not
positive-definiteness, and `min`/`max` are idempotent-semiring (tropical) read-outs with
no inner-product counterpart. (b) The analogy even predicts its own failure mode: just as
a non-PSD similarity has no implicit feature computation, the non-factoring `WeightSum`
has no scalar splice — and in *both* cases the recourse is to fall back to the explicit
object (here, the full histogram, or the §5 bracket).

## 7. CLT reconstruction at deep nodes

A deep node's length is a sum over many path stages, so for well-mixed nodes the
path-length distribution tends to Gaussian (Lindeberg CLT, when the stages are many and
comparable). That means the **moment jet `(M, m₁, m₂)` can reconstruct an approximate
histogram without ever building one**: read off `mean`, `var`, and emit a discretised
`Normal(μ, σ²)` truncated to the `(min, max)` support bracket.

**[Implemented — increment 1b]** `boundary_cache::HistRepr::MomentNormal { support, mean,
std, total }` (wire tag 6) is exactly this rung — the moment-matched discretised Normal,
the cheapest reconstruction (5 scalars, no EM). It is constructible from the jet alone via
`MomentJet::to_normal_repr`, and `fit_moment_normal(h)` routes through the same
`hist_moment_jet`, so the jet-built and histogram-fitted forms agree bit-for-bit. It joins
the candidate ladder under the same CDF gate (a bimodal node misses `ε_K` and is
rejected).

**[Implemented — third moment]** The jet carries `m₃` (and `m₄`, below) with a
`skewness()` read-out. The first payoff is a sharper **binomial**: `fit_binomial_moments`
fits `(n, p)` from the *mean and variance* (`p = 1 − var/mean`, `n = mean/p`) instead of
pinning `trials = support−1` and matching only the mean — so it recovers the true `n` of a
binomial embedded in a wider support, gets the spread right, and the skew corroborates it
(`moment_binomial_recovers_n_in_wider_support`). It returns `None` for over-dispersed data
(`var ≥ mean`), cleanly ceding to the beta-binomial.

**[Implemented — Gram–Charlier rung, complete]** The graded reconstruction family is now
fully built. The jet carries `m₄` (`MomentJet { mass, m1, m2, m3, m4 }`) with an
`excess_kurtosis()` read-out, and `HistRepr::GramCharlier { support, mean, std, skew,
kurtosis, total }` (wire tag 7) is the moment-Normal **plus skew *and* kurtosis
corrections** — a discretised Gaussian times `1 + (γ₁/6)·He₃(z) + (γ₂/24)·He₄(z)`
(`gram_charlier_pmf`; the tail can dip negative, a known artefact, so negatives are clamped
and renormalised). Constructible from the jet alone (`MomentJet::to_gram_charlier_repr`).
It is a *perturbation of a Gaussian*, so it is for **mildly non-normal, unimodal** nodes —
**not** strongly multimodal ones; the CDF gate enforces that. Validated by
`gram_charlier_beats_normal_on_a_skewed_unimodal` (a skewed Poisson),
`kurtosis_correction_beats_skew_only_on_leptokurtic` (a symmetric scale-mixture, where the
`m₄` term earns its place over skew-only), and `gram_charlier_rejected_for_bimodal`. The
`(M,m₁,m₂) → +m₃ → +m₄` reconstruction ladder is **complete** (the next term, `m₅`, would
buy diminishing returns and is not carried).

- This is the principled three-scalar payload for distribution *reconstruction* —
  `(min, max, mass)` cannot do it, because the range is a sample-size-dependent,
  badly-biased estimator of `σ`; you need the **second moment**, not the extremes.
- **Raw → central is exact; the model enters only at "moments → CDF."** Converting the
  propagated raw moments to central ones is pure algebra, no model
  (`μ₂ = m₂′ − μ²`, `μ₃ = m₃′ − 3μ·m₂′ + 2μ³`, … with `m_k′ = m_k/M`). A *model* is
  needed only for the last step — a finite moment set does not determine a distribution
  — and that choice gives a **graded reconstruction family** that extends this rung:
  `(M,m₁,m₂)` → Gaussian (CLT); `+m₃` → Gram–Charlier / Edgeworth (adds skew); `+m₄` →
  Edgeworth / Pearson family (adds kurtosis — mild non-normality); the full jet → the
  histogram. Propagating to order `2n` thus buys a *non*-Gaussian-but-still-cheap deep
  node before paying for the histogram, and the CDF gate still arbitrates — rejecting
  the closure wherever it does not fit.
- It slots into the existing exact→approximate ladder (boundary spec §9) as a new, very
  cheap **CDF-gated reconstruction rung**, *complementary* to the discretised-GMM:
  CLT-Gaussian for deep, well-mixed, unimodal nodes (3 scalars, no EM); GMM for shallow,
  structured, multimodal nodes (expensive). If a node is multimodal the moment-Gaussian
  misses the Kolmogorov gate and the chooser rejects it automatically — the correctness
  certificate already guards the approximation, so it never silently fires where CLT
  does not hold.
- The win is exactly the implicitness of §1/§6: the moments propagate by their own
  scalar recurrence, so the deep-node distribution estimate costs three accumulators and
  never touches a histogram.

## 8. Roadmap (increments)

1. **Payload on the ancestor space (exact, safe).** Carry `(min, max, mass, m₁, m₂)` over
   the acyclic ancestor space and validate *against the existing histogram* bucket-for-
   bucket (tropical pair = first/last nonzero index; moment jet = the histogram's weighted
   sums). **[1a DONE]** `boundary_cache::{MomentJet, Interval, suffix_moment_jet,
   suffix_interval}` propagate the two semirings directly (never forming the histogram),
   with the `convolve` splice laws; validated by `moment_jet_and_interval_equal_the_
   histogram`, `convolve_laws_match_spliced_histogram`, `interval_and_mass_bracket_d_eff`.
   Concrete functions, not yet a `PathSemiring` trait — the trait is deferred until the
   distance kernel gives a second instance to generalise over (and its star/closure
   contract is settled, §3). **[1b DONE]** the moment jet → discretised-Normal CDF-gated
   reconstruction rung (`HistRepr::MomentNormal`, §7). **[1c DONE]** the payload is fused
   into the live WamState path: `build_boundary_jets` precomputes the
   `(mass, m₁, m₂, min, max)` side-table (`boundary_jet`) without forming the histogram,
   and `collect_native_category_ancestor_boundary_jet` splices it at query time
   (`δ_depth ⊗ jet_B`), validated against the full-enumeration histogram's read-outs
   (`boundary_jet_splice_matches_histogram`). The end-to-end loop — propagate, splice,
   reconstruct — now runs without ever materialising a histogram. **[1d DONE]** the jet now
   carries `m₃` with a `skewness()` read-out, and `fit_binomial_moments` uses mean+variance
   to recover the true `n` of a binomial (the accurate-binomial payoff of the skew, §7).
   Remaining within increment 1: the higher-order Edgeworth/Pearson reconstruction *rungs*
   (use the carried `m₃`, and carry `m₄`).
1.5. **Per-payload closure characterization (still on acyclic data). [DONE]** Each payload's
   star/closure-or-truncation behaviour — the convergence table of §4 / §3-gap-(1) — is now
   an implemented, tested element star `a* = ⊕_{i≥0} aⁱ = one ⊕ a⊗a*`:
   - **min-plus**: closed, `a* = 0` (looping never shortens the shortest path) — the graph-level
     form is `min_distance_closure` (**[2a]**).
   - **counting / moment jet** (`MomentJet::star`): converges to a **budget-free closed form iff
     `mass < 1`** (`Σ massⁱ = 1/(1−mass)` finite), `None` otherwise. Pure path-counting has
     integer `mass ≥ 1` on any real loop, so it always diverges → the cycle must be
     **truncated** (the existing budget + visited-guard); a *discounted/weighted* jet (`mass < 1`)
     has the finite closed form. Checked against the explicit geometric histogram
     (`moment_jet_star_converges_iff_mass_below_one`).
   - **interval** (min-plus × max-plus, `Interval::star`): the min factor closes at `0`, but the
     **max** (longest-path) factor diverges on any positive-length loop, so the interval star is
     `None` except for the degenerate length-0 loop.

   So divergence is **per-payload and now explicit**, on the acyclic domain checked against the
   histogram: counting/max-plus need truncation, min-plus terminates, the weighted moment jet has
   a closed star. This was logically prior to the cyclic increment (which is `[DONE]` below), so
   it kept step 2's only genuinely new unknown to cyclic *control flow*, not per-payload
   divergence as well.
2. **Distance / shortest-path kernels + cyclic closure.** Point the now-generic
   machinery at `transitive_distance3`, then `weighted_shortest_path3` /
   `astar_shortest_path4` (boundary suffixes as ALT landmarks), adding the
   closed-semiring / budget-truncation path for cyclic up-sets — the only genuinely new
   correctness work.
   - **[2a DONE]** the min-plus closure foundation: `min_distance_closure` (a BFS fixpoint,
     `a* = 0`) computes cycle-correct shortest `node→root` distances, where the DFS+memo
     recurrences are unsound on cycles (§4). This is also the §1.5 closure characterization
     for the min-plus payload, settled before the kernel wiring.
   - **[2b DONE]** the weighted min-plus payload and the distance splice:
     `weighted_distance_closure` (a Bellman-Ford relaxation summing per-edge weights, the
     general closure of which the 2a BFS is the `weight ≡ 1` case) and `distance_splice`
     (`min_B (dist(seed→B) + dist(B→root))` — the ALT landmark identity, exact when the
     boundary is a cut), validated by `weighted_closure_respects_edge_weights` and
     `distance_splice_equals_full_closure` (unweighted and weighted).
   - **[2c DONE]** the distance closure + splice are fused into the live WamState path:
     `boundary_dist` (a `node -> dist(B->root)` side-table), `build_boundary_distances`
     (built from `min_distance_closure`, so cycle-correct), and
     `category_ancestor_boundary_distance` (a BFS from the seed that stops at a cached
     boundary and adds its suffix — the ALT-landmark prune; degrades to a plain correct
     BFS with an empty cache). Validated by `boundary_distance_splice_matches_closure`.
   - **[2d-i DONE]** the `PathSemiring` trait is extracted (§3): `MomentJet` and `Interval`
     implement it, and `suffix_moment_jet` / `suffix_interval` are now thin instances of one
     generic `suffix_value::<S>`. The cyclic star stays a per-payload free function (only
     min-plus is closed), the ⊕/⊗ asymmetry is on the trait, and the laws are guarded by
     `path_semiring_laws_and_generic_equivalence`.
   - **[2d-ii DONE]** the distance cache is wired into the kernel *codegen*. The faithful
     home turned out to be the existing **to-a-fixed-root** kernel, not `transitive_distance3`
     (which is a general *source→any-target* stream with no fixed root — the to-root cache
     does not apply without artificially pinning the target). So `boundary_optimization`
     gains a `boundary_result_extractor(shortest_distance)` mode: the upgraded
     `category_ancestor_boundary` wrapper returns the cycle-correct shortest hop-distance to
     root via `category_ancestor_boundary_distance` (the min-plus cache), not the histogram.
     Validated by `option_shortest_distance_extractor` (lowering) and
     `wrapper_shortest_distance_matches_closure` (cargo-gated exec, incl. the empty-cache
     fallback). **This closes increment 2.**
   - *Deferred (own track):* a dedicated fixed-target `transitive_distance3` variant and
     `astar_shortest_path4` ALT landmarks (`|d(u,L) − d(v,L)|`) — a general *between-nodes*
     query needs the landmark formulation, not the to-root splice.

3. **Between-nodes distance from the to-root cache (composite caret).** The *between-nodes*
   companion to increment 2, §5a.
   - **[3a DONE]** `caret_distance_upper` (the O(1) root-bridge caret — an *undirected*
     upper bound) and `caret_distance_lca` (the exact `∧`-distance through the lowest common
     ancestor). `directed_distance_lower` is the only valid cache lower bound, and only on
     the *directed* `d(u→v)` — the symmetric `|d_u − d_v|` is NOT a lower bound on the
     undirected distance off a tree (corrected; `alt_lower_bound_is_directed_only`).
     Validated on a tree (caret = true distance) and a DAG (caret = certified upper bound).
   - **[3b DONE]** the live WamState path: `category_caret_distance(u, v, acc)` (the exact
     between-nodes `∧`-distance by a joint upward BFS over the edge accessor) and
     `category_ancestor_astar(u, target, acc)` (directed shortest `u→ancestor` via **A\*
     with the ALT landmark heuristic** `h(n) = max(0, min_dist[n] − min_dist[target])` from
     the loaded distance-to-root table — admissible/consistent; degrades to Dijkstra with
     an empty `min_dist`). Validated by `live_caret_distance_matches_lca` and
     `astar_ancestor_distance_matches_closure`.
   - **[measured]** the A* prune is **structure-dependent, and that is inherent to a single
     root landmark.** `h(n) = d(n→root) − d(target→root)` is *exact* exactly when `target`
     **dominates** the path to root (every `n→root` shortest path crosses `target`) — there
     A* expands only the optimal path and prunes hard (`astar_alt_prunes_a_dominator_decoy`:
     ALT expands strictly fewer nodes than Dijkstra). Across a branch that **bypasses**
     `target`, the same `h` is a loose lower bound and cannot prune (a root landmark sits
     *behind* an ancestor target). So the distance-cache A* pays off for dominator-shaped
     ancestor queries; a *general* speedup wants **periphery** landmarks (classic ALT picks
     landmarks "beyond" the targets), which the boundary machinery could precompute but does
     not yet — the honest next measurement-driven step if A* on general graphs is wanted.
   - **[3d DONE]** `caret_distance_budgeted(u, v, parents, budget)` — the caret with a
     path-length **budget** on the joint up-walk, so the **budget is the bridge-level knob**:
     a small budget admits only a LOW common ancestor (a tight, local relation), a budget ≥
     the subtree height always reaches the bridge and equals `caret_distance_lca`. Its
     natural value is the **support upper bound** (`max` from the interval payload, increment
     1), which bounds depth-to-subtree-root — so the increment-1 payload feeds the
     increment-3 caret. Validated by `budgeted_caret_scopes_the_bridge_by_level` and
     `support_upper_bound_is_a_sufficient_caret_budget`.
   - **[3d′ DONE]** `caret_through_bridge(u, v, B)` — the caret through a *designated*
     reference `B` (§5b). The complement to the auto-LCA measure: the LCA caret collapses to
     the shortest path (no info beyond distance), while a fixed bridge keeps the level
     signal, with the exact gap `through(B) − lca = 2·d(LCA→B)`
     (`caret_through_bridge_vs_lca_and_the_gap`).
   - **[3e DONE]** a **real-data integration** on the Wikipedia category graph
     (`data/benchmark/{dev,300,10k,10x}/category_parent.tsv`). Harness:
     `wikipedia_category_subtree_end_to_end_3e` (env-var gated on `UW_CATEGORY_TSV`, skips in CI;
     `UW_CATEGORY_ROOT` / `UW_CATEGORY_MAXDEPTH` scope a single subtree). **Invariants held across
     four scales (≤25k edges, raw + scoped):** boundary caret `==` full caret, 3f cached-landmark
     caret `==` per-query `caret_min_over_hubs`, `min_distance_closure` terminates. **The lesson:
     scope first.** A nominally "Physics-rooted" crawl is *not* a subtree of Physics — its
     unbounded cone spans most of Wikipedia (7811/8247 nodes at 10k), is **cyclic** (so
     `descendant_minhash` → `None`, IC unavailable), and its fan-in hubs are **maintenance
     categories** (`Container_categories`, 1778 children; the hub-quantized caret then inflates to
     7 vs an exact 1). Restricting to the **bounded-depth descendant cone** (depth ≤ 3) fixes both:
     the subtree is **acyclic** (IC runs) and the hubs are semantic (`Subfields_of_physics`,
     `Matter`, `Energy`). On it the IC read-outs track real physics — `Electromagnetism`–`Optics`
     Lin 0.68 ≫ `Thermodynamics`–`Optics` 0.36 — and the quantization gap closes. **The deeper
     resolution:** the leak is a *downward-cone* problem (Wikipedia categories are associative, not
     is-a, so `Physics → Matter → Physical_objects → Organisms → …` is real, not a data bug); the
     **per-pair bidirectional bridge sidesteps it** — `caret_optimal_bridge(u, v, budget)` explores
     only the two nodes' *up-cones*, finds where *their* lineages mix, and on the **raw, uncurated,
     cyclic** graph recovers semantically-correct bridges (`Classical_mechanics`×`Electromagnetism`
     → `Subfields_of_physics`; `Electromagnetism`×`Optics` → `Electromagnetism`), stable across all
     scales, no cone needed. Residual: even scoped, fan-in can prefer `Physicists_by_nationality`
     over `Subfields_of_physics` — concrete motivation for the deferred semantic-diversity *global*
     hub selection (per-pair bridges are already good without it). Full write-up:
     `WAM_RUST_CARET_REALDATA_MEASUREMENT_2026-06-18.md`.
   - **[3f DONE]** the **landmark-cached designated-bridge caret** (§5c): `bridge_distance_fields`
     precomputes `d(·→B)` (one downward BFS per bridge over the shared children graph, `O(E +
     Σ_B|desc(B)|) ≤ O(K·V)`), and `caret_through_bridge_cached` / `caret_min_over_cached_bridges`
     then answer in **O(1)** / **O(#bridges)** per pair — the boundary-cache reuse applied to the
     caret. Wired into the live path: `WamState::build_caret_landmarks`,
     `category_caret_through_bridge`, `category_caret_min_over_landmarks`. This is the missing
     amortization that makes hub *selection* (§5c rungs) pay off: pick the convergence hubs as
     bridges, cache their fields once, and `caret_min_over_hubs` becomes O(#hubs) lookups instead
     of a BFS per hub. The cached field is the **min-plus distance functional** read off a compact
     precompute — never forming the path-length distribution between the pair, the §8 "carry the
     functional, not the distribution" theme again (formally a min-plus inner product over a
     2-hop-cover / hub-labeling structure — see the §5c theory-grounding note for the literature
     and the `O(K·V)` space-tightness result). The cached read **equals** the per-query
     `caret_min_over_hubs`, but note *that* is the true LCA caret only when an optimal common
     ancestor is among the bridges; with a sparser bridge set both are a valid **upper bound**
     (gap `2·d(LCA→nearest bridge)`, §5c). Validated by
     `bridge_landmarks_cached_caret_matches_per_query` and `live_bridge_landmarks_match_library`.
   - **[deferred]** the **nested-cut** hierarchy (§5c) — cache the suffix above each cut and
     compose by convolution; worth it only for *deep* hierarchies, and needs the
     dominator/cut property maintained on a DAG. And **[3c, optional]** a between-nodes
     *kernel* result mode in the codegen; periphery-landmark selection for general A*.

## 9. Relationship to the other docs

- `WAM_RUST_BOUNDARY_DISTRIBUTION_PHILOSOPHY.md` — why the boundary cache exists and
  what the measurements showed (the histogram instance).
- `WAM_RUST_BOUNDARY_DISTRIBUTION_SPECIFICATION.md` — the shipped histogram cache, the
  `g_B` basis, and the §9 approximation ladder this note's CLT rung extends.
- `WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md` — phase status of shipped work.
- This note — the algebraic generalization (product semirings, ancestor-space domain,
  the implicit-functional / kernel-trick framing) that the next increments build on.

## 10. References

Collected for understanding the theory and as a citation base for possible future write-up.
Each is referenced inline at the section that uses it.

**Information-content semantic similarity (§5d, §5e).**
- Resnik, P. (1995). *Using Information Content to Evaluate Semantic Similarity in a Taxonomy.*
  IJCAI-95. — `resnik_similarity = IC(MICA)`.
- Lin, D. (1998). *An Information-Theoretic Definition of Similarity.* ICML 1998. —
  `lin_similarity = 2·IC(MICA)/(IC(u)+IC(v))`.
- Jiang, J. J., & Conrath, D. W. (1997). *Semantic Similarity Based on Corpus Statistics and
  Lexical Taxonomy.* ROCLING X (also arXiv cmp-lg/9709008). — the JC distance `IC(u)+IC(v)−2·IC(MICA)`.
- Seco, N., Veale, T., & Hayes, J. (2004). *An Intrinsic Information Content Metric for Semantic
  Similarity in WordNet.* ECAI 2004. — the **intrinsic** (descendant-count) IC we use, `IC(t) =
  −log₂(|desc(t)|/N)`, ≈0.84 vs 0.79 human-benchmark correlation against corpus IC.
- Pirró, G., & Euzenat, J. (2010). *A Feature and Information Theoretic Framework for Semantic
  Similarity and Relatedness.* ISWC 2010. — the **FaITH** measure `IC(MICA)/(IC(u)+IC(v)−IC(MICA))`.

**MinHash / KMV sketches and distinct-count estimation (rung 4, §5d; descendant sketch, §5e).**
- Broder, A. Z. (1997). *On the Resemblance and Containment of Documents.* SEQUENCES 1997. —
  MinHash for Jaccard.
- Bar-Yossef, Z., Jayram, T. S., Kumar, R., Sivakumar, D., & Trevisan, L. (2002). *Counting
  Distinct Elements in a Data Stream.* RANDOM 2002. — k-minimum-values (KMV) distinct-count.
- Beyer, K., Haas, P. J., Reinwald, B., Sismanis, Y., & Gemulla, R. (2007). *On Synopses for
  Distinct-Value Estimation Under Multiset Operations.* SIGMOD 2007. — the bottom-`k` / KMV
  `(k−1)/ĥ_k` cardinality estimator (`sketch_card`).
- Cohen, E., & Kaplan, H. (2007). *Summarizing Data Using Bottom-k Sketches.* PODC 2007. — the
  one-pass bottom-`k` Jaccard estimator (`sketch_jaccard`).

**2-hop cover / hub labeling and landmark distance (§5c, roadmap 3f).**
- Cohen, E., Halperin, E., Kaplan, H., & Zwick, U. (2002/2003). *Reachability and Distance
  Queries via 2-Hop Labels.* SODA 2002 / SIAM J. Comput. 2003. — the 2-hop label framework that
  `bridge_distance_fields` instantiates.
- Abraham, I., Delling, D., Goldberg, A. V., & Werneck, R. F. (2012). *Hierarchical Hub Labelings
  for Shortest Paths.* ESA 2012.
- Akiba, T., Iwata, Y., & Yoshida, Y. (2013). *Fast Exact Shortest-Path Distance Queries on Large
  Networks by Pruned Landmark Labeling.* SIGMOD 2013.
- Tretyakov, K., Armas-Cervantes, A., García-Bañuelos, L., Vilo, J., & Dumas, M. (2011). *Fast
  Fully Dynamic Landmark-based Estimation of Shortest Path Distances in Very Large Graphs.* CIKM
  2011. — the `O(K·V)` space-tightness for O(1)-lookup landmarks.
- Storandt, S. (2022). *Algorithms for Landmark Hub Labeling.* ISAAC 2022. — `min_B(d(u→B)+d(v→B))`
  as a valid upper bound, exact iff an optimal meeting node is a landmark.

**Distribution reconstruction (§7).**
- Lindeberg, J. W. (1922). *Eine neue Herleitung des Exponentialgesetzes in der
  Wahrscheinlichkeitsrechnung.* Math. Z. 15. — the CLT condition for the moment-jet → Gaussian rung.
- Blinnikov, S., & Moessner, R. (1998). *Expansions for Nearly Gaussian Distributions.* A&A
  Suppl. Ser. 130. — practical Gram–Charlier / Edgeworth series (the `MomentNormal` → `GramCharlier`
  reconstruction rungs).
- Berry, A. C. (1941), *The Accuracy of the Gaussian Approximation to the Sum of Independent
  Variates* (Trans. AMS 49); Esseen, C.-G. (1942) — the **Berry–Esseen** `O(n^−½)` CLT convergence
  rate. For a *symmetric* sum (`γ₁ = 0`, the symmetric diamond chain) the leading `O(n^−½)` Edgeworth
  term vanishes, leaving `O(n^−1)` — why the chain's measured `0.0019` beats the `≈0.097` bound (§3).

**Moment-ratio admissibility and mixtures (§3, the `reconstruction_class` gate).**
- Pearson, K. (1894). *Contributions to the Mathematical Theory of Evolution.* Phil. Trans. R. Soc.
  London A, 185. — the **method of moments**, and the "dissection" of a frequency curve into a
  2-Gaussian mixture; the general *asymmetric* case reduces to a **9th-degree (nonic)** polynomial,
  the symmetric equal-variance case to the closed-form `δ = σ·(−γ₂/2)^¼`.
- Sharma, R., & Bhandari, R. (2015). *Skewness, Kurtosis and Newton's Inequality.* Rocky Mountain J.
  Math. 45(5). — bounds in the Pearson `(β₁, β₂)` plane; the universal `β₂ ≥ β₁ − 2`, attained only
  by two-point distributions (`d = 0`).
- Klaassen, C. A. J., & van Es, B. (2023). arXiv:2312.06212. — the **strictly-unimodal** floor
  `d ≥ 189/125 ≈ 1.512`; note it does *not* cover the flat/amodal uniform (`d ≈ 0.80`), which is why
  `reconstruction_class` keeps the conservative `d < 0.7` rather than raising to `≈1.5`.
- Jondeau, E., & Rockinger, M. (2001). *Gram–Charlier Densities.* J. Economic Dynamics and Control,
  25(10). — the **bounded `(γ₁, γ₂)` region** over which the Gram–Charlier A expansion is a valid
  (non-negative) density (roughly `|γ₁| ≲ 2`); outside it the §7 CDF gate rejects the result.
