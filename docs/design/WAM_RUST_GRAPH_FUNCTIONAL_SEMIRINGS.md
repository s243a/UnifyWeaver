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
  a cycle. So `star` is a per-payload free function, not a method some impls could not
  honour. (`suffix_value` itself is the acyclic recurrence.)
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
>   returning the overlap fraction in `[0,1]`.)
> - **Rung 4 — ancestor sketch + small-world lift (height-agnostic, baseline-corrected).** The
>   fixed-`up_hops` probe of rung 3 has a fatal flaw: it only sees a crossover *within* `k`
>   hops, but the **crossover height is unknown and varies per node** — too small misses deep
>   hubs, too large walks to root (the reachability we refuse). Worse, in a **small-world**
>   graph the up-cones cover most of the graph within a few hops, so *raw* overlap (at any `k`,
>   even exact) approaches 1 for **every** pair — it measures "are we in a small world," not "is
>   `B` a funnel." Two fixes, both O(k): (a) **height-agnosticism** — summarize each node's
>   *whole* lineage to root once, as a fixed-size **KMV/MinHash ancestor sketch**
>   `sig(B) = bottom-k( {B} ∪ ⋃_p sig(p) )`, one root→leaf pass; overlap (`sketch_jaccard`) is
>   then read at *any* depth with no knob (error ∝ 1/√k, not a depth cutoff). (b) **baseline
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
1.5. **Per-payload closure characterization (still on acyclic data).** Before any cyclic
   work, characterize each payload's star/closure-or-truncation behaviour — the
   convergence table of §4 / §3-gap-(1): counting needs truncation, min-plus terminates,
   the moment jet's star needs mass `< 1`. Do it on the acyclic domain where each can be
   checked against the histogram. This is logically prior to, and separable from, the
   cyclic increment, so step 2 then confronts **one** unknown (cyclic control flow), not
   two (control flow *and* per-payload divergence) at once.
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
   - **[3e next]** a **real-data integration** on a Wikipedia subtree (e.g. physics): extract
     the root-anchored region (`build_scoped_subtree_lmdb.py`), propagate the support
     interval, and compute multi-level budgeted carets between topics — the end-to-end
     composition on real (cyclic) data, where the 2a/2b cycle-correctness earns its keep.
   - **[3f, buildable]** the **landmark-cached designated-bridge caret** (§5c): precompute
     `d(·→B)` (one downward field) for each designated level `B`, so `caret_through_bridge`
     is O(1) per pair — the boundary-cache reuse applied to the caret. `O(K·V)` for `K`
     levels.
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
