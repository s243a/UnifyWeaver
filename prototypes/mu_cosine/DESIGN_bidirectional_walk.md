# Depth-balanced bidirectional sampler walk — design

*Detail for a future `gen_mu_pairs.py` option: a bidirectional random walk whose **depth doesn't drift**
(neither toward leaves nor toward the root). Complements the `--child-only` (downward) mode; the two are
**mixed** for training-data diversity. Theory cousin: §5d depth-stability and the λ-biased walk.*

## The goal

The sampler does short walks from a seed and emits `(start, endpoint)` pairs. We want the endpoint to land
at a depth *representative* of the seed — not systematically deeper (drifting toward leaves) nor shallower
(drifting toward the generic apex, the leak-conduit direction). Formally: **zero expected depth change per
step**.

## The handshake lemma — the means are equal

Every edge is exactly one `(child, parent)` relationship, so summed over all nodes,
`Σ children = Σ parents = #edges`. Therefore **`E[c] = E[p]` exactly**, in *any* DAG. The up/down asymmetry
is never about the mean degree (always equal) — it is about the **distribution shape**: parents are
concentrated (most nodes have 1–2), children are heavy-tailed (a few hubs have hundreds). That tail is what
`E[c²]` sees and `E[p²]` doesn't.

## The exact zero-drift rule (local)

Weight each child-edge (down) by `1` and each parent-edge (up) by `β`. At a node with `c` children and `p`
parents:

```
P(down) = c/(c+βp),   P(up) = βp/(c+βp),   E[Δdepth] = (c − βp)/(c + βp).
```

`E[Δdepth] = 0` at **every** node ⟺ `β = c/p`. Equivalently — and with no per-node arithmetic —
**flip a fair coin for up-vs-down, then pick uniformly within that direction** (`P(down)=P(up)=½`). This
makes the depth a **martingale**: `E[depth_t − depth_0] = 0` for any number of steps, regardless of the
graph. (Boundary caveat: the root has `p=0` (must go down) and leaves have `c=0` (must go up), so the
martingale holds in the bulk with reflecting-ish boundaries — fine for short walks from interior seeds.)

## The global-constant variant (the `E[c²]/E[p²]` weight)

If you'd rather a single up-weight `β` (one weighted graph, no per-node `c/p`), balance the branching the
walk *experiences*. A walk lands on nodes **size-biased by degree**, so the branching it sees is the
size-biased mean — `E[c²]/E[c]` going down, `E[p²]/E[p]` going up. Balancing them:

```
β = (E[c²]/E[c]) / (E[p²]/E[p]) = E[c²]/E[p²]      (using E[c]=E[p]).
```

So this is the **up-edge weight** that balances the walk *on average* — a size-biased mean-field estimate,
exact only in aggregate (the per-node `c/p` is exact pointwise). Estimate it once from the graph. **It is a
weight on the (few) parent edges, not the down/up probability ratio** — with it applied, the down/up
*probabilities* come out ≈ balanced, which is the point.

This is the Lyons–Pemantle–Peres λ-biased walk at criticality (up-weight = branching number ⟹ zero speed);
`c/p` and `E[c²]/E[p²]` are its local-exact and global-average forms.

## The `γ` family — one dial from depth-neutral to branching-following

"No drift" (coin-flip) is only *one* target. A different, equally valid goal is **coverage**: at a node
with a big subtree below, spend more steps going *down* (there's more to sample there). That is the
opposite of cancelling `c` — and the two live on one axis, a single knob `γ ∈ [0,1]`:

```
down/up odds = (c/p)^γ        ⟺      P(down) = 1 / (1 + (p/c)^γ)
```

- **γ = 0** → odds `1` → `P(down) = ½` → **coin-flip** (depth-neutral; ignores local branching).
- **γ = 1** → odds `c/p` → `P(down) = c/(c+p)` → the **natural undirected walk** (branching-following; drifts
  deep — the current default).
- **γ ∈ (0,1)** → high-child nodes go down *more*, tunably (at `c = p`, `P(down) = ½` for any γ — no spurious
  bias). This is the "many children ⇒ don't coin-flip" regime.

`γ` is the **depth-neutral ↔ coverage** dial: small γ keeps endpoints at a representative depth; large γ
pours the walk into the big subtrees (more coverage of where the structure is, deeper drift). No single γ
is "correct" — it's an empirical choice of what the training pairs should emphasize.

**Truer measure — subtree mass, not child count.** Raw `c` is a crude proxy for "how much is below" (3
children heading huge subtrees > 10 leaf children). We already have the descendant **sketches**, so the
faithful version weights down-steps by **mass below** rather than count:

```
down/up odds ∝ (mass_below / mass_above)^γ        # mass_below from |desc(child)| / μ-mass (sketches)
```

(the up-side normalization needs care — "above" is the rest of the graph; bound it to the parent's subtree
to keep it local). Same `γ`, better branching measure. `E[c²]/E[p²]` and `c/p` are the count-based special
cases; this is the mass-based generalization.

## Mix unidirectional + bidirectional (diversity)

The two walk modes sample **different relations**, so mixing them enriches the training distribution:

- **`--child-only` (downward):** stays inside the seed's subtree → in-domain ancestor→descendant pairs,
  deep coherent structure, zero domain drift (can't reach siblings). Cross-domain pairs come from pairing
  two roots' downward walks (see the Physics+Chemistry task).
- **bidirectional depth-balanced:** reaches *siblings and cousins* (up to a shared parent, back down) at a
  representative depth → the lateral, same-level relations a downward walk never sees (e.g. Physics↔
  Chemistry via `Physical_sciences`), without drifting deep or into the apex.

So **sample a mix** (e.g. a `--bidir-frac` knob): downward walks for vertical/in-domain structure,
depth-balanced bidirectional for lateral/cross-domain structure. The graded μ labels then cover both axes.

## Orthogonal: keep hub-down-weighting

Depth-balance (up vs down) is independent of the existing **hub-down-weighting** (`1/deg^β` *within* a
direction, to avoid stepping into leak-conduit apexes — §"real-data" cone-purity). They compose: balance
up/down, and *within* "down" still down-weight hub children. Depth-balance fixes *depth* drift; hub-down-
weighting fixes *domain* drift; you want both. (Bridge-aware seeding, philosophy use case 6, is the sharper
domain-drift fix later.)

## Implementation & validation

- Add to `gen_mu_pairs.py`: a `--bidir` mode with the **`γ` dial** (`--gamma`, the down/up odds exponent
  `(c/p)^γ`): `γ=0` = coin-flip, `γ=1` = natural undirected, `γ∈(0,1)` = branching-aware. Keep
  `--bidir-mode {count,mass}` to choose `c/p` vs the sketch-based `mass_below/mass_above`, and a
  `--bidir-frac` to mix bidirectional with `--child-only` walks. Keep hub-down-weighting within direction.
- **Validate the depth↔coverage trade on the real 10k graph:** from interior seeds, sweep `γ ∈ {0, 0.25,
  0.5, 1}` and report, per γ: the `depth(endpoint) − depth(seed)` distribution (mean/spread — expect ≈0 and
  tightest at γ=0, skewing deeper as γ→1) **and** a coverage measure (distinct endpoints / how much of the
  big subtrees get hit). The point is to *see the trade*, not to crown one γ. Also report `--child-only`
  (≥0 by construction) as a reference. **No LLM budget** — sampler engineering + measurement.
- **Domain-reach sanity:** confirm bidirectional from `Physics` actually reaches sibling domains
  (`Chemistry`, `Computer_science`) without the endpoint distribution drifting into generic apexes.
