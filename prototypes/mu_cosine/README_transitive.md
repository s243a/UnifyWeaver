# Transitive ordinal constraints — methods & usage

Teach the μ model **transitive decay** (`μ(A→C) ≤ min(links)`) from clean graph structure, as a soft ordinal
constraint. Theory: [`DESIGN_transitive_relations.md`](DESIGN_transitive_relations.md). Verified results:
[`REPORT_transitive_verification.md`](REPORT_transitive_verification.md) (generalises + no-collapse + survives
convergence, replicated 2-seed on a leakage-aware holdout).

## Pipeline (3 stages)

### 1. Generate — `transitive_closure.py` (pure graph code, no LLM)
Compose **tagged hierarchical** edges into transitive pairs, ranked by **product of link μ** (= highest-product
path = Dijkstra on `−log μ`; max-product dominant path — DESIGN §"Generation order", §"Multi-path").
```
python3 transitive_closure.py --edges context/*_edges.tsv --max-hops 3 --out triples.tsv
```
Output triples: `trans_src trans_dst trans_rel | bound_src bound_dst bound_rel | product min_link hops`.
The **bound edge** = the weakest link = the ordinal bound (`μ_trans ≤ μ_bound`, i.e. "transitive drop ≥ max
pairwise drop" — DESIGN §"is the bound the product? No").

### 2. Train with the ordinal constraint — `train_mu_attention.py --transitive`
Adds `L_trans = softplus(−s·(μ_bound − μ_trans − m))` = `−log σ(s·Δ)`, the ranking CE (DESIGN §"The loss",
§"statistical not logical"). **A Lagrangian relaxation** of the inequality — the weight is the multiplier λ.
Two ways to set λ:

**(a) Fixed λ** — the naive first cut (hand-set multiplier):
```
... --transitive triples.tsv --transitive-weight 1.0 [--transitive-margin 0.05 --transitive-scale 10]
```

**(b) Dual-ascent λ (recommended)** — adapt λ to a *target satisfaction rate* (DESIGN §"multi-factor",
loss-note "adaptive dual-ascent"): `λ ← clamp(λ + lr·(target − sat), 0, max)`. Removes the hand-tuned weight;
λ self-tunes to the *minimal* value that hits the target (decreases when over-satisfied).
```
... --transitive triples.tsv --transitive-weight 1.0 --transitive-target-sat 0.92 \
    [--transitive-lambda-lr 0.1 --transitive-lambda-max 20]
```

**Scale: heteroscedastic (DEFAULT) vs homoscedastic** (orthogonal to λ — this sets the logistic *sharpness*,
not the weight). The default is **per-pair** `s_pair = s/√(1+V)` (more principled, ~free — `V` is precomputed
in the triples, no extra forwards). Pass **`--transitive-homo`** to force the global-`s` approximation (e.g.
for an A/B). `V` is the **product-propagated chain variance** carried in the triples:
`V = Σ_links (1−μ)/μ` — the textbook product error-propagation (relative variances add; additive in
log-variance, the dual of the log-μ chaining). **Longer / weaker chains → larger `V` → softer constraint** —
which the global-`s` form cannot express. DESIGN §"The loss must be over the predicted DISTRIBUTION".

### 3. Evaluate — `--eval-transitive` (leakage-aware)
On a **held-out node-split** (hold out destination nodes so held-out pairs share no endpoints with training —
DESIGN §"Eval ... leakage-aware split"):
```
... --eval-transitive holdout_triples.tsv
```
Reports **constraint satisfaction** (`μ_trans ≤ μ_bound`) **and the anti-collapse / level guard** (mean
`μ_bound` must stay HIGH — not gamed by μ→0 — `μ_trans` below it). Both are needed: satisfaction alone is
gamed by collapse (DESIGN §"Eval ... anti-collapse").

**Leakage-aware split** (node-based, deterministic):
```
python3 transitive_closure.py --edges context/*_edges.tsv --out all.tsv
python3 - <<'PY'
import random
rows=[l.rstrip() for l in open('all.tsv') if not l.startswith('#')]; hdr=open('all.tsv').readline().rstrip()
nodes={n for l in rows for n in l.split('\t')[:2]}
H=set(random.Random(7).sample(sorted(nodes), int(0.2*len(nodes))))
tr=[l for l in rows if l.split('\t')[0] not in H and l.split('\t')[1] not in H]
ho=[l for l in rows if l.split('\t')[1] in H]
open('trans_train.tsv','w').write(hdr+'\n'+'\n'.join(tr)+'\n'); open('trans_hold.tsv','w').write(hdr+'\n'+'\n'.join(ho)+'\n')
PY
```

## Knobs
| flag | meaning | default |
|---|---|---|
| `--transitive PATH` | triples file (stage 1 `--out`) | — |
| `--transitive-weight` | fixed-λ multiplier (also the dual-ascent init) | 0.0 (off) |
| `--transitive-margin` | `m`: enforce `μ_bound − μ_trans ≥ m` | 0.05 |
| `--transitive-scale` | logistic `s` (global confidence; homoscedastic) | 10.0 |
| `--transitive-homo` | force homoscedastic global `s` (default is heteroscedastic per-pair `s/√(1+V)`) | off (→ hetero) |
| `--transitive-target-sat` | dual-ascent target satisfaction (0=fixed-λ) | 0.0 |
| `--transitive-lambda-lr` / `-max` | dual-ascent step / cap | 0.1 / 20 |
| `--eval-transitive PATH` | held-out triples → satisfaction + anti-collapse | — |

## Deferred methods (proposed, not built — see DESIGN open questions)
- **Heteroscedastic via superposition variance** — the design's *original* variance source (per-pair
  `Var[μ]` from the operator-superposition, needing R hard-cell forwards / MC). **Built instead:** the cheaper,
  cleaner **product-propagated chain variance** (the default, above) — structural, no extra forwards.
  The superposition-variance variant remains an alternative if a per-pair (non-chain) uncertainty is wanted.
- **Noisy-OR multi-path** — reinforcement when multiple paths exist; NOT a semiring closure (needs path
  enumeration), unlike the `max` default (DESIGN §"Multi-path: semiring closure vs path enumeration").
- **LLM-anchored multi-factor `μ_bound`** — a judge term anchoring the absolute bound, weighted by *inter-judge
  agreement* (not single-judge confidence — measured caveat) (DESIGN §"multi-factor loss", §"Measured caveat").
- **Product soft floor** — band `[floor, μ_bound − m]` to prevent over-decay (DESIGN §"too low risk").
