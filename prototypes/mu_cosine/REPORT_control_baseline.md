# Control baseline — symmetric MiniLM μ-encoder (#3287)

The symmetric MiniLM pairwise encoder (`train_cosine_mu_torch.py --mode pairs --minilm`, merged in
#3287) validated as the **control arm** for `DESIGN_directional_attention.md`. These are the numbers the
directional (e5 multi-operator) model must beat to justify its extra machinery. **No LLM budget spent** —
all analysis runs on the existing maps + the faithful `gated_ic`/`lin_from_ic` port.

Reproduce:
```bash
python3 train_cosine_mu_torch.py --mode pairs --pairs mu_pairs_scored.tsv --minilm \
    --holdout 0.2 --weight-decay 1.0 --epochs 2000 --lr 0.01 --save-encoder control.pt
python3 validate_lin_agreement.py --encoder control.pt
python3 emit_dense_mu.py --encoder control.pt --out dense_mu_control.tsv
python3 dense_mu_direct.py --model e5 --out dense_mu_e5.tsv          # proxy map for the flip test
python3 validate_control_baseline.py --encoder control.pt \
    --control-map dense_mu_control.tsv --proxy-map dense_mu_e5.tsv
```

## Control numbers

| metric | value |
|---|---|
| held-out pairwise-μ corr (40 held-out positives) | **+0.726** (MSE 0.065) |
| lin-agreement Pearson / Spearman (1275 pairs) | **+0.098 / +0.113** |
| lin-agreement on non-saturated pairs (the real signal) | **+0.124** (42 pairs) |
| lin saturation fraction | **96.7%** (1233/1275 pinned at 1.0) |
| decision-flip rate, band vs confident | **40% vs 4.0%** (proxy); 28% (5/18) ground-truth |
| cold-start OOD gate-leak (dist ≥ 5, never trained) | **1.1%** (49/4280, μ̄ 0.043) |
| gate-leak (5-probe, from #3287) | **0/5** |

## Q1 — `lin_from_ic` saturation ceiling

**96.7% (1233/1275)** of scored-physics-node pairs saturate at `Lin ≥ 0.999`. Cause (a **graph/gating
property, not the μ map**): `Lin = min(2·IC(MICA)/(IC(u)+IC(v)), 1)`, and **all 1233** saturated pairs
have `2·IC(MICA) ≥ IC(u)+IC(v)` — 1224 have an *un-clamped* `Lin > 1` (median **1.39**). This is because
**μ-gating prunes ancestor cones**: a common ancestor reachable only through low-μ (out-of-domain)
connectors has a tiny *gated* descendant cone and therefore a **high IC — often higher than the nodes it
sits above** (e.g. `Temperature`/`Fire`: IC 3.07, 5.24 but `MICA` IC = 6.24). IC is non-monotone up the
gated DAG (the same in-domain-leak pruning the Rust core does on purpose), so the Lin ratio overshoots 1
and clamps.

**Ceiling / headroom:** graph-Lin carries gradable signal on only **42/1275 (3.3%)** of pairs; the rest
are pinned at 1.0 and add only label-noise to the Pearson. **lin-agreement is a low-resolution metric** —
the directional model's real headroom is those 42 non-saturated pairs (control already **+0.124** there),
not the global `r`. *Conclusion: do not over-weight the global lin-agreement Pearson; it is structurally
capped low and the per-pair signal is thin.*

## Q2 — decision-flip rate (active-learning premise)

**360** categories sit in the control decision band `μ ∈ [0.2, 0.45]` (of 8 247). How many would a better
membership signal flip across the 0.3 gate?

- **(a) ground truth** vs the 90-node Haiku fixture: 18 band nodes are fixture-scored; **5 (28%) flip**.
- **(b) proxy** vs an independent map (e5-direct): **145/360 (40%) flip** — vs only **4.0%** in the
  confident region (μ outside the band).

The band flips **~10× more** than the confident region, confirming it is the genuinely-uncertain set —
**the active-learning premise holds**: a budget-bounded boundary rescore targeted at this band is well
spent (≈ a third of it changes the gate decision), whereas rescoring the confident region would be waste.

## Q3 — cold-start coverage (out-of-domain, never trained)

The shared encoder is frozen-MiniLM + learned blocks, so all 8 247 nodes are scored; only ~1 225 names
were ever in training. On unseen nodes:

- **OOD rejection:** of **4 280** nodes far from Physics (graph dist ≥ 5) and never trained, only
  **1.1% (49) leak** the 0.3 gate (μ̄ 0.043, max 0.67) — clean rejection.
- **Unseen in-domain generalisation:** the 9 never-trained nodes at dist ≤ 2 are scored by **semantics,
  not graph proximity** — physics-adjacent fields pass (Mechanics 0.75, Electricity 0.67, Chemistry 0.57,
  Academic_disciplines 0.51) while generic apex neighbours are correctly rejected (Geography 0.13,
  Everyday_life 0.08, Past/Chronology/Future ≈ 0).
- **No memorisation:** at *matched* graph distance, trained-vs-unseen gate-pass is identical for far
  nodes (**1% vs 1%**, n=4 280). The shared encoder generalises; it is not a per-node lookup. (The near
  d≤2 sample is only 9 unseen nodes — too small for a parity claim, but the per-node scores above show
  correct discrimination.)

## What the directional model must beat

`SYM`/relatedness and **gate-leak** are the head-to-head metrics (per the design's "control arm"). The
control sets: **gate-leak 0/5 (probe) and 1.1% (4 280-node OOD)**, **held-out pairwise-μ +0.726**, and
**non-saturated lin-agreement +0.124**. Two cautions the directional track inherits: (1) the global
lin-agreement Pearson is a **low-resolution metric** (96.7% saturated) — judge it on the non-saturated
pairs; (2) the **decision band is real** (40% flip) — that is where the budgeted `LLM_*` boundary labels
should go.
