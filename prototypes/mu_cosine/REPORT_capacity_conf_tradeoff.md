# 3 layers / 0.9 conf (memorize) vs 4 layers / 0.7 conf (principled) — a parity, trading in the predicted direction

Two deliberately-contrasted configurations, per the design dialogue:
- **3L / `c=0.9` — "memorize":** less regularization (operator commits ~90% to the label), 3 layers — spend
  capacity on *fitting*.
- **4L / `c=0.7` — "principled":** more regularization (heavier operator spread) *plus* the extra capacity
  to absorb it — the one-standard-error / AIC-flavoured choice (prefer the more-regularized model when fit is
  within noise).

Same harness as `REPORT_tagged_blend_sweep.md` (fuzzy/Complex round, warm-start `model_nodetype.pt`, 700
steps, 3 seeds). **Caveat:** the 4-layer's 4th layer is **random-init** (the warm-start only covers the 3
shared layers + heads) and fine-tunes for the same 700 steps — so the 4L config is mildly *handicapped* vs a
from-scratch 4-layer base.

## Result

| config | discrimination (3 seeds) | **disc mean** | SYM corr (3 seeds) | **SYM mean** |
|---|---|---|---|---|
| 3L / 0.9 (memorize) | 96 / 100 / 100% | **98.7%** | .678 / .621 / .726 | **+0.675** |
| 4L / 0.7 (principled) | 96 / 100 / 96% | **97.3%** | .672 / .630 / .765 | **+0.689** |

## Verdict — parity, with the small differences pointing exactly where the theory says
The two are **at parity** (both differences are ~1 probe / ~0.01 corr — within seed noise). But the *direction*
of the tiny gaps is the interesting part, and it matches the framing cleanly:

- **3L / 0.9 edges the discrimination probe** (98.7 vs 97.3%) — the more in-distribution "did it fit the
  domain structure" metric. Less regularization → marginally better *fit*. The "memorize" rationale.
- **4L / 0.7 edges the held-out SYM correlation** (+0.689 vs +0.675) — the more out-of-sample generalization
  metric (4L higher in 2 of 3 seeds). More regularization + capacity → marginally better *generalization*.
  The "principled" rationale — **and it achieves this despite the random-4th-layer handicap**, which makes
  the SYM edge mildly more notable than the number alone.

So neither dominates; they trade fit (disc) for generalization (SYM) by a hair, in the predicted directions.

## What this says about "do we need the 4th layer?"
On **this** data: **no.** The 4th layer does not unlock a clear win — at best it matches 3 layers (with a
handicap), buying a sliver of held-out generalization at a sliver of in-distribution fit. The capacity/AIC
argument for it only bites when we genuinely need the **0.5–0.7 heavy-regularization regime** (low-confidence
or heterogeneously-curated data, where the 3-layer model *underfit* in the sweep). This well-labeled
systems-theory probe isn't that regime, so 3 layers + moderate `c` remains the operating point. The 4th
layer is *banked* for the harder data, with this run confirming it at least doesn't hurt.

## Limitation (for a cleaner memorize-vs-generalize read)
Discrimination-vs-SYM is only a *proxy* for memorize-vs-generalize. The direct measure is the **train-vs-
held-out graded-fit gap** — which we did not capture here. If we want to *confirm* "3L/0.9 memorizes more,"
the next cheap addition is a train-set graded-MSE readout alongside the held-out one; the hypothesis predicts
3L/0.9 fits train tighter while 4L/0.7 closes the gap. Without it, the disc/SYM split is suggestive, not
conclusive. (And, as always: single-seed on the 25-probe metric is untrustworthy; even at 3 seeds these gaps
are within noise.)
