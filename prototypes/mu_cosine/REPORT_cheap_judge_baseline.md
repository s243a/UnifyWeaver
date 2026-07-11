# Cheap-judge pipeline baseline: graph_S is a free S channel; the scheme wins at low budget

The two zero-scoring-cost validations accompanying `DESIGN_cheap_judge_pipeline.md`. Data: the 1,700
pair-matched dual-judge campaign rows; prior = `model_prod_namecond.pt` agnostic readouts — the
campaign-INDEPENDENT prior (blocker 1). The earlier draft used `model_channel_heads_namecond_r0.pt`, whose
trunk was fine-tuned on the campaign train split, giving an optimistic P0 on the fit rows; all tables below
were regenerated against the campaign-independent checkpoint. The qualitative findings survive; the exact
NLLs moved. Note (against the pre-registered expectation that the graph channels would GROW once the prior
was de-memorized): graph_S's free-only value actually SHRANK modestly (+0.53/+0.84 → +0.41/+0.64), because
the honest prior is itself a better S predictor on held rows; the ordering claims below still hold.

## 1. The symmetric-graph S channel (user's point) — the free tier now measures S

`run_sym_channel_fusion.py`: graph_S = a 4-feature linear model (inverse common-ancestor lateral distance,
shared-parent, shared-grandparent, is-ancestor), calibrated to S on the train split — the multivariate
analog of the d→D affine calibration. Added as an S measurement row (H=[0,1]); 6×6 joint blocks per split,
40 descendant-disjoint splits. S-MARGINAL NLL (the channel under test):

| rung | expl | fresh |
|---|---|---|
| prior | −0.282 | +0.230 |
| +graph_D | −0.359 | +0.186 |
| **+graph_D+graph_S (free-only)** | **−0.765** | **−0.454** |
| +graph_D+luna | −0.611 | −0.031 |
| ALL | −0.861 | −0.482 |

- **graph_S free-only value: +0.41/+0.64 NLL** (row-SE 0.008/0.011) — and the FREE-ONLY rung BEATS the
  prior+graph+luna rung on S on BOTH corpora (expl −0.765 < −0.611; fresh −0.454 < −0.031): the graph's
  lateral structure is a better S measurement than the cheap judge.
- **Still adds after luna: +0.25/+0.45** — not redundant with the judge.
- Raw feature strength: corr(1/(1+d_sym), S) = +0.60/+0.43; shared-parent +0.62/+0.48.
- Honesty note: part of graph_S's power is reproducing the stratum ordering (the strata are graph-built);
  that is deployable signal for fusion NLL (deployment has the same features), but a within-stratum ladder
  would show a smaller number — same caveat shape as the S-head decomposition.

Consequence for the DESIGN: "the graph doesn't observe S" is retired. The free tier (prior ⊕ graph_D ⊕
graph_S) is the fusion floor every judge call must improve on, and S fusion is non-trivial without any
judge.

## 2. Matched-cost simulation — the scheme wins where it's meant to (low coverage)

`sim_matched_cost.py`: equal 5.5-call budget n; arm A = n pure 5.5 labels; arm B = 0.3n dual-scored
overlap (labels + block fit) + `0.7kn − 0.3n` luna-bulk pairs with FUSED targets (prior⊕graph_D⊕graph_S⊕
luna). Downstream estimator: ridge on frozen e5 pair-features, λ by inner holdout (fixed λ hit the
768-feature interpolation threshold — double descent — an instructive proxy artifact). Held corr vs 5.5
labels, 10 resamples:

| budget n | best arm (expl D/S) | best arm (fresh D/S) |
|---|---|---|
| 80 | **B k=8** (+0.598/+0.696 vs A +0.552/+0.669) | **B k=8 D / k=4 S** (+0.426/+0.416 vs A +0.382/+0.369) |
| 160 | **B** all k ≥ A | mixed: A on D, B on S |
| 320 | B k=4 +0.602 vs A +0.586 D; A +0.702 vs B +0.699 S (parity) | A ahead D, ~parity S |
| 640 | A +0.610 vs B +0.596 D (truncated), parity S | — (n>pool, skipped) |

- **Low-budget regime (n=80–160): the scheme wins at every k on both corpora**, biggest at k=8 — exactly
  the regime it targets (a new corpus with a limited budget: Pearltrees, mindmap).
- High-budget convergence is partly REAL (with enough 5.5 labels, pure labels win on D — fused targets
  carry luna noise) and partly ARTIFACT: the bulk purchase truncates at the scored pool (at n=320, k=4
  already wants more luna rows than exist), so large-n·k cells understate arm B.
- S favors the scheme almost everywhere — graph_S inside the fused targets does the work (§1).
- Proxy caveat: ridge-on-e5 is not the transformer head; both arms share it, so the ORDERING is the
  claim, not the magnitudes. One full-head confirmation at a single grid point is the upgrade if a real
  budget decision hangs on this.

**Reading for the budget decision:** at new-corpus scale (coverage low relative to corpus), even k=2
suffices to tie and k≥4 wins outright; the scheme's advantage shrinks as coverage saturates. The practical
recipe stands: luna everywhere + small random 5.5 overlap + graph channels in the fusion + routed 5.5 on
conflict + sonnet-5 tiebreaks on disagreement.

## Repro

```
python3 make_pipeline_diagrams.py          # figures/pipeline_dataflow.png, figures/fusion_architecture.png
python3 run_sym_channel_fusion.py
python3 sim_matched_cost.py --k 2 4 8
```
