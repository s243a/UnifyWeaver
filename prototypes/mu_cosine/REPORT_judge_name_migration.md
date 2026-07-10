# Judge-name migration: behavior preserved bit-close, parity retrain, transfer test passes (B2 step 1)

The REPORT_channel_campaign §6 migration, built and validated. Design precedent (user): the function/operator
input already received this mitigation — `anchored_basis.py`'s frozen e5-seeded values + learnable residual,
with a constraint pinning the known part. The judge pathway is that idiom applied to provenance:

```
cond_j = W · e5(card_j) + r_j          (NameFunctionCond, mu_attention.py)
```

frozen card embedding (the pinned interpretable part, judge_cards.py) + learned translation W (amplifies the
calibration-relevant axes of name space) + per-judge residual (only what the card doesn't say). The
behavior-preserving init plays the anchor-KL's anti-drift role: `migrate_judge_names.py` fits W by ridge
least squares to the informative judge_emb rows and sets `r_j = judge_emb[j] − W·e_j`.

## 1. Migration is behavior-preserving (acceptance §6.6, init half)

| checkpoint | fit rows | reconstruction max err | end-to-end forward max err |
|---|---|---|---|
| model_prod.pt | 5/9 | 2.24e-08 | 1.19e-07 |
| model_channel_heads_campaign.pt | 6/9 (incl. gpt-5.5-low, graph) | 2.24e-08 | 5.96e-08 |

Forward check = an all-judges synthetic provenance batch through old vs new model (catches wiring bugs the
row-level check can't). Ridge (default 0.1) matters because the cards are collinear (cosines ≳0.83); the
residual absorbs the ridge bias so reproduction stays exact while W extrapolates smoothly to unseen names.

**Diagnostic — the card geometry already explains most of the learned conditioning:** on the campaign
checkpoint the TRAINED rows' residuals are small relative to their name priors (gpt-5.5-low ‖r‖ 0.17 vs
‖W·e‖ 0.28; graph 0.24 vs 0.47). The name function isn't a re-parameterization trick; it captures real
structure in what training learned.

## 2. Retrain parity (within-stratum decomposition, eval_within_stratum.py)

Champion recipe from the migrated base (`--ckpt model_prod_namecond.pt --data campaign --steps 800
--lr 5e-4 --unfreeze-last`), single seed like the champion. WITHIN-stratum held-out corr (the honest number):

| | expl D | expl S | fresh D | fresh S |
|---|---|---|---|---|
| indexed champion (model_channel_heads_campaign.pt) | +0.530 | +0.355 | +0.506 | +0.306 |
| name-cond, resid-weight 1e-2 | +0.521 | +0.331 | +0.513 | **+0.224** |
| name-cond, resid-weight 0 | +0.547 | +0.345 | +0.506 | +0.279 |

(eval_within_stratum.py reproduces the §8 champion numbers exactly — method-identical comparison.)

**Verdict: parity at resid-weight 0** (all four cells within single-seed noise of the champion). The 1e-2
residual regularizer costs the scarcest cell (fresh S within −0.08) and buys nothing in transfer (§3) —
default changed to 0. The name-prior-as-default property for NEW judges is preserved by construction
(they onboard at r = 0 regardless).

## 3. The transfer test (acceptance §6.6, trained half) — PASSES on S, the channel it exists for

Held-out judge: gpt-5.6-luna — never trained, not in JUDGES. Labels: its own validation run on the fresh
Behavior 250 (excluded from campaign training). Conditioning the trained name-cond model three ways,
correlated against luna's labels:

| conditioning | D | S |
|---|---|---|
| zero-row (the old onboarding) | +0.569 | **−0.100** |
| name prior (r = 0, cond = W·e5(luna card)) | +0.544 | **+0.055** |
| gpt-5.5-low's own row (borrow-the-family-row reference) | +0.570 | +0.047 |

(resid-weight-0 checkpoint: +0.581/−0.084, +0.569/+0.050, +0.588/+0.041 — same picture.)

- **S**: the zero row is anti-correlated; the name prior flips it positive and matches the family-row
  reference — the name prior recovers essentially ALL of what borrowing 5.5's row would give, with no
  index assignment, no contamination of 5.5's calibration, and a residual slot ready to learn luna's
  measured +D/−S tilt. This is the §5 disposition gap, closed.
- **D**: all three tie — D routes barely by judge (D was never the bottleneck), consistent with the probe.
- Honest bounds on the S magnitude: the test set is all-transitive (the S-starved stratum; campaign
  within-stratum transitive S is only +0.13–0.18) and luna's S agreement with 5.5 caps at 0.35–0.44. The
  +0.05-vs-−0.10 direction is the signal, not the absolute size; a lateral-strata luna set would raise the
  ceiling (step 3 material).

## 3b. Luna onboarding (B2 step 3) — residual captures the tilt, touches nothing else

`fine_tune_luna_resid.py`: luna at JUDGES index 9, residual-only training (lr 1e-2, 300 steps) on its 250
labelled fresh pairs (175 train / 75 held, descendant-disjoint).

| | D corr (held) | S corr (held) | tilt μ_luna − μ_5.5 |
|---|---|---|---|
| r=0 name prior (before) | +0.662 | +0.052 | — |
| after residual training | +0.665 | +0.050 | D **+0.174** (expect +), S **−0.107** (expect −, measured −0.11..−0.13) |

- **Isolation verified**: max residual drift on every other judge = 0.00e+00 (gradient flow — only luna's
  row appears in batches; W and trunk frozen).
- The residual learned exactly what §7 measured about luna: the +D/−S calibration tilt. Correlations don't
  move because the name prior already put luna at its family ceiling on this slice, and a bias offset is
  correlation-invariant — the residual's job here IS the bias.
- Bounds: the 250 pairs are all-transitive (S variance tiny — the campaign's B1 lesson applies to the S
  number); luna's residual norm grew to 4.05 with a mid-training dip (single seed, no early stop) — a
  deployment-grade luna row wants the stratified-data treatment and light early stopping.

## 4. Status & handoff

- B2 step 1 CLOSED: `model_channel_heads_namecond_r0.pt` is the recommended working checkpoint (parity
  metrics + name architecture) for step 2 (fused head distilling the Lever-A Kalman posteriors).
- Luna onboarding (step 3) is now mechanical: add `"gpt-5.6-luna": 9` to JUDGES (card already in
  judge_cards.py), load via load_expanded → r = 0 name prior; train only its residual on luna-labelled
  rows to capture the measured +D/−S tilt.
- The same NameFunctionCond + migrate procedure applies verbatim to OPS/CORPORA when their turn comes.

Repro: `python3 migrate_judge_names.py --ckpt model_prod.pt` →
`python3 fine_tune_channel_heads.py --ckpt model_prod_namecond.pt --data campaign --steps 800 --lr 5e-4
--unfreeze-last --out model_channel_heads_namecond_r0.pt` →
`python3 eval_within_stratum.py --ckpt ...` / `python3 eval_luna_transfer.py --ckpt ...`
