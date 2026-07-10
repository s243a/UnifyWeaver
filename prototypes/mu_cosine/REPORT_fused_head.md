# Fused head (B2 step 2): mechanism works, value is a theory-consistent null on a single reliable judge

`fine_tune_fused_head.py` — the DESIGN_amortized_fusion_heads three-way learn's `mu_PoE`, distilling the
Lever-A Kalman posteriors into a name-conditioned readout on the campaign data. Single seed, champion recipe
(800 steps, lr 5e-4, last layer + readout + name pathway, agnostic-anchor honesty).

## 1. First consumer of the name architecture — a new head with NO new learned row

`kalman-fused` (JUDGES index 10) onboarded purely by NAME: card "composite fused judge, kalman filter
posterior of graph walk and LLM judge measurement channels", conditioning = `W·e5(card)` at r=0. The head
trained normally from that prior — the §6 onboarding story works for function-type identities, not just
LLM judges. (gpt-5.6-luna took index 9 at the same time; both ride `load_expanded`'s name-prior expansion.)

## 2. Target construction (honesty choices)

Posterior per campaign pair: prior = frozen base agnostic readouts (μ_D, μ_S); measurements = [calibrated
graph d → D, judge D, judge S]; prior/graph error blocks fit on the TRAIN split vs dequantized labels;
judge channel priced by the MEASURED self-consistency `R_judge` (D 0.0043, S 0.0041 — the campaign has one
judge run, so fitting R_judge in-sample would return the degenerate 0). Judge-error cross-correlations set
to 0 (unmeasurable without a second run on these pairs; Lever A showed correlation pricing moves error bars
far more than means, and the mean is what's distilled).

## 3. Results (held-out, within-stratum)

Distillation fidelity is fine (fused vs posterior, within: D +0.541/+0.468, S +0.399/+0.288 expl/fresh).
The deployment question — does the fused head beat the LLM head against labels? **No:**

| within-stratum vs labels | expl D | expl S | fresh D | fresh S |
|---|---|---|---|---|
| fused head (cw 0.25) | +0.522 | +0.337 | +0.447 | +0.230 |
| llm head (same ckpt) | +0.550 | +0.324 | +0.445 | +0.193 |
| fused head (cw 0)    | +0.501 | +0.301 | +0.391 | +0.221 |
| llm head (same ckpt) | +0.505 | +0.295 | +0.402 | +0.213 |

- **Conflict slice** (top-quartile |graph − prior_D|, the rows where fusion could differ): fused ≈ llm
  there too (expl +0.582 vs +0.631; fresh +0.327 vs +0.325).
- The apparent S edge at cw 0.25 vanishes at cw 0 (gaps +0.006/+0.008) — it was the consistency prior's
  shrinkage acting as a shared-trunk regularizer, not fusion value.

## 4. Why the null is the theory speaking, not a failure

With measured `R_judge ≈ 0.004` against prior/graph error variances ~0.05–0.1, the Kalman gain puts ~95%
of the weight on the judge: **the posterior sits mean|Δ| ≈ 0.02 off the label.** Distilling this posterior
is re-distilling the label with a whisper of graph/prior shrinkage — so the fused head necessarily learns
(and measured: learns exactly) the LLM head's function. Same shape as Lever A's NLL-routing null: the
value of fusion machinery is bounded by how much the fusion CAN deviate from its dominant channel, and a
judge this reliable leaves no room.

**Where the fused head earns its keep (the boundary of the null):**
1. **Noisy judges** — luna's R is ~5–10× 5.5's (S corr 0.35–0.44 vs 0.766 ceiling): a luna-measured fusion
   has real graph/prior weight. Step 3's onboarding produces exactly this setting.
2. **Multi-judge conflict** — two judges with opposite biases (luna +D/−S vs 5.5) make the fusion
   non-degenerate; requires per-row multi-judge scoring (exists only on the fresh 250 so far).
3. **The slow timescale as designed** — the two-timescale split stands: with a single reliable judge, the
   slow-timescale distillation loop ≡ ordinary channel training (which the campaign already does); the
   explicit filter remains the fast-timescale tool when measurements arrive.

## 5. Caveats & repro

Single seed; one target construction (constant blocks, no Σ(hop) — campaign lateral strata carry no hop);
independent-judge-channel approximation; `dequant` interior clipping.

```
python3 fine_tune_fused_head.py --ckpt model_channel_heads_namecond_r0.pt --steps 800 --lr 5e-4
python3 fine_tune_fused_head.py --consistency-weight 0 --out model_fused_head_cw0.pt
python3 fine_tune_fused_head.py --eval-only          # conflict slice off the saved ckpt
```
