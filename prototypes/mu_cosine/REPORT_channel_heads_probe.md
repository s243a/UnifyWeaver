# Lever B opening probe: the judge tokens don't route — per-channel heads must be trained

*`run_channel_heads_probe.py`, 2026-07-09. Before building `DESIGN_amortized_fusion_heads.md` step 2
(per-channel heads mu_graph, mu_LLM), measure whether the existing checkpoint already carries them: the
architecture has judge-conditioned provenance tokens (`JUDGES`), and `model_prod.pt` was trained with
judge-tagged rows. Probe: query the checkpoint on both corpora under three conditionings and correlate each
readout with each channel's ground truth.*

## Facts found

- `model_prod.pt` carries **5 judge rows** (haiku/graph/human/sonnet/opus) — it predates the gpt-5.5-low,
  blend, and dir-blend rows; no local checkpoint has more than 5. The checkpoint's trained LLM judge is HAIKU.
- **The judge tokens do not route.** All three conditionings (agnostic / judge=graph / judge=haiku) produce
  nearly identical readouts: |Δ| ≈ 0.02–0.07, and the graph-Δ and llm-Δ are CORRELATED (+0.44 to +0.62) — a
  shared shift, not channel differentiation. The graph-conditioned readout predicts the walk no better than
  agnostic (fresh: 0.283 vs 0.277); the llm-conditioned predicts LLM labels no better (0.572 vs 0.567).
  The judge rows learned calibration offsets, not channels. **Step 2 is NOT amortized; it must be trained.**
- **The SYM readout has ~ZERO correlation with the LLM's S channel** (−0.06 to +0.02, both corpora) while
  correlating +0.5 with D. The model's "symmetric" head tracks directionality, not the judge's associative
  signal — the mu_S feature fed to the fusion arcs was never really the S channel (the covariance fits priced
  this correctly, which is why fusion still worked). A trained S head is a MISSING CHANNEL, not a refinement —
  likely the largest single amortization win available.

## Build plan (next PRs)

1. **B1 — per-channel heads:** resize judge_emb 5→9 (copy 5 rows, zero-init the rest), fine-tune from
   model_prod.pt with channel-tagged rows: (pair, HIER, judge=gpt-5.5-low → D), (pair, SYM, judge=gpt-5.5-low
   → S), (pair, HIER, judge=graph → walk hit_prob). Data: the ~1,380 scored pairs + free graph targets.
   Descendant-disjoint eval: per-channel correlation under each conditioning (this probe re-run = the
   acceptance test: routing must appear) + trunk non-degradation on the standard eval.
2. **B2 — fused head:** distill the Lever-A fusion posteriors (judge token = a new "kalman" row) with the
   stop-grad consistency prior + measured anchor recipe from the DESIGN.
3. **B3 — class-mixture predictive** for the D-bimodality defect (acceptance gate in THEORY §11.5).

## The escalation ladder (user, from the Lever-A discussion)

The decision-flip signal generalizes beyond "call the judge": it prices an ESCALATION LADDER — more reasoning
effort, a stronger model, a k-model consensus — each rung a measurement channel with its own (R, cost).
Measured so far: R ≈ 0.004 at gpt-5.5-low; consensus of k gives R/k; higher effort presumably lower R per call.
The Kalman machinery converts each rung's R into expected flip-probability per dollar: inference spend as a
portfolio with computable marginals. Training-data generation for B1/B2 should use it (spend scoring budget
where flips concentrate — the conflict policy).

## Repro

```
python3 run_channel_heads_probe.py
```
