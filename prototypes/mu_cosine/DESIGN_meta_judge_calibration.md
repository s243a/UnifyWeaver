# Meta-judge: CE-calibrated μ as the Kalman training signal (two-timescale)

User proposal (2026-07-17), pinned after clarification. The idea: **train a judge by cross-entropy
on each batch to CALIBRATE the μ training signal, then use that calibrated μ to optimize the Kalman
filter's fused μ estimate.** The judge produces μ (it is not a separate ranking head); CE calibrates
that μ. At least one model the judge sees is NOT a Kalman channel (held-out sonnet-5) — that is what
makes it meta-learning ("learning how to learn" the combination, not memorizing the fusion inputs).
At INFERENCE there is no judge in the loop: folders are ranked by the Kalman-fused μ.

## Why calibration, not lineage injection

REPORT_pearltrees_candidate_lineage.md established that Pearltrees folder TITLES already carry the
lineage (a child's e5 embedding sits closer to its true parent, 0.877, than to a random folder,
0.816; explicit lineage only helped on unseen folders). So the μ ranker's weakness is not missing
structure — it is mis-CALIBRATED μ: the diagnostic (§2 there) showed e5 wins only the easy FAR
third and everyone is weak on the confusable NEAR two-thirds. The meta-judge targets exactly that:
calibrate μ so the ranking is right on the hard cases, using a stronger held-out judge as the
reference the fusion itself cannot supply.

## Objects

- **Candidate-ranking (LINEAGE_RANK) CE.** For a descendant node `d` with true parent folder `f*`,
  a candidate set `C = {f*} ∪ negatives(d)` (K folders sampled from the campaign folder pool,
  excluding d's own ancestors). The judge score is the model's directional μ `s(d,c) = μ(d | c, HIER,
  pearltrees, JUDGE)`. Loss = candidate-softmax cross-entropy `−log softmax_c[ s(d,c)/τ ]_{f*}`.
  This calibrates the μ so the true parent ranks first — the exact filing objective.
- **Held-out sonnet calibration anchor.** Sonnet-5 (subagent-scored, NOT a Kalman channel) supplies
  a held-out directional D on the overlap TRUE pairs. On those rows add
  `λ_cal · (s(d,f*) − D_sonnet(d,f*))²` — anchors the μ MAGNITUDE to an independent strong judge so
  the scale is not set circularly by the fusion's own channels (luna/graph). Sonnet-being-held-out
  is the meta hook.
- **Kalman inner estimate.** The existing fusion: prior ⊕ graph_D ⊕ graph_S ⊕ luna → fused (D,S)
  posterior, blocks (P0,R,C) refit per batch on the overlap-train rows. Sonnet is NOT a channel.
  The calibrated judge μ becomes the distillation target the μ heads are trained toward (replacing /
  augmenting the raw fused target), so the calibration flows into what the Kalman μ reads at
  inference.

## Two-timescale loop (per epoch over batches)

**Implemented mechanism, stated literally (audit finding 4).** The build below is aspirational in
two places the implementation simplifies: (i) the Kalman fused targets are PRECOMPUTED once by
run_pearltrees_fusion.py and held FIXED during training (the "inner Kalman refit per batch" is not
in the loop — the fast step is MSE distillation toward static fused targets); (ii) the slow step
updates ONLY the judge-name residual row via the SNR-gated manual update — no other meta-parameters.
Also: the ranking examples are h1–h5 ancestor pairs, so the CE trains ANCESTOR-ranking, not
exact-parent recovery (finding 3); exact-parent labels would need h1-only examples (too few at this
campaign size) or the harvested tree ground truth.

    for batch b:
        # INNER (fast): refit the Kalman blocks on this batch's overlap-train rows; compute the
        # fused μ posterior; one gradient step on the μ heads + trunk toward the fused targets
        # (the existing fine_tune step) — this is the μ-estimate optimization.
        fast_step(model, batch)                      # μ heads + last layer, lr_fast

        # OUTER (slow): every K batches, one candidate-ranking CE step (+ sonnet anchor) that
        # calibrates the judge μ; its gradient flows into the shared trunk so the calibration
        # persists into the Kalman μ. Slow = small lr / every-K so the judge drifts under the
        # faster inner estimate (the metastable-drift / amortized-fusion-heads timescale split).
        if b % K == 0:
            slow_step(model, ranking_batch, sonnet)  # candidate-softmax CE + λ_cal anchor, lr_slow

Timescale split (DESIGN_amortized_fusion_heads.md): the judge is the slow meta-parameter (the
calibration of μ); the Kalman blocks + μ heads are the fast inner estimator. IMPORTANT (added after
implementation): do NOT impose the split with `lr_slow ≪ lr_fast` or a `K`-batch cadence. The judge
uses a PRECISION/SNR-GATED update so the slow timescale EMERGES from the (low) information in the
quantization-noisy ranking CE — Adam and plain SGD both destroy this and silently reintroduce a
hand-tuned timescale. The full derivation is THEORY_emergent_timescale_learning.md; the fast μ heads
keep Adam (their MSE signal is high-SNR, so gating buys nothing there). Anchor loss (agnostic
readouts pinned to the base) stays on throughout, so untrained identities do not drift (the Filing v1
finding-6 lesson; checkpoint remains Pearltrees-only).

## Evaluation

- Primary: the FILING metric via the Kalman-fused μ ranker (eval_pearltrees_filing conditioned
  heads), meta-judge checkpoint vs the no-meta Filing v1 checkpoint vs e5-cos. Report acc@1/@5/MRR.
- **Focus on the confusable NEAR stratum** (diagnose_filing_e5_vs_mu.py) — the meta-judge's thesis
  is that calibration helps exactly where e5 is weak; report the stratified NEAR/MID/FAR table.
- Controls: (i) no sonnet anchor (CE-only) vs full — isolates the held-out judge's contribution;
  (ii) sonnet-in-the-Kalman (as a channel) vs held-out — tests that holding it out is what matters.
- Honesty: an in-sample calibration eval leaks; the ranking CE and the sonnet anchor are fit on
  train-split rows only, evaluated on the node-disjoint held folders (the Filing v1 split-first
  discipline). A null is reportable.

## Scope guards (out of this build)

- No learned output-metric space (asymmetric zᵀMz, cached folder vectors) — that is the heavier
  alternative if scalar CE-calibration underperforms (recorded in the Filing v1 §7 memory).
- Sonnet scores the 300-row overlap only (the calibration reference set), not the full bulk.
- Single corpus (Pearltrees), partial DAG — same coverage caveats as Filing v1.
