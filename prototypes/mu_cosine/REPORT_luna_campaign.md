# Luna campaign: the §7 verdict revised, stratified fusion value, and the fused head's first win

Stratified luna scoring (same 2,000 campaign pairs as the 5.5 run; 200/200 batches, 1,930 ingested, 0
failures, 111 min, ~0.30 pairs/s at gpt-5.6-luna low) → 1,700 pair-matched dual-judge rows. Three results.

## 1. The §7 "weak on S" verdict was a stratum artifact (compare_judges_campaign.py)

| stratum | n | D corr | D bias | S corr | S bias |
|---|---|---|---|---|---|
| trans | 660 | +0.795 | **+0.090** | +0.375 | −0.150 |
| sib | 332 | +0.606 | −0.063 | **+0.476** | −0.094 |
| cous | 331 | +0.665 | −0.047 | **+0.608** | −0.060 |
| rand | 377 | +0.807 | −0.001 | +0.550 | −0.007 |
| ALL | 1700 | **+0.877** | +0.013 | **+0.703** | −0.090 |

(ceilings: 5.5 same-judge repeat D 0.954, S 0.766)

The fresh-250 S number (0.35–0.44) reproduces exactly on the transitive stratum (+0.375) — §7 measured
luna's S where S doesn't vary. Where it varies (sib/cous), luna agrees at 0.48–0.61; pooled S +0.703 vs the
0.766 ceiling. **Luna ≈ 90% of 5.5's self-agreement at a fraction of the price.** Tilt refined: −S bias is
universal; the +D bias is transitive-specific (luna reads more hierarchy only in true hierarchy pairs).

## 2. Stratified luna channel value + empirical R (fine_tune_fused_head_luna.py, analytic part)

5×5 joint blocks fit per corpus on the train split from the dual-judge data (no imported R):
fitted R_luna D 0.030/0.040, S 0.015/0.032 (≈7–10× 5.5's self-R, as the multi-judge report estimated).
Ladder vs the 5.5 target (held rows): luna channel worth **+0.766/+0.721 NLL** over prior+graph
(expl/fresh) — larger than the transitive-only +0.451. Fusion non-degeneracy pull 0.080/0.095.

## 3. The fused head beats a channel head for the first time — in the predicted regime

kalman-fused retrained on the LUNA-fused posteriors (prior ⊕ graph ⊕ luna), alongside 5.5/graph/luna
channel supervision; held rows, WITHIN-stratum, vs the 5.5 labels:

| head (within-stratum vs 5.5) | expl D | fresh D | expl S | fresh S |
|---|---|---|---|---|
| luna channel head (raw cheap labels) | +0.351 | +0.407 | +0.373 | +0.312 |
| **luna-FUSED head** | **+0.396** | **+0.451** | +0.363 | +0.305 |
| 5.5 head (expensive labels — reference) | +0.411 | +0.503 | +0.369 | +0.323 |

- **D**: fusion recovers roughly HALF the gap between cheap-label and expensive-label heads
  (+0.045/+0.044 within-stratum), at zero additional scoring cost. This is REPORT_fused_head.md's null
  boundary confirmed constructively: with a noisy judge the posterior differs from the label (pull 0.08–0.10)
  and the fused head learns the difference (fidelity to its own target: within +0.381–0.588).
- **S**: a wash, as expected — the graph doesn't observe S, so the S fusion has only the weak prior to add.
- The luna channel head itself now has stratified S signal (within vs luna's own labels +0.25–0.28) —
  fixing the B2-step-3 residual's S starvation.

**Economics statement.** For future corpora labeled with the cheap judge, fuse-then-distill recovers ~half
of the D-channel quality gap to expensive labels for free. Combined with §1 (luna ≈ 90% of ceiling pooled),
the cheap-judge pipeline is live: luna labels + Kalman fusion + conflict-routed 5.5 calls (Lever A) is the
deployment recipe to beat.

## 3b. Multi-seed hardening (seeds 0–2, full-pipeline reseed: split + covariance fit + init + batches)

Fused vs raw-luna channel head, D WITHIN-stratum, held rows:

| seed | expl fused/luna/5.5h | fresh fused/luna/5.5h |
|---|---|---|
| 0 | +0.396 / +0.351 / +0.411 | +0.451 / +0.407 / +0.503 |
| 1 | +0.607 / +0.594 / +0.600 | +0.411 / +0.401 / +0.465 |
| 2 | +0.538 / +0.491 / +0.542 | +0.434 / +0.381 / +0.449 |

- **Fused > raw-luna in 6/6 corpus×seed cells** (mean Δ +0.035, sd 0.020) — the win's SIGN is robust;
  its magnitude varies with the split (seed 1 exploratory nearly ties).
- Refinement of the §3 claim: on EXPLORATORY the fused head effectively TIES the expensive-label 5.5 head
  (Δ −0.015/+0.007/−0.004); on FRESH it recovers ~half the gap (−0.052/−0.054/−0.015). "Half the gap" was
  the conservative read; on the denser corpus fusion closes it entirely.
- The analytic luna channel value is stable across seeds: +0.766/+0.767/+0.805 (expl), +0.721/+0.700/+0.640
  (fresh) NLL over prior+graph.
- Level differences between seeds reflect the split (descendant-disjoint held sets differ in composition) —
  comparisons are valid within a seed row, which is why the paired Δ is the reported statistic.

## Caveats

Single split per corpus per seed for the trained heads (the paired Δ handles level shifts);
target = the 5.5 operating judge, not ground truth (same-family privilege — a human-verified subset remains
the gold-standard upgrade); 14 non-dict response objects skipped at ingest (the §3 format-discipline guard);
luna's fitted R absorbs its tilt as variance (bias-augmented state would price it properly).

Repro:
```
python3 score_with_codex.py --pairs /tmp/mu_data/campaign_pairs.tsv --batch 10 \
    --model gpt-5.6-luna --judge gpt-5.6-luna --out /tmp/mu_data/campaign_scored_luna.tsv \
    --responses /tmp/mu_data/campaign_luna_responses.txt
python3 compare_judges_campaign.py
python3 fine_tune_fused_head_luna.py --ckpt model_channel_heads_namecond_r0.pt
```
