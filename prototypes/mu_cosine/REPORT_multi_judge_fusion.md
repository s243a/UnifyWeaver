# Multi-judge fusion: the fused-head null's boundary confirmed — luna's fusion is non-degenerate

`run_multi_judge_fusion.py` — boundary case #1 from REPORT_fused_head.md, tested on existing data (the
fresh 250 carries prior, graph, gpt-5.5 run 2, and gpt-5.6-luna against the gpt-5.5 run 1 target; zero new
scoring). Setup mirrors run_judge_channel.py: 7×7 joint residual covariance fit per split, so luna's
cross-correlations with 5.5 and the prior are PRICED; 40 descendant-disjoint splits, constant blocks.

## Ladder (target = judge1 = the operating campaign judge; NLL ↓)

| rung | NLL | Mahal/dim | channel value over prior+graph |
|---|---|---|---|
| prior | +0.408 | 1.42 | — |
| prior+graph | −0.012 | 1.40 | — |
| prior+graph+**luna** | −0.462 | 1.55 | **+0.451** (row-SE 0.026) |
| prior+graph+**j2** | −1.946 | 1.09 | **+1.935** (0.034) — Lever A's +1.94 replicates |
| all | −1.981 | 1.14 | luna's marginal after j2: **+0.034** (0.009) |

## The two findings

1. **The fused-head null's boundary is real.** Fusion non-degeneracy (mean |posterior_D − judge_D|):
   **0.133 with luna** vs 0.020 with j2. With the reliable judge the posterior collapses onto the
   measurement (why the fused head learned nothing beyond the LLM head); with the noisy judge the filter
   genuinely mixes prior, graph, and judge. A luna-fused posterior is a target that DIFFERS from any single
   channel — the setting where an amortized `mu_PoE` head has something to learn.

2. **Judge economics, extended.** Luna lost as a solo campaign judge (§7: not interchangeable, S at half
   the ceiling) but carries ~23% of the full-price judge's channel value through fusion, and a small
   positive consensus increment after 5.5. This prices the Lever-A escalation ladder's bottom rung: if
   luna is ≥~4× cheaper per call, luna-on-every-row + 5.5-on-conflict-routed-rows dominates 5.5-only at
   matched spend (subject to the caveats below).

## Caveats

- **Same-family target privilege**: y = gpt-5.5 run 1, so j2's fitted R absorbs shared 5.5 bias (looks
  better than against ground truth) and luna's fitted R absorbs its true +D/−S tilt (looks worse). This is
  the operationally-correct frame — 5.5 IS the campaign judge — but it lower-bounds luna against an
  independent truth; a human-verified subset remains the gold-standard upgrade (the §12(3) lesson).
- Luna-rung error bars mildly overconfident (Mahal/dim 1.55 vs ≈1.4 base) — the tilt is a bias term, and
  constant blocks fold it into R as variance; a bias-augmented state (DESIGN metastable §) would price it
  properly.
- All-transitive 250 (S channel starved as usual); constant blocks, no Σ(hop).

## Follow-up unlocked

Re-run the fused-head distillation with LUNA-fused targets — now non-degenerate (pull 0.133). Blocked on
data breadth: 250 all-transitive pairs are too thin/narrow to train a head; wants the stratified luna
scoring run (~2,000 pairs at luna prices ≈ well under the usual spend guideline).

Repro: `python3 run_multi_judge_fusion.py`
