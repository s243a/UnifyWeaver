# Anchored-basis A/B — first measurement (weights-only v1)

The first real test of DESIGN §8: compute the blend `op_weights` via attention over frozen e5-phrase relation
ANCHORS + 5 learnable ATOMS (`--anchored-basis`), vs the current JointPosterior+Dirichlet blend. Both
warm-started from the 4L/0.7 base on the combined 6-source round (`graded_all`, 3824 targets), early-stopped,
3 seeds.

## Result

| seed | baseline SYM | anchored SYM | baseline disc | anchored disc |
|---|---|---|---|---|
| 1 | +0.433 | **+0.578** | 88% | 88% |
| 2 | +0.785 | **+0.834** | 80% | **92%** |
| 3 | +0.767 | **+0.823** | 92% | 92% |
| **mean** | +0.662 | **+0.745** | 86.7% | **90.7%** |

**Anchored beats baseline on SYM in all 3 seeds (+0.083 mean) and is ≥ on discrimination in all 3 (+4pp).**
The 3/3 consistency (not 50/50) is the signal that it's more than seed noise — notable since this is the
stripped-down **weights-only v1**: the e5-phrase anchor *values* aren't the token yet, the query uses
warm-start μ (not refreshed), and there's no utilisation readout.

## Honest caveats
- **Attribution unclear.** The anchored query also carries the categoriser **provenance + raw e5 text**, which
  the baseline (JointPosterior on μ_vec alone) never sees. So the gain may be the anchored attention/atoms,
  the richer query, or both. A clean ablation (baseline+provenance, or anchored−provenance) would separate
  them. The A/B fairly shows "full anchored approach > baseline blend"; it does not yet attribute *why*.
- **Noisy metrics.** SYM is 40 held-out positives (baseline swings 0.433→0.785 across seeds); disc is 25
  probes. The effect is modest; read it as "consistent and promising," not "decisive."
- The baseline mean (0.662) is in line with the earlier `ft1` run (0.671) — sanity check passes.

## Verdict / next
A consistent, modest improvement in the right direction — enough to justify the v2 build:
1. **Ablation** to attribute (provenance-only vs anchored-attention).
2. **Value-token** (use the e5-phrase anchor values as the token, the full §8).
3. **μ-refresh** in the query (vs warm-start static).
4. **Utilisation readout** (`anchored_rel.utilization()` at eval) → is K=5 right / grow? (§8b)
5. then the **grow/prune controller**.
