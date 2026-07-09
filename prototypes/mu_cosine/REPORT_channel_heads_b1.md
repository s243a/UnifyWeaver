# B1: per-channel heads by light fine-tune — an informative negative; the bottleneck is data

*`fine_tune_channel_heads.py`, 2026-07-09. Follows the probe (judge tokens don't route; the S channel is
missing). Two fine-tunes from `model_prod.pt` (judge_emb expanded 5→9), channel-tagged rows on the two multihop
corpora (~1,050 rows / ~350 per channel), descendant-disjoint held sets, agnostic path protected.*

## Ladder

| run | trainable | trunk protection | outcome |
|---|---|---|---|
| B1 | judge_emb only (3,456 params) | BY CONSTRUCTION (p_mask_prov=0; drift 0.0 verified) | partial |
| B1b @5e-3 | + last layer + readout (1.19M) | agnostic-anchor distillation | **COLLAPSED** (saturated readout, drift 0.90) |
| B1b @5e-4 + grad-clip | same | same | stable (drift 0.075) |

## Held-out channel correlations

| channel | baseline (probe) | B1 | B1b stable | verdict |
|---|---|---|---|---|
| graph-d exploratory | +0.45 | +0.67 | **+0.75** | routes, but… |
| graph-d fresh | +0.28 | +0.19 | +0.18 | …DOESN'T TRANSFER — corpus-specific walk memorization |
| llm-D (both) | +0.53/+0.57 | ~+0.55 | ~+0.54 | saturated at baseline — no head gain |
| **llm-S (both)** | ~0.00 | ~0 unstable | **~0 unstable** | **channel NOT created** |

## Findings

1. **The missing S channel is not buildable at this data scale.** With capacity (1.2M params), healthy
   optimization, and the trunk anchored, ~350 S-labeled examples produced no stable positive correlation.
   Combined with the probe (S signal absent from frozen features), the deficit is now located precisely:
   **not routing, not capacity — data.**
2. **The graph channel's gain is memorization, not a head:** +0.75 on the training corpus's held descendants,
   flat on the fresh corpus. A real graph head must transfer; this one learned 100k_cats walk structure.
3. **llm-D is already saturated at baseline** — the agnostic trunk extracts as much D as these features carry.
4. **Training mechanics for this codebase, learned the hard way:** embeddings-only tolerates lr 1e-2; a
   transformer layer needs ~5e-4 + gradient clipping (5e-3 unclipped saturates the sigmoid readout and
   collapses everything, taking the anchor with it). The anchor loss holds at drift ≈ 0.075 when optimization
   is sane and cannot save a collapsed run.

## Disposition

Fusion-heads step 2 is BLOCKED ON DATA, not architecture. Next lever: a scoring CAMPAIGN — thousands of
S-labeled (and D-labeled) pairs, generated with the LLM judge at scale, budget allocated by the
escalation-ladder/conflict policy from the Lever-A routing result (spend where decisions flip). Until then,
the channels remain best served explicitly by the Kalman stack (which needed no per-pair training). B2 (fused
head distilling the Lever-A posteriors) has the same data economics and should ride the same campaign.

## Repro

```
python3 fine_tune_channel_heads.py --steps 800                                # B1 (frozen trunk)
python3 fine_tune_channel_heads.py --steps 800 --lr 5e-4 --unfreeze-last      # B1b (anchored)
python3 run_channel_heads_probe.py                                            # acceptance probe
```
