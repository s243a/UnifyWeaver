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

## Ablation — attributing the gain (the verification)

**What an ablation is:** change the system *one piece at a time* so each piece's contribution is isolated.
The first A/B changed two things at once (the **architecture** AND the **query**), so it couldn't say which
helped. Three arms, adding the changes one at a time:

| arm | architecture | query | SYM mean | disc mean (per-seed) |
|---|---|---|---|---|
| baseline | JointPosterior+Dirichlet | μ_vec | +0.662 | 86.7% (88/80/92) |
| **anchored-μ** | anchored attention | **μ_vec only** | +0.714 | **96.0% (96/96/96)** |
| anchored-full | anchored attention | §8c fusion | +0.745 | 90.7% (88/92/92) |

- **anchored-μ vs baseline** (same input, new architecture): **discrimination +9.3pp and perfectly consistent
  (96/96/96)**; SYM +0.052. ⇒ **the architecture genuinely helps** — it is *not* just the richer query.
- **anchored-full vs anchored-μ** (same architecture, + rich query): discrimination **−5.3pp**; SYM +0.031.
  ⇒ **the rich query HURTS discrimination** (small SYM gain, real disc cost).

## Verdict
- **The anchored-basis architecture is validated.** With the *same* input as baseline it beats it on both
  metrics — strongly and consistently on discrimination. The attention-over-anchors+atoms → op_weights
  mechanism (with the anchor-KL) is the real driver, not a query-richness artefact.
- **The §8c full query is NOT worth it as-is.** Adding the provenance + raw e5 text drops discrimination
  (−5.3pp) for a noisy SYM sliver. Likely the 384-d `e5_raw` (and/or the provenance) drowns the μ-evidence
  the discrimination probe needs.
- **Operating point: anchored architecture + `--anchor-query mu`** — best discrimination (96%, stable),
  good SYM, *and* simpler (no provenance/raw-text plumbing).

## Next (revised by the ablation)
- Adopt **anchored-μ** as the anchored default.
- If we still want the query signal: ablate it finer — **provenance-only** vs **raw-text-only** (the −5.3pp
  is probably the 384-d raw text; provenance alone might be neutral/positive). Project `e5_raw` down before
  fusing, or drop it.
- Then the deferred v2: value-token, μ-refresh, **utilisation readout** (is K=5 right?), grow/prune.
