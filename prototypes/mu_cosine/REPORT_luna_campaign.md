# Luna campaign: corrected prior/calibration rerun and historical comparison

Post-#3648 correction (2026-07-10): the original campaign report used the campaign-trained `r0` checkpoint
both to initialise the channel head and to define the Gaussian prior, and it left Luna's affine tilt inside
the residual covariance. The corrected run separates those roles: `model_channel_heads_namecond_r0.pt` remains
the training initialisation/anchor, while campaign-independent `model_prod_namecond.pt` supplies the prior;
Luna is globally affine-calibrated on the train split before covariance fitting. Graph traversal, PyTorch
threading, row sampling, and augmentation are deterministic. The old numbers remain below only as an exactly
reproducible historical configuration.

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

## 2. Corrected stratified Luna value + empirical R (`fine_tune_fused_head_luna.py`)

Five-by-five joint blocks are fit per corpus on the dual-judge train rows; no R is imported. Held-row NLL:

| configuration / corpus | fitted R Luna D / S | prior | +graph | +debiased Luna | Luna value | mean \|post_D − Luna_D\| |
|---|---:|---:|---:|---:|---:|---:|
| corrected exploratory | 0.0208 / 0.0149 | -0.285 | -0.568 | **-1.493** | **+0.925** | 0.030 |
| corrected fresh | 0.0273 / 0.0234 | +0.405 | -0.018 | **-1.101** | **+1.083** | 0.035 |
| historical exploratory | 0.0302 / 0.0153 | -0.212 | -0.472 | -1.238 | +0.766 | 0.080 |
| historical fresh | 0.0395 / 0.0321 | +0.569 | +0.187 | -0.533 | +0.721 | 0.095 |

The correction makes the scientific interpretation clearer: debiased Luna is the S workhorse and dominates
the analytic gain. The posterior legitimately moves less far from Luna after systematic bias is removed; a
small pull is no longer evidence that fusion is degenerate. These are one-split descriptive results (row SD,
not an independence-based SE); strict node-disjoint uncertainty is reported separately by
`run_sym_channel_fusion.py`.

## 3. Corrected deterministic fused-head run

kalman-fused retrained on the LUNA-fused posteriors (prior ⊕ graph ⊕ luna), alongside 5.5/graph/luna
channel supervision; held rows, WITHIN-stratum, vs the 5.5 labels:

| head (within-stratum vs 5.5) | expl D | fresh D | expl S | fresh S |
|---|---|---|---|---|
| luna channel head (raw cheap labels) | +0.373 | +0.398 | **+0.369** | **+0.296** |
| **luna-FUSED head** | **+0.402** | **+0.467** | +0.361 | +0.261 |
| 5.5 head (expensive labels — reference) | +0.419 | +0.481 | +0.354 | +0.287 |

- **D:** fusion improves over the Luna channel by +0.029/+0.069 and closes about 63%/83% of the gap to the
  expensive-label head in this single deterministic run. Fidelity to the analytic posterior is +0.350/+0.479
  within-stratum.
- **S:** fusion does not improve over the Luna channel (−0.008/−0.035). This agrees with the node-disjoint
  result that graph_S's small incremental value after debiased Luna is not confirmed. Posterior fidelity
  (+0.453/+0.461) shows this is not simply a failure to learn the target.
- The Luna channel retains stratified S signal (within vs its own labels +0.312/+0.224), fixing the earlier
  transitive-only S starvation.

**Economics statement.** The corrected evidence supports debiased Luna as the paid S workhorse and
fuse-then-distill as a promising D improvement. It does not yet justify a universal “half the gap for free”
claim: the recovery is corpus-dependent, S does not improve, and this head result is one
descendant-disjoint—not strict node-disjoint—split. Matched-cost and routing decisions should use the separate
paired-budget and node-disjoint validations rather than extrapolating this table.

## Caveats

Single deterministic seed and one descendant-disjoint split per corpus; target = the 5.5 operating judge,
not ground truth (same-family privilege — a human-verified subset remains the gold-standard upgrade); 14
non-dict response objects skipped at ingest (the §3 format-discipline guard). The implemented bias correction
is a global per-channel affine fit. Per-stratum/per-D-bin bias models and strict node-disjoint repeated head
training remain follow-up work.

Run provenance (SHA-256): channel init `797e0b79ceba8af6e98c2fafb21b963de238a68a427deb71a75029f2c97cf1de`;
campaign-independent prior `c1cfc3a3827e42a1993f4286b6a881aee7ff10eb56a76367735b9ec8fdf11f7d`;
GPT-5.5 campaign `c2acf399aadd35c3797171d5b42d64e45b07055802092cc28becf308d460ef09`;
Luna campaign `a8d951b4fd05f0ca111fbe9d9c23881bb47b790ccb335731113bfd4ae77ffe6e`;
corrected `/tmp` output `fa6a5778452a2a92968cbae14a2150d056d0069d4f20aaa36fbf9aac1a919338`.

Repro:
```
python3 score_with_codex.py --pairs /tmp/mu_data/campaign_pairs.tsv --batch 10 \
    --model gpt-5.6-luna --judge gpt-5.6-luna --out /tmp/mu_data/campaign_scored_luna.tsv \
    --responses /tmp/mu_data/campaign_luna_responses.txt
python3 compare_judges_campaign.py
python3 fine_tune_fused_head_luna.py --analytic-only \
    --ckpt model_channel_heads_namecond_r0.pt \
    --prior-ckpt model_prod_namecond.pt --luna-calibration global
python3 fine_tune_fused_head_luna.py --steps 800 --out /tmp/model_fused_head_luna_corrected.pt \
    --ckpt model_channel_heads_namecond_r0.pt \
    --prior-ckpt model_prod_namecond.pt --luna-calibration global

# Exact historical analytic configuration (no training):
python3 fine_tune_fused_head_luna.py --analytic-only \
    --ckpt model_channel_heads_namecond_r0.pt \
    --prior-ckpt model_channel_heads_namecond_r0.pt --luna-calibration none
```
