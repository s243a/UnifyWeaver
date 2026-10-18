---
name: uncertainty-estimation
description: Playbook for combining multiple signals into a μ / confidence / relation estimate in prototypes/mu_cosine. Use when the work involves fusing multiple μ sources, "confidence weighting", calibration, product-of-experts / precision / inverse-variance combining, selective prediction, or picking a confidence gate — so you start from the calibrated joint-posterior + margin-gate approach instead of hand-set independent weights. Trigger on: "confidence", "uncertainty", "combine sources", "fuse judges", "which source to trust", "calibrate", "AURC/ECE".
user-invocable: true
allowed-tools:
  - Read
  - Bash(python3 *)
  - Bash(ls *)
---

# Uncertainty / multi-source μ-estimation

You're about to combine multiple signals into a μ / confidence / relation estimate (or you reached for
"confidence weighting"). **Read `prototypes/mu_cosine/DESIGN_uncertainty_estimation_playbook.md` first** — it
encodes what this project learned the slow way. Do not re-derive it.

## The rule
Don't hand-set independent confidence weights over correlated sources. Fit a **learned, calibrated combiner**
(`JointPosterior` in `mu_posterior.py`) on a **held-out, node-disjoint** split, and use confidence as a
**margin gate** (top1−top2), not a per-item weight.

## Before writing any fusion code, check you're not about to:
1. inverse-variance / product-of-experts average per-source signals as if independent — but in this project
   every model readout consumes frozen e5, so they're correlated (+0.751) and it over-confidences;
2. treat two anti-correlated readouts (subcat vs element) as separate positive weights — their *joint* is the
   asymmetry;
3. use absolute μ *level* as the confidence signal — use **margin** (level is anti-correlated with correctness
   early in training);
4. use node **degree** as a "data-density" / confidence term — it doesn't measure nearby trained support;
5. measure a source's reliability on **training pairs** — leakage inflates *and reorders* it; use held-out,
   node-disjoint.

## The workflow
1. Assemble the source vector (`e5_mu_fn`, `model_readout_fn`, `struct_dist_fn`, external judges).
2. Print per-source **separability** + the **correlation matrix** — a new source must be *decorrelated* to add value.
3. Fit `JointPosterior` (LR or small MLP) on a **held-out node-disjoint** split; keep factored PoE as a control.
4. Report acc, log-loss, **ECE (stated binning)**, **AURC (margin gate, bootstrap CI)**.
5. **Ablate** each source (with vs without) on the same split — a source earns its keep iff the with-source AURC
   CI sits below the without-source point estimate.

Run: `python3 mu_posterior.py --pairs <graded _pairs.tsv> --e5-cache <e5.pt> --model <ckpt> --struct-emb <se.pt>
--split node-disjoint --boot 500`. See the playbook doc for the full pitfalls, references (#3356/#3357/#3359/
#3387/#3391), and the tool inventory.
