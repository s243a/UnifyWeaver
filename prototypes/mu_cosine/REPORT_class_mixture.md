# Class-mixture predictive for D (B2 step 4): the §11.5 gate PASSES

`run_class_mixture.py` — the fix for the atoms run's diagnosed residual defect (D's directional-or-not
bimodality vs an effectively-unimodal Gaussian predictive; the JointPosterior result #1 resurfacing),
judged against the THEORY_evidence_fusion §11.5 acceptance gate. Campaign data, both corpora, judge-free
context only (prior readouts, graph walk, stratum one-hots), all fits on the train split.

```
p(D | context) = P(dir | context) · N_dir(D)  +  (1 − P(dir | context)) · N_lat(D)
```

Classes anchored (gate 3) to the relation family — is the pair a hierarchy pair at all (label D ≥ 0.5) —
with P(dir | context) a logistic regression on judge-free observables, so the mixture is deployable before
any judge call. Per-class means = per-class affine-calibrated prior readouts; proper score = bin-mass NLL
(labels live on the 0.05 lattice — density NLL would be the wrong score, the atoms-run lesson).

| §11.5 criterion | exploratory | fresh | verdict |
|---|---|---|---|
| 1. held-out proper score improves | Δ **+0.667** NLL/row (row-SE 0.056) | Δ **+0.343** (0.060) | PASS |
| — trans-only slice (where the defect was diagnosed; stratum one-hot can't carry it) | Δ +0.625 (0.154) | Δ +0.318 (0.123) | survives |
| 2. class posterior calibrates | ECE 0.030 | ECE 0.056 | PASS (extreme bins near-perfect; mid bins small-n noisy) |
| 3. modes anchored to observable relation classes | by construction (directional family), predicted from judge-free context | — | PASS |
| 4. shape (support only) | end-masses 0.67 / 0.15 | 0.52 / 0.16 | consistent |

**Reading.** The D defect is real and the two-mode class mixture prices it: about two-thirds of a nat per
row of predictive improvement pooled, and ~0.3–0.6 nats even inside the transitive stratum — deep-ancestor
pairs split into "still a hierarchy relation" vs "effectively unrelated", and prior_D + the walk predict
which, so the predictive should be a mixture over that class, not one Gaussian straddling the gap.

**Epistemic reading (§11.5's closing point).** The bimodality is epistemic — the pair IS one thing. The
class posterior's middle bins (pred 0.4–0.6) are exactly the rows where a judge call collapses the modes:
this is the Lever-A conflict/ambiguity routing signal, now with a distributional justification. The
mixture and the routing policy are two faces of the same object.

**Bounds.** Single split per corpus (the fixed descendant-disjoint split, no seed sweep); 2 classes only
(see_also-vs-assoc substructure inside "lateral" not modeled — the atoms run found 3-class atom masses
already correct, so K=2 targets only the diagnosed defect); the class anchor is judge-derived at TRAINING
time (gate-3 anchoring is about the modes' identity, not label-freeness — deployment needs no judge).

Repro: `python3 run_class_mixture.py --ckpt model_channel_heads_namecond_r0.pt`
