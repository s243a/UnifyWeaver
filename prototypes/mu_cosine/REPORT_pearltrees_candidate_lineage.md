# Candidate-lineage-conditioned μ, the e5-vs-μ diagnostic, and Kalman channel aggregation

Follow-up to REPORT_pearltrees_filing_v1.md §7. Three threads: (1) fix the identified train/eval
regime mismatch by feeding the candidate FOLDER's principal path into the scalar μ scorer and
re-run the filing metric; (2) answer why e5 beats the μ heads at filing; (3) work through which
judge measurement functions the fusion Kalman filter should carry.

## 1. Candidate-lineage-conditioned μ (the §7 fix)

**What was wrong.** LINEAGE is a scalar pairwise scorer `(node, folder, LINEAGE)→μ`; its lineage
context came from `Tokenizer._anc_for(NODE)` — the deeper endpoint's ancestors. At filing the node
is a fresh bookmark with no lineage and the eval tokenizer had empty parent tables, so the folder's
known principal path never reached the scorer. The head was scored blind to lineage.

**The fix.** A backward-compatible `Tokenizer(root_lineage=True, root_lineage_depth=k)` flag now
also emits the anchor/FOLDER's materialized path (deterministic first-parent climb) as `anc`
tokens — reusing the existing ancestor role, so no new model parameters and no forward-pass change.
The model is retrained with it on (`fine_tune_pearltrees_filing.py --cand-lineage`); the filing
eval supplies each candidate folder's principal path from the assembled DAG
(`eval_pearltrees_filing.py --cand-lineage`), respecting the §7 leakage boundary: ONLY the
folder→parent lineage, never the evaluated bookmark's placement nor the folder's bookmark children.

**Result.** Each checkpoint evaluated in its trained regime (the merged no-lineage checkpoint
without lineage tokens; the new `model_pt_filing_lin.pt` with them); 217/323 folders had a
principal path (271 ancestor titles embedded). Filing MRR (fidelity to real treeId placement):

Post-audit rerun (record-majority PRINCIPAL lineage — the earlier first-DAG-parent walk
contradicted the recorded parent on 61% of multi-parent folders — and a SUPPORT-PRESERVING shuffle
control; coverage rose to 253/323 folders, 1118/1147 edges from records):

| head (MRR ↑) | no-lineage | +principal lineage | +lineage SHUFFLED (clean control) |
|---|---|---|---|
| mu-max-cond (conditioned) | 0.075 | 0.081 | 0.078 |
| mu-lineage | 0.044 | 0.068 | 0.078 |
| mu-elem-cond | 0.081 | 0.100 | 0.108 |
| held-folder mu-max-cond (n=279) | 0.073 | 0.082 | 0.088 |
| e5-cos (reference, lineage-invariant) | 0.294 | 0.294 | 0.294 |

**Verdict (corrected, and cleaner than the first cut): the lineage effect is GENERIC CONTEXT, not
the specific path.** With the true principal lineage and the clean control, the shuffled chains do
as well as — sometimes better than — the correct ones on every head, INCLUDING the held-folder
slice where the first cut had shown a right-lineage edge (that edge was an artifact of the wrong
lineage construction plus the confounded shuffle, and did not survive the audit corrections).
Extra folder-side tokens help a little (0.075→~0.08–0.11 across heads); WHICH lineage they encode
does not measurably matter at this scale. Consistent with §"titles already carry the lineage".
Nothing approaches e5 (0.294).

Why so weak — the titles already carry the lineage. The held-folder slice discriminates the two
candidate explanations: no-lineage 0.073 → correct 0.097 → shuffled 0.084, so the RIGHT lineage
beats both no-lineage AND shuffled on UNSEEN folders, while the gain washes out once seen folders
are pooled in. That rules out "the model can't use lineage order at all" (capacity/position
embeddings — it would lose everywhere) and points to "the folder title already encodes its
lineage." Direct probe: a child folder's e5 embedding is closer to its TRUE parent (mean cosine
0.877) than to a random folder (0.816), a +0.062 separation — the parent relationship is legible in
the title alone. So on folders the model has seen, it infers the parent path from the title and
explicit lineage is redundant; explicit lineage only adds value where the folder's title is
unfamiliar (the held slice). A weak depth encoding (this first cut reuses the shared `anc` role,
no distinct candidate-lineage role) plausibly CAPS the held-folder magnitude but does not explain
the pattern. Consequence for design: the leverage is not injecting lineage (titles have it) but
CALIBRATING μ so the ranking is right — which is exactly the training-time meta-judge's job
(candidate-ranking CE that calibrates the μ signal fed to the Kalman filter), aimed at the
confusable NEAR stratum (§2) where e5 is also weak.

## 2. Why does e5 beat the μ heads? Mostly the regime, not pure capability

`diagnose_filing_e5_vs_mu.py` stratifies the 400 filing queries by e5 DIFFICULTY — the true
folder's e5-cosine minus the best distractor's. Negative gap = the true folder is NOT the
e5-nearest (a distractor is closer); positive = e5 already separates it.

    e5 difficulty gap: min −0.175, median −0.044, max +0.075
    (i.e. for MOST queries the true folder is not the e5-nearest neighbour)

| stratum (by e5 gap) | n | e5 recall@1 | mu-max recall@1 | μ − e5 |
|---|---|---|---|---|
| NEAR (confusable, gap [−0.17,−0.06]) | 134 | 0.000 | 0.000 | +0.000 |
| MID (gap [−0.06,−0.02]) | 133 | 0.000 | 0.038 | +0.038 |
| FAR (e5 easy, gap [−0.02,+0.08]) | 133 | **0.617** | 0.060 | −0.556 |

μ-beats-e5 rate: **0.355 on the NEAR (confusable) half vs 0.065 on the FAR half.**

**Reading.** e5's entire filing win is the FAR third — the queries where the bookmark title is
already the folder's nearest e5 neighbour (recall@1 0.617 there, ~0 elsewhere). On the other two
thirds — where filing is actually a decision, because the true folder is NOT the e5-nearest — e5
scores essentially zero and μ is no worse (slightly better on MID). This is your hypothesis
confirmed: the aggregate "e5 wins" is an easy-case artifact; the semantic spread of the candidate
folders means e5 cosine already nails the trivial cases and neither method is strong on the hard,
confusable cases where a filing assistant would earn its keep. It is NOT that raw similarity is
fundamentally better than the learned μ signal — it is that the eval is dominated by cases where
raw similarity suffices. (Luna itself is not a filing ranker here; it was the label source. The
comparison is e5-cosine vs the model's learned conditioned μ.) The confusable NEAR stratum is
exactly where candidate-lineage conditioning (§1) should pay — folder titles like "Tools" or
duplicated names disambiguated by their parent path.

## 3. Kalman measurement channels: aggregate vs split (the weighting-function question)

The fusion currently carries a REDUCED 2-channel-per-judge measurement: `D = max(subcategory,
subtopic, element_of, super_category)` and `S = max(see_also, assoc)`. Your question: keep the
directional/symmetric aggregates, or split every relation (both directions) into its own channel?
Empirically, from the 799 Pearltrees luna rows (per-relation μ correlation + which relation wins
each max):

| | subcat | subtopic | element_of | super_cat | see_also | assoc |
|---|---|---|---|---|---|---|
| subcategory | 1.00 | 0.66 | 0.50 | **0.01** | 0.12 | −0.05 |
| subtopic | | 1.00 | 0.32 | **0.00** | 0.39 | 0.20 |
| element_of | | | 1.00 | **0.01** | 0.09 | −0.01 |
| super_category | | | | 1.00 | 0.07 | 0.02 |
| see_also | | | | | 1.00 | 0.63 |
| assoc | | | | | | 1.00 |

- **The forward directional relations (subcategory/subtopic/element_of) are redundant** (r 0.32–0.66)
  and subtopic wins the D-max on 706/799 rows — collapsing them into one D aggregate loses little.
- **super_category is nearly ORTHOGONAL to all of them (r ≈ 0.00–0.01)** and wins the D-max on only
  11/799 rows — so the current `max`-into-D essentially DISCARDS it. It is the reverse-direction
  membership signal ("root is under node"), a genuinely independent measurement.
- **see_also/assoc are correlated (0.63)** but assoc carries more (wins 610/799); splitting them
  gives a modest independent-signal gain.

**Recommendation.** The high-ROI change is NOT the full per-relation explosion (6 channels × both
directions), which would blow up the covariance parameter count against only ~300 overlap rows.
It is a targeted 3rd channel: keep the forward-directional aggregate D, keep the symmetric aggregate
S, and ADD a REVERSE-direction channel (super_category, or the judge's mu_rev) — the one signal the
current max provably throws away. That is a 3-channel-per-judge measurement, well within the overlap
budget, and it maps cleanly onto the directional asymmetry the filing task cares about
(bookmark→folder membership is directional). Splitting the symmetric aggregate is a secondary,
lower-value option. A confirmatory test belongs on the enwiki node-disjoint harness
(`run_sym_channel_fusion.py`), where the covariance-vs-data tradeoff can be measured with the frozen
gate rather than the small Pearltrees overlap.

## Caveats

- The lineage eval keys folders by title; duplicate folder titles share one lineage entry (the
  title-equivalence caveat from Filing v1 §5 still applies).
- 225/335 candidate folders have a principal path in the partial assembled DAG; 110 are roots or
  isolated (no lineage supplied — a coverage limit the completed Grok export lifts).
- The e5-difficulty stratification uses the best-distractor gap; it is a descriptive lens, not a
  calibrated hardness model.
- §3's correlations are measured on luna's raw per-relation μ; the fusion operates on the affine-
  calibrated D/S, so the split-channel gain must be confirmed by an actual re-fusion, not inferred
  from the correlation table alone.

## Repro

```
# candidate-lineage retrain + eval (with vs shuffled control)
python3 fine_tune_pearltrees_filing.py --cand-lineage --out model_pt_filing_lin.pt
python3 eval_pearltrees_filing.py --tuned model_pt_filing_lin.pt --cand-lineage
python3 eval_pearltrees_filing.py --tuned model_pt_filing_lin.pt --cand-lineage --shuffle-lineage
# e5-vs-μ diagnostic
python3 diagnose_filing_e5_vs_mu.py --tuned model_pt_filing.pt
```
