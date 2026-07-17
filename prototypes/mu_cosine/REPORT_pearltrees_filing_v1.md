# Pearltrees Filing v1: first live deployment of the cheap-judge labeling pipeline

Filing v1 steps 2–4 (ARC_filing_program.md §7): label the Pearltrees corpus with the validated
cheap-judge recipe (DESIGN_cheap_judge_pipeline.md — first production run), fine-tune the filing
model on the fused targets, and evaluate on the FILING metric (rank the true folder for real
bookmarks). Evaluation frame throughout: fidelity to the gpt-5.5-low operating judge (labels) and
to the user's actual filing decisions (retrieval eval) — never "semantic accuracy".

**Summary.** The labeling pipeline deployed cleanly (799 dual-strata pairs, 0 scoring failures,
every fusion rung pays); luna's D bias FLIPS SIGN on Pearltrees vs enwiki (§1 — the per-corpus
overlap is not optional); the shadow bias-states show larger bin structure than enwiki and win
their descriptive ladder (§3 — gate-protocol rerun recommended); on the LEAKAGE-FREE readout the
fused head still beats the raw-luna channel within-stratum on S and the LINEAGE head learns
lineage order (§4). On the filing metric the result is an HONEST NULL, strengthened by the
external-review corrections: e5-cosine ranks folders best (MRR 0.294), the like-for-like
conditioned ranker does NOT improve under the corrected fine-tune (mu-max-cond −0.037 paired MRR;
the earlier +0.041 claim is retracted as a random-baseline artifact), and only e5's own margins
are a usable escalation signal (§5). Deployment recipe today: e5 ranks with margin-routed
escalation; the fused model's value is the label factory and conflict router, not the ranking
head.

## 0. Corpus, coverage, and provenance

- **Graph**: the PARTIAL Pearltrees recovery — principal-path records from
  `.local/data/api_tree_paths_v8.jsonl` (752 retained records after privacy filtering, 1,294 path
  nodes) + `assembled_titles.tsv`. The RDF-truncation gap stands (396/880 multi-parent folders
  recovered; ARC §3); the completed Grok re-export was NOT available for this round (user
  confirmed) — coverage upgrades later re-run this same pipeline.
- **Sampling**: Codex's registered principal sampler (250 pairs, 50/hop h1..h5, seed 0, component
  round-robin, direction-conflict pairs excluded — manifest reproduced its documented pool counts
  1077/965/738/492/352) + the new lateral sampler `sample_pearltrees_lateral.py` (same
  privacy-filtered forest: 150 sib, 150 cous, 250 rand, deterministic blake2b ranking, group
  round-robin). 799 unique pairs after 1 duplicate-title-key drop.
- **Title policy**: the audited Pearltrees corrections (36 spelling-only fixes, frozen 2026-07-09,
  applied by endpoint id) are applied to every emitted title — raw titles preserved as aliases in
  the pairs TSV. Typos masquerading as lateral drift are the known Pearltrees pathology.
- **e5 cache**: 955 titles (879 endpoints + ancestor cones), `pt_campaign_e5.pt`.
- **Scoring** (score_with_codex, §14 contract; smoke-tested at 10 rows first): luna on all 799
  (80/80 batches, 0 failures, 47 min); 5.5 on a RANDOM 300-row overlap (30/30, 0 failures, 22 min);
  5.5 on 80 routed conflict rows (8/8, 0 failures, 5.5 min).
  SHA-256: luna `5ce78f6f268d6bcc…`, 5.5 overlap `c7686a316f4e0d31…`.
- **Spend this deployment**: 809 luna calls (799 + 10 smoke) + 390 gpt-5.5 calls (300 overlap +
  80 routed + 10 smoke). ≈ 78 min luna, ≈ 28 min 5.5 wall clock.
- Artifacts in the durable /tmp/mu_data (→ ~/mu_data): `pt_campaign_*` (pairs, score inputs,
  manifests, scored TSVs, fused targets, fusion manifest, e5 cache).

## 1. Luna-on-Pearltrees tilt — the enwiki tilt does NOT transfer on D

Pair-matched luna vs 5.5 on the random overlap (bias = luna − 5.5):

| stratum | n | D corr | D bias | S corr | S bias |
|---|---|---|---|---|---|
| principal (h1..h5) | 84 | +0.811 | **−0.114** | +0.361 | −0.156 |
| pt_sib | 63 | +0.837 | −0.079 | +0.381 | −0.121 |
| pt_cous | 50 | +0.490 | −0.048 | +0.642 | −0.151 |
| pt_rand | 103 | +0.642 | −0.075 | +0.660 | −0.089 |
| ALL | 300 | +0.805 | −0.082 | +0.607 | −0.125 |

vs the enwiki campaign (REPORT_bias_states §0): S bias negative everywhere on BOTH corpora
(pooled −0.089 enwiki / −0.125 here), but the D tilt FLIPS — enwiki transitive **+0.075** vs
Pearltrees principal **−0.114**. Luna over-reads Wikipedia category hierarchy and UNDER-reads
personal filing hierarchy relative to 5.5. Judge bias is corpus-dependent; the per-corpus random
overlap is not optional.

## 2. Fusion (the recipe, live)

Blocks fit on the 300-row random overlap only (routed rows excluded by construction; shrinkage
0.05; prior = `model_prod_namecond.pt` agnostic readouts; graph channels from the principal-path
DAG itself — hit_prob→D affine, 4-feature sym linear→S):

- Fitted R diag (graph_D, graph_S, luna_D, luna_S) = (0.0602, 0.0207, 0.0251, 0.0178);
  P0 diag (D, S) = (0.0783, 0.0254). Luna R here vs enwiki-fit values is same order; graph_D is
  noisier than on enwiki (the partial DAG's up-walk is deterministic on principal rows but
  uninformative off-lineage).
- Fused (D, S) posteriors written for all 799 rows (`pt_fused_targets.tsv`) with per-row posterior
  variances; overlap + routed 5.5 labels carried alongside.
- **Conflict routing**: top 80 non-overlap rows (10%) by |graph_D − prior_D| innovation
  (range 0.536–0.610) → scored by 5.5 as extra training labels, never entering the covariance fit.
  Sonnet tiebreaker skipped this round per plan.

Descriptive held-overlap ladder (one node-disjoint split of the overlap, 45 held rows; NLL ↓,
fidelity to 5.5) — the recipe's rungs all pay on Pearltrees:

| rung | joint | S-marg | D-marg |
|---|---|---|---|
| prior | +1.285 | +1.180 | +0.451 |
| +graph_D | +1.065 | +0.877 | −0.059 |
| +graph_D+graph_S | −0.375 | −0.289 | −0.066 |
| +graph_D+luna | −0.756 | −0.498 | −0.385 |
| ALL | **−1.072** | **−0.624** | **−0.451** |

(n=45, single split — descriptive; the confirmatory ladder machinery remains the enwiki
run_sym_channel_fusion.py.)

## 3. Shadow bias states — Pearltrees shows LARGER bin structure (as hypothesized)

`fit_bias_states` run in SHADOW (spec §5.1; promotion=false after the enwiki encouraging null;
never applied to the fused targets). Fit on 109 overlap-train rows, τ=0.5, rank 8/8, cond 19.4,
only the zero-support `missing` state prior-dominated. Notable offsets (vs enwiki's ≤|0.21|):
graph_D sib **−0.182** / h5 **+0.124** / rand +0.086; luna_S rand +0.060.

Same-split descriptive ladder, control vs shadow treatment on the 45 held rows:

| variant | ALL joint | ALL S | ALL D |
|---|---|---|---|
| affine (deployed) | −1.072 | −0.624 | −0.451 |
| affine+bins (shadow) | **−1.149** | **−0.662** | **−0.511** |

The shadow treatment wins all three metrics on this single small split — consistent with the
hypothesis that a NEW corpus is where bin structure matters most. This is descriptive evidence
only (n=45, one split, no bootstrap): promoting bins for Pearltrees requires the same frozen-gate
protocol (pre-declared primary, paired node-block bootstrap) on a bigger overlap — recommended for
the post-Grok-export round.

## 4. Fine-tune (Filing v1 step 3)

Champion recipe on the fused targets: warm-start `model_prod_namecond_full.pt`
(= model_prod_namecond.pt after `migrate_name_tables.py --tables ops,corpora`; behavior-preserving,
forward max|Δ| 1.8e-07 — the pearltrees corpus onboards BY CARD), 800 steps, lr 5e-4, bs 64,
grad-clip 1.0, agnostic-anchor loss (weight 1.0), trainables = last encoder layer + readout +
judge/op/corpus name residuals + nodetype embedding.

Rows (7-tuples carrying NODETYPE[pearltrees_collection]; train side of the node-disjoint campaign
split, seed 0): 5.5 channels (overlap + routed rows), raw-luna channels, graph walk channel,
kalman-fused distillation channels, and LINEAGE rows — single-path graded decay `0.85^(hop−1)` on
principal rows + 0.0 on pt_rand rows (lca_depth_frac omitted: the partial multi-record DAG has no
canonical depth — documented deviation from DESIGN_mindmap_lineage §3b; sib/cous excluded from
LINEAGE as filing-distance-ambiguous).

The base checkpoint predates the LINEAGE operator (4-row op tables): `load_with_lineage_ops` grows
op_emb/readout/op_name to len(OPS)=6 — old rows copied, new rows fresh (the train_lineage
precedent), LINEAGE's op-name card as the name prior with zero residual.

Two external-review corrections shape this section (both rerun before reporting): (a) the anchor
reference is a deepcopy of the exact initial model (an independently-grown ref carried different
random LINEAGE rows — the anchor pulled toward a different model); (b) the shared name-transform W
matrices are FROZEN (they map every identity's card; training them on pearltrees rows drifts
unrelated judges/corpora/ops) — only per-identity residual rows train (1.20M trainable params).

**Leakage-free held readout.** The first-cut evaluation let the full 300-row overlap fit the
target factory before the split; the corrected protocol splits FIRST and fits the entire factory
(calibrations + covariance + fused targets) on the 108 campaign-train overlap rows only
(`run_pearltrees_fusion.py --fit-overlap train-only`), so the 49 held overlap labels never touch
anything upstream of evaluation. Result (pooled / between / WITHIN vs 5.5):

| head | D within | S within |
|---|---|---|
| kalman-fused (distilled) | +0.197 | **+0.339** |
| luna channel (raw cheap labels) | +0.211 | +0.302 |
| 5.5 channel (expensive labels) | **+0.291** | **+0.462** |

The honest ordering: the direct-5.5 head is best on both channels (as it should be — it is
supervised by the target judge); the fused head still beats the RAW-LUNA channel on within-stratum
S (+0.339 vs +0.302, n=49, one split) — the economics claim (fusion upgrades cheap labels) holds
in direction, the stronger first-cut claim (fused ≈ 5.5 head) was partly leakage and is retracted.
Held LINEAGE (38 principal rows): rank-corr **+0.655**, MAE 0.329 — learns lineage ORDER,
level-miscalibrated (rank is what filing consumes).

The production checkpoint retrains with the full-overlap factory (all 300 rows — the deployment
configuration; its held-overlap table is no longer quoted since those labels entered its targets).

Checkpoint: `model_pt_filing.pt` (gitignored;
SHA-256 `c0a2f731876cf81a668d2e93f1f71337fced2528e3c7a40e946022e759378ac9`).

## 5. FILING metric (Filing v1 step 4 — the deliverable)

`eval_pearltrees_filing.py`: real filing decisions (harvested tree JSONs; ground truth = each
bookmark's actual treeId), candidates = folders with ≥3 bookmarks, base vs fine-tuned. The
agnostic-anchor loss pins the agnostic readouts by design, so the fine-tune is read through the
CONDITIONED rankers (corpus=pearltrees, judge=kalman-fused for ELEM/HIER/SYM; judge=graph for
LINEAGE; nodetypes page→pearltrees_collection). Held-folder subset = queries whose true folder
never appeared in the fine-tune's train node set.

400 bookmark queries over 335 candidate folders (min_bm 3, seed 7; sorted-query manifest sha256
a1735e5b…; 47 queries hit duplicate-title folder sets, graded via title-equivalence best-alias
ranks — a title-keyed model cannot distinguish aliases).

**Headline (honest null, strengthened by the review corrections):** e5-cosine remains the best
filing ranker (recall@1 0.205, MRR 0.294 — unchanged between checkpoints by construction). After
fixing the anchor-reference and shared-W bugs and grading with equivalence sets, the fine-tune
does NOT improve the like-for-like conditioned ranker — mu-max-cond MRR moves 0.112 → 0.075
(paired −0.037) — and the stock agnostic rankers drift slightly down (margin-gate 0.206 → 0.191).
Only the LINEAGE head improves (MRR 0.029 → 0.044), and per the review that mostly measures
training a previously-random readout row. The earlier "+0.041" claim is RETRACTED as an artifact
of comparing against a random LINEAGE baseline. The campaign's value on Pearltrees is the label
factory (§2–§4), not — on this evidence — the ranking head.

| ranker (MRR, equivalence-graded) | base | tuned |
|---|---|---|
| e5-cos | **0.294** | **0.294** |
| margin-gate (e5⊕μ-max) | 0.206 | 0.191 |
| mu-max (agnostic) | 0.132 | 0.110 |
| mu-max-cond (like-for-like) | 0.112 | 0.075 |
| mu-elem-cond | 0.105 | 0.081 |
| mu-lineage | 0.029 | 0.044 |
| mu-max+lineage | 0.041 | 0.055 |

Transductive held-folder subset (endpoint-only definition; n=279): e5-cos MRR 0.302;
mu-max-cond 0.105 → 0.073 — same shape as pooled.

Escalation margins on the DEPLOYED ranker (e5-cos; identical for both checkpoints; kept_n in
parens) — descriptive only, full policy evaluation (judge rescue, AURC, cluster bootstrap) needs
judge labels on routed queries:

| threshold | routed | kept R@1 |
|---|---|---|
| 0.02 | 0.585 (166 kept) | 0.235 |
| 0.05 | 0.973 (11 kept) | 0.545 |

e5's margin is informative (kept-decision accuracy rises with the threshold), so margin-routed
escalation on e5 is viable as a POLICY SHAPE, but a calibrated threshold + judge-utility curve
is future work. The tuned μ heads' margins are not currently a usable routing signal.

## 6. Caveats

External-review residues (2026-07-17, addressed where noted):
- Loader now FAILS CLOSED (unique score keys, judge-column verification, zero missing luna/e5
  rows); the 300-row overlap selection is a committed sampler step with a manifest
  (indices sha + file sha) and reproduces the original draw byte-for-byte.
- The held-folder eval slice is TRANSDUCTIVE, not fully held: its definition is endpoint-only, and
  ~19/277 true folders still appear as ancestor CONTEXT tokens during training.
- Duplicate folder titles (7 titles / 19 folder ids; ~40/400 queries) are graded via
  title-equivalence sets (best-alias rank) — a title-keyed model cannot distinguish them.
- The routed-row SELECTION used the full-overlap innovation ordering, so the eval model's train
  set composition retains a weak dependence on held overlap labels (selection only, not targets).
- Fusion covariance is fit on fitted residuals (in-sample); the reviewer's out-of-fold diagnostic
  put the R-diagonal understatement at 0.7–3.2% (mild) — the oof machinery exists in
  run_sym_channel_fusion for the confirmatory enwiki harness.
- Escalation curves are DESCRIPTIVE margin diagnostics on the deployed ranker; the full policy
  evaluation (judge rescue accuracy, AURC, cluster bootstrap, cost curve) needs judge labels on
  routed queries — future spend.

- Partial-recovery DAG (396/880 multi-parent folders; 752 path records): coverage-limited campaign;
  the completed export re-runs this pipeline unchanged.
- All labels are judge fidelity, not gold truth; the human-verified subset remains the deferred
  absolute frame.
- The overlap (300) is below the design's 500–700 recommendation (budget choice, task-specified);
  the block fit is ~20 numbers on 300 rows — conditioned, but the shadow-gate rerun should use the
  larger overlap.
- The overlap ladder and shadow-bins comparison are single-split descriptive numbers (n=45 held).
- LINEAGE targets use pure hop decay (no lca_depth_frac) — see §4.
- eval_filing's Tokenizer runs without lineage context (empty parents — prior art parity); a
  lineage-aware candidate context is a plausible upgrade for the next round.

## Repro

```
python3 sample_product_kalman_pearltrees_campaign.py --paths-jsonl ../../.local/data/api_tree_paths_v8.jsonl \
    --titles-tsv ../../.local/data/pearltrees_api/assembled_titles.tsv --pairs 250 --hmax 5 --seed 0 \
    --pairs-tsv /tmp/mu_data/pt_campaign_pairs.tsv --score-in /tmp/mu_data/pt_campaign_score_in.tsv \
    --manifest /tmp/mu_data/pt_campaign_manifest.json
python3 sample_pearltrees_lateral.py --paths-jsonl ../../.local/data/api_tree_paths_v8.jsonl \
    --titles-tsv ../../.local/data/pearltrees_api/assembled_titles.tsv \
    --merge-pairs /tmp/mu_data/pt_campaign_pairs.tsv \
    --out-score-in /tmp/mu_data/pt_campaign_all_score_in.tsv \
    --out-pairs /tmp/mu_data/pt_campaign_all_pairs.tsv \
    --manifest /tmp/mu_data/pt_campaign_lateral_manifest.json
python3 prep_pearltrees_e5.py --paths-jsonl ... --titles-tsv ...
# luna bulk + 5.5 overlap + routed rows via score_with_codex.py (see §0 for sizes)
python3 run_pearltrees_fusion.py            # log: /tmp/mu_data/pt_fusion_output.txt
python3 migrate_name_tables.py --ckpt model_prod_namecond.pt --out model_prod_namecond_full.pt
python3 fine_tune_pearltrees_filing.py --out model_pt_filing.pt
python3 eval_pearltrees_filing.py --base model_prod_namecond_full.pt --tuned model_pt_filing.pt
```
