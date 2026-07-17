# Pearltrees Filing v1: first live deployment of the cheap-judge labeling pipeline

Filing v1 steps 2–4 (ARC_filing_program.md §7): label the Pearltrees corpus with the validated
cheap-judge recipe (DESIGN_cheap_judge_pipeline.md — first production run), fine-tune the filing
model on the fused targets, and evaluate on the FILING metric (rank the true folder for real
bookmarks). Evaluation frame throughout: fidelity to the gpt-5.5-low operating judge (labels) and
to the user's actual filing decisions (retrieval eval) — never "semantic accuracy".

**Summary.** The labeling pipeline deployed cleanly (799 dual-strata pairs, 0 scoring failures,
every fusion rung pays); luna's D bias FLIPS SIGN on Pearltrees vs enwiki (§1 — the per-corpus
overlap is not optional); the shadow bias-states show larger bin structure than enwiki and win
their descriptive ladder (§3 — gate-protocol rerun recommended); the fine-tune's fused head beats
the raw-luna channel within-stratum and its LINEAGE head learns lineage order (§4). On the filing
metric itself the result is an HONEST NULL vs the deployment baseline: e5-cosine still ranks
folders best (MRR 0.294); the fine-tune improves the model's own μ rankers (paired +0.041 MRR)
and makes escalation margins informative, but μ does not yet beat e5 at ranking (§5). Deployment
recipe today: e5 ranks, the fused model routes conflicts and manufactures labels.

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

Run (seed 0): 1,839 train rows from 279 node-disjoint train pairs; loss 0.149 → 0.017 in 800
steps; 1.64M trainable of 4.02M params. Held-overlap readout (49 held 5.5-labeled rows;
pooled / between / WITHIN-stratum corr vs the 5.5 labels):

| head | D within | S within |
|---|---|---|
| kalman-fused (distilled) | +0.061 | **+0.187** |
| luna channel (raw cheap labels) | +0.075 | +0.096 |
| 5.5 channel (expensive labels) | +0.020 | +0.178 |

The fused head's within-stratum S beats the raw-luna channel and edges the direct-5.5 head —
fuse-then-distill transfers to Pearltrees (n=49, one split; direction consistent with the enwiki
fused-head result). D within-stratum is small for every head — Pearltrees principal rows are
near-saturated in D (the up-walk is deterministic), so within-stratum D variance is thin.
Held LINEAGE (38 principal rows): rank-corr **+0.608**, MAE 0.396 — the single-path head learns
the ORDER of lineage decay but is level-miscalibrated after one fresh readout row × 800 steps
(rank is what filing consumes; the MSE level is a follow-up).

Checkpoint: `model_pt_filing.pt` (gitignored;
SHA-256 `c0a2f731876cf81a668d2e93f1f71337fced2528e3c7a40e946022e759378ac9`).

## 5. FILING metric (Filing v1 step 4 — the deliverable)

`eval_pearltrees_filing.py`: real filing decisions (harvested tree JSONs; ground truth = each
bookmark's actual treeId), candidates = folders with ≥3 bookmarks, base vs fine-tuned. The
agnostic-anchor loss pins the agnostic readouts by design, so the fine-tune is read through the
CONDITIONED rankers (corpus=pearltrees, judge=kalman-fused for ELEM/HIER/SYM; judge=graph for
LINEAGE; nodetypes page→pearltrees_collection). Held-folder subset = queries whose true folder
never appeared in the fine-tune's train node set.

400 bookmark queries over 335 candidate folders (min_bm 3, seed 7); held-folder subset 277/400.

**Headline (honest null vs the deployment baseline):** e5-cosine remains the best filing ranker
(recall@1 0.198, MRR 0.294) — unchanged by construction between checkpoints. The fine-tune
improves the model's OWN conditioned/LINEAGE heads (paired MRR delta +0.041 on mu-max+lineage;
mu-lineage 0.021 → 0.050) but every μ ranker stays well below e5-cos on this OOD retrieval task.
Filing v1 deployment should ship e5-cos (or the margin-gate e5⊕μ blend) as the ranker, with the
fused model as the label factory and escalation policy — not (yet) as the ranker itself.

| ranker (MRR) | base | tuned |
|---|---|---|
| e5-cos | **0.294** | **0.294** |
| margin-gate (e5⊕μ-max) | 0.204 | 0.187 |
| mu-max (agnostic) | 0.130 | 0.120 |
| mu-elem-cond | 0.105 | 0.109 |
| mu-max-cond | 0.106 | 0.107 |
| mu-lineage | 0.021 | **0.050** |
| mu-max+lineage | 0.031 | **0.073** |

Held-folder subset (true folder unseen in fine-tune train nodes, n=277): mu-max-cond MRR
0.105 → 0.111, mu-max+lineage 0.033 → 0.064 — the conditioned gains generalize past the trained
folders; e5-cos 0.280 there. Note the small DOWNWARD drift of the agnostic rankers
(margin-gate 0.204 → 0.187): the anchor loss pins agnostic readouts only on campaign pairs, not
on the filing-eval distribution — a real (small) cost of the fine-tune to price in future rounds.

Escalation curve (route to the judge when the mu-max+lineage top-2 margin < t; kept-R@1 =
recall@1 among non-routed decisions):

| threshold | base routed | base kept R@1 | tuned routed | tuned kept R@1 |
|---|---|---|---|---|
| 0.02 | 0.795 | 0.012 | 0.498 | 0.030 |
| 0.05 | 0.987 | 0.000 | 0.775 | 0.044 |
| 0.10 | 1.000 | — | 0.913 | 0.114 |
| 0.15 | 1.000 | — | 0.968 | 0.154 |
| 0.20 | 1.000 | — | 0.995 | 0.500 |

The tuned model's margins are INFORMATIVE (kept-decision accuracy rises steeply with the
threshold; the base model's margins are degenerate — everything routes). That is the escalation
policy working in miniature, but at today's accuracy nearly all decisions still route: the model
earns its keep as the conflict-router and label factory, while e5 ranks.

## 6. Caveats

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
