# ARC: the μ-attention filing program — corpora, training methods, and where we are

The narrative thread of the exploration: what we're building, which corpora we trained on and what each
one's pathology taught us, the training methods and losses we tried, the path-learning story, the
fusion/judge-economics arc, a dated PR ledger (Claude + Codex), and the current plan (Filing v1).
Written 2026-07-16. Companion to the per-result REPORT_*/DESIGN_* docs indexed in §9.

## 1. The goal

A **bookmark-filing assistant**: given a new bookmark (or mindmap node), rank the folders of a personal
hierarchy (Pearltrees, SimpleMind) by directional fuzzy membership μ(node | folder), escalating to an LLM
judge only where the decision is uncertain. Everything else in this program — operators, judges, fusion,
campaigns — exists to make that ranking accurate, calibrated, and cheap.

## 2. The model in one paragraph

`MuAttention` (mu_attention.py; DESIGN_directional_attention.md): a small permutation-invariant
transformer over a SET of tagged tokens — operator token, anchor(root), node, sampled ancestors, a
maskable provenance token (corpus ⊗ judge ⊗ account) — over **frozen e5-small** embeddings; sigmoid μ
readout per operator. Asymmetry is structural (root = e5 `query:`, candidates = `passage:`). Operators:
SYM (symmetric relatedness), HIER (category hierarchy), ELEM (element-of), LINEAGE / LINEAGE_RANK
(materialized-path filing). Since 2026-07-09, identity conditioning is a **name function**
(`cond = W·e5(card) + residual`, NameFunctionCond) for judges, operators, and corpora — new identities
onboard from a text card at zero residual (REPORT_judge_name_migration.md).

## 3. Corpora and their pathologies

| corpus | scale used | hierarchy character | the problems it taught us |
|---|---|---|---|
| enwiki categories (10k/100k TSV; `enwiki_cats_correct` scoped LMDB "Behavior" slice) | 10k→100k pairs; 2.25M-node content subtree | **many parents, no principal one** — multiple valid hierarchies | dense-DAG transitive closure explodes (multipath merged closure → shallow local context only); ~14% admin categories had to be excluded (3-dump "Correct" ingest, content root Main_topic_classifications); deep-hop pairs split directional-vs-unrelated → the D **bimodality** (class-mixture fix, REPORT_class_mixture.md); high branching makes hop count only a proxy for D |
| simplewiki | earlier WAM/benchmark work; corpus slot 0 | as enwiki, smaller | mostly a scale stepping-stone |
| **Pearltrees** | 5,004 trees exported (truncated); 396/880 multi-parent filing folders recovered | **user-curated, has a principal parent** (the filing choice) | RDF export silently truncates ~24MB (#3400, 2026-07-02, partial recovery; chunked re-export + API backfill still pending — Filing v1 step 1); **misspellings/typos masquerade as lateral semantic drift** (user diagnosis) — addressed by Codex's title-policy/audit work (#3600/#3603 cross-corpus campaign 2026-07-09; title policies + audit manifests #3627/#3635 2026-07-09/10, REPORT_product_kalman_title_policies.md); OOD vs Wikipedia style (mu-pearltrees-ood #3392, 2026-07-01) |
| SimpleMind mindmaps (.smmx) | 491 lineage chains from 6 maps | single-parent tree | small data (needs fusion with other corpora to matter); privacy filtering in parse; "Root Node" sentinel dropped; harvesting more maps needs the .local harvester (gen_mindmap_lineage.py; DESIGN_mindmap_lineage.md) |

The corpus contrast drives the architecture: Wikipedia's no-principal-parent forced multipath/closure
thinking and the bimodality machinery; Pearltrees' principal parent makes single-path lineage correct and
puts data QUALITY (typos, truncation) rather than ambiguity at the top of its risk list.

## 4. Path learning: Pearltrees vs Wikipedia (and where cross-entropy came in)

- **LINEAGE** (op 4): materialized-path filing trained with **graded MSE** on μ magnitude along the
  root→…→node path (mindmap-lineage methodology #3426, 2026-07-03).
- **LINEAGE_RANK** (op 5): the same paths trained with **candidate-softmax cross-entropy** — the loss is
  over the ORDER of candidate folders, not the μ value (#3436, 2026-07-03; DESIGN_mindmap_lineage.md §3c).
  This is the CE exception the program's mostly-MSE training has: filing is ultimately a ranking decision,
  so the rank head optimizes exactly that.
- **PATH multipath experiment** (DESIGN_path_operator.md, #3395 2026-07-01; result recorded in
  REPORT-level memory 2026-07): multipath does NOT beat single-path LINEAGE on Pearltrees (3-seed;
  subset branch-recovery negative). **Decision:** single-path LINEAGE for Pearltrees (principal parent
  exists), multipath PATH reserved for Wikipedia (no principal parent). Dense-DAG merged closure explodes
  → keep Wikipedia context shallow/local.

## 5. Training methods, in order (with losses)

All training minimizes error (MSE on μ targets) unless noted; the model is always warm-started
("build up by fine-tuning" philosophy); torch jobs run one at a time (consumer WSL).

| when (2026) | method | PRs (sample) | what it added / lesson |
|---|---|---|---|
| June (early) | base directional attention + graded rounds; bridges to Wikipedia; 4-layer base | (pre-#3355 foundations) | frozen-e5 + tags is enough for direction; REPORT_directional.md, REPORT_graded_training.md, REPORT_4l_base_train.md |
| 06-25..27 | **inferred-operator superposition / blend regularizer** — operator as a random Dirichlet superposition; tagged-blend as regularizer; capacity pairing (3L vs 4L) | #3356–#3373 | uncertainty as INPUT noise; isolated-RNG A/B discipline (89→94% honest, not 97%); blend needs capacity headroom; DESIGN_inferred_operator_superposition.md |
| 06-27..28 | **anchored basis**: frozen e5 phrase anchors ++ K learnable atoms, anchor-confidence KL | #3370, #3371, #3374 | the open-set pattern later reused for judge NAME conditioning; explicit P(op) supervision did NOT pay (REPORT_anchored_basis_ab.md) |
| 06-28..29 | **Haiku tail augmentation** — LLM E[μ] targets on the inferred tail, `--tail-weight 6` | #3375, #3376 | +0.12 held-tail corr (3-seed); dilution is real; §14 judge prompt contract born here (REPORT_haiku_tail_pilot.md) |
| 06-29 | **transitive relations arc** (incl. heteroscedastic A/B) | #3377–#3384 | multi-hop μ targets; hetero weighting groundwork |
| 06-30..07-01 | production/hybrid + retrieval + coverage/data-quality + self-annealing + LLM rerank | #3385–#3395 | the APPLICATION scaffolding (DESIGN_model_applications.md) — what Filing v1 returns to |
| 07-02..03 | **Pearltrees data completion; mindmap LINEAGE + LINEAGE_RANK (CE)** | #3400, #3426, #3436 | §4 above |
| 07-03..05 | **SYM dual judge** (e5 ⊕ graph precision blend), ELEM membership ablation, cross-judge direction blend, two-judge posterior | #3445, #3464, #3488, #3496–#3517 | graph-as-judge enters the model; measured constants over learned weights where data is thin |
| 07-06..09 | **product-Kalman fusion program** (see §6) | #3527–#3618 | fusion theory validated on real data |
| 07-09 | **provenance channel heads**: B1 (judge rows only) → B1b (+last layer, agnostic-anchor distillation loss) → stratified campaign → S channel established | #3591, #3592, #3613, #3614 | DATA (stratification), not capacity, was the S bottleneck; within-stratum decomposition = the honest eval (+0.31–0.36, not the pooled +0.74) |
| 07-09 | **name-function migration + fused head + luna onboarding + class mixture (B2)** | #3621 | behavior-preserving init (ridge-lstsq W, r = old − W·e); fused-head **null** with a reliable judge (R=0.004 ⇒ posterior≈label); luna residual captures its +D/−S tilt with exact isolation; class-mixture passes the §11.5 gate (+0.34–0.67 nats) |
| 07-09..10 | **multi-judge fusion + luna campaign + cheap-judge pipeline** | #3623, #3634, #3648 | luna fusion is non-degenerate (pull 0.08–0.13); fused head's first win (D, 6/6 seed cells); §7 luna verdict was a stratum artifact (S 0.48–0.61 on laterals); matched-cost: cheap scheme wins at low coverage, k≥4 |
| 07-10 | **#3648 validity corrections** (delegated Opus agent) | on #3648 | campaign-independent prior; budget accounting; **luna debiasing FLIPPED "free tier beats cheap judge on S"**; paired per-split stats; Cholesky conditioning + tests — the strongest methodological lesson of the program: bias correction before covariance fitting |

Loss inventory: MSE on μ (nearly everything); **cross-entropy** for LINEAGE_RANK (filing order);
KL for anchor-confidence (anchored basis) and distillation anchors; L2 residual regularizers;
heteroscedastic MLE (per-row known noise) in the analytic statlin fits — flagged as the Tier-1 upgrade
for distillation losses (weight fused-target rows by posterior precision).

## 6. The fusion / judge-economics arc (condensed; 07-06 → 07-10)

Σ(hop) confirmatory pass (PAPER_sigma_hop_confirmatory.md) → fusion rungs on real data
(REPORT_product_kalman_{realdata,logit,gated,statlin,atoms}.md; #3527–#3580s, Claude results + ~30 Codex
infra PRs) → champion G_sl (statistical linearization; dual μ/logit mixture for distribution, μ expert for
points) → Lever A: judge as measurement channel, R_judge≈0.004, conflict routing, decision-flip value
(#3584) → campaigns (#3613) → name architecture + fused head + luna (#3621–#3648, §5 rows above).
Theory map: THEORY_evidence_fusion.md; the two-timescale design: DESIGN_amortized_fusion_heads.md;
scheme + batch-vs-dynamic statistics ladder: DESIGN_cheap_judge_pipeline.md (figures/*.png).

**Codex's post-#3648 theory chain** (≈07-10→15, from the PR titles): #3651 post-3648 validation + GPU
square-root/QR conditioning → #3666 batched sqrt/QR with correlation whitening → #3671 gate structured
residual covariance → #3675 covariance sensitivity + adjacency-aware batching → #3685 adjacency-residual
confidence, component-safe folds → #3695 PSD graph-geometry audit → #3701 preregistered repeated-judge
covariance campaign → #3707/#3726 fail-closed capacity/topology audits → #3735 dependence-aware
source-region topology bridge → #3742 corpus-specific source-dependence power harness → **Stage-A
immutable power run live** (results: ~/UnifyWeaver-runs/repeated-judge-source-power/d7e28a91/result.json).
Codex's three standing questions: graph geometry; how to measure correlation; how much correlation is
safe in inverse-square-root propagation (Householder/Potter). Division of labor (2026-07-10): **Codex =
theory/tools; Claude = application + training; Grok = delegated self-contained scripts.**

## 7. Where we are: Filing v1 (the application refocus)

The research earned what filing needs; methodology hit diminishing returns. Plan (memory:
project_filing_v1_arc.md):
1. Pearltrees data completion (chunked re-export + API backfill) — Grok-delegated script.
2. Label Pearltrees with the validated cheap pipeline (luna bulk + ~300-row random 5.5 overlap, debiased
   fusion, conflict-routed 5.5, sonnet-5-low tiebreaks).
3. Fine-tune the filing model (corpus card onboards by name; single-path LINEAGE; champion recipe).
4. Evaluate on the FILING metric (placement accuracy / rank of correct folder; decision-flip escalation).
5. Assistant loop (rank folders, escalate on conflict) — Grok-delegated CLI shell around our scorer.
sqrt-KF in training: target factory + Cholesky parameterization; no in-loop filter layer yet.

## 8. PR ledger (merged; both authors)

Foundations: #3355–#3395 (06-25→07-01, Claude — superposition, anchored basis, tail, transitive,
applications scaffolding). Data: #3400 Pearltrees completion (07-02); #3426/#3436 mindmap lineage
MSE/CE (07-03). Judges: #3445/#3464 SYM dual judge (07-03/04); #3488 ELEM membership (07-05);
#3496–#3517 blends + two-judge posterior (07-05/06). Fusion: #3527–#3598 product-kalman core + infra
(07-07→09, mostly Codex); Claude result rungs #3565 realdata, #3567 logit, #3570 gated, #3575 statlin,
#3577 atoms, #3579 theory, #3584 judge channel (07-09). Campaigns: #3600/#3603/#3610/#3618 cross-corpus
samplers incl. Pearltrees/SimpleMind + title/typo handling (Codex, 07-09); #3591/#3592 channel heads,
#3613 campaign, #3614 within-stratum (Claude, 07-09). B2 + economics: #3621 name-cond arc, #3623
multi-judge fusion, #3627/#3635 title audits (Codex), #3634 luna campaign (07-09/10). Pipeline: #3648
cheap-judge pipeline + corrections (07-10). Theory chain: #3651→#3742 (Codex, ≈07-10→15, §6).

## 9. Doc index (by theme)

- **Architecture**: DESIGN_directional_attention.md, DESIGN_provenance_and_representation.md,
  DESIGN_calibrated_judges.md, DESIGN_mu_sources_and_estimation.md.
- **Operators/paths**: DESIGN_path_operator.md, DESIGN_lineage_decoder.md, DESIGN_mindmap_lineage.md,
  DESIGN_transitive_relations.md, DESIGN_inferred_operator_superposition.md (+
  NOTES_model_selection_capacity.md), REPORT_element_operator.md, REPORT_transitive_*.md.
- **Judges/fusion theory**: THEORY_evidence_fusion.md, DESIGN_uncertainty_estimation_playbook.md,
  DESIGN_sym_dual_judge.md, DESIGN_sym_estimation_integration.md, DESIGN_cross_judge_direction.md,
  DESIGN_two_judge_posterior.md, DESIGN_product_kalman_poe.md, DESIGN_amortized_fusion_heads.md,
  DESIGN_cheap_judge_pipeline.md, PAPER_sigma_hop_confirmatory.md.
- **Fusion results**: REPORT_product_kalman_{realdata,logit,gated,statlin,atoms}.md,
  REPORT_judge_channel.md, REPORT_two_judge_posterior.md.
- **Channels/campaigns**: REPORT_channel_heads_probe.md, REPORT_channel_heads_b1.md,
  REPORT_channel_campaign.md, REPORT_judge_name_migration.md, REPORT_fused_head.md,
  REPORT_luna_campaign.md (⚠ predates luna debiasing — don't build on its exact numbers),
  REPORT_class_mixture.md, REPORT_multi_judge_fusion.md, REPORT_cheap_judge_baseline.md (post-correction
  numbers are authoritative).
- **Corpora/data quality**: DESIGN_wikipedia_sampling.md, DESIGN_graph_widening.md,
  REPORT_widen_enwiki.md, DESIGN_product_kalman_cross_corpus_campaign.md,
  REPORT_product_kalman_{pearltrees,simplemind}_sampling.md, REPORT_product_kalman_title_policies.md
  (the typo/data-quality work), REPORT_product_kalman_enwiki_topology.md.
- **Training/eval method**: DESIGN_sampling_and_grading.md, REPORT_graded_{round,training}.md,
  REPORT_haiku_tail_pilot.md, REPORT_eval_methodology.md, REPORT_train_consolidation.md,
  REPORT_anchored_basis_ab.md, REPORT_infer_blend*.md, REPORT_tagged_blend_sweep.md.
- **Application**: DESIGN_model_applications.md (retrieval, rerank, production/hybrid — Filing v1's
  scaffolding).
