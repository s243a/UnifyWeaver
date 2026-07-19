# Filing-engine architecture

Last updated: 2026-07-18

This is the map of the filing system that has materialized across the
`mu_cosine` work. It is an exploratory architecture document, not a
preregistration, audit, or substitute for the linked reports and protocols.
It may preserve hypotheses before they are proven, provided their maturity is
explicit.

## How to read and update this map

Every architectural element is one atomic row with one stable ID and exactly
one classification tag:

- `DECIDED` — a standing policy, convention, or adopted empirical operating
  conclusion. Its row includes a one-line evidence pointer.
- `IDENTITY` — a mathematical consequence of the stated model. Its row points
  to the derivation; changing it means changing the model or correcting an
  error, not revising a decision.
- `DEFAULT` — the working choice. It may be revised without a promotion
  ceremony.
- `OPEN` — unresolved. Its row names the owner and the evidence that would
  resolve it.

An empirical operating conclusion also records an **Envelope** (the scale and
coverage of its support) and a **Revisit trigger** (the named scale event that
reopens it). A scale-triggered revisit is a normal change-log event, not a
retraction—the map expects its empirical rows to move.

Append new rows instead of recycling IDs. When an item changes classification, update
that row and add a dated entry to the change log; do not rewrite unrelated
history. Promotion protocols and audits belong in separate documents and PRs.

| ID | Tag | Operating shape | Evidence |
|---|---|---|---|
| FLOW-001 | `DEFAULT` | Run outcome-blind candidate generation, frozen e5-cosine ranking, an optional thin feature blend, top-two-margin routing, and then the user-facing bookmark-filing workflow. Graph and judge machinery primarily improves candidates, labels, calibration, and routing; it does not replace e5 without held evidence. | The layer records below connect the [filing program arc](../../prototypes/mu_cosine/ARC_filing_program.md), [Filing v1](../../prototypes/mu_cosine/REPORT_pearltrees_filing_v1.md), and the [bookmark-filing skill](../../skills/skill_bookmark_filing.md). |

## 1. Ranking layer

| ID | Tag | Architectural element | Evidence, owner, or resolution |
|---|---|---|---|
| RANK-001 | `DECIDED` | Frozen e5 cosine is the filing ranking backbone. The learned μ heads are not the primary ranker. | [Pearltrees candidate-lineage report](../../prototypes/mu_cosine/REPORT_pearltrees_candidate_lineage.md): e5 MRR was 0.294 while μ-head MRRs were at most about 0.11; [Filing v1 report §5](../../prototypes/mu_cosine/REPORT_pearltrees_filing_v1.md) confirms the corrected null. **Envelope:** Pearltrees only: 400 real-placement queries; the base harness used 335 candidate folders, while the audited lineage rerun used 323 folders with 253 path-covered; the μ fine-tune updated 1.20M parameters and the comparator was frozen `intfloat/e5-small-v2`. **Revisit trigger:** completed Pearltrees export against a frozen non-transductive catalog, or at least 10× as many placement labels. |
| RANK-002 | `DECIDED` | Candidate recall is the current binding ceiling: over the fixed e5 top-50 candidate set, recall of 0.680 means no reranker can recover the remaining 0.320. | [PR #3865 corrected report](https://github.com/s243a/UnifyWeaver/pull/3865), candidate-recall readout. **Envelope:** corrected Pearltrees run with a 1,200-query manifest, 335-folder transductive catalog, e5 top `K=50`, and a 6,014-node partial graph; five overlapping repeated node-disjoint splits exposed only 699/1,200 queries as held. **Revisit trigger:** the first frozen non-transductive catalog, or completed Pearltrees export with the candidate generator rerun. |
| RANK-003 | `DEFAULT` | Keep the graph-derived ranker as a thin feature blend over e5, not a replacement model. Its corrected result is descriptive and prospectively untested. | [PR #3865](https://github.com/s243a/UnifyWeaver/pull/3865) retracted the original win after statistical corrections and a partial Dirichlet correction; the corrected blend is only descriptively ahead on exposed data and still requires an exact boundary rerun. |
| RANK-004 | `OPEN` | **Owner: engineering-lane; evaluation boundary: rigor-lane.** Build a placement-blind candidate generator that mixes e5 retrieval with structural candidates such as ancestors, siblings through common parents, and permitted topology-only sources. | Resolve by freezing the generator before held results, improving recall at matched `K=50`, and then improving prospective node-disjoint filing MRR under [PR #3875](https://github.com/s243a/UnifyWeaver/pull/3875). No placement count, filing label, or judge outcome may enter eligibility. |
| RANK-005 | `OPEN` | **Owner: rigor-lane; implementation support: engineering-lane.** Better bounded graph domains are the second priority lever because the current two-hop feature is dominated by its artificial boundary. | Resolve with nested-domain comparisons against a larger exact-Dirichlet reference: raw Green responses, normalized screening values, candidate ranks, cut-current/envelope diagnostics, conditioning, memory, and runtime must all be reported. See [local grounded diffusion](LOCAL_GROUNDED_DIFFUSION.md). |
| RANK-006 | `DEFAULT` | Route escalation using the e5 top-one minus top-two score margin. Treat existing curves as descriptive, and do not call the margin a calibrated probability. | [Filing v1 report §5](../../prototypes/mu_cosine/REPORT_pearltrees_filing_v1.md) found the e5 margin useful while tuned μ margins were not; the correlated-calibration pattern is summarized in the [uncertainty playbook](../../prototypes/mu_cosine/DESIGN_uncertainty_estimation_playbook.md). |

## 2. Geometry layer

| ID | Tag | Architectural element | Evidence, owner, or resolution |
|---|---|---|---|
| GEOM-001 | `DECIDED` | Grounded diffusion is the shared numerical primitive: topology supplies edge support, frozen semantic similarity modulates conductance on those edges, and positive leakage makes the precision strictly positive definite. | [Leaky graph diffusion](LEAKY_GRAPH_DIFFUSION.md), [local grounded diffusion](LOCAL_GROUNDED_DIFFUSION.md), and [graph-geometry decisions](../../prototypes/mu_cosine/DECISIONS_graph_geometry.md) record the #3852/#3857/#3867 design and correctness boundary. |
| GEOM-002 | `IDENTITY` | For a retained domain, deleting cut edges under a zero-bath Dirichlet boundary adds their conductances to the retained endpoints as the exact diagonal shunt `β`; silently dropping them changes the operator. | [Local grounded diffusion, “The exact Dirichlet cut”](LOCAL_GROUNDED_DIFFUSION.md) derives this retained principal block. |
| GEOM-003 | `DEFAULT` | Hop-based top-`K` selection is the transparent bounded-domain baseline; Nomic-resistance Dijkstra is the sanctioned semantic sensitivity. | [Local grounded diffusion, “Choosing the local domain”](LOCAL_GROUNDED_DIFFUSION.md) defines both selectors, deterministic ties, and provenance. |
| GEOM-004 | `OPEN` | **Owner: rigor-lane.** Replace boundary-dominated two-hop balls with a candidate-skeleton family: ancestors and common parents first, union all batch anchors, then attach informative local branches within a resource budget. | Resolve against nested larger-domain references. The current structural warning is 22% boundary nodes and an infeasible e-fold target in [PR #3865](https://github.com/s243a/UnifyWeaver/pull/3865). Acceptance requires stable raw responses and ranks, controlled cut current, and material memory/runtime savings. |
| GEOM-005 | `IDENTITY` | Exact reduction rules are constraints on every candidate skeleton: a one-port subtree may be Kron-reduced to a scalar shunt; an ungrounded dangling subtree has zero DC shunt; eliminating any multi-port subtree, including a tree, creates boundary-to-boundary fill. Fill is not a cycle-only effect. | These are direct Laplacian/Schur-complement identities and refine the exact-marginal alternative deferred in [Local grounded diffusion, “Rejected and deferred alternatives”](LOCAL_GROUNDED_DIFFUSION.md). |
| GEOM-006 | `OPEN` | **Owner: rigor-lane.** Test a budgeted sparse **graph-derived** boundary closure as an outcome-blind approximation to exterior Schur fill. An actual path through the represented exterior must license and weight every coupling; frozen semantic geometry may only prioritize which graph-connected port pairs receive expensive resistance/Schur calculations. Closure must **reallocate** the cut mass at each boundary between induced couplings, self-return, and residual grounding; it must not add couplings on top of the full Dirichlet `β`. | Resolve under the [prospective bounded-diffusion fidelity protocol](../../prototypes/mu_cosine/PROTOCOL_bounded_diffusion_fidelity.md) by matching a larger-domain or exact small-graph Schur reference better than plain Dirichlet at the same node budget while preserving symmetry, nonnegative conductances, M-matrix/SPD safety, and an auditable per-boundary mass ledger. An empty or negligible closure after common-parent retention is a valid result. |
| GEOM-007 | `OPEN` | **Owner: rigor-lane.** Test entropy/generality-weighted base conductance `c₀` as a new symmetric topology-only hypothesis for controlling hub leakage. The definition must be frozen, symmetric, nonnegative, and have a positive floor; `α` must be recalibrated after reweighting. Lower leakage is plausible, not automatic. | Resolve with cross-corpus diagnostics frozen before outcomes: hub leakage, Green responses, relational accuracy, rank stability, and conditioning. Pearltrees and a multi-parent corpus must both be represented. |
| GEOM-008 | `OPEN` | **Owner: user, with evidence from both lanes.** Rigor position: infeasible e-fold calibration should fail closed for a calibrated `h_s`. Engineering position: an SPD, minimum-`α` raw screening score may remain usable if explicitly labeled degraded. | Resolve by freezing both modes before outcomes, comparing each with a larger-domain reference, and reporting downstream utility/risk, coverage, and resource cost. A degraded raw value may never be described as calibrated. |
| GEOM-009 | `DECIDED` | For multi-anchor screening, select one union-of-anchors domain and use one shared precision matrix and factorization; never splice separately normalized per-anchor solves. | [Local grounded diffusion, “Multiple anchors and positive semidefiniteness”](LOCAL_GROUNDED_DIFFUSION.md) records the one-domain/one-factor convention needed for a coherent PSD geometry. |
| GEOM-010 | `IDENTITY` | A memoized traversal collision means only that a node or component was already discovered; it may indicate path convergence or a cycle, but it is not an articulation/cut point. A Dirichlet cut edge is defined by the retained/omitted partition, while articulation status requires an explicit topology calculation. | The [prospective bounded-diffusion fidelity protocol](../../prototypes/mu_cosine/PROTOCOL_bounded_diffusion_fidelity.md) applies this distinction to exterior-component discovery and forbids collision-induced grounding. |

### Grounded-diffusion authorization boundaries

The global primitive's boundary is quoted exactly from
[Leaky graph diffusion, Status and scope](LEAKY_GRAPH_DIFFUSION.md):

> The implementation in src/unifyweaver/graph/leaky_diffusion.py is a dense
> float64 correctness reference. It establishes the algebra and fail-closed API;
> it does not claim a CUDA crossover or authorize a learned cross-item
> covariance. The completed Stage-A repeated-judge source-power experiment
> failed closed and, within that dependence campaign, unlocked no
> dependence-candidate enumeration, Nomic packing, judge calls, covariance
> deployment, independent batching, QR specialization, or CUDA claim.
> This primitive therefore remains outcome-blind infrastructure.
>
> Separate deterministic application uses do not depend on that failed
> dependence gate: frozen-geometry candidate generation, raw per-anchor
> screening values as ranking features, and routing are allowed when geometry,
> leakage, and thresholds are chosen without placement or judge outcomes.
> They are not calibrated probabilities, learned covariance entries, or
> authorization for independent batching, QR, or sparse/CUDA claims.

The local primitive's boundary is quoted exactly from
[Local grounded diffusion, Status and scope](LOCAL_GROUNDED_DIFFUSION.md):

> Separate outcome-blind application uses are authorized now: frozen-geometry candidate generation, per-anchor `h_s` screening values as ranking features (not calibrated probabilities or covariance entries), and routing.
>
> Selection of `D`, `K`, `ell`, `epsilon`, `alpha`, and screening thresholds may use topology, frozen embeddings, resource limits, and numerical diagnostics, but no placement or judge outcomes.
>
> Any learned downstream consumer still requires train-only selection and held evaluation.
>
> This does not reopen the failed Stage-A dependence campaign or authorize judge covariance, independent batching, QR, sparse/CUDA, or performance claims.

## 3. Teaching layer

| ID | Tag | Architectural element | Evidence, owner, or resolution |
|---|---|---|---|
| TEACH-001 | `DECIDED` | Distinguish **walk as teacher** from **walk as inference feature**. When teaching succeeds, one expected manifestation is inference redundancy because the relational lesson has been distilled into the prior; that redundancy does not establish teaching irrelevance. | [Transitive-relations design](../../prototypes/mu_cosine/DESIGN_transitive_relations.md) defines the curriculum, while [PR #3865](https://github.com/s243a/UnifyWeaver/pull/3865) supplies only the narrower inference-feature null. |
| TEACH-002 | `DECIDED` | Graph judge v1 uses topology-derived `hit_prob` targets and an explicit transitive-relations curriculum as free, dataset-familiarizing supervision. | [Multihop-direction report](../../prototypes/mu_cosine/REPORT_multihop_direction.md) defines `hit_prob`; [transitive verification](../../prototypes/mu_cosine/REPORT_transitive_verification.md) validates the curriculum, and the [post-#3648 validation](../../prototypes/mu_cosine/REPORT_cheap_judge_post3648_validation.md) bounds the free channel's role. |
| TEACH-003 | `OPEN` | **Owner: rigor-lane for semantics and acceptance; engineering-lane for implementation.** Graph judge v2 is either a semantically gated **directed killed walk**, or a separate directional channel plus symmetric grounded diffusion only for proximity/screening. Symmetric resistor diffusion is not a drop-in replacement for directional parent-walk targets. | Resolve on multiple axes: hop-conditioned bias, NLL/calibration, relational accuracy against independent labels, and residual covariance. A flatter `Σ(hop)` alone is insufficient because lower variance can conceal systematic bias. |
| TEACH-004 | `DEFAULT` | Use the cheap-judge system as the current label factory: retain train-only global affine debiasing, correlated fusion, and the free graph channel as cheap supervision. Debiased Luna is the paid S workhorse in the present coverage regime. | [Cheap-judge baseline](../../prototypes/mu_cosine/REPORT_cheap_judge_baseline.md) records Luna debiasing and the graph-S increment; [post-#3648 validation](../../prototypes/mu_cosine/REPORT_cheap_judge_post3648_validation.md) shows that exact-budget economics change with coverage and partition. **Envelope:** 1,700 pair-matched `gpt-5.5-low`/Luna enwiki rows across exploratory and fresh corpora and 40 partitions; strict matched-cost economics covered `n={80,160}` on one held-node split with 10 paired subsamples; the first live Pearltrees factory covered 799 pairs. **Revisit trigger:** at least 10× labels, or the completed Pearltrees export rerun across frozen node partitions. |
| TEACH-005 | `DEFAULT` | Retain each judge's train-only global affine correction as the production calibration layer. | [Bias-state report §0](../../prototypes/mu_cosine/REPORT_bias_states.md) and the [bias-state design](../../prototypes/mu_cosine/DESIGN_bias_state_augmentation.md) preserve global affine calibration before any residual state offsets. |
| TEACH-006 | `DEFAULT` | Keep per-distance/bin bias states in shadow mode. The descriptive NLL improvement is encouraging but the global uncertainty interval includes zero, so bins remain opt-in diagnostics rather than a promoted production layer. | [Bias-state report](../../prototypes/mu_cosine/REPORT_bias_states.md) records the encouraging null and false promotion gate. |
| TEACH-007 | `DECIDED` | In the tested Wikipedia regime, use smooth `Σ(hop)` as an empirically supported teacher row-noise model, not as proof that graph-judge v2 is unbiased or that the model transfers unchanged to other corpora. | [Confirmatory Sigma-hop report](../../prototypes/mu_cosine/REPORT_sigma_hop_confirmatory.md) confirms hop dependence on a fresh slice; `TEACH-003` keeps bias and relational validity as separate acceptance axes. **Envelope:** one fresh no-overlap Wikipedia `Behavior` slice of 75,901 nodes; 250 `gpt-5.5-low` pairs (50 per hop 1–5), 40 descendant-disjoint splits averaging 75 held pairs, and 1,000 hop-shuffle permutations. **Revisit trigger:** the first full enwiki-scale run, or transfer to another corpus or judge family. |

## 4. Fusion and calibration layer

| ID | Tag | Architectural element | Evidence, owner, or resolution |
|---|---|---|---|
| FUSE-001 | `DECIDED` | The dense correlated-Gaussian conditioner is the interpretable fusion baseline. Correlated sources are modeled jointly rather than combined by hand-set confidence weights. | [Uncertainty playbook](../../prototypes/mu_cosine/DESIGN_uncertainty_estimation_playbook.md), [joint square-root/QR design](../../prototypes/mu_cosine/DESIGN_joint_square_root_qr_conditioner.md), and [JointPosterior report](../../prototypes/mu_cosine/REPORT_cheap_judge_joint_posterior.md) retain the dense Gaussian as the strong control. |
| FUSE-002 | `DEFAULT` | Keep `JointPosterior` as a different nonlinear learned model and comparator; it is not a numerical fallback, not the QR conditioner, and has not beaten the dense Gaussian control. The square-root/QR path is the numerical alternative for conditioning. | [JointPosterior report](../../prototypes/mu_cosine/REPORT_cheap_judge_joint_posterior.md) and [post-#3648 validation](../../prototypes/mu_cosine/REPORT_cheap_judge_post3648_validation.md) report paired intervals including zero. |
| FUSE-003 | `DECIDED` | Encode judge, operator, and corpus identity through name-conditioned parameters so checkpoints remain semantically stable as identities are added. | [Judge-name migration report](../../prototypes/mu_cosine/REPORT_judge_name_migration.md) documents the mechanism; judge identity landed in #3621 and operator/corpus identity in #3826, superseding stale merge-state wording in the program arc. |
| FUSE-004 | `DECIDED` | μ heads serve calibration, label fusion, and conflict routing—not primary filing ranking. | [Filing v1 report](../../prototypes/mu_cosine/REPORT_pearltrees_filing_v1.md) and [candidate-lineage report](../../prototypes/mu_cosine/REPORT_pearltrees_candidate_lineage.md) show the label-factory value and the ranking null. **Envelope:** Pearltrees only: 799 dual-strata pairs; the leakage-free factory fit used 108 overlap-train rows and one read on 49 held labels; filing rank comparison used 400 queries over 335 folders; the fine-tune exposed 1.20M trainable parameters. **Revisit trigger:** at least 10× judge-overlap labels with a completed-export, non-transductive filing evaluation. |
| FUSE-005 | `OPEN` | **Owner: rigor-lane.** Determine whether per-judge, per-distance, per-channel residual bias states generalize beyond the global affine layer. | Resolve with train-only fitting, node-disjoint held scoring, effective-sample-size/rank diagnostics, and one global retention decision. Per-state intervals may localize effects but may not replace the held global decision. See [bias-state design](../../prototypes/mu_cosine/DESIGN_bias_state_augmentation.md). |
| FUSE-006 | `OPEN` | **Owner: rigor-lane.** The dual-space residual state `δ̃` and state-level constraint covariance `R_c` remain parked. Row residual scatter `R̂_row` is an upper bound that mixes item noise with state discrepancy and must not be substituted as a tight `R_c`. | Resolve using replicated or cross-fitted bin/campaign effects that separate row scatter from state discrepancy, followed by held global benefit for the joint direct/logit state. See [bias-state design §3](../../prototypes/mu_cosine/DESIGN_bias_state_augmentation.md). **Envelope:** no state-level `R_c` estimate or fitted dual `δ̃` exists; available bias-state evidence is two 1,000-row enwiki campaign datasets, 40 node-disjoint splits, about 350 train rows and 36 fitted offset states per split, and only row-level `R̂_row`. **Revisit trigger:** the first replicated or cross-fitted multi-campaign bias run, operationally the at-least-10×-labels event that can separate observation scatter from state discrepancy. |

## 5. Corpus-structure axis

| ID | Tag | Architectural element | Evidence, owner, or resolution |
|---|---|---|---|
| CORP-001 | `DECIDED` | Corpus topology is an architectural input: principal-parent trees and genuine multi-parent DAGs require different path operators, leakage diagnostics, and candidate policies. | [Filing program arc §§2–3](../../prototypes/mu_cosine/ARC_filing_program.md) and [path-operator design](../../prototypes/mu_cosine/DESIGN_path_operator.md) separate the Pearltrees and Wikipedia regimes. |
| CORP-002 | `DECIDED` | Pearltrees uses record-majority principal parents and single-path `LINEAGE`; arbitrary sorted/first DAG parents are not semantically privileged. | [Candidate-lineage report](../../prototypes/mu_cosine/REPORT_pearltrees_candidate_lineage.md) found the arbitrary parent rule disagreed with recorded paths on most multi-parent folders; [path-operator design](../../prototypes/mu_cosine/DESIGN_path_operator.md) records the single-path decision. |
| CORP-003 | `DEFAULT` | Wikipedia uses multipath `PATH` because it lacks a privileged principal parent. This is the structurally meaningful default, not yet a demonstrated performance win. | [Path-operator design](../../prototypes/mu_cosine/DESIGN_path_operator.md) leaves a proper Wikipedia branch-recovery evaluation open. |
| CORP-004 | `DEFAULT` | Treat Pearltrees parent-walk features as probably redundant at inference after lineage teaching, with comparatively little hub leakage; do not generalize that null to teaching or to Wikipedia. | [PR #3865](https://github.com/s243a/UnifyWeaver/pull/3865) supplies the exploratory inference-feature null; `TEACH-001` states the narrower interpretation. |
| CORP-005 | `OPEN` | **Owner: rigor-lane.** Measure each corpus using structural parent multiplicity and path counts plus cross-validated conditional `R²` of `hit_prob` after distance and ancestor features. This supersedes a raw-correlation “tree-ness” score. | Resolve with leakage-free per-corpus estimates and uncertainty, including Pearltrees and a genuinely multi-parent corpus. The diagnostic must identify incremental multiplicity signal rather than merely reproduce distance. |
| CORP-006 | `OPEN` | **Owner: rigor-lane; implementation support: engineering-lane.** Determine whether semantic gating and entropy/generality conductance reduce multipath hub leakage without erasing useful multiplicity signal. | Resolve with the `CORP-005` diagnostic plus hop-bias, relational accuracy, Green-response/rank stability, and conditioning comparisons across the two corpus regimes. |

## 6. Evaluation and deployment discipline

| ID | Tag | Architectural element | Evidence, owner, or resolution |
|---|---|---|---|
| EVAL-001 | `DECIDED` | Geometry and candidate eligibility are outcome-blind. Topology, revision-pinned frozen embeddings, resource limits, and numerical diagnostics are permitted; placement labels and judge outcomes are not. | The exact authorization is quoted above from [Local grounded diffusion](LOCAL_GROUNDED_DIFFUSION.md). |
| EVAL-002 | `DECIDED` | Exposed seeds and splits are exploratory forever. [PR #3875](https://github.com/s243a/UnifyWeaver/pull/3875) governs reveal order, inner-only selection, and the untouched node-disjoint outer result for the ranker program. | [PR #3875](https://github.com/s243a/UnifyWeaver/pull/3875), Filing Ranker Evaluation Protocol, separates design evidence from the single confirmatory reveal. |
| EVAL-003 | `DECIDED` | Report fidelity to the operating judge separately from independent truth. Current filing labels evaluate agreement with the deployed `gpt-5.5-low` decision frame; they do not establish universal human correctness. | [Filing v1 report §2](../../prototypes/mu_cosine/REPORT_pearltrees_filing_v1.md) states the operating-judge frame; [uncertainty playbook](../../prototypes/mu_cosine/DESIGN_uncertainty_estimation_playbook.md) preserves the distinction. |
| EVAL-004 | `DEFAULT` | The deployment surface is selective escalation: rank every candidate cheaply, accept high-margin e5 decisions, and spend an LLM/judge call where ambiguity justifies it. | [Filing program arc](../../prototypes/mu_cosine/ARC_filing_program.md) defines the goal; `RANK-006` records the current descriptive gate. |
| EVAL-005 | `DECIDED` | The architecture stops at ranked candidates, calibrated uncertainty, and escalation advice. The bookmark-filing skill owns user interaction, candidate presentation, optional LLM consultation, and any approved filing action. | [Bookmark-filing skill](../../skills/skill_bookmark_filing.md) defines the deployment workflow and its confirmation boundary. |

## Consolidated decision record

This table is the short operational record. Mixed-maturity choices are split so
that a settled Pearltrees rule does not falsely promote a Wikipedia hypothesis.

| ID | Tag | Recorded choice | Evidence |
|---|---|---|---|
| DR-001 | `DECIDED` | Choose Pearltrees principal parents by record-majority over observed `path_ids`; use an assembled-edge fallback only when no path record covers the folder. | [Candidate-lineage report](../../prototypes/mu_cosine/REPORT_pearltrees_candidate_lineage.md), construction section. |
| DR-002 | `DECIDED` | Use single-path `LINEAGE` for principal-parent Pearltrees. | [Path-operator design](../../prototypes/mu_cosine/DESIGN_path_operator.md), standing Pearltrees decision. |
| DR-003 | `DEFAULT` | Use multipath `PATH` for Wikipedia while branch-recovery value remains unproven. | [Path-operator design](../../prototypes/mu_cosine/DESIGN_path_operator.md), Wikipedia section. |
| DR-004 | `DECIDED` | Fit the Luna global affine correction on training rows before covariance fitting; use debiased Luna as the primary paid S measurement and retain graph S as free supervision. This is a role decision, not a universal matched-cost victory. | [Cheap-judge baseline](../../prototypes/mu_cosine/REPORT_cheap_judge_baseline.md) and corrected [post-#3648 validation](../../prototypes/mu_cosine/REPORT_cheap_judge_post3648_validation.md). **Envelope:** role evidence spans 1,700 matched `gpt-5.5-low`/Luna enwiki rows over two corpora and 40 partitions, plus a 300-row Pearltrees random judge overlap inside the 799-pair live campaign; all comparisons measure operating-judge fidelity, not gold truth. **Revisit trigger:** completed Pearltrees export with a fresh corpus-specific overlap, or at least 10× judge labels. |
| DR-005 | `DECIDED` | Every newly materialized e5 cache pins model ID, exact revision, text construction, normalization, node order, and content hashes. | [PR #3875 §2](https://github.com/s243a/UnifyWeaver/pull/3875) freezes artifact identity before evaluation. |
| DR-006 | `DEFAULT` | Keep frozen `intfloat/e5-small-v2` as the filing backbone. Historical caches without a recorded remote revision are legacy evidence, not revision-pinned artifacts. | [Filing program arc §2](../../prototypes/mu_cosine/ARC_filing_program.md); exact revision backfill remains `OPENQ-011`. |
| DR-007 | `DEFAULT` | Prefer revision-pinned Nomic `nomic-embed-text-v1.5` with the exact `clustering: ` prefix for an external semantic geometry modifier; retain MiniLM as sensitivity and e5 as the shared-input redundancy control. “External/less redundant” does not mean statistically independent or ground truth. | [Graph-geometry decisions](../../prototypes/mu_cosine/DECISIONS_graph_geometry.md) and its confirmatory evidence pointer. **Envelope:** outcome-blind enwiki inventory only: 1,000 exploratory and 770 fresh campaign rows (764/777 within-descendant comparisons); pinned Nomic and MiniLM caches covered 2,681 unique titles; Nomic–e5 distance Spearman was 0.776/0.838 versus MiniLM–e5 0.810/0.868, and no held outcome benefit was tested. **Revisit trigger:** the first enwiki-scale held residual or filing comparison, or evaluation of a materially larger revision-pinned embedder. |
| DR-008 | `DECIDED` | Store durable campaign artifacts under `~/mu_data`; expose `/tmp/mu_data` only as a compatibility symlink, and hash artifacts/manifests. | [Bias-state report §0](../../prototypes/mu_cosine/REPORT_bias_states.md) records the reboot failure and durable-store correction. |
| DR-009 | `DECIDED` | Verify squash-merged work by tree/path contents or branch-to-main tree comparison; the absence of a “Merge pull request” commit is not evidence that files are absent. | [Filing program arc §6](../../prototypes/mu_cosine/ARC_filing_program.md), squash-merge verification note. |

## Open-questions register

The register is ordered by the two leverage points that currently dominate the
filing outcome, followed by dependencies and longer-horizon research. The
resolution column is a compact evidence contract, not a protocol.

| Priority | ID | Tag | Question | Owner and resolving evidence |
|---:|---|---|---|---|
| 1 | OPENQ-001 | `OPEN` | Can placement-blind structural candidate generation raise the 0.680 recall@50 ceiling? | **Owner: engineering-lane; boundary: rigor-lane.** Freeze eligibility and retrieval before held outcomes; improve recall at matched `K`, then prospective filing MRR under #3875. |
| 2 | OPENQ-002 | `OPEN` | Which bounded domain gives stable diffusion features without the two-hop boundary pathology? | **Owner: rigor-lane.** Compare nested candidate-skeleton domains with a larger exact-Dirichlet reference on responses, ranks, boundary diagnostics, conditioning, cost, and censoring. |
| 3 | OPENQ-003 | `OPEN` | Should infeasible e-fold calibration fail closed or expose a labeled-degraded raw `h_s`? | **Owner: user; evidence: both lanes.** Freeze both modes and compare reference error, downstream risk/coverage, and cost; never call degraded output calibrated. |
| 4 | OPENQ-004 | `OPEN` | Can candidate skeletons, exact single-port reductions, and sparse mass-conserving closure approximate larger domains at useful cost? | **Owner: rigor-lane.** Require the algebraic constraints in `GEOM-005/006`, SPD/M-matrix checks, a mass ledger, reference error, rank stability, and resource savings. |
| 5 | OPENQ-005 | `OPEN` | Does frozen entropy/generality-weighted conductance reduce hub leakage without discarding relational signal? | **Owner: rigor-lane.** Freeze a symmetric positive-floor definition, recalibrate `α`, and compare both corpus regimes on leakage, accuracy, responses, ranks, and conditioning. |
| 6 | OPENQ-006 | `OPEN` | Does the thin feature blend deserve promotion? | **Owner: engineering-lane.** Meet #3875's untouched outer threshold and paired node-block interval; exposed descriptive seeds cannot promote it. |
| 7 | OPENQ-007 | `OPEN` | What top-two-margin escalation policy gives useful quality/cost tradeoffs? | **Owner: engineering-lane.** Select thresholds inside training only and report untouched risk/coverage or AURC, paired uncertainty, random-routing control, and matched spend for any deployment claim. |
| 8 | OPENQ-008 | `OPEN` | Does graph judge v2 improve teaching rather than merely flatten residual covariance? | **Owner: rigor-lane; implementation: engineering-lane.** Require the directed semantics and multi-metric acceptance criteria in `TEACH-003`. |
| 9 | OPENQ-009 | `OPEN` | Which measured corpus properties predict when multipath teaching and gating help? | **Owner: rigor-lane.** Use structural multiplicity/path counts plus leakage-free conditional `R²`, not raw correlation. |
| 10 | OPENQ-010 | `OPEN` | Do per-distance and dual-space bias states generalize, and what is the state-level `R_c`? | **Owner: rigor-lane.** Separate `R̂_row` from `R_c` with replicated/cross-fitted effects and require held global benefit. |
| 11 | OPENQ-011 | `OPEN` | Can the operative historical e5 artifacts be given exact revision provenance? | **Owner: engineering-lane.** Regenerate or attest caches with exact Hugging Face commit, text/prefix contract, normalization, node order, and hashes. |

## Evidence index

- Filing program and ranking: [program arc](../../prototypes/mu_cosine/ARC_filing_program.md),
  [Filing v1](../../prototypes/mu_cosine/REPORT_pearltrees_filing_v1.md),
  [candidate lineage](../../prototypes/mu_cosine/REPORT_pearltrees_candidate_lineage.md),
  [PR #3865](https://github.com/s243a/UnifyWeaver/pull/3865), and
  [PR #3875](https://github.com/s243a/UnifyWeaver/pull/3875).
- Geometry: [leaky diffusion](LEAKY_GRAPH_DIFFUSION.md),
  [local grounded diffusion](LOCAL_GROUNDED_DIFFUSION.md), and
  [geometry decisions](../../prototypes/mu_cosine/DECISIONS_graph_geometry.md).
- Teaching and covariance: [transitive relations](../../prototypes/mu_cosine/DESIGN_transitive_relations.md),
  [Sigma-hop confirmation](../../prototypes/mu_cosine/REPORT_sigma_hop_confirmatory.md),
  [cheap-judge baseline](../../prototypes/mu_cosine/REPORT_cheap_judge_baseline.md), and
  [corrected post-#3648 validation](../../prototypes/mu_cosine/REPORT_cheap_judge_post3648_validation.md).
- Fusion and calibration: [uncertainty playbook](../../prototypes/mu_cosine/DESIGN_uncertainty_estimation_playbook.md),
  [JointPosterior report](../../prototypes/mu_cosine/REPORT_cheap_judge_joint_posterior.md),
  [QR design](../../prototypes/mu_cosine/DESIGN_joint_square_root_qr_conditioner.md),
  [bias-state design](../../prototypes/mu_cosine/DESIGN_bias_state_augmentation.md),
  [bias-state report](../../prototypes/mu_cosine/REPORT_bias_states.md), and
  [name migration](../../prototypes/mu_cosine/REPORT_judge_name_migration.md).

## Change log

| Date | Change |
|---|---|
| 2026-07-18 | Initial architecture map assembled from PRs #3648–#3875 and the candidate-skeleton/graph-judge-v2 design discussion. |
| 2026-07-18 | Split mathematical identities from decisions, including a separate union-of-anchors convention; added empirical evidence envelopes and scale-triggered revisit events; demoted regime-dependent cheap-judge economics and the untested Nomic preference to working defaults. |
| 2026-07-19 | Refined `GEOM-006` from semantic edge creation to graph-derived exterior closure: semantics is only an optional search filter, while topology licenses edges and resistance/Schur response sets their strength; explicitly allowed a zero-bridge result after common-parent retention; recorded `GEOM-010` so memoized loop collisions cannot be mistaken for cut points. |
