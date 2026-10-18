# Filing ranker blend: e5 + grounded diffusion + graph features on recorded placements

The application deliverable (ARC §7 / post-#3845 refocus): a thin, honest feature-blend ranker for
Pearltrees filing. No LLM scoring anywhere — ground truth is each bookmark's REAL recorded placement
(treeId), and every feature is OUTCOME-BLIND (placement labels touch only the blend fit and the
metrics). The standing bar: e5-cos MRR 0.294 on the 400-query harness; μ heads ≤0.11 as rankers.

## 1. Construction (all parameters recorded; filing_ranker.py)

- **Candidates**: e5-cos top-50 folders per bookmark, over the 335 candidate folders (≥3 bookmarks).
- **Graph universe**: the candidates' 2-hop titled undirected neighborhood of the assembled DAG
  (6,014 nodes). Undirected symmetrization is documented as appropriate for SCREENING
  (neighborhood closeness); the DIRECTIONAL features use the directed record-majority
  principal-parent graph (the #3845 audit rule — never a sorted/first-DAG-parent walk).
- **Grounded-diffusion screening h_s** (docs/design/LEAKY_GRAPH_DIFFUSION.md, dense float64
  reference): semantic conductance from e5 with ℓ = median edge e5 distance (0.5175), floor
  ε = 0.05. **DIRICHLET boundary (audit finding 1, fixed):** the 5,709 edges cut by the 2-hop
  truncation become shunts to the bath at their 1,296 boundary nodes (same RBF-with-floor
  conductance; total shunt mass 3,827) — the first run's insulating truncation was a wrong
  operator and its numbers are retracted. Uniform α is bisected ON TOP of the boundary shunts by
  the e-fold recipe; **the calibration SATURATED at its lower bound (α = 1e-4, final shell ratio
  0.047 vs the 1/e = 0.368 target)** — with 1,296/6,014 nodes on the boundary, bath leakage alone
  over-damps beyond the recipe's reach. This is recorded as a calibration failure of the shell
  recipe under high surface-to-volume truncation, not hidden: the operator used is the
  boundary-dominated one. Float64 contract passed (condition 5.6e3, reciprocal 1.8e-4). Source: unit current split over the bookmark's
  top-20 e5-nearest graph folders ∝ cosine⁺; h_s(f) = equilibrium response at candidate f via
  triangular solves against the precision root (no inverse).
- **Directional/topology features**: hit_prob(anchor→f), hit_prob(f→anchor), and
  sym_graph_features(anchor, f) (inv d_sym, shared parent, shared grandparent, is-ancestor), where
  anchor = the bookmark's e5-nearest graph folder (outcome-blind).
- **μ features** (--mu-feats): the UNTRAINED base model's agnostic ELEM/HIER/SYM μ — rankers that
  lose alone (≤0.11) may still earn their keep as features.
- **Blend**: weighted ridge (closed form, deterministic; positives ×K), standardized features.
- **Folds**: 5 node-disjoint seeds (node_disjoint_eval conventions over (bookmark, true-folder)
  pairs): held pairs' bookmarks AND folders are unseen as training identities, and held identities
  are excluded from training rows even as negatives (the #3845 audit finding-2 rule).
- **Grading**: title-equivalence (best-alias rank); a true folder outside the top-50 counts as a
  miss (rank ∞), and candidate-recall@50 is reported as the explicit ceiling.

## 2. Results

Held metrics across 5 node-disjoint folds (1,200 queries, manifest sha `fcf5e1d6…`;
candidate-recall@50 = 0.680 caps every ranker identically):

| ranker | MRR | recall@1 | recall@5 |
|---|---|---|---|
| e5-cos only | 0.294 ± 0.034 | 0.209 ± 0.028 | 0.384 ± 0.045 |
| blend | 0.308 ± 0.042 | 0.216 ± 0.038 | 0.400 ± 0.052 |

(± values are spread over OVERLAPPING repeated splits — descriptive, not intervals.)

**CONFIRMATORY (frozen seed-0 split, 192 held; paired two-endpoint node-block bootstrap on
per-query Δ(1/rank), 95% CI): blend − e5 = −0.003 [−0.061, +0.055] — a NULL.** The five repeated
splits are OVERLAPPING (held 188–202; only 699/1,200 queries ever held) and are reported only as
descriptive stability: per-split ΔMRR mean +0.0145, [−0.003, +0.041, +0.011, +0.017, +0.006],
4/5 positive. The earlier headline ("first ranker to beat e5, 5/5 folds") was computed on the
INSULATING-boundary operator and mislabeled overlapping splits as folds (statistical audit findings
1 and 3) — it is RETRACTED. The honest statement: the blend is descriptively ahead of e5 and
confirmatorily indistinguishable from it at this held size.

Drop-one-family ablations (ΔMRR vs the full blend; negative = the family was helping):

EXPLORATORY (five families, no multiplicity handling — audit finding 4; the pre-declared
confirmatory secondary is the frozen-split full−drop-diffusion bootstrap, which is also null:
−0.011 [−0.059, +0.035]):

| family dropped | MRR | Δ |
|---|---|---|
| e5-cos | 0.294 | −0.014 |
| grounded diffusion (h_s) | 0.294 | −0.014 |
| sym4 | 0.305 | −0.004 |
| walk (hit_prob fwd/rev) | 0.310 | +0.002 |
| μ features | 0.311 | +0.002 |

Descriptively, diffusion ties e5-cos as the most load-bearing family and the walk/μ families
contribute nothing (consistent with #3845: μ = calibration/fusion, not ranking) — but per the
confirmatory bootstrap this attribution is suggestive, not established.

Mean standardized blend weights: e5_cos +0.181, h_s +0.073, hit_rev +0.074, inv_d_sym −0.124,
mu_hier +0.046, others ≤ |0.03|. (inv_d_sym's negative sign is a conditional weight next to h_s —
the two share neighborhood-closeness variance; interpret families via the ablation, not raw signs.)

Escalation (fold-0 held; route to judge below top-2 blend margin t):

| t | routed | kept_n | kept R@1 |
|---|---|---|---|
| 0.005 | 0.031 | 186 | 0.226 |
| 0.020 | 0.156 | 162 | 0.222 |
| 0.050 | 0.281 | 138 | 0.239 |
| 0.100 | 0.458 | 104 | 0.308 |

DESCRIPTIVE only (audit finding 6): post-hoc thresholds, fold-0, no AURC/bootstrap or
matched-coverage comparator, and kept-R@1 is NOT monotone at the first thresholds (0.198 → 0.196
→ 0.201). The broad shape (kept R@1 0.198 unrouted → 0.284 at 43% routed) suggests usable margins;
a calibrated-threshold, bootstrap-intervaled routing evaluation is future work.

**Verdict (post-audit, corrected operator).** The deliverable stands as the DEPLOYED ranker —
e5 + graph features with a thin blend, never worse than e5 descriptively — but the claimed WIN is
retracted: on the correct Dirichlet operator with honest statistics, the confirmatory frozen-split
bootstrap is null (blend − e5 = −0.003 [−0.061, +0.055]) and the diffusion attribution is
suggestive only. Two structural findings matter more than the point estimate: (i) the 2-hop
truncation is boundary-dominated (22% of nodes are boundary; shunt mass ≫ interior leakage), so
the e-fold shell recipe cannot reach its target — a larger universe or a boundary-aware
calibration is required before h_s gets a fair test; (ii) candidate-recall@50 = 0.680 caps every
ranker. Both point the next round at candidate generation and a full-graph (sparse) diffusion —
exactly the #3867 million-node-universe direction — rather than blend tuning. No tuning beyond
the recorded protocol was performed.

## 3. Caveats

- The analysis is TRANSDUCTIVE (audit finding 2): the 335-folder candidate catalog derives from
  the same recorded placements (≥3-bookmark eligibility; 40 evaluated queries sit in folders at
  exactly the threshold). A frozen pre-evaluation catalog is the clean upgrade.
- Candidate-recall@50 caps every ranker identically (0.680); raising K or a candidate
  generator beyond pure e5 is future work and orthogonal to the blend comparison.
- The feature cache key now covers source-data/e5/checkpoint hashes and the feature schema
  (audit finding 5).
- The e5 embedding drives BOTH the candidate list and features (e5-cos, the diffusion conductance,
  the anchor choice) — the design doc's preference for a revision-pinned Nomic embedding as the
  independent semantic modifier (MiniLM sensitivity) stands as the outcome-blind upgrade.
- Partial-recovery DAG (Grok re-export pending); principal parents are record-majority where
  records exist, first-DAG-edge fallback elsewhere (coverage 1,919/6,014 universe nodes).
- Numbers are not directly comparable to the 400-query REPORT_pearltrees_candidate_lineage table
  (different query count/protocol: 1,200 queries, miss-as-∞ grading, fold-held subsets).

## Repro

```
python3 filing_ranker.py --mu-feats      # full run (one torch job; features cached)
python3 filing_ranker.py                 # torch-free feature set
```
