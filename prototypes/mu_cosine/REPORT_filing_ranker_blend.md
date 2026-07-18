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
  ε = 0.05; uniform leakage α calibrated OUTCOME-BLIND by the e-fold recipe on a frozen 2-hop
  shell (bisection to median shell/source response = 1/e; α = 3.8e-4, final shell ratio 0.368 ≈ 1/e);
  float64 contract passed (condition 1.68e5, reciprocal 5.94e-6). Source: unit current split over the bookmark's
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
| e5-cos only | 0.293 ± 0.033 | 0.208 ± 0.028 | 0.384 ± 0.045 |
| **blend** | **0.312 ± 0.041** | **0.225 ± 0.035** | **0.401 ± 0.056** |

**PAIRED per-fold ΔMRR (blend − e5): +0.0187 ± 0.0161 fold-SD; per-fold
[+0.003, +0.043, +0.016, +0.025, +0.007]; 5/5 folds positive.** The first ranker in the Filing
arc to beat e5 — a small, fold-consistent gain (+6% relative MRR).

Drop-one-family ablations (ΔMRR vs the full blend; negative = the family was helping):

| family dropped | MRR | Δ |
|---|---|---|
| e5-cos | 0.294 | −0.018 |
| **grounded diffusion (h_s)** | 0.296 | **−0.016** |
| sym4 | 0.307 | −0.005 |
| walk (hit_prob fwd/rev) | 0.315 | +0.003 |
| μ features | 0.315 | +0.003 |

The diffusion screening score is the second-most-valuable family — nearly as load-bearing as e5-cos
itself, i.e. genuinely ADDITIVE semantic-topological signal, not a proxy for e5. The directed-walk
and μ families contribute nothing here (dropping them slightly helps) — consistent with #3845's
verdict that the μ stack's value is calibration/fusion, not ranking.

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

Margins are informative (kept-decision accuracy rises with the threshold): a ~46%-routed policy
keeps decisions at R@1 0.308 vs 0.219 unrouted — the deployment shape from Filing v1 carries over
to the blend.

**Verdict.** The deliverable holds: e5 + graph features with a thin blend is the practical filing
ranker, and the grounded-diffusion screening score is the one new feature that pays — a
fold-consistent +0.019 MRR (5/5 folds) with the diffusion family carrying most of it. The gain is
modest in absolute terms and bounded above by candidate generation (recall@50 = 0.680), which is
now the highest-leverage lever. No tuning beyond the recorded protocol was performed.

## 3. Caveats

- Candidate-recall@50 caps every ranker identically (0.680); raising K or a candidate
  generator beyond pure e5 is future work and orthogonal to the blend comparison.
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
