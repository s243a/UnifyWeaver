# Product-Kalman Cross-Corpus Campaign Protocol

Status: pre-scoring exploratory protocol. This is not a preregistered confirmatory claim.

## Question

Do the Sigma(hop) and correlated Product-Kalman gains survive across enwiki, Pearltrees, and SimpleMind, and can
differences be separated into graph-structure effects versus title-quality effects?

There is no predeclared corpus winner. Enwiki has cleaner canonical titles but a multi-parent DAG. Pearltrees and
SimpleMind have a user-selected principal parent and are therefore more tree-like, but their titles may contain
typos, duplicate variants, or organizational labels. Either advantage can dominate.

## Corpus Views

| corpus | primary graph view | title source | primary risk |
| --- | --- | --- | --- |
| enwiki | category DAG below Main_topic_classifications | MediaWiki page titles | multi-parent and generic-apex ambiguity |
| Pearltrees | principal Collection containment (subtopic) | harvested collection titles | typos, incomplete harvests, and secondary annotations |
| SimpleMind | within-map principal parent, content-rooted only | raw node text after duplicate-title cleanup | typos, conflicted copies, and organizational super-layers |

Primary results use these views separately. Pearltrees section relations, shortcuts, and bridges are secondary
edges, not replacements for principal containment. SimpleMind cross-map and organizational ancestors are secondary
views. They may be evaluated as within-corpus sensitivity analyses, but must not be silently mixed into the primary
tree view.

## Verified Enwiki Starting Point

The current semantic-ready local artifact is:

    data/benchmark/enwiki_cats_correct/lmdb_scoped
    root title: Main_topic_classifications
    root page ID: 7345184
    scope depth cap: 10
    direct root children: 35
    retained category edges: 6,706,581
    real title rows: 2,245,288
    title tables: title_i2s, title_s2i
    title layer kind: mediawiki_page_titles

The depth-10 scoped store is sufficient for the initial hop-1..5 campaign.

## Graph-Title Join Contract

The numeric LMDB identity and adjacency tables are the intended fast path for graph algorithms. Distance and
traversal code operates only on uint32 IDs:

    category_parent: child ID -> duplicate parent IDs
    category_child: parent ID -> duplicate child IDs

Titles are a separate semantic overlay:

    title_i2s: uint32 ID -> UTF-8 category title
    title_s2i: UTF-8 category title -> uint32 ID

Resolve the scope root through title_s2i once, perform BFS, shortest-hop calculations, filtering, and sampling on
numeric IDs, then join only the retained endpoint IDs through title_i2s when producing judge/model rows. No title
lookup or string processing belongs in the inner graph traversal.

The full lmdb_resident graph can remain numeric. If it is used for semantic scoring later, add or attach a separate
real-title overlay without rewriting its numeric graph tables. The already-titled lmdb_scoped store is the primary
campaign source.

## Frozen Sampling Rules

1. Sample at least 250 unique descendant/ancestor pairs per corpus, balanced to 50 pairs at each shortest upward
   distance from hop 1 through hop 5.
2. Use the chosen primary graph view to compute hop. Do not mix full-DAG distance with principal-parent distance in
   one row set.
3. Exclude self-pairs, duplicate unordered pairs, missing-title nodes, administrative categories, and generic apexes
   as pair endpoints.
4. For enwiki, allocate each hop in deterministic round-robin order across eligible direct children of
   Main_topic_classifications; do not let the already-studied Behavior branch dominate the campaign.
5. For Pearltrees, stratify over harvested principal trees and record account/tree provenance. For SimpleMind,
   stratify over maps and retain only content-rooted chains after duplicate-title and organizational-layer cleanup.
6. Preserve raw IDs, raw titles, corpus, graph view, source branch/tree/map, hop, and every endpoint alias in the
   pair manifest.
7. Use one fixed random seed and deterministic casefold-title ordering. A rerun over the same source snapshot must
   produce identical pair IDs.

## Split And Leakage Rules

Calibration and final evaluation must be node-disjoint. Identity closure is stronger than literal ID equality:
explicit enwiki bridges, Pearltrees slugs, SimpleMind links, and frozen canonical-title aliases identify one concept
for split purposes. No identity-equivalent concept may cross calibration/evaluation boundaries.

Prefer branch/tree/map-level split units, omitting boundary rows whose endpoints cross units. Record omitted rows and
the retained hop distribution. Do not fit covariance blocks, title corrections, gates, or source weights on final
evaluation rows.

Each corpus gets its own calibration and evaluation result. A pooled fit may be reported only as an additional
domain-transfer experiment with corpus explicit in the model; it cannot replace per-corpus results.

## Title-Quality Audit

Title quality and tree-likeness are different variables. Keep both visible:

- Preserve raw_title unchanged.
- Add deterministic normalized_title only for matching: Unicode normalization, underscore-to-space conversion,
  whitespace collapse, and casefolding. Normalization is not a semantic correction.
- Add canonical_title only from an explicit bridge or a correction manifest frozen before scoring.
- Record flags for missing or identity-numeric titles, replacement characters, duplicate normalized titles,
  conflicted copies, and suspected spelling errors.
- Never use judge labels, model residuals, or Product-Kalman gains to decide which titles to correct.

Primary scores describe the corpus as it exists using raw titles. A matched audited-title sensitivity run reuses the
same pair IDs and graph distances while substituting only frozen canonical titles. The raw-to-audited delta estimates
title-channel cost. It does not rewrite graph truth.

## Topology Audit

Record per corpus and graph view:

- node and edge counts;
- mean, median, and p95 parent count;
- fraction of nodes with multiple parents;
- branching distribution and reachable depth;
- cycles or repeated-node paths encountered;
- principal versus secondary edge counts;
- branch/tree/map coverage of sampled rows.

These measurements can explain associations but do not identify causes by themselves. A claim that tree-likeness
offsets title noise requires a within-corpus graph-view or title-view ablation, not only a difference between corpora.

## Fixed Models And Measurements

Use the same model checkpoint, graph-channel construction, judge model, and prompt template across corpora. Compare:

1. prior model readout;
2. independent Gaussian PoE or zero-cross-covariance Kalman control;
3. correlated Product-Kalman;
4. hop-conditioned correlated Product-Kalman;
5. the registered calibrated JointPosterior baseline where the required source vector exists.

Report held-out NLL with paired row-resampling intervals, MSE, Mahalanobis-per-dimension and tail quantiles, marginal
PIT/central coverage, ECE with stated bins, and margin-gated AURC with bootstrap intervals. Also report source
correlations and separability before interpreting a fusion gain.

## Interpretation Rules

- Do not call raw-title errors semantic drift without an audited-title sensitivity result.
- Do not call a Pearltrees or SimpleMind advantage a tree-structure effect without a within-corpus graph-view
  comparison.
- Do not promote Product-Kalman over JointPosterior unless held-out NLL, calibration, and selective risk improve on
  a node-disjoint evaluation split.
- Treat corpus interactions as results, not nuisance terms to average away.
- Keep single-judge and cross-corpus ontology differences explicit limitations.

## Durable Artifacts

Each corpus run should emit a source manifest, topology audit, title audit, sampled-pair table, split manifest, score
JSON, row-level evaluation NPZ, and Markdown report. Record source hashes or LMDB fingerprints, title-table names and
counts, root IDs, graph-view rules, correction-manifest hash, model/judge identifiers, and code commit.

## Execution Order

1. Build and test the enwiki branch-stratified sampler against the titled scoped LMDB.
2. Materialize its pair, topology, and title-audit manifests before scoring.
3. Add Pearltrees principal-containment and SimpleMind cleaned within-map adapters with the same output schema.
4. Freeze any title-correction manifests.
5. Score all raw-title views with the same judge and model, then run matched audited-title sensitivities.
6. Evaluate each corpus separately, followed by explicit cross-corpus and domain-transfer summaries.
