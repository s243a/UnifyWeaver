# Confirmatory protocol: Pearltrees filing-ranker feature blend

**Status:** prospective and frozen before the next corrected rerun.  Results already seen in PR #3865 are
exploratory and cannot satisfy this protocol.  Seeds 0--4 have also been revealed and are ineligible for
the outer test.  This is a locked **transductive audit** of the frozen corpus snapshot: a clean inductive
deployment claim additionally requires a candidate catalog frozen before a later bookmark cohort exists.
If implementation or data constraints require a material change, amend this file and its preregistration
manifest before computing any outer-test metric.

## 1. Decision and estimand

The practical question is whether a thin, outcome-blind filing ranker improves over the deployed
e5-cosine baseline on held bookmark and folder identities within the frozen snapshot.  The primary
estimand is

\[
\Delta_{\mathrm{MRR}}
= \operatorname{mean}_{q\in T}
\left[RR_{\mathrm{blend}}(q)-RR_{\mathrm{e5}}(q)\right]
\]

on one frozen outer node-disjoint test set \(T\).  Both rankers use the identical frozen candidate catalog,
top-50 candidate list, queries, and grading rule.  A missing true folder has reciprocal rank zero.

The blend is promoted only when both conditions hold:

1. \(\Delta_{\mathrm{MRR}} \ge 0.010\); and
2. the lower endpoint of the paired 95% node-block bootstrap interval is greater than zero.

The absolute 0.010 floor is deliberately modest but nonzero.  The exploratory repeated-split standard
deviation of the corrected PR #3865 deltas was about 0.016; half of that is about 0.008, rounded upward.
That split-to-split spread is used only to size the practical floor, never as a standard error.

Report four possible outcomes without collapsing them: promoted; statistically positive but below the
practical floor; practically sized but inconclusive; or null/negative.  An honest null is a complete result.

## 2. Frozen population, identities, and grading

Before any outer scoring, commit a machine-readable manifest containing all choices and hashes below.

- Snapshot the Pearltrees tree JSON directory, assembled DAG, title table, and principal-path records.
  Hash a canonical sorted inventory of every input file and its bytes; an absolute path is not provenance.
- Enumerate the candidate catalog structurally from every folder present in the frozen tree snapshot and
  having a stable folder ID and title.  Do **not** filter candidates by bookmark count, recorded-placement
  frequency, judge score, or any statistic derived from the evaluation labels.  Record all exclusions and
  their structural reason.
- Give graph nodes typed stable identities, at minimum `("bookmark", pearl_id)` and
  `("folder", tree_id)`.  Titles are attributes, not identities.
- Deterministically choose at most 1,200 eligible bookmark queries by sorting stable bookmark IDs and then
  sampling with seed 7.  Selection may use structural availability, but not the bookmark's destination,
  rank, judge output, or feature value.  Commit the ordered query-ID manifest before fitting or scoring.
- Grade exact destination folder ID as the primary endpoint.  Duplicate-title/best-alias grading is a
  labeled sensitivity analysis because the deployed filing action targets an ID, not a title string.
- Retrieve the top 50 candidates by frozen e5 cosine from the full structural catalog.  Freeze the e5 model
  identifier, revision, prompt/text construction, normalization, tie rule, and embedding hashes.  Report
  candidate recall@50 as a descriptive ceiling, never as evidence that the blend ranks better.

The earlier `>=3 bookmarks` catalog is placement-derived and therefore compounds the transductive use of
the current snapshot.  It may be retained only as a separately labeled sensitivity analysis; it is not the
locked audit population.  Even the structural catalog is not a future-cohort catalog: the current tree
snapshot may reveal that a destination exists.  The final report must retain the transductive qualifier.

## 3. Leakage boundary and split

Use `node_disjoint_pair_split` with outer seed 3867, held fraction 0.40, 64 split candidates, and the typed
bookmark/folder IDs.  Seeds 0--4 must not be reused because their results have been inspected.  Persist the
exact train, cross, and held identity manifests and their full SHA-256 digests before feature fitting.  The
outer held labels must be unreadable to every step before the final single evaluation.  Cross rows are not
training rows.

All of the following are fit or selected using only outer-train data or outcome-blind graph diagnostics:

- feature inclusion and standardization;
- ridge strength, source-injection count, and any other supervised hyperparameter;
- bias/fusion calibration and any learned feature transform;
- routing thresholds; and
- missing-value or alias policy.

Placement labels—outer or train—must not enter candidate enumeration, graph construction, embeddings,
conductance, local-domain selection, \(\ell\), \(\epsilon\), \(\alpha\), diffusion sources, or cache reuse.
Graph statistics are outcome-blind only when computed from the frozen pre-evaluation topology, not from
held bookmark placements.  A revision-pinned Nomic embedding is the preferred independent semantic
geometry; e5 is permitted in this baseline round only if its dual role is stated and frozen.

For model selection, make repeated node-disjoint inner splits entirely inside the outer-train identities.
Average inner validation MRR is the selection score.  Overlapping inner splits are acceptable for selection
but are not folds and provide no inferential interval.  Resolve ties within 0.002 MRR toward fewer feature
families, then stronger ridge regularization, then the lexicographically first frozen configuration.

## 4. Frozen feature and hyperparameter family

The candidate configurations are nested so selection cannot invent a new family after seeing validation:

1. e5 cosine only (reference, never refit);
2. e5 cosine + corrected grounded-diffusion \(h_s\);
3. (2) + the four symmetric topology features;
4. (3) + forward/reverse principal-parent walk features;
5. (4) + the three untrained base-\(\mu\) features.

For fitted blends, search ridge strength in `{0.1, 1.0, 10.0}` and source-injection count in
`{8, 20, 32}`.  Standardization moments and ridge coefficients are learned on inner-training rows only.
After selection, refit the selected configuration once on all outer-train rows.

Exclude an outer-train query from within-list reranker fitting when its exact true folder is absent from
its top-50 list; an all-negative list does not define the intended ranking target.  Keep the same query in
evaluation with reciprocal rank zero.  Report the excluded training count and candidate-miss count.

This protocol does **not** authorize semantic shortcut edges, approximate Kron/Schur closure, learned
generality conductance, or candidate-skeleton reduction.  Those are new geometries requiring a separate
label-free calibration and held structural validation.  They must not be introduced into this rerun.

## 5. Grounded-diffusion contract

Use the operator and diagnostics in `docs/design/LOCAL_GROUNDED_DIFFUSION.md`, not a hand-written
two-hop variant.  For every scored batch:

- form one shared union-of-anchors domain and one factorization;
- enumerate every incident graph edge before truncation;
- convert every cut edge into its exact Dirichlet shunt \(\beta\);
- use symmetric nonnegative conductances and verify the grounded precision is SPD with M-matrix signs;
- calibrate uniform \(\alpha\) outcome-blind on the preregistered strict-interior shell using the tightest
  anchor, and record the per-anchor realized tail-envelope radius and censor flag; and
- record domain size, boundary size, cut count/mass, \(\alpha\), \(\ell\), \(\epsilon\), condition diagnostics,
  envelope \(p\), cut-current fraction, solve residuals, runtime, and peak memory.

The dense correctness API fails closed when a cut endpoint or required embedding is missing.  A future
scale adapter may substitute topological \(c_0\) for a missing exterior embedding only as licensed by the
design, with substitution count and mass recorded.  Silently dropping untitled exterior endpoints is not
allowed.  The currently reviewed #3865 head does so and therefore is not yet an exact Dirichlet run.

Choose local-domain size without labels: evaluate nested resource-feasible domains at \(K\), \(2K\), and
\(4K\), and choose the smallest size whose raw responses, \(h_s\), and candidate rankings meet the frozen
stability tolerances in the local-diffusion design.  Do not select \(K\) by held MRR.  If no feasible domain
passes, fail closed and report the resource limit.  If boundary shunts already attenuate the calibration
shell beyond \(e^{-1}\) at zero/numerical-minimum uniform leakage, record the target as infeasible; do not
silently return a bisection endpoint and call it calibrated.

## 6. Primary inference

Evaluate the refitted selected blend and frozen e5 reference once on the outer held queries.  Persist one
row per query with typed IDs, candidate IDs, scores, ranks, reciprocal ranks, and paired difference.

Construct a deterministic 95% percentile interval with 9,999 paired node-block bootstrap resamples using
`paired_node_bootstrap_ci` and seed 3,867,001.  Each resample applies identical weights to both rankers and
respects both bookmark and destination-folder dependence.  Report the point estimate, interval, bootstrap
mean and attempt count, held-query count, unique typed bookmark/folder counts, seed, resample count, and a
Monte Carlo rerun check.  Do not pool overlapping split results or call their spread an SE.

## 7. Secondary analyses and multiplicity

Use a gatekeeping hierarchy:

1. Test the primary blend-versus-e5 endpoint above.
2. Only if the blend is promoted, test the prespecified attribution contrast: selected full blend versus
   the same fitted family without grounded diffusion.  Require a paired node-block 95% interval excluding
   zero; use an absolute 0.005 MRR attribution floor.

All remaining drop-family ablations, alternative candidate catalogs, title-equivalence grading, repeated
splits, and Nomic/MiniLM/e5 geometry comparisons are exploratory and must be labeled as such.  Print all
tested configurations, including losses.  This hierarchy controls the two decision-bearing tests without
pretending the descriptive ablation table is confirmatory.

Routing is also descriptive unless its threshold is selected wholly inside outer-train data and evaluated
once on the outer test.  Report risk/coverage or AURC against e5 and a matched-coverage random router with a
paired node-block interval.  No claim of judge rescue, cost savings, or deployment benefit is permitted
without labeled routed outcomes and a matched-cost analysis.

The historical three-tier routed policy (margin <0.02: Sonnet-family judge with N=10 lineage menu;
0.02≤margin<0.03: the same judge with N=20 lineage menu; margin≥0.03: e5 top-1) was developed on
the standing 1,200-row benchmark and is therefore exploratory/transductive there.  Its frozen
prospective definition is `ROUTED_POLICY_three_tier_v1.json`.  A decision-bearing follow-up must
apply that artifact without modification to a later cohort or one untouched outer node-disjoint
test; any threshold, menu, judge, prompt, or lineage change restarts selection inside outer-train.
Repeated judge draws are required to expose judge variance. Task emission on that untouched set
must not report a rescue ceiling or any other placement-label-derived diagnostic before the
frozen judge outputs have been sealed.

Any task sent outside the local machine must be rebuilt from the certified-public Pearltrees
population.  Candidate folders and emitted lineage ancestors require explicit public node
visibility under `pearltrees-public-only-v1`; missing node visibility is quarantined and privacy
propagates down typed containment.  A bookmark is eligible only as content of a certified-public
folder and only when it has no private-title or restricted-visibility signal. Candidate and query
eligibility also use the frozen, outcome-blind
`pearltrees-public-alphanumeric-title-v1` rule: a destination title must contain at least one
letter or number, and queries recorded only in an excluded destination leave the evaluation
population. The task envelope
must bind the source inventory, privacy index, catalog, population, ranking, exact selection band,
and task rows. Exact score ties are broken by ascending frozen catalog column, matching the
exact-destination rank computation. Picks must bind the exact task bytes and declared judge/prompt provenance.
Missing, extra, duplicate, legacy headerless, cross-task, or tampered rows fail closed.  Content
hashes establish integrity, not provider authentication.

The current v2 tool binds exactly one complete response artifact per judge tier. It does not yet
represent multiple provider-call chunks or repeated judge draws. Before the next decision-bearing
run, add a parent-task manifest that binds every chunk, provider run, and draw to the same task ID,
plus a frozen aggregation rule. Manually concatenating partial responses is not an acceptable
substitute.

The current ranker receipt binds the e5 model name and exact candidate/query embedding arrays, so
upstream model drift fails reproduction rather than silently changing a result. Before a future
decision-bearing task is emitted, replace the `unresolved` Hub revision with an immutable e5
revision so those bound arrays can also be regenerated long-term.

## 8. Reproducibility and fail-closed audit

The run fingerprint must cover, by content rather than machine-local paths:

- source snapshot and canonical file inventory;
- candidate, query, split, and title-alias manifests;
- DAG, principal-parent choices, and every local-domain/cut manifest;
- embedding model IDs/revisions, text normalization, and embedding arrays;
- model checkpoints and calibration/fusion artifacts;
- exact feature schema, hyperparameter grid, tie rules, seeds, and software commit;
- numeric dtype, linear-algebra/runtime identity without absolute library paths; and
- cached per-query feature arrays plus the keys used to validate them.

Reject stale or partially matching caches.  Verify manifest reproduction, deterministic rerun, unique
joins, complete judge/source coverage where relevant, no duplicate IDs, and zero missing cut edges before
revealing outer metrics.  Record peak memory and wall time so statistical success cannot conceal an
unusable implementation.

The final report must distinguish: choices frozen by this protocol; label-free numerical calibration;
inner supervised selection; the one outer confirmatory readout; and exploratory sensitivities.  Any
post-outer change starts a new protocol/run rather than overwriting this result.
