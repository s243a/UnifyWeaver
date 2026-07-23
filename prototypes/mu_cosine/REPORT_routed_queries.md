# Routed-queries loop: exploratory end-to-end filing evidence and frozen follow-up

**Evidence status: strong exploratory/transductive result, not confirmatory.** On the historical
1200-query manifest (sha `fcf5e1d6…`, e5 @ K=100, title-equivalence grading), routing low-margin
queries (t = 0.02 → 695/1200 routed) to a cheap judge (Haiku, title-only 10-item menus) produced

- policy R@1 **0.258** vs no-routing baseline **0.203**
- **delta +0.055, paired bootstrap 95% CI [+0.037, +0.072] — EXCLUDES ZERO.**

The finite-benchmark arithmetic is reproducible, but the interval is conditional on a policy chosen
after inspecting this same benchmark. The threshold/menu grid, judge/context choice, and second
margin band were all selected with these 1200 placement labels available. The interval also
resampled individual queries, omitting destination-folder dependence, selection multiplicity, and
judge-draw variance. It therefore does not authorize a deployment or generalization claim.

As an engineering result, the signal is useful: the perfect-judge ceiling at (t=0.02, N=10) was
0.389 (+0.186), and the stored Haiku picks rescued 193 of the 351 menus containing the recorded
folder (55% rescue-pool conversion; 87/695 nulls).

## Historical protocol and prospective v2 contract

- `ceilings`: perfect-judge upper bounds over (t × N). Baseline 0.203; 0.389 @ t=0.02/N=10;
  0.512 @ t=0.03/N=20 — routing, not ranking, is where filing performance lives.
- Historical `emit`: outcome-blind task JSONL (menus in e5 order, truth never marked) → private
  `~/mu_data`.
- Judge spend: 4 parallel Haiku subagents (the roster's cheap judge), strict `{qid, pick|null}`
  contract, 695/695 scored.
- Historical `score`: required every routed qid but did **not** require a manifest header, reject
  extra qids, bind the menu/task bytes, or bind judge/prompt provenance. The stored pick files are
  consequently `legacy_unbound`; they remain descriptive and cannot be upgraded retroactively.
- Prospective v2 `emit` rebuilds a certified-public population and binds source inventory, privacy
  index, catalog, sampled population, ranking, exact band/menu/lineage selection, and task rows.
  It also freezes the exact declared provider/model/revision/interface/temperature and committed
  prompt before labels are returned. Raw output must echo the exact task ID before `seal-picks`
  will bind it, preventing a response from being reassigned to a same-QID task with different
  menus. Duplicate-title score ties use ascending frozen catalog column, the same rule as the
  exact-rank calculation. They therefore have zero margin and route to review rather than being
  silently merged; ID-specific ancestor lineage gives the judge context that title embeddings
  cannot supply. `score` rejects missing, extra, duplicate, cross-task, or tampered rows and uses
  exact destination ID as the primary grade (best title-equivalent destination is
  sensitivity-only) with a paired bookmark/folder connected-component bootstrap.
- `ROUTED_POLICY_three_tier_v1.json` freezes the current policy for a later cohort or one untouched
  outer node-disjoint test. No result on the development benchmark becomes confirmatory merely
  because the policy is now frozen.
- V2 currently seals one complete response artifact per tier. Large-call chunking and repeated
  judge draws need a bound parent-task/chunk/draw manifest before the future decision test; manual
  concatenation would lose per-call provenance and is not supported.

## Honest caveats

- Ground truth = recorded placement. A topically defensible alternative scores as wrong, while a
  historically recorded but subjectively suboptimal placement scores as right. The metric measures
  agreement with the recorded destination and may undercount acceptable alternatives; its
  direction relative to subjective best-choice quality is not identified.
- Single judge pass; per-chunk null rates varied (9/15/49/14 across the 4 subagent chunks) —
  judge-draw variance is not in the CI (which is over queries). A second judge/seed pass and a
  stronger judge (sonnet-tier) are the obvious next spends.
- Menus are title-only. Judge cards / folder lineage context (the §7 candidate-lineage machinery)
  are unexploited headroom, as is menu size (ceiling at N=20 is 0.453 for this t).
- Thresholds t and N were chosen from an outcome-revealing ceiling grid and were not preregistered.
  The later judge/context and band choices were also selected on this benchmark.
- Privacy audit found nonpublic candidate titles in 67/695 historical low-band menus and 50/227
  middle-band menus, plus one nonpublic-source query in the middle band. No such titles were
  committed to Git, but externally processed legacy artifacts must not be reused. V2 requires
  explicit-public visibility certificates for candidate and lineage nodes. A bookmark is eligible
  only inside a certified-public folder and only without a private-title or restricted-visibility
  signal; missing node visibility is quarantined.

## Where this leaves the program

The candidate loop is filing_assistant (e5 @ K=100 + margin gate) + judge escalation. On the
historical development benchmark it improved R@1 by +0.055 with Haiku and +0.113 under the final
three-tier policy. The next decision-bearing read must keep that policy fixed and use a later
bookmark cohort or one untouched outer node-disjoint split, public-only tasks, repeated judge
draws, and node-block inference. Remaining engineering levers are hypotheses, not authorized
post-hoc changes to that test.

## Lever 1 result: Sonnet judge + lineage-context menus (2026-07-23)

Same manifest, same routed set (t=0.02, N=10), menus augmented with each folder's principal-path
context (`--lineage`, §7 machinery, folder-side only / outcome-blind). Judge = Sonnet subagents
(4 disjoint chunks; null rates 13/11/11/14 — far steadier than Haiku's 9/15/49/14).

- **Policy R@1 0.290 vs 0.203 baseline: +0.087, 95% CI [+0.070, +0.107] — excludes zero.**
- **Paired vs Haiku on the 695 routed rows: +0.056, 95% conditional iid-bootstrap CI
  [+0.032, +0.084].** The corresponding whole-policy increment is 39/1200 = **+0.0325**.
  232/351 rescuable menus converted (66% vs Haiku's 55%); ~47% of the perfect-judge headroom.
  Pick agreement with Haiku was 0.60; one judge draw plus a simultaneous context change cannot
  separate model, lineage-context, and draw-variance effects.
- Confound note: this pass upgrades judge AND menu context together (deliberate — one redundant
  full pass to pick the production judge); the components are not separated.

**Coverage-efficiency rule (owner, standing):** full same-data re-scores are a one-time spend for
judge selection only. All future judge spends are coverage-first — calibrate judges/prompts on
~150–200-query subsamples; put the bulk of tokens on NEW queries (e.g. the t=0.03 increment) or
NEW menu tail (positions 11–20 where the N=10 menu missed), never on re-scoring covered rows.

## Lever 2 result: three-tier policy — margin-banded menus (coverage-first)

Per the standing rule, zero re-scoring: the 0.02≤margin<0.03 band (227 newly judged queries, but
not new or untouched evaluation rows; N=20 lineage menus, rescue ceiling 113/227) was judged once
by Sonnet (nulls 13/5). Frozen exploratory policy:
margin<0.02 → judge@N10; 0.02–0.03 → judge@N20; ≥0.03 → auto-file e5 top-1.

- **Policy R@1 0.316 vs 0.203 baseline: +0.113, 95% CI [+0.093, +0.134].**
- Paired vs lever-1 (band auto-filed): **+0.026, CI [+0.017, +0.036] — excludes zero.**
- In-band: judge 72/227 correct vs auto-file 41/227 — judging the band beats auto-filing it.

Historical development arc: R@1 0.203 → 0.316 (+56% relative) via routing alone; ranker
untouched. This exact numeric result belongs to the legacy, private-inclusive population. The
prospective public-only population has a new privacy/catalog/population fingerprint and the
outcome-blind `pearltrees-public-alphanumeric-title-v1` eligibility rule. It must be reported
separately rather than compared as though its rows were unchanged.
