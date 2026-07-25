# Prospective protocol: SM-FS retention and Pearltrees transfer

**Status:** prospective design lock, currently **blocked** until the negative-enriched SM-FS
ranking bundle, matched null ledger, behavior panel, and target cohort are frozen. No Pearltrees
score is authorized by this version. Retention and transfer are different estimands and may never
be collapsed into a single “SM-FS worked” result.

The existing `model_sm_fs_lin.pt` starts from a Pearltrees-trained checkpoint. It can contribute
only to a descriptive continual-learning/retention narrative. Its positive-only `+0.82` validation
correlation is neither ranking nor transfer evidence, and its originally cited `+0.37` warm-start
correlation is not reproducible from the current content-unbound run receipt.

## 1. Bound source artifacts

The future execution must bind these exact starting artifacts:

| role | artifact | SHA-256 |
|---|---|---|
| pre-Pearltrees base | `model_prod_namecond.pt` | `c1cfc3a3827e42a1993f4286b6a881aee7ff10eb56a76367735b9ec8fdf11f7d` |
| migrated pre-Pearltrees base | `model_prod_namecond_full.pt` | `9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef` |
| Pearltrees retention baseline | `model_pt_filing_lin.pt` | `55834a204093e1cd525b7189d372cdec6eea4b424518e938912955f5dfcf3c76` |
| positive-only pilot, descriptive only | `model_sm_fs_lin.pt` | `e541e8b812a00be457a27f3f7484ffcb53f5ba65c54c6c3bbd9ebcba90c9be19` |

The source bundle bindings are those in `PROTOCOL_sm_fs_lineage_ranking.md`. Both experiments must
consume a verified, frozen negative-enriched source ledger produced under that protocol. The
positive-only pilot ledger cannot substitute.

`model_prod_namecond_full.pt` is the **operative Track-T source**. The raw
`model_prod_namecond.pt` hash is provenance for its report-grounded, behavior-preserving migration
only; the historical migration's embedding cache is not revision-bound and is not required to
byte-reproduce. Before execution, verify the migrated checkpoint's registered preexisting behavior
against a frozen panel, then perform LINEAGE growth once per seed from those exact migrated bytes.

The migrated checkpoint has only four preexisting operator/readout rows and predates LINEAGE
training. Merely comparing a freshly expanded, unchanged LINEAGE head with a trained head would
mostly measure learning any head at all—the invalid contrast previously retracted in Filing v1.
The matched structure-scrambled source control below is therefore load-bearing.

## 2. Common execution rules

Use training seeds `3998101`, `3998102`, and `3998103`. Before any Pearltrees scoring, freeze:

- the negative source ledger and matched null-ledger hashes;
- code commit and every base, expanded, initialized, and output checkpoint hash;
- optimizer, precision, steps, batch size, learning rate, anchor weight, clipping;
- exact trainable parameter names and counts;
- source row/list normalization, deterministic order, augmentation RNG, and batching;
- CUDA, PyTorch, driver, GPU, and determinism settings; and
- prediction schema, catalog, tie rule, scoring code, and evaluation receipt.

Use one GPU job at a time. Paired arms share the same seed, initialized bytes, source examples,
batches, augmentation schedule, optimizer budget, and checkpoint-selection rule. Verify
byte-identical initial states before either optimizer is created. The already observed seed-0
pilot is excluded.

Source-only, lineage-blocked inner validation may select a configuration only if the entire finite
grid and tie rule were frozen before fitting. The 1,481 reserved and 1,755 quarantined SM-FS rows
remain unread for training, early stopping, or model selection. No Pearltrees label, rank, margin,
calibration, or outcome may select a source checkpoint or hyperparameter.

The current domain-specific node typing is frozen: SM-FS pairs use
`mindmap_node,mindmap_node`; Pearltrees uses its registered folder/collection type. A future generic
collection type is a separately preregistered symmetric arm and cannot be chosen using target
performance.

## 3. Matched structure-scrambled control

The null ledger preserves every source item and candidate text, list size, and the **list-level
multiset** of supervision payloads, folds, batches, and optimizer exposure. A payload is the joint
tuple `(target bytes, objective mass, positive/nonancestor status, relation, hardness)`. Within each
source candidate list, apply one deterministic, preregistered permutation assigning complete
payloads to different candidate identities. The row-wise type and weight are deliberately moved
with the target; only their list-level marginals are preserved.

Reject a permutation unless every payload index moves, every true ancestor receives a
nonancestor payload, and every positive payload moves to a true nonancestor. Fixed point means
payload **index**, not equality of numeric value: repeated `0.02` values make value-level
derangement impossible and uninformative. Record the fraction of rows whose numeric target remains
equal after permutation, require the target-1 parent payload to move, and bind the complete
permutation in the null-ledger manifest.

The null constructor receives only the verified SM-FS source bundle. It never receives a
Pearltrees identifier, label, candidate, embedding, rank, or score. Freeze its seed, complete
permutation, output hash, and verifier before training.

This control distinguishes information in correct SM-FS hierarchy from generic optimization,
exposure, and learning a newly added LINEAGE head.

## 4. Track R — continual-learning retention

**Question:** does continuing a Pearltrees-trained model on SM-FS preserve the capability already
present in its exact initial checkpoint?

For each training seed:

- `R0`: exact unchanged `model_pt_filing_lin.pt`;
- `R1_s`: byte-identical copy of `R0`, trained on the frozen correct SM-FS ranking ledger;
- `Rnull_s`: byte-identical copy of `R0`, trained for the same budget on the frozen
  structure-scrambled ledger.

Each trained arm's frozen reference is a deep copy of its exact initialized model. It is never
independently reloaded or regrown.

For Pearltrees query \(q\), define

\[
d_R(q)=\frac{1}{3}\sum_s
\left[RR_{R1_s}(q)-RR_{R0}(q)\right].
\]

Retention passes only when:

1. the lower endpoint of the paired 95% typed-node bootstrap is greater than `-0.005` MRR; and
2. no individual training seed loses more than `0.010` MRR.

The ranking panel uses correct candidate lineage because the baseline checkpoint was trained in
that regime according to the historical report; this setting is not recorded in a content-bound
checkpoint receipt. A decision-bearing run must reproduce the baseline behavior and candidate-
lineage scoring under a frozen receipt before fitting. Report `R1-Rnull`, conditioned `mu-max`,
head/calibration probes, and operator/type behavior panels descriptively. They cannot rescue a
failed primary retention endpoint.

The current three-field agnostic anchor is not a retention test. Before execution, freeze a
behavior panel spanning at least HIER, SYM, ELEM, and LINEAGE; agnostic and conditioned forms;
registered corpora/judges; endpoint-type combinations; positive, unrelated, and ranking rows.
Panel tolerances are fixed before training. A violation blocks a “retained” classification even if
MRR passes.

## 5. Track T — genuine zero-shot cross-corpus transfer

**Question:** starting from one common checkpoint with no Pearltrees training, does correct SM-FS
structure improve Pearltrees filing ranking?

For each seed:

1. load and verify exact `model_prod_namecond_full.pt` bytes;
2. set the seed;
3. run the pinned loading/growth path exactly once: expand LINEAGE operator/readout and any missing
   judge-name rows, while verifying that already-full corpus/type tables are unchanged;
4. seal the resulting common expanded state as `T0_s`; and
5. deep-copy those exact bytes into both trained arms.

The arms are:

- `T0_s`: common expanded base, unchanged;
- `T1_s`: the same state plus correct SM-FS ranking training;
- `Tnull_s`: the same state plus matched structure-scrambled source training.

Define total transfer:

\[
d_T(q)=\frac{1}{3}\sum_s
\left[RR_{T1_s}(q)-RR_{T0_s}(q)\right].
\]

Total transfer passes only when mean \(\Delta_{\mathrm{MRR}}\ge 0.010\) and the paired 95%
interval lower endpoint is greater than zero. Only after that gate passes, test specificity:

\[
d_S(q)=\frac{1}{3}\sum_s
\left[RR_{T1_s}(q)-RR_{Tnull_s}(q)\right].
\]

Specific SM-FS transfer requires mean \(\Delta_{\mathrm{MRR}}\ge 0.005\) and an interval lower
endpoint greater than zero. If total transfer passes but specificity fails, the allowed conclusion
is “additional training exposure helped”; correct SM-FS lineage receives no credit.

A separate deployment-safety comparison forms the per-query contrast `RR_T1 - RR_e5` and passes
only when its central-bootstrap 95% lower endpoint is strictly greater than the `-0.010` MRR
noninferiority margin. Real transfer and operational superiority are distinct outcomes.

The `0.010` total-transfer and `0.005` specificity floors inherit the filing-ranker protocol's
practical-gain and attribution conventions. The retention mean margin `-0.005` is half that
practical floor; the per-seed `-0.010` guard rejects a full-floor regression hidden by averaging.
These are prospective policy tolerances, not confidence limits estimated from the target data.

## 6. Target cohort and ranking contract

The standing 1,200-query manifest, old 400-query evaluation, split seeds 0--4, routing thresholds,
candidate-lineage variants, and related Pearltrees metrics are adaptively burned. They may be
reported only as historical sensitivities.

A decision-bearing target requires the first post-freeze certified-public harvester cohort:

1. freeze the public structural candidate catalog before reading cohort placement labels;
2. freeze all arms, hashes, inference code, candidate retrieval, and prediction schema;
3. admit only bookmark IDs absent from every historical task, training, evaluation, and pick
   manifest, and exclude any canonical URL/resource-content hash already present in those
   historical artifacts;
4. select queries by stable ID using an outcome-blind rule;
5. seal every arm's complete candidate predictions before joining destination labels; and
6. count a destination absent from the frozen catalog as a miss rather than enlarging the catalog.

If no fresh cohort exists, the locked run may use the historical population only as an
**exploratory/transductive locked audit**. It cannot confirm transfer or generalization.

All arms use one identical certified-public structural catalog and one frozen e5 top-100 pool,
pinned to E5 commit `ffb93f3bd4047442299a41ebb6fa998a38507c52`. Exact destination folder ID
is primary; a true folder outside top 100 has reciprocal rank zero; exact score ties break by
ascending frozen candidate ID; title-equivalence is sensitivity-only; recall@100 is descriptive.

The transfer primary is title-only LINEAGE scoring. The current SM-FS trainer used empty parent
tables and therefore never learned candidate-chain tokens. Adding Pearltrees chain tokens only at
evaluation would change regimes. Correct and shuffled candidate-lineage variants are
prespecified descriptive analyses. Retention uses correct candidate lineage as specified above.

## 7. Inference and decision hierarchy

Average each query's paired reciprocal-rank difference over the three training seeds before
bootstrap. Use typed endpoints

```text
(("resource", canonical_resource_hash), ("folder", tree_id))
```

with `paired_node_bootstrap_ci`, 9,999 resamples, seed `3998999`, confidence `0.95`, and identical
weights for every contrast. Report point estimate, percentile interval, bootstrap mean/attempts,
each seed, held-query count, unique pearl/resource/folder counts, and endpoint-connected components.
Fewer than 20 components makes every interval descriptive and unable to pass a gate.

For a prospective cohort, the typed query endpoint is
`("resource", canonical_resource_hash)`, not merely the new pearl ID; the second endpoint remains
`("folder", tree_id)`. This clusters duplicate URLs/content observed under distinct fresh IDs.
Exact pearl ID remains the grading/query identity and is reported separately. Missing canonical
resource identity blocks a confirmatory classification.

Intervals are conditional on the frozen target cohort and fitted checkpoints. They do not include
future-cohort, source-corpus, training-seed-population, or protocol-selection uncertainty.

Track R retention and Track T total transfer are the two root decisions. Both use central 95%
intervals, so each lower-tail test spends `0.025`; Bonferroni bounds their combined root-decision
family error by `0.05`. Specificity and e5 safety are tested only through the serial Track-T
gate. The gatekeeping order is:

1. Track R retention, classified independently;
2. Track T total transfer (`T1-T0`);
3. only if total transfer passes, transfer specificity (`T1-Tnull`);
4. only after specific transfer passes, e5 deployment noninferiority.

Secondary R@1/R@5, candidate-recalled-only MRR, hardness strata, duplicate-title slices,
conditioned variants, and individual mechanisms are descriptive. No secondary result rescues a
failed primary gate.

## 8. Privacy, artifacts, and release

Reverify the v3 bundle and privacy index on every run; copied IDs are insufficient. The four named
historical input checkpoints are currently ordinary mode-`0644` files. Before execution, read each
through a bound descriptor, verify its registered hash, and install a private mode-`0600`
single-link copy in the no-replace run-input bundle. All newly created SM-FS source rows, null
rows, embeddings, checkpoints, optimizer states, predictions, paths, and row-level reports stay
local in mode-`0700` directories and mode-`0600` no-replace files. No provider or API receives
SM-FS content.

Only reviewed aggregate metrics and content hashes may enter Git. Public-policy training provenance
does not authorize publishing a user-specific checkpoint. No checkpoint leaves the private
workspace without a separate memorization/privacy review.

## 9. Fail-closed conditions

Block rather than downgrade silently when:

- the negative ranking bundle or matched null ledger is absent or unverifiable;
- the migrated pre-Pearltrees checkpoint hash, provenance, or behavior equivalence cannot be
  verified;
- Pearltrees retention-baseline behavior and candidate-lineage scoring cannot be reproduced into a
  content-bound receipt;
- initial bytes differ within a paired seed;
- a transfer arm consumes any Pearltrees outcome before predictions are sealed;
- a reserved or quarantined SM-FS row is read;
- a candidate catalog uses placement counts or target outcomes;
- a prospective query overlaps any historical identity;
- privacy or source inventory cannot be rederived;
- candidates, tie handling, or scoring differ between arms;
- the behavior panel, determinism check, or artifact hashes fail; or
- the target cohort/catalog was not frozen in the required temporal order.

This version intentionally records `execution_authorized=false`. An amendment may turn it true only
after binding the negative and null ledgers, behavior panel, target cohort, exact training plan, and
all resulting content hashes. An honest blocked, null, nonretained, nonspecific, or operationally
inferior result completes the corresponding track.
