# Prospective protocol: process-expression P1 distillation

**Status:** frozen before construction of the P1 ledger, fitting, checkpoint selection, or held
scoring. This protocol governs the first process-expression experiment described in
`DESIGN_process_expression_implementation.md`. A material change requires a new protocol version
and preregistration ID before any affected held metric is computed.

P1 is an **exploratory, transductive representation study**. It cannot become confirmatory by
freezing a split now: the historical 1,200-query benchmark informed the routing thresholds, menu
sizes, judge/context choice, and process roster. A later bookmark cohort, drawn from a catalog
frozen before its placement labels exist, is required for a generalization or deployment claim.

## 1. Decision and estimand

The decision question is whether structured process-expression cards improve the same residual
ranker over flat learned process tokens. Merely beating an unconditioned pile does not answer that
question.

For each retained held query \(q\) and training seed \(s\), let
\(RR_{\mathrm{expr},s}(q)\) and \(RR_{\mathrm{flat},s}(q)\) be reciprocal rank under the identical
frozen candidate list, production-policy process selection, and exact-destination grade. The
primary paired value is

\[
d(q)=\frac{1}{3}\sum_s
\left[RR_{\mathrm{expr},s}(q)-RR_{\mathrm{flat},s}(q)\right],
\qquad
\Delta_{\mathrm{MRR}}=\operatorname{mean}_{q\in T}d(q).
\]

The expression arm is **superior** only when both:

1. \(\Delta_{\mathrm{MRR}}\ge 0.010\); and
2. the lower endpoint of the frozen 95% paired two-endpoint node-block interval is greater than
   zero.

If the interval lower endpoint is at least \(-0.005\), the result may be called
**in-distribution noninferior**, but not successful or superior. A later P3 zero-shot result may
justify retaining a noninferior representation; it must not retroactively change the P1
classification. Every other outcome is reported as practically positive but inconclusive,
statistically positive but below the floor, inconclusive, or inferior.

The expression arm also has a deployment-safety comparison against frozen e5. Failure to keep the
held MRR interval above the \(-0.010\) noninferiority margin bars deployment and selects the e5
rollback. This safety comparison is not evidence that expressions beat flat tokens.

## 2. Population and source-artifact gate

The historical counts—1,200 unique queries, 922 judged queries, 1,617 judge records, and 1,895
records after 278 automatic rows—describe the old development population. They are **not required
counts for P1**.

The historical judge tasks and picks are `legacy_unbound`, included nonpublic candidates, and
cannot be upgraded retroactively. They may not enter the primary ledger. They may be examined only
as a separately labeled, local-only sensitivity analysis that cannot tune P1 or satisfy an exit
criterion.

Primary P1 construction fails closed unless every judge-derived source has:

- a reproducible `unifyweaver.routed-task.v2` parent task from the certified-public population;
- `pearltrees-public-only-v1` and
  `pearltrees-public-alphanumeric-title-v1` receipts;
- a verified `unifyweaver.routed-execution-bundle.v1`;
- the byte-identical aggregate `unifyweaver.routed-picks.v2`; and
- the frozen policy, prompt, judge/model/revision/settings, e5 revision, task, plan, attempt-set,
  bundle, and pick identifiers.

The verifier must call the repository's task and execution-bundle re-derivation paths; copying
stored IDs into the ledger is insufficient. If no such bundles exist, the honest result is
`blocked_no_eligible_v2_labels`. Producing new labels is a separate, explicit spending decision.

Automatic rows are regenerated from the same verified public-only population and frozen policy.
They may not be appended from the historical population. Therefore the final primary-ledger count
is whatever survives the prospective privacy, population, execution, and join gates—not 1,895 by
fiat.

## 3. Process identity and ledger contract

The ledger is built and hashed before training. Process identity uses:

```
sha256(REGISTRY_VERSION + "|" + canonical_process_expression)
```

as the full 64-hex-character digest, plus the canonical expression, registry version, factory
fingerprint, and applicable task/policy/bundle identifiers. The current 16-hex `ast_sha()` is a
compact cache/display key and must never be the sole ledger or provenance identity.
`PROCESS_EXPRESSION_P1_PREREG.json` pins the registry version, renderer version, canonical string,
and full digest for every P1 process. Registry or canonicalization drift requires a new
preregistration rather than silently changing a process identity.

Each process-target record contains at least:

- stable typed bookmark ID and frozen query text hash;
- ordered candidate folder IDs, menu positions, and candidate-score content record;
- canonical process expression, full process digest, factory fingerprint, and card-cache keys;
- source kind (`judge_pick` or `auto_top1`);
- task ID, pick ID, bundle ID, execution-policy ID, policy ID, privacy-manifest digest, and full
  file content records where applicable;
- selected folder ID or explicit null reason (`null_majority` or `no_consensus`);
- process-normalized training weight; and
- split assignment.

The training ledger must not contain the recorded destination. Destination labels live in a
separately sealed evaluation artifact. The splitter may use typed bookmark/destination identities
to construct the split manifest, but training and checkpoint selection receive only the
outer-train labels. Outer-held destinations remain unreadable until all arms, checkpoints, and
inference code are frozen.

Every rendering and process record for one query stays on the same split side. Joins are one-to-one
on stable IDs and fail on missing, extra, duplicate, cross-task, cross-population, or ambiguous
title-only matches.

### Nulls and loss normalization

A null pick does not identify a positive folder and therefore contributes no primary listwise
ranking loss. Null-majority and no-consensus rows remain in the ledger and reports. Treating all
menu items as fractional negatives is an inner-training sensitivity only; it cannot be selected
after held metrics are visible.

Non-null rows use one normalized listwise cross-entropy per process record: the chosen folder is
positive and the other frozen menu folders are alternatives. The per-record loss is normalized
over its menu, then process weights are scaled so each process has equal total training mass.
This prevents N=20 or a duplicated process cohort from dominating by pair count.

## 4. Frozen split and training boundary

Use `node_disjoint_pair_split` over typed `(bookmark_id, destination_folder_id)` pairs with:

- outer seed `3980001`;
- held-node fraction `0.40`;
- `64` outcome-blind candidate assignments; and
- process/band labels as coverage strata.

Cross pairs are excluded from both fitting and evaluation. Persist the exact train, held, cross,
typed-node, and per-process manifests with full SHA-256 digests. All rows sharing a query follow
the query's pair assignment.

Hyperparameters, early stopping, checkpoint choice, loss details, trainable parameter sets, and
the candidate text/scoring pipeline are selected only through repeated node-disjoint inner splits
inside outer-train. Before the first fit, install a content-bound training-plan manifest containing
the complete grid, tie rule, maximum steps, optimizer, precision, determinism settings, and package
versions. No arm gets an additional search or training budget.

The three frozen training seeds are `3980101`, `3980102`, and `3980103`. Every arm starts from the
same base checkpoint and uses matched data order, batches, optimization budget, candidate lists,
and early-stopping rule for each seed.

## 5. Arms and conditioning

The frozen arms are:

1. **expression:** frozen card embedding and shared name projection plus one trainable residual row
   per registered process;
2. **flat:** one trainable process row per process with the same residual dimension and condition
   dropout, but no card-derived prior;
3. **merged:** no process identity; and
4. **shuffled:** expression architecture with training cards permuted across distinct process
   identities under five frozen train-only permutations.

V0/V1/V2/V3 sampling probabilities are `0.10/0.60/0.25/0.05`. The draw for a row and epoch is a
deterministic function of training seed, epoch, and stable ledger-row ID. The flat arm receives the
same 10% condition dropout; its other three outcomes map to the same flat process row.

Expression and flat arms must expose matched trainable-parameter and initialization reports.
Parameters common to both arms are initialized byte-identically per seed. Any unavoidable
parameter-count difference is reported and prevents a claim that representation alone caused the
gain.

The shuffled arm is a diagnostic, not part of the primary decision. Its permutation is derived
only from outer-train rows and never changes labels, split membership, process frequencies, or
verbosity frequencies.

## 6. Primary evaluation

Each held query produces exactly one ranking under the frozen production policy:

- margin below 0.02: `sonnet-lin-n10`;
- margin from 0.02 inclusive to 0.03 exclusive: `sonnet-lin-n20`; and
- margin at least 0.03: `e5-auto`.

Haiku rows are auxiliary supervision and a per-process diagnostic, not a second primary observation
for low-band queries. Both primary arms rank the same frozen public-only e5 top-100 catalog.
Missing true destinations receive reciprocal rank zero. Exact destination ID is primary;
best-title-equivalent grading is sensitivity-only.

P1 neither retunes the e5 margin thresholds nor calls them calibrated probabilities. They are a
frozen routing input inherited from the exploratory filing work. Any later confidence-policy
development must fit a joint calibrated combiner on inner node-disjoint data and evaluate
margin-gated risk/coverage separately; it is not an implicit part of this representation test.

Average each query's reciprocal rank across the three trained seeds before forming the paired arm
difference. Report each seed separately. The primary percentile interval uses
`paired_node_bootstrap_ci` with 9,999 resamples, seed `3980999`, confidence 0.95, and the same
bookmark/folder weights for both arms. Report the number of held queries, unique typed endpoints,
connected components, bootstrap attempts, bootstrap mean, and a deterministic rerun check.

This interval is conditional on the frozen label corpus and three fitted seeds. It does not include
judge-draw, provider-session, prompt-call, label-factory-selection, or future-cohort variation.
If the held graph has fewer than 20 endpoint-connected components, the interval is descriptive and
cannot satisfy the superiority gate.

## 7. Secondary analyses and multiplicity

Secondary, explicitly descriptive outputs are:

- expression versus merged;
- expression and flat versus e5;
- R@1 and NLL;
- per-process and per-band results;
- all individual training seeds;
- shuffled-card loss of conditioning gain;
- null-majority, no-consensus, abstention, and candidate-miss rates; and
- historical legacy-population sensitivity, if run entirely in the private workspace.

No secondary result rescues a failed primary contrast. P2 is authorized only by P1 superiority.
P3 may proceed as a separately preregistered mechanism study after P1 noninferiority, but not after
P1 inferiority or deployment-safety failure.

## 8. Privacy, artifacts, and reporting

Task, execution, ledger, split, label, feature, prediction, checkpoint, optimizer, and report-row
artifacts remain local-only. Nothing containing titles, browsing interests, private/nonpublic
membership, raw provider output, or model derivatives is committed.

Before any checkpoint is exported or published, conduct a separate release review. Public-only
training provenance does not by itself authorize publishing a user-specific filing model.
Historical private-inclusive sensitivity taints all transitive outputs as private.

The final report includes every tested configuration, including failures; exact source and
protocol IDs; ledger counts before and after each gate; process weights; split retention; candidate
recall; null reasons; all seeds; interval diagnostics; peak memory; and wall time. Allowed evidence
language is restricted to this finite, exploratory, transductive study. “Confirmed,” “production
gain,” “generalizes,” and “calibrated confidence” are not authorized.

An honest blocked, null, noninferior, or negative result completes P1.
