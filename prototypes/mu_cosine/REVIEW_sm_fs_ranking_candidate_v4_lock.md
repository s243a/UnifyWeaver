# Independent review: SM-FS ranking candidate v4

**Verdict:** authentic and materially improved static artifact; **request changes before sequence
step 4**. Candidate v4 does not authorize fitting, and this review does not amend either the
ranking or retention/transfer preregistration.

This review was completed before any decision-bearing optimizer step, held-fold score, reserve
access, or step-4 amendment. It binds:

- candidate lock
  `~/mu_data/sm_fs_ranking_run_v4/candidate_lock_v4.json`, SHA-256
  `b5a2a7e7034a937d8241e437d48536592c579bca13dcd7af59b0d9bc4e21f4fa`;
- landed commit `fe87d08c8c96b5b0b886dc3154dc18d14efd8250`;
- training plan SHA-256
  `1cdb659bd3baa291af1847de764f46148318df98f95ea4b286270868f8a486d1`; and
- the candidate-v3 rejection review
  `421fbab584df2c02eb34d11d4d4a8ee3a536ef06ab38f38ba8eb585b68da71f6`.

The scientific parents remain ranking preregistration
`0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2`
(protocol SHA-256 `25bac71a67b317700f6a5123af238f01e9b57225bf4deaf71a6470486bd194f3`)
and still-blocked retention/transfer preregistration
`f3e1d123e6f81191489689d7f7e0fd121e25ff891ada330fe668b8b3238497c1`
(protocol SHA-256 `5cbe305ed3eb0c2ecbb78b6f032e326c5c0c72bf415d218c538bbe3abdacd3d8`).

The accompanying JSON is the machine-checkable authority. The local candidate path is provenance,
not publication or fitting permission.

## 1. What passed

Candidate v4 closes several concrete v3 failures:

- its exact SHA-256, schema, landed commit, nine code hashes, title table, source manifest,
  preregistration hashes, initialized checkpoints, and invariant environment reproduce;
- the plan contains exactly 30 unique decision jobs covering five folds, two arms, and three
  seeds, and lists the two descriptive controls;
- all five train-only projections retain their reviewed byte identities and exclude the
  corresponding held queries;
- all three initialized checkpoints load through `load_expanded` and resolve the same 18 tensors
  and 1,195,782 trainable parameters;
- genuine optimizer construction and a synthetic fit step pass;
- the frozen sampler and bootstrap remain correct: 82 lineage blocks, 82 draws per replicate,
  9,999 replicates at seed `3997999`, endpoints 249 and 9749, query-weighted aggregation, and the
  registered contrast direction and gate;
- fit receipts now carry the intended chain metadata, and the evaluator verifies a fit receipt
  and checkpoint bytes before opening held rows;
- the decision worker rejects nonfinite or wrong-length score vectors and recomputes rank and
  reciprocal rank instead of trusting stored `rank` or `rr`;
- rollback errors are surfaced and successful rollback is directory-fsynced; and
- candidate, ranking preregistration, and retention preregistration all remain blocked, while the
  v4 namespace contains no fit, evaluation, decision, final-lock, or verification-receipt output.

These are real advances. They authenticate the static preflight and improve the record format, but
they do not yet authenticate the scientific evidence carried by those records.

## 2. Blocking findings

### CAND4-1 — arbitrary score vectors still pass the primary decision

Candidate v3 trusted a caller-supplied reciprocal rank. Candidate v4 correctly recomputes rank
from `row["scores"]`, but it never proves that those scores were produced by the bound checkpoint
and evaluator. `validate_fit_receipt` accepts a syntactically valid checkpoint hash plus copied
public chain fields; `cmd_decide` never opens that checkpoint or reproduces a score.

Independent safe reproductions created the complete 30-job receipt population in `/tmp`, used all
82 frozen lineage blocks and the real 9,999-replicate bootstrap, and created **no checkpoint
files**. Finite caller-selected scores made the graded arm win:

```text
FABRICATED_SCORES_ACCEPTED True
DELTA 0.5 CI [0.5, 0.5]
AUTH new-reserve-preregistration-only
```

A second reproduction produced delta and interval `0.987804878...`, also passing. Evaluation
receipts could omit environment, held count, R@1/R@5/R@10, weighted errors, slices, AUC, nDCG,
title sensitivity, and Git provenance without rejection.

This is not fixed by hashing the same caller-authored receipt. Either the decision must reproduce
predictions from authenticated checkpoint bytes and the frozen evaluator/input bytes, or a
separately authenticated verifier must bind that complete computation. The same provenance issue
applies one stage earlier: a compatible checkpoint and field-correct self-authored fit receipt can
cross the held-data boundary without proof that the authorized fitter produced it.

### CAND4-2 — the post-review preregistration amendment is unrestricted

The final-lock whitelist restricts which **lock keys** may change, but permits the entire ranking
and retention preregistration hashes and IDs to be replaced. `verify_prereg_pair` then checks only:

- ranking hash, `model_fitting_authorized: true`, and a self-derived ranking ID; and
- retention hash and `execution_authorized: false`.

It does not recover the candidate-bound starting documents, structurally diff amendments, enforce
allowed JSON pointers and values, rederive the retention ID, verify the retention-to-ranking link,
or verify either live protocol blob.

Independent reproductions passed after changing the primary metric and practical floor, changing
the passing action to open the reserve, marking reserve data opened/scored, enabling reserve,
quarantined-row, and checkpoint-release access, and inserting an arbitrary retention ID. Merely
leaving top-level retention `execution_authorized` false was sufficient:

```text
ARBITRARY_PREREG_REWRITES_ACCEPTED
ranking.primary.metric = ATTACKER_CHANGED_AFTER_REVIEW
retention.reserve_rows_authorized = true
```

The candidate review therefore does not preserve the estimand, privacy boundary, or narrow
retention cascade. Starting preregistration bytes must be recovered from the candidate commit and
hash-checked, then compared structurally with exact permitted amendments. Both protocol blobs
must also be authenticated.

### CAND4-3 — Git tracking and a four-field receipt are not independent authority

The proposed committed-review seam is directionally useful, but `verify_accepted_review` accepts
any schema beginning with a prefix and any accepting document committed at local `HEAD`. A caller
can invent and locally commit its own review:

```text
ARBITRARY_PREFIX_SCHEMA_ACCEPTED
unifyweaver.sm-fs-ranking-candidate-review.attacker-defined
```

Git tracking proves byte stability, not reviewer identity or merge approval. Use one exact schema
and path, invoke its complete verifier, and anchor acceptance to the repository's chosen trust
root (for example a user-merged review commit or cryptographic countersignature). The review
should bind candidate SHA and commit, plan SHA, human report SHA, and the exact authorized
amendment.

There is also no production final-lock or verification-receipt emitter.
`verify_receipt` accepts four public copied fields, so a caller-authored receipt passes. Since
`chain_context` already recomputes the final state, this receipt is redundant rather than
independent evidence.

### CAND4-4 — the registered reporting and control contract remains incomplete

Evaluation emits useful per-query material, including full scores, weighted global MSE/MAE,
relation/hardness slice MSE, AUC, nDCG@10, R@1/R@5/R@10, and a title-equivalence field. But the
decision neither validates nor aggregates those fields. The two controls have no required
fold/seed job matrix or completeness gate, and `binary_zero_negative` is absent. Consequently
controls and secondary outputs can be selectively run or reported after scores exist.

The title-equivalence implementation compares path-leaf strings instead of the bound title table.
Those happen to induce identical groups for all 359 candidates in this corpus, so this does not
change the present result, but it is not the registered data source. The exact nDCG gain and cutoff
also need freezing rather than relying on an implementation choice.

A final decision artifact must carry the complete prespecified descriptive report—without allowing
any secondary result to rescue the primary—and must state the registered non-pass outcome.

### CAND4-5 — verified-byte handoff and decision-bearing tests remain incomplete

`chain_context` reads the final lock and receipt without the private-mode requirement, and
`verify_final_state` reopens the candidate with plain `open`. More importantly,
`load_pairs_verified` reopens the live manifest, pairs, and folds after final-state verification
without accepting the candidate-bound manifest hash or carrying verified bytes across the handoff.
A consistent source replacement in that gap is not tied to the reviewed candidate.

The ten supplied tests never call `verify_final_state`, `verify_prereg_pair`, `verify_receipt`,
`chain_context`, `cmd_evaluate`, or `cmd_decide`. The test described as decision recomputation
calls only `recompute_rank` on three numbers. That coverage gap is why the full forged-chain and
fabricated-score paths survive.

## 3. Required replacement candidate

Keep both preregistrations blocked while the engineering lane:

1. defines one exact accepted-review schema and fixed path, verifies its full schema-specific
   contract, and anchors it to the repository's independent merge/countersign trust root;
2. binds candidate SHA and commit, plan SHA, review-report SHA, protocol blobs, and an explicit
   JSON-pointer/value amendment contract;
3. recovers both starting preregistration blobs from the candidate commit, checks their
   candidate-bound hashes, and structurally rejects every non-authorized change;
4. rederives both amended preregistration IDs, verifies the retention-to-ranking link, and enforces
   the exact still-blocked retention cascade and all privacy/release fields;
5. emits the final lock and verification receipt through reviewed private no-replace transactions,
   with a genuine independent verifier identity or drops the redundant receipt claim;
6. makes the decision reproduce scores from authenticated checkpoints and frozen evaluator/input
   bytes, or uses an actually isolated authenticated scoring verifier;
7. gives fitted checkpoint provenance the same protection before held bytes can be opened;
8. carries candidate-bound source bytes or expected source hashes through every held-data loader;
9. completes the registered controls, binary-zero sensitivity, secondary diagnostics, exact nDCG
   definition, title-table equivalence, and whole-population report;
10. adds true end-to-end tests for candidate → review → exact amendment → final lock → receipt →
    fit → evaluate → decide, plus arbitrary-primary, reserve, protocol, checkpoint, score-table,
    missing-diagnostic, and source-substitution attacks; and
11. emits candidate v5 from a fresh private no-replace namespace and clean landed commit.

That candidate receives another independent review. Do not amend or repair candidate v4 in place.

## 4. Conditions preserved for the later amendment

The frozen estimand remains exploratory and catalog-transductive: 361 queries, 359 candidates,
five folds, 82 unsplit lineage blocks, two paired arms, three seeds, exact-destination MRR,
`graded-negative-minus-positive-only`, a `+0.010` minimum point gain, strict lower CI greater than
zero, ascending frozen catalog-column tie rule, and no secondary rescue. A pass authorizes only a
new reserve preregistration; it never authorizes opening the 1,481-row reserve.

The bootstrap remains the correct implementation: average seeds within query before contrast,
draw 82 UTF-8-sorted blocks with replacement using the versioned SHA-256 rejection sampler and
identical multiplicities, retain the query-weighted mean, use 9,999 replicates at seed `3997999`,
and take nearest-rank endpoints 249 and 9749.

The later retention cascade remains narrow and blocked. Only the source ranking protocol hash/ID
and verified negative-bundle status/hash may change. Track T must independently regrow its
pre-Pearltrees base under seeds `3998101`--`3998103`; ranking initializations and fitted
checkpoints remain forbidden there.

Until a replacement candidate passes review, step 4, the retention cascade, model fitting,
held-fold scoring, reserve access, and checkpoint release all remain unauthorized.
