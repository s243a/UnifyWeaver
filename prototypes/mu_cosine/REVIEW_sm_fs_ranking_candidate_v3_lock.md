# Independent review: SM-FS ranking candidate v3

**Verdict:** authentic and materially improved static artifact; **request changes before sequence
step 4**. Candidate v3 does not authorize fitting, and this review does not amend either the
ranking or retention/transfer preregistration.

This review was completed before any decision-bearing optimizer step, held-fold score, reserve
access, or step-4 amendment. It binds:

- candidate lock
  `~/mu_data/sm_fs_ranking_run_v3/candidate_lock_v3.json`, SHA-256
  `33628b5da5f3bed7e6f7afb0f21a633decac27f348bc256bd52079dd4f47efcf`;
- landed commit `db5dea709dff77f39a1b772c3bb9ea690b41e077`; and
- the candidate-v2 rejection review
  `916cad2f7b3a9d99fa4b8549cd4423b4849c3f205075cd0734e3672a9044aa2d`.

The scientific parents remain ranking preregistration
`0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2`
(protocol SHA-256 `25bac71a67b317700f6a5123af238f01e9b57225bf4deaf71a6470486bd194f3`)
and still-blocked retention/transfer preregistration
`f3e1d123e6f81191489689d7f7e0fd121e25ff891ada330fe668b8b3238497c1`
(protocol SHA-256 `5cbe305ed3eb0c2ecbb78b6f032e326c5c0c72bf415d218c538bbe3abdacd3d8`).

The accompanying JSON is the machine-checkable authority. The local candidate path is provenance,
not publication or fitting permission.

## 1. What passed

Candidate v3 closes important failures from both earlier candidates:

- its exact SHA-256, schema, landed commit, eight code hashes, title table, ranking manifest,
  manifest-bound pair/fold files, and three initialized checkpoint hashes reproduce;
- plan SHA-256
  `ff81a8fda9f2cea9c21c21e0d27341661d42046368925fd04e4c3c11f6e2856d`
  binds 30 unique jobs covering five folds, two arms, and three seeds;
- all five train-only projections rederive byte-for-byte from the 129,599 source pairs and exclude
  every corresponding held query:

  | fold | held queries | train rows | projection SHA-256 |
  |---:|---:|---:|---|
  | 0 | 73 | 103,392 | `3bd7feb3742895d72866d5ba5dc90427db34ad973ec95fa43fe5f878f664b539` |
  | 1 | 72 | 103,751 | `1699ff88e2d6e3bfe3f7622e01bd5c6f1aed841c9dc603189d3a9752791d6f20` |
  | 2 | 72 | 103,751 | `9c605c9abaea5142c78d551d355abeedfa23026a19e5246610f7c32c1e3084cc` |
  | 3 | 72 | 103,751 | `eb9ea3a4633d2289a8279af856578f7eb9e00a011889d038bf8388a9cfd0e0b5` |
  | 4 | 72 | 103,751 | `420fbb7772d9f173001fc3c371ea06ce60ea2af7f98ef70893f37ab93e102891` |

- all three exact initialized checkpoints load through `load_expanded` and resolve the
  candidate-bound 18 tensors / 1,195,782 parameters;
- a genuine optimizer construction and one synthetic fit step succeed on both CPU and the
  GTX 1660 SUPER GPU without reading held or reserved rows;
- all 30 counter-based selection schedules execute for all 800 steps with identical common slots
  between paired arms;
- the training sampler reproduces all four preregistered known answers;
- the decision bootstrap correctly implements the canonical SHA-256 rejection sampler, 82 draws
  per replicate, 9,999 replicates at seed `3997999`, query-weighted block means, endpoints
  `sorted[249]` and `sorted[9749]`, the registered contrast direction, and the
  `+0.010`/strict-positive-lower-bound gate;
- exact-score finiteness and ascending frozen-column tie handling are correct in the real scorer;
  and
- both live preregistrations and the candidate still say false, and the v3 namespace contains no
  fit, evaluation, decision, final-lock, or verification-receipt output.

These are real advances. They establish an executable training mechanism and authentic static
preflight, but they do not establish the authorization and evidence chain around that mechanism.

## 2. Blocking findings

### CAND3-1 — the accepted-review and final-state chain is caller-forgeable

`sm_fs_ranking_lock_verify.py` still emits only a candidate. It has no final-lock emitter and no
independent verification-receipt emitter. `verify_final_lock` accepts any JSON file whose basename
starts with `SM_FS_` when it contains three caller-selected values: `verdict: "accepted"`, the
candidate SHA, and the same arbitrary review ID copied into the lock.

Nothing proves that this artifact is Git-tracked, equals the blob at `HEAD`, follows a registered
review schema, or has a canonically derived review ID. This is directly exploitable because
`lane_clean()` deliberately ignores untracked files. Independent reproduction placed an untracked
caller-authored `SM_FS_*.json` beside the code and obtained:

```text
UNTRACKED_REVIEW_ACCEPTED True
```

Binding the candidate SHA does not bind the execution state. The final verifier compares the
final lock's fields with the *current* files, never with the reviewed candidate's fields.
Reproduction changed `steps` from the candidate's `800` to a hypothetical live/final `801`; the
old candidate SHA and an “accepted” review of that old candidate still passed:

```text
POST_REVIEW_STATE_DRIFT_ACCEPTED candidate_steps=800 final_steps=801
```

Thus arbitrary post-review code, plan, artifact, optimizer, or environment changes can be
presented as a new final state while retaining the reviewed candidate SHA.

`fitting_allowed` also treats a four-field caller-authored receipt as independent verification.
It validates no verifier authority or committed receipt bytes. The candidate and verifier bind
neither starting preregistration document, the allowed amendment diff, nor the retention
preregistration. A live ranking document with a self-consistent new ID and `model_fitting_authorized:
true` can change the estimand, arms, thresholds, bootstrap, or reserve boundary and can skip the
required narrow retention cascade without rejection.

This leaves CAND2-3 open. Finalization must be implemented and reviewed before candidate
generation, not hand-authored after the review.

### CAND3-2 — held evaluation and the primary decision still trust self-authored receipts

`validate_fit_receipt` checks schema/job strings, the registered initialization string, the
projection string, that an arbitrary inventory list has length 18, that an arbitrary environment
contains `dtype: float32`, and that a checkpoint hash looks like 64 characters. It does not check
the plan SHA, final-lock SHA, review/candidate chain, exact inventory, optimizer/budget,
environment, code commit, or receipt provenance. A nine-field receipt containing 18 copies of
`["forged", [1]]` is accepted. An arbitrary compatible checkpoint plus its own hash can therefore
reach the evaluator without proof that authorized fitting produced it.

The decision worker is weaker still. It ignores the evaluation checkpoint and fit-receipt hashes,
score-vector length and finiteness, destination, stored rank, catalog size, held count, environment,
and every diagnostic. It does not recompute rank or reciprocal rank from sealed scores. It trusts
each row's caller-supplied `rr`.

Independent reproduction created the required 30 mode-`0600` evaluation receipts containing only
job/catalog/tie metadata and `{query, rr}` rows—no checkpoint, fit receipt, scores, rank,
destination, or diagnostics. Positive-only rows used `0.1`; graded-negative rows used `0.6`.
The real decision worker emitted:

```text
delta_mrr: 0.49999999999999994
ci95: [0.4999999999999999, 0.49999999999999994]
passed_exploratory_gate: true
authorizes: new-reserve-preregistration-only
```

This is a direct false authorization path. CAND2-4 is not closed: held-fold access and the primary
decision are not capability-separated until every receipt is derived, chained, and recomputed.

### CAND3-3 — the registered output and runtime contract remains incomplete

The evaluator stores the full score vector, which is useful, but the decision never validates or
uses it. The evaluator computes unweighted MSE/MAE although section 5 of the protocol requires
weighted MSE/MAE. It does not emit relation or hardness slices, R@1/R@5/R@10, title-equivalent
sensitivity, or the registered descriptive controls. Its `per_seed_mrr` result contains fifteen
fold-by-seed job means per arm rather than three whole-population seed-specific results.

The environment split correctly allows invariant verification in a CPU-only sandbox while
recording the observed GPU descriptively. However, the protocol requires CUDA, GPU, and driver
provenance before execution; the driver is not recorded. Evaluation receipt validation checks
only `dtype`, so paired jobs from different runtime contexts are accepted.

These defects need not alter the primary estimand, but they leave reporting and reproducibility
choices after scoring and contradict the candidate's claim that the complete diagnostics and
environment chain are enforced.

### CAND3-4 — transaction rollback and descriptor handoff are not fully durable

The successful private-install path is substantially stronger: real mode-`0700` parent checks,
mode-`0600` staging, data fsync, hard-link no-replace, two directory fsyncs, single-link
verification, and content re-read all work.

On failure, cleanup suppresses unlink failures and performs no directory fsync after rollback.
An injected first directory-fsync failure plus target-unlink failure raised to the caller while
leaving the installed target behind. Even when unlink succeeds, its rollback is not made durable.
Parent checking and subsequent operations are path-based rather than held-dirfd-relative, and
the fitter reopens plan/title/projection/checkpoint paths instead of consuming the bytes or
descriptors authenticated by finalization.

These are perimeter rather than estimand defects, but the candidate explicitly claims the stronger
crash-atomic rollback and descriptor/byte-handoff contract.

### CAND3-5 — the claimed decision-bearing integration tests do not exist

The six new tests do establish a real checkpoint load, allowlist, optimizer, one synthetic CPU
step, finite scorer, bootstrap fragments, and successful-path transaction behavior. None calls
`cmd_evaluate`, `cmd_decide`, `verify_final_lock`, or `fitting_allowed`; none checks a real
fit/evaluation/finalization receipt chain; none recomputes rank from a receipt; and none injects a
post-link transaction failure.

The test named for receipt substitution deliberately accepts a nine-field fixture that omits the
final-lock and plan chain. The bootstrap test checks four block draws and rejection cases, not a
full decision known answer. This coverage gap is why both caller-forgeable authorization paths
survived despite the candidate claiming CAND2-8 was closed.

## 3. Required replacement candidate

Keep both preregistrations blocked while the engineering lane:

1. emits final-lock and verification-receipt artifacts through explicit no-replace transactions;
2. requires a registered review schema whose canonical ID is rederived, whose bytes are proven
   Git-tracked at the exact reviewed commit, and whose accepted candidate is authenticated;
3. compares every final execution binding with the reviewed candidate and rejects all drift except
   an explicit, machine-checked preregistration amendment whitelist;
4. binds and verifies both starting preregistrations and protocols, the exact ranking amendment,
   and the narrow still-blocked retention cascade;
5. makes fit receipts exact and candidate/final/review-bound, then requires that chain before a
   loadable checkpoint can expose held outcomes;
6. makes evaluation receipts exact and recomputes destination, rank, and reciprocal rank from the
   bound finite score vector before decision use;
7. validates the complete 30-job population, catalog, query/fold population, checkpoint and fit
   chain, runtime, and registered diagnostics inside the decision worker;
8. implements weighted diagnostics and the frozen secondary reporting contract, including three
   whole-population seed-specific results;
9. completes rollback fsync/error handling and descriptor-relative or verified-byte handoff;
10. adds adversarial end-to-end tests for finalization, held access, evaluation, primary decision,
    post-review drift, forged/untracked reviews, forged receipts, and rollback failure; and
11. emits a fresh plan and candidate in a new no-replace namespace from the clean landed commit.

That candidate receives another independent review. Do not amend or repair candidate v3 in place.

## 4. Conditions preserved for the later amendment

The frozen estimand remains exploratory and catalog-transductive: 361 queries, 359 candidates,
five folds, 82 unsplit lineage blocks, two paired arms, three seeds, exact-destination MRR,
`graded-negative-minus-positive-only`, `+0.010` minimum point gain, strict lower CI greater than
zero, ascending frozen catalog-column tie rule, and no secondary rescue. A pass authorizes only a
new reserve preregistration; it never authorizes opening the 1,481-row reserve.

The bootstrap remains the now-correct v3 implementation: average seeds within query before the
contrast, draw 82 UTF-8-sorted blocks with replacement using the versioned SHA-256 rejection
sampler and identical multiplicities, retain the query-weighted mean, use 9,999 replicates at seed
`3997999`, and take nearest-rank endpoints 249 and 9749.

The later retention cascade remains narrow and blocked. Only the source ranking protocol hash/ID
and verified negative-bundle status/hash may change. Track T must independently regrow its
pre-Pearltrees base under seeds `3998101`--`3998103`; ranking initializations and fitted
checkpoints remain forbidden there.

Until a replacement candidate passes review, step 4, the retention cascade, model fitting,
held-fold scoring, reserve access, and checkpoint release all remain unauthorized.
