# Independent review: SM-FS ranking candidate lock

**Verdict:** authentic, reproducible preflight; **request changes before sequence step 4**.
The candidate does not authorize fitting, and this review does not amend the ranking or
retention/transfer preregistrations.

This review was completed before any ranking optimizer step, held-fold score, reserve access, or
step-4 amendment. It binds:

- candidate lock
  `~/mu_data/sm_fs_ranking_run_v1/candidate_lock.json`, SHA-256
  `6c4801cf387f0f8af075a22fbca5a411b8b0a659af899e99ca219f670d09f881`;
- landed commit `094de6345e3b7b88b36657fc8fc9c0d8ed340dcf`; and
- training plan SHA-256
  `01f34ab74478a9642a54e328fb4e2dec7f290c33fd95c695b7eb81ea03ea26df`.

The scientific parents are ranking preregistration
`0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2`
(protocol SHA-256 `25bac71a67b317700f6a5123af238f01e9b57225bf4deaf71a6470486bd194f3`)
and still-blocked retention/transfer preregistration
`f3e1d123e6f81191489689d7f7e0fd121e25ff891ada330fe668b8b3238497c1`
(protocol SHA-256 `5cbe305ed3eb0c2ecbb78b6f032e326c5c0c72bf415d218c538bbe3abdacd3d8`).

The accompanying JSON is the machine-checkable authority. The local path is provenance, not
publication permission.

## 1. What passed

The candidate is not fabricated or stale. Independent reproduction established:

- byte-identical candidate re-emission at the recorded commit;
- all nine recorded code hashes match that commit and the local checkout;
- the warm start, three authoritative initialized checkpoints, source target projection, ranking
  manifest, and training plan match their recorded hashes;
- the 30 jobs equal five folds by two arms by three seeds;
- the five training-query population hashes and global sampler-population hash rederive;
- the counter sampler reproduces its known-answer vectors and common-positive slots match across
  arms;
- the run and ranking directories are mode `0700`; the reviewed artifacts, including the repaired
  title cache, are regular, single-link, non-symlink mode-`0600` files;
- the recorded deterministic environment re-emits exactly on the reviewed host; and
- the candidate and current preregistration keep fitting false. Supplying the candidate to `fit`
  exits `3`.

These checks make the artifact a useful preflight receipt. They do not make the disabled scaffold
an executable training transaction.

## 2. Blocking findings

### CAND-1 — the reviewed trainer is intentionally incomplete

`sm_fs_ranking_train.py` contains the schedule primitives and a fitting gate, but `cmd_fit`
unconditionally stops after accepting a lock, while `cmd_evaluate` and `cmd_decide` unconditionally
stop. There is no model/optimizer loop, held-fold evaluator, exact-candidate scorer, bootstrap,
decision implementation, output receipt, or checkpoint transaction to review.

This violates the ordering frozen in `REVIEW_sm_fs_ranking_execution_lock.md`: the complete
trainer/coordinator/evaluator and candidate verifier must land **before** candidate generation and
independent review. Adding them during step 4 would change both the reviewed trainer hash and the
training-plan hash after review.

### CAND-2 — the final-lock gate accepts a self-asserted authorization

`fitting_gate` trusts fields supplied by the same caller-controlled JSON:

- final-looking schema;
- `fitting_authorized=true`;
- current preregistration ID and its easily derived digest; and
- `independently_verified=true`.

It does not require the live preregistration itself to authorize fitting and does not verify a
separate review receipt, final-lock hash, training-plan hash, code hashes, bundle, initialized
bytes, environment, modes, or link shape. A hand-written five-field lock therefore passes the
gate even while the live preregistration says `model_fitting_authorized=false`. No optimizer is
reachable today only because `cmd_fit` has a second unconditional hard stop.

Replace this with a cycle-free verifier of an independently emitted verification receipt bound to
the exact final-lock SHA. The live preregistration must also say fitting is authorized.

### CAND-3 — held outcomes are not capability-separated

`load_bundle` reads all 129,599 pair rows, including every held query-to-destination association,
into the fitter process. The statement “held rows filtered before any training materialization”
is present only as plan text.

The trusted coordinator must create and verify a train-only, descriptor-bound projection for one
fold. The fitter may receive the shared transductive catalog and title inputs, but it must not
receive held outcome associations. A separate evaluator may receive held rows only after the
checkpoint and fit receipt are sealed.

### CAND-4 — required execution bindings are absent

The candidate and plan do not bind or enforce all fields required by section 4 of the ranking
protocol:

- title-table and E5-cache hashes plus E5 revision and tokenizer structure;
- complete Adam options;
- exact 18-name/shape trainable allowlist and frozen-parameter assertion;
- frozen-reference construction;
- augmentation parameters and NumPy/Torch RNG reset and consumption order;
- dtype and complete runtime provenance;
- evaluator, scorer, bootstrap, and decision code;
- tie handling and output/checkpoint/prediction/receipt schemas; and
- the exact initialized-checkpoint reuse checks at every fold/arm job.

Code hashes cannot substitute for fields or enforcement, especially when the bound code is a
scaffold.

### CAND-5 — no candidate-bound execution or final-lock verifier existed

At the reviewed commit, the candidate schema occurs only in the emitter. The existing
execution-review verifier recognizes the earlier proposal, not
`unifyweaver.sm-fs-ranking-execution-lock.candidate.v1`. No candidate-bound test performs
install/verify/tamper rejection or final-lock/independent-receipt verification. This review adds an
authenticity-and-**rejection** verifier for the frozen candidate; it is deliberately not the missing
execution/authorization verifier.

### CAND-6 — the transaction and source-isolation contract remains partial

The reviewed files happen to have the right modes and link shape. The implementation, however,
writes directly to final pathnames with `O_EXCL` and file fsync. It has no private staging
transaction, parent-directory fsync, no-replace rename, rollback, or enforced pre/post
mode/link/symlink verification. A crash can leave a partial permanent target.

Inputs are hash-checked and then reopened by pathname. They are not read from the same verified
descriptor, so the promised descriptor-bound isolation is not implemented.

### CAND-7 — secondary fail-closed perimeter gaps

- Critical emitter and environment checks use `assert`, which `python -O` disables.
- Git subprocess return codes are not checked.
- The cleanliness statement means tracked `prototypes/mu_cosine` files only; untracked lane files
  and other tracked paths are excluded.
- Environment provenance omits Python, NumPy, CUDA/device/driver, dtype, interop threads, and RNG
  state/reset order.
- Tests bound by the candidate at commit `094de6345` cover the negative gate and sampler fragments,
  not schedule exhaustion, capability separation, optimizer/trainables, evaluation, inference,
  atomicity, or executable candidate/final-lock verification. This review's tests cover only the
  exact rejected candidate's authenticity and inability to self-upgrade.

## 3. Required replacement candidate

Keep the current preregistration blocked while the engineering lane:

1. removes hard-coded preregistration IDs from hash-bound execution/sampler code (derive IDs from
   canonical live documents, or extract an ID-free stable sampler module);
2. lands the complete, still-gated fit/evaluate/decide transaction and its tests;
3. replaces the self-asserted lock gate with independent receipt verification;
4. implements train-only capability separation and post-seal held evaluation;
5. binds and enforces every field in CAND-4;
6. installs verified inputs and outputs through crash-atomic private transactions;
7. implements candidate/final-lock verification with tamper tests; and
8. emits a fresh plan and candidate in a new no-replace namespace from a clean landed commit.

That replacement candidate then receives a new independent review. Do not overwrite or “repair”
the reviewed candidate in place.

## 4. Conditions for the later step-4 amendment

After a replacement candidate passes, preserve the existing exploratory,
catalog-transductive estimand, 361 queries, 359 candidates, five folds, 82 unsplit lineage blocks,
two paired arms, three ranking seeds, budgets, initialized bytes, no early stopping, and untouched
1,481-row reserve. Preserve fold-assignment SHA-256
`b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37`,
evidence scope `map-near-lineage-blocked-catalog-transductive`, exact-destination MRR,
`graded-negative-minus-positive-only`, the `+0.010` point floor, strict lower-CI `>0` gate,
ascending frozen catalog-column tie rule, no secondary rescue, and the consequence
`new-reserve-preregistration-only`.

Mechanically freeze the primary inference:

1. compute each arm's reciprocal rank for every query and seed;
2. average seeds within query before forming the paired arm difference;
3. for each replicate, draw exactly 82 block indices with replacement from the canonical
   UTF-8-sorted 82-block list, using identical multiplicities for both arms;
4. compute
   `sum_b multiplicity_b * sum_{q in b} d(q) / sum_b multiplicity_b * n_b`, retaining the
   query-weighted mean when block sizes differ;
5. use a versioned SHA-256 rejection sampler whose canonical-JSON key contains exactly schema,
   sampler ID, seed, replicate, draw, and retry; interpret the digest as an unsigned big-endian
   256-bit integer and reject above `floor(2^256/82)*82` before modulo;
6. use 9,999 resamples, seed `3997999`, and nearest-rank central-percentile endpoints: zero-based
   sorted replicate indices `249` and `9749`; and
7. fail closed on missing/nonfinite predictions, inconsistent candidates, or fewer than 20
   eligible blocks in the **observed** population. A replicate may contain fewer than 20 unique
   blocks after sampling with replacement.

Avoid an ID/hash cycle before generating the replacement candidate: derive preregistration IDs
from canonical documents, and do not hard-code a new ID into a module whose hash the same
preregistration binds.

The later retention cascade is narrow. Only the source ranking protocol hash/ID and verified
negative-bundle status/hash may change. Derive a new retention ID while preserving
`execution_authorized=false`, null-ledger hash `null` and blocked status, behavior-panel hash
`null`, the fresh-cohort block, reserve/quarantine false, Track-R base/arms/margins, Track-T
arms/seeds/title-only primary/gates, inference and Bonferroni fields, and every privacy/release
gate. Track T must still regrow independently from `model_prod_namecond_full.pt` under seeds
`3998101`--`3998103`; ranking initializations and fitted checkpoints remain forbidden in Track T.

Until a replacement candidate passes review, step 4, the retention cascade, model fitting,
held-fold scoring, reserve access, and checkpoint release all remain unauthorized.
