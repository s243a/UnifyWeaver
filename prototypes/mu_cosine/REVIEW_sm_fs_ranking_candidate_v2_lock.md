# Independent review: SM-FS ranking candidate v2

**Verdict:** authentic static artifact; **request changes before sequence step 4**.
Candidate v2 does not authorize fitting, and this review does not amend either the ranking or
retention/transfer preregistration.

This review was completed before any ranking optimizer step, held-fold score, reserve access, or
step-4 amendment. It binds:

- candidate lock
  `~/mu_data/sm_fs_ranking_run_v2/candidate_lock_v2.json`, SHA-256
  `a490cff99178ca0187c1e11a880c6618e00594bc4b9e7e8757171440159440e3`;
- landed commit `64ee948ed94d502038e5c46421eab5844c9dd899`; and
- the prior rejection review
  `68e036c59c210002c50eafcbf6333cdb43393383fe916cd1449942f9fcc584f1`.

The scientific parents remain ranking preregistration
`0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2`
(protocol SHA-256 `25bac71a67b317700f6a5123af238f01e9b57225bf4deaf71a6470486bd194f3`)
and still-blocked retention/transfer preregistration
`f3e1d123e6f81191489689d7f7e0fd121e25ff891ada330fe668b8b3238497c1`
(protocol SHA-256 `5cbe305ed3eb0c2ecbb78b6f032e326c5c0c72bf415d218c538bbe3abdacd3d8`).

The accompanying JSON is the machine-checkable authority. The local candidate path is provenance,
not publication permission.

## 1. What passed

The candidate file is the artifact Fable reported, rather than a fabricated replacement:

- its SHA-256 and landed commit match the handoff;
- every recorded code hash, the ranking-manifest hash, title-table hash, and all three initialized
  checkpoint hashes match disk;
- the manifest's `pairs.jsonl` and `fold_assignment.tsv` output hashes match disk;
- the source bundle contains 361 queries, 359 candidates, 129,599 pairs, 82 lineage blocks, and
  fold populations `73/72/72/72/72`;
- the actual fold-0 projection is mode `0600`, hash-consistent, contains 103,392 rows, and excludes
  all 73 fold-0 queries; and
- the live ranking and retention preregistrations remain blocked, so the first optimizer step is
  not currently reachable.

For complete finite inputs, the evaluator's exact-destination rank, ascending catalog-column tie
rule, arm direction, three-seed within-query averaging, query-weighted block aggregation,
`+0.010` point floor, and strict lower-CI `> 0` gate are directionally correct.

These facts authenticate useful pieces of the replacement. They do not make the candidate an
executable or independently countersigned training transaction.

## 2. Blocking findings

### CAND2-1 — the complete fitter cannot load the countersigned checkpoints

Both `cmd_fit` and `cmd_evaluate` construct `MuAttention(**blob["cfg"])`. Every countersigned
initialization stores the established legacy config:

```text
{"d_model": 384, "heads": 4, "layers": 3, "judge_name": true,
 "ridge": 0.1, "op_name": true, "corpus_name": true}
```

`MuAttention.__init__` accepts `n_heads` and `n_layers`, not `heads` and `layers`, and does not
accept the remaining legacy flags. Independent reproduction raises
`TypeError: MuAttention.__init__() got an unexpected keyword argument 'heads'` before optimizer
construction. The established compatible loading path is `load_expanded`; the new transaction
does not use or reproduce it.

Thus CAND-1 is not closed: source text for a loop exists, but no registered fit can execute.

### CAND2-2 — the preregistration ID cycle remains and no fresh plan exists

Candidate-bound `sm_fs_protocols.py` still hard-codes both current preregistration IDs. Step 4 must
derive new ranking and retention IDs, which forces a change to candidate-bound code or makes the
validators reject the amended documents. Either outcome invalidates this reviewed candidate.

The v2 namespace has no training plan. It contains a candidate, title table, and one fold-0
projection. The only `training_plan.json` is the obsolete v1 plan bound to the rejected scaffold.
Candidate v2 consequently binds no exact 30-job matrix, per-fold projection identities, schedule
population, or plan hash. The prior review explicitly required an ID-free sampler/validator seam
and a fresh plan plus candidate; neither was delivered.

### CAND2-3 — the final lock and “independent” receipt are not candidate-bound

`sm_fs_ranking_lock_verify.py` emits only a candidate. It has no final-lock emitter, no
verification-receipt emitter, and no verifier for the provenance or authority of a receipt.

After the live preregistration becomes true, `fitting_allowed` accepts a receipt containing the
right schema, final-lock hash, preregistration ID, and any nonempty string other than `"self"` in
the `verifier` field. A caller can write both JSON objects. The final lock contains no reviewed
candidate SHA or review ID, and its common verification is recomputed against whatever code and
data happen to exist after step 4. Post-review changes can therefore be presented as a wholly new
accepted final lock.

A valid finalization must derive from this exact candidate, bind an independently generated
review receipt and review ID, and prove that the only intervening changes are the narrowly
authorized preregistration amendments. A name string is not independent verification.

### CAND2-4 — train/evaluate/decide capabilities trust self-authored receipts

The honest fold-0 projection is clean, but the fitter does not authenticate its origin. It trusts
`projection_sha256` from the adjacent caller-controlled metadata and does not verify metadata
schema, requested fold, source-manifest hash, held-query exclusion, or a final-lock-bound
projection identity. A projection containing a held association plus a matching self-authored
metadata file passes its checks.

The evaluator likewise checks a checkpoint only against a SHA supplied by the same fit receipt.
It does not validate receipt schema, fold, arm, seed, initialization, projection, final-lock chain,
commit, environment, or trainable contract. Worse, it opens the complete pair bundle—including
held outcomes—before attempting to deserialize the checkpoint. Arbitrary bytes plus a
self-consistent receipt therefore cross the held-outcome capability boundary.

The decision worker trusts evaluation rows without validating their schema, job identity,
checkpoint, catalog, tie rule, finiteness, query uniqueness, exact fold membership, or complete
361-query population. Self-authored reciprocal ranks can consequently be attributed to registered
arms and seeds.

The sealed outputs are also insufficient to reproduce the registered report. Evaluation stores
only destination rank and reciprocal rank, not the per-candidate scores needed to recompute exact
ranking or the registered nDCG, AUC, MSE/MAE, relation, and hardness diagnostics. Decision output
omits the required bootstrap mean/attempts, held-query count, unique-destination count, and
seed-specific results. Those omissions leave decision-bearing reporting choices after scoring.

### CAND2-5 — several claimed bindings are fields but not enforced bindings

The supplied verifier omits `environment`, `tokenizer_structure`, `frozen_reference`,
`augmentation`, `early_stopping`, and `cleanliness_scope` from `_verify_common`. Mutating all of
those fields is accepted. Unsandboxed WSL host execution sees the recorded GTX 1660 SUPER and
re-emits the candidate exactly; sandboxed verification sees no CUDA device and emits different
environment bytes, yet the supplied verifier still reports success. That cross-context acceptance
is direct evidence that the environment lock is descriptive rather than enforced.

The verifier copies initialization hashes from source constants without reading the checkpoint
files, and it verifies only the ranking manifest bytes—not the manifest-bound pair and fold files.
The candidate records only the number of trainable tensors and total parameters, not the required
exact 18 names and shapes. The fitter reopens title, projection, and checkpoint paths after lock
verification; the exact verified descriptors/bytes are not carried into execution.

### CAND2-6 — the frozen inference algorithm is not implemented

The prior review froze a versioned SHA-256 rejection sampler with exactly 82 block draws per
replicate and nearest-rank sorted endpoints at zero-based indices 249 and 9749. `cmd_decide`
instead uses NumPy `default_rng` and linearly interpolated `np.percentile` endpoints. Candidate
v2 binds only the seed, resample count, confidence, and minimum block count, omitting the sampler
schema, sampler ID, key fields, rejection rule, draw count, and endpoint rule.

The evaluator also does not reject nonfinite predictions. If the destination score is NaN, every
greater-than and equality comparison is false and the destination is silently assigned rank 1;
NaN competitor scores are silently ignored. This is a direct false-positive path and violates the
frozen fail-closed requirement.

### CAND2-7 — the private transaction still has incomplete rollback and directory guarantees

`install_private` correctly stages mode-`0600` data, fsyncs it, hard-links without replacement,
and checks the resulting target. It does not remove the installed target when parent fsync or
post-install verification fails, despite claiming rollback. It fsyncs the directory before
unlinking the staging name but not after cleanup, so the final one-link namespace state is not the
state made durable. Existing parent directories are not checked for mode, ownership, or symlink
substitution, and `read_bound` does not require private input mode.

These are fail-closed perimeter gaps, but they matter because the candidate claims the stronger
transaction as an authorization premise.

### CAND2-8 — decision-bearing paths have no executable tests

The new suite adds four unit tests. It never loads a real countersigned checkpoint, constructs the
optimizer, executes even one fit step, scores a candidate list, validates a fit/evaluation
receipt, exercises final-lock/receipt acceptance, checks the bootstrap known-answer sequence, or
tests the primary decision. This is why the incompatible loader, receipt substitutions, NaN rank,
and wrong bootstrap survived.

## 3. Required replacement candidate

Keep both preregistrations blocked while the engineering lane:

1. extracts the sampler into an ID-free stable module and removes hard-coded preregistration IDs
   from every candidate-bound validator;
2. fixes checkpoint construction and tests all three exact initialized bytes through model load,
   exact allowlist resolution, optimizer construction, and a deterministic one-step smoke;
3. freezes all five train-only projections before candidate generation and independently verifies
   their schema, fold, source manifest, row population, and zero held-query overlap;
4. emits a fresh training plan binding the exact 30 jobs, projections, code, artifacts,
   environment, exact trainable names/shapes, and complete inference contract;
5. implements a candidate-derived final-lock emitter and an independent review-receipt verifier
   that bind the candidate SHA, review ID, and a whitelist of permitted preregistration changes;
6. chains and verifies every fit and evaluation receipt before held access or decision use, with
   exact job/population checks and finite predictions;
7. implements the frozen SHA-256 block sampler and nearest-rank endpoints exactly;
8. completes rollback, second directory fsync, directory privacy, and descriptor/byte handoff
   guarantees;
9. adds known-answer and adversarial tests over the actual load/fit/evaluate/decide paths; and
10. emits a fresh plan and candidate in a new no-replace namespace from the clean landed commit.

That candidate then receives another independent review. Do not amend or “repair” candidate v2 in
place.

## 4. Conditions preserved for the later amendment

The frozen estimand remains exploratory and catalog-transductive: 361 queries, 359 candidates,
five folds, 82 unsplit lineage blocks, two paired arms, three seeds, exact-destination MRR,
`graded-negative-minus-positive-only`, `+0.010` minimum point gain, strict lower CI greater than
zero, ascending frozen catalog-column tie rule, and no secondary rescue. A pass authorizes only a
new reserve preregistration; it never authorizes opening the 1,481-row reserve.

The bootstrap remains the exact algorithm frozen in the prior review: average seeds within query,
form the paired contrast, draw 82 UTF-8-sorted blocks with replacement using the versioned
SHA-256 rejection sampler and identical multiplicities for both arms, retain the query-weighted
mean, use 9,999 replicates at seed `3997999`, and take nearest-rank endpoints 249 and 9749.

The later retention cascade remains narrow and still blocked. Only the source ranking protocol
hash/ID and verified negative-bundle status/hash may change. Track T must independently regrow its
pre-Pearltrees base under seeds `3998101`--`3998103`; ranking initializations and fitted
checkpoints remain forbidden there.

Until a replacement candidate passes review, step 4, the retention cascade, model fitting,
held-fold scoring, reserve access, and checkpoint release all remain unauthorized.
