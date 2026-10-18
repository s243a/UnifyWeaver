# Rigor review: SM-FS lineage-ranking execution proposal

**Status:** the warm-start choice in PR #4010 is countersigned. The proposed execution lock is
not a fitting authorization and the ranking preregistration remains blocked. This review was
completed before any ranking optimizer step, held-fold score, or reserve access.

This document reviews local proposal
`~/mu_data/sm_fs_ranking_v1/execution_lock.json`, SHA-256
`0752cc16256a5699924e150d665dff3a556d0f92c3739e652b9892de97d4e47e`, against
`PROTOCOL_sm_fs_lineage_ranking.md` and preregistration
`0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2`.
The accompanying JSON review is the machine-checkable authority. Paths in the local proposal are
not publication permissions.

## 1. Countersigned warm start

Use the migrated pre-Pearltrees checkpoint `model_prod_namecond_full.pt`, SHA-256
`9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef`, as the common
ranking initialization. This is the correct choice for the frozen estimand:

- it precedes Pearltrees and SM-FS outcome training;
- within a seed, both decision arms can start from the exact same expanded checkpoint bytes;
- the comparison remains the fixed-budget effect of replacing the second positive slot with a
  graded structural alternative, rather than adding negative exposure to one arm; and
- it does not import the positive-only pilot or a Pearltrees-trained checkpoint into the
  decision-bearing contrast.

The result will be conditional on this checkpoint and the three frozen seeds. It will not
establish transfer or initialization-robustness. Do not add a Pearltrees-trained warm-start
decision arm: that would answer a different and contaminated question.

The expansion is more than “LINEAGE growth.” It adds the `LINEAGE` and `LINEAGE_RANK` operator
rows, expands the current judge table, refreshes name-card embeddings, and creates fresh
operator-readout rows. The exact initialized checkpoint bytes, not a future re-execution of this
growth process, are the authoritative training inputs:

| seed | initialized checkpoint SHA-256 | canonical state SHA-256 |
|---:|---|---|
| 3997001 | `a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e` | `0d842ed6eeb8e69036605e50697aa2cb3a5f8955255deb7becfbfa1c8b3154ad` |
| 3997002 | `fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796` | `f3eae318f41f17332f5bdc589958dd09d80d411afab58cca0fe63bbb7c90cdd3` |
| 3997003 | `f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a` | `e9cedab4e2df1405c929e9c42af5adc439c37117ba62ef5e2b43e29b40cac8ce` |

Their canonical configuration SHA-256 is
`01a1236ab7debd57b42818210e7e4fa2d711244ea67f408cd02472417374e98d`.
The state digest iterates state-dictionary keys in ascending Unicode order and feeds, for each
tensor, `key UTF-8`, NUL, `str(dtype)`, NUL, `str(tuple(shape))`, NUL, and contiguous CPU NumPy
C-order bytes into SHA-256. The configuration digest is SHA-256 over sorted-key compact JSON with
one terminal LF.
For every seed, reload that seed's one checkpoint byte-for-byte before each of the five folds and
both arms. Never continue one fold or arm from another.

Track T remains a separate experiment. It must independently expand the same pre-Pearltrees base
under its own `3998101`--`3998103` seeds, behavior panel, execution lock, and output namespace.
Ranking initializations and fitted checkpoints may not enter Track T.

## 2. Verified proposal contents

The negative bundle is reserve-free and reproduces its frozen inventory:

| binding | SHA-256 / value |
|---|---|
| ranking manifest | `e01e1e48b5464bd315cff3c982e035f390cab8ba2b4c3ee60322dac65bf35894` |
| exhaustive pairs | `a0f3e30ce091567516db3dde2cdf6025ba44f526a7588fbd6d01125d448c9b26` |
| fold file bytes | `a03e2ef7d584f58adc1b4f6b1abb6d042c7d0cfbff1d7ca18b5d132f81703c8e` |
| logical fold assignment | `b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37` |
| constructor file | `40ed4212f218817166df305b78280966a6af9561d5e3a12440d40af9a2ff67bc` |
| constructor commit | `b8b969ac1` (merged by `e18076a9b`) |
| queries / candidates / pairs | 361 / 359 / 129,599 |
| positives / nonancestors | 2,792 / 126,807 |
| hard / medium / easy | 1,819 / 7,814 / 117,174 |

The 720-entry title table contains exactly 361 certified map titles and 359 certified candidate
leaf titles. Exact paths are opaque identities and do not become encoder text:

| binding | SHA-256 / value |
|---|---|
| ordered names | `9c81701c10e00215f1ff6e734bd1910dfe89ee18fa903959b4dce8c6226cde42` |
| ordered identity-to-title table | `60c8b3bfb7a3739b1e057a407603ec1d9138d5e94f17c5621f28413618763391` |
| E5 cache | `bb9342a06cc9c62eedd664bb88c76833829f35e14e464beac0661feba81ed23f` |
| E5 revision | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |

The proposed optimizer (`torch.optim.Adam`, learning rate `0.0005`, betas `[0.9,0.999]`,
epsilon `1e-8`, no weight decay) and the unique 18-tensor, 1,195,782-parameter inventory match
the intended onboarding recipe. These are reviewed proposal fields, not enforcement: a final
trainer must freeze every parameter, enable exactly the bound allowlist, assert its
names/shapes/count, and only then construct the optimizer.

## 3. Why fitting remains blocked

### 3.1 No ranking trainer or final training plan

No repository program consumes `sm_fs_ranking_v1/pairs.jsonl` to run the registered
five-fold-by-two-arm-by-three-seed transaction. `fine_tune_sm_fs.py` remains the earlier
positive-only diagnostic: it chooses a new random validation split, does not consume the graded
bundle, and does not use the frozen sampler. The proposal's `training_plan_code` hashes the
emitter, which performs no optimizer step.

The final plan must bind and enforce:

- exact one-initialization-per-seed reload before every fold and arm;
- the complete fold/arm/seed job ledger and common two-slot sampler schedule;
- held-fold byte denial until the corresponding checkpoint is sealed;
- exact tokenizer structure, augmentation parameters, and NumPy/Torch RNG reset and consumption
  order;
- the 18-tensor trainable allowlist, frozen reference, Adam options, step budget, clipping, and no
  early stopping;
- deterministic runtime settings as observed assertions, not desired booleans;
- output checkpoint, prediction, scoring, bootstrap, and receipt schemas; and
- private crash-atomic no-replace installation and verification.

### 3.2 Proposal provenance is not a clean landed execution snapshot

The lock records Git commit `e18076a9b`, which predates and does not contain
`emit_ranking_execution_lock.py`. The emitter itself landed as `f4a172a6e` and was merged by
`01da72ba3`. Its file hash is
`97b49380dec4c041c25cd05a3efdf3f4a910f7975dca7899c3701e01de7899f2`.
A final lock must be generated from a clean landed commit containing the emitter, trainer,
evaluator, verifier, and complete plan.

Growth provenance is also incomplete. It transitively uses `fine_tune_channel_heads.py`,
`judge_cards.py`, and local judge/operator card caches, none of which the proposal records.
Because the initialized checkpoint bytes are now authoritative, this does not invalidate them;
the final receipt must nevertheless distinguish authoritative initialized bytes from descriptive
growth provenance.

### 3.3 Privacy and transaction contract is violated

`ranking_text_e5.pt` is mode `0644`, while the protocol requires every local ranking artifact to
be `0600`. The emitter and constructor use in-place truncating writes. They do not install an
immutable run-input tree through private staging, single-link/non-symlink checks, fsync, and
no-replace rename. A rerun can therefore replace a proposed frozen input.

Regenerate or copy verified inputs into a new mode-`0700` run directory using a crash-atomic,
no-replace transaction. Every regular input must be mode `0600`, single-link, non-symlink, and
content-verified after installation.

### 3.4 Environment and source isolation are promises, not enforcement

The proposal writes desired values for cuDNN determinism, TF32, precision, and dtype but does not
set or assert them. It omits deterministic-algorithm mode, cuBLAS workspace configuration,
thread counts, and relevant runtime/library versions. The final runner must set these before
model construction and fail closed if observed state differs.

The emitter also reopens the live SM-FS target path without authenticating its frozen hash or
receiving descriptor-bound verified bytes. Current outputs reproduce and are clean; the final
transaction must use verified, descriptor-bound inputs so a later rerun cannot mix a stale
ranking bundle with a changed source projection.

## 4. Exact next handoff

The final authorization sequence is deliberately two-stage so the execution lock cannot become
stale when the preregistration changes:

1. land the ranking trainer/coordinator/evaluator, complete training plan, and candidate-lock
   verifier;
2. generate a candidate proposal from that clean landed commit without reading any score;
3. independently review the candidate proposal;
4. land the authorized ranking-protocol amendment, new preregistration ID, and cycle-free
   verifier binding, then cascade the new ranking hash and ID into the still-blocked
   retention/transfer preregistration;
5. regenerate the **final** execution lock from that exact clean authorized commit, binding the
   new preregistration ID;
6. independently verify the final lock against the landed preregistration and code; and
7. only then permit the first optimizer step.

If code validates the preregistration ID, it should derive and check the ID from the document
rather than hard-code an ID into a file whose own hash participates in the plan. This avoids a
hash/ID cycle.

Until step 6 passes, construction remains valid, the pre-Pearltrees warm start is fixed, and model
fitting remains forbidden.
