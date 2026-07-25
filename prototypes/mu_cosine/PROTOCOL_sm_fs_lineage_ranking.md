# Prospective protocol: SM-FS LINEAGE negative construction and ranking

**Status:** frozen after the positive-only onboarding run and before constructing any negative
ledger, fitting a negative-trained checkpoint, or scoring a ranking metric. This is an
**exploratory, catalog-transductive mechanism study** over the 361 certified exploration maps. It
does not open or score the 1,481 reserved maps. A material change requires a new protocol version
and preregistration ID before an affected score is computed.

The positive-only run remains useful, but its evidence is narrower than initially reported. The
trained checkpoint reproduces a validation correlation near `+0.82`; the exact currently available
warm start instead reproduces `-0.112`, not the reported `+0.37`, and the run receipt did not hash
or evaluate its initialization. Moreover, 356 of 384 validation rows reuse an ancestor path seen
in training and an ancestor-mean lookup reaches correlation `+0.630`. The result therefore shows
shared-hierarchy interpolation on unseen maps. It is not a ranking, unseen-lineage, or transfer
result and supplies no decision-bearing observation for this protocol.

## 1. Decision and estimand

The question is whether topology-derived non-ancestor supervision improves exact-destination
ranking relative to the same model trained on positive ancestors only.

For held exploration map \(q\) and training seed \(s\), let
\(RR_{\mathrm{neg},s}(q)\) and \(RR_{\mathrm{pos},s}(q)\) be reciprocal ranks of the recorded
destination under identical 359-candidate lists. The paired value is

\[
d(q)=\frac{1}{3}\sum_s
\left[RR_{\mathrm{neg},s}(q)-RR_{\mathrm{pos},s}(q)\right],
\qquad
\Delta_{\mathrm{MRR}}=\operatorname{mean}_{q}d(q).
\]

The graded-negative mechanism passes this exploratory gate only when:

1. \(\Delta_{\mathrm{MRR}}\ge 0.010\); and
2. the lower endpoint of the frozen 95% paired adaptive-lineage-block bootstrap is greater than
   zero.

A passing result authorizes writing a new, prospective protocol for the untouched reserve. It
does not itself authorize reserve scoring, a Pearltrees transfer claim, deployment, or publication
of a user-specific checkpoint. Every other outcome is reported as practically positive but
inconclusive, statistically positive but below the floor, inconclusive, or negative.

The exact destination path is the sole primary relevant item. Duplicate leaf titles remain
distinct identities. Best-title-equivalent grading is a sensitivity only.

## 2. Source and privacy perimeter

Construction first verifies the exact local SM-FS v3 bundle:

| binding | SHA-256 / value |
|---|---|
| manifest file | `466adfcf2e7c5914dad27b548c3f009804fec12e633498e187b0b4df71a98d61` |
| internal manifest ID | `e36164b7fef0569acfaeafb667628062cf37ced7ad04df0f34543c899086efe2` |
| ledger | `b45ff8ded88f2e5d5ded78664b2d381a017a5722464f956f6b44d660f5060b40` |
| positive target projection | `c3b298d5335ee901111f4985bbf5f7c5feb017c503a8c81db60dba1b947ac051` |
| privacy policy | `public-only` |
| process | `lineage(fs,decay=0.85)` |
| E5 revision | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |

The trusted integrity verifier may read the full bundle to authenticate it. It then passes a
descriptor-bound copy of only `lineage_fs_targets.tsv`, a redacted receipt, and this specification
to a pure constructor. The redacted receipt may contain opaque source hashes and the already
reported aggregate split counts, but no reserve or cross-lineage-excluded row identity, title,
destination, association, or placement-derived distribution. The constructor and every training
or scoring worker are denied the source ledger.

The candidate catalog is **not** `ledger["catalog"]`: that 3,133-path catalog was derived partly
from held placements. The only eligible catalog is the sorted union of exact ancestor paths in the
2,792 exploration target rows. It must reproduce exactly 359 paths.

No provider, LLM, embedding model, source `.smmx`, network service, placement frequency, or
reserved record participates in negative construction. “Public-only” is an owner-policy
classification, not permission to publish a derivative model. Row-level artifacts, paths,
embeddings, predictions, and checkpoints remain local.

## 3. Exhaustive candidate and target construction

For each of the 361 exploration maps \(q\), rank the identical UTF-8-byte-sorted 359-path catalog.
Let \(d_q\) be the map's recorded destination and \(A_q\) its complete root-to-destination ancestor
set in the verified target projection.

- `positive_parent`: \(c=d_q\), target \(1\);
- `positive_ancestor`: \(c\in A_q\setminus\{d_q\}\), target conceptually
  \(0.85^{\operatorname{hop}(q,c)-1}\);
- `structural_nonancestor`: every \(c\notin A_q\).

The canonical positive value is the six-decimal ASCII target already certified in
`lineage_fs_targets.tsv`, parsed once as IEEE-754 binary64 and recorded with its source bytes and
`float.hex()`. It is not recomputed at greater precision.

Non-ancestors are structural alternatives, not assertions that a folder is semantically
unsuitable. Their primary target retains the existing graph-judge **functional form**, but freezes
the SM-FS process decay `0.85` rather than the earlier prototype's default `0.6`:

\[
\mu_{\mathrm{graph}}(q,c)=
\max\left(
0.02,\;
0.85^{h(d_q,c)}
\frac{\operatorname{depth}(\operatorname{LCA}(d_q,c))}
     {\operatorname{depth}(d_q)}
\right).
\]

Different roots use the registered unreachable distance and an LCA fraction of zero, so their
target is the `0.02` floor. Compute each new value first as the exact reduced rational
`max(1/50,(17/20)^h * lca_n/lca_d)`, then convert it once to correctly rounded binary64. The
constructor records the exact hop count or unreachable marker, reduced target and LCA fractions,
and target `float.hex()`; the verifier independently recomputes all of them.

Every nonancestor pair also receives two orthogonal, model-score-blind, placement-derived
structural labels.
Treat paths as component sequences, let \(\ell\) be their longest-common-prefix length, and let
\(h=|d_q|+|c|-2\ell\):

- `descendant` when \(d_q\) is a strict prefix of \(c\);
- `sibling` when \(d_q\ne c\), both have the same immediate parent, and equal depth;
- `near_branch` when \(\ell>0\), neither prior relation holds, and \(h\le4\);
- `same_root_far` when \(\ell>0\), neither prior relation holds, and \(h\ge5\);
- `cross_root` when \(\ell=0\);
- hardness: `hard` for reachable distance 1--2, `medium` for 3--4, and `easy` for distance at
  least 5 or a cross-root pair.

The pre-score inventory is frozen as a construction gate:

| record class | count |
|---|---:|
| queries | 361 |
| candidates | 359 |
| exhaustive pairs | 129,599 |
| positive rows | 2,792 |
| structural non-ancestors | 126,807 |
| hard non-ancestors | 1,819 |
| medium non-ancestors | 7,814 |
| easy non-ancestors | 117,174 |

These counts were computed before any negative-trained model score existed. A mismatch blocks the
run rather than silently changing the population.

### Objective mass and exact minibatch sampler

Every query has equal total supervised mass. Positive ancestors receive total mass `1/2`, uniform
within the query. Non-ancestors receive total mass `1/2`. Within the non-ancestor half, nonempty
hard, medium, and easy buckets receive relative mass `3:2:1`; the ratio is renormalized across the
nonempty buckets, and each bucket's mass is uniform across its members. An empty family remains
empty and is recorded; its mass is never reassigned by selecting extra candidates.

The stored weights define the graded arm's target sampling distribution; they are not multiplied
into the loss a second time. At every one of 800 steps, derive exactly 24 training-query draws with
replacement, uniformly over the canonical UTF-8-byte-sorted training-query list. Use sampler ID
`sm-fs-ranking-sampler-v1`. Its key is canonical JSON schema
`unifyweaver.sm-fs-ranking-sampler-key.v1` with exactly the sorted fields `bucket`, `draw`, `fold`,
`query_id`, `retry`, `role`, `sampler_id`, `schema`, `seed`, and `step`, compact separators,
standard JSON UTF-8 escaping, and one terminal LF (the repository's `canonical_json_bytes`).
`fold`, `step`, `draw`, and `retry` are zero-based nonnegative JSON integers; `seed` is the frozen
nonnegative training seed. Unused `query_id` and `bucket` fields are the empty string.

Hash that exact byte string with SHA-256. Domain-separate draws with role `query`,
`common-positive`, `contrast-positive`, `negative-bucket`, or `negative-candidate`. For every
role after `query`, bind the selected exact query ID; `negative-candidate` additionally binds its
bucket. Each query draw creates two batch slots:

1. a positive slot, uniform over that query's ancestors and byte-identical across both arms;
2. a contrast slot: `positive_only` draws another independently keyed uniform ancestor, while
   `graded_negative` chooses a nonempty hardness bucket with renormalized `3:2:1` probability and
   then a uniform candidate within that bucket.

Thus each arm performs 48 row evaluations and one optimizer update per step. The arms pair the
query schedule, first positive slot, optimizer initialization, augmentation stream, and compute
budget; their contrast-slot examples differ by design. Each batch loss is the arithmetic mean of
its 48 squared errors. The agnostic anchor is computed **only** from the 24 common-positive slots,
with the same three-field rows, frozen-reference outputs, and no training augmentation in both
arms; contrast slots never enter the anchor. Add its identical mean squared error with weight
`1.0`. Sampling is with replacement across and within steps. Query, ancestor, and candidate lists
use canonical UTF-8 byte order. Eligible negative buckets are traversed in fixed
`hard,medium,easy` order with integer weights `3,2,1`: draw an integer on `[0,sum(weights))` and
select the first cumulative interval containing it; absent buckets contribute neither an interval
nor weight. The counter sampler interprets SHA-256 as an unsigned big-endian 256-bit integer,
increments only the zero-based `retry` field under the same domain-separated key, and rejects
values at or above
`floor(2^256 / n) * n` before taking modulo \(n\), so index selection is exactly specified.

The machine-readable preregistration contains four known-answer vectors spanning query,
positive, bucket, and negative-candidate draws. An implementation must reproduce their canonical
preimage digest, accepted retry, and selected index before constructing a schedule.

The primary graded targets above are not interchangeable with binary zero targets. A zero-target
arm is a prespecified sensitivity and cannot replace the primary after results are visible.

## 4. Frozen folds, arms, and optimization

Use five deterministic, model-score-blind folds grouped by adaptive destination-lineage blocks.
The split is allowed to use the exploration destinations because they are training-corpus
structure; it is not outcome-blind in the stronger prospective sense. Begin with exact depth-3
destination prefixes; recursively deepen a block containing more than
`max(1,floor(0.08 * 361))` maps using the same rule as `sm_fs_freeze.derive_split`, and never
divide an unresolved exact-destination block. This must reproduce cap `28`, 14 recursively
deepened prefixes, 82 final blocks, and block sizes from 1 through 28.

For assignment, compute SHA-256 over the UTF-8 bytes of
`sm-fs-lineage-ranking-fold-v1`, followed by one NUL byte, followed by `block_utf8`. Sort blocks by descending map count,
then digest, then UTF-8 block bytes, and greedily put each whole block into the fold with the
smallest current map count (tie: lowest fold number 0--4). The frozen assignment has map counts
`73/72/72/72/72`, block counts `16/16/16/17/17`, and canonical sorted
`block<TAB>fold<TAB>count` SHA-256
`b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37`
(UTF-8 lines joined by one LF, with no terminal LF).
Persist and verify this mapping before fitting.

This removes direct map and near-lineage overlap, but high ancestors and the global exploration
catalog remain shared. The evidence class is therefore
**map/near-lineage-blocked, catalog-transductive**, not fully lineage-independent.

For every fold and seed:

- training receives no held query-to-destination association or held query rows; the shared
  transductive catalog necessarily exposes held destination identities as candidates;
- the checkpoint and full run receipt are sealed before the scorer receives held rows;
- steps = `800`, batch size = `48`, learning rate = `0.0005`, anchor weight = `1.0`, gradient
  clipping norm = `1.0`, and no early stopping;
- query order, common positive slots, augmentation draws, initialized bytes, and budget are paired
  across arms according to the sampler above;
- seeds are `3997001`, `3997002`, and `3997003`.

The decision-bearing arms are:

1. `positive_only`: positive ancestors, retrained inside every fold under the frozen budget;
2. `graded_negative`: the same positives plus exhaustive graded non-ancestors.

The unchanged warm start is a descriptive reference. `binary_zero_negative` is a prespecified
sensitivity. A listwise `LINEAGE_RANK`/softmax model would be a separate future arm and may not be
introduced as an implementation detail of the graded-regression contrast.

This version authorizes the private constructor and training-plan implementation, but **not model
fitting**. Before fitting, amend and reseal the preregistration with the exact initialized
checkpoint and post-growth hash, loading/growth code hash, tokenizer/text-table hash, optimizer
class and options, trainable parameter names/counts, frozen-reference construction, augmentation
algorithm, numeric precision, deterministic environment, and the complete training-plan hash.

Exact paths are opaque identities. Embedding text is limited to the certified map title and
candidate leaf title. Path components, destination ancestors, fold labels, and private or reserve
text may not enter the text encoder.

## 5. Evaluation and inference

Score all 359 candidates for each held map. Exact-score ties break by ascending frozen catalog
column. Average the three seed-specific reciprocal ranks per query before taking the paired
contrast.

The primary 95% interval uses a paired adaptive-lineage-block bootstrap with 9,999 resamples, seed
`3997999`, and confidence `0.95`. Resample whole frozen lineage blocks and apply identical weights
to both arms. Fewer than 20 blocks makes the interval descriptive and unable to pass the gate.
Report the point estimate, percentile interval, bootstrap mean and attempts, held queries, lineage
blocks, unique exact destinations, and every seed-specific result.

This is a fixed-fit, conditional interval over the frozen exploration corpus and the five fitted
fold models. It does not estimate uncertainty from drawing a new corpus, choosing a different
lineage split, fitting another seed population, or the earlier privacy/freezer decisions. The
`0.010` practical floor follows the filing-ranker protocol's split-stability convention; it is not
estimated from this held ranking result.

Prespecified secondary outputs, all descriptive, are:

- exact-path R@1, R@5, and R@10;
- nDCG against the complete graded graph target;
- ancestor-versus-nonancestor AUC and weighted MSE/MAE;
- relation and hardness slices;
- best-title-equivalent sensitivity;
- unchanged warm-start and frozen e5 title-cosine controls; and
- the binary-zero-negative sensitivity.

No subgroup, alternate grade, or secondary arm rescues a failed primary result.

## 6. Frozen local artifact contract

The constructor emits schema `unifyweaver.sm-fs-lineage-ranking-bundle.v1`:

- `source_receipt.json`;
- `spec.json`;
- `catalog.jsonl`;
- `queries.jsonl`;
- `pairs.jsonl`;
- `folds.jsonl`; and
- `manifest.json`.

Every pair binds full query/map identity, exact candidate path, relation, hardness, graph distance,
LCA fraction, exact target, query-normalized loss weight, fold, lineage block, source-target hash,
and specification hash. The verifier rederives every byte; a copied stored field is not evidence.

Install through a private staging directory using no-replace atomic rename, mode `0700` for the
directory and `0600` for regular single-link files. The fitter consumes descriptor-bound verified
bytes rather than reopening named files. Checkpoints, prediction tables, and run receipts use the
same private, crash-atomic, no-replace discipline and record full—not shortened—hashes for source,
initialized checkpoint, output checkpoint, code, text table, fold, and environment.

## 7. Fail-closed and authorization rules

Block before fitting unless:

- the exact certified v3 bundle and owner exclusions reproduce;
- only the 2,792 exploration target rows reach the constructor;
- the counts in section 3 reproduce exactly;
- each query has exactly one target-1 parent and at least one non-ancestor;
- every candidate is supported by an exploration positive path;
- no non-ancestor is a true ancestor and no duplicate pair or candidate exists;
- all targets are finite and in `[0.02, 1]`;
- each query's positive and negative weights independently sum to `1/2`;
- UTF-8 ordering, folds, blocks, targets, weights, and bytes rederive identically;
- no reserve, cross-lineage-excluded, private, unknown, or owner-excluded content appears; and
- no network or model call occurs during construction.

The existing positive-only checkpoint and observed `+0.82` run are ineligible for the paired
decision. The trusted source verifier may authenticate aggregate reserve membership, but no
reserve row evidence reaches construction, fitting, scoring, or decision workers, and no reserve
map is scored. A passing exploration result authorizes only drafting and freezing a separate
reserve transaction.

## 8. Minimum implementation tests

Before execution, tests must cover:

- byte-identical reproduction under input and mapping-order changes;
- brute-force catalog and pair completeness;
- target, LCA, relation, hardness, and mass recomputation;
- duplicate titles preserving distinct exact paths;
- reserve sentinels absent from every derived artifact;
- stable block/fold assignment with no block split;
- tamper-and-reseal rejection for every artifact;
- wrong bundle, privacy policy, code, E5 revision, or specification rejection;
- atomic no-replace behavior, permissions, symlink/hardlink rejection, and races;
- loader outputting the exact seven-field model tuple;
- held-fold bytes unavailable to the fitter before checkpoint sealing; and
- no path string entering embedding text.

An honest blocked, null, or negative outcome completes this exploratory mechanism study.
