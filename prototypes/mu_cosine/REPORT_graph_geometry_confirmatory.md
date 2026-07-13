# Graph geometry for residual covariance — theory, mechanism audit, and outcome-blind inventory

## Bottom line

The graph-geometry infrastructure is ready, but no real covariance or QR deployment is unlocked.

- PSD graph geometry is now structural: closed-neighborhood, sparse walk-feature, cumulative-walk, exact heat,
  and exact regularized-Laplacian references all produce unit-diagonal Gram matrices without eigenvalue repair.
- The original synthetic comparison failed honestly because common path amplitude `alpha` did not mean common
  covariance strength across kernels.  Its family-wise null control passed, but selected held likelihood was
  usually worse.
- A preregistered v2 matched maximum off-item coupling.  At `rho_max=0.20`, every non-deranged truth had positive
  held gain, 86.0--97.5% predictive-class recovery, and 97.0--98.5% wins over equal-energy topology controls.
  Block-null nonzero selection was 5.5%.  At `rho_max=0.10`, power was mixed and the heat case had essentially
  zero selected gain.
- The first outcome-blind campaign inventory rejected the original same-hop walk concatenation: it was nearly
  identity and did not make adjacent roots closer.  Cumulative walk fixes that while remaining a sparse PSD
  feature Gram.
- Nomic and MiniLM are different models from the e5-based operating pipeline, but their geometry is not
  statistically independent of e5.  Nomic is modestly less redundant and remains the primary external
  semantic candidate; MiniLM is a lower-cost sensitivity comparator.

The next scientific step is the repeated-judge campaign described below.  The current independent/block item
covariance stays in production.

## Theory implemented

### PSD comes before interpretability

An arbitrary graph distance is useful for sampling but is not automatically a covariance kernel.  In
particular, a Gaussian RBF over general graph shortest-path distances is not guaranteed PSD.  The accepted
constructions are:

```text
explicit Gram:       K = Phi Phi.T
heat reference:      K = exp(-t L)
resolvent reference: K = (I + tau L)^-1
convex mixture:      K = sum_g theta_g K_g, theta_g >= 0
interaction:         K = K_graph * K_embed       # Schur product
```

Every kernel is diagonal-normalized to correlation form.  Same-descendant gating is a one-hot Gram multiplied
elementwise by the root kernel, so it also preserves PSD.  The heat and regularized-Laplacian references follow
the graph spectral constructions of [Kondor and Lafferty](https://people.cs.uchicago.edu/~risi/papers/diffusion-kernels.pdf)
and [Smola and Kondor](https://people.cs.uchicago.edu/~risi/papers/SmolaKondor.pdf).

### The scalable graph candidate

The original same-hop feature map was

```text
concat_h sqrt(w_h) p_h(root).
```

It compares landing distributions at equal diffusion time.  That is valid and PSD, but it misses the
cross-hop overlap `p_0(a)` versus `p_1(b)` that represents a direct edge.  The accepted scalable candidate is

```text
Phi_cumulative(root) = sum_h sqrt(w_h) p_h(root),
K_cumulative = normalize(Phi_cumulative Phi_cumulative.T).
```

Sparse CSR propagation and direct sparse Gram accumulation avoid whole-graph diagonalization.  The original
same-hop kernel remains a negative/control geometry.

### Why one selected item kernel matters numerically

For one item kernel,

```text
R = I_n tensor B0 + K_theta tensor Bg,
K_theta = U Lambda U.T,
```

the transform `U tensor I_m` reduces `R` to independent channel blocks

```text
B0 + lambda_i Bg,  i=1,...,n.
```

At the proposed `n=128,m=32`, this suggests one 128-dimensional item eigendecomposition and 128 parallel
32-dimensional factorizations instead of a single unstructured 4096-dimensional factorization.  Multiple
noncommuting `K_g tensor B_g` terms generally lose this reduction, which is another reason not to promote a
large kernel zoo.  This is future optimization theory, not a CUDA performance claim; dense QR remains the
correctness reference.

## Embedding contract

The cache builder pins model revision, task prefix, title transformation, normalization, node order, array
shape/dtype, package versions, and content hashes.  Evaluation never downloads a model.

| candidate | exact revision | role |
|---|---|---|
| Nomic `nomic-embed-text-v1.5` | `e9b6763023c676ca8431644204f50c2b100d9aab` | primary external semantic geometry; `clustering:` title prefix |
| MiniLM `all-MiniLM-L6-v2` | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` | lower-cost sensitivity comparator |
| e5 | existing frozen caches | shared-input redundancy control, not an independent candidate |

Nomic's clustering prefix is deliberate: this experiment groups semantically similar items and is not a
query/document retrieval task.  Query/document prefixes are deferred because they inject directional
membership semantics closer to the quantities being judged.  Nomic's model family and open training design are
described in [Nussbaum et al.](https://arxiv.org/abs/2402.01613); MiniLM's distilled base architecture is
described in [Wang et al.](https://arxiv.org/abs/2002.10957).

Both pinned models loaded from local cache and produced deterministic, normalized campaign caches covering
2,681 unique endpoint titles.  Manifest SHA-256:

- MiniLM: `35fc09608d16dc2b5111686f587a57d5f60895611465b109de6898e864758569`;
- Nomic: `702f5f4af33c8994d25ad259c6a46ac7f7c762f4d6ef2a47f4335f9bc912e3d2`.

## Synthetic audit chronology

### V1 failed: common alpha was not matched cost

V1 calibrated its full candidate/alpha search with 1,000 block-null simulations, selected on 12 train fields,
and scored 64 independent held fields.  The block-null nonzero rate was 6.0%, but the frozen planted gates
failed.  For example:

| truth | alpha | selected held NLL gain/scalar | topology-control win rate |
|---|---:|---:|---:|
| closed | 0.10 | -0.001128 | 76.5% |
| closed | 0.20 | +0.001091 | 87.5% |
| walk same-hop | 0.10 | -0.001072 | 59.0% |
| walk same-hop | 0.20 | -0.001216 | 67.5% |
| heat | 0.20 | -0.000953 | 84.0% |
| resolvent | 0.20 | -0.001680 | 70.0% |

The reason was visible in outcome-blind kernel diagnostics: off-diagonal RMS ranged from 0.0720 walk to 0.3063
closed, while heat and resolvent were 0.999 correlated.  Common `alpha` neither matched covariance energy nor
identified exact family labels.  Field-count diagnostics through 96 reduced the threshold but did not repair
that comparison.  V1 is retained as failed evidence.

### V2 passed its matched-coupling strong-effect gates

V2 froze `rho_max=max_{i!=j}|C_ij|` as the common effect and grouped the outcome-blind
`{closed,heat,resolvent}` predictive class.  It used 48 train fields and the same 200 replicates / 1,000 null
calibrations / 64 held fields.

| truth | rho_max | nonzero selection | equivalence-class selection | truth beats derangement/base | selected held NLL gain/scalar |
|---|---:|---:|---:|---:|---:|
| closed | 0.10 | 60.5% | 52.0% | 83.5% | +0.000906 |
| closed | 0.20 | 98.5% | 97.5% | 98.5% | +0.014632 |
| walk same-hop | 0.10 | 40.0% | 33.5% | 81.0% | +0.000290 |
| walk same-hop | 0.20 | 90.5% | 86.0% | 97.0% | +0.008174 |
| heat | 0.10 | 48.0% | 42.0% | 82.0% | -0.000036 |
| heat | 0.20 | 94.5% | 93.5% | 97.5% | +0.008939 |
| resolvent | 0.10 | 45.5% | 43.0% | 78.0% | +0.000143 |
| resolvent | 0.20 | 93.0% | 92.0% | 98.5% | +0.009957 |

The v2 block-null nonzero rate was 5.5%.  All frozen `rho_max=0.20` gates passed.  The mixed 0.10 rows are the
most relevant warning for real data: accurate weak covariance can help in oracle sensitivity curves while a
finite-sample selector still fails to exploit it reliably.

Neither synthetic result is end-to-end power.  Both assume known zero mean, known scalar channel covariance,
one fixed 12-node topology, and many independent residual fields.

## Outcome-blind campaign geometry inventory

No judge score or residual entered this inventory.  Campaign rows supplied endpoint identities only.

| corpus | rows | within-descendant row pairs | direct-adjacent pairs |
|---|---:|---:|---:|
| exploratory | 1,000 | 764 | 84 |
| fresh | 770 | 777 | 206 |

### Distance rank correlations

| pair | exploratory | fresh |
|---|---:|---:|
| Nomic vs shared e5 | 0.776 | 0.838 |
| MiniLM vs shared e5 | 0.810 | 0.868 |
| MiniLM vs Nomic | 0.894 | 0.907 |
| closed graph vs Nomic | 0.481 | 0.396 |
| cumulative walk vs Nomic | 0.590 | 0.493 |
| closed graph vs cumulative walk | 0.730 | 0.885 |

Thus Nomic is more independent of e5 than MiniLM, but neither is an independent vote.  Graph geometry remains
the more distinct channel.  Cumulative walk makes adjacent roots closer (`0.857` vs `0.996` exploratory;
`0.770` vs `0.982` fresh), while the rejected same-hop walk did not.

The strongest disagreement quartiles between cumulative walk and Nomic contain only 13 exploratory and 22
fresh row-pairs (1.7% / 2.8%).  Existing data are too thin for a graph-by-embedding interaction claim.  A new
campaign must deliberately oversample those cells.

## Alternatives rejected or deferred

| alternative | disposition | reason / reconsideration condition |
|---|---|---|
| shortest-path Gaussian covariance | reject as direct kernel | not PSD on an arbitrary graph; retain distance for sampling unless negative type is proved |
| ordinary PPR/random-walk matrix | reject | asymmetric on irregular graphs; use a feature Gram or symmetric resolvent |
| concatenated same-hop walk as adjacency model | reject as primary | topology inventory showed near-identity and wrong adjacency ordering; retain as negative control |
| e5 as primary semantic geometry | reject | shared input to current mu readouts; retain as redundancy control |
| observed same-judge mu distance | reject | endogenous: the same noise enters geometry and residual |
| cross-fitted mu-hat distance | defer diagnostic | valid conditional covariate only with component-disjoint, preferably leave-one-judge-out construction |
| Nomic and MiniLM as independent votes | reject | their distance ranks correlate 0.89--0.91 |
| separate closed/heat/resolvent/cumulative selector votes | reject on current topology | nearly equivalent kernels inflate search multiplicity; reopen only if outcome-blind correlation drops on another graph |
| effective-resistance RBF | defer | mathematically promising but needs disconnected-component and global-solve infrastructure |
| unconstrained dense learned kernel | reject | low-sample overfit and PSD/identification risk |
| graph x embedding Schur product | defer secondary | PSD but can erase graph-local signal when the embedder misses domain-specific proximity |
| full-graph eigendecomposition/CUDA now | defer | statistical covariance is not validated yet |
| lower deployment confidence | reject | changes accepted error rate rather than adding information |

The append-only rationale and reconsideration conditions are in `DECISIONS_graph_geometry.md`.

## Next confirmatory campaign

1. Sample more than 400 independent endpoint components; use a full-procedure simulation to set the final count.
2. Within components, balance anchor, direct/local positive, and matched distant/hard-negative triples.
3. Obtain at least three independent calls per judge family and retain call identity.
4. Oversample graph/Nomic disagreement cells rather than relying on their 1.7--2.8% natural frequency.
5. Primary covariance candidates: block, closed-neighborhood, cumulative-walk, Nomic, and one nonnegative
   graph-plus-Nomic mixture.  MiniLM and e5 are sensitivity/redundancy controls, not extra selector votes.
6. Refit calibration, regional mean, marginal/channel covariance, geometry, and amplitude inside whole-component
   outer folds.
7. Gate on held joint residual NLL, posterior NLL/calibration/coverage, decision log-loss, margin AURC, topology
   derangement, loading, and the 95% simultaneous spectral envelope.
8. Only after those gates pass, compare dense QR with the single-kernel eigenmode conditioner and matched CUDA
   timing at `n=128,m=32`.

## Reproduction

```bash
python3 prototypes/mu_cosine/run_graph_geometry_synthetic.py \
  --out /tmp/graph_geometry_synthetic_final_v1_portable.json

python3 prototypes/mu_cosine/run_graph_geometry_synthetic_v2.py \
  --out /tmp/graph_geometry_synthetic_final_v2_portable.json

python3 prototypes/mu_cosine/independent_embedding_cache.py \
  --pairs-tsv /tmp/mu_data/campaign_scored.tsv --preset minilm \
  --out-prefix /tmp/campaign_minilm_geometry --device cpu

python3 prototypes/mu_cosine/independent_embedding_cache.py \
  --pairs-tsv /tmp/mu_data/campaign_scored.tsv --preset nomic \
  --out-prefix /tmp/campaign_nomic_geometry --device cpu

python3 prototypes/mu_cosine/run_graph_geometry_inventory.py \
  --artifact-repo /home/s243a/Projects/UnifyWeaver \
  --campaign /tmp/mu_data/campaign_scored.tsv \
  --minilm-prefix /tmp/campaign_minilm_geometry \
  --nomic-prefix /tmp/campaign_nomic_geometry --lmdb-no-lock \
  --out /tmp/graph_geometry_inventory_cumulative_final.json
```

Portable output SHA-256:

- v1 mechanism: `9cd6c0b6f6e7b4efdbf6e8afcff88607b871348fc3a8c212c421a8c8e5d2ee51`;
- v2 mechanism: `24333d08941ac2c356bb33fdd602a0009317de0daa6d0f7788e176c6bcc33969`;
- outcome-blind inventory: `81c35eaa8f9fa82afbbcf6a9040f4905f97140b4b91d7134ecfb69957b9fb9a1`.

The graph-geometry focused suite has 34 passing tests.  Including the inherited adjacent-residual,
structured-covariance, covariance-sensitivity, and NumPy/Torch square-root-conditioner suites gives
`157 passed, 9 skipped`.
