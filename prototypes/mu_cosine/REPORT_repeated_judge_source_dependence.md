# Repeated-judge source dependence — topology bridge report

## Bottom line

The hard three-hop-core design failed because it discarded too much of both graphs.  The dependence-aware
replacement is structurally feasible: using the complete exclusive source regions, every frozen
`K in {64,96,128}` passes the optimistic four-endpoint capacity, cap-constrained allocation, minimum-region,
and PSD checks in both corpora at every registered `G in {160,320,512,800}`.

That is a **structural bridge, not a power result or covariance result**.  Region-average graph exposure is a
stipulated sensitivity geometry, not measured judge-residual correlation and not a worst-case candidate-level
bound.  No `K` is selected.  The audit reads no history, candidates, embeddings, labels, or judge responses,
and every authorization remains false.  Its matrices are fixed inputs to the next source-dependent
full-procedure null/power simulation.

The information diagnostics show why this explicit step matters.  At nominal `G=800` and stipulated
`rho=.20`, the prospective allocations have only 103--235 effective components in exploratory and 225--365 in
fresh.  Allocation-free guarded floors are lower still, 19--25 and 33--37.  Treating all 800 components as
independent would therefore be anti-conservative under this sensitivity model even though most cross-region
exposures are small.

## Frozen construction

The audit reuses the deterministic induced-connected source regions from the failed core audit but sets the
halo radius to zero: every graph node remains in exactly one region, and every region remains inside one true
weak component.  Source region is a concentration, fold, and sensitivity unit; it is not renamed a connected
component and does not assert independence.

For each region `r`, start a random walk uniformly over its full node set.  With the already frozen weights
`w=(1,.5,.25,.125)`, form

```text
Z_r = sum_h sqrt(w_h) U_r P^h,
E = normalize_rows(Z) normalize_rows(Z).T.
```

This computes only sparse `K x V` landing distributions and a dense `K x K` Gram; it never creates a
`V x V` node kernel.  `E` is PSD and unit diagonal by construction.  The audit stores the complete matrix,
spectrum, off-diagonal quantiles, and hop-specific outside-region landing mass.  The latter is not first-exit
mass because a walk can leave and return.

For each `(corpus,K,G)`, a prospective greedy quota adds the next component to the eligible region minimizing
the exact increment `2(E n)_r+E_rr`, with stable region ID as the only tie-break.  Region capacity is

```text
min(floor(region_nodes/4), floor(.10 G)).
```

The quota is an outcome-blind design target, not proof that the later four-endpoint, 32-cell, history-disjoint
candidate packing exists.  With one-hot component-to-region incidence `H`, the stipulated source path is

```text
C_rho = (1-rho) I_G + rho H E H.T,
rho in {0,.025,.05,.10,.20}.
```

Distinct components assigned to the same source region have correlation `rho`; a cross-region pair has
correlation `rho E_rs`.  For the equal-component scalar mean,

```text
G_eff = G^2 / [(1-rho)G + rho n.T E n].
```

This is exact for the stipulated separable path.  It is an information diagnostic, not the power of the
repeated-judge selector.  The audit also gives a numerically guarded allocation-free lower ESS by upper-
bounding `n.T E n` with the tighter of a nonnegative-row-sum/capacity-squared bound and a nonnegative
row-capacity bound.

## Frozen inputs and reproducibility

| corpus | graph | nodes | undirected edges | content identity |
|---|---|---:|---:|---|
| exploratory | SimpleWiki `100k_cats/category_parent.tsv` | 84,136 | 196,876 | `4881beed...5dc8ec` |
| fresh | enwiki `Behavior` LMDB slice | 75,901 | 99,971 | `3bcfe59a...a5d690` |

The fresh loader continues to bind the exploratory graph used for title exclusion.  LMDB `lock.mdb` remains
excluded runtime state.  The runner hashes the design, preregistration, implementation, graph inputs, and
loader dependencies at startup; it reloads and compares graph content records and loader metadata after the
audit, then rechecks the scientific files before output.  The path-free Python/NumPy/BLAS identity is recorded,
and the CLI requests one BLAS thread through `threadpoolctl` when available.  Paths and elapsed time are absent
from the JSON.

Two final computations produced the same complete 2,767,735-byte payload record, SHA-256
`bf9a09c35e54bd36c2e7efea19c432ccf1e9105ff67c4154cfc1c6e744a843b2`.  The tracked review projection is
308,090 bytes, SHA-256 `89fe1e33568badc39c4a68eb419bceac40cdb4f06714f488b556b32879e0ec20`;
it retains all decisions and scalar diagnostics plus content records for omitted exact matrices and quotas.

## Full-region capacity

The table is the optimistic number of four-endpoint components available after only region size and the 10%
cap.  Every entry exceeds its requested `G`; exact candidate packability remains untested.

| corpus | `K` | `U4(160)` | `U4(320)` | `U4(512)` | `U4(800)` | regions used at `G=800` |
|---|---:|---:|---:|---:|---:|---:|
| exploratory | 64 | 538 | 1,053 | 1,645 | 2,534 | 40 |
| exploratory | 96 | 1,050 | 2,077 | 3,283 | 5,037 | 72 |
| exploratory | 128 | 1,546 | 3,050 | 4,789 | 7,219 | 104 |
| fresh | 64 | 1,024 | 2,048 | 3,261 | 5,088 | 64 |
| fresh | 96 | 1,536 | 3,069 | 4,850 | 7,517 | 96 |
| fresh | 128 | 2,048 | 4,068 | 6,379 | 9,757 | 128 |

The exploratory graph contains small weak components and some regions with fewer than four nodes, so not all
nominal regions receive a component.  The used counts still exceed the frozen minimum of 20.  The fresh graph
uses every region.

## Exposure geometry

| corpus | `K` | numerical rank | effective rank | largest eigenvalue | off-diagonal 95th percentile | maximum | mean outside landing hop 1 / 2 / 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exploratory | 64 | 64 | 62.38 | 1.661 | .0244 | .535 | 17.89% / 18.99% / 25.17% |
| exploratory | 96 | 96 | 91.31 | 2.077 | .0230 | .774 | 25.31% / 27.38% / 36.12% |
| exploratory | 128 | 128 | 123.10 | 1.934 | .0207 | .584 | 28.63% / 32.25% / 41.87% |
| fresh | 64 | 64 | 63.79 | 1.251 | .0087 | .246 | 6.54% / 9.55% / 12.84% |
| fresh | 96 | 96 | 95.55 | 1.366 | .0063 | .362 | 6.63% / 10.00% / 13.22% |
| fresh | 128 | 128 | 127.48 | 1.370 | .0046 | .364 | 6.73% / 10.08% / 13.44% |

The matrices are full rank and have effective rank near `K`, so the average region landing profiles are mostly
distinct.  This does **not** mean cross-region dependence is absent.  The high maxima show a small number of
strongly exposed neighboring region pairs, while the low 95th percentiles show that most region pairs barely
overlap.  Exploratory loses much more random-walk mass across its cuts than fresh, consistent with the earlier
core failure.

Average-region exposure can understate a candidate set concentrated near cut boundaries.  The later builder
must recompute the lifted exposure for the realized candidate incidence; it cannot inherit these averages as
a certified upper bound.

## Effective-information sensitivity

The table gives the exact prospective-quota `G_eff` at the weakest nonzero and strongest frozen source
couplings.  Parentheses contain the allocation-free numerically guarded floor at `rho=.20`.

| corpus | `K` | `G=160`, `.025` / `.20` | `G=320`, `.025` / `.20` | `G=512`, `.025` / `.20` | `G=800`, `.025` / `.20` (floor) |
|---|---:|---:|---:|---:|---:|
| exploratory | 64 | 141.1 / 77.3 | 244.7 / 92.5 | 337.1 / 99.4 | 434.1 / 103.3 (24.7) |
| exploratory | 96 | 150.3 / 105.4 | 276.0 / 140.7 | 401.6 / 160.1 | 552.1 / 174.2 (21.2) |
| exploratory | 128 | 154.2 / 123.1 | 290.4 / 176.4 | 433.6 / 209.4 | 615.4 / 235.3 (19.0) |
| fresh | 64 | 153.0 / 117.3 | 287.4 / 167.9 | 428.1 / 199.4 | 606.1 / 224.7 (36.5) |
| fresh | 96 | 156.3 / 134.7 | 299.6 / 207.1 | 456.0 / 258.2 | 663.6 / 302.6 (33.8) |
| fresh | 128 | 158.0 / 145.3 | 306.0 / 234.2 | 471.3 / 302.8 | 696.4 / 365.3 (33.3) |

Finer partitions improve the exact greedy-quota information but are not declared winners: the audit has no
effect size, selector, multiplicity calibration, prompt incidence, or exact candidate universe.  The guarded
floor is intentionally allocation-agnostic and can be lower at larger `K` because it protects against any
cap-feasible concentration, not the prospective quota.  Neither diagnostic is a confidence interval.

## Decision and next work

All three `K` values define a jointly feasible structural bridge, and all continue forward.  No region count
is selected.  Every authorization field is false and the normal runner exits 2 after writing the completed
artifact; `--audit-only` is the explicit report-mode zero exit and changes no decision.

The next no-spend PR is Stage A: extend the complete repeated-judge simulation rather than jump to history or
candidate construction:

1. lift these fixed source matrices through the registered component allocations in the persistent-error DGP;
2. make source regions indivisible in outer and inner folds while keeping prompt blocks split-contained;
3. include prompt and source incidence jointly in the estimator and a two-way prompt/source or graph-aware
   simultaneous inference path;
4. use the upper `rho=.20` dependence envelope for error control, while keeping any later conditioner benefit
   on the separate train-only `s_safe` shrinkage path;
5. rerun the complete family-wise selector under block-null, mean-only, planted topology, and deranged controls
   and require at least 80% primary-event power in both synthetic corpora; and
6. only after that gate passes, begin Stage B by unlocking the attempted-input identity inventory and exact
   structural candidate packing.  Realized candidate exposure and prompt incidence must remain inside the
   powered envelope or trigger recalibration before Nomic or judge work.

`JointPosterior` remains the calibrated learned decision comparator.  It is not this source kernel and is not
the square-root/QR factorization.  Dense QR remains the numerical reference, and no QR/CUDA optimization is
eligible until a real covariance model passes the later statistical gates.

## Verification and reproduction

The new focused module/runner tests cover PSD and unit diagonal, input-order invariance, direct equivalence to
the canonical cumulative-walk feature map, boundary and isolated-node outside-landing mass, exact
cap/allocation behavior, analytic ESS, exhaustive toy verification of the guarded bound, malformed inputs,
two-corpus joint logic, all-false authorization, path-free numerical-runtime identity, portable atomic output,
and start/end scientific and external-input provenance drift.  The 35 focused tests pass; together with the
inherited source-region, capacity, and graph-geometry suites, 99 tests pass.  Python compilation and
`git diff --check` are clean.  The broader repeated-judge set has 146 passing tests when its multiprocessing
power-runner file is executed in the required fresh pytest process; loading graph/SciPy libraries first in the
same parent intentionally changes the recorded BLAS identity and makes that provenance test fail closed.

```bash
python prototypes/mu_cosine/run_repeated_judge_source_dependence.py \
  --artifact-repo /path/to/UnifyWeaver-with-ignored-graph-artifacts \
  --out /tmp/repeated_judge_source_dependence.json \
  --audit-only

python prototypes/mu_cosine/run_repeated_judge_source_dependence.py \
  --artifact-repo /path/to/UnifyWeaver-with-ignored-graph-artifacts \
  --out prototypes/mu_cosine/repro/repeated_judge_source_dependence/summary.json \
  --summary-only --audit-only
```

The tracked artifact is `repro/repeated_judge_source_dependence/summary.json`.  `--summary-only` changes only
the review projection, not scientific computation or decisions.  Omitting `--audit-only` still writes the
requested complete/summary artifact and exits 2 because this PR intentionally unlocks nothing.
