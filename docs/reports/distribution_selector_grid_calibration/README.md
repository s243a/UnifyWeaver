# Distribution Selector Grid Calibration

This directory contains a decision-oriented calibration of the distribution
representation selector on numeric-keyed category LMDB fixtures.

Compared with the earlier smoke run, this pass adds:

- per-budget and per-child-depth representation selection tables;
- parametric CDF pass counts under the strict selector gate;
- an enwiki grid that avoids the budget-8 expansion-cap regime;
- a broader SimpleWiki sample from the Articles-like root.

The selector gate is `max_cdf_error <= 0.001`; no `max_w1_error` gate is
enabled.  Runs use bounded simple parent paths and skip repeated nodes per
path.

## Fixtures

| graph | root | depths | budgets | targets/depth | caps |
|-------|------|--------|---------|---------------|------|
| simplewiki Articles-like root | `2` | `1,2,3` | `4,5,6` | `10` | `path_cap=20000`, `expansion_cap=50000` |
| enwiki Main topic classifications | `7345184` | `2,3,4` | `4,5,6` | `10` | `path_cap=20000`, `expansion_cap=50000` |

## Headline Results

| graph | target-budget rows | reachable | capped | prefix winner | arbitrary-functional winner |
|-------|-------------------:|----------:|-------:|---------------|-----------------------------|
| simplewiki | 60 | 60 | 0 | `quantized_cdf_table` | `packed_sparse_histogram` |
| enwiki | 90 | 90 | 0 | `quantized_cdf_table` | `packed_sparse_histogram` |

## SimpleWiki

The broader SimpleWiki sample is still essentially near-chain.  Histograms have
about one path and one bin on average across budgets 4, 5, and 6.  The full
parent degree is not tiny, but the root-reaching parent degree is close to
one, which is what matters for root-path histograms.

| budget | reachable | mean paths | mean bins | mean full parent degree | mean root-reaching parent degree |
|-------:|----------:|-----------:|----------:|------------------------:|---------------------------------:|
| 4 | 20 | 1.050 | 1.050 | 4.950 | 1.050 |
| 5 | 20 | 1.050 | 1.050 | 4.950 | 1.050 |
| 6 | 20 | 1.050 | 1.050 | 4.950 | 1.050 |

Parametric fits mostly pass the CDF gate because the distributions are usually
single-bin.  Even then, the selected representations remain packed exact:

- prefix-mass workload: `quantized_cdf_table`;
- arbitrary functional workload: `packed_sparse_histogram`.

## Enwiki

The enwiki grid is uncapped at budgets 4, 5, and 6 and shows increasing path
and support growth.

| budget | reachable | mean paths | p95 paths | mean bins | p95 bins | mean nodes expanded |
|-------:|----------:|-----------:|----------:|----------:|---------:|--------------------:|
| 4 | 30 | 4.200 | 11.000 | 1.933 | 3.000 | 745.5 |
| 5 | 30 | 11.167 | 36.000 | 2.967 | 4.000 | 3861.3 |
| 6 | 30 | 30.000 | 81.000 | 4.033 | 5.000 | 18888.7 |

Current parametric families are not competitive under the strict CDF gate:

| model | rows | mean L1 | p95 L1 | max L1 | mean CDF error |
|-------|-----:|--------:|-------:|-------:|---------------:|
| `binomial_fit` | 90 | 0.241081 | 0.814815 | 0.982400 | 0.068143 |
| `shifted_gamma_fit` | 90 | 0.834301 | 1.277469 | 1.352708 | 0.402378 |

Parametric CDF pass counts drop as budget grows:

| child depth | budget | model rows | parametric CDF pass rows |
|------------:|-------:|-----------:|-------------------------:|
| 2 | 4 | 20 | 2 |
| 2 | 5 | 20 | 0 |
| 2 | 6 | 20 | 0 |
| 3 | 4 | 20 | 11 |
| 3 | 5 | 20 | 0 |
| 3 | 6 | 20 | 0 |
| 4 | 4 | 20 | 19 |
| 4 | 5 | 20 | 9 |
| 4 | 6 | 20 | 0 |

The selector still picks packed exact encodings for every row:

- prefix-mass workload: `quantized_cdf_table`, mean 29.956 bytes,
  mean CDF error `0.000003`;
- arbitrary functional workload: `packed_sparse_histogram`, mean 57.867 bytes,
  exact CDF/W1.

## Policy Implication

Under a strict `max_cdf_error <= 0.001` gate, the current binomial and shifted
Gamma fits should not be used as default replacements for enwiki parent-path
histograms.  The next implementation step should improve packed exact storage
and only add richer parametric families after a grid shows packed exact losing
on byte cost or runtime.

## Artifacts

- [SimpleWiki summary](lmdb_parent_histogram_benchmark_summary_simplewiki_articles_selector_grid_20260613T051706Z.md)
- [SimpleWiki JSONL](lmdb_parent_histogram_benchmark_simplewiki_articles_selector_grid_20260613T051706Z.jsonl)
- [Enwiki summary](lmdb_parent_histogram_benchmark_summary_enwiki_mtc_selector_grid_20260613T051654Z.md)
- [Enwiki JSONL](lmdb_parent_histogram_benchmark_enwiki_mtc_selector_grid_20260613T051654Z.jsonl)
