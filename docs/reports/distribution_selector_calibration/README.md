# Distribution Selector Calibration

This report records a small smoke calibration of the distribution
representation selector on numeric-keyed category LMDB fixtures.

The runs use bounded simple parent paths, so cycles are skipped per path and
rows can be capped by `path_cap` or `expansion_cap`.  The active selector gate
is `max_cdf_error <= 0.001`; no `max_w1_error` gate is applied.

## Fixtures

| graph | root | depths | budgets | targets/depth | notes |
|-------|------|--------|---------|---------------|-------|
| simplewiki Articles | `2` | `2,3,4` | `4,6,8` | `5` | Fixture metadata calls this the best Articles-like root; the sampled frontier only produced one depth-2 target. |
| enwiki Main topic classifications | `7345184` | `2,3,4` | `4,6,8` | `5` | Correct-mode enwiki category LMDB rooted at `Main_topic_classifications`. |

## Results

### SimpleWiki

The smoke sample is near-chain: all three reachable target-budget rows have
one path and one histogram bin.

| workload | selected representation | rows | mean bytes | mean CDF error |
|----------|-------------------------|------|------------|----------------|
| prefix mass | `quantized_cdf_table` | 6 | 26.000 | 0.000000 |
| arbitrary functional | `packed_sparse_histogram` | 6 | 36.000 | 0.000000 |

### Enwiki

The enwiki smoke sample shows the expected parent branching pressure.  At
budget 6, reachable rows average 18.8 paths and 3.8 bins; at budget 8 every row
hits the expansion cap, so those rows should be treated as bounded smoke data,
not final measurements.

| workload | selected representation | rows | mean bytes | mean CDF error | mean W1 error |
|----------|-------------------------|------|------------|----------------|---------------|
| prefix mass | `quantized_cdf_table` | 84 | 30.714 | 0.000004 | 0.000009 |
| arbitrary functional | `packed_sparse_histogram` | 84 | 60.857 | 0.000000 | 0.000000 |

The current parametric fits do not pass the strict CDF gate on this enwiki
sample:

| model | rows | mean L1 | mean CDF error |
|-------|------|---------|----------------|
| `binomial_fit` | 42 | 0.309813 | 0.091852 |
| `shifted_gamma_fit` | 42 | 0.734725 | 0.345684 |

## Interpretation

- Packed exact encodings dominate the current binomial/Gamma fits under a
  strict `0.001` CDF gate.
- `quantized_cdf_table` is the right first-choice representation for prefix
  mass workloads.
- `packed_sparse_histogram` remains the right first-choice representation for
  arbitrary functionals because CDF-only tables do not preserve enough PMF
  detail.
- The enwiki budget-8 rows are capped; a deeper calibration should lower target
  counts or raise caps before treating those numbers as policy thresholds.

## Artifacts

- [SimpleWiki summary](lmdb_parent_histogram_benchmark_summary_simplewiki_articles_selector_smoke_20260613T044723Z.md)
- [SimpleWiki JSONL](lmdb_parent_histogram_benchmark_simplewiki_articles_selector_smoke_20260613T044723Z.jsonl)
- [Enwiki summary](lmdb_parent_histogram_benchmark_summary_enwiki_mtc_selector_smoke_20260613T044735Z.md)
- [Enwiki JSONL](lmdb_parent_histogram_benchmark_enwiki_mtc_selector_smoke_20260613T044735Z.jsonl)
