# Pearltrees Product-Kalman Campaign Sampling

Status: pre-scoring exploratory artifact. No judge labels or Product-Kalman result were produced in this run.

## Primary View

The primary Pearltrees view is path-local lineage, not the unioned multi-parent DAG. Each
`api_tree_paths_v8.jsonl` record supplies one account-tagged materialized `path_ids` lineage. Candidate pairs are
formed only within a recorded lineage. The account token is provenance and is not a graph node.

This distinction matters in the current snapshot. Individual paths are tree-like, but their union can cycle. For
example, some records place `ethics` beneath `philosophy`, while others place `philosophy` beneath `ethics`. The
sampler excludes an endpoint pair whenever records disagree about its direction or hop; it does not break the cycle
or select one context post hoc. Alias and secondary-reference edges in `assembled_dag.tsv` remain a separate
sensitivity view.

## Source

| artifact | bytes | mtime_ns | SHA-256 |
| --- | ---: | ---: | --- |
| `.local/data/api_tree_paths_v8.jsonl` | 347,589 | 1769651833578690785 | `89c1ceff7ef95293a9771eb343c21f5a01fe88a1b1e0efab41840f19b998ebc5` |
| `.local/data/pearltrees_api/assembled_titles.tsv` | 472,564 | 1782970573157666345 | `65bd587b680f2092bd4e3cac586bf80d54ed2804515be221946159b8c5316ded` |

Command:

```bash
python3 prototypes/mu_cosine/sample_product_kalman_pearltrees_campaign.py \
  --paths-jsonl .local/data/api_tree_paths_v8.jsonl \
  --titles-tsv .local/data/pearltrees_api/assembled_titles.tsv \
  --pairs 250 --hmax 5 --seed 0 \
  --pairs-tsv /tmp/mu_data/pearltrees_campaign_pairs_unscored.tsv \
  --score-in /tmp/mu_data/pearltrees_campaign_score_in_unscored.tsv \
  --manifest /tmp/mu_data/pearltrees_campaign_manifest_unscored.json
```

## Source Audit

| measurement | value |
| --- | ---: |
| path records | 881 |
| private paths removed before title aliasing | 129 |
| retained path records | 752 |
| unique path nodes | 1,294 |
| unique consecutive path edges | 1,188 |
| consecutive edge observations | 2,920 |
| title-covered path nodes | 1,224 |
| missing-title path nodes | 70 |
| retained path/table title disagreements preserved as aliases | 152 |
| endpoint pairs rejected for cross-record direction conflict | 20 |

One direction-conflict example is tree IDs `10311488` and `10391040`: 29 observations point one way and 41 point
the other, all at hop 1. Such pairs are absent from the campaign table and retained as manifest diagnostics.

## Sample

The emitted table contains 250 unique unordered endpoint pairs, 50 at each hop from 1 through 5.

| hop | eligible pool | represented top-level components | maximum selected from one component |
| ---: | ---: | ---: | ---: |
| 1 | 1,077 | 50 | 1 |
| 2 | 965 | 41 | 2 |
| 3 | 738 | 30 | 2 |
| 4 | 492 | 25 | 3 |
| 5 | 352 | 16 | 4 |

The 250 rows contain 284 unique endpoint IDs, 282 unique raw titles, and 281 unique normalized titles. Three pairs
of endpoint IDs collapse into duplicate normalized-title groups. No semantic corrections were applied. The pair
table preserves assembled raw titles plus path-record aliases; only normalized copies are used for matching.

## Ephemeral Artifacts

| artifact | SHA-256 |
| --- | --- |
| `/tmp/mu_data/pearltrees_campaign_pairs_unscored.tsv` | `586c81c9065ef4f003f8039c78054f0f96bf35a0c311d1a8c3399533733f230f` |
| `/tmp/mu_data/pearltrees_campaign_score_in_unscored.tsv` | `f05b404632b8649b0e89a37a0d1b192a64bd18d8601a5cbdf9ebee910db2098a` |
| `/tmp/mu_data/pearltrees_campaign_manifest_unscored.json` | `51d0c970abce1d382c5f7b3d93653f7a30e42eacebf0632534aed2306593c4d5` |

These files are local and ephemeral. The committed sampler, source fingerprints, seed, and command are the durable
regeneration anchors.

## Interpretation Guardrails

- The sample is judge-ready, not scored evidence.
- Path-local tree-likeness does not establish that Pearltrees calibrates better than enwiki.
- Title aliases are provenance, not automatic corrections.
- Direction-conflicted pairs are excluded rather than resolved from model residuals or judge labels.
- Product-Kalman still requires node-disjoint calibration/evaluation and comparison with the registered
  `JointPosterior` baseline on NLL, calibration, and selective risk.
