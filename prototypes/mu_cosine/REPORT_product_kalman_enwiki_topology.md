# Enwiki Product-Kalman Campaign Topology Audit

Status: pre-scoring descriptive audit. These measurements are not model inputs, confidence estimates, or causal
evidence for a calibration outcome.

## Source

The audit used the semantic-ready scoped category store:

    data/benchmark/enwiki_cats_correct/lmdb_scoped
    scope root: Main_topic_classifications (page ID 7345184)
    data.mdb size: 1,319,403,520 bytes
    data.mdb mtime_ns: 1783430413932204744
    graph tables: category_parent, category_child
    title tables: title_i2s, title_s2i
    title layer kind: mediawiki_page_titles

Command:

```bash
python3 prototypes/mu_cosine/audit_product_kalman_lmdb_topology.py \
  --lmdb-dir data/benchmark/enwiki_cats_correct/lmdb_scoped \
  --scope-root Main_topic_classifications \
  --corpus enwiki \
  --graph-view category_dag \
  --output /tmp/mu_data/enwiki_campaign_topology_audit.json
```

The JSON is an ephemeral local artifact whose SHA-256 for this run is
`522863dce41808ae5282cda402dc71391688747a833710d810d2d4e99378b7b7`. The committed audit code and the source
fingerprint above are the durable regeneration anchors.

## Results

| measurement | value |
| --- | ---: |
| graph nodes (union of numeric adjacency keys) | 2,248,538 |
| category-parent entries | 6,706,581 |
| category-child entries | 6,706,581 |
| reciprocal table entry counts match | yes |
| title_i2s rows | 2,245,288 |
| direct children of scope root | 35 |

Parent count uses all graph nodes as its denominator, including the root's zero:

| parent-count statistic | value |
| --- | ---: |
| zero parents | 1 |
| more than one parent | 2,076,218 (92.3364%) |
| mean | 2.9826 |
| median | 3 |
| p95 | 5 |
| p99 | 7 |
| maximum | 202 |

Child count also uses all graph nodes, including leaves:

| child-count statistic | value |
| --- | ---: |
| zero children | 1,103,713 |
| more than one child | 768,996 (34.1998%) |
| mean | 2.9826 |
| median | 1 |
| p95 | 12 |
| p99 | 36 |
| maximum | 25,053 |

## Interpretation

The scoped enwiki category view is strongly multi-parent: it should be treated as a DAG, not approximated as a
principal-parent tree for the primary campaign. That creates a useful structural contrast with the proposed
principal-parent Pearltrees and SimpleMind views.

This does not establish that tree-likeness improves Product-Kalman calibration, nor that it offsets title noise.
Those claims require held-out within-corpus graph-view/title-view ablations. In particular, node degree is not a
proxy for trained-neighbor density and must not be used as a confidence feature.

The streaming audit checks equality of the two reciprocal table entry counts, not every reciprocal edge pair, and
does not test graph acyclicity. Those limitations are recorded in the JSON output.
