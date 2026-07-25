# SimpleMind filesystem filing corpus — a fresh, frozen-holdout filing dataset

**The owner's insight, validated:** the .smmx maps live inside a real Dropbox folder hierarchy,
so each map's path is a RECORDED FILING DECISION — map root ↔ parent directory, the same
single-principal-folder task as Pearltrees, on a corpus the eval program has never touched.

## Corpus + discipline (sm_fs_filing.py)

4,385 maps scanned; privacy-filtered (any path component containing 'private' dropped) → 4,367
with a filing folder; catalog = 270 distinct directory names (313 dirs ≥3 maps, the PT min_bm
analog) → 1,272 eligible queries. **Frozen holdout at first touch (finding-5 lesson): 508
queries (40%, seed 0) RESERVED before any scoring** — hashes digested in ~/mu_data/sm_fs_ledger
.json (`eb02e3694be3b71d`); this script never scores them. Exploration split n=764.

## Result (exploration split, e5 ranking)

| corpus | catalog | R@1 | MRR | R@50 |
|---|---|---|---|---|
| **SM-FS filing** | 270 | **0.478** | **0.560** | 0.819 |
| PT filing (standing) | 335 | 0.203 | 0.291 | — |
| SM in-map parent | 200 | 0.180 | 0.320 | — |

## Interpretation (revises the cross-corpus claim)

Diagnostic: exact title==dir only 1.3%, but **78% substring overlap** — maps are topic-named and
filed in topic-named folders, so the query→folder semantic gap is small. Pearltrees is hard
because bookmarks are wild page titles far from folder vocabulary. Refinement of
REPORT_wiki_multiparent's conclusion: task SHAPE (one principal parent) sets the metric
structure, but the QUERY→FOLDER SEMANTIC DISTANCE distribution sets the difficulty level —
PT ≈ SM-in-map (wild/deep queries), SM-FS much easier (topic-identity queries).

## Uses

1. **Training corpus:** 4,367 real filing decisions with materialized directory paths — the
   LINEAGE target-factory shape (`lineage(fs, decay=…)` as a process expression), 5.5× the 799
   campaign rows, available NOW while the harvester drains and P1 awaits v2 bundles.
2. **OPENQ-012 cross-corpus arm:** a fourth task-matched corpus (PT / SM-in-map / wiki / SM-FS).
3. **Confirmatory reserve:** the 508 frozen queries are this program's first
   designed-before-touched holdout — usable for confirmatory tests under a preregistered
   protocol only.

Caveats: exploration-split numbers are descriptive (single seed, no CIs); catalog uses directory
NAMES (duplicates title-equivalent); depth-0 maps (no filing folder) excluded; per-item data
stays in ~/mu_data (paths/names may be personal), repo carries aggregates only.
