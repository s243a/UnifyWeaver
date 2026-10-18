# Wikipedia multi-parent hit@k + strategic harvest prioritization

## Multi-parent study (wiki_multiparent_hits.py)

enwiki_named correct-ingest graph (9.9M title edges, admin cats excluded); 400 sampled categories
with ≥4 parents (median 5, max 14); catalog = 5,000 (1,453 true-parent union + 3,547 random
distractors); e5 query→passage ranking.

| k | ≥1 parent | ≥2 | ≥3 | all 4 |
|---|---|---|---|---|
| 1 | 0.945 | 0 | 0 | 0 |
| 5 | 0.973 | 0.920 | 0.752 | 0.347 |
| 10 | 0.975 | 0.935 | 0.800 | 0.400 |
| 50 | 0.985 | 0.958 | 0.868 | 0.448 |
| 100 | 0.990 | 0.970 | 0.890 | 0.470 |
| 500 | 0.998 | 0.988 | 0.922 | 0.580 |

k to capture the 1st parent: median 1 (p90 1). k to capture all 4: **median 181, p75 1,806,
p90 3,623** (of 5,000).

Reading: corpus structure dominates. Wikipedia's multiple valid parents make ANY-parent trivial
(0.945@1 vs Pearltrees single-folder R@1 0.203 — one right answer is the harder per-target task,
as predicted). But parents are FACETS (decade × genre × country × …): e5 surfaces the semantically
dominant facets within k≈5, and the last parent is typically an ORTHOGONAL facet with little
lexical overlap — full multi-parent recovery is facet enumeration, not similarity ranking.
Consequence for Pearltrees' 880 multi-parent filing folders: recovering non-principal parents
needs facet-style candidate generation (graph/structural), not a better similarity ranker.
Caveats: single seed; catalog-bounded (5k); distractors uniform-random (no hard negatives).

## Strategic harvest prioritization (prioritize_harvest_queue.py)

The augmented harvest queue (4,739 trees) drains over weeks at account-safe pacing, so ORDER is
the lever: each queued tree is scored by max e5 similarity to the 741 filing-failure bookmarks
(rank>5 on the standing manifest) and the queue is re-sorted. Top priorities came out as the
politics/society gap topics (Authority, Deception, Injustice, Misinformation …) — matching
REPORT_hybrid_candidates §B″ (the gap mass is news/politics, not STEM). Harvester relaunched on
the prioritized queue at unchanged pacing; batch_state dedup makes reordering safe/resumable.

## Cross-corpus table (sm_filing_hits.py adds SimpleMind)

| task (single right answer unless noted) | catalog | R@1 | MRR |
|---|---|---|---|
| SM parent-level (immediate parent) | 200 | 0.180 | 0.320 |
| PT single-folder (standing) | 335 | 0.203 | 0.291 |
| SM root-level (map root + PT distractors) | 346 | 0.128 | 0.218 |
| wiki any-parent (multi-target) | 5,000 | 0.945 | — |
| wiki all-4-parents | 5,000 | median k=181 | — |

Owner's claim CONFIRMED: Pearltrees and SimpleMind are the same task — same shape (one principal
parent), same difficulty band (R@1 0.18–0.20, MRR 0.29–0.32) across different corpora and catalog
sizes. Task structure, not corpus identity, sets ranking difficulty; Wikipedia's facet structure
is the outlier in both directions (any-parent trivial, all-parents deep-tail). Root-level is
harder than parent-level (0.128): deep nodes drift semantically from their map root, so
root↔filing matching is best done at the map level (shallow nodes/roots), not per deep node.
488 SM lineage rows (6 maps, privacy-filtered at parse); single seed; descriptive.
