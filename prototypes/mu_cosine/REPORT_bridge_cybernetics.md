# Bridging the Cybernetics mindmap to Wikipedia — 1 / 2 / 3 hops (fused, privacy-scrubbed)

The "1, 2, 3 hop from the cybernetics mindmap, with bridging" run, using `fuse_corpus.py` — the unified
cross-corpus walker — over the **Cybernetics** SimpleMind map + its Pearltrees harvests + Wikipedia. A "hop"
is ANY cross-corpus edge (mm↔mm, mm↔pt, pt↔pt, pt↔enwiki), not only paths that reach enwiki.

## Pipeline
1. `parse_smmx.py Cybernetics.smmx` → **32 nodes, 23 carry a Pearltrees tree-id** (the harvest seeds). The
   map itself has **0 direct enwiki anchors** — all bridges come via Pearltrees.
2. Harvested all 23 seeds (depth 1) via the private `.local` harvester → **1389 bridges (267 to enwiki
   Categories) over 20 concepts**.
3. `fuse_corpus.py --start cybernetics --hops {1,2,3}` over the fused, **privacy-scrubbed** graph.

## Result

| hops | nodes | by corpus | edges | bridges |
|---|---|---|---|---|
| 1 | 14 | mm 13, pt 1 | 13 | 1 |
| 2 | 196 | mm 19, pt 27, **wiki 150** | 280 | **207** |
| 3 | 1269 | mm 27, pt 222, **wiki 1020** | 1476 | **1151** |

Cybernetics bridges **densely** to Wikipedia: by 2 hops a mindmap of 32 concepts reaches **150 enwiki
nodes** through 207 bridges; by 3 hops, **1020 enwiki nodes** / 1151 bridges. The 1-hop neighbourhood is
almost pure mindmap (the mm↔mm `subtopic`/`see_also`/`super_category` structure); pt and wiki enter at hop 2
(the seeds' own collections + their enwiki references) and explode at hop 3 (the collections' members).

> **Correction (later):** these `bridge` counts predate the same-concept gate. Most pt→wiki links here are
> the collection's cross-dataset *references* (different concepts), now correctly typed `see_also`, not
> `bridge`. Post-fix the 2-hop split is bridge **19** / see_also 196 (not 207 bridges). See
> `REPORT_graded_round.md` §Correction.

## Privacy — the scrub-everywhere filter fired on real data
The fresh harvest pulled **2 collections containing `*private*` markers** (`Intelegence & mind`,
`System Perspective (Philosophy)` — each a `*private*` shortcut). `fuse_corpus.py` now applies the same
scrub-everywhere rule to the raw `.pt_cache` harvests (which do **not** pass through `parse_pearltrees.py`):
it dropped those private rows + their inherited subtree, and the emitted fused TSVs contain **no** private
data (verified). This closed a real gap — before this run, the Pearltrees side of a fused walk was
unscrubbed, so private collections would have leaked into the fused output.

## Notes
- The privacy rule now lives in one place (`privacy.py`), imported by `parse_smmx.py`,
  `parse_pearltrees.py`, and `fuse_corpus.py`, so it can't drift between corpora.
- The mindmap lives under the gitignored `context/`; `.pt_cache/` and the fused `*_nodes/_edges.tsv` are
  local/regenerable (contain private-tier Pearltrees data) and are **not** committed.

## Next
Feed the 2-hop fused neighbourhood (196 nodes: mm 19 / pt 27 / wiki 150, with 207 bridges) into the graded
round — relation→μ targets, `Team <name> <id>` e5-text, and the `corpus`/`judge`/`account`/`node-type` tags
— then train with `--use-nodetype` on. This is the within-operator type diversity `REPORT_nodetype.md` found
missing, now privacy-clean.
