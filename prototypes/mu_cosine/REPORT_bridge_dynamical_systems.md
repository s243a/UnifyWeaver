# Bridging the Dynamical systems mindmap to Wikipedia — 1 / 2 / 3 hops (fused, privacy-scrubbed)

Same `fuse_corpus.py` walk as `REPORT_bridge_cybernetics.md`, this time from the **Dynamical systems**
SimpleMind map. A "hop" is ANY cross-corpus edge (mm↔mm, mm↔pt, pt↔pt, pt↔enwiki).

## Pipeline
1. `parse_smmx.py "Dynamical systems.smmx"` → **57 nodes, 35 carry a Pearltrees tree-id** (seeds); **0
   direct enwiki anchors** (all bridges via Pearltrees).
2. Harvested all 35 seeds (depth 1) → **977 bridges (88 to enwiki Categories) over 30 concepts**.
3. `fuse_corpus.py --start dynamical-systems --hops {1,2,3}` over the fused, privacy-scrubbed graph.

## Result

| hops | nodes | by corpus | edges | bridges |
|---|---|---|---|---|
| 1 | 12 | mm 11, pt 1 | 11 | 1 |
| 2 | 77 | mm 32, pt 29, wiki 16 | 123 | 34 |
| 3 | 753 | mm 42, pt 175, **wiki 536** | 932 | **626** |

> **Correction (later):** these `bridge` counts predate the same-concept gate — most pt→wiki links are
> cross-dataset *references* (different concepts), now typed `see_also`. Post-fix the 3-hop split is bridge
> **43** / see_also 586 (not 626 bridges). See `REPORT_graded_round.md` §Correction.

## Profile vs Cybernetics — bridge depth differs by how the map was curated
| map | seeds | bridges (harvest) | 2-hop wiki | 3-hop wiki |
|---|---|---|---|---|
| Cybernetics | 23 | 1389 | **150** | 1020 |
| Dynamical systems | 35 | 977 | **16** | 536 |

Despite **more** seeds, Dynamical systems exposes far fewer enwiki nodes at 2 hops (16 vs 150) — its
Pearltrees collections carry their enwiki references **deeper** (the bridges show up at hop 3: 536 wiki /
626 bridges), whereas Cybernetics collections reference enwiki right at the top level. The 2-hop
neighbourhood here is **pt-heavy** (29 pt vs 16 wiki) — more cross-collection (`assoc`) structure than
direct Wikipedia bridging. Useful signal: a single graded-round depth won't be uniform across maps; pick the
hop count per map by where its bridges actually land (2 for Cybernetics, 3 for Dynamical systems to get
comparable wiki coverage).

## Privacy
No `*private*` markers in this map's harvests (cybernetics had 2); the scrub-everywhere filter ran and
emitted clean TSVs (verified, 0 private).

## Notes
- Map under gitignored `context/`; `.pt_cache/` + fused `*_nodes/_edges.tsv` are local/regenerable (private-
  tier Pearltrees data) and not committed.

## Next
Graded round off the fused neighbourhoods — for Dynamical systems, the **3-hop** cut (753 nodes: mm 42 /
pt 175 / wiki 536, 626 bridges) gives wiki coverage comparable to Cybernetics 2-hop. relation→μ targets +
`Team <name> <id>` e5-text + `corpus`/`judge`/`account`/`node-type` tags, then train with `--use-nodetype`.
