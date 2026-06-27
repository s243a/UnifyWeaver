# Bridging the Emergence Pearltrees tree to Wikipedia — 1 / 2 / 3 hops (PT-rooted, no mindmap)

The first **Pearltrees-rooted** bridging walk (no SimpleMind map). Motivation: "Emergence" is a **leaf** in our
Wikipedia category graph (`enwiki_named`: parents `Ontology`/`Holism`, but **0 children**), so a wiki-only
bidirectional walk can't go down from it. The user's Pearltrees Emergence tree
(`pearltrees.com/s243a/emergence`, id **60457194**) has real category connectivity — **223 enwiki bridges**
across 4 sub-trees (Modularity, Superorganisms, …) — so we root the walk there instead.

## Enabler: `fuse_corpus.py --pt-seed SLUG:TREEID`
New option to root the fused walk **directly at a Pearltrees tree** (a start `mm:` node bridged to the PT
tree), bypassing the mindmap. `--smmx` is now optional; reusable for any PT-rooted walk.

## Pipeline
1. Harvest: `fetch_pearltrees_tree.py --tree-id 60457194 --depth 3` → 252 edges / 19 sections / **223 enwiki
   bridges** over 4 trees.
2. `fuse_corpus.py --pt-seed emergence:60457194 --start emergence --hops {1,2,3} --section-method fuzzy`.

## Result

| hops | nodes | by corpus | edges | inferred |
|---|---|---|---|---|
| 1 | 2 | mm 1, pt 1 | 1 | 0 |
| 2 | 118 | **wiki 108**, pt 9, mm 1 | 133 | 2 |
| 3 | 196 | **wiki 178**, pt 17, mm 1 | 217 | 3 |

Reaches Wikipedia **immediately** (108 wiki at hop 2) — the PT tree's PagePearls carry direct enwiki anchors.
3-hop relation diversity: **element_of 167, super_category 32, subcategory 14, bridge 4**. Only 3/217
inferred. (Membership-heavy, as expected for a PagePearl-rich tree; lighter on `see_also`/`assoc` than the
mindmap walks.)

## Notes
- PT harvest + fused TSVs are private-tier / gitignored (`.pt_cache/`, `context/`); not committed.
- Adds the Emergence region to the data pool — and validates the PT-rooted path for future "the Pearltrees
  page is better connected than the wiki node" cases.
