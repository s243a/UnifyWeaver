# Bridging the Chaos theory mindmap to Wikipedia — 1 / 2 / 3 hops (fused, privacy-scrubbed)

Same `fuse_corpus.py` walk as the other systems-theory maps, from the **Chaos theory** SimpleMind map
(`Subjects/.../sys/sys/complex/chaos/`). Fuzzy section categorisation.

## Pipeline
1. `parse_smmx.py "Chaos theory.smmx"` → **37 nodes, 20 carry a Pearltrees tree-id**.
2. `bridge_seeds.py --depth 1` → harvested **20/20 seeds**, 732 bridges (50 to enwiki Categories) over 15 concepts.
3. `fuse_corpus.py --start chaos-theory --hops {1,2,3} --section-method fuzzy`.

## Result

| hops | nodes | by corpus | edges | inferred |
|---|---|---|---|---|
| 1 | 8 | mm 7, pt 1 | 8 | 0 |
| 2 | 98 | wiki 66, pt 19, mm 13 | 132 | 2 |
| 3 | 378 | **wiki 294**, pt 63, mm 21 | 438 | 8 |

3-hop relation diversity: **element_of 291, subcategory 50, super_category 45, see_also 20, bridge 18,
subtopic 12, assoc 2**. Only 8/438 inferred — the fuzzy layer tagged almost every section relation.

## Notes
- Map + fused TSVs in gitignored `context/`; `.pt_cache/` harvests are private-tier, not committed.
- Folds into the systems-theory fine-tuning pool (cybernetics + dynamical + LTI + complex + chaos).
