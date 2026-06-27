# Bridging the Complex systems theory mindmap to Wikipedia — 1 / 2 / 3 hops (fused, privacy-scrubbed)

Same `fuse_corpus.py` walk as the cybernetics / dynamical-systems / LTI reports, from the **Complex systems
theory** SimpleMind map (`Subjects/sci/phy/math/calc/der/diff/sys/sys/complex/`) — the fourth systems-theory
starting point. Section relations categorised with the **fuzzy** method (typo + tag/qualifier + parent-signal).

## Pipeline
1. `parse_smmx.py "Complex systems theory.smmx"` → **41 nodes, 23 carry a Pearltrees tree-id**, 0 direct
   enwiki anchors.
2. Pearltrees seeds harvested **23 / 23** (`bridge_seeds.py --depth 1`; 13 newly harvested via the private
   `.local` harvester, 10 already cached) → 972 bridges, 115 to enwiki Categories, over 20 concepts.
3. `fuse_corpus.py --start complex-systems-theory --hops {1,2,3} --section-method fuzzy`.

## Result (full 23/23-seed coverage)

| hops | nodes | by corpus | edges | bridges | inferred |
|---|---|---|---|---|---|
| 1 | 6 | mm 5, pt 1 | 7 | 1 | 0 |
| 2 | 97 | **wiki 74**, pt 9, mm 14 | 127 | 7 | 3 |
| 3 | 301 | **wiki 249**, pt 30, mm 22 | 399 | 18 | 5 |

Unlike LTI / dynamical-systems (where enwiki sat **deep** — 2 wiki nodes at hop 2 → 200+ at hop 3), Complex
systems theory reaches Wikipedia **fast**: 74 wiki nodes already at hop 2. Its seeds carry direct enwiki
bridges, so the 2-hop cut already has real Wikipedia coverage. (The 2-hop cut is identical at 10/23 vs 23/23
coverage — the 13 newly-harvested seeds sit *deeper* in the map and only add at 3 hops: +35 edges, pt 29→30,
bridges 12→18.)

Good relation diversity at 3 hops: **element_of 266, super_category 53, subcategory 39, bridge 18,
subtopic 16, see_also 6, assoc 1**.

## Almost nothing inferred (the fuzzy layer earns its keep)
Only **5 / 399** edges are inferred at 3 hops — and all are `bridge` (conf 0.80, the title-match heuristic),
**not** section-inferred. The fuzzy section categoriser (typo + tag/qualifier + paren + parent-signal guard)
tagged *every* recognised section relation, so this map contributes essentially zero section-inferred noise —
consistent with `REPORT_section_embedding.md`'s finding that the lexical layers already capture the
relation-header signal.

## Notes
- Map under gitignored `context/`; `.pt_cache/` + fused TSVs (`context/cx_fused_*`) are local/regenerable
  (private-tier Pearltrees) and not committed.

## Next
- (Optional) harvest the 13 missing seeds for fuller pt→enwiki coverage, then re-walk.
- Fold this map into the **combined systems-theory graded round** (cybernetics + dynamical + LTI + complex)
  and **re-run the infer-blend A/B** — the larger, more diverse round where the posterior blend is predicted
  to separate from the v1 fixed-breadth heuristic.
