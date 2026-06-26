# Bridging the Complex systems theory mindmap to Wikipedia — 1 / 2 / 3 hops (fused, privacy-scrubbed)

Same `fuse_corpus.py` walk as the cybernetics / dynamical-systems / LTI reports, from the **Complex systems
theory** SimpleMind map (`Subjects/sci/phy/math/calc/der/diff/sys/sys/complex/`) — the fourth systems-theory
starting point. Section relations categorised with the **fuzzy** method (typo + tag/qualifier + parent-signal).

## Pipeline
1. `parse_smmx.py "Complex systems theory.smmx"` → **41 nodes, 23 carry a Pearltrees tree-id**, 0 direct
   enwiki anchors.
2. Pearltrees cache coverage: **10 of 23 seeds** already harvested in `.pt_cache`; **13 not yet harvested**
   (the walk below uses the 10 + the full mindmap structure — see *Coverage* note).
3. `fuse_corpus.py --start complex-systems-theory --hops {1,2,3} --section-method fuzzy`.

## Result

| hops | nodes | by corpus | edges | bridges | inferred |
|---|---|---|---|---|---|
| 1 | 6 | mm 5, pt 1 | 7 | 1 | 0 |
| 2 | 97 | **wiki 74**, mm 14, pt 9 | 127 | 7 | 3 |
| 3 | 300 | **wiki 249**, pt 29, mm 22 | 364 | 12 | 4 |

Unlike LTI / dynamical-systems (where enwiki sat **deep** — 2 wiki nodes at hop 2 → 200+ at hop 3), Complex
systems theory reaches Wikipedia **fast**: 74 wiki nodes already at hop 2. Its cached seeds carry direct
enwiki bridges, so the 2-hop cut already has real Wikipedia coverage.

Good relation diversity at 3 hops: **element_of 249, super_category 47, subcategory 33, subtopic 16,
bridge 12, see_also 6, assoc 1**.

## Almost nothing inferred (the fuzzy layer earns its keep)
Only **4 / 364** edges are inferred at 3 hops — and all 4 are `bridge` (conf 0.80, the title-match
heuristic), **not** section-inferred. The fuzzy section categoriser (typo + tag/qualifier + paren +
parent-signal guard) tagged *every* recognised section relation, so this map contributes essentially zero
section-inferred noise — consistent with `REPORT_section_embedding.md`'s finding that the lexical layers
already capture the relation-header signal.

## Coverage
This walk used **10 of 23** pt seeds (the cached ones). The 13 un-harvested tree-ids
(`10963876 13730217 14699448 14699455 60213399 60213481 60343836 60458362 60458440 60480476 61158790
61158910 61159614`) would add more pt → enwiki bridges; harvesting them needs the **private** `.local`
harvester + a Pearltrees session (not run here without confirmation). Even at 10/23, the 3-hop cut reaches
249 Wikipedia nodes.

## Notes
- Map under gitignored `context/`; `.pt_cache/` + fused TSVs (`context/cx_fused_*`) are local/regenerable
  (private-tier Pearltrees) and not committed.

## Next
- (Optional) harvest the 13 missing seeds for fuller pt→enwiki coverage, then re-walk.
- Fold this map into the **combined systems-theory graded round** (cybernetics + dynamical + LTI + complex)
  and **re-run the infer-blend A/B** — the larger, more diverse round where the posterior blend is predicted
  to separate from the v1 fixed-breadth heuristic.
