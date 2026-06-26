# Bridging the LTI System Theory mindmap to Wikipedia — 1 / 2 / 3 hops (fused, privacy-scrubbed)

Same `fuse_corpus.py` walk as the cybernetics / dynamical-systems reports, from the **LTI System Theory**
SimpleMind map — the third systems-theory starting point (the user is sweeping systems-theory mindmaps, then
cybernetics).

## Pipeline
1. `parse_smmx.py "LTI System Theory.smmx"` → **77 nodes, 40 carry a Pearltrees tree-id** (the most so far),
   2 direct enwiki anchors.
2. Harvested all 40 seeds (depth 1) → **299 bridges (40 to enwiki Categories) over 28 concepts**.
3. `fuse_corpus.py --start lti-system-theory --hops {1,2,3}`.

## Result

| hops | nodes | by corpus | edges | bridges | inferred |
|---|---|---|---|---|---|
| 1 | 13 | mm 12, pt 1 | 12 | 1 | 0 |
| 2 | 50 | mm 25, pt 23, wiki 2 | 86 | 13 | 24 |
| 3 | 336 | mm 37, pt 91, **wiki 208** | 394 | 29 | 233 |

Like dynamical-systems, the enwiki references sit **deep** (2 wiki nodes at hop 2 → 208 at hop 3); the 3-hop
cut is the one with real Wikipedia coverage. Good relation diversity at 3 hops: **subcategory 44, element_of
205, see_also 29, super_category 17, bridge 29**.

## Combined systems-theory round (cybernetics 2-hop + dynamical 3-hop + LTI 3-hop)
`build_graded_round.py --fused cyb_fused_2 --fused ds_fused_3 --fused lti_fused_3`:
- **1027 nodes, 1607 edges → 2900 graded targets** (was 2488 without LTI).
- by op: **WIKI 778** (subcategory 436, super_category 178, subtopic 164), **ELEM 1496**, SYM 626.
- **inferred 500 / 2900** (was 400) — +100 from LTI, and notably more *subcategory* (436 vs 396), the
  directional axis the infer-blend posterior most needs diversity on.

## Notes
- Map under gitignored `context/`; `.pt_cache/` + fused / graded TSVs are local/regenerable (private-tier
  Pearltrees data) and not committed.

## Next
- Keep accumulating systems-theory maps (then cybernetics), then **re-run the infer-blend A/B on the combined
  round** — this is the larger, more *diverse* inferred set where the posterior blend should start to separate
  from the v1 fixed-breadth heuristic (which only switches element_of→subcategory).
- **Parser improvement for more LABEL data (high value):** add the `fuzzy` method to `pt_sections.py` — a
  normalised / edit-distance match against the section keywords — so typo'd headers (`Subtoipcs`,
  `More Subtoipcs`) are *categorised* (→ tagged `element_of`) instead of falling through to the inferred
  structural default. That both **grows the labelled set** (better joint-head fit + graded data) and **shrinks
  the inferred-noise** the blend has to absorb.
