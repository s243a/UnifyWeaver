# Bridging the SimpleMind mindmaps to Wikipedia — via Pearltrees (3 hops)

The end-to-end "try what we built" run: connect the **systems-theory SimpleMind mindmap** to the Wikipedia
category/page graph, using Pearltrees as the bridge corpus.

## Pipeline
1. `parse_smmx.py "System Theory.smmx"` → 110 nodes; **91 carry a Pearltrees tree-id** (the harvest seeds).
2. `bridge_seeds.py` takes each seed and reads its harvested Pearltrees subtree (harvesting via the private
   `.local` cookie harvester if not cached) for the enwiki links each collection carries (its
   `Wiki / Encyclopedia type References`).
3. Output: **mindmap-concept → enwiki** `bridge` edges — the concept's own Wikipedia category plus its
   neighbourhood.

## Result (sample of 6 seeds, depth 1)
6 concepts → **208 bridges (43 to enwiki Categories)**. Each concept reaches its own category *and* a rich
neighbourhood:

| mindmap concept | principal bridge | + neighbourhood (sample) |
|---|---|---|
| `chaos-theory` | `Category:Chaos_theory` | Butterfly_effect, Catastrophe_theory, Feigenbaum_constants, `Category:Quantum_chaos_theory`, ~50 more |
| `bifurcation-theory` | `Category:Bifurcation_theory` | Pitchfork/Saddle-node/Transcritical bifurcation, Feigenbaum_constants |
| `dynamical-systems` | `Category:Dynamical_systems` | `Category:Ergodic_theory`, `Category:Stability_theory`, `Category:Symbolic_dynamics` |
| `celestial-mechanics` | `Category:Celestial_mechanics` | `Category:Astrodynamics`, `Category:Orbits`, Barycenter |
| `artificial-life` | `Category:Artificial_life` | Boids, Langton's_ant, Tierra, ~50 ALife pages |
| `butterfly-effect` | `Butterfly_effect` (page) | — |

So the bridge is **dense**: a single mindmap concept maps not to one Wikipedia node but to its category +
dozens of member pages / sibling categories — exactly the bridge-rich behaviour expected of Pearltrees.

## Notes
- **Harvester relocation fix.** Moving `fetch_pearltrees_tree.py` into `.local` broke its cookie path
  (it assumed the old `prototypes/mu_cosine` location and doubled `.local/tools`). Fixed in the private
  `.local` copy: cookies now resolved as `<script>/../pearltrees_cookies.txt`.
- `bridge_seeds.py` is **ref-based** and committed; the harvest cache (`.pt_cache/`) and `*_bridges.tsv`
  are gitignored (regenerable, contain private Pearltrees data). `--no-harvest` runs on cached harvests
  only (graceful on clones without `.local`).

## Next
Feed the `bridge` edges (+ the mindmap `subtopic`/`see_also`/… edges, tagged `corpus=mindmap, judge=human`,
node-types `mindmap_node`/`category`/`page`) into a graded round, then flip `--use-nodetype` on — these
mindmap↔enwiki bridges are exactly the within-operator type diversity `REPORT_nodetype.md` found missing.
