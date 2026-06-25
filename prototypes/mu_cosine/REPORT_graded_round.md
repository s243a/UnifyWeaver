# The multi-corpus graded round — calibration & distribution

`build_graded_round.py` turns the PRIVACY-SCRUBBED fused neighbourhoods (`fuse_corpus.py`) into a calibrated,
tagged graded round for the trainer. This documents the **relation→operator→μ calibration** (the heart of
it) and the distribution it produces on the two banked neighbourhoods (cybernetics 2-hop + dynamical-systems
3-hop).

## Calibration — relation → operator + μ

Corpus differences are **not** baked into the target μ; they ride the maskable provenance token so the model
conditions/marginalizes them (per `DESIGN_provenance_and_representation.md`). One human-judge μ per relation:

| relation | operator | kind | μ(member\|container) | μ(reverse) | rationale |
|---|---|---|---|---|---|
| `subcategory` | WIKI | directional | 0.90 | 0.10 | explicit narrower category |
| `subtopic` | WIKI | directional | 0.85 | 0.12 | narrower topic (mindmap), looser than a category |
| `element_of` | ELEM | directional | 0.90 | 0.10 | page/collection membership |
| `super_category` | WIKI | directional (dst broader) | 0.85 | 0.12 | src belongs to the broader dst |
| `see_also` | SYM | symmetric | 0.40 | 0.40 | weakly related |
| `assoc` | SYM | symmetric | 0.30 | 0.30 | associative cross-link |
| `sequence` | SYM | symmetric | 0.30 | 0.30 | reading-order (navigation) |
| `bridge` | SYM | symmetric | 0.90 | 0.90 | the SAME concept across corpora |

Reverse targets (0.10–0.12) are the **semantic-drift floor**: a member is mostly-not its container. Each
directional edge emits BOTH directions so the model learns the asymmetry, not just the positive.

**Provenance = the human-curated endpoint.** A bridge is mm/pt ↔ wiki; enwiki didn't assert it, the
mindmap/pearltrees curation did — so *both* directions of a bridge get the mm/pt side's `corpus`/`judge`
(priority mindmap > pearltrees > enwiki), never `enwiki/graph`.

## Distribution (cybernetics 2-hop + dynamical-systems 3-hop)

916 nodes, 1212 fused edges → **2334 graded targets**.

- **by operator:** WIKI 244, SYM 2084, ELEM 6
- **by relation:** bridge 1654, assoc 408, subtopic 222, see_also 22, super_category 22, element_of 6
- **by corpus⊗judge:** pearltrees/human 2122, mindmap/human 212
- **by μ:** 0.90 ×1657, 0.85 ×122, 0.40 ×22, 0.30 ×408, 0.12 ×122, 0.10 ×3

## Two things the trainer step must handle

1. **Bridge dominance.** 1654 / 2334 targets are `bridge` at μ=0.90. That is the cross-corpus **node-type
   diversity** `--use-nodetype` needs (mm↔pt↔wiki, distinct node-types under one relation) — but left
   unweighted it would teach "everything bridged is similar" and swamp the directional WIKI/ELEM signal.
   The trainer should **subsample or down-weight** bridges (and balance per operator).
2. **ELEM is thin here (6).** These neighbourhoods are collection/category/bridge-heavy, few member pages;
   ELEM is already trained on the wiki page data, so this round mainly contributes WIKI (mindmap hierarchy)
   + SYM (bridges/assoc/see_also) with the new cross-corpus type diversity.

## Outputs (not committed — derived from gitignored fused/`.pt_cache` data)
- `<out>_pairs.tsv`: `node  root  mu  op  relation  node_type  root_type  corpus  judge`
- `<out>_nodes.tsv`: `key  corpus  node_type  title  embed_text` — `embed_text` is the e5 string per node
  (the `Team <name> <id>` prefix hook lives here; inert until the `s243a_groups`/teams account is harvested).

## Next
Wire the trainer to consume `<out>_pairs.tsv`: a mixed-operator graded path (per-row `op`, MSE to μ +
directional margins), bridge down-weighting, the fused nodes unioned into the e5 build, then train with
`--use-nodetype` on and measure whether type diversity now helps (`REPORT_nodetype.md` found it collinear
*without* this data).
