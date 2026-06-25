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

## Correction: `bridge` is same-concept only (not every cross-dataset link)

A `bridge` means the **same concept** across corpora (identity, μ≈0.9). `fuse_corpus.py` /
`parse_pearltrees.py` originally labelled *every* pt→wiki link `bridge`, but most are the collection's
cross-dataset **references** to *different* things (`Cybernetics` collection → `Centrifugal governor` page) —
those are `see_also` (associative, μ≈0.4), not identity. A **same-concept gate** (normalised titles match ⇒
`bridge`, else `see_also`) corrects this: on the two neighbourhoods, **bridge 1654 → 122**, **see_also 22 →
1550**. The remaining 122 bridges are genuine identity links (high e5; the ensemble keeps 60/61). This makes
the round semantically honest — the model learns `centrifugal-governor ~0.4 cybernetics` (related), not
`=0.9` (identical). Bridge dominance is no longer even an issue; `see_also` now carries the cross-dataset mass
at its correct associative μ.

## Bridge quality — negatives + an e5-prior review gate

Two refinements to how bridges are handled (a bridge asserts "same concept across corpora" at μ≈0.9):

### Bridge negative sampling (`--bridge-neg-ratio`, default 1.0)
Bridges were all *positive*, so the model never saw "these two cross-corpus nodes are NOT the same concept" —
down-weighting controls dominance but not discrimination. Now each surviving bridge source also gets a
random wiki node it is *not* bridged to, at μ≈0.1 (SYM, `rel=bridge_neg`, **full-weight**). On the two
neighbourhoods: 92 negative pairs → 184 targets.

### e5-prior review gate (`--e5-cache`, `--bridge-min-cos`)
The frozen e5 already gives a similarity prior, so a bridge whose endpoints are *far* in e5 is suspect — a
bad link, or a non-obvious synonym e5 can't see — exactly the case to **quarantine for LLM review before
training** rather than feed a possibly-wrong μ=0.9. e5 cosines are **compressed** (here min 0.784, median
0.874, max 1.0), so the threshold is e5-calibrated (default 0.80 ≈ suspect bottom decile). It quarantined
**8** bridges to `<out>_bridge_review.tsv` — a clean mix of genuinely-questionable (`ladybird-of-szeged`,
`norbit`) and legit-but-non-obvious (`BELBIC` a real control architecture; `towards-a-new-socialism` =
cybernetic planning) — precisely the set worth an LLM call. The LLM adjudication itself is **deferred (Haiku
budget-gated)**; the gate produces the candidate list for free.

### Honest A/B (fine-tune ±negatives+quarantine, both `--use-nodetype`)
| metric | no neg | +neg +quarantine |
|---|---|---|
| discrimination | 94% | 92% |
| SYM held-out | +0.838 | +0.832 |
| ELEM corr | +0.702 | **+0.731** |
| graded WIKI fit (r) | +0.757 | **+0.817** |

Headline metrics are **flat within noise** (±1 discrimination example). This *confirms* the earlier read that
**bridge down-weighting was already sufficient** — negatives are cheap, sound insurance (they stop "all
cross-corpus pairs look similar") but don't move the needle at this scale. (The graded-SYM held-out r is not
comparable across the two: the negatives make that held-out set wider/harder.) The e5-prior **quarantine** is
the clearer, training-independent win — a data-quality gate.

## Next
- **LLM-review the quarantined bridges** (budget-gated) → re-include the legit non-obvious ones, drop the
  bad links. Re-measure as the bridge set grows (negatives may matter more at larger scale).
- Wire `account` + the `Team <name> <id>` e5-text once the `s243a_groups` account is harvested.
