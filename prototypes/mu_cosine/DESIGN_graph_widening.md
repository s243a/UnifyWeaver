# Category-graph widening — extraction spec

*The current benchmark is a **Physics-rooted 10k-article slice of Simple English Wikipedia** (8,247 of
simplewiki's ~76,740 categories). It is physics-dense and thin/absent in Math / CS / Engineering / AI — a
**data-coverage ceiling** that bounds the μ model (discrimination), the sampler (drift, thin pools), and
the bridge detector (`τ_pure` tuned to this slice). Widening the graph is the single move that unblocks
all three. This spec is the **how**; it's forward-looking — the source DB isn't currently in the repo.*

## Why now

Every recent result hit the *data* ceiling, not a *method* ceiling:
- μ model: 4-domain discrimination 18/20, but **Math 3/5** purely because its pool is 9 nodes vs Physics 38
  (`REPORT_4roots.md`); AI is **absent** (no anchor exists); the `Applied_sciences → CS → AI` spine is
  truncated in the slice.
- sampler: `desc(Physics) ≈ the whole 8.2k slice` (it was crawled from Physics), so "the subtree" isn't a
  clean domain — every domain leans on `depth-bound ∩ μ-coherence` to stay separable.
- bridge detector: in-domain-fraction purities are tiny (~0.001–0.04) *because* `|desc|` ≈ the whole
  exploded slice, forcing a slice-specific `τ_pure`.

A wider, multi-rooted, domain-balanced graph fixes the root cause of all three.

## Source

- **Simple English Wikipedia categories DB** (`data/simplewiki/simplewiki_categories.db`, ~76,740
  categories) — referenced in `metadata.json` but **not in the repo**. Widening needs it back in place (or
  regenerated from a simplewiki dump). The existing slices (300/1k/5k/10k articles) were extracted from it.
- **Trade-off — simplewiki is shallow.** Even its *full* graph may cover AI/CS thinly and may lack
  `Subfields_of_computer_science` (the slice does). So: start by widening *within simplewiki*; if AI/CS/
  engineering coverage is still too thin, escalate the extraction to **full English Wikipedia** (much
  bigger, much deeper, heavier to process). The provenance token (just added to `MuAttention`) is what lets
  simplewiki and enwiki data be **mixed** — tag each label with its corpus.

## The key change: multi-root + balanced extraction

The current gaps come from a **single Physics root + N-nearest-articles** crawl. Fix:

1. **Multi-root seeding.** Seed from a *set* of domain roots, not just Physics — e.g.
   `{Physics, Chemistry, Mathematics, Computer_science, Applied_sciences/Engineering, Biology}` plus
   `Branches_of_science` as the shared spine. For **AI**, explicitly include the
   `Computer_science → Subfields_of_computer_science → Artificial_intelligence → Machine_learning/Robotics`
   branch **if present in the source**; if simplewiki lacks it, that's the trigger to escalate to enwiki.
2. **Per-domain balance.** Aim for *comparable* pool sizes per root. The Math-9-vs-Physics-38 imbalance is
   exactly what biased `Calculus`/`Differential_equations` toward Physics (the bigger pool wins the
   argmax). Budget each root a comparable article/category count so no single domain dominates.
3. **Depth-bound + μ-coherence on the *extraction*** (not just the sampler) — keep each domain's region
   coherent and stop it re-exploding into one giant cross-linked blob (the `desc(Physics) ≈ whole graph`
   problem). The same discipline we apply when building sampler pools applies to choosing what to extract.
4. **Size: a ladder, bounded.** Bigger than 8.2k but kept iterable — e.g. `25k` / `50k` category tiers
   alongside a small `dev`, mirroring the existing 300/1k/5k/10k ladder, so fast iteration survives.

## Output (drop-in, no pipeline changes)

Emit the **same files/format** as the current slices so `dense_mu_direct.py`, `mu_attention.py`,
`gen_multidomain_pairs.py`, and the Rust bridge detector all run unchanged:
- `category_parent.tsv` (`child<TAB>parent`), `article_category.tsv`, `reference_output.tsv`,
- `root_categories.tsv` — now listing **all** seed roots,
- `metadata.json` — record `source`, the **root list**, per-root counts, `n_categories`, `max_depth`.

## Downstream recalibration (rides with the widening)

- **Bridge detector:** re-derive `τ_pure` from the new graph's purity distribution (the self-calibration
  task) — the slice-specific 0.01 won't transfer.
- **Sampler:** rebuild the per-domain μ-coherence pools (now *balanced*); the depth-balanced bidirectional
  walk (`gen_mu_pairs.py --bidir`) and `--child-only` modes apply unchanged.
- **μ model:** regenerate the dense μ map; retrain; tag the new labels' **provenance** (corpus = the new
  slice) so a model trained on mixed slices conditions on / marginalizes over the source.

## Validation after extraction

- **Per-domain coverage + balance:** every target anchor present; pool sizes comparable across domains
  (Math no longer 4× smaller than Physics).
- **The AI path intact** (if targeting AI): `Computer_science → … → Artificial_intelligence` reachable; else
  record that simplewiki is insufficient and escalate to enwiki.
- **Directed-cyclicity stays low** (the slice is ~99.4% acyclic now — keep it a clean DAG so the cone /
  ancestor / IC machinery holds; the "95% cyclic" earlier claim was undirected and wrong).
- **No re-explosion:** `desc(root)` for each domain root should be a *coherent* fraction of the graph, not
  ~everything.

## Open decision

**Simplewiki-wider vs enwiki.** Widening within simplewiki is the cheap first step and keeps the corpus
homogeneous, but Simple English is shallow — AI/CS/engineering may stay thin. Full enwiki gives real depth
but is a heavier extraction and a different corpus (hence the provenance token, so the two can be mixed
rather than chosen). Recommended order: **widen within simplewiki first** (multi-root, balanced), measure
whether Math/CS/Engineering pools are now adequate, and escalate to enwiki only for the domains that remain
too thin (likely AI).
