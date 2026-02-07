# Proposal: Reproducible Wikipedia Physics Embedding Datasets

## Problem

The current `wikipedia_physics_nomic.npz` (300 items, 768D) and `wikipedia_physics_articles.npz`
(200 items, 768D) datasets have unknown provenance — we can't determine exactly what text was
embedded or which model version was used. Their embeddings have only 0.67 average cosine similarity
despite covering the same articles, making comparisons unreliable.

## Proposed Solution

Generate three documented embedding variants from the same 300 Wikipedia physics articles,
each embedding a different "view" of the article:

| Dataset | Embeds | Purpose |
|---------|--------|---------|
| `wikipedia_physics_nomic_titles.npz` | Article title only | Baseline, matches mindmap builder use case |
| `wikipedia_physics_nomic_text.npz` | Article content (text_preview) | Richer semantic signal |
| `wikipedia_physics_nomic_paths.npz` | Materialized Pearltrees path | Organizational/hierarchical signal |

All three use the same model (`nomic-embed-text-v1.5`), same prefix (`search_document:`),
and same normalization. A metadata field records the exact generation parameters.

### Phase 1: Titles and Text (straightforward)

Source data exists in `reports/wikipedia_physics_articles.jsonl` (300 articles with title
and text_preview). Script:

1. Load 300 titles and text_previews from JSONL
2. Embed titles with `"search_document: {title}"` → `_titles.npz`
3. Embed text with `"search_document: {text_preview}"` → `_text.npz`
4. Save with metadata: model name, prefix, timestamp, source file hash

### Phase 2: Materialized Paths (complex)

Currently only 14 of 300 articles have direct path matches in the training data.
Generating paths for the remaining 286 requires navigating Wikipedia's category hierarchy
up to a known Pearltrees folder.

#### Challenges

**1. Multiple category paths per article**

A Wikipedia article like "Bose-Einstein statistics" belongs to multiple categories:
- Category:Quantum mechanics → Category:Physics → ...
- Category:Statistical mechanics → Category:Physics → ...
- Category:Particle statistics → ...

Each category path may lead to a different Pearltrees folder, producing multiple
valid materialized paths. The existing bridge uses effective distance with dimension
n=5 to combine these, but for embedding we need a single text representation.

**Options:**
- (a) Embed the shortest/best path only
- (b) Embed all paths concatenated (longer text, richer signal)
- (c) Embed each path separately, average the embeddings
- (d) Embed the path with lowest effective distance to the target folder

**2. Non-unique Pearltrees matches**

A Wikipedia category like "Physics" might match multiple Pearltrees folders:
- `science > Math, Physical Sciences & Engineering > Physics`
- `s243a > Subjects of learning > Physics` (if it exists)

The current bridge handles this by returning all matches with distances, but for
embedding we'd need to pick one or embed multiple.

**Options:**
- (a) Use only the "canonical" Pearltrees tree (e.g., `science` tree)
- (b) Pick the match with shortest path
- (c) Embed all matches, let the model learn which is relevant

**3. Missing Pearltrees IDs in the path**

The materialized path format has two parts:
```
/10388356/11110453/10647426        ← Pearltrees numeric IDs
- science
  - Math, Physical Sciences & Engineering
    - Physics
      - Atomic Physics            ← human-readable hierarchy
```

For articles matched via Wikipedia categories (not direct Pearltrees entries),
we don't have Pearltrees IDs for the article-level node. We could:
- (a) Omit the ID line entirely (only embed the hierarchy text)
- (b) Use `?` or `*` as placeholder for unknown IDs
- (c) Use the parent folder's ID path + `/?` for the leaf

**4. Category database required**

Walking the Wikipedia category hierarchy requires `categorylinks.db` (built by
`scripts/fetch_wikipedia_categories.py` from a Wikipedia SQL dump). This is a
large download (~700MB compressed) and may not be available on all machines.

**Option:** Pre-compute the paths once and store them in a JSONL, so the embedding
script only needs the pre-computed paths (not the category database).

#### Recommended Approach for Phase 2

1. **Pre-compute paths** using the existing `WikipediaCategoryBridge`:
   - For each of the 300 articles, walk Wikipedia categories up to Pearltrees folders
   - Store all matched paths in `data/wikipedia_physics_article_paths.jsonl`
   - Include: article title, all matched paths, effective distances, via-categories

2. **Select best path per article**:
   - Use the path with lowest effective distance
   - If no match found, use a generic path: `- science\n  - Math, Physical Sciences & Engineering\n    - Physics\n      - {title}`

3. **Embed the hierarchy text only** (no Pearltrees IDs):
   - The ID line is numeric and unlikely to carry semantic meaning for the embedding model
   - The hierarchy text captures the organizational structure

4. **Save with full provenance**:
   - Which path was selected and why
   - All alternative paths in metadata

### Existing datasets (keep as reference)

Rename for clarity:
- `wikipedia_physics_nomic.npz` → keep as-is (referenced in Flask API code)
- `wikipedia_physics_articles.npz` → keep as-is (referenced in Flask API code)
- Both marked as "unknown provenance" in documentation

### Phase 3: Embedding Blending (future)

Different embedding views (titles, text, paths, projected) capture different
aspects of the same articles. Rather than choosing one view, blending allows
combining them with a tunable ratio:

```
blended = α * embedding_A + (1 - α) * embedding_B
```

Where `α ∈ [0, 1]` controls the mix. After blending, L2-normalize.

**Use cases:**
- **Layout blending**: Blend title embeddings (categorical structure) with text
  embeddings (content similarity) for a 2D layout that balances taxonomy and
  topic — e.g., 70% title + 30% text keeps the hierarchical hub structure
  while pulling content-related articles slightly closer.
- **Tree blending**: Blend projected embeddings (organizational distance) with
  raw embeddings (semantic distance) for tree construction — smoothly
  transition between purely organizational and purely semantic trees.

**Challenges:**
- Blending requires embeddings to be in the same space and dimension. Title
  and text embeddings from the same model (nomic-embed-text-v1.5, 768D) can
  be blended directly. Projected embeddings (64D from Bivector model) cannot
  be blended with raw 768D embeddings without projection.
- The blend ratio is an additional parameter for every operation (layout + tree),
  increasing UI complexity. A single slider per purpose (layout blend, tree
  blend) is manageable, but the interaction between dataset choice, model
  choice, and blend ratio creates many combinations.
- Blending in the embedding space is not the same as blending in the distance
  space. For tree construction, blending distances
  (`d_blend = α * d_A + (1-α) * d_B`) may be more principled than blending
  embeddings and computing distances on the blend.

**Recommended approach:** Start with a single blend slider for layout (title ↔ text),
keeping tree distances unblended. This covers the most useful case (tuning the
2D visualization between categorical and semantic layouts) without combinatorial
complexity. Add tree blending later if needed.

## Open Questions

1. Should Phase 2 use all 2198 category entries or just the 300 article subset?
2. For articles with no category path at all, should we use a fallback template
   or exclude them from the paths dataset?
3. Should the embedding script also generate a 200-item subset of each variant
   (to match the mindmap builder's article set)?
4. Is `nomic-embed-text-v1.5` the right model, or should we use v1 for consistency
   with the `generate_nomic_embeddings.py` script?
5. For blending, should the API accept a blend ratio parameter, or should the
   client compute the blended embeddings and send them via `/api/compute/from-embeddings`?
