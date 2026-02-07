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

### Phase 3: Embedding Blending (partially implemented)

Different embedding views (titles, text, paths, projected) capture different
aspects of the same articles. Rather than choosing one view, blending allows
combining them with a tunable ratio.

#### Implemented: Input↔Output Space Blending

A Blend tab in the density explorer supports blending between input (raw) and
output (model-transformed) embedding spaces. Four modes are available:

1. **None** — No blending, standard behavior.
2. **Visualization** — Single slider (α ∈ [0,1]). SVD each space to 2D
   independently, normalize to same scale, blend 2D coordinates:
   `pos = α·pos_input + (1-α)·pos_output`. Tree distances unchanged.
3. **Tree Distance** — Single slider. Compute cosine distance matrices in each
   space, blend: `d = α·d_input + (1-α)·d_output`. 2D layout unchanged.
4. **Both** — Two independent sliders for layout and tree.

**Key finding:** With the Bivector Paired model (designed to preserve input
geometry for good hit@k retrieval), tree blending has minimal visible effect —
only 8/299 MST edges change between pure input and pure output. The model's
information-preserving design means input and output distance orderings are
nearly identical. Visualization blending is more effective because SVD captures
principal variance directions, which shift even when relative distances don't.

This confirms a tension: models optimized for retrieval preservation produce
similar distances (good for search, minimal tree impact), while models that
reshape distances for hierarchy (like the learned metric) would produce more
dramatic tree differences.

#### Proposed: Custom Multi-Space Blending

Cross-space blending (embedding × weights × learned metric × Wikipedia physics)
involves different dimensionalities and distance semantics. This is addressed
in a separate proposal: `docs/proposals/custom_distance_blending.md`.

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
