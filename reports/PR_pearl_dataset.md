# PR: Pearl Dataset Generator & Physics Model

## Title
feat(pearltrees): Add pearl dataset generator and PagePearl support

## Summary
This PR adds the ability to generate training datasets that include **Pearls (bookmarks)** in addition to Trees (folders). This enables semantic search to find specific items like articles or web pages, not just folders.

It also includes a Fix for `PagePearl` parsing which was missing from the multi-account generator.

## Key Changes

### 1. `pearltrees_multi_account_generator.py`
- Added parsing logic for `<pt:PagePearl>` elements.
- Now correctly extracts headers/bookmarks from RDF exports.

### 2. `scripts/generate_pearl_dataset.py`
- New script to generate training data with pearls.
- Supports **Query Styles** for zero-shot prompting:
  - `locate_node({title})` -> Folders
  - `locate_url({title})` -> PagePearls (Bookmarks)
  - `locate_pearl({title})` -> Other Pearls
- Filters by path (e.g. "Physics").

### 3. Generated Physics Subset
- `reports/pearltrees_targets_physics_pearls.jsonl`
- Contains 1468 targets (268 folders, 1200 pearls).

## Validation

Trained a small model (`models/pearltrees_physics_pearls.pkl`) on the physics subset.

**Test 1:**
Query: `locate_url(Feynman Lectures on Physics)`
Result: `The Feynman Lectures on Physics` (Rank 1)

**Test 2:**
Query: `locate_url(Category:Energy properties)`
Result: `Category:Energy properties` (Rank 1)

### Observations on Query Steering
The query styles effectively steer the search intent:

| Query Template | Top Result | Composition of Top 5 | Behavior |
| :--- | :--- | :--- | :--- |
| `locate_url(...)` | **Bookmark** | 5 Bookmarks | Targets content pages/bookmarks. |
| `locate_node(...)` | **Bookmark** | 4 Bookmarks, **1 Folder** | **Promotes Folders**. Successfully retrieved parent folder "Energy (Physics)" in top 5. |
| `locate_object(...)` | **Bookmark** | 5 Bookmarks | Zero-shot fallback (works but lower confidence). |

This demonstrates that `locate_node(...)` successfully biases the model towards retrieving structural elements (Trees), which is essential for filing tasks.

## Usage

generate dataset:
```bash
python3 scripts/generate_pearl_dataset.py \
    --account s243a data/s243a.rdf \
    --account s243a_groups data/s243a_groups.rdf \
    --filter-path "Physics" \
    --query-style locate \
    --output reports/pearltrees_targets_physics_pearls.jsonl
```

train model:
```bash
python3 scripts/train_pearltrees_federated.py \
    reports/pearltrees_targets_physics_pearls.jsonl \
    models/pearltrees_physics_pearls.pkl \
    --cluster-method path_depth
```

inference:
```bash
python3 scripts/infer_pearltrees_federated.py \
    --model models/pearltrees_physics_pearls.pkl \
    --query "locate_url(Energy)"
```

## Co-authored-by
Co-authored-by: Claude <claude@anthropic.com>
Co-authored-by: John William Creighton (s243a) <JohnCreighton_@hotmail.com>
