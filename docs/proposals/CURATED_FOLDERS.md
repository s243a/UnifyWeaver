# Curated Folders: Hierarchy-Preserving Folder Organization

## Problem Statement

When generating mind maps from hierarchical bookmark systems (Pearltrees, browser folders, etc.), we face conflicting requirements:

1. **Deep nesting**: Real user hierarchies can be 20+ levels deep
2. **Many clusters**: Per-tree clustering produces thousands of unique clusters (e.g., 5,821)
3. **Filesystem limits**: Path length limits, too many files per folder
4. **Navigation**: Users expect to start from their actual root, not a computed "semantic center"

Current `--mst-folders` uses semantic similarity to organize folders and picks the cluster closest to the global centroid as root. This loses the user's mental model of their hierarchy.

## Proposal: `--curated-folders`

A folder organization strategy that:
1. **Preserves user's actual root** as the filesystem root
2. **Groups similar trees** into shared folders (reduces folder count)
3. **Maintains per-tree clusters** for W matrices (optional)
4. **Uses MST with constrained root** for folder hierarchy

### Key Distinction

| Aspect | `--mst-folders` | `--curated-folders` |
|--------|-----------------|---------------------|
| Root selection | Semantic center (closest to global centroid) | User's actual root node |
| Folder grouping | One folder per MST node | Multiple trees per folder (K-clustered) |
| Hierarchy source | Computed from embeddings | Derived from user's folder structure |
| W matrices | Per MST cluster | Per-tree OR per-folder (configurable) |

## Algorithm

### Phase 1: Build User Hierarchy Tree

```
Input: JSONL with cluster_id (parent) relationships
Output: Tree structure rooted at user's actual root

1. Parse cluster_id → tree_id relationships
2. Build parent→children adjacency list
3. Identify root (node with no parent in dataset)
4. Validate: single connected tree (warn on orphans)
```

### Phase 2: Compute Tree Centroids

Each tree's centroid is the mean of its child item embeddings (target embeddings).

**Why centroid, not tree's own embedding?**

The target embeddings are trained on hierarchical lists:
```
- account
  - Subjects
    - Science
      - Physics
        - Item Title
```

Child items already capture the tree's path context in their embeddings. The centroid naturally represents where the tree sits in semantic space, since all children share the path prefix.

```
For each tree T in hierarchy:
    items = load_items_in_tree(T)
    embeddings = [get_target_embedding(item) for item in items]
    centroid[T] = mean(embeddings)
```

### Phase 3: K-Cluster Trees into Folder Groups

```
Input: Tree centroids, target_folder_count K
Output: Mapping tree_id → folder_group_id

Option A: K-means on centroids
    folder_groups = kmeans(centroids, K)

Option B: Hierarchical cut
    Use user's hierarchy depth to determine cuts
    Trees at depth > max_folder_depth share parent's folder

Option C: MST edge cutting
    Build MST on tree centroids
    Cut K-1 longest edges
    Each component = one folder group
```

### Phase 4: MST with Fixed Root

```
Input: Folder group centroids, user's root tree
Output: Folder hierarchy with user's root at top

1. Compute folder group centroids (mean of member tree centroids)
2. Find which folder group contains user's root tree
3. Build MST on folder group centroids
4. Re-root MST at the folder group containing user's root
5. Convert to folder paths
```

### Phase 5: Assign Trees to Folders

```
For each tree T:
    folder_group = tree_to_folder_group[T]
    folder_path = folder_group_to_path[folder_group]
    output_path = folder_path / f"{tree_id}.smmx"
```

## Configuration Options

```bash
python3 scripts/generate_simplemind_map.py \
    --cluster-url "https://..." \
    --recursive \
    --curated-folders \
    --folder-count 100 \           # Target number of folders (K)
    --folder-method kmeans \        # kmeans | hierarchical | mst-cut
    --max-folder-depth 5 \          # Flatten deeper levels
    --root-tree-id id12345 \        # Explicit root (optional, auto-detect)
    --output-dir output/
```

## W Matrix Options

The curated folder organization is **independent** of W matrix clustering:

| W Matrix Mode | Clusters | Folders | Use Case |
|---------------|----------|---------|----------|
| `--cluster-method per-tree` | 5,821 (one per tree) | ~100 (grouped) | Maximum precision |
| `--cluster-method per-folder` | ~100 (one per folder) | ~100 (same) | Reduced model size |
| `--cluster-method mst` | Variable (MST cuts) | ~100 (grouped) | Balance |

This allows:
- High-precision per-tree W matrices with manageable folder count
- Or matching W clusters to folders for simplicity

## Folder Naming

Within curated folders, trees are named by their tree_id:

```
output/
  root/                          # User's actual root
    root.smmx
    science/                     # Folder group (multiple trees)
      id12345.smmx              # Physics tree
      id12346.smmx              # Chemistry tree
      id12347.smmx              # Biology tree
    humanities/
      id23456.smmx
      id23457.smmx
```

Future: LLM-generated mnemonic folder names based on member tree titles.

## Parent Links Interaction

`--parent-links` continues to follow the **actual user hierarchy**, not the folder structure:

```
File: output/science/id12345.smmx (Physics)
Parent link: → output/root/root.smmx (actual Pearltrees parent)

NOT: → output/science/science.smmx (folder group parent)
```

This preserves navigation semantics while organizing files for manageability.

## Implementation Phases

### Phase A: Design & Documentation (this PR)
- [ ] Design document
- [ ] Algorithm specification
- [ ] Configuration options

### Phase B: Core Implementation
- [ ] Build user hierarchy from JSONL
- [ ] K-clustering of trees into folder groups
- [ ] MST with fixed root

### Phase C: Integration
- [ ] `--curated-folders` flag in generate_simplemind_map.py
- [ ] Folder naming options
- [ ] Tests with real Pearltrees data

### Phase D: W Matrix Integration (Optional)
- [ ] `--cluster-method per-folder` option
- [ ] Folder-aligned W matrices in federated training

## Design Decisions

### Orphan Handling

Trees not connected to the main hierarchy go to a separate `_orphans/` folder:

```
output/
  root/
    ...
  _orphans/                      # Disconnected trees
    id99999.smmx
    id99998.smmx
```

This keeps the main hierarchy clean while preserving all data.

### Cross-Account Mode

Configurable via `--account-mode`:

| Mode | Flag | Behavior |
|------|------|----------|
| **Primary account** | `--account-mode primary` | Start from designated primary account (default: first in dataset). Other accounts' trees accessed via cross-account links are included but hierarchy follows primary. |
| **Unified root** | `--account-mode unified` | Single synthetic root with each account as top-level folder. |
| **Per-account** | `--account-mode separate` | Separate output directories per account. |

**Current embeddings context**: The hierarchical lists use `s243a` as the principal account, with the tree following links across accounts. This maps naturally to `--account-mode primary`.

```bash
# Default: primary account mode
python3 scripts/generate_simplemind_map.py \
    --curated-folders \
    --primary-account s243a \
    --output-dir output/

# Unified: all accounts under one root
python3 scripts/generate_simplemind_map.py \
    --curated-folders \
    --account-mode unified \
    --output-dir output/
```

## Open Questions

1. **Folder naming**: Currently tree_id based
   - Future: LLM mnemonics, path abbreviations

## Related Documentation

- [MINDMAP_ROADMAP.md](../MINDMAP_ROADMAP.md) - Phase 6: MST-Based Folder Organization
- [FEDERATED_MODEL_FORMAT.md](../design/FEDERATED_MODEL_FORMAT.md) - Per-tree clustering
- [TRANSFORMER_DISTILLATION.md](TRANSFORMER_DISTILLATION.md) - Per-tree vs MST comparison
