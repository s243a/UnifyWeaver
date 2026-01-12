# Mindmap Tools

Tools for managing SimpleMind mindmap files (.smmx) with relative linking support.

## Overview

These tools enable cross-linking between mindmaps using relative paths (`cloudmapref`), allowing navigation between related mindmaps without relying on external URLs.

## Tools

### build_index.py

Build an index mapping tree IDs to mindmap file paths.

```bash
# Build index from mindmap directory
python3 scripts/mindmap/build_index.py output/mindmaps_curated/ -o output/mindmaps_curated/index.json

# Build TSV index (awk-friendly)
python3 scripts/mindmap/build_index.py output/mindmaps_curated/ -o output/mindmaps_curated/index.tsv
```

The index maps tree IDs (e.g., `"10390825"`) to relative file paths (e.g., `"Gods_of_Earth_and_Nature/id10390825.smmx"`).

### add_relative_links.py

Add `cloudmapref` links to existing mindmaps using the index.

```bash
# Preview changes (dry-run)
python3 scripts/mindmap/add_relative_links.py output/mindmaps_curated/*.smmx \
  --index output/mindmaps_curated/index.json --dry-run --verbose

# Apply changes
python3 scripts/mindmap/add_relative_links.py output/mindmaps_curated/*.smmx \
  --index output/mindmaps_curated/index.json

# Specify root directory for path computation
python3 scripts/mindmap/add_relative_links.py /tmp/repaired_map.smmx \
  --index output/mindmaps_curated/index.json \
  --root output/mindmaps_curated/
```

For each Pearltrees URL in a mindmap, looks up the corresponding local mindmap in the index and adds a `cloudmapref` attribute for local navigation.

### build_reverse_index.py

Build a reverse index tracking which mindmaps link to each mindmap (backlinks).

```bash
python3 scripts/mindmap/build_reverse_index.py output/mindmaps_curated/ \
  -o output/mindmaps_curated/reverse_index.json
```

Useful for:
- Finding all mindmaps that reference a given mindmap
- Updating links when a mindmap is renamed or moved

### rename_mindmap.py

Rename mindmap files and automatically update all `cloudmapref` references.

```bash
# Single file: auto-generate titled filename from root topic
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --titled

# Single file: explicit new name
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --new-name "Technology_10380971.smmx"

# Batch: rename all to titled format
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps_curated/ \
  --titled

# Preview changes without making them
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --titled --dry-run --verbose
```

Features:
- Generates `Title_ID.smmx` filenames from root topic text
- Scans for all mindmaps with `cloudmapref` pointing to the renamed file
- Updates relative paths (including `../../../` paths)
- Batch mode updates all references before renaming files
- Optional index update via `--index`

### index_store.py

Abstraction layer for index storage with pluggable backends:

- **JSONStore**: Human-readable, good for small indexes
- **TSVStore**: Tab-separated, awk-friendly for shell scripting
- **SQLiteStore**: Efficient for large indexes with frequent lookups
- **CachedStore**: Wrapper adding in-memory caching to any backend

```python
from mindmap.index_store import create_index_store

# Auto-detect format from extension
store = create_index_store('index.json')
store = create_index_store('index.tsv')
store = create_index_store('index.db')

# With caching
store = create_index_store('index.json', cache=True)

# Usage
path = store.get('10390825')  # Returns 'Gods_of_Earth_and_Nature/id10390825.smmx'
store.set('12345678', 'subfolder/id12345678.smmx')
```

## Integration with generate_mindmap.py

The main mindmap generator supports the `--index` option to create both URL and cloudmapref links during generation:

```bash
python3 scripts/generate_mindmap.py \
  --data reports/pearltrees_targets.jsonl \
  --cluster-url "https://www.pearltrees.com/s243a/people/id10390825" \
  --output output/mindmaps_curated/id10390825.smmx \
  --index output/mindmaps_curated/index.json
```

When `--index` is provided:
- Main nodes get `urllink` to Pearltrees website
- Child nodes (square, unlabeled) get `cloudmapref` to local mindmap
- Only trees found in the index get cloudmapref links

You can override this behavior with `--url-nodes`:
- `--url-nodes url` (default with index): URL on main, cloudmapref on child
- `--url-nodes map`: cloudmapref on main, URL on child
- `--url-nodes url-label`: cloudmapref on main, URL on labeled child

## Workflow

### Initial Setup

1. Generate mindmaps recursively:
   ```bash
   python3 scripts/generate_mindmap.py \
     --cluster-url "https://www.pearltrees.com/s243a" \
     --recursive --output-dir output/mindmaps_curated/
   ```

2. Build the index:
   ```bash
   python3 scripts/mindmap/build_index.py output/mindmaps_curated/ \
     -o output/mindmaps_curated/index.json
   ```

3. Add relative links to all mindmaps:
   ```bash
   python3 scripts/mindmap/add_relative_links.py output/mindmaps_curated/**/*.smmx \
     --index output/mindmaps_curated/index.json
   ```

### Regenerating a Single Mindmap

When regenerating a mindmap, use the index to include relative links:

```bash
python3 scripts/generate_mindmap.py \
  --cluster-url "https://www.pearltrees.com/s243a/science/id12345678" \
  --output output/mindmaps_curated/science/id12345678.smmx \
  --index output/mindmaps_curated/index.json
```

### Handling Renames/Moves

Use `rename_mindmap.py` to rename a mindmap and automatically update all references:

```bash
# Rename with auto-generated title from root topic
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --titled --dry-run

# Rename with explicit new name
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --new-name "Technology_10380971.smmx"

# Batch rename all mindmaps to titled format (Title_ID.smmx)
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps_curated/ \
  --titled --dry-run
```

The tool:
1. Finds all mindmaps with `cloudmapref` pointing to the renamed file
2. Updates their relative paths to the new filename
3. Renames the file
4. Optionally updates the index

## Folder Organization Tools

Tools for semantic-based folder organization suggestions.

### build_folder_projections.py

Build folder projection matrices for semantic similarity scoring.

For each folder, computes:
- **Centroid**: Average embedding (for fast top-k filtering)
- **W matrix**: Procrustes projection (for precise fit scoring)

```bash
# Full rebuild (~10 seconds for 66 folders)
python3 scripts/mindmap/build_folder_projections.py \
  --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
  --index output/mindmaps_curated/index.json \
  --output output/mindmaps_curated/folder_projections.db

# Incremental (only changed folders)
python3 scripts/mindmap/build_folder_projections.py \
  --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
  --index output/mindmaps_curated/index.json \
  --output output/mindmaps_curated/folder_projections.db \
  --incremental

# Specific folders only
python3 scripts/mindmap/build_folder_projections.py \
  --folders "Media_Reviews" "Economics"
```

The W matrix captures each folder's "semantic projection style" - how titles in that folder relate to their hierarchical context.

### suggest_folder.py

Suggest the best folder for a mindmap using two-stage scoring:

1. **Stage 1**: Top-k centroid filtering (cheap, ~50K ops)
2. **Stage 2**: Procrustes fit scoring (precise, evaluates k candidates)

```bash
# Suggest folder for a mindmap by tree ID
python3 scripts/mindmap/suggest_folder.py --tree-id 12345678

# With verbose scoring details
python3 scripts/mindmap/suggest_folder.py --tree-id 12345678 --verbose

# Suggest folder for a new title (embeds on the fly)
python3 scripts/mindmap/suggest_folder.py --title "Machine Learning Tutorial"

# Check all mindmaps in a folder for misplacements
python3 scripts/mindmap/suggest_folder.py \
  --check-folder output/mindmaps_curated/Economics/ \
  --threshold 0.5

# JSON output for scripting
python3 scripts/mindmap/suggest_folder.py --tree-id 12345678 --json
```

**Signal interpretation:**
| Probability | Interpretation |
|-------------|----------------|
| >50% | Strong match, clear winner |
| 25-50% | Good match, minor ambiguity |
| <25% | Ambiguous, multiple valid options |

**Example output:**
```
Tree ID: 2492215
Current folder: Hacktivism

Suggested folders:
  1. Hacktivism: 51.2% (fit=0.9667) <-- current
  2. eyes-symbols-history-s243a: 20.3% (fit=0.8743)
  3. Online_Hacktivists_groups: 18.6% (fit=0.8655)
```

### Algorithm Details

The Procrustes fit score measures how well an item's input/output embedding pair matches a folder's learned projection:

```
fit_score = cosine(input_embedding @ W_folder, output_embedding)
```

Where W_folder is the orthogonal matrix minimizing ||X @ W - Y|| for all trees in the folder (computed via SVD).

**Storage:**
- Centroids: ~200 KB (all folders in RAM)
- W matrices: ~150 MB (lazy-loaded from SQLite)
- Peak memory: ~11 MB (k=5 candidates)

### batch_rename_folders.py

Batch rename folders using LLM-generated names based on folder contents.

```bash
# Dry run - preview what would be renamed
python3 scripts/mindmap/batch_rename_folders.py \
  --base-dir output/mindmaps_curated/ \
  --dry-run --verbose

# Rename all folders with descriptions
python3 scripts/mindmap/batch_rename_folders.py \
  --base-dir output/mindmaps_curated/ \
  --descriptions output/mindmaps_curated/folder_descriptions.json

# Only rename root-level folders
python3 scripts/mindmap/batch_rename_folders.py \
  --base-dir output/mindmaps_curated/ \
  --depth 1

# Limit to first 5 folders (for testing)
python3 scripts/mindmap/batch_rename_folders.py \
  --base-dir output/mindmaps_curated/ \
  --limit 5 --dry-run
```

Features:
- Processes folders from root to leaves (parent context informs child naming)
- Uses LLM with JSONL context for semantic folder naming
- Generates abbreviated names (Corp_Intel, Econ, Tech, etc.)
- Updates all `cloudmapref` links after renaming
- Updates the mindmap index with new paths
- Saves multi-length descriptions (short/medium/long) to JSON

**Context levels** (via `--context-level`):
- `titles`: Just item titles (~100 tokens)
- `paths`: Hierarchy paths (~500 tokens)
- `jsonl`: Full JSONL records (~2K tokens, best quality)

## LLM Folder Naming

The `generate_mindmap.py` script supports LLM-based folder naming:

```bash
# Generate folder structure with LLM names
python3 scripts/generate_mindmap.py \
  --cluster-url "https://www.pearltrees.com/s243a" \
  --recursive --output-dir output/mindmaps_curated/ \
  --llm-folder-context jsonl \
  --llm-folder-descriptions output/folder_descriptions.json
```

Options:
- `--llm-folder-context`: Context level (titles/paths/full/jsonl)
- `--llm-folder-descriptions`: Save descriptions to JSON file

The LLM uses abbreviations (Corp_Intel, Econ, Tech, Govt) and considers parent folder context to avoid redundant naming

## MST Semantic Clustering

Tools for organizing mindmaps into semantically coherent folder hierarchies using Minimum Spanning Tree (MST) partitioning.

### mst_folder_grouping.py

Compute MST-based folder groupings from embeddings.

```bash
# Test on physics subset
python3 scripts/mindmap/mst_folder_grouping.py \
  --subset physics --target-size 8 --max-depth 3 --verbose

# Run on all trees
python3 scripts/mindmap/mst_folder_grouping.py \
  --trees-only --target-size 10 --max-depth 8 \
  -o output/mst_folder_structure_trees.json

# Limit for testing
python3 scripts/mindmap/mst_folder_grouping.py \
  --trees-only --limit 2000 --target-size 10 --verbose

# Use arithmetic internal cost for tighter semantic grouping
python3 scripts/mindmap/mst_folder_grouping.py \
  --trees-only --target-size 10 --max-depth 8 \
  --internal-cost arithmetic --verbose

# Use curated hierarchy (respects Pearltrees parent-child structure)
python3 scripts/mindmap/mst_folder_grouping.py \
  --subset physics --tree-source curated --target-size 8 --max-depth 3 --verbose

# Use hybrid mode (curated + greedy orphan attachment with blended embeddings)
python3 scripts/mindmap/mst_folder_grouping.py \
  --subset physics --tree-source hybrid --embed-blend 0.3 --target-size 8 --max-depth 3 --verbose
```

**Options Summary:**

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--target-size` | int | 8 | Target items per folder |
| `--max-depth` | int | 4 | Maximum folder hierarchy depth (soft constraint) |
| `--min-size` | int | 2 | Minimum items per folder |
| `--subdivision-method` | `multilevel`, `bisection` | `multilevel` | How to split oversized folders |
| `--size-cost` | `gm_maximize`, `quadratic`, `geometric` | `gm_maximize` | Size cost (gm_maximize is scale-invariant) |
| `--internal-cost` | `none`, `arithmetic`, `geometric` | `none` | Cost function for internal edges |
| `--tree-source` | `mst`, `curated`, `hybrid` | `mst` | Tree source for partitioning |
| `--embed-blend` | 0.0–1.0 | 0.3 | Blend weight for hybrid mode (0.3 = 30% input, 70% output) |
| `--stats`, `-s` | flag | off | Print statistics tables (markdown format) |
| `--verbose`, `-v` | flag | off | Print detailed progress |

**Tree Source Modes:**

| Mode | Description | Compute Cost | Best For |
|------|-------------|--------------|----------|
| `mst` | Build MST from embeddings | O(N²) or O(N*k) | Fresh organization, items without clear hierarchy |
| `curated` | Use Pearltrees parent-child hierarchy | O(N) | Respecting existing curation, enhancing structure |
| `hybrid` | Curated structure + greedy orphan attachment | O(N) | Mixed content with orphans from other accounts |

**Tree Source Details:**

- `curated`: Uses hierarchy paths from JSONL `target_text` field. Edge weights computed from embedding distances.

- `hybrid`: Combines curated hierarchy (fixed) with greedy orphan attachment:
  - Uses blended embeddings: `embed_blend * input + (1 - embed_blend) * output`
  - Orphan nodes attach to minimize semantic distance (non-binary, can attach to any node)
  - Attachment order optimized (closest orphans attached first)
  - Newly attached orphans become valid attachment points for subsequent orphans

**Tangent Deviation Metric:**

When using `curated` or `hybrid` modes, the `--stats` output includes a tangent deviation metric that measures how much the final tree differs from the original curated structure:

| Metric | Description |
|--------|-------------|
| Mean deviation | Average deviation across all nodes (0 = identical, 2 = opposite) |
| Max deviation | Worst-case node deviation |
| Nodes compared | Number of nodes with both curated and final neighbors |

The deviation is computed as `1 - cosine_similarity(t_curated, t_final)` where:
- `t_curated` = average direction to curated neighbors (tangent vector)
- `t_final` = average direction to final tree neighbors

**Intuition**: "Do the two graphs point you in the same direction at each point?" Low deviation means the final tree preserves the curated structure's local geometry.

**Subdivision Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| `multilevel` | Top-cut/bottom-cuts approach, grows bands from tree root | Flexible partitioning, uneven clusters |
| `bisection` | Removes highest-weight edges for balanced splits | Even splits, simpler structure |

**Size Cost Modes:**

| Mode | Formula | Scale-Invariant | Effect |
|------|---------|-----------------|--------|
| `quadratic` | (size - target)² | No | Fixed target, aggressive splitting. Requires tuning. |
| `gm_maximize` | split if size > 4×GM | **Yes** | Derived from maximizing GM. Threshold adapts dynamically. |
| `geometric` | *(deprecated)* | — | Incorrect implementation; use gm_maximize instead. |

**Internal Cost Modes:**

| Mode | Formula | Effect |
|------|---------|--------|
| `none` | — | No internal edge cost (fastest) |
| `arithmetic` | Σ(edge weights) | Favors tight clusters (low total distance) |
| `geometric` | n / GM(weights) | Detects uneven clusters (low GM = natural cut points) |

**Algorithm:**
1. Computes pairwise cosine distances between embeddings
2. Builds Minimum Spanning Tree (MST) from distance matrix
3. Partitions MST into "circles" (folders) using incremental growth
4. Recursively subdivides oversized circles using cost-based decisions
5. Uses soft depth penalty (exponential) to balance depth vs folder size

**Memory modes:**
- Dense (N < 5000): Full N×N distance matrix
- Sparse (N >= 5000): k-NN graph with 50 neighbors

**Output:** JSON hierarchy with folder names, tree IDs, and nested children.

**Method Comparison** (13,279 trees, target=10, max-depth=8):

| Method | Size Cost | Folders | GM | Max Size | Scale-Invariant |
|--------|-----------|---------|-----|----------|-----------------|
| Default | quadratic | 1,073 | 5.31 | 20 | No |
| GM-maximize | gm_maximize | 1,077 | 5.29 | 20 | **Yes** |

*Note: Similar results but different reasoning. Quadratic has slightly higher GM with fewer folders, but this isn't necessarily better - higher GM could just mean larger folders on average. For a balanced tree, the key is appropriate folder counts, not maximizing GM alone.*

**Tradeoffs:**
- **Quadratic**: Fixed target, aggressive splitting. Requires tuning `target_size`.
- **GM-maximize**: Scale-invariant, derived from maximizing geometric mean. Threshold `4×GM` adapts dynamically. Less parameter tuning needed.
- **Arithmetic internal**: Favors tight semantic clusters (low total edge distance).
- **Geometric internal**: Detects uneven clusters with natural cut points (low GM = outlier edges).

**Parameter Scaling Note:**
If using `quadratic` mode, the `target_size` parameter needs tuning per dataset. In principle, if you know the expected geometric mean for your dataset, you could scale the quadratic parameters to match. However, `gm_maximize` avoids this problem entirely by computing the threshold dynamically from the current folder distribution.

**Future Work:**
- **Branch permutation for better boundaries**: Currently, the MST subdivision finds cut points but doesn't optimize which subtrees go to which child folder. Branch permutation would try swapping subtrees between sibling folders to improve semantic coherence at folder boundaries. This could reduce cases where semantically similar items end up in different folders due to arbitrary cut ordering.

**Example Statistics** (13,279 trees, target=10, max-depth=8, multilevel, none):

| Metric | Value |
|--------|-------|
| Total items | 13,279 |
| Total folders | 1,073 |
| Max depth | 8 |
| Folder size range | 3–20 |
| Average folder size | 6.5 |

| Depth | Folders |
|-------|---------|
| 0 | 1 |
| 1 | 103 |
| 2 | 13 |
| 3 | 18 |
| 4 | 72 |
| 5 | 104 |
| 6 | 164 |
| 7 | 231 |
| 8 | 383 |

| Size | Folders |
|------|---------|
| 3 | 414 |
| 4 | 130 |
| 5 | 107 |
| 6 | 60 |
| 7 | 51 |
| 8 | 26 |
| 9 | 25 |
| 10 | 140 |
| 11–19 | 49 |
| 20 | 71 |

### generate_mst_mindmaps.py

Generate mindmaps organized by MST folder structure with cloudmapref cross-links.

```bash
# Generate all mindmaps
python3 scripts/mindmap/generate_mst_mindmaps.py \
  --mst-structure output/mst_folder_structure_trees.json \
  --output output/mst_mindmaps/ \
  --root-name "Pearltrees_Collection"

# Test with limit
python3 scripts/mindmap/generate_mst_mindmaps.py \
  --mst-structure output/mst_folder_structure_trees.json \
  --output output/mst_mindmaps/ \
  --limit 100

# Dry run
python3 scripts/mindmap/generate_mst_mindmaps.py \
  --mst-structure output/mst_folder_structure_trees.json \
  --output output/mst_mindmaps/ \
  --dry-run
```

**Features:**
- Extracts tree→pearl relationships from embedding URIs
- Creates one mindmap per tree (tree as root, pearls as children)
- Organizes mindmaps into MST folder hierarchy
- Adds `cloudmapref` links between sibling mindmaps in same folder
- Shows related maps with dashed lines

**Example output structure:**
```
output/mst_mindmaps/
└── Pearltrees_Collection/
    ├── Quantum_Mechanics/
    │   ├── Quantum_Mechanics_id10380971.smmx
    │   ├── Quantum_Invariant_id10381234.smmx
    │   └── ...
    ├── Thermodynamics/
    │   └── ...
    └── ...
```

### Workflow: MST-Organized Mindmaps

1. **Generate MST folder structure:**
   ```bash
   python3 scripts/mindmap/mst_folder_grouping.py \
     --trees-only --target-size 10 --max-depth 5 \
     -o output/mst_folder_structure_trees.json --verbose
   ```

2. **Generate mindmaps with cross-links:**
   ```bash
   python3 scripts/mindmap/generate_mst_mindmaps.py \
     --mst-structure output/mst_folder_structure_trees.json \
     --output output/mst_mindmaps/ \
     --root-name "My_Collection"
   ```

3. **Open in SimpleMind:**
   - Open any .smmx file
   - Click on "→ Related Map" nodes to navigate
   - Cross-links work offline via cloudmapref
