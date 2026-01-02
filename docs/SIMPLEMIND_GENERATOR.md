# SimpleMind Mind Map Generator

Generate SimpleMind-compatible mind maps (`.smmx` files) from Pearltrees clusters using semantic embeddings for hierarchical organization.

## Overview

The generator creates visual mind maps from Pearltrees bookmark collections by:
1. Loading items from a cluster (folder) in the JSONL training data
2. Matching items to pre-computed embeddings for semantic similarity
3. Building a hierarchy using K-means micro-clustering (4-8 children per node)
4. Applying radial layout with level-appropriate spacing
5. Force-directed optimization to eliminate overlaps
6. Edge crossing minimization for cleaner visual layout
7. Outputting SimpleMind XML format (`.smmx` zip file)

## Usage

```bash
# Basic usage - by cluster URL
python3 scripts/generate_simplemind_map.py \
    --cluster-url "https://www.pearltrees.com/s243a/differential-geometry/id11563719" \
    --data reports/pearltrees_targets_full_pearls.jsonl \
    --output output/differential_geometry.smmx

# By cluster name search
python3 scripts/generate_simplemind_map.py \
    --cluster "Poles and Zeros" \
    --data reports/pearltrees_targets_full_multi_account.jsonl \
    --output output/poles_zeros.smmx

# With optimization (recommended)
python3 scripts/generate_simplemind_map.py \
    --cluster-url "..." \
    --output output/optimized.smmx \
    --optimize

# Full optimization with crossing minimization
python3 scripts/generate_simplemind_map.py \
    --cluster-url "..." \
    --output output/full_optimized.smmx \
    --optimize --minimize-crossings

# Quick pass for large clusters (fewer crossing passes)
python3 scripts/generate_simplemind_map.py \
    --cluster-url "..." \
    --output output/quick.smmx \
    --optimize --minimize-crossings --crossing-passes 3

# Output raw XML for debugging
python3 scripts/generate_simplemind_map.py \
    --cluster-url "..." \
    --output output/debug.smmx \
    --xml-only

# Visual distinction: Trees as rectangles, Pearls as ellipses
python3 scripts/generate_simplemind_map.py \
    --cluster-url "..." \
    --output output/styled.smmx \
    --tree-style rectangle \
    --pearl-style ellipse

# Recursive generation with cross-cluster links
python3 scripts/generate_simplemind_map.py \
    --cluster-url "https://www.pearltrees.com/s243a/hactivism/id10818216" \
    --output-dir output/linked_maps/ \
    --recursive \
    --max-depth 3

# Unlimited depth recursive generation
python3 scripts/generate_simplemind_map.py \
    --cluster "Hactivism" \
    --output-dir output/full_hierarchy/ \
    --recursive
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster` | - | Cluster name/title to search for |
| `--cluster-url` | - | Exact Pearltrees folder URL |
| `--data` | `reports/pearltrees_targets_full_multi_account.jsonl` | Training data JSONL |
| `--embeddings` | `models/dual_embeddings_full.npz` | Pre-computed embeddings |
| `--output` | - | Output `.smmx` file path (required unless `--recursive`) |
| `--xml-only` | false | Output raw XML instead of zip |
| `--max-children` | 8 | Max children before micro-clustering |
| `--min-children` | 4 | Min clusters when splitting |
| `--optimize` | false | Apply force-directed optimization (eliminates overlaps) |
| `--optimize-iterations` | 100 | Number of force-directed iterations |
| `--minimize-crossings` | false | Apply edge crossing minimization after force-directed |
| `--crossing-passes` | 10 | Max passes for crossing minimization |
| `--no-scaling` | false | Disable node size scaling by descendant count |
| `--tree-style` | None | Node shape for Tree items: half-round, ellipse, rectangle, diamond |
| `--pearl-style` | None | Node shape for Pearl items: half-round, ellipse, rectangle, diamond |
| `--recursive` | false | Recursively generate linked maps for child Trees |
| `--output-dir` | - | Output directory for recursive generation (required with `--recursive`) |
| `--max-depth` | unlimited | Maximum depth for recursive generation |
| `--parent-links` | false | Add "back to parent" nodes in child maps (not yet implemented) |

## How It Works

### 1. Embedding Lookup

Items are matched to embeddings using (in priority order):
- `pearl_uri` - Unique identifier for PagePearls
- `uri` - Unique identifier for Trees
- `tree_id` - Legacy identifier
- `raw_title` - Fallback for items without IDs

### 2. Micro-Clustering

When a node has more than `max_children` items:
1. Extract embeddings for all items
2. Apply K-means to create `min_children` to `max_children` clusters
3. Select the most central item in each cluster as representative
4. Recursively apply to each cluster until all nodes have ≤ `max_children`

This creates semantically coherent groupings rather than arbitrary divisions.

### 3. Radial Layout

Nodes are positioned in concentric rings:
- **Level 0**: Root at center
- **Level N**: Radius calculated to maintain `min_spacing` between nodes

Formula: `radius = cumulative(n_nodes * min_spacing / (2 * pi))`

Each node's children occupy an angular sector proportional to their position among siblings.

### 4. Force-Directed Optimization (`--optimize`)

Physics simulation to eliminate node overlaps:

- **Mass-based repulsion**: Hub nodes (many descendants) push apart more strongly
- **Connection-aware**: Parent-child pairs repel weakly (0.3x), non-connected pairs repel strongly (1.5x)
- **Tethered leaves**: Leaf nodes stay close to parents (attraction inversely proportional to mass)
- **Result**: Zero overlapping nodes while preserving semantic clustering

### 5. Edge Crossing Minimization (`--minimize-crossings`)

Reduces visual edge crossings for cleaner layout:

- **Per-node adjustment**: Nodes processed by depth (shallowest first)
- **Angular/radial search**: Tries ±60° rotations and distance adjustments
- **Sibling edge detection**: Heuristic for curved line crossings from same parent
- **Iterative refinement**: Multiple passes until no improvement

**Performance note**: O(n²) complexity per pass. For large clusters (200+ nodes), use `--crossing-passes 3` for faster results.

### 6. Node Sizing

Nodes are scaled by descendant count:
- **Leaf nodes**: 1.2x base font
- **Hub nodes**: Up to 3.5x base font
- **Formula**: `scale = 1.2 + log2(1 + descendants) * 0.4`

### 7. Text Wrapping

Long titles are wrapped with `\N` line breaks targeting a 2:1 width-to-height ratio for rounder node shapes.

## Output Format

The `.smmx` file is a ZIP archive containing `document/mindmap.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE simplemind-mindmaps>
<simplemind-mindmaps generator="UnifyWeaver" gen-version="1.0.0" doc-version="3">
  <mindmap>
    <meta>
      <guid guid="..."/>
      <title text="Cluster Name"/>
      <style key="system.soft-palette"/>
    </meta>
    <topics>
      <topic id="0" parent="-1" x="500" y="500" text="Root Node">
        <link urllink="https://..."/>
        <style><font scale="2.5" bold="True"/></style>
      </topic>
      <!-- child topics -->
    </topics>
    <relations/>
  </mindmap>
</simplemind-mindmaps>
```

## Results

Tested performance on sample clusters:

| Cluster | Nodes | Overlaps | Crossings (before) | Crossings (after) |
|---------|-------|----------|-------------------|-------------------|
| people | 47 | 0 | 5 | 0 |
| graphlpane | 43 | 0 | 6 | 1 |
| differential_geometry | 225 | 0 | 147 | 29 |

## Data Requirements

### Training Data (JSONL)

Each line contains:
```json
{
  "type": "PagePearl",
  "raw_title": "Article Title",
  "uri": "https://...",
  "pearl_uri": "https://...#item123",
  "cluster_id": "https://parent-folder-url",
  "tree_id": "123456"
}
```

### Embeddings (NPZ)

Generated by `scripts/generate_dual_embeddings.py`:
- `input_nomic`: 768-dim Nomic embeddings
- `titles`: Raw titles for lookup
- `tree_ids`: Tree identifiers
- `uris`: Unique URIs (optional, for precise matching)

## Known Limitations

1. **Curved Line Approximation**: Edge crossing detection uses straight lines; SimpleMind renders curves
2. **Sibling Crossings**: Heuristic detection may miss some curved line crossings
3. **Large Clusters**: Crossing minimization is O(n²); use fewer passes for 200+ nodes
4. **Title Matching**: Without URIs in embeddings, relies on exact title match

## Image Rendering

Render `.smmx` files to SVG or PNG images for visualization, sharing, or LLM-assisted layout refinement.

### Usage

```bash
# Basic SVG output
python3 scripts/render_mindmap.py input.smmx output.svg

# With SimpleMind-style curved edges
python3 scripts/render_mindmap.py input.smmx output.svg --curves

# PNG output (requires cairosvg)
python3 scripts/render_mindmap.py input.smmx output.png --curves

# Custom canvas size
python3 scripts/render_mindmap.py input.smmx output.svg --width 2000 --height 1500

# Override node style for all nodes
python3 scripts/render_mindmap.py input.smmx output.svg --node-style ellipse
```

### Render Options

| Option | Default | Description |
|--------|---------|-------------|
| `--curves` | false | Use cubic Bezier curves (SimpleMind-style) |
| `--straight` | true | Use straight lines (matches crossing detection) |
| `--width` | auto | Output width in pixels |
| `--height` | auto | Output height in pixels |
| `--scale` | 1.0 | Scale factor for PNG output |
| `--node-style` | half-round | Override node shape for all nodes |

### Node Styles

The renderer supports SimpleMind's borderstyle values:

| Style | Description | SimpleMind Value |
|-------|-------------|------------------|
| `half-round` | Rounded corners, rx = min(w,h)/6 | `sbsHalfRound` (default) |
| `ellipse` | Full oval, rx = w/2, ry = h/2 | `sbsEllipse` |
| `rectangle` | Sharp corners, rx = 0 | `sbsRectangle` |
| `diamond` | 45° rotated square | `sbsDiamond` |

By default, the renderer uses each node's `borderstyle` from the `.smmx` file. Use `--node-style` to override all nodes with a single style.

### Curve Rendering

Cubic Bezier curves match SimpleMind's visual style:
- **Start tangent**: Radial from parent node center
- **End tangent**: Tangent to nearest reference line (0°, 45°, 90°, 135°, etc.)
- **Control point distance**: 40% of edge length

This creates smooth curves that arrive at nodes from predictable directions.

### PNG Conversion

PNG output requires `cairosvg`:

```bash
pip install cairosvg
```

Alternative: `svglib` + `reportlab`:

```bash
pip install svglib reportlab
```

---

## Multi-Format Export

Convert `.smmx` files to other mind map and graph formats:

```bash
# OPML - outline format for outline editors, RSS readers
python3 scripts/export_mindmap.py input.smmx output.opml

# GraphML - graph format for yEd, Gephi, Cytoscape
python3 scripts/export_mindmap.py input.smmx output.graphml

# VUE - Tufts Visual Understanding Environment
python3 scripts/export_mindmap.py input.smmx output.vue

# FreeMind - free desktop mind mapping software
python3 scripts/export_mindmap.py input.smmx output.mm
```

### Supported Formats

| Format | Extension | Applications |
|--------|-----------|--------------|
| OPML | `.opml` | OmniOutliner, Dynalist, Workflowy, SimpleMind, RSS readers |
| GraphML | `.graphml` | yEd, Gephi, Cytoscape, NetworkX |
| VUE | `.vue` | Tufts Visual Understanding Environment |
| FreeMind | `.mm` | FreeMind, Freeplane, Mind42 |

### Format Notes

- **OPML**: Hierarchical outline only (no layout coordinates). Best for text-based outline tools.
- **GraphML**: Preserves node positions and colors. yEd-compatible with full layout.
- **VUE**: Preserves positions and colors. Free academic tool with good concept mapping features.
- **FreeMind**: Hierarchical with left/right positioning. Works with free desktop mind mappers.

---

## Cross-Cluster Linking

Generate linked mind maps for an entire Pearltrees hierarchy using `--recursive`.

### How It Works

1. **Recursive Generation**: Starting from a root cluster, the generator creates `.smmx` files for every child Tree recursively.

2. **Deterministic GUIDs**: Each node gets a deterministic GUID based on `hash(cluster_url + "_" + node_id)`. This allows creating links to nodes before their map is generated.

3. **cloudmapref Links**: Tree nodes (folders) include `cloudmapref` attributes pointing to their child `.smmx` files:
   ```xml
   <link urllink="https://..." cloudmapref="./id12345.smmx" element="BASE64_GUID"/>
   ```

4. **File Naming**: Output files are named by tree_id (e.g., `id10818216.smmx`) for consistent, unique filenames.

### Example Output Structure

```
output/linked_maps/
  id2492416.smmx      # Root: "Hacktivism's Literacy"
  id2595428.smmx      # Child: "FR"
  id2595429.smmx      # Child: "EN"
  id2596001.smmx      # Grandchild: "Some topic"
  ...
```

### Navigation in SimpleMind

After opening a generated `.smmx` file in SimpleMind:
- Click on a Tree node (folder) to see both URL link and mind map link options
- The mind map link opens the child cluster's `.smmx` file
- The `element` attribute jumps directly to the linked node

### Depth Control

Use `--max-depth` to limit recursion:
- `--max-depth 0`: Only root cluster
- `--max-depth 1`: Root + immediate children
- `--max-depth 3`: Three levels deep
- No flag: Unlimited depth (generates entire hierarchy)

---

## Integration with SimpleMind

The generated `.smmx` files can be:
1. Opened directly in SimpleMind (iOS, Android, macOS, Windows)
2. Manually refined - drag nodes to resolve remaining issues
3. Extended with additional relationships
4. Exported to other formats (PDF, image, etc.)

## Future Work

- [x] Force-directed overlap resolution
- [x] Node sizing by descendant count
- [x] Edge crossing minimization
- [x] Connection-aware repulsion (parent-child vs non-connected)
- [x] Sibling edge crossing detection (curved line heuristic)
- [x] Image rendering (SVG/PNG) with straight and curved edges
- [x] Per-node borderstyle support (half-round, ellipse, rectangle, diamond)
- [x] Visual distinction: `--tree-style` and `--pearl-style` options
- [x] Multi-format export (OPML, GraphML, VUE)
- [x] Cross-cluster linking via `cloudmapref` (recursive generation)
- [ ] Time budget control (`--time-limit`)
- [ ] Spatial indexing for O(n log n) crossing detection
- [ ] Angular rebalancing based on subtree size
- [ ] LLM-guided layout refinement
- [ ] Parent back-links (--parent-links)

See [mindmap_layout_optimization.md](mindmap_layout_optimization.md) for algorithm details.
