# Mind Map Generator Roadmap

## Completed (v1.0)

- [x] SimpleMind `.smmx` generation from Pearltrees clusters
- [x] K-means micro-clustering on embeddings (4-8 children per node)
- [x] Radial layout with level-based spacing
- [x] Force-directed optimization (zero overlaps)
- [x] Edge crossing minimization
- [x] Node sizing by descendant count
- [x] Connection-aware repulsion (parent-child vs non-connected)

---

## Phase 1: Image Generation ✓

**Goal:** Render mind maps to images for visualization and LLM-assisted refinement.

### 1.1 Basic Rendering (Straight Lines) ✓
- [x] Render nodes as rounded rectangles with text
- [x] Draw edges as straight lines between node centers
- [x] Match our crossing detection model
- [x] Output formats: PNG, SVG

### 1.2 Cubic Bezier Rendering (SimpleMind-style) ✓
- [x] Implement curve rendering matching SimpleMind's behavior:
  - Start tangent: radial from parent node center
  - End tangent: tangent to nearest reference line (0°, 45°, 90°, etc.)
  - Curves arrive smoothly at nodes matching SimpleMind's visual style
- [x] Control point distance: 40% of edge length

### 1.3 Rendering Options ✓
- [x] Node colors by palette (8-color SimpleMind palette)
- [x] Font scaling based on node importance
- [x] Configurable canvas size (`--width`, `--height`)
- [x] Node shape styles: half-round (default), ellipse, rectangle, diamond
- [x] Per-node borderstyle parsing from .smmx files
- [ ] Optional: edge labels, icons (future)

### 1.4 LLM Integration (Optional)
- [ ] Generate image → send to multimodal LLM
- [ ] Parse layout suggestions
- [ ] Apply adjustments iteratively

---

## Phase 2: Multi-Format Export ✓

**Goal:** Support various mind map applications beyond SimpleMind.

Export script: `scripts/export_mindmap.py`

### 2.1 OPML (Outline Format) ✓
- [x] Simple hierarchical outline
- [x] Compatible with many outline/note apps (OmniOutliner, Dynalist, Workflowy)
- [x] Preserves text and URLs (loses layout)

### 2.2 GraphML ✓
- [x] Standard graph format for yEd, Gephi, Cytoscape
- [x] Node/edge attributes (label, url)
- [x] Preserves layout coordinates (x, y)

### 2.3 VUE (Tufts Visual Understanding Environment) ✓
- [x] XML format with node/link elements
- [x] `<shape xsi:type="roundRect"/>` for node shapes
- [x] `<resource>` elements for URL links
- [x] Preserves positions and colors

### 2.4 FreeMind/Freeplane ✓
- [x] FreeMind `.mm` format
- [x] Left/right positioning based on node x-coordinates
- [x] Compatible with FreeMind, Freeplane, Mind42

### 2.5 Other Formats (Future)
- [ ] Mermaid (text-based diagrams)
- [ ] Markmap (markdown to mindmap)

---

## Phase 3: Visual Distinction (Pearls vs Trees)

**Goal:** Visually differentiate folder nodes (Trees) from page nodes (Pearls).

### 3.1 Shape-Based Distinction ✓

Implemented via `--tree-style` and `--pearl-style` options:
- [x] `--tree-style`: Set borderstyle for Tree (folder) nodes
- [x] `--pearl-style`: Set borderstyle for Pearl (page/link) nodes
- [x] Supported styles: half-round, ellipse, rectangle, diamond
- [x] Default: no distinction (uses SimpleMind default)

Example: `--tree-style rectangle --pearl-style ellipse`

### Design Options to Explore (with LLM assistance)

#### Option A: Shape-Based ✓
- Trees: Configurable (rectangle recommended)
- Pearls: Configurable (ellipse recommended)

#### Option B: Icon-Based
- Folder icon embedded in Tree nodes
- Page/link icon for Pearls

#### Option C: Structural
- Create explicit grouping nodes:
  - "URLs" parent node for Pearls
  - "Subfolders" or "Categories" parent node for Trees
- Matches manual mindmap style

#### Option D: Color Scheme
- Trees: One color palette
- Pearls: Different palette
- Configurable themes

#### Option E: Hybrid/Configurable
- User selects preferred distinction method
- `--tree-style shape|icon|group`
- `--pearl-style shape|icon|color`

### Implementation
- [ ] Define configuration schema for visual styles
- [ ] Generate sample images with each approach
- [ ] Use LLM to evaluate readability/aesthetics
- [ ] Implement chosen approach(es)

---

## Phase 4: Cross-Cluster Linking (In Progress)

**Goal:** Link related mind maps using relative paths, creating a navigable hierarchy.

### 4.1 Relative Links (`cloudmapref`) ✓
- [x] Generate `<link cloudmapref="./id12345.smmx"/>` for Tree children
- [x] Deterministic GUIDs for predictable cross-file linking
- [x] `--recursive` flag for batch generation
- [x] `--output-dir` for multi-file output
- [x] `--max-depth` to limit recursion depth
- [x] Flat directory structure (all files in one folder)
- [ ] Optional: Hierarchical directory structure mirroring Pearltrees

### 4.2 Parent Links
- [ ] `--parent-links` flag (CLI added, implementation pending)
- [ ] Attach parent folder link to root node
- [ ] Square-shaped "parent" indicator node
- [ ] Configurable link style

### 4.3 "See Also" Relations (AliasPearl/RefPearl)
- [ ] Include AliasPearl nodes with cloudmapref links
- [ ] SimpleMind `<relations>` section for non-hierarchical connections
- [ ] Cross-reference related clusters
- [ ] Optional relation label nodes

### 4.4 Navigation Structure
- [ ] Generate index/root mindmap linking all clusters
- [ ] Breadcrumb-style navigation nodes
- [ ] Bidirectional linking (parent ↔ child)

---

## Phase 5: Advanced Optimization

**Goal:** Improve performance and layout quality for large clusters.

### 5.1 Performance
- [ ] Spatial indexing (R-trees) for O(n log n) crossing detection
- [ ] Incremental crossing updates (only affected edges)
- [ ] Time budget control (`--time-limit`)
- [ ] Progressive refinement with checkpoints

### 5.2 Layout Quality
- [ ] Angular rebalancing based on subtree size
- [ ] Density-aware node sizing
- [ ] Better sibling ordering (minimize crossings before positioning)

### 5.3 Curved Line Crossing Detection
- [ ] Sample Bezier curves for accurate crossing detection
- [ ] Or: solve cubic intersection equations

---

## Phase 6: MST-Based Folder Organization ✓

**Goal:** Organize generated mind maps into semantically meaningful folder hierarchies.

### 6.1 MST from Cluster Centroids ✓
- [x] `--mst-folders` flag to enable MST-based subfolder organization
- [x] Compute item-based centroids (mean of item embeddings per cluster)
- [x] Build MST using cosine distance between centroids
- [x] MST root = cluster closest to global centroid
- [x] Relative `cloudmapref` paths across folders (`../sibling/id.smmx`)

### 6.2 Folder Structure Controls ✓
- [x] `--max-folder-depth N` - Limit subfolder nesting
- [x] `--min-folder-children N` - Only create subfolders if ≥N children
- [x] `--max-folder-children N` - Only first N children get subfolders

### 6.3 Dual Centroid Types (Future Proposal)

**Problem:** Current item-based centroid represents "what's directly in the cluster" but not "what the cluster leads to" (its descendants).

**Proposal:** Compute two centroid types per cluster:

1. **Item-based centroid** (current):
   ```
   centroid = mean(embeddings of items IN cluster)
   ```
   Good for: Matching cluster's direct content

2. **Descendant-based centroid** (proposed):
   ```
   centroid = weighted_mean(child_centroids, weights=item_counts)
   ```
   Good for: Representing what's deep within the hierarchy

**Challenge:** To use descendant-based centroids for MST construction, we'd need adaptive clustering - clusters would need to exchange members to optimize the MST. This is a joint optimization problem:

```
repeat until converged:
    1. Compute descendant-based centroids
    2. Build MST from centroids
    3. Re-assign cluster members to minimize MST edge costs
```

**Status:** Item-based centroid works well with fixed Pearltrees cluster membership. Descendant-based approach requires fundamentally different algorithm (future research).

**Note on W matrices:** The descendant-based centroid approach resembles hierarchical smoothing, but experimental results show minimal transforms (Procrustes alignment) outperform smoothing for W matrices in federated training. This suggests descendant-based centroids may not be the right direction for embedding alignment - the simpler item-based approach may remain preferable.

### 6.4 Per-Tree vs MST Clustering

Two clustering approaches are available for federated models used in mind map generation:

| Approach | Root Selection | Best For |
|----------|----------------|----------|
| **Per-tree** (`--cluster-method per-tree`) | User's actual Pearltrees folder | Preserving navigation structure |
| **MST/Embedding** (`--cluster-method embedding`) | Semantic center (closest to global centroid) | Optimal distillation quality |

**Per-tree clustering:**
- One cluster per Pearltrees folder (uses `cluster_id` field)
- Mind maps start from user's **real root node**
- Navigation matches user's mental model
- Lower distillation quality (depends on user's organization style)
- Useful for misfiling detection (items with low recall@k may be poorly filed)

**MST/Embedding clustering:**
- Groups semantically similar items regardless of folders
- Mind maps start from computed semantic center
- Better transformer distillation (semantically coherent clusters)
- May reorganize items differently than user expects

Choose per-tree when preserving the user's actual folder structure is more important than optimal semantic organization.

---

## Future Ideas

- **LLM-Guided Hierarchy**: Use LLM to suggest alternative groupings
- **Semantic Relations**: Auto-detect "see also" candidates from embeddings
- **Collaborative Editing**: Export to formats supporting real-time collaboration
- **Animation**: Generate animated layout optimization visualization
- **3D Layout**: Explore 3D mind map representations
- **Adaptive Clustering for MST**: Joint optimization of cluster membership and tree structure

---

## Priority Summary

| Phase | Status | Effort | Dependencies |
|-------|--------|--------|--------------|
| 1. Image Generation | ✓ Complete | Medium | None |
| 2. Multi-Format Export | ✓ Complete | Medium | None |
| 3. Visual Distinction | ✓ Complete | Low-Medium | Phase 1 (for LLM feedback) |
| 4. Cross-Cluster Linking | **In Progress** | Medium | Phase 3 (visual clarity) |
| 5. Advanced Optimization | Pending | High | None |
| 6. MST-Based Folder Organization | ✓ Complete | Medium | Phase 4 (recursive generation) |
