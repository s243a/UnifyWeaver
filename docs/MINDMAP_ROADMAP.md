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

## Phase 1: Image Generation

**Goal:** Render mind maps to images for visualization and LLM-assisted refinement.

### 1.1 Basic Rendering (Straight Lines)
- [ ] Render nodes as rounded rectangles with text
- [ ] Draw edges as straight lines between node centers
- [ ] Match our crossing detection model
- [ ] Output formats: PNG, SVG

### 1.2 Cubic Bezier Rendering (SimpleMind-style)
- [ ] Implement curve rendering matching SimpleMind's behavior:
  - Start tangent: radial from parent node center
  - End tangent: axis-aligned (vertical or horizontal)
  - Control point orientation switches at 45° boundary between nodes
- [ ] Configurable curve tension

### 1.3 Rendering Options
- [ ] Node colors by palette/cluster
- [ ] Font scaling based on node importance
- [ ] Configurable canvas size and zoom
- [ ] Optional: edge labels, icons

### 1.4 LLM Integration (Optional)
- [ ] Generate image → send to multimodal LLM
- [ ] Parse layout suggestions
- [ ] Apply adjustments iteratively

---

## Phase 2: Multi-Format Export

**Goal:** Support various mind map applications beyond SimpleMind.

### 2.1 VUE (Tufts Visual Understanding Environment)
- [ ] XML format with `<child>` nodes
- [ ] `<shape xsi:type="roundRect"/>` for node shapes
- [ ] `<resource>` elements for URL links
- [ ] Layer support

### 2.2 GraphML
- [ ] Standard graph format for yEd, Gephi, etc.
- [ ] Node/edge attributes
- [ ] Preserves layout coordinates

### 2.3 OPML (Outline Format)
- [ ] Simple hierarchical outline
- [ ] Compatible with many outline/note apps
- [ ] Loses layout but preserves structure

### 2.4 Other Formats (Future)
- [ ] Mind42 (cloud-based)
- [ ] FreeMind/Freeplane (`.mm`)
- [ ] Mermaid (text-based diagrams)
- [ ] Markmap (markdown to mindmap)

---

## Phase 3: Visual Distinction (Pearls vs Trees)

**Goal:** Visually differentiate folder nodes (Trees) from page nodes (Pearls).

### Design Options to Explore (with LLM assistance)

#### Option A: Shape-Based
- Trees: Rounded rectangle (current)
- Pearls: Square, circle, or diamond

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

## Phase 4: Cross-Cluster Linking

**Goal:** Link related mind maps using relative paths, creating a navigable hierarchy.

### 4.1 Relative Links (`cloudmapref`)
- [ ] Generate `<link cloudmapref="./subfolder/child.smmx"/>` for Tree children
- [ ] Create directory structure mirroring Pearltrees hierarchy
- [ ] Batch generation of linked mindmaps

### 4.2 Parent Links
- [ ] Attach parent folder link to root node
- [ ] Square-shaped "parent" indicator node
- [ ] Configurable link style

### 4.3 "See Also" Relations
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

## Future Ideas

- **LLM-Guided Hierarchy**: Use LLM to suggest alternative groupings
- **Semantic Relations**: Auto-detect "see also" candidates from embeddings
- **Collaborative Editing**: Export to formats supporting real-time collaboration
- **Animation**: Generate animated layout optimization visualization
- **3D Layout**: Explore 3D mind map representations

---

## Priority Summary

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|--------------|
| 1. Image Generation | High | Medium | None |
| 2. Multi-Format Export | Medium | Medium | None |
| 3. Visual Distinction | Medium | Low-Medium | Phase 1 (for LLM feedback) |
| 4. Cross-Cluster Linking | Medium | Medium | Phase 3 (visual clarity) |
| 5. Advanced Optimization | Low | High | None |
