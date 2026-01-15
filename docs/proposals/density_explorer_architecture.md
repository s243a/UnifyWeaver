# Density Explorer: Multi-Approach Architecture Proposal

**Status:** Draft
**Author:** Claude
**Date:** 2026-01-14
**Branch:** `feature/density-explorer` (to be created)

---

## 1. Philosophy

### 1.1 Why This Tool?

The density manifold visualization reveals the **geometric structure of knowledge**. When we embed concepts (documents, articles, bookmarks) into high-dimensional space:

- **Clusters** emerge where related concepts group together
- **Density peaks** mark topic centers
- **Low-density regions** separate distinct domains
- **Trees** show hierarchical relationships

Understanding this structure helps us:
1. **Organize information** - See natural category boundaries
2. **Navigate knowledge** - Find related concepts by proximity
3. **Build better hierarchies** - Use density-aware tree construction (J-guided)
4. **Debug embeddings** - Spot anomalies, verify quality

### 1.2 Design Principles

1. **Shared Core, Multiple Interfaces**
   - One computation engine, many frontends
   - Same JSON format everywhere
   - Test once, deploy anywhere

2. **Progressive Enhancement**
   - Start simple (static image)
   - Add interactivity (Streamlit)
   - Full customization (Vue.js)
   - Client-side option (Pyodide)

3. **Security-Conscious**
   - Pyodide option keeps data local
   - No mandatory server for sensitive embeddings

4. **Leverage Existing Work**
   - Reuse `MindMapViewport.vue` (zoom/pan)
   - Reuse `pyodide-matrix` example (WASM loader)
   - Reuse `hierarchy_objective.py` (J-guided tree)

### 1.3 Target Users

| User | Need | Best Approach |
|------|------|---------------|
| Developer (quick look) | Fast iteration | Streamlit |
| Researcher (analysis) | Full control | Flask + Vue |
| End user (privacy) | No data upload | Pyodide |
| Embedded widget | Integration | Vue component |

---

## 2. Specification

### 2.1 Functional Requirements

#### Core Features (All Approaches)

| ID | Feature | Description |
|----|---------|-------------|
| F1 | Load embeddings | From .npz file or JSON |
| F2 | SVD projection | Project to 2D, show variance explained |
| F3 | Density heatmap | KDE with configurable bandwidth |
| F4 | Contour lines | Equipotential surfaces |
| F5 | Data points | Scatter plot with hover labels |
| F6 | Tree overlay | MST or J-guided, filterable by depth |
| F7 | Peak detection | Label density maxima |
| F8 | Export | Download JSON, PNG |

#### Interactive Features (Vue.js Approaches)

| ID | Feature | Description |
|----|---------|-------------|
| I1 | Zoom/Pan | Mouse wheel, drag |
| I2 | Click to select | Highlight node, show details |
| I3 | Branch focus | Click node → show only its subtree |
| I4 | Hide/show layers | Toggle contours, tree, points |
| I5 | Live controls | Sliders update visualization |

#### Advanced Features (Future)

| ID | Feature | Description |
|----|---------|-------------|
| A1 | Generate mindmap | Export subtree as .smmx |
| A2 | Pearltrees view | Alternative tree layout |
| A3 | Animation | Morph between bandwidth settings |
| A4 | 3D view | Three.js with 3 SVD components |

### 2.2 Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Initial load (Streamlit) | < 3 sec |
| Initial load (Pyodide) | < 15 sec |
| Interaction latency | < 100 ms |
| Max points supported | 5,000 |
| Browser support | Chrome, Firefox, Safari (modern) |

### 2.3 Data Format Specification

```typescript
interface DensityManifoldData {
  // Points in 2D projected space
  points: Array<{
    id: number;
    title: string;
    x: number;
    y: number;
  }>;

  // Density grid (row-major)
  density_grid: {
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
    grid_size: number;
    values: number[][];  // [row][col]
    bandwidth: number;
  };

  // Tree structure (optional)
  tree?: {
    nodes: Array<{
      id: number;
      title: string;
      parent_id: number | null;
      depth: number;
      x: number;
      y: number;
    }>;
    edges: Array<{
      source_id: number;
      target_id: number;
      weight: number;
    }>;
    root_id: number;
    tree_type: 'mst' | 'j-guided';
  };

  // Density peaks (optional)
  peaks?: Array<{
    x: number;
    y: number;
    density: number;
    nearest_node_id: number;
    title: string;
  }>;

  // Projection metadata
  projection: {
    variance_explained: [number, number];
    singular_values: [number, number];
  };

  n_points: number;
}
```

### 2.4 API Specification (Flask Approach)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/health` | GET | Health check |
| `POST /api/compute` | POST | Compute full manifold |
| `POST /api/density` | POST | Compute density grid only |
| `POST /api/tree` | POST | Compute tree only |
| `GET /api/embeddings/{id}` | GET | Get cached embeddings |

**POST /api/compute Request:**
```json
{
  "embeddings_path": "datasets/wikipedia_physics.npz",
  "top_k": 300,
  "bandwidth": 0.1,
  "grid_size": 100,
  "tree_type": "mst",
  "include_peaks": true,
  "n_peaks": 5
}
```

---

## 3. Implementation Plan

### 3.1 Branch Strategy

```
main
 └── feature/density-explorer
      ├── Phase 1: Shared core (merged)
      ├── Phase 2: Streamlit (PR #1)
      ├── Phase 3: Vue components (PR #2)
      ├── Phase 4: Flask API (PR #3)
      └── Phase 5: Pyodide (PR #4)
```

### 3.2 Phase Details

#### Phase 1: Shared Core ✅ (Complete)

**Goal:** Computation engine that works everywhere

**Deliverables:**
- [x] `shared/data_format.py` - Data classes, JSON schema
- [x] `shared/density_core.py` - NumPy/SciPy computation
- [x] `shared/__init__.py` - Public API

**Validation:**
```python
from shared import load_and_compute
data = load_and_compute("datasets/wikipedia_physics.npz", top_k=300)
print(data.to_json())  # Valid JSON output
```

---

#### Phase 2: Streamlit Prototype (1-2 days)

**Goal:** Working interactive demo

**Deliverables:**
- [x] `streamlit_app.py` - Basic version
- [ ] Add J-guided tree option
- [ ] Add click-to-select (st.plotly_events)
- [ ] Add gradient lines toggle
- [ ] Test with multiple datasets

**Validation:**
```bash
streamlit run tools/density_explorer/streamlit_app.py
# Opens browser, all controls work
```

---

#### Phase 3: Vue.js Components (3-5 days)

**Goal:** Reusable frontend components

**Deliverables:**
- [ ] `frontend/package.json` - Dependencies (Vue 3, Plotly, Vite)
- [ ] `DensityViewport.vue` - Main canvas with Plotly
- [ ] `TreeOverlay.vue` - SVG tree with depth filtering
- [ ] `ControlsSidebar.vue` - Settings panel
- [ ] `NodeDetails.vue` - Click-to-see info panel
- [ ] `types.ts` - TypeScript interfaces matching JSON schema

**Validation:**
```bash
cd tools/density_explorer/frontend
npm run dev
# Load mock JSON, all components render
```

---

#### Phase 4: Flask API (1-2 days)

**Goal:** REST backend for Vue frontend

**Deliverables:**
- [ ] `api/server.py` - Flask app with endpoints
- [ ] `api/cache.py` - LRU cache for computed results
- [ ] `frontend/src/lib/api.ts` - API client
- [ ] Connect frontend to backend

**Validation:**
```bash
# Terminal 1
python tools/density_explorer/api/server.py

# Terminal 2
curl -X POST http://localhost:5001/api/compute \
  -H "Content-Type: application/json" \
  -d '{"embeddings_path": "datasets/wikipedia_physics.npz"}'
# Returns valid JSON
```

---

#### Phase 5: Pyodide Version (2-3 days)

**Goal:** Client-side WASM version

**Deliverables:**
- [ ] `wasm/index.html` - Loader page with progress
- [ ] `wasm/worker.py` - Pyodide script (runs density_core)
- [ ] `frontend/src/lib/pyodide.ts` - Pyodide integration
- [ ] `frontend/src/composables/usePyodide.ts` - Vue composable
- [ ] Switch frontend between API/Pyodide modes

**Validation:**
```bash
cd tools/density_explorer/wasm
python -m http.server 8080
# Open browser, wait for Pyodide load, test computation
```

---

#### Phase 6: Comparison & Documentation (1 day)

**Goal:** Benchmark, document, decide

**Deliverables:**
- [ ] Benchmark results (load time, computation time)
- [ ] User feedback summary
- [ ] Final README with usage instructions
- [ ] Decision on primary approach

---

### 3.3 Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Shared Core | Done | - |
| 2. Streamlit | 1-2 days | Phase 1 |
| 3. Vue Components | 3-5 days | Phase 1 |
| 4. Flask API | 1-2 days | Phase 3 |
| 5. Pyodide | 2-3 days | Phase 3 |
| 6. Compare | 1 day | All |

**Total: ~10-14 days**

---

## 4. Approaches (Detailed)

### 4.1 Streamlit + Plotly (Python-centric)

```
┌─────────────────────────────────────────┐
│           Streamlit App                  │
│  ┌─────────────────────────────────────┐ │
│  │  Sidebar Controls (st.slider, etc.) │ │
│  ├─────────────────────────────────────┤ │
│  │  Plotly Figure (st.plotly_chart)    │ │
│  │  - Heatmap (density)                │ │
│  │  - Scatter (points, peaks)          │ │
│  │  - Lines (tree edges)               │ │
│  ├─────────────────────────────────────┤ │
│  │  Info Panel / Export                │ │
│  └─────────────────────────────────────┘ │
│                  │                       │
│          ┌───────▼───────┐               │
│          │  shared/      │               │
│          │  density_core │               │
│          └───────────────┘               │
└─────────────────────────────────────────┘
```

**Pros:**
- Fastest to build (mostly done)
- Python-only, familiar
- Good interactivity via Plotly
- Easy deployment (Streamlit Cloud)

**Cons:**
- Less customizable UI
- Hard to embed in other apps
- Server required

**Files:**
- `tools/density_explorer/streamlit_app.py`

---

### 4.2 Flask API + Vue.js (Service-oriented)

```
┌────────────────────┐     ┌────────────────────────────┐
│    Flask API       │     │      Vue.js Frontend       │
│  ┌──────────────┐  │     │  ┌──────────────────────┐  │
│  │ /api/compute │◄─┼─────┼──│ DensityViewport.vue  │  │
│  │ /api/tree    │  │JSON │  │ - Plotly.js heatmap  │  │
│  │ /api/peaks   │  │     │  │ - Cytoscape.js tree  │  │
│  └──────┬───────┘  │     │  │ - Controls sidebar   │  │
│         │          │     │  └──────────────────────┘  │
│  ┌──────▼───────┐  │     │  ┌──────────────────────┐  │
│  │  shared/     │  │     │  │ MindMapViewport.vue  │  │
│  │  density_core│  │     │  │ (existing, zoom/pan) │  │
│  └──────────────┘  │     │  └──────────────────────┘  │
└────────────────────┘     └────────────────────────────┘
```

**Pros:**
- Most flexible
- Reuse existing Vue components
- API can serve multiple frontends
- Full UI customization

**Cons:**
- More setup (two services)
- Need to maintain frontend build
- Server required

**Files:**
- `tools/density_explorer/api/server.py`
- `tools/density_explorer/frontend/src/`

---

### 4.3 Pyodide + Vue.js (Client-side WASM)

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (No Server)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │                 Vue.js Frontend                    │  │
│  │  ┌─────────────────┐  ┌─────────────────────────┐ │  │
│  │  │ Controls        │  │ DensityViewport.vue     │ │  │
│  │  │ (same as #2)    │  │ (same as #2)            │ │  │
│  │  └────────┬────────┘  └────────────▲────────────┘ │  │
│  │           │                        │              │  │
│  │           ▼                        │              │  │
│  │  ┌────────────────────────────────────────────┐   │  │
│  │  │           Pyodide (Python WASM)            │   │  │
│  │  │  ┌──────────────────────────────────────┐  │   │  │
│  │  │  │  shared/density_core.py              │  │   │  │
│  │  │  │  (same Python code as #1 and #2)     │  │   │  │
│  │  │  └──────────────────────────────────────┘  │   │  │
│  │  │  + NumPy, SciPy (pre-bundled)              │   │  │
│  │  └────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- No server needed
- Data never leaves browser (privacy)
- Works offline
- Leverage existing pyodide-matrix example

**Cons:**
- 5-10 sec initial load
- 3-5x slower computation
- Limited to Pyodide-compatible packages

**Files:**
- `tools/density_explorer/wasm/index.html`
- `tools/density_explorer/wasm/worker.py`
- `tools/density_explorer/frontend/src/` (shared with #2)

---

## 5. Shared Components Summary

### 5.1 Python (Backend/WASM)

| Component | File | Used By |
|-----------|------|---------|
| Data format | `shared/data_format.py` | All |
| Core computation | `shared/density_core.py` | All |
| J-guided tree | `scripts/mindmap/hierarchy_objective.py` | All |

### 5.2 Vue.js (Frontend)

| Component | Description | Used By |
|-----------|-------------|---------|
| `DensityViewport.vue` | Main visualization canvas | Flask+Vue, Pyodide |
| `TreeOverlay.vue` | SVG tree edges with filtering | Flask+Vue, Pyodide |
| `ControlsSidebar.vue` | Sliders, toggles, inputs | Flask+Vue, Pyodide |
| `NodeDetails.vue` | Click-to-see node info | Flask+Vue, Pyodide |
| `MindMapViewport.vue` | Existing zoom/pan (reuse) | Flask+Vue, Pyodide |

### 5.3 Visualization Libraries

| Library | Purpose | Used By |
|---------|---------|---------|
| Plotly.js | Heatmap, contours, scatter | All |
| Cytoscape.js | Tree/graph overlay (optional) | Flask+Vue, Pyodide |
| D3.js | Custom SVG (optional) | Flask+Vue, Pyodide |

---

## 6. Decision Criteria

| Criterion | Weight | Streamlit | Flask+Vue | Pyodide |
|-----------|--------|-----------|-----------|---------|
| Time to build | 20% | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Flexibility | 25% | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| Performance | 15% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Security/Privacy | 15% | ⭐ | ⭐ | ⭐⭐⭐ |
| Offline support | 10% | ❌ | ❌ | ⭐⭐⭐ |
| Embeddability | 15% | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**Weighted Score:**
- Streamlit: ~60%
- Flask+Vue: ~75%
- Pyodide: ~70%

Flask+Vue scores highest for flexibility and embeddability, but Streamlit is fastest to prototype. Pyodide wins on security/privacy and offline support.

---

## 7. File Structure

```
tools/density_explorer/
├── shared/                     # Shared across all approaches
│   ├── __init__.py
│   ├── data_format.py          # Data structures, JSON schema
│   └── density_core.py         # Core computation (NumPy/SciPy)
│
├── streamlit_app.py            # Approach 1: Streamlit
│
├── api/                        # Approach 2: Flask backend
│   ├── __init__.py
│   └── server.py
│
├── wasm/                       # Approach 3: Pyodide
│   ├── index.html
│   └── worker.py
│
├── frontend/                   # Shared Vue.js (Approaches 2 & 3)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.vue
│       ├── components/
│       │   ├── DensityViewport.vue
│       │   ├── TreeOverlay.vue
│       │   ├── ControlsSidebar.vue
│       │   └── NodeDetails.vue
│       └── lib/
│           ├── api.ts          # Flask API client
│           └── pyodide.ts      # Pyodide loader
│
└── README.md
```

---

## 8. Recommendation

**Start with Streamlit** to quickly validate the shared core and UX, then build Vue.js components for the more customizable approaches. This allows:

1. Fast feedback loop on visualization design
2. Shared Python code proven before building frontends
3. Vue components can serve both Flask and Pyodide backends

---

## 9. References

- Existing code: `scripts/mindmap/visualize_density_manifold.py`
- Pyodide example: `examples/pyodide-matrix/`
- Vue.js viewport: `src/unifyweaver/examples/pearltrees/vue/MindMapViewport.vue`
- Streamlit example: `tools/agentRag/src/agent_rag/core/web_ui.py`
