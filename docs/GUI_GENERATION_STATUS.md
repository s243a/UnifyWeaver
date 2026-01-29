# GUI Generation Status

This document tracks the status of GUI/UI app generation in UnifyWeaver.

## Summary

| Category | Count |
|----------|-------|
| Fully generated apps | 4 |
| Partially generated apps | 3 |
| Manual apps (need generators) | 4 |
| Untested visualization generators | 13 |

---

## Fully Generated Apps

These apps can be completely regenerated from Prolog via `generate.pl` or `generate.sh`.

### 1. storybook-react
**Path:** `examples/storybook-react/`
**Generator:** `generate.pl`
**Architecture:** Prolog → `react_generator.pl` → React/TypeScript
**Output:** Button.tsx, TextInput.tsx components
**Status:** ✅ Tested in Storybook

### 2. flutter-cli
**Path:** `examples/flutter-cli/`
**Generator:** `generate.pl`
**Architecture:** Prolog → `flutter_generator.pl` → Dart/Flutter
**Output:** main.dart, app.dart with Material widgets
**Status:** ✅ Syntax validated (Flutter SDK not available on Termux)

### 3. tui-cli
**Path:** `examples/tui-cli/`
**Generator:** Uses `tui_generator.pl` (echo mode)
**Architecture:** Prolog → `tui_generator.pl` → Bash + ANSI escape codes
**Output:** http_cli.sh with colored terminal UI
**Status:** ✅ Tested

### 4. tui-dialog-cli
**Path:** `examples/tui-dialog-cli/`
**Generator:** Uses `tui_generator.pl` (dialog mode)
**Architecture:** Prolog → `tui_generator.pl` → Bash + dialog/whiptail
**Output:** dialog_cli.sh with ncurses menus
**Status:** ✅ Tested

---

## Partially Generated Apps

These apps have Prolog modules that generate some code, but HTML/scaffolding is manual.

### 5. http-cli-server
**Path:** `examples/http-cli-server/`
**Generator:** `generate.sh` calls multiple generators
**Architecture:**
- Prolog → `http_server_generator.pl` → TypeScript/Express server
- Prolog → `html_interface_generator.pl` → HTML + embedded Vue.js
- Prolog → `auth_generator.pl` → Authentication middleware

**Generated:**
- `generated/server.ts` - Express server with routes
- `generated/index.html` - Vue.js single-page app

**Manual:** spec.pl (declarative specification - intentionally manual)
**Status:** ✅ Tested with HTTPS and authentication

### 6. pyodide-matrix
**Path:** `examples/pyodide-matrix/`
**Generator:** `matrix_module.pl` has `generate_all/0`
**Architecture:** Prolog → Python (NumPy) + TypeScript bindings
**Generated:**
- matrix_lib.py - NumPy operations
- matrix_wasm.ts - TypeScript bindings

**Manual:** index.html (23KB) - needs HTML generator
**Status:** ⚠️ Generation partial, browser app untested

### 7. curve-plot
**Path:** `examples/curve-plot/`
**Generator:** `curve_module.pl` has `generate_curve_wasm/2`
**Architecture:** Prolog → LLVM IR → WebAssembly + TypeScript/Chart.js
**Generated:**
- LLVM IR for curve functions
- TypeScript Chart.js bindings

**Manual:** index.html (16KB) - needs HTML generator
**Status:** ⚠️ Generation partial, browser app untested

---

## Manual Apps (Need Generators)

These apps exist but have no Prolog generation - they were handcrafted.

### 8. wasm-graph
**Path:** `examples/wasm-graph/`
**Current:** Manual TypeScript + HTML
**Needed:** `generate.pl` using `graph_generator.pl`
**Architecture:** Should be Prolog → `graph_generator.pl` → Cytoscape.js/React
**Status:** ❌ No generator

### 9. react-cli
**Path:** `examples/react-cli/`
**Current:** Manual React/Vite app
**Needed:** `generate.pl` using `react_generator.pl` + `project_scaffold.pl`
**Architecture:** Should be Prolog → React/TypeScript via Vite
**Status:** ❌ No generator

### 10. react-http-cli
**Path:** `examples/react-http-cli/`
**Current:** Manual React app (24KB App.tsx)
**Needed:** `generate.pl` using `react_generator.pl` + `html_interface_generator.pl`
**Architecture:** Should be Prolog → React/TypeScript + HTTP client
**Status:** ❌ No generator

### 11. rust-ffi-node
**Path:** `examples/python-bridges/rust-ffi-node/`
**Current:** Manual Express + React frontend
**Needed:** `generate.pl` for frontend
**Architecture:** Should be Prolog → React frontend + Express API
**Status:** ❌ No generator for frontend

---

## Visualization Generators (Untested)

These generators exist in `src/unifyweaver/glue/` but have no example apps.

### Chart Types

| Generator | Output Target | Description | Example Needed |
|-----------|---------------|-------------|----------------|
| `chord_generator.pl` | React + D3 | Chord diagrams for relationships | Storybook story or standalone |
| `funnel_chart_generator.pl` | React + Chart.js | Funnel/pipeline visualization | Storybook story or standalone |
| `gauge_chart_generator.pl` | React + Chart.js | Gauge/meter with thresholds | Storybook story or standalone |
| `radar_chart_generator.pl` | React + Chart.js | Radar/spider charts | Storybook story or standalone |
| `sankey_generator.pl` | React + D3 | Sankey flow diagrams | Storybook story or standalone |
| `plot3d_generator.pl` | React + Plotly.js | 3D surface/scatter/line | Storybook story or standalone |
| `heatmap_generator.pl` | React + D3/seaborn | Heatmap grids | Storybook story or standalone |
| `treemap_generator.pl` | React + Plotly.js | Hierarchical treemaps | Storybook story or standalone |

### Python Visualization

| Generator | Output Target | Description | Example Needed |
|-----------|---------------|-------------|----------------|
| `matplotlib_generator.pl` | Python | matplotlib plotting code | Python script example |

### UI Features

| Generator | Output Target | Description | Example Needed |
|-----------|---------------|-------------|----------------|
| `animation_generator.pl` | CSS/React | Keyframes, transitions, sequences | Storybook animation gallery |
| `interaction_generator.pl` | React | Pan/zoom, brush selection, tooltips | Interactive demo |
| `virtual_scroll_generator.pl` | React | Efficient large list rendering | 10K+ item list demo |
| `lazy_loading_generator.pl` | React | Pagination, infinite scroll, chunked | Infinite scroll demo |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prolog Specifications                         │
│  (ui_primitives, http_cli_ui, graph specs, chart specs, etc.)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Web Targets   │   │ Mobile        │   │ Terminal      │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ react_gen     │   │ flutter_gen   │   │ tui_gen       │
│ vue_gen       │   │ (swiftui)     │   │ (echo mode)   │
│ html_iface    │   │               │   │ (dialog mode) │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ React/TS      │   │ Dart/Flutter  │   │ Bash + ANSI   │
│ Vue/TS        │   │ Swift/SwiftUI │   │ Bash + dialog │
│ HTML+Vue      │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Generator → Target Matrix

| Generator | React | Vue | Flutter | TUI | HTML | Python |
|-----------|-------|-----|---------|-----|------|--------|
| `ui_primitives.pl` | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| `react_generator.pl` | ✅ | - | - | - | - | - |
| `vue_generator.pl` | - | ✅ | - | - | - | - |
| `flutter_generator.pl` | - | - | ✅ | - | - | - |
| `tui_generator.pl` | - | - | - | ✅ | - | - |
| `html_interface_generator.pl` | - | ✅ | - | - | ✅ | - |
| `graph_generator.pl` | ✅ | ✅ | - | - | - | - |
| `curve_plot_generator.pl` | ✅ | - | - | - | - | - |
| `matplotlib_generator.pl` | - | - | - | - | - | ✅ |
| Chart generators | ✅ | - | - | - | - | - |

---

## Next Steps

### Priority 1: Add generators to existing apps
1. `wasm-graph/generate.pl` - Use `graph_generator.pl`
2. `react-cli/generate.pl` - Use `react_generator.pl` + `project_scaffold.pl`
3. `pyodide-matrix/` - Add HTML generation to `matrix_module.pl`
4. `curve-plot/` - Add HTML generation to `curve_module.pl`

### Priority 2: Test visualization generators
Add Storybook stories for each chart type:
- Chord, Funnel, Gauge, Radar, Sankey, Plot3D, Heatmap, Treemap

### Priority 3: Test UI feature generators
- Animation gallery in Storybook
- Virtual scroll with 10K items
- Lazy loading / infinite scroll demo
- Interaction demo (pan/zoom/tooltips)
