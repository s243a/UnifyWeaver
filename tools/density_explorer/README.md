# Density Manifold Explorer

Interactive visualization of embedding density manifolds with tree overlays.

## Versions

| Version | Path | Description |
|---------|------|-------------|
| Streamlit | `streamlit_app.py` | Python/Streamlit version with Plotly |
| Web (hand-crafted) | `web/index.html` | Full-featured browser version |
| Web (generated) | `generated/index.html` | Prolog-generated version |

## Generated Version

The `density_module.pl` generates a functional density explorer from declarative Prolog specifications.

### Generate

```bash
cd tools/density_explorer
swipl -g "consult('density_module.pl'), density_module:generate_all" -t halt
```

This creates `generated/index.html` (does NOT overwrite `web/index.html`).

### View

```bash
python3 -m http.server 8080
# Open http://localhost:8080/generated/index.html
```

## Features

### Both Versions
- Density heatmap visualization with Plotly.js
- Display toggles: Points, Peaks, Contours, Tree
- Tree overlay with max depth control
- Search nodes
- Export mindmap (JSON)
- Mobile-responsive layout

### Generated Version Specific
- Declarative control definitions in Prolog
- Orientation-aware colorbar (bottom in portrait, right in landscape)
- Touch controls: drag to pan, pinch to zoom
- Two-finger scroll for page navigation on mobile

### Hand-crafted Version Additional Features
- Pyodide integration for in-browser Python
- Multiple dataset sources
- Custom projection axes
- Multiple export formats (VUE, FreeMind, Mermaid, OPML, GraphML)
- ML embedding models
- More tree types (MST, J-guided)

## Architecture

```
density_module.pl
├── app_config/1        # App configuration (title, colors, theme)
├── tab/3               # Tab definitions (View, Data, Compute)
├── checkbox/3          # Checkbox controls
├── control/4           # Slider/select controls
├── export_format/3     # Export format definitions
├── dataset/3           # Dataset definitions
├── generate_all/0      # Main entry point
└── generate_html/1     # Complete HTML generation
    ├── generate_css/1
    ├── generate_body/1
    └── generate_javascript/1
```

## Mobile Support

- **Portrait**: Colorbar at bottom (toggle with "Show Density Scale")
- **Landscape**: Colorbar on right, plot fills available space
- **Touch**: Drag to pan, pinch to zoom, two-finger for page scroll
- **Orientation change**: Auto-resizes and repositions elements

## Dependencies

- SWI-Prolog (for code generation)
- Modern browser with ES6 support
- Plotly.js (loaded from CDN)
