# WASM Graph Visualization Example

Demonstrates UnifyWeaver's graph visualization with Cytoscape.js.

**Fully Generated:** All output files (HTML, TypeScript) are generated from `graph_module.pl`.

## Quick Start

### Generate All Files

```bash
cd examples/wasm-graph
swipl -g "consult('graph_module.pl'), graph_module:generate_all" -t halt
```

This generates:
- `index.html` - Complete web application with Cytoscape.js
- `graph_wasm.ts` - TypeScript bindings for WASM (with JS fallback)

### View in Browser

```bash
python3 -m http.server 8080
# Open http://localhost:8080
```

## Files

| File | Description |
|------|-------------|
| `graph_module.pl` | Prolog module - source of truth for all generation |
| `index.html` | (Generated) Complete web app with Cytoscape.js |
| `graph_wasm.ts` | (Generated) TypeScript bindings with JS fallback |
| `family_graph.ts` | (Legacy) Old TypeScript bindings |
| `build.sh` | (Legacy) WASM build script |

## Graph Definition

Define graphs in Prolog:

```prolog
% Nodes with properties
node(abraham, [label("Abraham"), type(person), generation(1)]).
node(isaac, [label("Isaac"), type(person), generation(2)]).

% Edges with properties
edge(abraham, isaac, [relation(parent)]).

% Graph configuration
graph_config([
    title("Family Tree"),
    layout(cose),
    theme(dark),
    node_color('#7c3aed'),
    edge_color('#00d4ff')
]).
```

## Features

The generated app includes:
- Family tree with 8 nodes and parent relationships
- Layout options: Force, Circle, Grid, Tree
- Add custom edges (creates nodes automatically)
- Remove edges individually
- Click nodes to see details
- Load/clear sample data
- Responsive Cytoscape.js visualization

## Architecture

```
graph_module.pl
├── node/2              # Node definitions
├── edge/3              # Edge definitions
├── graph_config/1      # Graph configuration
├── generate_all/0      # Main entry point
├── generate_html/1     # Complete HTML app
└── generate_ts_bindings/1  # TypeScript WASM bindings
```

### Pipeline

```
Prolog (node/2, edge/3)
    ↓ graph_module.pl
HTML + JavaScript (Cytoscape.js)
    ↓
Browser Graph Visualization
```

## Advanced: WASM Backend (Optional)

For high-performance graph operations, compile to WASM:

```bash
# Generate LLVM IR
swipl -g "use_module('src/unifyweaver/targets/llvm_target'),
          compile_wasm_string_module([func(ancestor, 2, transitive_closure)],
                                      [module_name(family_graph)], Code),
          write_llvm_program(Code, 'examples/wasm-graph/family_graph.ll')" -t halt

# Build WASM
./build.sh
```

The TypeScript bindings automatically use WASM if available, otherwise fall back to JavaScript.

## Dependencies

- SWI-Prolog (for code generation)
- Modern browser with ES6 module support
- Cytoscape.js (loaded from CDN)

Optional (for WASM):
- LLVM 14+ with wasm32 target
- wasm-ld (from lld package)

## Reference

Original hand-written index.html preserved in git history:
https://github.com/s243a/UnifyWeaver/blob/main/examples/wasm-graph/index.html (pre-generation)
