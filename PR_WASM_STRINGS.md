# feat: Add WASM String Support and Graph Visualization

## Summary

Extends LLVM/WASM target to handle string types, enabling graph visualization with Cytoscape.js.

## New Features

### WASM String Runtime
- `alloc`/`dealloc` - Bump allocator for linear memory
- `memcpy_wasm`/`strlen_wasm` - String helpers
- `addEdge`/`getEdge`/`getEdgeCount` - Edge storage for graphs

### TypeScript Bindings
```typescript
const graph = await GraphWasm.load('family_graph.wasm');
graph.addEdge('tom', 'bob');
graph.getEdges(); // [['tom', 'bob']]
```

### Cytoscape.js Demo
Interactive graph visualization with dark theme UI.

## Files Changed

### Implementation
```
src/unifyweaver/targets/llvm_target.pl   [MODIFIED] +290 lines
- Phase 5: generate_wasm_string_runtime/1
- compile_wasm_string_module/3
- generate_edge_storage/1
- generate_ts_string_bindings/2
```

### Examples
```
examples/wasm-graph/                     [NEW]
├── README.md                            Quick start guide
├── build.sh                             LLVM → WASM script
├── index.html                           Cytoscape demo (dark theme)
└── family_graph.ts                      Generated GraphWasm bindings
```

### Documentation
```
docs/proposals/wasm_string_support.md    [NEW] Full architecture proposal
docs/LLVM_TARGET.md                      [MODIFIED] Phase 5 section
```

## Architecture

```
Prolog (parent/2, ancestor/2)
    ↓ compile_wasm_string_module/3
LLVM IR (family_graph.ll)
    ↓ llc + wasm-ld
WASM (family_graph.wasm)
    ↓ GraphWasm.load()
TypeScript + Cytoscape.js
```

## Technical Notes

- **Bump allocator**: Simple, no fragmentation, no GC needed
- **Pointer+length convention**: UTF-8 in TypeScript, null-terminated in WASM
- **Max 256 edges**: Current implementation limit (easily expandable)
- **cose layout**: Cytoscape's compound spring embedder for animated graphs

## Demo

![Sample graph](examples/wasm-graph/screenshot.png)

```bash
cd examples/wasm-graph
npx serve .
# Open http://localhost:3000
```

Click "Load Sample Data" → shows family tree graph.

## Related

- PR #352: JavaScript Family + TypeScript Target
- PR #362: Haskell Parsec
