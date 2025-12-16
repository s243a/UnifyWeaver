# WASM Graph Visualization Example

Demonstrates UnifyWeaver's WASM string support with Cytoscape.js graph rendering.

## Quick Start

```bash
# 1. Generate WASM module from Prolog
cd /path/to/UnifyWeaver
swipl -g "use_module('src/unifyweaver/targets/llvm_target'),
          compile_wasm_string_module([func(ancestor, 2, transitive_closure)], 
                                      [module_name(family_graph)], Code),
          write_llvm_program(Code, 'examples/wasm-graph/family_graph.ll')" -t halt

# 2. Build WASM (requires LLVM with WASM backend)
./build.sh

# 3. Serve and view
npx serve .
# Open http://localhost:3000
```

## Files

| File | Description |
|------|-------------|
| `build.sh` | Compiles .ll → .wasm |
| `family_graph.ts` | Generated TypeScript bindings |
| `graph.ts` | Cytoscape.js integration |
| `index.html` | Demo page |

## Architecture

```
Prolog (parent/2, ancestor/2)
    ↓ compile_wasm_string_module/3
LLVM IR (family_graph.ll)
    ↓ llc + wasm-ld
WASM (family_graph.wasm)
    ↓ GraphWasm.load()
TypeScript + Cytoscape.js
    ↓
Browser Graph Visualization
```

## Example Usage

```typescript
import { GraphWasm } from './family_graph';
import cytoscape from 'cytoscape';

const graph = await GraphWasm.load('family_graph.wasm');

// Add edges
graph.addEdge('tom', 'bob');
graph.addEdge('bob', 'alice');
graph.addEdge('alice', 'eve');

// Render with Cytoscape
const edges = graph.getEdges();
const cy = cytoscape({
  container: document.getElementById('graph'),
  elements: edges.flatMap(([from, to]) => [
    { data: { id: from } },
    { data: { id: to } },
    { data: { source: from, target: to } }
  ])
});
```

## Dependencies

- LLVM 14+ with wasm32 target
- wasm-ld (from lld package)
- Node.js / npm
- Cytoscape.js (`npm install cytoscape`)
