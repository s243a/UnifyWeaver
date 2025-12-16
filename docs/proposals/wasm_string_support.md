# WASM Graph Visualization Pipeline

## Overview

End-to-end demo proving the UnifyWeaver component system:

```
Prolog (ancestor/2) → Go/Rust → LLVM → WASM → TypeScript → Cytoscape.js
```

## Motivation

| Goal | Description |
|------|-------------|
| **Prove components** | Cross-target integration works |
| **Extend WASM** | String support for real-world use |
| **Visual demo** | Graph shows transitive closure |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Prolog (Source)                        │
│   parent(tom, bob).  parent(bob, alice).                   │
│   ancestor(X, Y) :- parent(X, Y).                          │
│   ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).          │
└─────────────────────┬───────────────────────────────────────┘
                      │ compile_transitive_closure (existing)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Go/Rust Target                            │
│   Generates BFS/DFS with string node names                 │
└─────────────────────┬───────────────────────────────────────┘
                      │ LLVM IR
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              WASM Module (NEW: String Support)              │
│   exports.addEdge(from: ptr, to: ptr)                      │
│   exports.findAncestors(node: ptr) → ptr[]                 │
│   exports.getEdges() → [ptr, ptr][]                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ generate_ts_bindings/2 (extended)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              TypeScript Bindings                            │
│   async function loadFamilyGraph(): FamilyGraph            │
│   graph.addEdge("tom", "bob")                              │
│   graph.findAncestors("tom") // ["bob", "alice"]           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Cytoscape.js Visualization                     │
│   Directed graph with nodes and edges                      │
│   Interactive (zoom, pan, select)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Existing Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| `compile_transitive_closure` | ✅ Go, Rust, LLVM | `recursive_compiler.pl` |
| WASM compilation | ✅ Numeric only | `llvm_target.pl` |
| `generate_ts_bindings/2` | ✅ Numeric only | `llvm_target.pl` |
| TypeScript target | ✅ New | `typescript_target.pl` |

---

## Implementation Phases

### Phase 1: WASM String Support

**Goal**: Extend WASM target to handle string parameters and returns.

| Task | Description |
|------|-------------|
| Memory management | `alloc`/`dealloc` exports |
| String encoding | UTF-8 encode/decode |
| String params | Pass strings to WASM |
| String returns | Return string/arrays |

**WASM Memory Model**:
```
┌─────────────────────────────────────────────────────┐
│                 WASM Linear Memory                  │
├─────────────────────────────────────────────────────┤
│  ptr → [length: i32][UTF-8 bytes...]               │
└─────────────────────────────────────────────────────┘
```

### Phase 2: TypeScript Bindings for Strings

**Goal**: Extend `generate_ts_bindings/2` to handle strings.

```typescript
interface FamilyGraph {
  memory: WebAssembly.Memory;
  
  // High-level API (generated)
  addEdge(from: string, to: string): void;
  findAncestors(node: string): string[];
  getEdges(): [string, string][];
}
```

### Phase 3: Graph Data Export

**Goal**: WASM exports edge list for visualization.

```typescript
// Returns all edges as array of [from, to] pairs
getEdges(): [string, string][]
```

### Phase 4: Cytoscape.js Integration

**Goal**: Render graph in browser.

```typescript
import cytoscape from 'cytoscape';

const cy = cytoscape({
  container: document.getElementById('graph'),
  elements: edges.map(([from, to]) => ([
    { data: { id: from } },
    { data: { id: to } },
    { data: { source: from, target: to } }
  ])).flat(),
  style: [
    { selector: 'node', style: { 'label': 'data(id)' } },
    { selector: 'edge', style: { 'target-arrow-shape': 'triangle' } }
  ]
});
```

### Phase 5: Example & Documentation

**Goal**: Working example in `examples/wasm-graph/`.

```
examples/wasm-graph/
├── README.md
├── family.pl          # Prolog source
├── build.sh           # Compile pipeline
├── family_graph.wasm  # Generated
├── family_graph.ts    # Generated bindings
├── index.html         # Demo page
└── graph.ts           # Cytoscape integration
```

---

## Graph Library Decision

| Library | Pros | Cons | Verdict |
|---------|------|------|---------|
| **Cytoscape.js** | Network diagrams, simple API, TypeScript | Medium bundle | ✅ Recommended |
| ts-graphviz | DOT format, Graphviz | Needs Graphviz | ❌ External dep |
| Vizdom | Rust→WASM, fast | Very new (2024) | ❌ Experimental |
| D3.js | Flexible | Steep learning | ❌ Overkill |

**Decision**: Cytoscape.js - designed for network graphs, minimal config.

---

## Success Criteria

```typescript
// 1. Load WASM
const graph = await loadFamilyGraph();

// 2. Add data (or load from WASM)
graph.addEdge("tom", "bob");
graph.addEdge("bob", "alice");
graph.addEdge("alice", "eve");

// 3. Query
console.log(graph.findAncestors("tom")); 
// ["bob", "alice", "eve"]

// 4. Visualize
renderGraph(graph.getEdges());
// Shows directed graph: tom → bob → alice → eve
```

---

## Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| LLVM + wasm-ld | WASM compilation | `apt install lld` |
| Node.js | TypeScript runtime | Already installed |
| Cytoscape.js | Graph rendering | `npm install cytoscape` |

---

## Timeline

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | WASM string support | Core work |
| 2 | TypeScript bindings | Build on Phase 1 |
| 3 | Graph data export | Small addition |
| 4 | Cytoscape integration | Example code |
| 5 | Documentation | Writeup |

---

## See Also

- [wasm_target_design.md](wasm_target_design.md) - Original WASM proposal
- [llvm_target.pl](../../src/unifyweaver/targets/llvm_target.pl) - LLVM/WASM implementation
- [Cytoscape.js docs](https://js.cytoscape.org/)
