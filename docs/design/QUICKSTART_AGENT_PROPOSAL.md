# Quickstart Agent Proposal

## Problem Statement

UnifyWeaver has grown into a multi-faceted project with capabilities spanning:
- Prolog compilation to multiple targets
- Mindmap organization tools
- GUI/app generation from declarative specs
- Semantic search and federated models
- Security infrastructure (policy and webapp)

New users face a "where do I start?" problem. Existing documentation is:
- **Reference-oriented** ("What does X do?") rather than **task-oriented** ("I want to accomplish Y")
- **Target-specific** (organized by compilation target) rather than **general-to-specific**
- **Built early** in the project lifecycle, missing newer capabilities

## Proposed Solution

A **Quickstart Agent** backed by a curated **task-oriented Q&A database** that:

1. Meets users where they are ("I have mindmaps to organize")
2. Routes them to appropriate entry points (skills, docs, commands)
3. Guides from general concepts to specific implementations
4. Covers the full capability tree with user-focused questions

## Design Principles

### General to Specific

Structure knowledge so users learn patterns before instances:

```
"How do I compile?"                    (general, parameterized)
    ↓
"How do I compile to Bash?"            (specific, TARGET=bash)
```

This mirrors how the code itself is structured:
- `compile(Source, Target, Opts)` → general predicate
- `compile_bash(Source, Opts)` → specific instantiation

### User Intent First

Focus on what users want to accomplish, not what the system can do:

| System-Centric (old) | User-Centric (new) |
|----------------------|-------------------|
| "compiler_driver.pl supports 10 targets" | "I want to deploy logic as a shell script" |
| "MST clustering uses hierarchy objective" | "I have 500 mindmaps to organize" |
| "Federated models use Procrustes projection" | "I want to find similar bookmarks" |

### Separate DB, Then Merge

Build the quickstart database separately for:
- Focused curation and review
- Clear coverage tracking
- Quality control before integration

Later, distill both quickstart and general Q&A databases into a single transformer.

```
Development                          Deployment
───────────────────────────────────────────────────
┌─────────────────┐
│ Quickstart DB   │ ──┐                 ┌─────────────┐
│ (task-oriented) │   ├─ distill ──────→│ Single      │
└─────────────────┘   │                 │ Transformer │
┌─────────────────┐   │                 └─────────────┘
│ General Q&A DB  │ ──┘
│ (reference)     │
└─────────────────┘
```

## Capability Tree

The full system organized general-to-specific:

### Compilation

```
├── Compilation
│   ├── General: compiler_driver.pl --target {TARGET} --input {FILE}
│   │
│   ├── Implementation Paradigms
│   │   ├── Stream/Procedural (template-based, Unix pipes, LINQ iterators)
│   │   ├── Query Engine (IR + runtime, semi-naive evaluation)
│   │   ├── Generator-Based (lazy evaluation, Python generators, memoization)
│   │   └── Declarative Output (SQL for external execution)
│   │
│   ├── Targets
│   │   ├── bash (stream/procedural)
│   │   ├── go (native binary, no deps)
│   │   ├── rust (native binary)
│   │   ├── sql (declarative)
│   │   ├── python (generator-based)
│   │   ├── powershell (pure or BaaS mode)
│   │   ├── awk (text processing)
│   │   ├── csharp-stream (LINQ iterators)
│   │   ├── csharp-query (query runtime, semi-naive)
│   │   └── prolog (dialect transpilation)
│   │
│   ├── Recursion Patterns
│   │   ├── Linear recursion
│   │   │   └── Examples: parent/child, list traversal
│   │   ├── Tail recursion
│   │   │   └── Examples: accumulator patterns, iteration
│   │   ├── Tree recursion
│   │   │   └── Examples: fibonacci, tree traversal, XML parsing
│   │   ├── Transitive closure
│   │   │   └── Examples: ancestor, reachability, dependencies
│   │   └── Mutual recursion
│   │       └── Examples: even/odd, parser grammar, state machines
│   │
│   ├── Data
│   │   ├── Input Sources
│   │   │   ├── JSON (nested path extraction)
│   │   │   ├── CSV/TSV
│   │   │   ├── XML
│   │   │   └── Custom plugins
│   │   │
│   │   ├── Databases
│   │   │   ├── SQLite (embedded)
│   │   │   ├── bbolt (Go embedded)
│   │   │   ├── PostgreSQL
│   │   │   └── MongoDB
│   │   │
│   │   ├── Patterns
│   │   │   ├── Aggregation (count, sum, min, max, group_by)
│   │   │   ├── Joins (inner, outer, cross)
│   │   │   ├── Deduplication
│   │   │   └── Cycle detection
│   │   │
│   │   └── Output Formats
│   │       ├── JSON
│   │       ├── TSV
│   │       └── SQL views
│   │
│   └── Advanced
│       ├── Enhanced chaining (fan-out, merge, routing, filtering)
│       └── Binding system (map predicates to stdlib functions)
```

### Semantic Search

```
├── Semantic Search
│   ├── General: How does semantic similarity work?
│   │
│   ├── Approaches
│   │   ├── Input (how queries/documents are encoded)
│   │   │   ├── Embedding models (see below)
│   │   │   ├── Tokenization strategies
│   │   │   └── Materialized path encoding (target_text)
│   │   │
│   │   ├── Output/Answer (how rankings are computed)
│   │   │   ├── Minimal Projection (Procrustes, recommended)
│   │   │   │   └── Federated W matrices per cluster
│   │   │   ├── LDA (topic modeling, not recommended)
│   │   │   ├── Direct cosine similarity
│   │   │   └── BM25 (sparse, lexical)
│   │   │
│   │   └── Blended (ensemble methods)
│   │       ├── Score fusion (weighted combination)
│   │       ├── Rank fusion (RRF)
│   │       └── Cascade (coarse → fine)
│   │
│   ├── Embedding Models
│   │   ├── nomic-embed-text-v1.5 (default, recommended)
│   │   ├── sentence-transformers
│   │   ├── BERT
│   │   ├── modernBERT (requires venv on older Ubuntu)
│   │   ├── OpenAI embeddings
│   │   └── Custom/local models
│   │
│   ├── Clustering Methods
│   │   ├── MST (minimum spanning tree, recommended)
│   │   ├── k-means
│   │   ├── Hierarchical agglomerative
│   │   └── DBSCAN
│   │
│   ├── Hierarchy Objectives
│   │   ├── General: J = D / (1 + H)
│   │   │
│   │   ├── Probability/Entropy Sources
│   │   │   ├── Fisher
│   │   │   ├── BERT
│   │   │   └── modernBERT (requires venv on older Ubuntu)
│   │   │
│   │   └── Distance metrics
│   │       ├── Cosine
│   │       └── Euclidean
│   │
│   ├── Target Implementations
│   │   ├── Python (reference implementation)
│   │   ├── Go (native inference)
│   │   ├── Rust (native inference)
│   │   └── Browser/WASM (proposed)
│   │
│   ├── Training
│   │   ├── train_pearltrees_federated.py
│   │   ├── Clustering method selection
│   │   └── Model format (.pkl)
│   │
│   └── Inference
│       ├── infer_pearltrees_federated.py
│       ├── Folder suggestion (suggest_folder.py)
│       └── Bookmark filing (bookmark_filing_assistant.py)
```

### Mindmap

```
├── Mindmap
│   ├── General: Tools for organizing .smmx SimpleMind files
│   │
│   ├── Linking
│   │   ├── link_pearltrees.py (attach Pearltrees URLs to nodes)
│   │   └── Prerequisites: federated model, embeddings
│   │
│   ├── Organization
│   │   ├── MST folder grouping (semantic clustering)
│   │   ├── Folder suggestion (where should this go?)
│   │   └── Hierarchy objective → (see Semantic Search)
│   │
│   ├── Cross-References
│   │   ├── build_index.py (tree_id → path mapping)
│   │   ├── add_relative_links.py (cloudmapref attributes)
│   │   ├── rename_mindmap.py (rename with link updates)
│   │   └── build_reverse_index.py (backlinks)
│   │
│   └── Index Formats
│       ├── JSON (human-readable)
│       ├── TSV (shell scripting)
│       └── SQLite (large collections)
```

### GUI

```
├── GUI
│   ├── General: Visual tools and app generation
│   │
│   ├── App Generation (from Prolog specs)
│   │   ├── General: generate_complete_project/4
│   │   │
│   │   ├── Frontend Targets
│   │   │   ├── Vue 3 (Vite + TypeScript)
│   │   │   ├── React Native (Expo)
│   │   │   ├── Flutter (Dart)
│   │   │   └── SwiftUI (iOS/macOS)
│   │   │
│   │   ├── Backend Targets
│   │   │   └── FastAPI
│   │   │
│   │   ├── Component Library (20+ components)
│   │   │   ├── Modals
│   │   │   ├── Toasts
│   │   │   ├── Cards
│   │   │   └── (others)
│   │   │
│   │   ├── Layout System
│   │   │   ├── CSS Grid
│   │   │   ├── Flexbox
│   │   │   └── Placement
│   │   │
│   │   ├── Data Binding
│   │   │   ├── State management
│   │   │   ├── Reactivity
│   │   │   └── WebSocket sync
│   │   │
│   │   ├── Accessibility
│   │   │   ├── ARIA attributes
│   │   │   └── Keyboard navigation
│   │   │
│   │   ├── Responsive Design
│   │   │   ├── Breakpoints
│   │   │   └── Adaptive layouts
│   │   │
│   │   └── Theming
│   │       ├── Colors
│   │       ├── Typography
│   │       └── Dark mode
│   │
│   └── Visualization Tools
│       ├── Density Explorer
│       │   ├── Flask API
│       │   ├── Vue frontend
│       │   └── Embedding space visualization
│       │
│       └── (other tools)
```

### Security

```
└── Security
    ├── Policy/Firewall
    │   ├── General: Firewall/policy system concept
    │   ├── Network policies
    │   └── Service restrictions
    │
    └── Webapp Security (PR 608)
        ├── General: Declarative security for generated apps
        ├── Navigation guards (route-level auth)
        ├── Auth backends (mock, text_file, sqlite, oauth2, ldap)
        ├── TLS config (proxy, passthrough, mutual)
        └── Shell sandbox (none, namespace, docker, firejail)
```

## User Journey Questions

Instead of mapping questions to tree items, we start with what users want to do:

### New User Personas

| Persona | Goal | Likely First Question |
|---------|------|----------------------|
| **Logic Developer** | Deploy Prolog rules as production code | "How do I compile Prolog to a runnable script?" |
| **Knowledge Organizer** | Organize large mindmap collection | "I have hundreds of mindmaps, how do I organize them?" |
| **App Builder** | Generate frontend from specs | "Can I generate a Vue app from Prolog?" |
| **Data Engineer** | Build ETL pipelines | "How do I process JSON and aggregate results?" |
| **Security Architect** | Add auth to generated apps | "How do I add authentication to my app?" |

### Sample User Questions

**Getting Started:**
- "What is UnifyWeaver?"
- "What can I do with this project?"
- "Where do I start?"

**Compilation:**
- "How do I compile Prolog?"
- "I need a standalone binary with no dependencies"
- "How do I handle recursive relationships like ancestor?"
- "Can I compile to SQL for my database?"

**Mindmaps:**
- "I have 500 mindmaps to organize by topic"
- "How do I link mindmaps to my Pearltrees?"
- "Can mindmaps link to each other?"
- "Where should this mindmap go?"

**Semantic Search:**
- "How do I find similar items?"
- "I want to file bookmarks automatically"
- "How do I train a model for my data?"

**GUI/Apps:**
- "Can I generate a mobile app from Prolog?"
- "How do I add dark mode?"
- "I need a login screen"

**Security:**
- "How do I restrict network access?"
- "I need OAuth login for my app"
- "How do I sandbox shell commands?"

## Coverage Analysis

For each user question, we track which tree nodes it touches. Good coverage means:
- Every major branch has at least one common question
- Specific questions naturally lead to specific tree nodes
- No orphaned nodes that users would never discover

## Next Steps

1. **Philosophy Doc** - Define principles for question/answer design
2. **Specification** - Schema for quickstart database entries
3. **Implementation Plan** - Build sequence and tooling
4. **Pilot Questions** - Draft first 20-30 entries covering major branches

## Related Documents

- `AGENT_INSTRUCTIONS.md` - Project-level agent startup instructions
- `skills/SKILL_ROUTER.md` - Existing skill routing system
- `skills/SKILL_INDEX.md` - Flat skill index
