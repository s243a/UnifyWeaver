# Skill Coverage Gaps Analysis

Analysis of Q&A coverage against the capability tree, identifying gaps and proposed solutions.

## Current Coverage Summary

From 183 Q&A entries generated across 14 skills:

| Branch | Coverage | Notes |
|--------|----------|-------|
| Compilation | Good | Targets, JSON, recursion covered |
| Semantic Search | Good | Training, inference, Procrustes covered |
| Mindmap | Good | Linking, organization, cross-refs covered |
| GUI | Good | App generation, all frontend targets covered |
| Security | Good | Auth, guards, TLS, sandbox covered |

## Identified Gaps

### High Priority Gaps

#### 1. Hierarchy Objectives (J = D/(1+H))

**Gap:** No coverage of the mathematical foundation behind MST clustering.

**Nature:** This is both theoretical (the objective function) and practical (entropy sources, distance metrics).

**Proposed Skill:** `skill_hierarchy_objective.md`

**Content should cover:**
- The objective function J = D / (1 + H)
- What D (distance/dispersion) and H (entropy) represent
- Entropy source options: Fisher, BERT, modernBERT
- Distance metrics: cosine, euclidean
- When to use different entropy sources
- Connection to `scripts/mindmap/hierarchy_objective.py`

**Educational references:**
- `book-14-ai-training/` - training concepts
- `book-13-semantic-search/07_density_scoring.md` - related scoring

---

#### 2. Embedding Models Selection

**Gap:** No guidance on choosing between embedding models or setting up modernBERT (requires venv).

**Proposed Skill:** `skill_embedding_models.md`

**Content should cover:**
- Available models: nomic-embed-text-v1.5, sentence-transformers, BERT, modernBERT
- Dimension differences (768 vs 384)
- Performance vs quality tradeoffs
- modernBERT venv setup on older Ubuntu
- Model selection by use case

**Educational references:**
- `book-14-ai-training/02_embedding_providers.md`

---

#### 3. Density Explorer Tool

**Gap:** Visualization tool exists (`tools/density_explorer/`) but no skill documentation.

**Proposed Skill:** `skill_density_explorer.md`

**Content should cover:**
- What the density explorer does (embedding space visualization)
- Flask API setup
- Vue frontend usage
- Interpreting visualizations
- Connection to semantic search debugging

---

### Medium Priority Gaps

#### 4. Transpiler Extension / Binding System

**Gap:** No coverage of how to extend the transpiler or bind predicates to stdlib functions.

**Nature:** This is about extending and generalizing the transpiler, not just the builtin target features.

**Proposed Skill:** `skill_transpiler_extension.md`

**Content should cover:**
- How the transpiler architecture works
- Adding new target languages
- Binding predicates to stdlib functions
- Template system extension
- Target-specific optimizations

**Note:** This is an advanced topic for users who want to contribute to UnifyWeaver or add custom targets.

---

#### 5. SQL Target / Declarative Output

**Gap:** SQL as a compilation target has different semantics (declarative, external execution).

**Proposed Skill:** `skill_sql_target.md`

**Content should cover:**
- Prolog to SQL translation
- Recursive CTEs for transitive closure
- Aggregation mapping
- Views vs queries
- Target database compatibility

**Educational references:**
- `book-10-sql-target/`

---

#### 6. Query Runtime (C# Query Engine)

**Gap:** The query engine paradigm (IR + runtime, semi-naive evaluation) is undocumented.

**Proposed Skill:** `skill_csharp_query_target.md`

**Content should cover:**
- Query engine vs stream paradigm
- Semi-naive evaluation
- When to use query runtime
- Performance characteristics

**Educational references:**
- `book-03-csharp-target/`

---

#### 7. Ensemble/Blended Search

**Gap:** No coverage of combining multiple search approaches.

**Proposed Skill:** `skill_ensemble_search.md`

**Content should cover:**
- Score fusion (weighted combination)
- Rank fusion (RRF)
- Cascade approaches (coarse → fine)
- When to use ensemble methods

---

### Lower Priority Gaps

#### 8. Mindmap Index Formats

**Gap:** index_store.py supports JSON/TSV/SQLite but this isn't documented.

**Proposed Skill:** `skill_mindmap_index.md` or add to existing cross-links skill.

---

#### 9. Responsive Design

**Gap:** GUI responsive design features (breakpoints, adaptive layouts) not covered.

**Proposed Skill:** `skill_responsive_design.md` or add to `skill_app_generation.md`.

**Educational references:**
- `book-gui-generation/07_responsive_design.md`

---

## Recommended Action Plan

### Phase 1: Theory + Practice Skills
1. `skill_hierarchy_objective.md` - Mathematical foundation with practical guidance
2. `skill_embedding_models.md` - Model selection and setup

### Phase 2: Tool Documentation
3. `skill_density_explorer.md` - Visualization tool

### Phase 3: Advanced Topics
4. `skill_transpiler_extension.md` - For contributors
5. `skill_sql_target.md` - Declarative paradigm
6. `skill_csharp_query_target.md` - Query engine paradigm

### Phase 4: Incremental Improvements
7. Update existing skills with missing sub-topics
8. Add ensemble search concepts to semantic search skills

## Coverage Tree (with counts)

```
├── Compilation
│   ├── General ─────────────────────────────── ✓ 22
│   ├── Implementation Paradigms
│   │   ├── Stream/Procedural ───────────────── ✓ 4
│   │   ├── Query Engine ────────────────────── ○ 0  ← skill_csharp_query_target.md
│   │   ├── Generator-Based ─────────────────── ✓ 6
│   │   └── Declarative Output ──────────────── ○ 0  ← skill_sql_target.md
│   ├── Targets ─────────────────────────────── ✓ (all covered)
│   ├── Recursion Patterns ──────────────────── ✓ 6
│   ├── Data ────────────────────────────────── ✓ 65
│   └── Advanced
│       ├── Enhanced chaining ───────────────── ✓ 1
│       └── Binding system ──────────────────── ○ 0  ← skill_transpiler_extension.md
│
├── Semantic Search
│   ├── General ─────────────────────────────── ✓ 19
│   ├── Approaches
│   │   ├── Procrustes ──────────────────────── ✓ 23
│   │   ├── LDA ─────────────────────────────── ✓ 2
│   │   └── Blended/Ensemble ────────────────── ○ 0  ← skill_ensemble_search.md
│   ├── Embedding Models
│   │   ├── nomic/sentence-transformers ─────── ✓ 14
│   │   └── BERT/modernBERT ─────────────────── ○ 0  ← skill_embedding_models.md
│   ├── Clustering Methods ──────────────────── ✓ 23
│   ├── Hierarchy Objectives ────────────────── ○ 0  ← skill_hierarchy_objective.md
│   ├── Training ────────────────────────────── ✓ 19
│   └── Inference ───────────────────────────── ✓ 51
│
├── Mindmap
│   ├── General ─────────────────────────────── ✓ 54
│   ├── Linking ─────────────────────────────── ✓ 9
│   ├── Organization ────────────────────────── ✓ 28
│   ├── Cross-References ────────────────────── ✓ 19
│   └── Index Formats ───────────────────────── ✓ 4 (low)
│
├── GUI
│   ├── App Generation ──────────────────────── ✓ 62
│   ├── Component Library ───────────────────── ✓ 6
│   ├── Layout/Binding/Theming ──────────────── ✓ 8
│   ├── Responsive Design ───────────────────── ○ 0  ← add to skill_app_generation.md
│   └── Visualization Tools
│       └── Density Explorer ────────────────── ○ 0  ← skill_density_explorer.md
│
└── Security
    ├── Policy/Firewall ─────────────────────── ✓ 3
    └── Webapp Security ─────────────────────── ✓ 26
```

## Notes

- The "theory + how-to" nature of some topics (like hierarchy objectives) suggests a different skill format that combines conceptual explanation with practical commands
- The transpiler extension topic is really about the architecture and extensibility, not just binding to stdlib
- Some gaps can be addressed by updating existing skills rather than creating new ones
