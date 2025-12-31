# UnifyWeaver Architecture Diagrams

This document contains architecture diagrams for UnifyWeaver's component registry and LDA projection systems.

## Component Registry Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT REGISTRY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   RUNTIME   │    │   SOURCE    │    │   BINDING   │    │  TRANSFORM  │  │
│  │  Category   │    │  Category   │    │  Category   │    │  Category   │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │    Types    │    │    Types    │    │    Types    │    │    Types    │  │
│  │ - lda_proj  │    │ - csv       │    │ - bash      │    │ - filter    │  │
│  │ - embedding │    │ - python    │    │ - python    │    │ - aggregate │  │
│  │ - multi_head│    │ - sqlite    │    │ - powershell│    │ - expand    │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Instances  │    │  Instances  │    │  Instances  │    │  Instances  │  │
│  │ + Config    │    │ + Config    │    │ + Config    │    │ + Config    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Registry API Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           API FLOW                                        │
└──────────────────────────────────────────────────────────────────────────┘

  1. DEFINE CATEGORY                    2. REGISTER TYPE
  ────────────────────                  ──────────────────
  define_category(                      register_component_type(
    runtime,                              runtime,
    "Runtime components",                 lda_projection,
    [lazy_init(true)]                     lda_projection_module,
  )                                       [supports(project/2)]
                                        )

  3. DECLARE INSTANCE                   4. INVOKE
  ─────────────────────                 ────────
  declare_component(                    invoke_component(
    runtime,                              runtime,
    semantic_search,                      semantic_search,
    lda_projection,                       project(QueryEmb),
    [model_path('W.npy'),                 Projected
      temperature(0.1)]                 )
  )
```

## LDA Projection System

### Global Projection (Original)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GLOBAL LDA PROJECTION                               │
└─────────────────────────────────────────────────────────────────────────┘

   TRAINING                                    INFERENCE
   ────────                                    ─────────

   Q-A Clusters                                Query
   ┌─────────────┐                             │
   │ Cluster 1   │──┐                          ▼
   │ q1,q2,q3→a1 │  │                    ┌───────────┐
   ├─────────────┤  │    ┌─────────┐     │  Embed    │
   │ Cluster 2   │──┼───▶│Compute W│     │  (MiniLM) │
   │ q4,q5,q6→a2 │  │    └────┬────┘     └─────┬─────┘
   ├─────────────┤  │         │                │
   │ Cluster n   │──┘         ▼                ▼
   │ ...→an      │      ┌───────────┐    ┌───────────┐
   └─────────────┘      │  W matrix │───▶│  W @ q    │
                        │  (d × d)  │    │ projected │
                        └───────────┘    └─────┬─────┘
                                               │
                                               ▼
                                         ┌───────────┐
                                         │  Cosine   │
                                         │ Similarity│
                                         │ to answers│
                                         └─────┬─────┘
                                               │
                                               ▼
                                          Top-K Results
```

### Multi-Head Projection (New)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MULTI-HEAD LDA PROJECTION                            │
└─────────────────────────────────────────────────────────────────────────┘

   TRAINING                                    INFERENCE
   ────────                                    ─────────

   Per-Cluster Heads                           Query
   ┌─────────────────┐                         │
   │ Head 1 (sqlite) │                         ▼
   │  centroid_1     │                   ┌───────────┐
   │  answer_emb_1   │                   │  Embed    │
   ├─────────────────┤                   │  (MiniLM) │
   │ Head 2 (http)   │                   └─────┬─────┘
   │  centroid_2     │                         │
   │  answer_emb_2   │                         ▼
   ├─────────────────┤              ┌──────────────────────┐
   │ Head n (...)    │              │  Centroid Similarity │
   │  centroid_n     │              │  sim_k = q · c_k     │
   │  answer_emb_n   │              └──────────┬───────────┘
   └─────────────────┘                         │
                                               ▼
                                    ┌──────────────────────┐
                                    │   Softmax Routing    │
                                    │   w = softmax(s/τ)   │
                                    │   τ = temperature    │
                                    └──────────┬───────────┘
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         │                     │                     │
                         ▼                     ▼                     ▼
                   ┌──────────┐          ┌──────────┐          ┌──────────┐
                   │ w_1=0.85 │          │ w_2=0.10 │          │ w_n=0.05 │
                   │ × a_1    │          │ × a_2    │          │ × a_n    │
                   └────┬─────┘          └────┬─────┘          └────┬─────┘
                        │                     │                     │
                        └──────────┬──────────┴──────────┬──────────┘
                                   │                     │
                                   ▼                     │
                            ┌────────────┐               │
                            │  Σ w_k·a_k │◀──────────────┘
                            │ (projected)│
                            └─────┬──────┘
                                  │
                                  ▼
                            ┌───────────┐
                            │  Cosine   │
                            │ Similarity│
                            │ to answers│
                            └─────┬─────┘
                                  │
                                  ▼
                             Top-K Results
```

## Softmax Temperature Effect

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TEMPERATURE EFFECT ON ROUTING                         │
└─────────────────────────────────────────────────────────────────────────┘

  Input similarities: [0.85, 0.70, 0.60]

  τ = 1.0 (diffuse)         τ = 0.1 (sharp)          τ → 0 (argmax)
  ──────────────────         ─────────────────         ──────────────

  ┌────┐                     ┌────┐                    ┌────┐
  │████│ 0.39                │████│                    │████│
  │████│                     │████│                    │████│
  │████│                     │████│ 0.82               │████│ 0.99
  ├────┤                     │████│                    │████│
  │███ │ 0.34                │████│                    │████│
  │███ │                     ├────┤                    ├────┤
  ├────┤                     │█   │ 0.15               │    │ 0.01
  │██  │ 0.27                ├────┤                    ├────┤
  └────┘                     │    │ 0.03               │    │ 0.00
  h1 h2 h3                   └────┘                    └────┘
                             h1 h2 h3                  h1 h2 h3

  Result: Blended            Result: Head 1 dominates  Result: Winner-take-all
  (poor retrieval)           (good retrieval)          (hard routing)
```

## RAG Pipeline with Multi-Head LDA

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE WITH MULTI-HEAD LDA                      │
└─────────────────────────────────────────────────────────────────────────┘

  User Query
      │
      ▼
┌───────────────┐
│   Embedding   │
│   (MiniLM)    │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│           MULTI-HEAD LDA RETRIEVER            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Head 1  │  │ Head 2  │  │ Head n  │       │
│  │(sqlite) │  │ (http)  │  │  (...)  │       │
│  └────┬────┘  └────┬────┘  └────┬────┘       │
│       │            │            │             │
│       └────────────┼────────────┘             │
│                    ▼                          │
│            Softmax Routing                    │
│            (temperature=0.1)                  │
│                    │                          │
│                    ▼                          │
│           Projected Query                     │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Vector Search  │
          │  (Top-K Docs)   │
          └────────┬────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              RELEVANT DOCS                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ Doc A   │  │ Doc B   │  │ Doc C   │     │
│  │playbook │  │ example │  │  ref    │     │
│  └─────────┘  └─────────┘  └─────────┘     │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│                    LLM                       │
│         (Reasoning & Generation)             │
│                                              │
│   Context: Retrieved docs                    │
│   Task: User query                           │
│   Output: Generated code/answer              │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
              Final Response
```

## Database Schema (Simplified)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LDA DATABASE SCHEMA                               │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│     answers      │      │   qa_clusters    │      │    questions     │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ answer_id (PK)   │◀─┐   │ cluster_id (PK)  │   ┌─▶│ question_id (PK) │
│ text             │  │   │ name             │   │  │ text             │
│ source_file      │  │   │ description      │   │  │ length_type      │
│ record_id        │  │   └────────┬─────────┘   │  └──────────────────┘
└──────────────────┘  │            │             │
                      │            │             │
              ┌───────┴────────┐   │   ┌─────────┴────────┐
              │cluster_answers │   │   │cluster_questions │
              ├────────────────┤   │   ├──────────────────┤
              │ cluster_id(FK) │◀──┴──▶│ cluster_id (FK)  │
              │ answer_id (FK) │       │ question_id (FK) │
              └────────────────┘       └──────────────────┘

┌──────────────────────┐      ┌──────────────────────┐
│ multi_head_projections│      │    cluster_heads     │
├──────────────────────┤      ├──────────────────────┤
│ mh_projection_id (PK)│◀────▶│ head_id (PK)         │
│ model_id (FK)        │      │ mh_projection_id(FK) │
│ temperature          │      │ cluster_id (FK)      │
│ num_heads            │      │ centroid_path        │
│ recall_at_1          │      │ answer_emb_path      │
└──────────────────────┘      └──────────────────────┘

┌──────────────────┐      ┌──────────────────┐
│    embeddings    │      │embedding_models  │
├──────────────────┤      ├──────────────────┤
│ embedding_id(PK) │      │ model_id (PK)    │
│ model_id (FK)    │◀────▶│ name             │
│ entity_type      │      │ dimension        │
│ entity_id        │      └──────────────────┘
│ vector_path      │
└──────────────────┘
```

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW SUMMARY                                │
└─────────────────────────────────────────────────────────────────────────┘

  TRAINING PHASE                          INFERENCE PHASE
  ──────────────                          ───────────────

  JSON Q-A Pairs                          User Query
       │                                       │
       ▼                                       ▼
  ┌─────────────┐                        ┌─────────────┐
  │   Import    │                        │   Embed     │
  │  to SQLite  │                        │  (MiniLM)   │
  └──────┬──────┘                        └──────┬──────┘
         │                                      │
         ▼                                      ▼
  ┌─────────────┐                        ┌─────────────┐
  │   Embed     │                        │   Route     │
  │ Q-A pairs   │                        │  (softmax)  │
  └──────┬──────┘                        └──────┬──────┘
         │                                      │
         ▼                                      ▼
  ┌─────────────┐                        ┌─────────────┐
  │  Compute    │                        │   Project   │
  │ Centroids   │                        │   Query     │
  └──────┬──────┘                        └──────┬──────┘
         │                                      │
         ▼                                      ▼
  ┌─────────────┐                        ┌─────────────┐
  │   Store     │                        │   Search    │
  │   Heads     │                        │   Top-K     │
  └─────────────┘                        └──────┬──────┘
                                                │
                                                ▼
                                         Retrieved Docs
```
