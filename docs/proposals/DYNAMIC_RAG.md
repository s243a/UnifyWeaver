# Dynamic RAG

**Status:** Proposed
**Date:** 2025-01-22
**Prerequisites:** Skill indexing, embedding infrastructure, stream aggregation

## Problem Statement

Traditional RAG systems pre-compute embeddings and store them in a static index. This creates tension between:

1. **Freshness** - Pre-indexed content becomes stale when source documents change
2. **Coverage** - Only indexed content is searchable
3. **Cost** - Maintaining large indexes requires persistent storage and compute

Static indexes work well for stable corpora but poorly for dynamic content or real-time information needs.

**Goal:** Build a RAG pipeline that fetches and processes content at query time, ensuring answers always reflect current source material while remaining cost-effective for personal/small-scale deployment.

## Proposed Solution

A hybrid architecture that combines:
- **Semantic routing** via lightweight pre-computed embeddings (for finding relevant sources)
- **Live content fetching** at query time (for freshness)
- **On-demand synthesis** from current source material
- **Optional caching** for frequently-asked queries

### Architecture Overview

```
                    Query
                      │
                      ▼
              ┌───────────────┐
              │ Embedding API │  ← External service (flexible)
              └───────────────┘
                      │
                      ▼ query vector
              ┌───────────────┐
              │ Route/Search  │  ← Find relevant sources
              └───────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Source 1│ │ Source 2│ │ Source N│  ← Fetch live content
    └─────────┘ └─────────┘ └─────────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌───────────────┐
              │  Aggregate    │  ← Merge, rank, dedupe
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Synthesize   │  ← LLM generates answer
              └───────────────┘
                      │
                      ▼
                   Answer
```

### Design Flexibility

The architecture is modular with pluggable components:

| Component | Options | Trade-offs |
|-----------|---------|------------|
| **Embedding** | External API (OpenAI, Cohere), Local model (ONNX), Self-hosted | API: easy start, cost per query. Local: free but slower |
| **Search/Routing** | Semantic embeddings, grep/ripgrep, Elasticsearch, hybrid | Semantic: understands intent. Keyword: always current |
| **Source Fetch** | Local files, HTTP, database, search API | Local: fast. Remote: network latency |
| **Synthesis** | External LLM API, local model | API: quality + cost. Local: private but slower |

**Recommended starting point:** External embedding API + local file sources + external LLM API. This minimizes infrastructure while maintaining quality.

### Scaling Path

```
Personal (current)          Small Team              Production
─────────────────          ──────────              ──────────
Single machine             Multi-node              Distributed cluster
Local files                Shared storage          Sharded index
Sequential fetch           Parallel fetch          Parallel across nodes
~100ms routing             ~50ms routing           ~20ms routing
```

For distributed deployment:

```
              Query + Vector
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Node 1      Node 2      Node 3    ← Each has index shard
     (shard A)   (shard B)   (shard C)    Local SSD, CPU only
        │           │           │
        └───────────┼───────────┘
                    ▼
              Aggregator
                    │
                    ▼
                LLM API
```

Cloud nodes only need CPU + SSD (no GPUs). ML compute is outsourced to embedding/LLM APIs.

## UnifyWeaver Integration

Express the pipeline declaratively in Prolog:

```prolog
%% Source definitions (pluggable backends)
:- source(search, skills, [
    backend(semantic_embeddings),
    index_path('datasets/skills_qa/skill_index.npz'),
    embedding_api(openai)  % or: local, cohere, voyage
]).

:- source(search, documents, [
    backend(hybrid),        % semantic + keyword
    paths(['docs/', 'playbooks/'])
]).

%% Pipeline definition
dynamic_rag(Query, Answer) :-
    embed_query(Query, Vector),
    route_to_sources(Vector, Sources, TopK),
    parallel_fetch(Sources, Contents),
    aggregate_ranked(Query, Contents, Context),
    synthesize(Query, Context, Answer).

%% With caching layer
cached_dynamic_rag(Query, Answer) :-
    (   cache_lookup(Query, CachedAnswer, Score),
        Score > 0.9
    ->  Answer = CachedAnswer
    ;   dynamic_rag(Query, Answer),
        cache_store(Query, Answer)
    ).
```

### Stream Processing for Large Results

For queries returning many sources, use stream aggregation:

```prolog
stream_rag(Query, Answer) :-
    embed_query(Query, Vector),
    route_to_sources(Vector, Sources, 10),
    stream_fetch(Sources, ContentStream),
    windowed_aggregate(ContentStream, window(5), TopChunks),
    synthesize(Query, TopChunks, Answer).
```

This processes results as they arrive rather than waiting for all fetches to complete.

### Parallel Execution

Leverage `partitioning_parallel.pl` for concurrent fetching:

```prolog
parallel_fetch(Sources, Contents) :-
    partition_work(Sources, Batches, batch_size(4)),
    parallel_map(Batches, fetch_batch, Results),
    flatten(Results, Contents).
```

## Implementation Phases

### Phase 1: Local Dynamic RAG (Current)

- [x] Skill document indexing (`index_skills.py`)
- [x] Semantic routing to relevant skills
- [x] Live content fetching from local files
- [x] LLM synthesis via CLI
- [x] Cache fallback for fast responses

### Phase 2: Pluggable Embedding Backend

- [ ] Abstract embedding interface
- [ ] OpenAI embeddings adapter
- [ ] Cohere embeddings adapter
- [ ] Local ONNX fallback
- [ ] Batch embedding for efficiency

### Phase 3: Hybrid Search

- [ ] Combine semantic + keyword search
- [ ] Configurable blend weights
- [ ] Query analysis to choose strategy
- [ ] Fallback chain (semantic → keyword → full scan)

### Phase 4: Distributed Architecture

- [ ] Index sharding strategy
- [ ] Node coordination protocol
- [ ] Result aggregation
- [ ] Kubernetes deployment manifests
- [ ] Helm chart for easy deployment

### Phase 5: Confidence and Quality

- [ ] Answer confidence scoring
- [ ] Source attribution
- [ ] Uncertainty quantification
- [ ] Feedback loop for improvement

## Cost Analysis

### Personal Deployment (Current)

| Component | Cost | Notes |
|-----------|------|-------|
| Embedding (local) | $0 | CPU, ~100ms/query |
| Storage | $0 | Local SSD |
| LLM API | ~$0.01-0.05/query | Claude Haiku/Sonnet |
| **Total** | **~$0.01-0.05/query** | |

### With External Embedding API

| Component | Cost | Notes |
|-----------|------|-------|
| Embedding API | ~$0.0001/query | OpenAI ada-002 |
| Storage | $0 | Local SSD |
| LLM API | ~$0.01-0.05/query | |
| **Total** | **~$0.01-0.05/query** | Negligible embedding cost |

### Distributed (estimated)

| Component | Cost | Notes |
|-----------|------|-------|
| 3x cloud nodes | ~$150-300/mo | CPU + SSD instances |
| Embedding API | ~$10-50/mo | Volume dependent |
| LLM API | ~$100-500/mo | Volume dependent |
| **Total** | **~$300-1000/mo** | For moderate traffic |

## Hybrid Blending

For resource-constrained environments (personal computers), blend static and dynamic approaches:

```
Query
  │
  ├─────────────────┬────────────────────┐
  ▼                 ▼                    ▼
Static RAG       grep (fast)         Semantic route
(cached Q/A)        │                    │
  │                 ▼                    ▼
  │         Raw results (many)     Fetch live docs
  │                 │                    │
  │                 ▼                    │
  │         LLM Filter (Flash) ────────►│
  │         - Filter irrelevant         │
  │         - Smart chunking            │
  │         - Add metadata              │
  │         - Suggest follow-up greps   │
  │         - Generate Q/A pairs        │
  │                 │                    │
  └────────────┬────┴────────────────────┘
               ▼
         Blend/Rank results
               │
               ▼
         Final synthesis
```

### Trade-offs on Personal Hardware

| Resource | Constraint | Mitigation |
|----------|------------|------------|
| CPU | Can't process many results | Offload filtering to LLM API |
| Memory | Can't hold large indexes | Use grep for fresh, small static index for routing |
| Latency | User expects ~2-5s response | Parallel grep + API calls |

### LLM as Smart Filter

Send grep results to a fast/cheap model (e.g., Gemini Flash) for:

1. **Relevance filtering** - Remove results that match keywords but not intent
2. **Smart chunking** - Break large matches into meaningful segments
3. **Metadata extraction** - Identify file type, section headers, code vs prose
4. **Follow-up queries** - Suggest refined grep patterns for better results
5. **Q/A pair generation** - Identify what questions each result could answer

```prolog
hybrid_rag(Query, Answer) :-
    % Parallel: static lookup + fresh grep
    parallel([
        static_cache_lookup(Query, CacheResults),
        grep_search(Query, GrepResults)
    ], [CacheHits, RawGrep]),

    % Offload processing to LLM API
    llm_filter_and_enrich(RawGrep, Query, ProcessedGrep, FollowUps),

    % Optional: follow-up greps for refinement
    (   FollowUps \= []
    ->  maplist(grep_search, FollowUps, MoreResults),
        flatten([ProcessedGrep | MoreResults], AllGrep)
    ;   AllGrep = ProcessedGrep
    ),

    % Blend and rank
    blend_results(CacheHits, AllGrep, BlendedContext),
    synthesize(Query, BlendedContext, Answer).
```

## Learned Query-to-Grep Transformation

### The Problem

Natural language questions need conversion to effective grep patterns:

| Question | Naive grep | Effective grep |
|----------|------------|----------------|
| "How do I compile Prolog?" | `compile Prolog` | `compile\|transpile\|compiler_driver` |
| "authentication setup" | `authentication setup` | `auth\|jwt\|login\|password_hash` |
| "stream aggregation" | `stream aggregation` | `stream.*aggregat\|window\|batch` |

### Learnable Transformation

This is a transformation from question embeddings to a joint (grep pattern + result) space.

#### Embedding Target Options

| Approach | Dimensions | Description |
|----------|------------|-------------|
| Concat embeddings | 2×dim (768d) | `[grep_emb; answer_emb]` - expensive |
| Embed concatenated text | dim (384d) | `embed(grep + result)` - natural, efficient |
| **Structured with context** | dim (384d) | `embed(context + chunk + grep)` - **preferred** |

The third option embeds a single text that includes context, the code/content chunk, and the search pattern. This captures *why* the result matters, not just *what* it is.

#### Context Styles

**Natural language context (preferred)** - LLM generates human-readable framing:

*Javadoc-style intro:*
```
This predicate compiles Prolog code to target languages like Bash or Python.
Found in compiler_driver.pl, it's the main entry point for transpilation.

compile(Predicate, Options, Code) :-
    analyze_dependencies(Predicate, Deps),
    generate_target(Deps, Options, Code).

Search: compile|transpile|compiler_driver
```

*Usage-focused:*
```
When you need to convert Prolog rules into executable scripts, use this:

compile(Predicate, Options, Code) :-
    analyze_dependencies(Predicate, Deps),
    generate_target(Deps, Options, Code).

Typically called with: compile(my_pred/2, [output_dir('.')], Result)
Pattern: compile|transpile
```

*Why-it's-relevant:*
```
For transpiling declarative Prolog logic to imperative Bash scripts,
the compiler driver handles dependency analysis and code generation.

compile(Predicate, Options, Code) :- ...

This is the answer when users ask about Prolog-to-Bash compilation.
grep: compile|transpile|compiler_driver
```

**Database-style metadata** - Structured but less semantically rich:

```yaml
grep: compile|transpile|compiler_driver
file: src/unifyweaver/core/compiler_driver.pl
type: predicate_definition
chunk: |
  compile(Predicate, Options, Code) :-
      analyze_dependencies(Predicate, Deps),
      generate_target(Deps, Options, Code).
```

#### Side-by-Side Comparison

**Same content, different styles:**

| Aspect | Natural Language Context | Database-Style Metadata |
|--------|-------------------------|------------------------|
| **Format** | Prose, human-readable | YAML/JSON, machine-readable |
| **Semantics** | Captures "why" and "when" | Captures "what" and "where" |
| **Embedding quality** | Rich - model understands intent | Sparse - keys/values less meaningful |

**Example: JWT Authentication**

*Natural language:*
```
When building secure APIs that need stateless authentication, JWT tokens
are the recommended approach. This configuration sets up token-based auth
with bcrypt password hashing and automatic session management.

auth([backend(text_file), password_hash(bcrypt), token_type(jwt)])

Use this when you need: login endpoints, protected routes, token refresh.
Search: jwt|token|auth|password_hash|session
```

*Database-style:*
```yaml
grep: jwt|token|auth|password_hash
file: src/unifyweaver/glue/auth_backends.pl
category: authentication
predicate: auth/1
params: [backend, password_hash, token_type]
chunk: "auth([backend(text_file), password_hash(bcrypt), token_type(jwt)])"
```

**Example: Stream Aggregation**

*Natural language:*
```
For processing large datasets that don't fit in memory, stream aggregation
lets you compute running totals, moving averages, and windowed statistics
as data flows through. This is essential for real-time analytics.

windowed_aggregate(Stream, window(Size), Aggregator, Results)

Handles: time-series data, log processing, sliding window computations.
Pattern: stream.*window|aggregat|batch|sliding
```

*Database-style:*
```yaml
grep: stream.*window|aggregat|batch
file: src/unifyweaver/streams/aggregation.pl
type: predicate
arity: 4
inputs: [stream, window_spec, aggregator]
output: results
chunk: "windowed_aggregate(Stream, window(Size), Aggregator, Results)"
```

Natural language context is preferred because:
- Embeddings capture semantic meaning better from prose
- The "why" and "when to use" are explicitly encoded
- More robust to varied question phrasings

**Note:** From the LLM's perspective, the best context is **brief and terse**, similar to a skill document:

```
Compile Prolog to Bash scripts.

compile(Predicate, Options, Code)

Use when: transpiling declarative logic to shell scripts.
See: compiler_driver.pl
Search: compile|transpile|compiler_driver
```

The LLM should generate concise context, not verbose prose:
- Fits embedding model token limits
- Less noise in the embedding
- Skill documents already model this format well
- Dense information, no filler
- Faster to generate, cheaper API cost

#### Other Metadata Types

Additional metadata (in either prose or structured format) may include:

| Metadata | Prose Example | Structured Example |
|----------|---------------|-------------------|
| **Line number** | "Found at line 142" | `line: 142` |
| **File name** | "From compiler_driver.pl" | `file: compiler_driver.pl` |
| **Related docs** | "See also: skill_deployment.md, EXTENDED_README.md" | `related: [skill_deployment.md, EXTENDED_README.md]` |
| **Module/namespace** | "In the unifyweaver.core module" | `module: unifyweaver.core` |
| **Dependencies** | "Requires: stream_utils, json_parser" | `deps: [stream_utils, json_parser]` |
| **Version/date** | "Added in v2.3, updated Jan 2025" | `version: 2.3, updated: 2025-01` |

Example combining both styles:

```
Compile Prolog to Bash scripts.
From compiler_driver.pl, line 142. See also: skill_unifyweaver_compile.md

compile(Predicate, Options, Code)

Use when: transpiling declarative logic to shell scripts.
Requires: analyzer, target_bash
Search: compile|transpile|compiler_driver
```

#### Positional Priority in Context

Position matters for both embedding models and LLMs:

**Embedding models:**
- Often weight the beginning of text more heavily
- Truncation typically cuts from the end
- Key semantic content should appear early

**LLMs (lost-in-the-middle phenomenon):**
- Strong attention to beginning and end
- Information in the middle may be underweighted
- Critical details should bookend the context

**Recommended structure:**

```
[PURPOSE - what this does]           ← Start: high attention
[KEY CODE/CONTENT]                   ← Early: captured by embeddings
[USAGE CONTEXT - when to use]        ← Middle: supporting info
[METADATA - file, line, deps]        ← Middle-end: reference info
[SEARCH PATTERN]                     ← End: high attention, easy to extract
```

Example with positional awareness:

```
Compile Prolog predicates to executable Bash scripts.    ← Purpose first

compile(Predicate, Options, Code) :-                     ← Key content early
    analyze_dependencies(Predicate, Deps),
    generate_target(Deps, Options, Code).

Use when transpiling declarative logic. Handles          ← Usage middle
recursive predicates and dependency analysis.
From compiler_driver.pl:142. Requires: analyzer.         ← Metadata

Search: compile|transpile|compiler_driver|bash.*target   ← Pattern at end
```

This structure ensures:
- Embeddings capture the purpose and key content
- LLMs see purpose at start, search pattern at end
- Metadata is present but doesn't dominate attention

#### Markdown-Structured Context

Use markdown formatting for context that is both human-readable (e.g., viewable in Obsidian) and embedding-friendly:

```markdown
# Compile Prolog to Bash

> **Purpose:** Transpile declarative Prolog predicates to executable shell scripts

```prolog
compile(Predicate, Options, Code) :-
    analyze_dependencies(Predicate, Deps),
    generate_target(Deps, Options, Code).
```

> **Use when:** Converting logic rules to bash for deployment
> **File:** compiler_driver.pl:142
> **Related:** skill_unifyweaver_compile.md

<!-- search: compile|transpile|compiler_driver -->
```

Benefits of this format:

| Aspect | Benefit |
|--------|---------|
| **Human readable** | View/edit in Obsidian, VS Code, any markdown renderer |
| **Prose-like** | Embedding models understand natural prose structure |
| **Structured metadata** | Callouts (`>`) and comments (`<!-- -->`) for parsing |
| **Code blocks** | Fenced code preserves formatting, indicates language |
| **Hierarchical** | Headings provide semantic structure |

Markdown elements and their roles:

| Element | Example | Role |
|---------|---------|------|
| `#` Heading | `# Compile Prolog` | Purpose/title (high attention) |
| `>` Callout | `> **Purpose:** ...` | Metadata, visually distinct |
| ``` Code fence | ````prolog ... ```` | Key content, language-tagged |
| `<!-- -->` Comment | `<!-- search: ... -->` | Machine-readable, minimal embedding impact |
| `**bold**` | `**File:**` | Key labels within prose |

This dual-purpose format means:
- The same file works as documentation AND training data
- No separate "structured" vs "readable" versions needed
- Metadata is embedded in context, not separate

#### Training Data Format

```jsonl
{"question": "How do I compile Prolog to bash?", "target_text": "This predicate compiles Prolog code to target languages... compile(Predicate, Options, Code) :- ... Search: compile|transpile|compiler_driver"}
{"question": "set up JWT authentication", "target_text": "For stateless authentication, configure JWT tokens... auth([token_type(jwt)...]) Pattern: jwt|token|auth.*backend"}
```

The minimal transformation learns:
- **W matrix**: Maps question embedding → target text embedding
- At inference: find nearest target text, extract grep pattern, run search

### Bootstrap Process

1. Start with hand-crafted patterns for common queries
2. Log which grep patterns yield useful results
3. Train transformation on (question, effective_pattern) pairs
4. Iterate: model suggests patterns → user feedback → retrain

### Q/A Generation from Grep Results

Instead of just filtering grep results, generate Q/A pairs:

```
Grep result: "generate_dockerfile(Service, Options, Dockerfile)"

LLM generates:
- Q: "How do I generate a Dockerfile?"
- Q: "What function creates Docker containers?"
- Q: "How do I containerize a service?"
```

This dynamically enriches the Q/A cache with questions the corpus can answer, discovered through grep exploration.

### Related Questions

When answering a query, identify what related questions the grep results could also answer:

```
User asks: "How do I deploy to Kubernetes?"

Grep finds: deployment_glue.pl

LLM identifies related questions:
- "How do I generate Helm charts?"
- "How do I configure K8s replicas?"
- "How do I set resource limits?"
```

These become suggestions for the user or seeds for cache warming.

## Open Questions

1. **Cache invalidation** - How to detect when source documents change and invalidate cached answers?

2. **Source trust** - When fetching from multiple sources, how to weight by reliability?

3. **Latency budget** - What's acceptable query latency? This determines parallelism strategy.

4. **Chunk strategy** - For long documents, how to chunk for optimal retrieval?

5. **Grep pattern learning** - How much training data needed for effective query→grep transformation?

6. **Q/A generation quality** - How to validate LLM-generated questions match the source content?

## Related Work

- Retrieval-Augmented Generation (RAG) patterns
- Federated search architectures
- Stream processing systems
- UnifyWeaver stream aggregation (`skill_stream_aggregation.md`)
- UnifyWeaver parallel execution (`partitioning_parallel.pl`)

## Files

| File | Purpose |
|------|---------|
| `scripts/index_skills.py` | Build semantic index for routing |
| `scripts/live_rag_query.py` | Query interface with live fetching |
| `scripts/export_skills_qa_model.py` | Static Q/A cache generation |
| `scripts/score_answer_confidence.py` | Confidence scoring pipeline |
