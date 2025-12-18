# Proposal: Q/A Knowledge Graph for Learning Paths

**Status:** Implemented (Phase 1 & 2)
**Version:** 1.0
**Date:** 2025-12-17
**Extends:** [LDA_DATABASE_SCHEMA.md](LDA_DATABASE_SCHEMA.md)
**Implementation:** `src/unifyweaver/targets/python_runtime/kg_topology_api.py`

## Executive Summary

Extend the Q/A database with typed relationships between Q/A pairs to create a knowledge graph. This enables:
- Learning path navigation (prerequisites → main → extensions)
- Contextual exploration (find related concepts)
- Model search space (explore options when answering)

The key insight is that Q/A pairs don't exist in isolation - they have conceptual dependencies and natural progressions that a knowledge graph can capture.

## Motivation

### Current State

Semantic search finds the nearest Q/A pair to a query:
```
Query → Embed → Project → Find nearest answer → Return
```

This is stateless - each query is independent with no understanding of:
- What the user already knows
- What prerequisites might be missing
- Where they might want to go next
- How concepts relate to each other

### Desired State

A knowledge graph layer that enables:
```
Query → Find nearest Q/A → Traverse graph → Understand context
                              ↓
                    Foundational: "What does this depend on?"
                    Preliminary: "What should I do first?"
                    Compositional: "How can I extend this?"
                    Transitional: "What comes next?"
```

### Use Cases

1. **Learning paths**: Guide users through prerequisite → main → extension
2. **Debugging help**: If user is confused, suggest foundational concepts
3. **Exploration**: Show related concepts for deeper understanding
4. **Agent planning**: Model can explore graph to understand task context

## Relationship Types

### Core Categories

| Type | Direction | Description | Example |
|------|-----------|-------------|---------|
| **foundational** | A ← B | B depends on principles from A | JWT auth ← cryptographic signing |
| **preliminary** | A ← B | B requires completing A first | parse CSV ← install pandas |
| **compositional** | A → B | B extends or builds upon A | basic query → query with joins |
| **transitional** | A → B | B is a natural next step after A | create user → assign permissions |

### Scope Relations (Specificity)

| Type | Direction | Description | Example |
|------|-----------|-------------|---------|
| **refined** | A → B | B is a more specific variant of A | read CSV → read CSV with headers |
| **general** | A ← B | A is broader in scope than B | parse delimited data ← parse CSV |

**Scope** is about breadth of applicability:
- `refined`: Same domain, narrower focus (CSV → CSV edge cases)
- `general`: Same domain, broader applicability (CSV → all delimited formats)

### Abstraction Relations

| Type | Direction | Description | Example |
|------|-----------|-------------|---------|
| **generalization** | A → B | B is an abstract pattern derived from A | JWT refresh → Token refresh pattern |
| **implementation** | A ← B | A is code that realizes pattern B | JWT refresh code ← Token pattern |
| **axiomatization** | A → B | B is abstract theory derived from A | Arithmetic → Ring theory |
| **instance** | A ← B | A is a domain that satisfies theory B | Arithmetic ← Ring theory |
| **example** | A ← B | A illustrates/demonstrates concept B | JWT tutorial ← Token-based auth |

**Abstraction** has multiple dimensions:

- `generalization` / `implementation`: **Pattern ↔ Code** (design patterns, architectural patterns)
- `axiomatization` / `instance`: **Theory ↔ Domain** (mathematical structures, formal systems)
- `example`: **Pedagogical** (this demonstrates that concept for learning)

### Scope vs Abstraction: Key Distinction

These are orthogonal dimensions:

```
                    Abstraction Level
                    (pattern ↔ implementation)
                           ↑
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        │  "Token Pattern" │  "Auth Pattern"  │  ← more abstract
        │        │         │        │         │
        │        │ instance│        │         │
        │        ▼         │        ▼         │
        │   "JWT Refresh"  │  "OAuth Flow"   │  ← concrete
        │        │         │        │         │
        │        │ refined │        │         │
        │        ▼         │        ▼         │
        │  "JWT Refresh    │  "OAuth with    │  ← more specific
        │   with Rotation" │   PKCE"         │
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
               ◀───────────┴───────────▶
                     Scope
               (general ↔ refined)

### Directionality

Relations are stored as directed edges from source to target:
- `(A, B, foundational)` means "A is foundational to B" (A ← B in learning order)
- `(A, B, transitional)` means "B is a natural next step after A" (A → B in learning order)

### Relation Semantics

**Foundational** - Principles and concepts:
- Theoretical knowledge the answer assumes
- "You need to understand X to use Y"
- Example: Understanding async/await before using CompletableFuture

**Preliminary** - Practical prerequisites:
- Setup steps, installations, configurations
- "You need to do X before you can do Y"
- Example: Creating a database before running migrations

**Compositional** - Extensions and combinations:
- Building on the base concept
- "Once you know X, you can extend it with Y"
- Example: Basic CRUD → adding validation → adding caching

**Transitional** - Workflow progression:
- Natural next steps in a workflow
- "After doing X, you typically do Y"
- Example: Write tests → run tests → deploy

**Refined** - More specific variant:
- Narrower scope, same domain
- "X, but with additional constraints or edge cases"
- Example: Read CSV → Read CSV with custom delimiters

**General** - Broader scope:
- Wider applicability, same domain
- "The broader category that X belongs to"
- Example: Parse delimited data ← Parse CSV

**Generalization** - Abstract pattern:
- Moving up the abstraction ladder (pattern level)
- "The abstract pattern behind X"
- Example: JWT refresh → Token refresh pattern

**Implementation** - Code realizing pattern:
- Code that follows a design pattern
- "This code realizes pattern X"
- Example: JWT refresh code ← Token pattern

**Axiomatization** - Abstract theory:
- Moving up the abstraction ladder (theory level)
- "The formal theory behind X"
- Example: Arithmetic → Ring theory

**Instance** - Domain satisfying theory:
- A concrete domain that satisfies abstract axioms
- "X is an instance of theory Y"
- Example: Arithmetic ← Ring theory

**Example** - Pedagogical illustration:
- Demonstrates a concept for learning
- "This illustrates how to do X"
- Example: JWT tutorial ← Token-based auth concept

## Database Schema

### Recommended Approach: Bind to Answers (Examples)

In the playbook context:
- **Answers = Examples** (concrete playbook artifacts)
- **Questions = Discovery paths** (ways to find examples)
- **Relations = Conceptual connections between examples**

Since examples are the primary unit being curated, relations should bind to answers:

```sql
-- Extend answer_relations with learning/conceptual relation types
-- This reuses the existing table structure from LDA_DATABASE_SCHEMA.md

-- New relation types to add:
--   Learning flow:
--     'foundational'    - source is a foundational concept for target
--     'preliminary'     - source is a prerequisite step for target
--     'compositional'   - target extends/builds upon source
--     'transitional'    - target is a natural next step after source
--   Scope:
--     'refined'         - target is a more specific variant of source
--     'general'         - source is broader in scope than target
--   Abstraction:
--     'generalization'  - target is an abstract pattern of source
--     'implementation'  - source is code that realizes pattern target
--     'axiomatization'  - target is abstract theory of source
--     'instance'        - source is domain satisfying theory target
--     'example'         - source illustrates/demonstrates target

-- The existing answer_relations table already supports this:
CREATE TABLE answer_relations (
    relation_id INTEGER PRIMARY KEY,
    from_answer_id INTEGER REFERENCES answers(answer_id),
    to_answer_id INTEGER REFERENCES answers(answer_id),
    relation_type TEXT NOT NULL,
    metadata TEXT,                        -- JSON for extra info
    UNIQUE(from_answer_id, to_answer_id, relation_type)
);

-- Just add the new relation types to the existing table!
-- Existing types: chunk_of, summarizes, variant_of, next_chunk, related_to, etc.
-- New types: foundational, preliminary, compositional, transitional
```

**Advantage:** No new tables needed - just extend `answer_relations` with new relation types.

### Why Answers (Examples) as the Primary Binding?

| Binding | Pros | Cons |
|---------|------|------|
| **Answers (Examples)** | Primary artifact; questions are just discovery paths; examples are what you curate | - |
| Clusters | Groups related Q/A | Extra indirection; cluster may have multiple answers |
| Questions | Fine-grained | Too granular; same example reached by many questions |

**The insight:** You're building a library of examples. Questions help users find examples, but the conceptual relationships are between the examples themselves.

```
Example: "JWT Authentication"
  ├── foundational: "Cryptographic Signing", "HTTP Headers"
  ├── preliminary: "Install JWT Library"
  ├── compositional: "Refresh Tokens", "Role-Based Access"
  └── transitional: "Secure Endpoints", "Token Revocation"
```

### Coexistence with Document Relations

The existing `answer_relations` already handles **document structure**:
- `chunk_of` - document chunking
- `summarizes` - abstraction levels
- `variant_of` - different text representations

The new relation types handle **conceptual structure**:

*Learning flow:*
- `foundational` - learning dependencies
- `preliminary` - practical prerequisites
- `compositional` - extensions
- `transitional` - workflow progression

*Scope:*
- `refined` - more specific variants
- `general` - broader scope

*Abstraction:*
- `generalization` - abstract patterns (design level)
- `implementation` - code realizing patterns
- `axiomatization` - abstract theories (formal level)
- `instance` - domains satisfying theories
- `example` - pedagogical illustrations

These coexist naturally in the same table - just different relation_type values.

### Alternative: Flexible Entity Binding

Relations could bind to different entity types - questions, answers, or clusters:

```sql
-- Flexible relations that can bind to any entity type
CREATE TABLE knowledge_relations (
    relation_id INTEGER PRIMARY KEY,

    -- Source can be question, answer, or cluster
    source_type TEXT NOT NULL CHECK(source_type IN ('question', 'answer', 'cluster')),
    source_id INTEGER NOT NULL,

    -- Target can be question, answer, or cluster
    target_type TEXT NOT NULL CHECK(target_type IN ('question', 'answer', 'cluster')),
    target_id INTEGER NOT NULL,

    relation_type TEXT NOT NULL CHECK(relation_type IN (
        -- Learning flow
        'foundational',
        'preliminary',
        'compositional',
        'transitional',
        -- Scope
        'refined',
        'general',
        -- Abstraction
        'generalization',
        'implementation',
        'axiomatization',
        'instance',
        'example'
    )),
    strength REAL DEFAULT 1.0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_type, source_id, target_type, target_id, relation_type)
);

-- Indexes for flexible lookup
CREATE INDEX idx_knowledge_rel_source ON knowledge_relations(source_type, source_id);
CREATE INDEX idx_knowledge_rel_target ON knowledge_relations(target_type, target_id);
CREATE INDEX idx_knowledge_rel_type ON knowledge_relations(relation_type);
```

**When to bind to each type:**

| Bind to | Use when |
|---------|----------|
| Question | Follow-up relates to specific phrasing or intent |
| Answer | Follow-up relates to the content/solution |
| Cluster | Follow-up relates to the whole concept |

**Example:**
```python
# "How do I read CSV?" might have preliminary: "install pandas"
# But the answer might have compositional: "handle encoding issues"
db.add_relation('question', q_id, 'cluster', setup_pandas, 'preliminary')
db.add_relation('answer', a_id, 'cluster', csv_encoding, 'compositional')
```

### Knowledge Gap Detection

**Key insight:** A Q/A pair with NO follow-up relations indicates a potential knowledge gap.

```sql
-- Find clusters with no outgoing relations (missing next steps)
SELECT c.cluster_id, c.name
FROM qa_clusters c
LEFT JOIN knowledge_relations kr ON
    kr.source_type = 'cluster' AND kr.source_id = c.cluster_id
WHERE kr.relation_id IS NULL;

-- Find clusters with no incoming foundational relations (orphaned concepts)
SELECT c.cluster_id, c.name
FROM qa_clusters c
LEFT JOIN knowledge_relations kr ON
    kr.target_type = 'cluster' AND
    kr.target_id = c.cluster_id AND
    kr.relation_type = 'foundational'
WHERE kr.relation_id IS NULL;
```

**Gap types:**
- **No foundational**: Concept appears standalone - either truly basic or missing prerequisites
- **No preliminary**: No setup steps - either self-contained or missing practical prerequisites
- **No compositional**: No extensions - either complete or missing advanced topics
- **No transitional**: Dead end - no natural next steps defined

**Actionable insight:** Gaps can drive content creation priorities:
```python
def find_knowledge_gaps(db):
    """Find clusters that need more relations."""
    gaps = {
        'orphaned': db.query_clusters_without('foundational', 'incoming'),
        'dead_ends': db.query_clusters_without('transitional', 'outgoing'),
        'no_extensions': db.query_clusters_without('compositional', 'outgoing'),
    }
    return gaps
```

**Trade-offs for flexible binding:**
- More flexible but more complex queries
- Can represent richer relationships
- Requires careful thought about which entity to bind to

**Recommendation:** Start with answer-level relations (extending `answer_relations`). Use flexible binding only if you need question-specific relations.

## API Extensions

```python
class LDAProjectionDB:
    # ... existing methods ...

    # Example (Answer) Relations - extends existing add_relation method
    def add_example_relation(
        self,
        from_answer_id: int,
        to_answer_id: int,
        relation_type: str,  # 'foundational', 'preliminary', 'compositional', 'transitional'
        metadata: dict = None
    ) -> int:
        """Create a conceptual relation between examples."""
        return self.add_relation(from_answer_id, to_answer_id, relation_type, metadata)

    def get_example_relations(
        self,
        answer_id: int,
        relation_type: str = None,
        direction: str = 'outgoing'  # 'outgoing', 'incoming', 'both'
    ) -> List[dict]:
        """Get conceptual relations for an example."""
        ...

    def get_foundational(self, answer_id: int) -> List[dict]:
        """Get concepts this example depends on."""
        return self.get_example_relations(answer_id, 'foundational', 'incoming')

    def get_prerequisites(self, answer_id: int) -> List[dict]:
        """Get steps required before this example."""
        return self.get_example_relations(answer_id, 'preliminary', 'incoming')

    def get_extensions(self, answer_id: int) -> List[dict]:
        """Get examples that extend this one."""
        return self.get_example_relations(answer_id, 'compositional', 'outgoing')

    def get_next_steps(self, answer_id: int) -> List[dict]:
        """Get natural next steps after this example."""
        return self.get_example_relations(answer_id, 'transitional', 'outgoing')

    def get_learning_path(
        self,
        answer_id: int,
        include_foundational: bool = True,
        include_prerequisites: bool = True
    ) -> List[dict]:
        """Get ordered learning path leading to this example."""
        ...

    # Enhanced search with graph context
    def search_with_context(
        self,
        query_text: str,
        projection_id: int,
        top_k: int = 5,
        include_relations: bool = True
    ) -> List[dict]:
        """
        Search with knowledge graph context.

        Returns results with related examples:
        {
            "answer_id": 1,
            "score": 0.95,
            "foundational": [...],
            "prerequisites": [...],
            "extensions": [...],
            "next_steps": [...]
        }
        """
        ...
```

## Example Usage

### Creating Relations Between Examples

```python
db = LDAProjectionDB("lda.db", "embeddings/")

# Add examples (answers) to the database
jwt_basics = db.add_answer("playbooks/auth/jwt.md", "JWT authentication example...")
crypto_signing = db.add_answer("playbooks/crypto/signing.md", "Cryptographic signing...")
http_headers = db.add_answer("playbooks/http/headers.md", "HTTP headers basics...")
jwt_refresh = db.add_answer("playbooks/auth/refresh.md", "JWT refresh tokens...")
setup_auth_lib = db.add_answer("playbooks/setup/auth_lib.md", "Install auth library...")
secure_endpoints = db.add_answer("playbooks/auth/secure.md", "Securing endpoints...")

# Add conceptual relations between examples
db.add_example_relation(crypto_signing, jwt_basics, 'foundational')  # signing is foundational to JWT
db.add_example_relation(http_headers, jwt_basics, 'foundational')    # headers is foundational to JWT
db.add_example_relation(setup_auth_lib, jwt_basics, 'preliminary')   # setup required before JWT
db.add_example_relation(jwt_basics, jwt_refresh, 'compositional')    # refresh extends JWT
db.add_example_relation(jwt_basics, secure_endpoints, 'transitional') # securing is next step
```

### Querying the Graph

```python
# User searches for JWT authentication
results = db.search_with_context("How do I add JWT auth?", proj_id)

# Result includes graph context:
{
    "answer_id": 42,
    "cluster_id": jwt_basics,
    "score": 0.92,
    "text": "To implement JWT authentication...",
    "foundational": [
        {"cluster": "crypto_signing", "name": "Cryptographic Signing"},
        {"cluster": "http_headers", "name": "HTTP Headers Basics"}
    ],
    "prerequisites": [
        {"cluster": "setup_auth_lib", "name": "Install JWT Library"}
    ],
    "extensions": [
        {"cluster": "jwt_refresh", "name": "JWT Refresh Tokens"}
    ],
    "next_steps": [
        {"cluster": "secure_endpoints", "name": "Secure API Endpoints"}
    ]
}
```

### Agent Usage

```python
# Agent can use graph for planning
def plan_task(task_description: str):
    results = db.search_with_context(task_description, proj_id)
    main_answer = results[0]

    # Check if user might need prerequisites
    if user_seems_confused():
        suggest(main_answer['foundational'])

    # After completing task, suggest next steps
    if task_completed():
        suggest(main_answer['next_steps'])
```

## Visualization

The knowledge graph can be visualized as a directed graph:

```
                    ┌─────────────────┐
                    │ crypto_signing  │
                    │  (foundational) │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ http_headers │───▶│   jwt_basics    │───▶│  jwt_refresh    │
│(foundational)│    │     (main)      │    │ (compositional) │
└──────────────┘    └────────┬────────┘    └─────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│setup_auth_lib│    │secure_endpoints │    │token_revocation │
│(preliminary) │    │ (transitional)  │    │ (transitional)  │
└──────────────┘    └─────────────────┘    └─────────────────┘
```

## Integration with Smoothing Basis

The knowledge graph structure could enhance smoothing basis projection:

1. **Similar clusters by relation**: Clusters with the same foundational concepts might benefit from shared basis
2. **Relation-weighted similarity**: Use graph distance as additional similarity signal
3. **Hierarchical basis**: Foundational concepts could define basis, derived concepts learn coefficients

This is speculative but worth exploring.

## Implementation Status

### Phase 1: Schema & API (Complete)
- [x] `answer_relations` table extended with 11 relation types
- [x] Indexes for efficient traversal
- [x] `add_kg_relation()` - create relations between answers
- [x] `get_relations()` - query relations by type and direction
- [x] Convenience methods: `get_foundational()`, `get_prerequisites()`, `get_extensions()`, `get_next_steps()`, `get_refined()`, `get_general()`, `get_generalizations()`, `get_implementations()`, `get_instances()`, `get_examples()`
- [x] `search_with_context()` - semantic search with graph context
- [x] `get_learning_path()` - ordered path including prerequisites

### Phase 2: Semantic Interfaces (Complete)
- [x] Interface schema with centroids and topics
- [x] `create_interface()`, `update_interface()`, `delete_interface()`
- [x] `map_query_to_interface()` - softmax routing over interface centroids
- [x] `search_via_interface()` - search with optional optimizations
- [x] Interface metrics and health monitoring

### Phase 3: Data Population (Future)
- [ ] Define relations for existing playbook clusters
- [ ] Create tooling to suggest relations based on content similarity

### Phase 4: Agent Integration (Future)
- [ ] Expose graph traversal via MCP tools
- [ ] Add learning path suggestions to search results

## Open Questions

1. **Relation discovery**: Can we automatically infer relations from content?
   - Foundational: concepts mentioned in answer but not explained
   - Transitional: "next, you might want to..." phrases

2. **Strength weighting**: How to determine relation strength?
   - Manual curation?
   - Based on co-occurrence in user sessions?

3. **Bidirectional relations**: Should `foundational` and `depends_on` be separate or implied inverses?

4. **Cross-domain relations**: Can relations span different playbook domains?

## References

- [LDA_DATABASE_SCHEMA.md](LDA_DATABASE_SCHEMA.md) - Base schema this extends
- [SMOOTHING_BASIS_PROJECTION.md](SMOOTHING_BASIS_PROJECTION.md) - Potential integration
- [SEED_QUESTION_TOPOLOGY.md](SEED_QUESTION_TOPOLOGY.md) - Hash-based anchor linking and provenance
- [SMALL_WORLD_ROUTING.md](SMALL_WORLD_ROUTING.md) - Distributed routing using relation topology
- Knowledge graphs in QA systems literature
