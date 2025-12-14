# Proposal: LDA Projection Database Schema

## Overview

Store embeddings, transformations, and Q-A mappings in a database to enable agents to find useful playbook examples via semantic search with learned projections.

## Use Case

1. Agent receives a task description
2. System embeds the task as a query (input model)
3. Projects query using learned W matrix
4. Compares to answer embeddings (output model)
5. Returns most similar playbook examples

## Asymmetric Embeddings

Different models can be used for queries vs answers:

| Role | Typical Length | Example Model | Dimension |
|------|----------------|---------------|-----------|
| Input (queries) | Short (2-50 words) | all-MiniLM-L6-v2 | 384 |
| Output (answers) | Long (100-1000+ words) | ModernBERT | 1024 |

The projection W transforms from input space to output space:
```
W: ℝ^(input_dim) → ℝ^(output_dim)
W shape: (output_dim, input_dim)

projected_query = W @ query_embedding  # (output_dim,)
similarity = projected_query · answer_embedding
```

This allows:
- Fast, lightweight model for queries
- High-capacity model for long documents
- W learns the cross-space mapping

## Schema

### Core Tables

```sql
-- Answer documents (playbook examples)
-- Supports hierarchy: chunks reference parent documents
CREATE TABLE answers (
    answer_id INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES answers(answer_id),  -- NULL for root documents
    source_file TEXT NOT NULL,           -- e.g., "playbooks/examples_library/csv_examples.md"
    record_id TEXT,                       -- e.g., "unifyweaver.execution.csv_data_source"
    text TEXT NOT NULL,                   -- answer text (may vary by model)
    text_variant TEXT DEFAULT 'default',  -- 'default', 'short', 'long', or model name
    chunk_index INTEGER,                  -- position within parent (NULL for root)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Answer relationships (graph structure)
CREATE TABLE answer_relations (
    relation_id INTEGER PRIMARY KEY,
    from_answer_id INTEGER REFERENCES answers(answer_id),
    to_answer_id INTEGER REFERENCES answers(answer_id),
    relation_type TEXT NOT NULL,
    metadata TEXT,                        -- JSON for extra info
    UNIQUE(from_answer_id, to_answer_id, relation_type)
);

-- Relation types:
--
-- Hierarchy:
--   'chunk_of'     - this is a chunk of the target (chunk → full doc)
--   'summarizes'   - this summarizes the target (summary → full doc)
--   'abbreviates'  - abbreviated version (short → long)
--
-- Sequence:
--   'next_chunk'   - next chunk in sequence (chunk N → chunk N+1)
--   'prev_chunk'   - previous chunk (chunk N → chunk N-1)
--
-- Variants:
--   'variant_of'   - same content, different text (model-specific)
--   'translates'   - language translation (EN → FR, etc.)
--                    metadata: {"from_lang": "en", "to_lang": "fr"}
--
-- Semantic:
--   'related_to'   - semantically related but distinct
--   'see_also'     - cross-reference
--   'supersedes'   - newer version replaces older
--
-- Examples:
--   (2, 1, 'chunk_of')      - answer 2 is a chunk of answer 1
--   (3, 2, 'next_chunk')    - answer 3 follows answer 2 in sequence
--   (4, 1, 'summarizes')    - answer 4 is a summary of answer 1
--   (5, 1, 'translates', {"from_lang": "en", "to_lang": "es"})
--   (6, 1, 'variant_of')    - answer 6 is a ModernBERT-optimized variant

CREATE INDEX idx_relations_from ON answer_relations(from_answer_id);
CREATE INDEX idx_relations_to ON answer_relations(to_answer_id);
CREATE INDEX idx_relations_type ON answer_relations(relation_type);

-- Query questions
CREATE TABLE questions (
    question_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    length_type TEXT CHECK(length_type IN ('short', 'medium', 'long')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Q-A cluster mappings
CREATE TABLE qa_clusters (
    cluster_id INTEGER PRIMARY KEY,
    name TEXT,                            -- e.g., "csv_data_source"
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE cluster_answers (
    cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
    answer_id INTEGER REFERENCES answers(answer_id),
    PRIMARY KEY (cluster_id, answer_id)
);

CREATE TABLE cluster_questions (
    cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
    question_id INTEGER REFERENCES questions(question_id),
    PRIMARY KEY (cluster_id, question_id)
);
```

### Embedding Tables

```sql
-- Embedding models
CREATE TABLE embedding_models (
    model_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,            -- e.g., "all-MiniLM-L6-v2"
    dimension INTEGER NOT NULL,           -- e.g., 384
    backend TEXT,                         -- 'python', 'rust', 'go'
    max_tokens INTEGER,
    notes TEXT
);

-- Stored embeddings
CREATE TABLE embeddings (
    embedding_id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES embedding_models(model_id),
    entity_type TEXT CHECK(entity_type IN ('answer', 'question')),
    entity_id INTEGER,                    -- references answers or questions
    vector BLOB NOT NULL,                 -- numpy array serialized
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, entity_type, entity_id)
);

-- Index for fast lookup
CREATE INDEX idx_embeddings_entity ON embeddings(model_id, entity_type, entity_id);
```

### Projection Tables

```sql
-- Trained W matrices
-- W transforms from input embedding space to output embedding space
-- Shape: (output_dim, input_dim)
CREATE TABLE projections (
    projection_id INTEGER PRIMARY KEY,
    input_model_id INTEGER REFERENCES embedding_models(model_id),   -- for questions
    output_model_id INTEGER REFERENCES embedding_models(model_id),  -- for answers
    name TEXT,                            -- e.g., "v1_playbook_examples"
    W_matrix BLOB NOT NULL,               -- numpy array (output_dim × input_dim)
    lambda_reg REAL,
    ridge REAL,
    num_clusters INTEGER,
    num_queries INTEGER,
    -- Evaluation metrics
    recall_at_1 REAL,
    mrr REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Note: When input_model_id == output_model_id, W is square (d × d)
-- When different, W is rectangular (d_out × d_in)
-- Example: input=all-MiniLM-L6-v2 (384), output=ModernBERT (1024)
--          W shape = (1024, 384)

-- Track which clusters were used for training
CREATE TABLE projection_clusters (
    projection_id INTEGER REFERENCES projections(projection_id),
    cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
    PRIMARY KEY (projection_id, cluster_id)
);
```

### Query Log (for continuous improvement)

```sql
-- Log queries for future training data
CREATE TABLE query_log (
    log_id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    model_id INTEGER REFERENCES embedding_models(model_id),
    projection_id INTEGER REFERENCES projections(projection_id),
    results TEXT,                         -- JSON array of answer_ids
    selected_answer_id INTEGER,           -- which one the agent used
    was_helpful BOOLEAN,                  -- feedback
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Storage Options

### Option 1: SQLite + numpy files

Simple, portable:
- SQLite for metadata and mappings
- `.npy` files for large vectors/matrices
- `vector BLOB` stores path to file

### Option 2: SQLite with sqlite-vss

Vector similarity search built-in:
- Uses sqlite-vss extension
- Can do `SELECT * FROM embeddings WHERE vss_search(vector, ?)`
- More complex setup

### Option 3: DuckDB

Analytical queries, good numpy integration:
- Native array types
- Fast aggregations
- Parquet export

**Recommendation**: Start with Option 1 (SQLite + numpy), migrate to Option 2 if vector search becomes a bottleneck.

## API Design

```python
class LDAProjectionDB:
    def __init__(self, db_path: str, embeddings_dir: str):
        ...

    # Answers
    def add_answer(self, source_file: str, text: str, record_id: str = None) -> int:
        ...

    # Questions
    def add_question(self, text: str, length_type: str = 'medium') -> int:
        ...

    # Clusters
    def create_cluster(self, name: str, answer_ids: List[int], question_ids: List[int]) -> int:
        ...

    # Embeddings
    def embed_and_store(self, model_name: str, entity_type: str, entity_id: int) -> np.ndarray:
        ...

    def get_embedding(self, model_name: str, entity_type: str, entity_id: int) -> np.ndarray:
        ...

    # Projections
    def train_projection(self, model_name: str, cluster_ids: List[int], name: str = None) -> int:
        ...

    def get_projection(self, projection_id: int) -> Tuple[np.ndarray, dict]:
        ...

    # Search
    def search(self, query_text: str, projection_id: int, top_k: int = 5) -> List[dict]:
        """
        1. Get projection (includes input_model, output_model, W)
        2. Embed query with input_model
        3. Apply projection: projected = W @ query_embedding
        4. Compare to answer embeddings (already stored with output_model)
        5. Return top-k with metadata
        """
        ...

    # Graph traversal
    def get_related(self, answer_id: int, relation_type: str = None) -> List[dict]:
        """Get answers related to this one (optionally filter by type)."""
        ...

    def get_full_version(self, answer_id: int) -> Optional[dict]:
        """Follow 'chunk_of', 'summarizes', 'abbreviates' to find full doc."""
        ...

    def get_variants(self, answer_id: int) -> List[dict]:
        """Get all variants (different text representations)."""
        ...

    def add_relation(self, from_id: int, to_id: int, relation_type: str, metadata: dict = None):
        """Create a relation between answers."""
        ...

    # Feedback
    def log_query(self, query_text: str, results: List[int], selected: int = None, helpful: bool = None):
        ...
```

## Example Usage

```python
db = LDAProjectionDB("playbooks/lda-training-data/lda.db", "playbooks/lda-training-data/embeddings")

# Add training data
a1 = db.add_answer("playbooks/examples_library/csv_examples.md", "CSV source: source(csv, ...")
q1 = db.add_question("How do I read CSV files?", "medium")
q2 = db.add_question("csv data loading", "short")

c1 = db.create_cluster("csv_data_source", [a1], [q1, q2])

# Train
proj_id = db.train_projection("all-MiniLM-L6-v2", [c1], name="v1")

# Search
results = db.search(
    "I need to load tabular data from a file",
    model_name="all-MiniLM-L6-v2",
    projection_id=proj_id,
    top_k=3
)
# Returns: [{"answer_id": 1, "source_file": "...", "score": 0.95, ...}, ...]
```

## Migration Path

1. **Phase 1**: Create schema and basic CRUD
2. **Phase 2**: Import existing `qa_pairs_v1.json` into database
3. **Phase 3**: Update training script to use database
4. **Phase 4**: Add search API for agents
5. **Phase 5**: Add query logging for continuous learning

## Integration with Agents

```prolog
% Prolog interface
find_examples(TaskDescription, TopK, Examples) :-
    invoke_component(runtime, lda_search,
        search(TaskDescription, TopK),
        results(Examples)).
```

Or via MCP tool for Claude:
```json
{
  "name": "find_playbook_examples",
  "description": "Find relevant playbook examples for a task",
  "parameters": {
    "task_description": "string",
    "top_k": "integer"
  }
}
```
