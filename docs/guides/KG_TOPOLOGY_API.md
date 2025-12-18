# Knowledge Graph Topology API Reference

**Module:** `src/unifyweaver/targets/python_runtime/kg_topology_api.py`
**Class:** `KGTopologyAPI`
**Extends:** `LDAProjectionDB`

## Overview

The KG Topology API extends the LDA Projection Database with:

1. **Knowledge Graph Relations** - 11 typed relationships between Q-A pairs
2. **Seed Level Provenance** - Track expansion depth from original dataset
3. **Hash-Based Anchoring** - Content-addressable links between questions and answers
4. **Semantic Interfaces** - Presentation layer for focused semantic identities
5. **Scale Optimizations** - Interface-first routing and transformer distillation

## Quick Start

```python
from kg_topology_api import KGTopologyAPI, create_kg_database

# Create or connect to database
db = create_kg_database("my_kg.db", "embeddings/")

# Add a knowledge graph relation
db.add_kg_relation(from_answer_id=1, to_answer_id=2, relation_type='foundational')

# Search with graph context
results = db.search_with_context(
    query_text="How do I parse CSV files?",
    model_id=1,
    top_k=5
)

# Create a semantic interface
interface_id = db.create_interface(
    name="csv_expert",
    description="CSV parsing and data manipulation",
    topics=["csv", "parsing", "data"]
)

# Search via interface
results = db.search_via_interface(
    query_text="read csv with headers",
    interface_id=interface_id,
    model_id=1
)
```

## Relation Types

The API supports 11 relation types across 3 categories:

### Learning Flow (4 types)

| Type | Direction | Meaning |
|------|-----------|---------|
| `foundational` | A ← B | B depends on principles from A |
| `preliminary` | A ← B | B requires completing A first |
| `compositional` | A → B | B extends/builds upon A |
| `transitional` | A → B | B is a natural next step after A |

### Scope (2 types)

| Type | Direction | Meaning |
|------|-----------|---------|
| `refined` | A → B | B is a more specific variant of A |
| `general` | A ← B | A is broader in scope than B |

### Abstraction (5 types)

| Type | Direction | Meaning |
|------|-----------|---------|
| `generalization` | A → B | B is an abstract pattern of A |
| `implementation` | A ← B | A is code realizing pattern B |
| `axiomatization` | A → B | B is abstract theory of A |
| `instance` | A ← B | A is domain satisfying theory B |
| `example` | A ← B | A illustrates/demonstrates B |

## API Reference

### Knowledge Graph Relations

#### `add_kg_relation(from_answer_id, to_answer_id, relation_type, strength=1.0, metadata=None)`

Create a relation between two answers.

```python
# JWT depends on cryptographic signing
db.add_kg_relation(
    from_answer_id=crypto_signing_id,
    to_answer_id=jwt_auth_id,
    relation_type='foundational'
)

# JWT refresh extends basic JWT
db.add_kg_relation(
    from_answer_id=jwt_auth_id,
    to_answer_id=jwt_refresh_id,
    relation_type='compositional'
)
```

**Parameters:**
- `from_answer_id` (int): Source answer ID
- `to_answer_id` (int): Target answer ID
- `relation_type` (str): One of the 11 relation types
- `strength` (float): Relation strength 0.0-1.0 (default 1.0)
- `metadata` (dict): Optional JSON metadata

**Returns:** Relation ID (int)

#### `get_relations(answer_id, relation_type=None, direction='both')`

Get relations for an answer.

```python
# Get all outgoing relations
outgoing = db.get_relations(answer_id, direction='outgoing')

# Get only foundational dependencies
deps = db.get_relations(answer_id, relation_type='foundational', direction='incoming')
```

**Parameters:**
- `answer_id` (int): Answer to query
- `relation_type` (str): Filter by type (optional)
- `direction` (str): 'incoming', 'outgoing', or 'both'

**Returns:** List of relation dicts with `relation_id`, `from_answer_id`, `to_answer_id`, `relation_type`, `strength`, `metadata`

#### Convenience Methods

```python
db.get_foundational(answer_id)      # Concepts this answer depends on
db.get_prerequisites(answer_id)      # Steps required before this answer
db.get_extensions(answer_id)         # Answers that extend this one
db.get_next_steps(answer_id)         # Natural next steps after this
db.get_refined(answer_id)            # More specific variants
db.get_general(answer_id)            # Broader scope versions
db.get_generalizations(answer_id)    # Abstract patterns
db.get_implementations(answer_id)    # Code realizing patterns
db.get_instances(answer_id)          # Domain instances
db.get_examples(answer_id)           # Pedagogical examples
```

### Search Methods

#### `search_with_context(query_text, model_id, top_k=5, max_graph_depth=2)`

Search with knowledge graph context included in results.

```python
results = db.search_with_context(
    query_text="How do I add JWT authentication?",
    model_id=1,
    top_k=5,
    max_graph_depth=2
)

for result in results:
    print(f"Answer: {result['answer_id']} (score: {result['score']:.3f})")
    print(f"  Foundational: {result['foundational']}")
    print(f"  Prerequisites: {result['prerequisites']}")
    print(f"  Extensions: {result['extensions']}")
    print(f"  Next steps: {result['next_steps']}")
```

**Returns:** List of dicts containing:
- `answer_id`, `score`, `text`
- `foundational`, `prerequisites`, `extensions`, `next_steps` - related answers

#### `get_learning_path(answer_id, max_depth=3)`

Get ordered learning path including all prerequisites.

```python
path = db.get_learning_path(jwt_auth_id, max_depth=3)
# Returns ordered list: [http_headers, crypto_signing, jwt_basics, ...]
```

### Seed Level Provenance

#### `set_seed_level(question_id, seed_level, discovered_from=None, discovery_relation=None)`

Set provenance tracking for a question.

```python
# Original dataset
db.set_seed_level(q1, seed_level=0)

# Discovered from q1 via refinement
db.set_seed_level(q2, seed_level=1, discovered_from=q1, discovery_relation='refined')
```

#### `get_seed_level(question_id)`

Get seed level for a question. Returns `None` if not set.

#### `get_questions_at_seed_level(model_id, seed_level, cluster_id=None)`

Query questions by seed level.

```python
# All seed(0) questions
originals = db.get_questions_at_seed_level(model_id=1, seed_level=0)

# Seed(1) questions in a specific cluster
expanded = db.get_questions_at_seed_level(model_id=1, seed_level=1, cluster_id=5)
```

### Hash-Based Anchoring

#### `compute_content_hash(text)` (static)

Compute SHA-256 hash of text content.

```python
hash_val = KGTopologyAPI.compute_content_hash("How do I read a CSV?")
```

#### `set_anchor_question(answer_id, anchor_question_text, seed_level=0)`

Link an answer to its anchor question via content hash.

```python
db.set_anchor_question(
    answer_id=42,
    anchor_question_text="How do I read a CSV file?",
    seed_level=0
)
```

#### `get_anchor_question(answer_id)`

Get the anchor question hash for an answer.

### Semantic Interfaces (Phase 2)

#### `create_interface(name, description=None, topics=None)`

Create a semantic interface.

```python
interface_id = db.create_interface(
    name="python_expert",
    description="Python programming and best practices",
    topics=["python", "programming", "libraries"]
)
```

#### `get_interface(interface_id)` / `get_interface_by_name(name)`

Retrieve interface details.

#### `list_interfaces(active_only=True)`

List all interfaces.

#### `update_interface(interface_id, name=None, description=None, topics=None, active=None)`

Update interface properties.

#### `delete_interface(interface_id)`

Delete an interface (cascades to cluster mappings).

### Interface-Cluster Mapping

#### `add_cluster_to_interface(interface_id, cluster_id, weight=1.0)`

Associate a cluster with an interface.

```python
db.add_cluster_to_interface(python_interface, csv_cluster, weight=1.0)
db.add_cluster_to_interface(python_interface, json_cluster, weight=0.8)
```

#### `remove_cluster_from_interface(interface_id, cluster_id)`

Remove cluster association.

#### `get_interface_clusters(interface_id)`

Get all clusters for an interface.

### Interface Centroids

#### `set_interface_centroid(interface_id, model_id, centroid)`

Set interface centroid embedding.

```python
centroid = np.array([0.1, 0.2, ...])  # Embedding vector
db.set_interface_centroid(interface_id, model_id, centroid)
```

#### `get_interface_centroid(interface_id, model_id)`

Retrieve interface centroid.

#### `compute_interface_centroid(interface_id, model_id, method='mean')`

Compute centroid from cluster centroids.

```python
centroid = db.compute_interface_centroid(
    interface_id,
    model_id,
    method='mean'  # or 'weighted_mean'
)
```

### Query Routing

#### `map_query_to_interface(query_embedding, model_id, temperature=1.0)`

Map query to interfaces via softmax routing.

```python
mapping = db.map_query_to_interface(query_emb, model_id, temperature=0.5)
# Returns: [
#   {'interface_id': 1, 'name': 'python_expert', 'probability': 0.7},
#   {'interface_id': 2, 'name': 'data_science', 'probability': 0.3}
# ]
```

#### `search_via_interface(query_text, interface_id, model_id, top_k=10, **options)`

Search through a specific interface.

```python
results = db.search_via_interface(
    query_text="parse csv with headers",
    interface_id=python_interface,
    model_id=1,
    top_k=10,
    # Optimization options:
    use_interface_first_routing=False,  # Pre-filter by centroid
    similarity_threshold=0.3,           # For interface-first routing
    max_distance=None                   # Post-filter by distance
)
```

### Scale Optimizations

#### `get_scale_config()` / `set_scale_config(**kwargs)`

Manage scale optimization configuration.

```python
# Get current config
config = db.get_scale_config()

# Set thresholds
db.set_scale_config(
    interface_first_routing_enabled='auto',  # 'auto', 'true', 'false'
    interface_first_routing_threshold=50000,
    transformer_distillation_enabled='auto',
    transformer_distillation_threshold=100000
)
```

#### `should_use_interface_first_routing()` / `should_use_transformer_distillation()`

Check if optimization should be used based on config and data scale.

```python
result = db.should_use_interface_first_routing()
# Returns: {'use': True, 'reason': 'auto: 75000 Q-A pairs >= 50000 threshold'}
```

#### `get_optimization_status()`

Comprehensive status for all optimizations.

```python
status = db.get_optimization_status()
# Returns: {
#   'config': {...},
#   'qa_count': 75000,
#   'interface_first_routing': {'use': True, 'reason': '...'},
#   'transformer_distillation': {'use': False, 'reason': '...'}
# }
```

#### `check_distillation_recommended(qa_threshold=100000)`

Check if transformer distillation is recommended.

#### `get_distillation_training_embeddings(model_id, sample_size=None)`

Get embeddings for training a distillation transformer.

```python
embeddings = db.get_distillation_training_embeddings(model_id=1, sample_size=10000)
# Returns: {'question_ids': [...], 'embeddings': np.array(...)}
```

### Interface Metrics

#### `set_interface_metric(interface_id, metric_name, value)`

Set a metric value for an interface.

#### `get_interface_metrics(interface_id)`

Get all metrics for an interface.

#### `compute_interface_coverage(interface_id)`

Compute coverage metrics.

```python
coverage = db.compute_interface_coverage(interface_id)
# Returns: {
#   'cluster_count': 5,
#   'answer_count': 150,
#   'question_count': 450,
#   'total_answers': 1000,
#   'coverage_ratio': 0.15
# }
```

#### `get_interface_health(interface_id)`

Get health status and recommendations.

```python
health = db.get_interface_health(interface_id)
# Returns: {
#   'status': 'healthy',  # or 'warning', 'unhealthy'
#   'metrics': {...},
#   'issues': [],
#   'recommendations': []
# }
```

## Auto-Generation

#### `auto_generate_interfaces(model_id, min_clusters=2, similarity_threshold=0.7)`

Automatically generate interfaces from cluster analysis.

```python
interfaces = db.auto_generate_interfaces(
    model_id=1,
    min_clusters=2,
    similarity_threshold=0.7
)
```

## Database Schema

The API adds these tables to the LDA database:

```sql
-- Seed level provenance
CREATE TABLE question_seed_levels (
    question_id INTEGER PRIMARY KEY,
    seed_level INTEGER NOT NULL DEFAULT 0,
    discovered_from_question_id INTEGER,
    discovery_relation TEXT
);

-- Anchor linking
CREATE TABLE answer_anchors (
    answer_id INTEGER PRIMARY KEY,
    anchor_question_hash TEXT NOT NULL,
    seed_level INTEGER DEFAULT 0
);

-- Semantic interfaces
CREATE TABLE semantic_interfaces (
    interface_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    topics TEXT,  -- JSON array
    active INTEGER DEFAULT 1
);

-- Interface-cluster mapping
CREATE TABLE interface_clusters (
    interface_id INTEGER,
    cluster_id INTEGER,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (interface_id, cluster_id)
);

-- Interface centroids
CREATE TABLE interface_centroids (
    interface_id INTEGER,
    model_id INTEGER,
    centroid_path TEXT NOT NULL,  -- File path to numpy array
    PRIMARY KEY (interface_id, model_id)
);

-- Interface metrics
CREATE TABLE interface_metrics (
    interface_id INTEGER,
    metric_name TEXT,
    value REAL,
    PRIMARY KEY (interface_id, metric_name)
);

-- Scale configuration
CREATE TABLE kg_settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

## See Also

- [ROADMAP_KG_TOPOLOGY.md](../proposals/ROADMAP_KG_TOPOLOGY.md) - Development roadmap
- [QA_KNOWLEDGE_GRAPH.md](../proposals/QA_KNOWLEDGE_GRAPH.md) - Relation type design
- [SEED_QUESTION_TOPOLOGY.md](../proposals/SEED_QUESTION_TOPOLOGY.md) - Provenance tracking design
- [TRANSFORMER_DISTILLATION.md](../proposals/TRANSFORMER_DISTILLATION.md) - Distillation optimization
