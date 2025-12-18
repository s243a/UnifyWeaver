# Knowledge Graph Topology - Usage Examples

Practical examples for common KG Topology API workflows.

## Building a Knowledge Graph

### Example 1: Creating Relations for a Tutorial Series

```python
from kg_topology_api import create_kg_database

db = create_kg_database("tutorials.db", "embeddings/")

# Assume we have these answers already in the database:
# 1: "Introduction to Python"
# 2: "Variables and Data Types"
# 3: "Control Flow (if/else)"
# 4: "Loops (for/while)"
# 5: "Functions"
# 6: "Classes and OOP"

# Create the learning flow
# "Introduction" is foundational to everything
db.add_kg_relation(1, 2, 'foundational')  # Intro → Variables
db.add_kg_relation(1, 3, 'foundational')  # Intro → Control Flow

# Natural progression
db.add_kg_relation(2, 3, 'transitional')  # Variables → Control Flow
db.add_kg_relation(3, 4, 'transitional')  # Control Flow → Loops
db.add_kg_relation(4, 5, 'transitional')  # Loops → Functions
db.add_kg_relation(5, 6, 'transitional')  # Functions → Classes

# Functions are foundational to Classes
db.add_kg_relation(5, 6, 'foundational')

# Classes extend Functions
db.add_kg_relation(5, 6, 'compositional')
```

### Example 2: Querying the Learning Path

```python
# What should I learn before Classes?
path = db.get_learning_path(answer_id=6, max_depth=5)
# Returns ordered list: [1, 2, 3, 4, 5, 6]
# (Introduction → Variables → Control Flow → Loops → Functions → Classes)

# What are the immediate prerequisites?
prereqs = db.get_foundational(6)  # [Functions, Introduction]
```

## Search with Context

### Example 3: Contextual Search Results

```python
results = db.search_with_context(
    query_text="How do I create a class in Python?",
    model_id=1,
    top_k=3
)

for result in results:
    print(f"\n=== {result['text'][:100]}... ===")
    print(f"Score: {result['score']:.3f}")

    if result['foundational']:
        print(f"Prerequisites: {[r['to_answer_id'] for r in result['foundational']]}")

    if result['next_steps']:
        print(f"Next steps: {[r['to_answer_id'] for r in result['next_steps']]}")
```

Output:
```
=== Classes and OOP: To create a class in Python, use the class keyword... ===
Score: 0.923
Prerequisites: [5, 1]  # Functions, Introduction
Next steps: []
```

## Semantic Interfaces

### Example 4: Creating Domain-Specific Interfaces

```python
# Create interfaces for different expertise areas
python_basics = db.create_interface(
    name="python_basics",
    description="Core Python language features",
    topics=["python", "syntax", "basics"]
)

python_oop = db.create_interface(
    name="python_oop",
    description="Object-oriented programming in Python",
    topics=["classes", "inheritance", "polymorphism"]
)

data_science = db.create_interface(
    name="data_science",
    description="Data analysis and ML with Python",
    topics=["pandas", "numpy", "scikit-learn"]
)

# Map clusters to interfaces
# Assume clusters: 1=basics, 2=oop, 3=pandas, 4=numpy
db.add_cluster_to_interface(python_basics, cluster_id=1)
db.add_cluster_to_interface(python_oop, cluster_id=2)
db.add_cluster_to_interface(data_science, cluster_id=3)
db.add_cluster_to_interface(data_science, cluster_id=4)
```

### Example 5: Computing and Using Interface Centroids

```python
# Compute centroids from cluster centroids
db.compute_interface_centroid(python_basics, model_id=1, method='mean')
db.compute_interface_centroid(python_oop, model_id=1, method='mean')
db.compute_interface_centroid(data_science, model_id=1, method='mean')

# Route a query to interfaces
import numpy as np
query_embedding = db._embed_query("How do I use pandas?", "model_name")

routing = db.map_query_to_interface(query_embedding, model_id=1, temperature=0.5)
for r in routing:
    print(f"{r['name']}: {r['probability']:.2%}")

# Output:
# data_science: 72.3%
# python_basics: 18.5%
# python_oop: 9.2%
```

### Example 6: Searching via Interface

```python
# Search within a specific interface
results = db.search_via_interface(
    query_text="read csv file",
    interface_id=data_science,
    model_id=1,
    top_k=5
)

# With interface-first routing optimization (for large datasets)
results = db.search_via_interface(
    query_text="read csv file",
    interface_id=data_science,
    model_id=1,
    use_interface_first_routing=True,
    similarity_threshold=0.3
)
```

## Seed Level Provenance

### Example 7: Tracking Question Expansion

```python
# Original dataset questions (seed level 0)
original_questions = [
    (1, "How do I read a CSV file?"),
    (2, "How do I parse JSON?"),
    (3, "How do I connect to a database?"),
]

for q_id, text in original_questions:
    db.set_seed_level(q_id, seed_level=0)
    # Also set anchor hash
    db.set_anchor_question(answer_id=q_id, anchor_question_text=text, seed_level=0)

# First expansion (seed level 1) - discovered from originals
expanded = [
    (4, "How do I read CSV with headers?", 1, 'refined'),     # From Q1
    (5, "How do I read CSV with encoding?", 1, 'refined'),    # From Q1
    (6, "How do I parse nested JSON?", 2, 'refined'),         # From Q2
]

for q_id, text, parent_q, relation in expanded:
    db.set_seed_level(
        q_id,
        seed_level=1,
        discovered_from=parent_q,
        discovery_relation=relation
    )
```

### Example 8: Querying by Seed Level

```python
# Get all original questions
originals = db.get_questions_at_seed_level(model_id=1, seed_level=0)
print(f"Original dataset: {len(originals)} questions")

# Get first expansion
first_expansion = db.get_questions_at_seed_level(model_id=1, seed_level=1)
print(f"First expansion: {len(first_expansion)} questions")

# Get expansion within a specific cluster
csv_expanded = db.get_questions_at_seed_level(
    model_id=1,
    seed_level=1,
    cluster_id=csv_cluster_id
)
```

## Scale Optimizations

### Example 9: Configuring Scale Optimizations

```python
# Check current optimization status
status = db.get_optimization_status()
print(f"Q-A Count: {status['qa_count']}")
print(f"Interface-First Routing: {status['interface_first_routing']}")
print(f"Transformer Distillation: {status['transformer_distillation']}")

# Configure thresholds
db.set_scale_config(
    interface_first_routing_enabled='auto',
    interface_first_routing_threshold=50000,
    transformer_distillation_enabled='auto',
    transformer_distillation_threshold=100000
)

# Check if optimizations should be used
routing_check = db.should_use_interface_first_routing()
if routing_check['use']:
    print(f"Using interface-first routing: {routing_check['reason']}")

distill_check = db.should_use_transformer_distillation()
if distill_check['use']:
    print(f"Distillation recommended: {distill_check['reason']}")
```

### Example 10: Preparing for Transformer Distillation

```python
# Check if distillation is recommended
check = db.check_distillation_recommended(qa_threshold=100000)
if check['recommended']:
    print(f"Distillation recommended: {check['qa_count']} Q-A pairs")

    # Get training embeddings
    data = db.get_distillation_training_embeddings(
        model_id=1,
        sample_size=50000  # Sample for faster training
    )

    print(f"Training samples: {len(data['question_ids'])}")
    print(f"Embedding shape: {data['embeddings'].shape}")

    # Use with ProjectionTransformer
    from projection_transformer import ProjectionTransformer

    transformer = ProjectionTransformer(
        input_dim=data['embeddings'].shape[1],
        hidden_dim=256,
        num_heads=4,
        num_layers=2
    )

    # Train via distillation...
```

## Interface Health Monitoring

### Example 11: Monitoring Interface Health

```python
# Check all interfaces
for interface in db.list_interfaces():
    health = db.get_interface_health(interface['interface_id'])

    print(f"\n{interface['name']}: {health['status']}")

    if health['issues']:
        print("  Issues:")
        for issue in health['issues']:
            print(f"    - {issue}")

    if health['recommendations']:
        print("  Recommendations:")
        for rec in health['recommendations']:
            print(f"    - {rec}")

# Compute coverage metrics
for interface in db.list_interfaces():
    coverage = db.compute_interface_coverage(interface['interface_id'])
    print(f"{interface['name']}: {coverage['coverage_ratio']:.1%} of answers")
```

## Complete Workflow Example

### Example 12: Building a Domain Expert System

```python
from kg_topology_api import create_kg_database
import numpy as np

# 1. Initialize database
db = create_kg_database("domain_expert.db", "embeddings/")

# 2. Add answers and questions (assume already done)
# ...

# 3. Build knowledge graph relations
db.add_kg_relation(basics_answer, intermediate_answer, 'transitional')
db.add_kg_relation(basics_answer, advanced_answer, 'foundational')
db.add_kg_relation(intermediate_answer, advanced_answer, 'transitional')

# 4. Set up seed levels for provenance
for q_id in original_question_ids:
    db.set_seed_level(q_id, seed_level=0)

# 5. Create semantic interfaces
beginner_interface = db.create_interface(
    name="beginner",
    description="Getting started",
    topics=["basics", "introduction", "tutorial"]
)

expert_interface = db.create_interface(
    name="expert",
    description="Advanced topics",
    topics=["advanced", "optimization", "internals"]
)

# 6. Map clusters to interfaces
db.add_cluster_to_interface(beginner_interface, basics_cluster)
db.add_cluster_to_interface(expert_interface, advanced_cluster)

# 7. Compute centroids
db.compute_interface_centroid(beginner_interface, model_id=1)
db.compute_interface_centroid(expert_interface, model_id=1)

# 8. Search with full context
results = db.search_with_context(
    query_text="How do I optimize my code?",
    model_id=1,
    top_k=5
)

# 9. Route to appropriate interface
query_emb = db._embed_query("How do I optimize my code?", "model_name")
routing = db.map_query_to_interface(query_emb, model_id=1)
best_interface = max(routing, key=lambda x: x['probability'])

print(f"Routing to: {best_interface['name']} ({best_interface['probability']:.0%})")

# 10. Search via that interface
final_results = db.search_via_interface(
    query_text="How do I optimize my code?",
    interface_id=best_interface['interface_id'],
    model_id=1
)
```

## See Also

- [KG_TOPOLOGY_API.md](KG_TOPOLOGY_API.md) - Full API reference
- [../proposals/ROADMAP_KG_TOPOLOGY.md](../proposals/ROADMAP_KG_TOPOLOGY.md) - Development roadmap
