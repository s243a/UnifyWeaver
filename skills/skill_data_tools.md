# Skill: Data Tools (Master)

Master skill for data processing, querying, machine learning, and data source management in UnifyWeaver.

## When to Use

- User asks "how do I process data?"
- User needs SQL, streaming aggregation, or fuzzy search
- User wants to train or run ML models
- User needs to read JSON/JSONL data sources
- User asks about embeddings, semantic search, or hierarchy optimization
- User wants to generate synthetic training data

## Skill Hierarchy

```
skill_data_tools.md (this file - MASTER)
├── skill_query_patterns.md (sub-master)
│   ├── skill_sql_target.md - SQL generation (SQLite, PostgreSQL, MySQL)
│   ├── skill_stream_aggregation.md - Runtime aggregation (Go, C#, Perl, Ruby)
│   ├── skill_aggregation_patterns.md - GROUP BY, window functions, score fusion
│   └── skill_fuzzy_search.md - Fuzzy logic, RRF, score blending
├── skill_ml_tools.md (sub-master)
│   ├── skill_embedding_models.md - Model selection (nomic, MiniLM, BERT)
│   ├── skill_train_model.md - Federated model training
│   ├── skill_semantic_inference.md - Running inference
│   └── skill_hierarchy_objective.md - J = D/(1+H) optimization
├── skill_data_sources.md (sub-master)
│   ├── skill_json_sources.md - JSON/JSONL data sources
│   └── skill_extract_records.md - Markdown record extraction
└── skill_synthetic_data.md (sub-master)
    ├── skill_qa_generation.md - Q&A from skills/docs
    ├── skill_answer_tailoring.md - Reword answers with LLM
    └── skill_pearl_dataset.md - Pearltrees training data from RDF
```

Note: `skill_density_explorer.md` moved to `skill_mindmap_bookmark_tools.md` as it serves both mindmap and bookmark visualization.

## Quick Reference

### Query Patterns

```prolog
% SQL generation
:- use_module('src/unifyweaver/targets/sql_target').
compile_predicate_to_sql(my_pred/2, [dialect(postgres)], SQL).

% Stream aggregation
aggregate_all(count, item(_), Count).
aggregate_all(sum(Price), order(_, Price), Region, Total).

% Fuzzy score blending
:- use_module('src/unifyweaver/fuzzy/fuzzy').
blend_scores(0.7, SemanticScores, KeywordScores, Combined).
```

### Machine Learning

```bash
# Train a federated model
python3 scripts/train_pearltrees_federated.py \
  reports/pearltrees_targets.jsonl \
  models/federated.pkl \
  --model nomic-ai/nomic-embed-text-v1.5

# Run inference
python3 scripts/infer_pearltrees_federated.py \
  --model models/federated.pkl \
  --query "quantum computing basics"

# Evaluate hierarchy
python3 scripts/mindmap/hierarchy_objective.py \
  --tree hierarchy.json \
  --embeddings embeddings.npy
```

### Data Sources

```prolog
% JSON source with column projection
:- source(json, order_totals, [
    json_file('data/orders.json'),
    columns(['order.customer.name', 'items[0].total'])
]).

% JSONL with null handling
:- source(json, events, [
    json_file('data/events.jsonl'),
    record_format(jsonl),
    null_policy(skip)
]).
```

```bash
# Extract records from markdown
perl scripts/utils/extract_records.pl \
  -f json \
  -q "pattern" \
  path/to/file.md
```

### Synthetic Data

```bash
# Generate Q&A from skills
python training-data/scripts/generate_qa_from_skills.py \
  --all --model haiku

# Tailor/reword answers
python scripts/generate_tailored_answers.py \
  --input training-data/expanded

# Generate Pearltrees dataset
python scripts/generate_pearl_dataset.py \
  --rdf data/export.rdf \
  --output reports/targets.jsonl
```

## Capabilities Overview

### Query & Aggregation

| Capability | Description | Targets |
|------------|-------------|---------|
| SQL Generation | Views, CTEs, recursive queries | SQLite, PostgreSQL, MySQL |
| Stream Aggregation | Runtime COUNT/SUM/AVG | Go, C#, Perl, Ruby |
| Window Functions | RANK, LAG, LEAD | SQL targets |
| Fuzzy Logic | f_and, f_or, weighted terms | All targets |
| Score Fusion | Blend, RRF, multiplication | Python, Prolog |

### Machine Learning

| Capability | Description | Tool |
|------------|-------------|------|
| Embedding Models | nomic, MiniLM, BERT, ModernBERT | Python, Go, Rust, C# |
| Model Training | Procrustes projection, MST clustering | train_pearltrees_federated.py |
| Inference | Folder suggestion, semantic search | infer_pearltrees_federated.py |
| Visualization | Density explorer, tree overlay | tools/density_explorer/ |
| Hierarchy Optimization | J = D/(1+H) objective | hierarchy_objective.py |

### Data Sources

| Source | Features | Status |
|--------|----------|--------|
| JSON Files | Column projection, JSONPath, nested records | Implemented |
| JSONL Streams | Line-by-line, null policies | Implemented |
| Markdown Records | Extract structured data from docs | Implemented |

### Synthetic Data

| Capability | Description | Tool |
|------------|-------------|------|
| Q&A Generation | Generate Q&A from skills/docs | generate_qa_from_skills.py |
| Answer Tailoring | Reword answers with LLM | generate_tailored_answers.py |
| Pearl Dataset | Training data from Pearltrees RDF | generate_pearl_dataset.py |
| Cluster Expansion | Expand Q&A clusters to pairs | expand_clusters_to_pairs.py |

## Common Workflows

### Semantic Search Pipeline

1. **Prepare data** - Extract text from sources (`skill_extract_records.md`)
2. **Generate embeddings** - Choose model (`skill_embedding_models.md`)
3. **Train projection** - Procrustes federation (`skill_train_model.md`)
4. **Run inference** - Query for similar items (`skill_semantic_inference.md`)
5. **Visualize** - Explore clusters (`skill_density_explorer.md`)

### Query Pipeline

1. **Declare sources** - JSON/JSONL (`skill_json_sources.md`)
2. **Define aggregation** - SQL or stream (`skill_aggregation_patterns.md`)
3. **Combine scores** - Fuzzy logic (`skill_fuzzy_search.md`)
4. **Generate code** - Target language (`skill_sql_target.md`, `skill_stream_aggregation.md`)

### Hierarchy Optimization

1. **Build initial tree** - MST or J-guided (`skill_hierarchy_objective.md`)
2. **Evaluate quality** - Compute J = D/(1+H) (`skill_hierarchy_objective.md`)
3. **Visualize structure** - Density explorer (`skill_density_explorer.md`)
4. **Iterate** - Adjust parameters, re-root, add intermediate nodes

### Synthetic Data Pipeline

1. **Generate Q&A** - From skills or docs (`skill_qa_generation.md`)
2. **Expand clusters** - To individual pairs (`skill_synthetic_data.md`)
3. **Tailor answers** - Reword with LLM (`skill_answer_tailoring.md`)
4. **Generate embeddings** - For training (`skill_embedding_models.md`)
5. **Train model** - On generated data (`skill_train_model.md`)

## Child Skills

- `skill_query_patterns.md` - SQL, streaming aggregation, fuzzy search
- `skill_ml_tools.md` - Embeddings, training, inference, visualization
- `skill_data_sources.md` - JSON sources, record extraction
- `skill_synthetic_data.md` - Q&A generation, answer tailoring, pearl datasets

## Related

**Sibling Masters:**
- `skill_server_tools.md` - Backend services, APIs, IPC
- `skill_gui_tools.md` - Frontend/GUI generation
- `skill_mindmap_bookmark_tools.md` - Mindmaps and bookmarks

**Documentation:**
- `docs/BINDING_MATRIX.md` - Target feature matrix
- `education/book-13-semantic-search/` - Semantic search concepts
- `education/book-14-ai-training/` - Training and embeddings

**Code:**
- `src/unifyweaver/targets/sql_target.pl` - SQL compilation
- `src/unifyweaver/fuzzy/` - Fuzzy logic DSL
- `scripts/train_pearltrees_federated.py` - Model training
- `tools/density_explorer/` - Visualization tool
- `scripts/generate_tailored_answers.py` - Answer tailoring
- `training-data/scripts/generate_qa_from_skills.py` - Q&A generation
- `scripts/generate_pearl_dataset.py` - Pearltrees dataset
