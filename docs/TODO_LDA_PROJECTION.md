# TODO: LDA Semantic Projection Feature

## Overview

This document tracks the remaining work for the LDA-based semantic projection feature.
See: `docs/proposals/SEMANTIC_PROJECTION_LDA.md` and `docs/proposals/COMPONENT_REGISTRY.md`

## Completed

- [x] SEMANTIC_PROJECTION_LDA.md proposal
- [x] COMPONENT_REGISTRY.md proposal
- [x] Component registry implementation (`src/unifyweaver/core/component_registry.pl`)
- [x] LDA projection type module (`src/unifyweaver/runtime/lda_projection.pl`)
- [x] Python projection class (`src/unifyweaver/targets/python_runtime/projection.py`)
- [x] Unit tests (Python and Prolog)
- [x] LDA_DATABASE_SCHEMA.md proposal (asymmetric embeddings, graph relations)
- [x] LDA_TRAINING_APPROACH.md documentation
- [x] SQLite database layer (`src/unifyweaver/targets/python_runtime/lda_database.py`)
- [x] Database unit tests (33 tests) (`tests/core/test_lda_database.py`)
- [x] Migration script (`scripts/migrate_to_lda_db.py`)
- [x] Q-A pairs migrated to database (`playbooks/lda-training-data/lda.db`)

## Documentation

- [x] Update docs/README.md with LDA section
- [x] Update scripts/README.md with training scripts
- [x] Update proposals/README.md with new proposals
- [ ] Add architecture diagram showing component registry flow

## Testing & Validation

- [x] Create sample Q-A dataset for testing (`playbooks/lda-training-data/raw/qa_pairs_v1.json`)
- [x] Train a test W matrix from sample data (`playbooks/lda-training-data/trained/all-MiniLM-L6-v2/W_matrix.npy`)
- [x] Validate projection improves retrieval over direct cosine similarity
  - Result: 100% vs 93.33% Recall@1 on novel queries
- [ ] Benchmark projection latency (single query, batch)

## Training Pipeline

- [x] Create training script (`scripts/train_lda_projection.py`)
  - [x] Load Q-A pairs from file (JSON)
  - [x] Embed questions and answers (sentence-transformers)
  - [x] Compute W matrix using `compute_W()` with pseudoinverse
  - [x] Save W matrix to .npy file
  - [x] Cross-validate λ parameter

- [x] Create validation script (`scripts/validate_lda_projection.py`)
  - Tests with novel queries not in training data
  - Compares projected vs direct cosine similarity

- [ ] Create Q-A pair collection tools
  - Script to extract Q-A pairs from playbook runs
  - Format: `{answer_doc_id, [question_texts]}`

## Database

- [x] Design schema with asymmetric embedding support (input_model/output_model)
- [x] Implement answer relations graph (chunk_of, summarizes, translates, etc.)
- [x] SQLite + numpy files for vector storage
- [x] Search API with projection and logging
- [x] Training batch tracking with file hash detection
- [ ] Prolog interface for `find_examples/3`:
  ```prolog
  find_examples(TaskDescription, TopK, Examples) :-
      invoke_component(runtime, lda_search, search(TaskDescription, TopK), results(Examples)).
  ```

### Training Batch Tracking (Implemented)

Tracks which Q-A data files have been trained on with SHA256 file hashes:

```bash
# Scan for new/modified files
python3 scripts/migrate_to_lda_db.py --scan --input playbooks/lda-training-data/raw/

# Process all pending batches
python3 scripts/migrate_to_lda_db.py --process-pending

# Retry failed batches
python3 scripts/migrate_to_lda_db.py --retry-failed

# List all batches
python3 scripts/migrate_to_lda_db.py --list-batches
```

Status history is tracked with timestamps for each transition:
- pending → importing → embedding → training → completed
- Failed batches record error messages for debugging

## Integration

- [ ] Update PtSearcher to optionally use LDA projection
- [ ] Add projected search mode to Go embedder
- [ ] Create Prolog predicates for semantic search with projection:
  ```prolog
  semantic_search(Query, TopK, Results) :-
      invoke_component(runtime, embedding_provider, embed(Query), embedding(QueryEmb)),
      invoke_component(runtime, semantic_projection, query(QueryEmb), projected(ProjEmb)),
      vector_search(ProjEmb, TopK, Results).
  ```

## Multi-Head Projection (Implemented)

Analogous to transformer attention heads, each cluster gets its own projection:

- **Per-cluster heads**: Each cluster stores centroid (question mean) and answer embedding
- **Soft routing**: Query similarity to centroids determines routing weights via softmax
- **Temperature control**: Lower temperature = sharper routing (0.1 recommended)

Results on novel queries:
- Multi-head (temp=0.1): 76.7% Recall@1
- Direct similarity: 70.0% Recall@1
- **Improvement: +6.7%**

```bash
# Train multi-head projection
python3 scripts/train_multi_head_projection.py \
    --db playbooks/lda-training-data/lda.db \
    --model all-MiniLM-L6-v2 \
    --temperature 0.1 \
    --validate

# Validate with novel queries
python3 scripts/validate_multi_head.py \
    --db playbooks/lda-training-data/lda.db \
    --mh-id 1
```

## Future Enhancements

- [ ] Go backend for LDA projection (avoid Python subprocess overhead)
- [ ] Rust backend for LDA projection
- [ ] MCP tool for Claude integration (useful for server-based deployments)
- [ ] Hot-reload support for W matrix updates
- [ ] Per-domain projection matrices
- [x] Multiple attention heads (separate input/output projections)

## Notes

### Quick Test Commands

```bash
# Run Python tests
python3 tests/core/test_lda_projection.py
python3 tests/core/test_lda_database.py

# Run Prolog tests
swipl tests/core/test_component_registry.pl
swipl tests/core/test_lda_projection.pl

# Train W matrix from Q-A pairs
python3 scripts/train_lda_projection.py \
    --input playbooks/lda-training-data/raw/qa_pairs_v1.json \
    --model all-MiniLM-L6-v2 \
    --output playbooks/lda-training-data/trained/all-MiniLM-L6-v2/W_matrix.npy

# Validate with novel queries
python3 scripts/validate_lda_projection.py

# Migrate Q-A pairs to database
python3 scripts/migrate_to_lda_db.py

# Use database for search (Python API)
python3 -c "
from src.unifyweaver.targets.python_runtime.lda_database import LDAProjectionDB
from sentence_transformers import SentenceTransformer
db = LDAProjectionDB('playbooks/lda-training-data/lda.db')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
results = db.search_with_embedder('How to read CSV?', 1, embedder, top_k=3, log=False)
for r in results: print(f'{r[\"score\"]:.3f} {r[\"record_id\"]}')
"
```

### Sample Training Data Format

```json
{
  "clusters": [
    {
      "answer": "Document about authentication...",
      "answer_id": "doc_001",
      "questions": [
        "How do I log in?",
        "What's the authentication process?",
        "How to authenticate users?"
      ]
    },
    {
      "answer": "Document about database queries...",
      "answer_id": "doc_002",
      "questions": [
        "How to query the database?",
        "What SQL commands are supported?",
        "How do I fetch data?"
      ]
    }
  ]
}
```
