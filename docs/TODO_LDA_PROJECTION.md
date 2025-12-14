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

## Documentation

- [ ] Update main README with component registry section
- [ ] Add usage examples for LDA projection
- [ ] Document how to train W matrix from Q-A pairs
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

## Future Enhancements

- [ ] Go backend for LDA projection (avoid Python subprocess overhead)
- [ ] Rust backend for LDA projection
- [ ] Hot-reload support for W matrix updates
- [ ] Per-domain projection matrices
- [ ] Multiple attention heads (separate input/output projections)

## Notes

### Quick Test Commands

```bash
# Run Python tests
python3 tests/core/test_lda_projection.py

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
