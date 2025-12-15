# LDA Projection Training Approach

## Overview

This document describes the approach for training the LDA projection matrix W from Q-A pairs.

## Training Data Structure

### Core Concept

Each training cluster consists of:
- **Answer**: A document/example that should be retrieved
- **Questions**: Multiple query phrasings that should retrieve that answer

```
Cluster = (answer_embedding, [question_embedding_1, question_embedding_2, ...])
```

### Starting Small

Begin with a single example from our playbooks:
1. Pick one example document
2. Generate ~5 query questions that would be good search keys
3. Vary query styles (short keywords, natural questions, detailed queries)

### Query Length Considerations

Different embedding models have different optimal input characteristics:

| Model | Max Tokens | Optimal Query Length | Backend | Notes |
|-------|-----------|---------------------|---------|-------|
| all-MiniLM-L6-v2 | 256 | Short-medium | Python/ONNX | Fast, good for keywords |
| e5-small-v2 | 512 | Medium | Python/ONNX | Instruction-tuned |
| e5-base-v2 | 512 | Medium-long | Python/ONNX | Better semantic understanding |
| BGE models | 512 | Varies | Python/ONNX | Some prefer "query:" prefix |
| ModernBERT | 8192 | Long documents | Rust | Longest context, use cross-target glue |

**Implication**: We may need separate training datasets per embedding model, or at minimum include query variations of different lengths.

### Dataset Organization

```
playbooks/
├── lda-training-data/
│   ├── raw/
│   │   └── qa_pairs_v1.json  # Text before embedding
│   └── trained/
│       ├── all-MiniLM-L6-v2/
│       │   └── W_matrix.npy
│       └── e5-small-v2/
│           └── W_matrix.npy
```

## Query Generation Strategy

For each answer document, generate queries at multiple levels:

### 1. Keyword Queries (2-4 words)
Short, search-engine style queries:
```
"authentication login"
"user password reset"
```

### 2. Natural Questions (5-15 words)
How users naturally ask:
```
"How do I log in to the system?"
"What's the process for resetting my password?"
```

### 3. Detailed Queries (15-30 words)
Context-rich queries:
```
"I'm trying to authenticate a user using OAuth2 and need to understand the token refresh flow"
```

### 4. Paraphrases
Same intent, different words:
```
"sign in" vs "log in" vs "authenticate"
"credentials" vs "username and password"
```

## Mathematical Foundation

The training computes the transformation matrix W using the **closed-form LDA solution with pseudoinverse**:

```
W = A · Q̄ᵀ · pinv(Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ + μ · I)
```

Where:
- **A**: Answer embeddings matrix (d × m)
- **Q̄**: Cluster centroids matrix (d × m), weighted means of questions per cluster
- **Δw**: Weighted residuals (deviations from centroids)
- **λ**: Regularization for residual suppression (cross-validated)
- **μ**: Ridge regularization for numerical stability (default: 1e-6)
- **pinv**: Moore-Penrose pseudoinverse (handles ill-conditioned matrices)

### Why Pseudoinverse?

The covariance matrix `(Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ + μ · I)` may be ill-conditioned when:
- Few clusters relative to embedding dimension (m << d)
- Clusters are highly correlated
- Numerical precision issues

Using `np.linalg.pinv()` instead of `np.linalg.inv()`:
1. Handles rank-deficient matrices gracefully
2. Produces minimum-norm solution
3. More numerically stable

The additional ridge term `μ · I` provides extra stability but pinv handles remaining issues.

### Weighted Centroid Computation

Centroids are computed iteratively to solve the bootstrapping problem:

```python
# Initialize with uniform weights
weights = [1/n, 1/n, ..., 1/n]

for _ in range(3):  # 2-3 iterations usually sufficient
    # Compute weighted centroid
    centroid = sum(weight_i * question_i)
    centroid = normalize(centroid)

    # Update weights based on similarity to centroid
    similarities = [question_i · centroid for each question]
    weights = softmax(similarities)

return centroid, weights
```

This ensures representative questions (close to centroid) have higher weight, while outliers are downweighted.

## Training Pipeline

### Step 1: Collect Q-A Pairs (Text)

```json
{
  "model_agnostic_pairs": [
    {
      "answer_id": "playbook_example_001",
      "answer_text": "...",  // or path to file
      "queries": {
        "short": ["auth login", "user signin"],
        "medium": ["How do I authenticate?", "What's the login process?"],
        "long": ["I need to implement user authentication with session management..."]
      }
    }
  ]
}
```

### Step 2: Embed with Target Model

```python
def embed_clusters(qa_pairs, embedder, model_name):
    """Embed Q-A pairs using specified model."""
    clusters = []
    for pair in qa_pairs:
        # Embed answer
        answer_emb = embedder.embed(pair["answer_text"])

        # Embed all query variants
        all_queries = []
        for length_type, queries in pair["queries"].items():
            all_queries.extend(queries)

        question_embs = [embedder.embed(q) for q in all_queries]

        clusters.append((answer_emb, np.array(question_embs)))

    return clusters
```

### Step 3: Compute W Matrix

```python
from projection import compute_W, save_W

# Cross-validate lambda
best_lambda = None
best_score = 0

for lambda_val in [0.01, 0.1, 1.0, 10.0]:
    W = compute_W(train_clusters, lambda_reg=lambda_val)
    score = evaluate_retrieval(W, test_clusters)
    if score > best_score:
        best_score = score
        best_lambda = lambda_val

# Train final model
W = compute_W(all_clusters, lambda_reg=best_lambda)
save_W(W, f"datasets/lda_training/{model_name}/W_matrix.npy")
```

### Step 4: Evaluate

Metrics:
- **Recall@k**: Is the correct answer in top-k results?
- **MRR**: Mean Reciprocal Rank of correct answer
- **Improvement**: Compare projected vs. direct cosine similarity

## Example: First Training Cluster

Let's start with a concrete example from a playbook.

### Answer Document
(Pick an example from `examples/` or a playbook reference)

### Query Variations

| Type | Query |
|------|-------|
| Short | `?` |
| Short | `?` |
| Medium | `?` |
| Medium | `?` |
| Long | `?` |

*(To be filled in based on selected example)*

## Bootstrapping with Few Examples

With only a few Q-A clusters, the matrix may be underdetermined. Strategies:

1. **Strong ridge regularization**: Higher μ in `(Q̄Q̄ᵀ + λΔwΔwᵀ + μI)⁻¹`
2. **Start with identity**: W = I + learned_adjustment
3. **Low-rank constraint**: Learn W = UVᵀ where U,V are tall-thin matrices
4. **Iterative expansion**: Start with W≈I, add clusters, recompute

## Next Steps

1. [ ] Select first example document from playbooks
2. [ ] Write 5 query variations (mix of lengths)
3. [ ] Embed using all-MiniLM-L6-v2 (our current default)
4. [ ] Compute W (even with 1 cluster, to test pipeline)
5. [ ] Validate: does projecting the queries improve similarity to answer?
6. [ ] Add more clusters, retrain, measure improvement

## Open Questions

1. How many Q-A clusters needed for meaningful improvement?
2. Should we weight clusters by importance/frequency?
3. Can we use synthetic query generation (LLM) to bootstrap?
4. Should the projection be query→answer or bidirectional?
