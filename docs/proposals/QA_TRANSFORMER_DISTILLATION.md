# Proposal: Q/A Transformer Distillation with Enhanced Routing

## Overview

This document describes the federated Q/A model training and transformer distillation pipeline, reports experimental results, and proposes enhancements using per-question routing with probability-weighted softmax.

## What We Did

### Data Characteristics

| Attribute | Value |
|-----------|-------|
| Q/A Pairs | 1,888 |
| Embedding Model | nomic-ai/nomic-embed-text-v1.5 |
| Embedding Dimension | 768 |
| Clusters | 124 (auto-selected via effective rank) |
| Cluster Method | K-means on answer embeddings |

The data consists of skills-related Q/A pairs where:
- **Questions**: Natural language queries about UnifyWeaver capabilities
- **Answers**: Detailed explanations with code examples and references

### Training Pipeline

1. **Embedding Generation**: Both questions and answers embedded with nomic model
2. **Clustering**: K-means on answer embeddings, cluster count determined by effective rank criterion
3. **W Matrix Training**: Per-cluster Procrustes (minimal transform) projecting question space → answer space
4. **Transformer Distillation**: Compress federated model (124 W matrices) into compact transformer

### Federated Model

The federated model uses:
- **Routing**: Query similarity to cluster centroids → softmax → weighted W combination
- **Projection**: `projected = Σ weight_i × (query @ W_i)` for top-K clusters
- **Temperature**: 0.1 for softmax sharpness

## Experimental Results

### Transformer Architecture Comparison

| Architecture | Capacity | Parameters | Compression | Train Loss | Test Cosine Sim |
|--------------|----------|------------|-------------|------------|-----------------|
| H=12, L=2 | 144 | 10.6M | 6.9x | 0.030 | 0.673 |
| H=6, L=3 | 216 | 15.4M | 4.8x | 0.024 | **0.705** |
| H=4, L=4 | 256 | 20.1M | 3.6x | 0.021 | 0.693 |

**Observations**:
- **6³ architecture achieved best generalization** (0.705 cosine sim) despite lower capacity than 4⁴
- **4⁴ had lowest training loss** but slightly worse test performance, suggesting mild overfitting
- **12² was most efficient** but insufficient depth for complex projection patterns
- **3 layers appears optimal** - enough depth for moderate complexity without overfitting

### Interpretation

The ~0.70 cosine similarity ceiling suggests:
1. The projection task has inherent difficulty that more capacity doesn't solve
2. The nomic embedding model handles most semantic understanding
3. The transformer learns a relatively simple mapping function

## Proposed Enhancement: Per-Question Routing

### W Matrix Options

Two approaches for the projection matrices:

1. **Per-Q/A pair W**: One W matrix per question-answer pair
   - Most expressive
   - Impractical: 1,888 × 768 × 768 × 4 bytes ≈ 4.4 GB storage

2. **Per-cluster W** (what we use): Single W matrix per cluster via minimal transform
   - Practical storage: 124 × 768 × 768 × 4 bytes ≈ 290 MB
   - Better results than attempting to smooth/blend W matrices
   - Each cluster's W is computed via Procrustes on all Q/A pairs in that cluster

### Current Approach (Centroid-Based Routing)

```
query → sim(query, 124 centroids) → softmax → select cluster W → query @ W
```

- Routes based on 124 cluster centroids
- Coarse-grained: all questions in a cluster share the same centroid
- Training signal limited to centroid-level patterns

### Proposed Approach (Question-Based Routing)

```
query → sim(query, 1888 questions) → softmax → aggregate to cluster weights → select cluster W
```

**Key insight**: Each training question acts like an **attention head**. The same cluster W is selected, but the question similarities weight the softmax routing in **non-linear ways**.

- The W matrices remain fixed (one per cluster)
- Questions provide fine-grained routing signal
- Non-linearity comes from softmax over question similarities, then aggregating to cluster selection

**Benefits**:
1. **Richer routing signal**: 1,888 question similarities vs 124 centroid similarities
2. **Non-linear attention**: Questions near cluster boundaries create nuanced soft-routing
3. **Better generalization**: Learns question-level routing patterns

**Challenges**:
1. **Similarity computation**: O(N²) for question similarities vs O(N×K) for centroids
2. **Aggregation**: Need to map question-level softmax weights to cluster-level selection

**Computational Optimization**:

Since W is the same for all questions in a cluster, aggregation is simple:

```python
# 1. Compute question similarities (dot products)
sims = query @ questions.T  # [1, N] @ [N, D].T = [1, N]

# 2. Softmax to get fractional probabilities
probs = softmax(sims / temperature)  # [N]

# 3. Aggregate to cluster weights (sum of fractional probabilities per cluster)
cluster_weights = zeros(K)
for i, q in enumerate(questions):
    cluster_weights[cluster[i]] += probs[i]

# 4. Single matrix multiply per cluster
result = sum(cluster_weights[k] * (query @ W[k]) for k in range(K))
```

The cluster weight is just the sum of fractional probabilities for questions in that cluster. This reduces to K matrix multiplies regardless of N.

**Final softmax routing over clusters**:

```python
# Cluster weights now have complex weighting from question-level attention
# Apply final softmax routing over clusters
final_weights = softmax(cluster_weights / temperature)
result = sum(final_weights[k] * (query @ W[k]) for k in range(K))
```

### Analogy to Statistical Mechanics

This two-level routing is analogous to **degeneracy (state multiplicity)** in statistical mechanics:

- **Cluster** = energy level
- **Questions in cluster** = degenerate microstates at that energy
- **Question similarity** = Boltzmann factor for each microstate
- **Cluster weight** = partition function contribution (sum over microstates)

In stat mech: `P(energy E) ∝ g(E) × exp(-E/kT)`

Where `g(E)` is the degeneracy (number of microstates at energy E).

In our routing: `weight(cluster k) ∝ Σ_q∈k sim(query, q)`

Clusters with more similar questions have higher "degeneracy" and thus higher routing weight - not just because of centroid similarity, but because of the multiplicity of matching questions.

## Probability-Weighted Softmax

### Motivation

Not all training examples are equally informative. We can modify softmax weights based on **prediction confidence/uncertainty** to:
- Upweight uncertain/boundary cases (more informative)
- Downweight highly confident predictions (less informative)

### Standard Softmax

```
weights_i = exp(sim_i / τ) / Σ exp(sim_j / τ)
```

### Probability-Weighted Softmax

```
weights_i = p_i × exp(sim_i / τ) / Σ p_j × exp(sim_j / τ)
```

Where `p_i` is a probability/importance weight derived from model confidence.

## Deriving Probability from Entropy

### Approach 1: Fisher Information

Fisher information measures how much information an observation provides about model parameters:

```
I(θ) = E[(∂/∂θ log p(x|θ))²]
```

For our routing context:
- **High Fisher information**: Sample is informative (near decision boundary)
- **Low Fisher information**: Sample is uninformative (clearly belongs to one cluster)

**Practical computation**:
```python
# Compute gradient of log-likelihood w.r.t. routing weights
grad = ∂L/∂weights
fisher_score = ||grad||²

# Weight by Fisher information (upweight informative samples)
importance = fisher_score / mean(fisher_scores)
```

**Entropy-based approximation**:
```python
# Softmax distribution entropy
H = -Σ weights_i × log(weights_i)

# High entropy = uncertain = informative
# Low entropy = confident = less informative
importance = H / log(num_clusters)  # Normalized entropy
```

### Approach 2: BERT Logit Values

The nomic embedding model (based on BERT architecture) produces logits before the final projection. These logits contain uncertainty information:

**Token-level uncertainty**:
```python
# Get logits from BERT's masked language model head
logits = bert.cls(hidden_states)  # [batch, seq, vocab]

# Convert to probabilities
probs = softmax(logits, dim=-1)

# Compute per-token entropy
token_entropy = -Σ probs × log(probs)

# Aggregate to sequence-level
sequence_uncertainty = mean(token_entropy)
```

**Embedding-level uncertainty** (if using contrastive models):
```python
# Some models expose confidence scores
# For nomic, we can use the embedding norm as a proxy
embedding_confidence = ||embedding|| / expected_norm

# Or use dropout-based uncertainty
with_dropout = [model(x, dropout=True) for _ in range(N)]
uncertainty = std(with_dropout)
```

### Combining Approaches

```python
def compute_importance_weight(query_emb, cluster_weights, bert_logits=None):
    # Routing entropy (from softmax weights)
    routing_entropy = -sum(w * log(w) for w in cluster_weights if w > 0)

    # Optional: BERT-based uncertainty
    if bert_logits is not None:
        bert_entropy = compute_token_entropy(bert_logits)
        combined = α * routing_entropy + (1-α) * bert_entropy
    else:
        combined = routing_entropy

    # Normalize to [0, 1] importance weight
    # High entropy = high importance (upweight uncertain samples)
    importance = combined / max_entropy

    return importance
```

## Implementation Plan

### Phase 1: Per-Question Routing
1. Modify `FederatedProjectionWrapper.project()` to use question-level routing
2. Add cluster aggregation: question weights → cluster weights
3. Benchmark against centroid-based routing

### Phase 2: Importance Weighting
1. Compute routing entropy for each training sample
2. Apply importance weights during distillation loss computation
3. Compare: uniform weights vs entropy-weighted vs Fisher-weighted

### Phase 3: BERT Uncertainty Integration
1. Extract logits/hidden states from nomic model during embedding
2. Compute token-level entropy
3. Combine with routing entropy for final importance weight

## Expected Outcomes

| Enhancement | Expected Impact |
|-------------|-----------------|
| Per-question routing | +5-10% cosine sim (richer training signal) |
| Entropy weighting | Better generalization on boundary cases |
| BERT uncertainty | More principled importance weights |

## Architecture Selection via AIC

### The Problem

Given K clusters and N questions, how do we choose the transformer architecture (H heads, L layers)?

- Constraint: H^L ≥ K (sufficient capacity)
- Trade-off: More parameters → better fit but potential overfitting
- Our experiment: 12², 6³, 4⁴ all valid, but 6³ generalized best

### Akaike Information Criterion (AIC)

Standard AIC balances fit vs complexity:
```
AIC = 2k - 2ln(L)
```
Where k = parameters, L = likelihood.

For regression with Gaussian errors (variance σ²):
```
AIC_gaussian = n·ln(MSE) + 2k
```

### Student's t Distribution

The Gaussian assumption treats all errors equally. But:
- An error of 0.1 when σ=0.01 is significant
- An error of 0.1 when σ=0.5 is noise

Student's t-distribution addresses this by:
1. Standardizing residuals: z_i = (r_i - μ) / σ
2. Using heavier tails (more robust to outliers)
3. The relevance of each error depends on σ

Log-likelihood under t(ν):
```
ln L = Σ ln(t_pdf(z_i; ν))
```

Where t_pdf involves gamma functions and the degrees of freedom ν.

### Estimating Degrees of Freedom

We estimate ν from the data using kurtosis:
```
For t(ν): excess_kurtosis = 6/(ν-4)  for ν > 4
Therefore: ν ≈ 4 + 6/kurtosis
```

High kurtosis (heavy tails) → low ν → t-distribution
Low kurtosis (near Gaussian) → high ν → approaches normal

### Implementation

```python
from scipy import stats

def compute_aic_student_t(residuals, n_params):
    # Standardize
    z = (residuals - residuals.mean()) / residuals.std()

    # Estimate df from kurtosis
    kurtosis = stats.kurtosis(z)
    df = max(3, 4 + 6/kurtosis) if kurtosis > 0.1 else 30

    # Log-likelihood under t(df)
    log_L = np.sum(stats.t.logpdf(z, df=df))

    # AIC
    return 2 * n_params - 2 * log_L
```

### Usage

```bash
# Suggest architectures
python3 scripts/distill_federated_to_transformer.py \
  models/federated.pkl models/out.pt \
  --architecture suggest --num-suggestions 4

# Compare with AIC selection
python3 scripts/distill_federated_to_transformer.py \
  models/federated.pkl models/out.pt \
  --architecture compare \
  --selection-criterion aic_t \
  --compare-epochs 50
```

Selection criteria:
- `best_cosine`: Highest mean cosine similarity
- `aic_t`: AIC with Student's t (default, balances fit + complexity)
- `aic_gaussian`: AIC with Gaussian assumption
- `bic_t`: BIC with Student's t (stronger complexity penalty)

### Effective Degrees of Freedom

Standard AIC uses raw parameter count (k = millions), which dominates the
criterion for neural networks. We offer alternative "effective parameter count"
options via `--effective-df`:

| Option | k = | Rationale |
|--------|-----|-----------|
| `params` | 10.6M, 15.4M, 20.1M | Raw weights (default, but problematic) |
| `capacity` | H^L = 144, 216, 256 | Discrete routing patterns |
| `log_capacity` | log₂(H^L) ≈ 7-8 | Bits to encode routing |

---

### What We Know (Empirical Facts)

1. **Raw parameter count dominates AIC**: With k in millions, the 2k penalty
   overwhelms the likelihood term, making AIC essentially "smallest model wins"

2. **Capacity-based AIC is numerically balanced**: When k = H^L (144-256),
   the penalty term 2k ≈ 300-500 is comparable to -2ln(L) ≈ 535, producing
   meaningful trade-offs

3. **Experimental results**:
   - 12² (144 capacity): 0.673 cosine similarity
   - 6³ (216 capacity): 0.705 cosine similarity (best generalization)
   - 4⁴ (256 capacity): 0.693 cosine similarity

4. **Training dynamics**: Deeper networks (6³, 4⁴) reach higher cosine
   similarity faster, but 12² can catch up with more epochs

---

### Philosophical Position (Proposed Interpretation)

We propose that **capacity (H^L) represents the effective parameter count**
because:

1. **Discrete routing structure**: The transformer implements H^L distinct
   attention routing patterns. The millions of linear weights are secondary
   parameters that realize these discrete patterns.

2. **Analogy to established models**:
   - Mixture of K Gaussians: complexity is O(K), not O(K×d²)
   - Decision trees: complexity is number of leaves, not threshold values
   - Our transformer: complexity is routing patterns, not weight values

3. **Logical vs implementation complexity**: The model operates at a logical
   level (which attention heads to select), while linear mappings are the
   continuous implementation of those discrete choices.

4. **Problem characterization**: We believe transformer distillation is more
   like a discrete routing problem than a continuous optimization problem.

---

### Open Questions (Needs Research)

1. **Theoretical justification**: We lack formal proof that H^L is the correct
   effective parameter count. Possible approaches:
   - Vapnik-Chervonenkis dimension analysis
   - PAC-Bayes generalization bounds
   - Minimum Description Length formulation

2. **Bias-variance in architecture selection**: Our experiments suggest:
   - 12²: higher bias (underfitting)
   - 6³: balanced
   - 4⁴: higher variance (slight overfitting)

   A principled criterion should capture this, but AIC doesn't directly
   measure bias-variance trade-off.

3. **Errors within vs exterior to model**: Are 12² errors due to model
   misspecification (can't express the true function) while 6³ errors are
   noise around the expressible truth? This relates to whether the true
   solution lies on the model's manifold.

4. **Training efficiency as model fit**: 6³ learns faster than 12². Could
   training efficiency (epochs to convergence, learning curve area) serve
   as a model selection criterion? This might indicate architecture-problem
   alignment.

5. **Cross-validation vs information criteria**: Our manual comparison
   (effectively cross-validation) selected 6³, while AIC selected 12².
   Which is more appropriate for neural architecture selection?

---

### Theoretical Concepts to Explore

The following theoretical frameworks may be relevant to understanding
effective degrees of freedom and model selection in this context:

**Statistical Learning Theory:**

- **Vapnik-Chervonenkis (VC) Dimension**: Measures the capacity of a model
  class by the largest set it can shatter (classify arbitrarily). For our
  transformers, the VC dimension might be related to H^L routing patterns
  rather than raw parameter count.

- **Rademacher Complexity**: Another capacity measure based on how well the
  model can fit random labels. Related to generalization bounds.

- **PAC (Probably Approximately Correct) Learning**: Framework for bounding
  generalization error. PAC-Bayes bounds incorporate prior beliefs about
  model complexity.

**Information-Theoretic Model Selection:**

- **Akaike Information Criterion (AIC)**: 2k - 2ln(L). Asymptotically optimal
  for prediction. Tends to select larger models.

- **Bayesian Information Criterion (BIC)**: k·ln(n) - 2ln(L). Consistent
  (selects true model as n→∞). Stronger complexity penalty.

- **Minimum Description Length (MDL)**: Model selection as data compression.
  Best model minimizes: length(model) + length(data|model). May naturally
  capture the discrete/continuous distinction.

- **Fisher Information**: Measures information an observation provides about
  parameters. Related to model curvature and effective degrees of freedom.

**Bias-Variance Decomposition:**

- **Bias**: Error from model limitations (can't express true function)
- **Variance**: Error from sensitivity to training data
- **Bias-Variance Tradeoff**: Simple models → high bias, low variance.
  Complex models → low bias, high variance. Optimal is balanced.

- **James-Stein Estimation**: Sometimes biased estimators have lower total
  error than unbiased ones. Relates to why 6³ might beat 4⁴.

**Effective Degrees of Freedom:**

- **Ridge Regression**: Effective df = trace(H) where H is hat matrix.
  Can be much less than parameter count p.

- **Smoothing Splines**: Effective df controlled by smoothing parameter.
  Bridges between interpolation and underfitting.

- **Neural Network Compression**: Pruned networks show that effective
  complexity << parameter count. Many weights are redundant.

**Geometry of Learning:**

- **Loss Landscape Geometry**: Smoother landscapes → easier optimization.
  Architecture affects landscape structure.

- **Implicit Regularization**: SGD dynamics implicitly prefer simpler
  solutions. Different architectures have different implicit biases.

- **Neural Tangent Kernel (NTK)**: In infinite-width limit, neural nets
  behave as kernel methods. The kernel depends on architecture.

- **Manifold Hypothesis**: Real data lies on low-dimensional manifolds.
  Model errors can be decomposed into on-manifold (expressible) and
  off-manifold (model limitation) components.

**Relevant Literature:**

- Akaike, H. (1974). "A new look at the statistical model identification"
- Schwarz, G. (1978). "Estimating the dimension of a model" (BIC)
- Rissanen, J. (1978). "Modeling by shortest data description" (MDL)
- Vapnik, V. (1995). "The Nature of Statistical Learning Theory"
- Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"
- Jacot et al. (2018). "Neural Tangent Kernel"
- Belkin et al. (2019). "Reconciling modern machine learning with the
  bias-variance trade-off" (double descent)

---

## Related Work

- **Knowledge Distillation**: Hinton et al. (2015) - temperature-scaled softmax
- **Importance Sampling**: Prioritizing informative training examples
- **Uncertainty Quantification**: Gal & Ghahramani (2016) - dropout-based uncertainty
- **Fisher Information in Deep Learning**: Martens (2014) - natural gradient methods
- **Model Selection**: Akaike (1974) - AIC; Schwarz (1978) - BIC
- **Robust Regression**: Lange et al. (1989) - t-distribution for outlier-robust estimation

## Files

| File | Description |
|------|-------------|
| `scripts/train_pearltrees_federated.py` | Federated model training with cluster criterion |
| `scripts/distill_federated_to_transformer.py` | Transformer distillation |
| `src/unifyweaver/targets/python_runtime/projection_transformer.py` | Transformer architecture |
| `models/skills_qa_federated_optimized.pkl` | Trained federated model (124 clusters) |
| `models/skills_qa_transformer_6x3.pt` | Best distilled transformer (H=6, L=3) |
