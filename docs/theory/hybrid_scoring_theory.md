# Theory: Hybrid Scoring for Bookmark Filing

## Overview

The UnifyWeaver Filing Assistant uses a hybrid scoring mechanism to retrieve the best folders for a given bookmark. It combines two distinct signals:
1.  **Projected Score**: Represents structural/organizational affinity (based on a user-specific embedding projection).
2.  **Raw Score**: Represents semantic affinity (based on the base embedding model).

This document explains the mathematical justification for the blending strategy.

## 1. The Projections

### Softmax Routing (Upstream)
The **Projected Query** $Q_{proj}$ is constructed by routing the raw query $Q$ through a set of cluster-specific projection matrices $\{W_k\}$.
$$ Q_{proj} = \sum_{k} w_k (Q \cdot W_k) $$
The weights $w_k$ are calculated using a **Softmax** function with temperature $\tau$, based on the similarity of $Q$ to cluster representatives.
$$ w_k = \frac{\exp(Sim(Q, C_k)/\tau)}{\sum_j \exp(Sim(Q, C_j)/\tau)} $$
This is chosen because:
*   It enforces sparsity (winner-take-all behavior for clear routing).
*   It aligns with attention mechanisms in deep learning.

### Raw Query
The **Raw Query** is simply the embedding $Q$ from the base model (e.g. Nomic v1.5).

## 2. The Scoring Problem

We obtain two sets of cosine similarity scores for every target $T_i$:
*   $S_{proj}^{(i)} = \cos(Q_{proj}, T_i)$
*   $S_{raw}^{(i)} = \cos(Q, T_i)$

**Problem**: These scores have different distributions.
*   $S_{raw}$ is often "sharp" and high for exact matches (0.6 - 0.8).
*   $S_{proj}$ is often "diffuse" and lower (0.1 - 0.3) due to the smoothing effect of projection averaging.

Linearly summing them ($S = \alpha S_{proj} + (1-\alpha) S_{raw}$) would cause the Raw Score to dominate unjustly, unless $\alpha$ is tuned extremely high (e.g. 0.9).

## 3. The Blending Solution: L1 Probability Mass

To solve the scale mismatch, we treat both score vectors as **Unnormalized Probability Distributions**.

We apply:
1.  **ReLU (Filtering)**: We discard negative scores (opposites), as they represent dissimilarity.
    $$ S' = \max(0, S) $$
2.  **L1 Normalization (Scaling)**: We normalize the vector so it sums to 1. This converts absolute "similarity magnitude" into "relative probability mass".
    $$ P_i = \frac{S'_i}{\sum_j S'_j} $$

### Why L1 Normalization?
By normalizing both vectors to sum to 1, we ensure that:
*   A "sharp" distribution (Raw) distributes its mass to fewer items ($p \approx 0.9$).
*   A "diffuse" distribution (Projected) distributes its mass to many items ($p \approx 0.05$).

When we blend:
$$ P_{final} = \alpha P_{proj} + (1-\alpha) P_{raw} $$
The blending coefficient $\alpha$ strictly controls the contribution of each *signal source* to the final decision, independent of the absolute magnitude of the underlying cosine scores.

### Comparison with Softmax Blending
We deliberately chose **L1 Normalization** over Softmax Normalization for the blending step.
*   **Softmax** ($\exp(S/\tau)$) introduces exponential distortion. It effectively re-routes the query again.
*   **L1** is linear. If a raw score is twice as high as another, its probability mass is roughly twice as high. This preserves the relative ranking within each signal more faithfully.

## 4. Conclusion
The hybrid scoring uses **Softmax Routing** for constructing the projection (to select the right transformation expert) but uses **L1 Probability Blending** for combining the final scores (to fairly weight the structural vs semantic evidence).
