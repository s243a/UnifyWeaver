# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Smoothing Basis Projection for Multi-Cluster LDA.

Uses gradient-based basis with alternating optimization:
- Coefficients: closed-form least squares (given basis)
- Basis: gradient descent updates

This approach leverages answer similarity to regularize when questions are sparse.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def frobenius_inner(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius inner product: ⟨A, B⟩_F = trace(A^T B) = sum(A * B)"""
    return np.sum(A * B)


def normalize_matrix(M: np.ndarray) -> np.ndarray:
    """Normalize matrix to unit Frobenius norm."""
    norm = np.sqrt(frobenius_inner(M, M))
    if norm > 1e-10:
        return M / norm
    return M


def orthogonalize_basis(basis: List[np.ndarray]) -> List[np.ndarray]:
    """Gram-Schmidt orthogonalization of basis matrices."""
    result = []
    for M in basis:
        M_orth = M.copy()
        for B in result:
            M_orth = M_orth - frobenius_inner(M_orth, B) * B
        M_norm = normalize_matrix(M_orth)
        if frobenius_inner(M_norm, M_norm) > 0.1:  # Not degenerate
            result.append(M_norm)
    return result


def compute_gradient(W: np.ndarray, Q: np.ndarray, A: np.ndarray,
                     cosine_weight: float = 0.5) -> np.ndarray:
    """
    Compute gradient of combined MSE + cosine loss.

    L = (1-λ)||QW - A||² + λ(1 - cos_sim(QW, A))
    """
    pred = Q @ W
    n = len(Q)

    # MSE gradient: 2 * Q^T @ (QW - A) / n
    grad_mse = 2 * Q.T @ (pred - A) / n

    # Cosine gradient (simplified but effective)
    pred_norms = np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8
    A_norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8

    pred_normalized = pred / pred_norms
    A_normalized = A / A_norms

    # Gradient to maximize cosine similarity
    grad_cos = -Q.T @ A_normalized / n

    return (1 - cosine_weight) * grad_mse + cosine_weight * grad_cos


def solve_for_alpha(Q: np.ndarray, A: np.ndarray,
                    basis: List[np.ndarray]) -> np.ndarray:
    """
    Given basis, solve for optimal coefficients via least squares.

    We want: Σ_k α_k (Q @ G_k) ≈ A
    This is linear in α, solved via least squares.
    """
    K = len(basis)
    if K == 0:
        return np.array([])

    # Handle sparse clusters: broadcast A to match Q's row count
    if A.shape[0] == 1 and Q.shape[0] > 1:
        A = np.tile(A, (Q.shape[0], 1))

    # Project each basis through Q: P[:, k] = vec(Q @ G_k)
    P = np.column_stack([(Q @ G_k).ravel() for G_k in basis])

    # Least squares: P @ α ≈ vec(A)
    alpha, _, _, _ = np.linalg.lstsq(P, A.ravel(), rcond=None)

    return alpha


def reconstruct_W(alpha: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
    """Reconstruct W from coefficients and basis: W = Σ_k α_k G_k"""
    if len(basis) == 0:
        return np.zeros((basis[0].shape if basis else (1, 1)))
    return sum(alpha[k] * basis[k] for k in range(len(basis)))


class SmoothingBasisProjection:
    """
    Multi-cluster projection using smoothing basis.

    Clusters share a common basis of projection matrices.
    Each cluster has coefficients determining its linear combination.
    """

    def __init__(self, num_basis: int = 4, cosine_weight: float = 0.5):
        """
        Args:
            num_basis: Number of basis matrices K
            cosine_weight: Weight for cosine loss (0=MSE only, 1=cosine only)
        """
        self.num_basis = num_basis
        self.cosine_weight = cosine_weight
        self.basis: List[np.ndarray] = []
        self.alpha: np.ndarray = None  # Shape (N, K)
        self.centroids: List[np.ndarray] = []  # For routing

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]],
              num_iterations: int = 100, lr: float = 0.01,
              log_interval: int = 20) -> List[float]:
        """
        Train smoothing basis projection.

        Args:
            clusters: List of (Q_i, A_i) - question embeddings and answer embedding
                      Q_i shape: (n_questions, d) or (d,) for single question
                      A_i shape: (d,) - the answer embedding
            num_iterations: Training iterations
            lr: Learning rate for basis updates
            log_interval: Log every N iterations

        Returns:
            List of loss values per iteration
        """
        # Normalize inputs and broadcast A to match Q's row count
        processed_clusters = []
        for Q, A in clusters:
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            # Broadcast A to match Q's row count for consistent training
            if A.shape[0] == 1 and Q.shape[0] > 1:
                A = np.tile(A, (Q.shape[0], 1))
            processed_clusters.append((Q, A))

        clusters = processed_clusters
        N = len(clusters)
        d = clusters[0][0].shape[1]
        K = min(self.num_basis, N)  # Can't have more basis than clusters

        logger.info(f"Training smoothing basis: N={N} clusters, K={K} basis, d={d}")

        # Store centroids for routing
        self.centroids = [np.mean(Q, axis=0) for Q, A in clusters]

        # Initialize basis from per-cluster regularized solutions
        W_init = []
        for Q, A in clusters:
            # Handle sparse clusters: broadcast A and use regularization
            if A.shape[0] == 1 and Q.shape[0] > 1:
                A = np.tile(A, (Q.shape[0], 1))
            reg = 0.01 * np.eye(d)
            W = np.linalg.solve(Q.T @ Q + reg, Q.T @ A)
            W_init.append(W)

        # Extract orthogonal basis from initial solutions
        self.basis = orthogonalize_basis(W_init[:K])
        K = len(self.basis)  # May be reduced if some were degenerate

        if K == 0:
            logger.warning("Could not extract any basis vectors!")
            return []

        logger.info(f"Extracted {K} orthogonal basis matrices")

        # Initialize coefficients via least squares
        self.alpha = np.zeros((N, K))
        for i, (Q, A) in enumerate(clusters):
            self.alpha[i] = solve_for_alpha(Q, A, self.basis)

        # Alternating optimization
        losses = []
        for iteration in range(num_iterations):
            total_loss = 0.0

            # Step 1: Fix basis, solve for coefficients (closed form)
            for i, (Q, A) in enumerate(clusters):
                self.alpha[i] = solve_for_alpha(Q, A, self.basis)

            # Step 2: Compute loss and gradients for basis update
            total_grad = [np.zeros_like(self.basis[k]) for k in range(K)]

            for i, (Q, A) in enumerate(clusters):
                W_i = reconstruct_W(self.alpha[i], self.basis)

                # Compute loss
                pred = Q @ W_i
                mse = np.mean((pred - A) ** 2)

                # Cosine similarity
                pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
                A_norm = A / (np.linalg.norm(A) + 1e-8)
                cos_sim = np.mean(np.sum(pred_norm * A_norm, axis=1))

                loss_i = (1 - self.cosine_weight) * mse + self.cosine_weight * (1 - cos_sim)
                total_loss += loss_i

                # Compute gradient
                grad_W = compute_gradient(W_i, Q, A, self.cosine_weight)

                # Accumulate gradient for each basis element
                for k in range(K):
                    total_grad[k] += self.alpha[i, k] * grad_W

            # Update basis via gradient descent
            for k in range(K):
                self.basis[k] = self.basis[k] - lr * total_grad[k]

            # Re-orthogonalize periodically
            if iteration % 10 == 0:
                self.basis = orthogonalize_basis(self.basis)
                K = len(self.basis)
                if K < self.alpha.shape[1]:
                    self.alpha = self.alpha[:, :K]

            losses.append(total_loss / N)

            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                logger.info(f"Iter {iteration}: loss={total_loss/N:.6f}")

        return losses

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query using smoothing basis with soft routing.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing

        Returns:
            Projected embedding (d,)
        """
        if len(self.basis) == 0 or self.alpha is None:
            return query_emb

        # Compute routing weights based on centroid similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        similarities = []
        for centroid in self.centroids:
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            sim = np.dot(query_norm, centroid_norm)
            similarities.append(sim)

        similarities = np.array(similarities)

        # Softmax routing
        scaled = similarities / temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        # Weighted combination of cluster projections
        projected = np.zeros_like(query_emb)
        for i, w in enumerate(weights):
            W_i = reconstruct_W(self.alpha[i], self.basis)
            projected += w * (query_emb @ W_i)

        return projected


class MultiHeadLDABaseline:
    """
    Standard multi-head LDA for comparison.
    Each cluster has independent centroid and answer embedding.
    """

    def __init__(self):
        self.heads = []  # List of (centroid, answer_emb)

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Train multi-head LDA.

        Args:
            clusters: List of (Q_i, A_i) - question embeddings and answer embedding
        """
        self.heads = []
        for Q, A in clusters:
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)

            centroid = np.mean(Q, axis=0)
            answer_emb = np.mean(A, axis=0)  # In case A has multiple rows
            self.heads.append((centroid, answer_emb))

        logger.info(f"Trained multi-head LDA with {len(self.heads)} heads")

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query using multi-head routing.
        """
        if len(self.heads) == 0:
            return query_emb

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        similarities = []
        for centroid, _ in self.heads:
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            sim = np.dot(query_norm, centroid_norm)
            similarities.append(sim)

        similarities = np.array(similarities)

        # Softmax routing
        scaled = similarities / temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        # Weighted combination of answer embeddings
        projected = np.zeros_like(query_emb)
        for i, (_, answer_emb) in enumerate(self.heads):
            projected += weights[i] * answer_emb

        return projected


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=== Smoothing Basis Test ===\n")

    # Create synthetic data
    np.random.seed(42)
    d = 64  # Embedding dimension
    N = 6   # Number of clusters

    # Create clusters with related answers
    clusters = []
    for i in range(N):
        # 1-2 questions per cluster (sparse!)
        n_questions = np.random.randint(1, 3)
        Q = np.random.randn(n_questions, d)
        A = np.random.randn(1, d) + (i // 2) * 0.5  # Groups of 2 have similar answers
        clusters.append((Q, A))

    print(f"Created {N} clusters with 1-2 questions each")

    # Train smoothing basis
    print("\nTraining Smoothing Basis...")
    sb = SmoothingBasisProjection(num_basis=3, cosine_weight=0.5)
    losses = sb.train(clusters, num_iterations=50, lr=0.01, log_interval=10)

    # Train baseline
    print("\nTraining Multi-Head LDA Baseline...")
    baseline = MultiHeadLDABaseline()
    baseline.train(clusters)

    # Test projection
    print("\nTesting projection...")
    test_query = np.random.randn(d)

    sb_proj = sb.project(test_query)
    baseline_proj = baseline.project(test_query)

    print(f"Query norm: {np.linalg.norm(test_query):.3f}")
    print(f"Smoothing basis projection norm: {np.linalg.norm(sb_proj):.3f}")
    print(f"Baseline projection norm: {np.linalg.norm(baseline_proj):.3f}")

    print("\n=== Test Complete ===")
