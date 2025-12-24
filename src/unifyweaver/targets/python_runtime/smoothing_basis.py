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


class ResidualBasisProjection:
    """
    Hybrid projection: FFT prior + basis-learned residuals.

    Formulation:
        W_i = W_fft_i + ΔW_i
        ΔW_i = Σ_k α_ik G_k

    Where:
        - W_fft is the FFT-smoothed projection (global structure)
        - ΔW is the residual learned by basis decomposition (local refinement)
        - α is regularized toward zero (defaults to FFT unless deviation helps)

    This combines:
        - FFT's strength at capturing global cross-cluster structure
        - Basis decomposition's ability to learn cluster-specific corrections

    The basis vectors span the "tangent space" of deviations from FFT.
    """

    def __init__(self, num_basis: int = 4, fft_cutoff: float = 0.5,
                 fft_blend: float = 0.7, alpha_reg: float = 0.1,
                 cosine_weight: float = 0.5):
        """
        Args:
            num_basis: Number of basis matrices for residual
            fft_cutoff: FFT low-pass filter cutoff
            fft_blend: FFT blend factor (1.0 = pure FFT prior)
            alpha_reg: L2 regularization on basis coefficients
            cosine_weight: Weight for cosine loss (0=MSE only, 1=cosine only)
        """
        self.num_basis = num_basis
        self.fft_cutoff = fft_cutoff
        self.fft_blend = fft_blend
        self.alpha_reg = alpha_reg
        self.cosine_weight = cosine_weight

        # FFT components
        self.W_fft: List[np.ndarray] = []
        self.centroids: List[np.ndarray] = []

        # Residual basis components
        self.basis: List[np.ndarray] = []
        self.alpha: np.ndarray = None  # Shape (N, K)

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]],
              num_iterations: int = 50, lr: float = 0.01,
              log_interval: int = 20) -> dict:
        """
        Train hybrid FFT + residual basis projection.

        Args:
            clusters: List of (Q_i, A_i) tuples
            num_iterations: Iterations for residual basis training
            lr: Learning rate for basis updates
            log_interval: Logging interval

        Returns:
            Dict with training stats
        """
        # Import FFT here to avoid circular import
        try:
            from .fft_smoothing import FFTSmoothingProjection
        except ImportError:
            from fft_smoothing import FFTSmoothingProjection

        # Normalize inputs
        processed = []
        for Q, A in clusters:
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            if A.shape[0] == 1 and Q.shape[0] > 1:
                A = np.tile(A, (Q.shape[0], 1))
            processed.append((Q, A))

        N = len(processed)
        d = processed[0][0].shape[1]
        K = min(self.num_basis, N)

        logger.info(f"Training ResidualBasis: N={N}, K={K}, d={d}")
        logger.info(f"  FFT: cutoff={self.fft_cutoff}, blend={self.fft_blend}")
        logger.info(f"  Basis: alpha_reg={self.alpha_reg}")

        # Store centroids
        self.centroids = [np.mean(Q, axis=0) for Q, A in processed]

        # Step 1: Train FFT smoothing to get W_fft
        fft_proj = FFTSmoothingProjection(
            cutoff=self.fft_cutoff,
            blend_factor=self.fft_blend
        )
        fft_proj.train(clusters)
        self.W_fft = fft_proj.W_smoothed.copy()

        # Step 2: Compute target W matrices (per-cluster least squares)
        W_target = []
        for Q, A in processed:
            reg = 0.01 * np.eye(d)
            W = np.linalg.solve(Q.T @ Q + reg, Q.T @ A)
            W_target.append(W)

        # Step 3: Compute residuals R_i = W_target_i - W_fft_i
        residuals = [W_target[i] - self.W_fft[i] for i in range(N)]

        # Step 4: Initialize basis from residuals
        self.basis = orthogonalize_basis(residuals[:K])
        K = len(self.basis)

        if K == 0:
            logger.warning("No basis extracted from residuals - using pure FFT")
            self.alpha = np.zeros((N, 1))
            return {'fft_only': True, 'losses': []}

        logger.info(f"Extracted {K} orthogonal residual basis matrices")

        # Step 5: Initialize coefficients
        self.alpha = np.zeros((N, K))
        for i in range(N):
            # Solve for α_i: Σ_k α_ik G_k ≈ R_i
            # This is least squares on the residual
            P = np.column_stack([G.ravel() for G in self.basis])
            alpha_i, _, _, _ = np.linalg.lstsq(P, residuals[i].ravel(), rcond=None)
            self.alpha[i] = alpha_i

        # Step 6: Alternating optimization with regularization
        losses = []
        for iteration in range(num_iterations):
            total_loss = 0.0

            # Fix basis, solve for regularized coefficients
            for i, (Q, A) in enumerate(processed):
                # Target for residual: what we want ΔW to achieve
                # W_target = W_fft + ΔW, so ΔW = W_target - W_fft
                # But we also want to minimize ||ΔW||, so we regularize

                # Least squares with L2 regularization on α
                P = np.column_stack([(Q @ G).ravel() for G in self.basis])
                target = (A - Q @ self.W_fft[i]).ravel()

                # Regularized least squares: (P^T P + λI) α = P^T target
                reg_matrix = self.alpha_reg * np.eye(K)
                self.alpha[i] = np.linalg.solve(
                    P.T @ P + reg_matrix,
                    P.T @ target
                )

            # Compute loss
            for i, (Q, A) in enumerate(processed):
                W_i = self.W_fft[i] + reconstruct_W(self.alpha[i], self.basis)
                pred = Q @ W_i

                # MSE loss
                mse = np.mean((pred - A) ** 2)

                # Cosine loss
                pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
                A_norm = A / (np.linalg.norm(A) + 1e-8)
                cos_sim = np.mean(np.sum(pred_norm * A_norm, axis=1))

                # Regularization loss
                reg_loss = self.alpha_reg * np.sum(self.alpha[i] ** 2)

                loss_i = ((1 - self.cosine_weight) * mse +
                          self.cosine_weight * (1 - cos_sim) +
                          reg_loss)
                total_loss += loss_i

            # Gradient update for basis (simplified - mainly α does the work)
            if iteration < num_iterations // 2:
                # Early iterations: also update basis
                total_grad = [np.zeros_like(G) for G in self.basis]

                for i, (Q, A) in enumerate(processed):
                    W_i = self.W_fft[i] + reconstruct_W(self.alpha[i], self.basis)
                    grad_W = compute_gradient(W_i, Q, A, self.cosine_weight)

                    for k in range(K):
                        total_grad[k] += self.alpha[i, k] * grad_W

                for k in range(K):
                    self.basis[k] = self.basis[k] - lr * total_grad[k]

                # Re-orthogonalize
                if iteration % 10 == 0:
                    self.basis = orthogonalize_basis(self.basis)
                    K = len(self.basis)
                    if K < self.alpha.shape[1]:
                        self.alpha = self.alpha[:, :K]

            losses.append(total_loss / N)

            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                avg_alpha_norm = np.mean(np.linalg.norm(self.alpha, axis=1))
                logger.info(f"Iter {iteration}: loss={total_loss/N:.6f}, "
                           f"avg_|α|={avg_alpha_norm:.4f}")

        return {
            'fft_only': False,
            'losses': losses,
            'final_alpha_norm': float(np.mean(np.linalg.norm(self.alpha, axis=1))),
            'num_basis': len(self.basis),
        }

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query using FFT prior + residual correction.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing

        Returns:
            Projected embedding (d,)
        """
        if len(self.W_fft) == 0:
            return query_emb

        # Compute routing weights
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        similarities = []
        for centroid in self.centroids:
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            similarities.append(np.dot(query_norm, centroid_norm))

        similarities = np.array(similarities)

        # Softmax routing
        scaled = similarities / temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        # Weighted combination of projections
        projected = np.zeros_like(query_emb)
        for i, w in enumerate(weights):
            # W = W_fft + ΔW
            if len(self.basis) > 0 and self.alpha is not None:
                delta_W = reconstruct_W(self.alpha[i], self.basis)
                W_i = self.W_fft[i] + delta_W
            else:
                W_i = self.W_fft[i]
            projected += w * (query_emb @ W_i)

        return projected

    def get_residual_stats(self) -> dict:
        """Get statistics about the learned residuals."""
        if self.alpha is None or len(self.basis) == 0:
            return {'has_residuals': False}

        # Compute ΔW norms
        delta_norms = []
        fft_norms = []
        for i in range(len(self.W_fft)):
            delta_W = reconstruct_W(self.alpha[i], self.basis)
            delta_norms.append(np.linalg.norm(delta_W))
            fft_norms.append(np.linalg.norm(self.W_fft[i]))

        return {
            'has_residuals': True,
            'num_basis': len(self.basis),
            'mean_delta_norm': float(np.mean(delta_norms)),
            'mean_fft_norm': float(np.mean(fft_norms)),
            'delta_to_fft_ratio': float(np.mean(delta_norms) / (np.mean(fft_norms) + 1e-8)),
            'alpha_sparsity': float(np.mean(np.abs(self.alpha) < 0.01)),
        }


class SmoothingKernel:
    """Enumeration of available smoothing kernels."""
    GAUSSIAN_RBF = "gaussian"
    MATERN_32 = "matern_32"
    MATERN_52 = "matern_52"
    LAPLACIAN = "laplacian"
    RATIONAL_QUADRATIC = "rational_quadratic"


class KernelSmoothingProjection:
    """
    Kernel-based smoothing for multi-cluster projections.

    Uses kernel-weighted averaging to smooth W matrices based on
    semantic distance between cluster centroids. This respects the
    full multi-dimensional similarity structure, unlike 1D FFT.

    Theory:
        W_smoothed_i = Σ_j K(centroid_i, centroid_j) W_j / Σ_j K(...)

        The kernel K defines how influence decays with distance.
        This is equivalent to solving the Graph Laplacian smoothing
        problem via Green's functions.

    Kernels:
        - gaussian: exp(-r²/2σ²) - Infinitely smooth, stable
        - matern_32: (1+√3r/ℓ)exp(-√3r/ℓ) - C² smooth
        - matern_52: (1+√5r/ℓ+5r²/3ℓ²)exp(-√5r/ℓ) - C⁴ smooth (recommended)
        - laplacian: exp(-r/ξ) - Rough, can be unstable
        - rational_quadratic: (1 + r²/2αℓ²)^(-α) - Scale mixture of Gaussians

    See docs/proposals/KERNEL_SMOOTHING_THEORY.md for details.
    """

    def __init__(
        self,
        kernel: str = SmoothingKernel.MATERN_52,
        length_scale: float = 1.0,
        alpha: float = 1.0,  # For rational quadratic
        regularization: float = 1e-6,
        normalize_kernel: bool = True,
        cosine_weight: float = 0.5,
    ):
        """
        Initialize kernel smoothing projection.

        Args:
            kernel: Kernel type (gaussian, matern_32, matern_52, laplacian, rational_quadratic)
            length_scale: Controls how quickly influence decays with distance
            alpha: Shape parameter for rational quadratic kernel
            regularization: Small value added to diagonal for numerical stability
            normalize_kernel: If True, normalize kernel rows to sum to 1
            cosine_weight: Weight for cosine loss (vs MSE) when computing W_target
        """
        self.kernel = kernel
        self.length_scale = length_scale
        self.alpha = alpha
        self.regularization = regularization
        self.normalize_kernel = normalize_kernel
        self.cosine_weight = cosine_weight

        # Computed during training
        self.centroids: List[np.ndarray] = []
        self.W_target: List[np.ndarray] = []  # Per-cluster target W
        self.W_smoothed: List[np.ndarray] = []  # Kernel-smoothed W
        self.K: np.ndarray = None  # Kernel matrix
        self.K_normalized: np.ndarray = None  # Row-normalized kernel

    def _compute_kernel_value(self, r: np.ndarray) -> np.ndarray:
        """
        Compute kernel values for given distances.

        Args:
            r: Distance matrix (N x N) or distance values

        Returns:
            Kernel values K(r)
        """
        # Avoid division by zero
        r = np.maximum(r, 1e-10)
        scaled = r / self.length_scale

        if self.kernel == SmoothingKernel.GAUSSIAN_RBF:
            return np.exp(-scaled**2 / 2)

        elif self.kernel == SmoothingKernel.MATERN_32:
            sqrt3 = np.sqrt(3) * scaled
            return (1 + sqrt3) * np.exp(-sqrt3)

        elif self.kernel == SmoothingKernel.MATERN_52:
            sqrt5 = np.sqrt(5) * scaled
            return (1 + sqrt5 + sqrt5**2 / 3) * np.exp(-sqrt5)

        elif self.kernel == SmoothingKernel.LAPLACIAN:
            return np.exp(-scaled)

        elif self.kernel == SmoothingKernel.RATIONAL_QUADRATIC:
            return (1 + scaled**2 / (2 * self.alpha))**(-self.alpha)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_kernel_matrix(self, centroids: List[np.ndarray]) -> np.ndarray:
        """
        Compute kernel matrix from cluster centroids.

        Args:
            centroids: List of centroid vectors

        Returns:
            K: Kernel matrix (N x N)
        """
        N = len(centroids)
        C = np.array(centroids)

        # Compute pairwise distances
        # ||c_i - c_j||² = ||c_i||² + ||c_j||² - 2<c_i, c_j>
        norms_sq = np.sum(C**2, axis=1)
        distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * (C @ C.T)
        distances = np.sqrt(np.maximum(distances_sq, 0))

        # Compute kernel values
        K = self._compute_kernel_value(distances)

        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(K, 1.0)

        # Add regularization for numerical stability
        K += self.regularization * np.eye(N)

        return K

    def _compute_target_W(self, Q: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Compute target W matrix for a cluster using pseudoinverse.

        Args:
            Q: Query matrix (n_queries x d)
            A: Answer matrix (n_queries x d)

        Returns:
            W: Projection matrix (d x d)
        """
        # Pseudoinverse solution: W = pinv(Q) @ A
        W, _, _, _ = np.linalg.lstsq(Q, A, rcond=None)
        return W

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]],
              log_interval: int = 10) -> dict:
        """
        Train kernel smoothing projection.

        Args:
            clusters: List of (Q_i, A_i) tuples per cluster
            log_interval: Logging interval (unused, for API compatibility)

        Returns:
            Dict with training stats including kernel properties
        """
        # Normalize inputs
        processed = []
        for Q, A in clusters:
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            if A.shape[0] == 1 and Q.shape[0] > 1:
                A = np.tile(A, (Q.shape[0], 1))
            processed.append((Q, A))

        N = len(processed)
        d = processed[0][0].shape[1]

        logger.info(f"Training KernelSmoothing: N={N}, d={d}")
        logger.info(f"  Kernel: {self.kernel}, length_scale={self.length_scale}")

        # Step 1: Compute centroids
        self.centroids = [np.mean(Q, axis=0) for Q, A in processed]

        # Step 2: Compute target W for each cluster (unconstrained solution)
        self.W_target = []
        for Q, A in processed:
            W = self._compute_target_W(Q, A)
            self.W_target.append(W)

        # Step 3: Compute kernel matrix
        self.K = self._compute_kernel_matrix(self.centroids)

        # Step 4: Normalize kernel rows (optional)
        if self.normalize_kernel:
            row_sums = self.K.sum(axis=1, keepdims=True)
            self.K_normalized = self.K / row_sums
        else:
            self.K_normalized = self.K

        # Step 5: Apply kernel smoothing to W matrices
        # W_smoothed_i = Σ_j K_normalized(i,j) W_j
        W_target_stacked = np.array(self.W_target)  # Shape: (N, d, d)
        W_target_flat = W_target_stacked.reshape(N, -1)  # Shape: (N, d*d)

        W_smoothed_flat = self.K_normalized @ W_target_flat  # Shape: (N, d*d)
        W_smoothed_stacked = W_smoothed_flat.reshape(N, d, d)

        self.W_smoothed = [W_smoothed_stacked[i] for i in range(N)]

        # Compute stats
        kernel_sparsity = np.mean(self.K < 0.01)
        kernel_condition = np.linalg.cond(self.K)

        stats = {
            "num_clusters": N,
            "embedding_dim": d,
            "kernel": self.kernel,
            "length_scale": self.length_scale,
            "kernel_sparsity": float(kernel_sparsity),
            "kernel_condition_number": float(kernel_condition),
            "mean_kernel_value": float(np.mean(self.K)),
        }

        logger.info(f"  Kernel condition number: {kernel_condition:.2f}")
        logger.info(f"  Mean kernel value: {np.mean(self.K):.4f}")

        return stats

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query using soft routing and kernel-smoothed W matrices.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing

        Returns:
            Projected embedding (d,)
        """
        if not self.centroids or not self.W_smoothed:
            raise ValueError("Model not trained. Call train() first.")

        query_emb = query_emb.flatten()

        # Compute similarities to all centroids
        similarities = []
        for centroid in self.centroids:
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            sim = np.dot(query_norm, centroid_norm)
            similarities.append(sim)

        similarities = np.array(similarities)

        # Softmax routing
        exp_sim = np.exp((similarities - np.max(similarities)) / temperature)
        weights = exp_sim / (np.sum(exp_sim) + 1e-8)

        # Weighted combination of projected queries
        projected = np.zeros(len(query_emb))
        for i, (W, weight) in enumerate(zip(self.W_smoothed, weights)):
            projected += weight * (query_emb @ W)

        return projected

    def get_kernel_stats(self) -> dict:
        """Get statistics about the learned kernel structure."""
        if self.K is None:
            return {"error": "Model not trained"}

        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(self.K)

        return {
            "kernel": self.kernel,
            "length_scale": self.length_scale,
            "num_clusters": len(self.centroids),
            "condition_number": float(np.linalg.cond(self.K)),
            "min_eigenvalue": float(np.min(eigenvalues)),
            "max_eigenvalue": float(np.max(eigenvalues)),
            "effective_rank": float(np.sum(eigenvalues > 0.01 * np.max(eigenvalues))),
            "mean_off_diagonal": float(np.mean(self.K[~np.eye(len(self.K), dtype=bool)])),
        }


if __name__ == "__main__":
    # Test code
    print("=== Smoothing Basis Test ===")

    # Create synthetic test data
    np.random.seed(42)
    N = 10  # Number of clusters
    d = 32  # Embedding dimension

    clusters = []
    for i in range(N):
        n_questions = np.random.randint(1, 3)
        Q = np.random.randn(n_questions, d)
        A = np.random.randn(1, d)
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

    # Test kernel smoothing methods
    print("\n=== Testing Kernel Smoothing Methods ===")

    kernels_to_test = [
        (SmoothingKernel.GAUSSIAN_RBF, 1.0),
        (SmoothingKernel.MATERN_32, 1.0),
        (SmoothingKernel.MATERN_52, 1.0),
        (SmoothingKernel.LAPLACIAN, 1.0),
        (SmoothingKernel.RATIONAL_QUADRATIC, 1.0),
    ]

    for kernel, length_scale in kernels_to_test:
        print(f"\nTraining KernelSmoothing ({kernel})...")
        ks = KernelSmoothingProjection(
            kernel=kernel,
            length_scale=length_scale,
        )
        stats = ks.train(clusters)

        ks_proj = ks.project(test_query)
        kernel_stats = ks.get_kernel_stats()

        print(f"  Condition number: {kernel_stats['condition_number']:.2f}")
        print(f"  Effective rank: {kernel_stats['effective_rank']:.0f}")
        print(f"  Projection norm: {np.linalg.norm(ks_proj):.3f}")

    print("\n=== Test Complete ===")
