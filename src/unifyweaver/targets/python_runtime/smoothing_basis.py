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


def conjugate_gradient(
    A_func,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    precond_func=None,
) -> Tuple[np.ndarray, int, float]:
    """
    Conjugate Gradient solver for Ax = b.

    Args:
        A_func: Function that computes A @ x (matrix-free)
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        max_iter: Maximum iterations
        tol: Convergence tolerance (relative residual)
        precond_func: Optional preconditioner M^{-1} @ r

    Returns:
        x: Solution vector
        iterations: Number of iterations used
        residual: Final relative residual
    """
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)

    r = b - A_func(x)
    if precond_func:
        z = precond_func(r)
    else:
        z = r

    p = z.copy()
    rz_old = np.dot(r, z)

    b_norm = np.linalg.norm(b) + 1e-10

    for i in range(max_iter):
        Ap = A_func(p)
        alpha = rz_old / (np.dot(p, Ap) + 1e-10)

        x = x + alpha * p
        r = r - alpha * Ap

        rel_residual = np.linalg.norm(r) / b_norm
        if rel_residual < tol:
            return x, i + 1, rel_residual

        if precond_func:
            z = precond_func(r)
        else:
            z = r

        rz_new = np.dot(r, z)
        beta = rz_new / (rz_old + 1e-10)

        p = z + beta * p
        rz_old = rz_new

    return x, max_iter, np.linalg.norm(r) / b_norm


class UnifiedKernelBasisProjection:
    """
    Unified projection combining:
    1. Matérn-5/2 kernel smoothing (graph Laplacian regularization)
    2. K basis vectors (tangent space / shared structure)
    3. Coupled optimization via Conjugate Gradient

    Loss function:
        L = Σ_i ||Q_i W_i - A_i||² + λ Σ_{i,j} K(i,j) ||W_i - W_j||²

    With basis decomposition W_i = Σ_k α_ik G_k:
        L = Σ_i ||Q_i (Σ_k α_ik G_k) - A_i||² + λ · trace(α^T L_K α)

    Key insight: orthonormal basis decouples the smoothness term per component,
    making the coupled system tractable with sparse CG.

    Optimization:
        1. Fix basis G → solve for α via CG with sparse Laplacian (coupled)
        2. Fix α → gradient descent on G with Gram-Schmidt

    See docs/proposals/KERNEL_SMOOTHING_THEORY.md for theoretical foundation.
    """

    def __init__(
        self,
        num_basis: int = 4,
        length_scale: float = 0.5,
        smoothing_strength: float = 0.1,
        k_neighbors: Optional[int] = None,
        cg_max_iter: int = 50,
        cg_tol: float = 1e-5,
        use_jacobi_precond: bool = True,
        diagonal_init_scale: float = 1.0,
        noise_scale: float = 0.1,
        cosine_weight: float = 0.5,
    ):
        """
        Initialize unified kernel basis projection.

        Args:
            num_basis: Number of shared basis matrices (K)
            length_scale: Matérn-5/2 kernel length scale
            smoothing_strength: λ - Laplacian regularization strength
            k_neighbors: If set, sparsify kernel to k nearest neighbors
            cg_max_iter: Maximum CG iterations
            cg_tol: CG convergence tolerance
            use_jacobi_precond: Use diagonal (Jacobi) preconditioning
            diagonal_init_scale: Scale for diagonal initialization
            noise_scale: Scale for random noise in initialization
            cosine_weight: Weight for cosine loss (vs MSE)
        """
        self.num_basis = num_basis
        self.length_scale = length_scale
        self.smoothing_strength = smoothing_strength
        self.k_neighbors = k_neighbors
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.use_jacobi_precond = use_jacobi_precond
        self.diagonal_init_scale = diagonal_init_scale
        self.noise_scale = noise_scale
        self.cosine_weight = cosine_weight

        # Trained components
        self.basis: List[np.ndarray] = []
        self.alpha: np.ndarray = None  # Shape (N, K)
        self.centroids: List[np.ndarray] = []
        self.L_sparse: np.ndarray = None  # Sparse graph Laplacian
        self.K_matrix: np.ndarray = None  # Kernel matrix

        # Statistics
        self.cg_iterations: List[int] = []
        self.losses: List[float] = []

    def _compute_matern52_kernel(self, centroids: List[np.ndarray]) -> np.ndarray:
        """Compute Matérn-5/2 kernel matrix from centroids."""
        N = len(centroids)
        C = np.array(centroids)

        # Pairwise Euclidean distances
        norms_sq = np.sum(C**2, axis=1)
        distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * (C @ C.T)
        distances = np.sqrt(np.maximum(distances_sq, 0))

        # Matérn-5/2: (1 + √5r/ℓ + 5r²/3ℓ²) exp(-√5r/ℓ)
        scaled = np.sqrt(5) * distances / self.length_scale
        K = (1 + scaled + scaled**2 / 3) * np.exp(-scaled)

        # Diagonal is 1
        np.fill_diagonal(K, 1.0)

        return K

    def _sparsify_kernel(self, K: np.ndarray, k: int) -> np.ndarray:
        """Sparsify kernel to k nearest neighbors per row."""
        N = len(K)
        K_sparse = np.zeros_like(K)

        for i in range(N):
            # Find k largest values (including self)
            top_k_idx = np.argsort(K[i])[-k:]
            K_sparse[i, top_k_idx] = K[i, top_k_idx]

        # Symmetrize
        K_sparse = (K_sparse + K_sparse.T) / 2

        return K_sparse

    def _compute_graph_laplacian(self, K: np.ndarray) -> np.ndarray:
        """Compute graph Laplacian L = D - K."""
        D = np.diag(K.sum(axis=1))
        L = D - K
        return L

    def _initialize_basis_diagonal(self, d: int) -> List[np.ndarray]:
        """
        Initialize basis as diagonally dominant + noise.

        Creates K basis matrices, each centered around identity
        with random noise to break symmetry.
        """
        basis = []
        for k in range(self.num_basis):
            # Start with scaled identity
            G = self.diagonal_init_scale * np.eye(d)
            # Add random noise
            G += self.noise_scale * np.random.randn(d, d)
            basis.append(G)

        return orthogonalize_basis(basis)

    def _initialize_basis_from_clusters(
        self, clusters: List[Tuple[np.ndarray, np.ndarray]], d: int
    ) -> List[np.ndarray]:
        """
        Initialize basis from per-cluster least-squares solutions.

        This provides a much better starting point than random initialization.
        """
        # Compute per-cluster W matrices
        W_init = []
        for Q, A in clusters:
            reg = 0.01 * np.eye(d)
            W = np.linalg.solve(Q.T @ Q + reg, Q.T @ A)
            W_init.append(W)

        # Extract orthogonal basis from first K solutions
        K = min(self.num_basis, len(W_init))
        basis = orthogonalize_basis(W_init[:K])

        # If we need more basis vectors, add diagonal + noise
        while len(basis) < self.num_basis:
            G = self.diagonal_init_scale * np.eye(d)
            G += self.noise_scale * np.random.randn(d, d)
            # Orthogonalize against existing
            for B in basis:
                G = G - frobenius_inner(G, B) * B
            G_norm = normalize_matrix(G)
            if frobenius_inner(G_norm, G_norm) > 0.1:
                basis.append(G_norm)
            else:
                break

        return basis

    def _compute_cluster_design_matrices(
        self, clusters: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute design matrices M_i and target vectors b_i for each cluster.

        For cluster i with Q_i, A_i:
            ||Σ_k α_ik (Q_i G_k) - A_i||² = α_i^T M_i α_i - 2 α_i^T b_i + const

        where:
            M_i[k,l] = trace((Q_i G_k)^T (Q_i G_l))
            b_i[k] = trace((Q_i G_k)^T A_i)
        """
        K = len(self.basis)
        M_list = []
        b_list = []

        for Q, A in clusters:
            # Compute P_ik = Q @ G_k for each basis
            P = [Q @ G for G in self.basis]

            # M_i: K x K matrix
            M_i = np.zeros((K, K))
            for k in range(K):
                for l in range(K):
                    M_i[k, l] = np.sum(P[k] * P[l])

            # b_i: K vector
            b_i = np.array([np.sum(P[k] * A) for k in range(K)])

            M_list.append(M_i)
            b_list.append(b_i)

        return M_list, b_list

    def _solve_coupled_alpha(
        self,
        M_list: List[np.ndarray],
        b_list: List[np.ndarray],
        alpha_init: np.ndarray,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Solve coupled system for α using Conjugate Gradient.

        The system is:
            (block_diag(M_i) + λ (L ⊗ I_K)) α = b

        where:
            - block_diag(M_i) is block-diagonal (data fidelity)
            - L ⊗ I_K is Laplacian regularization (couples clusters)
        """
        N = len(M_list)
        K = len(self.basis)

        # Flatten α: shape (N, K) → (N*K,)
        # Storage: α[i*K + k] = α_ik
        x0 = alpha_init.ravel()

        # Build right-hand side: concatenate b_i
        b = np.concatenate(b_list)

        # Define A @ x operator (matrix-free)
        def A_func(x):
            """Compute (block_diag(M_i) + λ L ⊗ I_K) x."""
            x_2d = x.reshape(N, K)
            result = np.zeros_like(x_2d)

            # Data fidelity term: M_i @ α_i for each cluster
            for i in range(N):
                result[i] = M_list[i] @ x_2d[i]

            # Laplacian regularization: λ L @ α for each basis
            # L acts on the cluster dimension
            for k in range(K):
                result[:, k] += self.smoothing_strength * (self.L_sparse @ x_2d[:, k])

            return result.ravel()

        # Optional Jacobi (diagonal) preconditioner
        precond_func = None
        if self.use_jacobi_precond:
            # Diagonal of Hessian: M_ii[k,k] + λ L_ii
            diag = np.zeros(N * K)
            for i in range(N):
                for k in range(K):
                    diag[i * K + k] = M_list[i][k, k] + self.smoothing_strength * self.L_sparse[i, i]
            diag = np.maximum(diag, 1e-6)  # Ensure positive

            def precond_func(r):
                return r / diag

        # Solve with CG
        x_sol, iterations, residual = conjugate_gradient(
            A_func, b, x0, self.cg_max_iter, self.cg_tol, precond_func
        )

        alpha_new = x_sol.reshape(N, K)
        return alpha_new, iterations, residual

    def train(
        self,
        clusters: List[Tuple[np.ndarray, np.ndarray]],
        num_iterations: int = 50,
        lr: float = 0.01,
        log_interval: int = 10,
    ) -> dict:
        """
        Train unified kernel basis projection.

        Args:
            clusters: List of (Q_i, A_i) tuples
            num_iterations: Alternating optimization iterations
            lr: Learning rate for basis updates
            log_interval: Logging interval

        Returns:
            Dict with training statistics
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
        K = min(self.num_basis, N)

        logger.info(f"Training UnifiedKernelBasis: N={N}, K={K}, d={d}")
        logger.info(f"  Matérn-5/2: length_scale={self.length_scale}")
        logger.info(f"  Smoothing: λ={self.smoothing_strength}")
        if self.k_neighbors:
            logger.info(f"  Sparse: k={self.k_neighbors} neighbors")

        # Store centroids
        self.centroids = [np.mean(Q, axis=0) for Q, A in processed]

        # Step 1: Compute kernel and Laplacian
        self.K_matrix = self._compute_matern52_kernel(self.centroids)

        if self.k_neighbors:
            self.K_matrix = self._sparsify_kernel(self.K_matrix, self.k_neighbors)

        self.L_sparse = self._compute_graph_laplacian(self.K_matrix)

        # Step 2: Initialize basis from per-cluster solutions
        self.basis = self._initialize_basis_from_clusters(processed, d)
        K = len(self.basis)

        logger.info(f"Initialized {K} basis matrices from cluster solutions")

        # Step 3: Initialize α from per-cluster solutions
        self.alpha = np.zeros((N, K))
        for i, (Q, A) in enumerate(processed):
            self.alpha[i] = solve_for_alpha(Q, A, self.basis)

        # Step 4: Alternating optimization
        self.losses = []
        self.cg_iterations = []

        for iteration in range(num_iterations):
            # Step 4a: Fix basis → solve for α via CG
            M_list, b_list = self._compute_cluster_design_matrices(processed)
            self.alpha, cg_iters, cg_residual = self._solve_coupled_alpha(
                M_list, b_list, self.alpha
            )
            self.cg_iterations.append(cg_iters)

            # Step 4b: Compute loss
            total_loss = 0.0
            total_smooth_loss = 0.0

            for i, (Q, A) in enumerate(processed):
                W_i = reconstruct_W(self.alpha[i], self.basis)
                pred = Q @ W_i

                # MSE
                mse = np.mean((pred - A) ** 2)

                # Cosine
                pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
                A_norm = A / (np.linalg.norm(A) + 1e-8)
                cos_sim = np.mean(np.sum(pred_norm * A_norm, axis=1))

                loss_i = (1 - self.cosine_weight) * mse + self.cosine_weight * (1 - cos_sim)
                total_loss += loss_i

            # Smoothness loss: λ trace(α^T L α)
            for k in range(K):
                total_smooth_loss += self.smoothing_strength * (
                    self.alpha[:, k] @ self.L_sparse @ self.alpha[:, k]
                )

            self.losses.append(total_loss / N + total_smooth_loss / N)

            # Step 4c: Fix α → update basis via gradient descent
            total_grad = [np.zeros_like(G) for G in self.basis]

            for i, (Q, A) in enumerate(processed):
                W_i = reconstruct_W(self.alpha[i], self.basis)
                grad_W = compute_gradient(W_i, Q, A, self.cosine_weight)

                for k in range(K):
                    total_grad[k] += self.alpha[i, k] * grad_W

            for k in range(K):
                self.basis[k] = self.basis[k] - lr * total_grad[k]

            # Re-orthogonalize periodically
            if iteration % 5 == 0:
                self.basis = orthogonalize_basis(self.basis)
                K_new = len(self.basis)
                if K_new < K:
                    self.alpha = self.alpha[:, :K_new]
                    K = K_new

            # Logging
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                logger.info(
                    f"Iter {iteration}: loss={self.losses[-1]:.6f}, "
                    f"CG_iters={cg_iters}, CG_res={cg_residual:.2e}"
                )

        # Compute final statistics
        kernel_sparsity = np.mean(self.K_matrix < 0.01)
        effective_neighbors = np.mean(np.sum(self.K_matrix > 0.01, axis=1))

        return {
            "num_clusters": N,
            "num_basis": len(self.basis),
            "embedding_dim": d,
            "length_scale": self.length_scale,
            "smoothing_strength": self.smoothing_strength,
            "kernel_sparsity": float(kernel_sparsity),
            "effective_neighbors": float(effective_neighbors),
            "final_loss": float(self.losses[-1]) if self.losses else 0.0,
            "avg_cg_iterations": float(np.mean(self.cg_iterations)),
            "total_cg_iterations": int(np.sum(self.cg_iterations)),
        }

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query using soft routing and coupled W matrices.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing

        Returns:
            Projected embedding (d,)
        """
        if not self.centroids or not self.basis:
            raise ValueError("Model not trained. Call train() first.")

        query_emb = query_emb.flatten()

        # Compute routing weights
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

        # Weighted combination
        projected = np.zeros(len(query_emb))
        for i, weight in enumerate(weights):
            W_i = reconstruct_W(self.alpha[i], self.basis)
            projected += weight * (query_emb @ W_i)

        return projected

    def get_stats(self) -> dict:
        """Get detailed statistics about the trained model."""
        if self.alpha is None:
            return {"trained": False}

        # Analyze basis structure
        basis_norms = [np.linalg.norm(G) for G in self.basis]

        # Analyze alpha distribution
        alpha_norms = np.linalg.norm(self.alpha, axis=1)

        # Analyze W matrices
        W_norms = []
        for i in range(len(self.centroids)):
            W_i = reconstruct_W(self.alpha[i], self.basis)
            W_norms.append(np.linalg.norm(W_i))

        # Laplacian eigenvalues (for condition analysis)
        L_eigenvalues = np.linalg.eigvalsh(self.L_sparse)

        return {
            "trained": True,
            "num_basis": len(self.basis),
            "basis_norms": basis_norms,
            "mean_alpha_norm": float(np.mean(alpha_norms)),
            "std_alpha_norm": float(np.std(alpha_norms)),
            "mean_W_norm": float(np.mean(W_norms)),
            "laplacian_condition": float(L_eigenvalues[-1] / (L_eigenvalues[1] + 1e-10)),
            "laplacian_spectral_gap": float(L_eigenvalues[1]),
            "total_cg_iterations": int(np.sum(self.cg_iterations)),
            "losses": self.losses,
        }


class KernelSmoothedBasisProjection:
    """
    Hybrid approach: Train basis locally, then smooth with kernel.

    This is a simpler approach than full coupled optimization:
    1. Train SmoothingBasisProjection (uncoupled per-cluster optimization)
    2. Apply Matérn-5/2 kernel smoothing to the resulting W matrices

    This combines:
    - Basis decomposition for parameter efficiency and regularization
    - Kernel smoothing for cross-cluster consistency

    The key insight is that these work better in sequence than jointly.
    """

    def __init__(
        self,
        num_basis: int = 4,
        length_scale: float = 0.5,
        kernel_blend: float = 0.5,
        cosine_weight: float = 0.5,
    ):
        """
        Args:
            num_basis: Number of shared basis matrices
            length_scale: Matérn-5/2 kernel length scale
            kernel_blend: Blend between unsmoothed (0) and smoothed (1) W
            cosine_weight: Weight for cosine loss in basis training
        """
        self.num_basis = num_basis
        self.length_scale = length_scale
        self.kernel_blend = kernel_blend
        self.cosine_weight = cosine_weight

        # Components
        self.basis_proj: Optional[SmoothingBasisProjection] = None
        self.W_smoothed: List[np.ndarray] = []
        self.centroids: List[np.ndarray] = []
        self.K_matrix: np.ndarray = None

    def _compute_matern52_kernel(self, centroids: List[np.ndarray]) -> np.ndarray:
        """Compute Matérn-5/2 kernel matrix."""
        N = len(centroids)
        C = np.array(centroids)

        norms_sq = np.sum(C**2, axis=1)
        distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * (C @ C.T)
        distances = np.sqrt(np.maximum(distances_sq, 0))

        scaled = np.sqrt(5) * distances / self.length_scale
        K = (1 + scaled + scaled**2 / 3) * np.exp(-scaled)
        np.fill_diagonal(K, 1.0)

        return K

    def train(
        self,
        clusters: List[Tuple[np.ndarray, np.ndarray]],
        num_iterations: int = 50,
        lr: float = 0.01,
        log_interval: int = 20,
    ) -> dict:
        """
        Train in two phases: basis learning, then kernel smoothing.
        """
        # Phase 1: Train basis projection
        self.basis_proj = SmoothingBasisProjection(
            num_basis=self.num_basis,
            cosine_weight=self.cosine_weight,
        )
        losses = self.basis_proj.train(
            clusters, num_iterations=num_iterations, lr=lr, log_interval=log_interval
        )

        # Get per-cluster W matrices
        N = len(clusters)
        self.centroids = self.basis_proj.centroids
        W_unsmoothed = []
        for i in range(N):
            W_i = reconstruct_W(self.basis_proj.alpha[i], self.basis_proj.basis)
            W_unsmoothed.append(W_i)

        # Phase 2: Kernel smoothing
        self.K_matrix = self._compute_matern52_kernel(self.centroids)

        # Normalize kernel rows
        K_normalized = self.K_matrix / self.K_matrix.sum(axis=1, keepdims=True)

        # Apply kernel smoothing
        d = W_unsmoothed[0].shape[0]
        W_stack = np.array(W_unsmoothed).reshape(N, -1)  # (N, d*d)
        W_smooth_flat = K_normalized @ W_stack
        W_smooth_stack = W_smooth_flat.reshape(N, d, d)

        # Blend unsmoothed and smoothed
        self.W_smoothed = []
        for i in range(N):
            W_blend = (1 - self.kernel_blend) * W_unsmoothed[i] + self.kernel_blend * W_smooth_stack[i]
            self.W_smoothed.append(W_blend)

        return {
            "basis_losses": losses,
            "num_basis": len(self.basis_proj.basis),
            "kernel_condition": float(np.linalg.cond(self.K_matrix)),
        }

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """Project query using smoothed W matrices."""
        if not self.W_smoothed:
            raise ValueError("Model not trained")

        query_emb = query_emb.flatten()

        # Compute routing weights
        similarities = []
        for centroid in self.centroids:
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            similarities.append(np.dot(query_norm, centroid_norm))

        similarities = np.array(similarities)
        exp_sim = np.exp((similarities - np.max(similarities)) / temperature)
        weights = exp_sim / (np.sum(exp_sim) + 1e-8)

        # Weighted projection
        projected = np.zeros(len(query_emb))
        for i, (W, weight) in enumerate(zip(self.W_smoothed, weights)):
            projected += weight * (query_emb @ W)

        return projected


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


class AdaptiveConditioningProjection:
    """
    Adaptive hybrid projection using conditioning-based regularization.

    For each cluster, finds the minimum λ (Tikhonov regularization) needed
    to achieve a target condition number via bisection. This is principled:
    - Well-conditioned clusters get λ=0 (no regularization)
    - Ill-conditioned clusters get just enough λ to stabilize

    Key insight: For k basis vectors, Q^T Q is only k×k. Computing its
    condition number is O(k³) — essentially free for small k.

    This is NOT unified/coupled optimization. Each cluster is solved
    independently with per-cluster adaptive λ.

    See docs/proposals/COUPLED_OPTIMIZATION_IMPROVEMENTS.md, Proposal 7.
    """

    def __init__(
        self,
        num_basis: int = 4,
        target_cond: float = 50.0,
        bisection_tol: float = 1e-3,
        bisection_max_iter: int = 20,
        cosine_weight: float = 0.5,
    ):
        """
        Initialize adaptive conditioning projection.

        Args:
            num_basis: Number of shared basis matrices (K)
            target_cond: Target condition number for regularized problem
            bisection_tol: Tolerance for bisection search
            bisection_max_iter: Maximum bisection iterations
            cosine_weight: Weight for cosine loss (vs MSE) in basis training
        """
        self.num_basis = num_basis
        self.target_cond = target_cond
        self.bisection_tol = bisection_tol
        self.bisection_max_iter = bisection_max_iter
        self.cosine_weight = cosine_weight

        # Trained components
        self.basis: List[np.ndarray] = []
        self.alpha: np.ndarray = None  # Shape (N, K)
        self.centroids: List[np.ndarray] = []
        self.lambdas: np.ndarray = None  # Per-cluster λ values

        # Statistics
        self.cluster_conditions_before: List[float] = []
        self.cluster_conditions_after: List[float] = []

    def _effective_condition(self, QTQ: np.ndarray, lam: float) -> float:
        """
        Compute condition number of regularized normal equations.

        cond(Q^T Q + λI) for Tikhonov regularization.

        Args:
            QTQ: k×k Gram matrix (Q^T Q in projected space)
            lam: Regularization parameter

        Returns:
            Condition number
        """
        regularized = QTQ + lam * np.eye(QTQ.shape[0])
        s = np.linalg.svd(regularized, compute_uv=False)
        return s[0] / (s[-1] + 1e-10)

    def _find_min_lambda(self, QTQ: np.ndarray) -> Tuple[float, float, float]:
        """
        Find minimum λ that achieves target condition number via bisection.

        Args:
            QTQ: k×k Gram matrix

        Returns:
            (lambda, cond_before, cond_after)
        """
        cond_before = self._effective_condition(QTQ, 0.0)

        # Already well-conditioned?
        if cond_before <= self.target_cond:
            return 0.0, cond_before, cond_before

        # Find upper bound where condition is satisfied
        lam_high = 1.0
        while self._effective_condition(QTQ, lam_high) > self.target_cond:
            lam_high *= 2
            if lam_high > 1e6:
                # Extremely ill-conditioned
                cond_after = self._effective_condition(QTQ, lam_high)
                return lam_high, cond_before, cond_after

        # Bisect to find minimum sufficient λ
        lam_low = 0.0
        for _ in range(self.bisection_max_iter):
            lam_mid = (lam_low + lam_high) / 2
            if self._effective_condition(QTQ, lam_mid) > self.target_cond:
                lam_low = lam_mid
            else:
                lam_high = lam_mid

            if lam_high - lam_low < self.bisection_tol:
                break

        cond_after = self._effective_condition(QTQ, lam_high)
        return lam_high, cond_before, cond_after

    def train(
        self,
        clusters: List[Tuple[np.ndarray, np.ndarray]],
        num_iterations: int = 50,
        lr: float = 0.01,
        log_interval: int = 10,
    ) -> dict:
        """
        Train adaptive conditioning projection.

        Two-phase approach:
        1. Learn shared basis (same as SmoothingBasisProjection)
        2. Solve per-cluster with adaptive λ via bisection

        Args:
            clusters: List of (Q_i, A_i) tuples
            num_iterations: Alternating optimization iterations
            lr: Learning rate for basis updates
            log_interval: Logging interval

        Returns:
            Dict with training statistics
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
        K = min(self.num_basis, N)

        logger.info(f"Training AdaptiveConditioning: N={N}, K={K}, d={d}")
        logger.info(f"  Target condition: {self.target_cond}")

        # Store centroids
        self.centroids = [np.mean(Q, axis=0) for Q, A in processed]

        # Phase 1: Initialize basis from per-cluster solutions
        W_init = []
        for Q, A in processed:
            reg = 0.01 * np.eye(d)
            W = np.linalg.solve(Q.T @ Q + reg, Q.T @ A)
            W_init.append(W)

        self.basis = orthogonalize_basis(W_init[:K])
        K = len(self.basis)

        logger.info(f"Initialized {K} basis matrices")

        # Phase 2: Alternating optimization (basis learning)
        self.alpha = np.zeros((N, K))
        losses = []

        for iteration in range(num_iterations):
            total_loss = 0.0

            # Step 1: Fix basis, solve for coefficients
            for i, (Q, A) in enumerate(processed):
                self.alpha[i] = solve_for_alpha(Q, A, self.basis)

            # Step 2: Compute loss
            for i, (Q, A) in enumerate(processed):
                W_i = reconstruct_W(self.alpha[i], self.basis)
                pred = Q @ W_i

                mse = np.mean((pred - A) ** 2)
                pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
                A_norm = A / (np.linalg.norm(A) + 1e-8)
                cos_sim = np.mean(np.sum(pred_norm * A_norm, axis=1))

                loss_i = (1 - self.cosine_weight) * mse + self.cosine_weight * (1 - cos_sim)
                total_loss += loss_i

            losses.append(total_loss / N)

            # Step 3: Update basis via gradient descent
            total_grad = [np.zeros_like(G) for G in self.basis]

            for i, (Q, A) in enumerate(processed):
                W_i = reconstruct_W(self.alpha[i], self.basis)
                grad_W = compute_gradient(W_i, Q, A, self.cosine_weight)

                for k in range(K):
                    total_grad[k] += self.alpha[i, k] * grad_W

            for k in range(K):
                self.basis[k] = self.basis[k] - lr * total_grad[k]

            # Re-orthogonalize periodically
            if iteration % 5 == 0:
                self.basis = orthogonalize_basis(self.basis)
                K_new = len(self.basis)
                if K_new < K:
                    self.alpha = self.alpha[:, :K_new]
                    K = K_new

            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                logger.info(f"Iter {iteration}: loss={losses[-1]:.6f}")

        # Phase 3: Adaptive λ per cluster via bisection
        logger.info("Phase 3: Computing adaptive λ per cluster...")

        self.lambdas = np.zeros(N)
        self.cluster_conditions_before = []
        self.cluster_conditions_after = []

        for i, (Q, A) in enumerate(processed):
            # Project Q through basis: P = [Q @ G_1, ..., Q @ G_K] flattened appropriately
            # For regularization, we work in the coefficient space
            P = np.column_stack([(Q @ G).ravel() for G in self.basis])

            # Gram matrix in projected space: P^T P (K×K for k basis vectors)
            PTP = P.T @ P

            # Find optimal λ via bisection
            lam_i, cond_before, cond_after = self._find_min_lambda(PTP)

            self.lambdas[i] = lam_i
            self.cluster_conditions_before.append(cond_before)
            self.cluster_conditions_after.append(cond_after)

            # Re-solve with adaptive regularization
            target = A.ravel()
            reg_matrix = lam_i * np.eye(K)
            self.alpha[i] = np.linalg.solve(PTP + reg_matrix, P.T @ target)

        # Log statistics
        n_regularized = np.sum(self.lambdas > 0)
        mean_lambda = np.mean(self.lambdas[self.lambdas > 0]) if n_regularized > 0 else 0
        mean_cond_before = np.mean(self.cluster_conditions_before)
        mean_cond_after = np.mean(self.cluster_conditions_after)

        logger.info(f"  Clusters regularized: {n_regularized}/{N}")
        logger.info(f"  Mean λ (regularized): {mean_lambda:.4f}")
        logger.info(f"  Mean condition: {mean_cond_before:.1f} → {mean_cond_after:.1f}")

        return {
            "num_clusters": N,
            "num_basis": len(self.basis),
            "embedding_dim": d,
            "target_cond": self.target_cond,
            "clusters_regularized": int(n_regularized),
            "mean_lambda": float(mean_lambda),
            "max_lambda": float(np.max(self.lambdas)),
            "mean_cond_before": float(mean_cond_before),
            "mean_cond_after": float(mean_cond_after),
            "final_loss": float(losses[-1]) if losses else 0.0,
            "losses": losses,
        }

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query using soft routing and adaptively-regularized W matrices.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing

        Returns:
            Projected embedding (d,)
        """
        if not self.centroids or not self.basis:
            raise ValueError("Model not trained. Call train() first.")

        query_emb = query_emb.flatten()

        # Compute routing weights
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

        # Weighted combination
        projected = np.zeros(len(query_emb))
        for i, weight in enumerate(weights):
            W_i = reconstruct_W(self.alpha[i], self.basis)
            projected += weight * (query_emb @ W_i)

        return projected

    def get_stats(self) -> dict:
        """Get detailed statistics about the trained model."""
        if self.alpha is None:
            return {"trained": False}

        return {
            "trained": True,
            "num_basis": len(self.basis),
            "num_clusters": len(self.centroids),
            "target_cond": self.target_cond,
            "lambdas": self.lambdas.tolist(),
            "conditions_before": self.cluster_conditions_before,
            "conditions_after": self.cluster_conditions_after,
            "mean_lambda": float(np.mean(self.lambdas)),
            "clusters_needing_regularization": int(np.sum(self.lambdas > 0)),
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
