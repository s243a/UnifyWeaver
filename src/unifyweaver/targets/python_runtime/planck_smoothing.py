"""
Experimental: Planck-like spectral smoothing for projection matrices.

STATUS: EXPERIMENTAL - Based on intuition, needs empirical validation.

THEORY:
    Multiple perspectives suggest high-frequency suppression should help.
    These seem related, but THE EXACT FORMAL RELATIONSHIP IS NOT OBVIOUS.

    1. Nyquist (signal processing):
       - With n clusters, frequency bin k has O(n/k) effective samples
       - High-frequency bins are fundamentally under-sampled
       - Bin n/2 (Nyquist) has only ~2 samples → very unreliable

    2. Power/information concentration:
       - Finite sample (rect) → sinc in frequency domain
       - Sinc has most power at low frequencies
       - Power ≈ information (Shannon's channel capacity)
       - High-freq bins: low power AND under-sampled → doubly unreliable

    3. Statistical moments:
       - Higher moments require more data to estimate reliably
       - Mean (1st): n samples, Variance (2nd): n-1 DoF, etc.
       - High frequencies seem analogous to high moments

    4. Planck / ultraviolet catastrophe (physics):
       - Quantization prevents infinite energy at high frequencies
       - Finite data might play analogous role to quantization

    5. Inverse temperature / risk aversion (economics):
       - With uncertain estimates, be "risk averse" toward fine distinctions
       - α as inverse "data temperature"

    All of these point the same direction: suppress high frequencies when
    data is limited. But we don't have a unified formal theory connecting
    them. Perspectives 1-2 are signal processing (rigorous); 3-5 are
    suggestive analogies that may share deeper structure.

WHAT'S SOLID:
    - FFT smoothing empirically improves MRR (tested)
    - High-frequency removal helps (tested)
    - Less data → less reliable fine distinctions (standard statistics)

WHAT'S SPECULATIVE:
    - The exact Planck-like functional form
    - How to derive α from data properties
    - Whether soft decay beats hard cutoff

The Planck-like spectrum:
    S(ω) ∝ |ω| / (exp(α|ω|) - 1)

    - Low ω: S(ω) ≈ 1/α (approximately flat)
    - High ω: S(ω) ~ |ω| * exp(-α|ω|) (exponential decay)

Reference: context/Obsidian/A modernized Republic/A sequal/Coupled Kalman Filter/
           simulator/multi-scale implication/kernals/RKHS/01 kernel space in functional analysis.md
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.fft import fft, ifft
import logging

logger = logging.getLogger(__name__)


def planck_spectrum(omega: np.ndarray, alpha: float) -> np.ndarray:
    """
    Planck-like spectrum (linear in ω, not cubic like blackbody).

    S(ω) = |ω| / (exp(α|ω|) - 1)

    Args:
        omega: Frequency values (normalized to [0, 1] typically)
        alpha: Inverse "data temperature" - larger = more HF suppression

    Returns:
        Spectrum values
    """
    omega_abs = np.abs(omega) + 1e-10
    # Avoid overflow in exp
    exp_term = np.exp(np.minimum(alpha * omega_abs, 50))
    return omega_abs / (exp_term - 1 + 1e-10)


def planck_filter(omega: np.ndarray, alpha: float) -> np.ndarray:
    """
    Filter weights derived from Planck spectrum.

    We want to suppress frequencies where the "noise spectrum" dominates.
    Weight = S(ω) / S(0) normalized so DC component is preserved.

    Args:
        omega: Frequency values
        alpha: Inverse data temperature

    Returns:
        Filter weights in [0, 1]
    """
    S = planck_spectrum(omega, alpha)
    # Normalize so max (at low freq) is 1
    S_max = np.max(S)
    if S_max > 0:
        return S / S_max
    return np.ones_like(omega)


class PlanckSmoothingProjection:
    """
    EXPERIMENTAL: Soft spectral smoothing using Planck-like decay.

    Instead of hard frequency cutoff, applies exponential suppression
    of high frequencies based on a data-derived "temperature" parameter.

    The intuition: with limited data, high-frequency distinctions between
    clusters are unreliable. The amount of suppression should depend on
    how much data we have (more data → can trust finer distinctions).
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        alpha_scale: float = 1.0,
        blend_factor: float = 0.6,
        min_alpha: float = 0.5,
        max_alpha: float = 5.0,
    ):
        """
        Initialize Planck smoothing projection.

        Args:
            alpha: Inverse data temperature. If None, derived from data.
            alpha_scale: Scaling factor when deriving alpha from data
            blend_factor: Blend between original (0) and smoothed (1)
            min_alpha: Minimum alpha (even with lots of data)
            max_alpha: Maximum alpha (even with little data)
        """
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.blend_factor = blend_factor
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        # Trained state
        self.W_original: Dict[int, np.ndarray] = {}
        self.W_smoothed: Dict[int, np.ndarray] = {}
        self.centroids: List[np.ndarray] = []
        self.cluster_order: List[int] = []
        self.derived_alpha: float = 1.0
        self.samples_per_cluster: List[int] = []

    def _derive_alpha(self, clusters: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Derive α from data properties.

        EXPERIMENTAL: This is speculative. The intuition is:
        - More samples → lower estimation variance → can trust finer distinctions
        - α ∝ 1/√(mean_samples) like standard error scaling

        Args:
            clusters: List of (Q, A) tuples

        Returns:
            Derived α value
        """
        samples = [Q.shape[0] for Q, A in clusters]
        self.samples_per_cluster = samples
        mean_samples = np.mean(samples)

        # α ∝ 1/√n - more samples means lower α, more HF allowed
        # This is speculative but matches intuition about estimation variance
        alpha = self.alpha_scale / np.sqrt(max(1, mean_samples))

        # Clamp to reasonable range
        alpha = np.clip(alpha, self.min_alpha, self.max_alpha)

        logger.info(f"Derived α={alpha:.3f} from mean_samples={mean_samples:.1f}")
        return alpha

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
        """
        Train Planck smoothing projection.

        Args:
            clusters: List of (Q_i, A_i) tuples

        Returns:
            Training statistics
        """
        N = len(clusters)
        d = clusters[0][0].shape[1]

        # Derive or use provided α
        if self.alpha is None:
            self.derived_alpha = self._derive_alpha(clusters)
        else:
            self.derived_alpha = self.alpha

        logger.info(f"Training PlanckSmoothing: N={N}, d={d}, α={self.derived_alpha:.3f}")

        # Compute per-cluster W matrices (least squares)
        for i, (Q, A) in enumerate(clusters):
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)

            # Store centroid
            self.centroids.append(np.mean(Q, axis=0))

            # Solve for W
            reg = 0.01 * np.eye(d)
            W = np.linalg.solve(Q.T @ Q + reg, Q.T @ A)
            self.W_original[i] = W

        # Order clusters by centroid similarity for FFT
        self._compute_cluster_order()

        # Apply Planck smoothing in frequency domain
        self._apply_planck_smoothing()

        return {
            "num_clusters": N,
            "embedding_dim": d,
            "alpha": float(self.derived_alpha),
            "blend_factor": self.blend_factor,
            "mean_samples_per_cluster": float(np.mean(self.samples_per_cluster)) if self.samples_per_cluster else 0,
        }

    def _compute_cluster_order(self):
        """Order clusters by centroid similarity (greedy nearest neighbor)."""
        N = len(self.centroids)
        if N == 0:
            return

        # Normalize centroids
        norms = [np.linalg.norm(c) + 1e-8 for c in self.centroids]
        normalized = [c / n for c, n in zip(self.centroids, norms)]

        # Greedy ordering starting from cluster 0
        remaining = set(range(N))
        order = [0]
        remaining.remove(0)

        while remaining:
            last = order[-1]
            best_sim = -2
            best_next = None

            for candidate in remaining:
                sim = np.dot(normalized[last], normalized[candidate])
                if sim > best_sim:
                    best_sim = sim
                    best_next = candidate

            order.append(best_next)
            remaining.remove(best_next)

        self.cluster_order = order

    def _apply_planck_smoothing(self):
        """Apply Planck-like spectral smoothing to W matrices."""
        N = len(self.W_original)
        if N == 0:
            return

        # Stack W matrices in similarity order
        W_stacked = np.stack([self.W_original[i] for i in self.cluster_order])
        original_shape = W_stacked.shape  # (N, d, d) or (N, d, d_out)

        # Flatten spatial dimensions for FFT along cluster axis
        W_flat = W_stacked.reshape(N, -1)

        # FFT along cluster dimension
        W_freq = fft(W_flat, axis=0)

        # Create frequency axis (normalized to [0, 1])
        freqs = np.fft.fftfreq(N)

        # Apply Planck filter
        filter_weights = planck_filter(freqs, self.derived_alpha)

        # Apply filter (broadcast across spatial dimensions)
        W_freq_filtered = W_freq * filter_weights[:, np.newaxis]

        # Inverse FFT
        W_smoothed_flat = np.real(ifft(W_freq_filtered, axis=0))
        W_smoothed_stacked = W_smoothed_flat.reshape(original_shape)

        # Store smoothed W matrices (mapped back to original indices)
        for i, orig_idx in enumerate(self.cluster_order):
            W_orig = self.W_original[orig_idx]
            W_smooth = W_smoothed_stacked[i]

            # Blend original and smoothed
            self.W_smoothed[orig_idx] = (
                (1 - self.blend_factor) * W_orig +
                self.blend_factor * W_smooth
            )

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query embedding using soft routing.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing

        Returns:
            Projected embedding (d,)
        """
        if not self.centroids or not self.W_smoothed:
            raise ValueError("Model not trained. Call train() first.")

        query_emb = query_emb.flatten()

        # Compute routing weights via softmax over centroid similarities
        similarities = []
        for centroid in self.centroids:
            c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            similarities.append(np.dot(q_norm, c_norm))

        similarities = np.array(similarities)
        exp_sim = np.exp((similarities - np.max(similarities)) / temperature)
        weights = exp_sim / (np.sum(exp_sim) + 1e-8)

        # Weighted combination of projections
        projected = np.zeros(len(query_emb))
        for i, weight in enumerate(weights):
            W_i = self.W_smoothed[i]
            projected += weight * (query_emb @ W_i)

        return projected

    def get_diagnostics(self) -> dict:
        """Get diagnostics about the smoothing applied."""
        if not self.W_original:
            return {"trained": False}

        N = len(self.W_original)

        # Compute how much smoothing changed each W
        changes = []
        for i in range(N):
            diff = np.linalg.norm(self.W_smoothed[i] - self.W_original[i])
            orig_norm = np.linalg.norm(self.W_original[i])
            changes.append(diff / (orig_norm + 1e-8))

        return {
            "trained": True,
            "num_clusters": N,
            "alpha": self.derived_alpha,
            "blend_factor": self.blend_factor,
            "mean_relative_change": float(np.mean(changes)),
            "max_relative_change": float(np.max(changes)),
            "samples_per_cluster": self.samples_per_cluster,
        }
