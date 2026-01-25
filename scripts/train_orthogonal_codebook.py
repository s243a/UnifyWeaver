#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Fast Orthogonal Bivector Codebook for rotation-based projection.

This implements the orthogonal extension from PR_principal_bivector_codebook.md Appendix B:

Key insight: If rotation planes are orthogonal, rotations commute and we can use
Rodrigues formula instead of matrix exponential - O(K×d) vs O(d³).

For d=768, K=64: Potentially 9000× faster than matrix exponential!

Architecture:
1. Build/orthogonalize codebook of bivectors with non-overlapping rotation planes
2. Transformer predicts routing weights via cosine similarity
3. Apply weighted rotations using Rodrigues formula (very fast)

Usage:
    # Build orthogonal codebook from existing PCA codebook
    python scripts/train_orthogonal_codebook.py \
        --codebook models/bivector_codebook.npz \
        --orthogonalize \
        --output models/orthogonal_codebook.npz

    # Train with orthogonal codebook
    python scripts/train_orthogonal_codebook.py \
        --federated-model models/federated.pkl \
        --codebook models/orthogonal_codebook.npz \
        --epochs 50
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle

import numpy as np
from scipy.linalg import schur, expm

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def log_info(msg: str):
    """Log info message and flush stdout to prevent buffer stalls."""
    logger.info(msg)
    sys.stdout.flush()


# =============================================================================
# Schur Decomposition and Plane Extraction
# =============================================================================

def extract_rotation_planes(B: np.ndarray, tol: float = 1e-10) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Extract rotation planes and angles from a bivector (antisymmetric matrix).

    Uses real Schur decomposition to find the block-diagonal form where each
    2x2 block represents a rotation in a specific plane.

    Args:
        B: (d, d) antisymmetric matrix (bivector)
        tol: tolerance for zero angles

    Returns:
        planes: List of (u, v, theta) tuples defining each rotation plane
                u, v are orthonormal vectors spanning the plane
                theta is the rotation angle in that plane
    """
    d = B.shape[0]

    # Real Schur decomposition: B = Q @ T @ Q.T
    # For antisymmetric B, T is block diagonal with 2x2 antisymmetric blocks
    T, Q = schur(B, output='real')

    planes = []
    i = 0
    while i < d:
        if i + 1 < d and abs(T[i, i+1]) > tol:
            # 2x2 block found: rotation plane
            # For antisymmetric 2x2: [[0, -θ], [θ, 0]]
            theta = T[i+1, i]  # Rotation angle
            u = Q[:, i].copy()       # First basis vector of plane
            v = Q[:, i+1].copy()     # Second basis vector of plane
            planes.append((u, v, theta))
            i += 2
        else:
            # Zero block (no rotation in this direction)
            i += 1

    return planes


def get_dominant_plane(B: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Get the dominant (largest angle) rotation plane of a bivector.

    Args:
        B: (d, d) antisymmetric matrix

    Returns:
        (u, v, theta) tuple for dominant plane, or None if no rotation
    """
    planes = extract_rotation_planes(B)
    if not planes:
        return None
    # Sort by absolute angle, return largest
    planes.sort(key=lambda p: abs(p[2]), reverse=True)
    return planes[0]


def bivector_from_plane(u: np.ndarray, v: np.ndarray, theta: float = 1.0) -> np.ndarray:
    """
    Construct a bivector (antisymmetric matrix) from a rotation plane.

    Args:
        u, v: Orthonormal vectors spanning the rotation plane
        theta: Rotation angle

    Returns:
        B: (d, d) antisymmetric matrix where exp(B) rotates by theta in (u,v) plane
    """
    return theta * (np.outer(u, v) - np.outer(v, u))


# =============================================================================
# Codebook Orthogonalization
# =============================================================================

def orthogonalize_codebook(
    codebook_bivectors: np.ndarray,
    method: str = "dominant"
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    Orthogonalize codebook bivectors so their rotations commute.

    Args:
        codebook_bivectors: (n_components, d, d) array of antisymmetric matrices
        method: "dominant" uses dominant plane from each bivector
                "all" tries to preserve all rotation planes (not implemented)

    Returns:
        orthogonal_bivectors: (n_components, d, d) with orthogonal rotation planes
        planes: List of (u, v, theta) tuples for each orthogonal plane
    """
    n_components, d, _ = codebook_bivectors.shape
    log_info(f"Orthogonalizing {n_components} bivectors in {d}D...")

    # Extract dominant plane from each bivector
    dominant_planes = []
    for i, B in enumerate(codebook_bivectors):
        plane = get_dominant_plane(B)
        if plane:
            dominant_planes.append(plane)
        else:
            log_info(f"  Warning: bivector {i} has no dominant plane")
            # Create a placeholder
            u = np.zeros(d)
            v = np.zeros(d)
            u[2*i % d] = 1.0
            v[(2*i + 1) % d] = 1.0
            dominant_planes.append((u, v, 0.0))

    log_info(f"  Extracted {len(dominant_planes)} dominant planes")

    # Orthogonalize planes using Gram-Schmidt on the subspaces
    orthogonal_planes = []
    used_basis = []  # Track basis vectors we've used

    for idx, (u, v, theta) in enumerate(dominant_planes):
        # Project out components from already-used basis vectors
        u_orth = u.copy()
        v_orth = v.copy()

        for basis_vec in used_basis:
            u_orth = u_orth - np.dot(u_orth, basis_vec) * basis_vec
            v_orth = v_orth - np.dot(v_orth, basis_vec) * basis_vec

        # Check if there's enough remaining
        u_norm = np.linalg.norm(u_orth)
        v_norm = np.linalg.norm(v_orth)

        if u_norm < 1e-8 or v_norm < 1e-8:
            # Not enough orthogonal space left, use arbitrary unused dimensions
            for dim in range(d):
                test_vec = np.zeros(d)
                test_vec[dim] = 1.0
                is_used = any(abs(np.dot(test_vec, b)) > 0.9 for b in used_basis)
                if not is_used:
                    if u_norm < 1e-8:
                        u_orth = test_vec
                        u_norm = 1.0
                    elif v_norm < 1e-8:
                        v_orth = test_vec
                        v_norm = 1.0
                    if u_norm >= 1e-8 and v_norm >= 1e-8:
                        break

        # Normalize u
        if u_norm > 1e-10:
            u_orth = u_orth / u_norm
        else:
            u_orth = np.zeros(d)
            u_orth[2*idx % d] = 1.0

        # Make v orthogonal to u, then normalize
        v_orth = v_orth - np.dot(v_orth, u_orth) * u_orth
        v_norm = np.linalg.norm(v_orth)
        if v_norm > 1e-10:
            v_orth = v_orth / v_norm
        else:
            v_orth = np.zeros(d)
            v_orth[(2*idx + 1) % d] = 1.0
            v_orth = v_orth - np.dot(v_orth, u_orth) * u_orth
            v_orth = v_orth / (np.linalg.norm(v_orth) + 1e-10)

        # Add to used basis
        used_basis.append(u_orth)
        used_basis.append(v_orth)

        orthogonal_planes.append((u_orth, v_orth, theta))

        if (idx + 1) % 20 == 0:
            log_info(f"  Orthogonalized {idx + 1}/{n_components} planes")

    # Reconstruct orthogonal bivectors
    orthogonal_bivectors = np.zeros_like(codebook_bivectors)
    for i, (u, v, theta) in enumerate(orthogonal_planes):
        orthogonal_bivectors[i] = bivector_from_plane(u, v, theta)

    log_info(f"  Done. Created {len(orthogonal_planes)} orthogonal bivectors")

    return orthogonal_bivectors, orthogonal_planes


def build_canonical_orthogonal_codebook(d: int, n_components: int) -> Tuple[np.ndarray, List]:
    """
    Build a codebook of n_components orthogonal rotation planes using canonical basis.

    This uses dimensions (2i, 2i+1) for the i-th rotation plane, guaranteeing
    perfect orthogonality but not data-driven.

    Args:
        d: Embedding dimension (e.g., 768)
        n_components: Number of codebook entries (must be <= d/2)

    Returns:
        codebook: (n_components, d, d) orthogonal bivectors
        planes: List of (u, v, theta) tuples
    """
    assert n_components <= d // 2, f"Can have at most {d//2} orthogonal planes in {d}D"

    log_info(f"Building canonical orthogonal codebook: {n_components} planes in {d}D")

    codebook = np.zeros((n_components, d, d))
    planes = []

    for i in range(n_components):
        j, k = 2 * i, 2 * i + 1
        u = np.zeros(d)
        v = np.zeros(d)
        u[j] = 1.0
        v[k] = 1.0

        codebook[i, j, k] = -1.0  # Antisymmetric
        codebook[i, k, j] = 1.0
        planes.append((u, v, 1.0))

    return codebook, planes


# =============================================================================
# Fast Orthogonal Codebook Transformer
# =============================================================================

class FastOrthogonalCodebook:
    """
    Codebook with orthogonal rotation planes for fast composition.

    Since all bivectors rotate in orthogonal planes, their rotations commute:
        exp(Σ wᵢBᵢ) = Π exp(wᵢBᵢ)

    We use Rodrigues formula for O(K×d) computation instead of O(d³) matrix exp.
    """

    def __init__(self, codebook_bivectors: np.ndarray, codebook_keys: Optional[np.ndarray] = None):
        """
        Args:
            codebook_bivectors: (n_components, d, d) orthogonal bivectors
            codebook_keys: (n_components, d) routing keys for cosine similarity
        """
        self.bivectors = codebook_bivectors
        self.n_components, self.d, _ = codebook_bivectors.shape

        # Extract plane info for each bivector
        log_info(f"Extracting rotation planes from {self.n_components} bivectors...")
        self.planes = []  # List of (u, v, theta) tuples
        for B in codebook_bivectors:
            plane = get_dominant_plane(B)
            if plane is None:
                plane = (np.zeros(self.d), np.zeros(self.d), 0.0)
            self.planes.append(plane)

        # Store as arrays for vectorized computation
        self.plane_u = np.array([p[0] for p in self.planes])  # (n_components, d)
        self.plane_v = np.array([p[1] for p in self.planes])  # (n_components, d)
        self.plane_theta = np.array([p[2] for p in self.planes])  # (n_components,)

        # Codebook keys for routing
        if codebook_keys is not None:
            self.codebook_keys = codebook_keys
        else:
            # Use plane_u as default keys
            self.codebook_keys = self.plane_u.copy()
            norms = np.linalg.norm(self.codebook_keys, axis=1, keepdims=True)
            self.codebook_keys = self.codebook_keys / (norms + 1e-8)

        log_info(f"FastOrthogonalCodebook ready: {self.n_components} orthogonal planes")

    def apply_weighted_rotation(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Apply weighted rotation composition using Rodrigues formula.

        For orthogonal planes, we can apply each rotation independently.
        Rodrigues formula for simple bivector: exp(θB) acts only on the 2D plane.

        Args:
            x: (batch, d) input vectors
            weights: (batch, n_components) blending weights

        Returns:
            y: (batch, d) rotated vectors
        """
        batch_size = x.shape[0]
        y = x.copy()

        for i in range(self.n_components):
            u = self.plane_u[i]
            v = self.plane_v[i]
            theta = self.plane_theta[i]

            if abs(theta) < 1e-10:
                continue

            w = weights[:, i]  # (batch,)

            # Weighted angle
            w_theta = w * theta  # (batch,)

            # Rodrigues: rotate within the (u, v) plane
            sin_wt = np.sin(w_theta)[:, None]  # (batch, 1)
            cos_wt = np.cos(w_theta)[:, None]  # (batch, 1)

            # Project y onto the rotation plane
            y_u = (y @ u)[:, None]  # (batch, 1) - component along u
            y_v = (y @ v)[:, None]  # (batch, 1) - component along v

            # Rotation within the plane:
            # y_u' = cos(wθ) * y_u - sin(wθ) * y_v
            # y_v' = sin(wθ) * y_u + cos(wθ) * y_v
            new_y_u = cos_wt * y_u - sin_wt * y_v
            new_y_v = sin_wt * y_u + cos_wt * y_v

            # Update y: remove old components, add new
            y = y - y_u * u - y_v * v + new_y_u * u + new_y_v * v

        return y

    def apply_weighted_rotation_vectorized(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Vectorized version of apply_weighted_rotation.

        Processes all planes simultaneously for better performance.

        Args:
            x: (batch, d) input vectors
            weights: (batch, n_components) blending weights

        Returns:
            y: (batch, d) rotated vectors
        """
        batch_size = x.shape[0]

        # Compute weighted angles for all planes: (batch, n_components)
        w_theta = weights * self.plane_theta[None, :]

        # Compute sin and cos: (batch, n_components)
        sin_wt = np.sin(w_theta)
        cos_wt = np.cos(w_theta)

        # Project x onto all planes at once
        # x @ plane_u.T gives (batch, n_components) - components along each u
        x_u = x @ self.plane_u.T  # (batch, n_components)
        x_v = x @ self.plane_v.T  # (batch, n_components)

        # Rotated components
        new_x_u = cos_wt * x_u - sin_wt * x_v  # (batch, n_components)
        new_x_v = sin_wt * x_u + cos_wt * x_v  # (batch, n_components)

        # Compute delta for each plane and sum
        # delta = (new_x_u - x_u) * u + (new_x_v - x_v) * v
        delta_u = new_x_u - x_u  # (batch, n_components)
        delta_v = new_x_v - x_v  # (batch, n_components)

        # (batch, n_components) @ (n_components, d) = (batch, d)
        delta = delta_u @ self.plane_u + delta_v @ self.plane_v

        return x + delta

    def route_and_apply(
        self,
        x: np.ndarray,
        top_k: int = 8,
        scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Route query to top-K codebook entries and apply rotation.

        Args:
            x: (batch, d) input vectors
            top_k: Number of codebook entries to blend
            scale: Output scaling factor

        Returns:
            y: (batch, d) rotated and scaled vectors
            weights: (batch, top_k) routing weights
            indices: (batch, top_k) selected codebook indices
        """
        batch_size = x.shape[0]

        # Normalize for cosine similarity
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity to codebook keys
        similarities = x_norm @ self.codebook_keys.T  # (batch, n_components)

        # Top-K selection
        indices = np.argsort(similarities, axis=1)[:, -top_k:]  # (batch, top_k)
        top_k_sims = np.take_along_axis(similarities, indices, axis=1)  # (batch, top_k)

        # Normalize weights (positive only)
        top_k_sims_pos = np.maximum(top_k_sims, 0.0)
        weight_sum = top_k_sims_pos.sum(axis=1, keepdims=True) + 1e-8
        weights = top_k_sims_pos / weight_sum  # (batch, top_k)

        # Expand to full weight vector
        full_weights = np.zeros((batch_size, self.n_components))
        for b in range(batch_size):
            full_weights[b, indices[b]] = weights[b]

        # Apply rotation
        y = self.apply_weighted_rotation_vectorized(x, full_weights)

        # Scale
        y = y * scale

        return y, weights, indices


# =============================================================================
# PyTorch Implementation for Training
# =============================================================================

def _import_torch():
    """Lazy import torch."""
    global torch, nn, F
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        return _torch, _nn, _F
    except ImportError:
        return None, None, None


class FastOrthogonalTransformer:
    """
    Transformer that routes to orthogonal codebook using Rodrigues formula.

    Much faster than BivectorCodebookTransformer because it avoids matrix exponential.
    """

    def __init__(
        self,
        codebook: FastOrthogonalCodebook,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 256,
        top_k: int = 8,
        dropout: float = 0.1,
        device: str = "auto",
    ):
        torch, nn, F = _import_torch()
        if torch is None:
            raise ImportError("PyTorch required for FastOrthogonalTransformer")

        self.codebook = codebook
        self.embed_dim = codebook.d
        self.n_components = codebook.n_components
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.top_k = min(top_k, self.n_components)

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Convert codebook to tensors
        self.plane_u = torch.from_numpy(codebook.plane_u).float().to(self.device)
        self.plane_v = torch.from_numpy(codebook.plane_v).float().to(self.device)
        self.plane_theta = torch.from_numpy(codebook.plane_theta).float().to(self.device)
        self.codebook_keys = torch.from_numpy(codebook.codebook_keys).float().to(self.device)

        # Build model
        self._build_model(nn, ff_dim, dropout)

        log_info(f"FastOrthogonalTransformer: {self.n_components} planes, "
                 f"top_k={self.top_k}, layers={num_layers}, device={self.device}")

    def _build_model(self, nn, ff_dim, dropout):
        """Build the transformer model."""
        # Input projection
        self.input_proj = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        ).to(self.device)

        # Routing head
        self.routing_head = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        # Scale head
        self.scale_head = nn.Sequential(
            nn.Linear(self.embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, 1)
        ).to(self.device)

    def forward(self, query):
        """
        Forward pass with Rodrigues rotation.

        Args:
            query: (batch, embed_dim) input queries

        Returns:
            projected: (batch, embed_dim) rotated outputs
            weights: (batch, top_k) routing weights
            top_k_idx: (batch, top_k) selected indices
            scale: (batch,) scale factors
        """
        torch, nn, F = _import_torch()
        batch_size = query.shape[0]

        # Encode
        x = self.input_proj(query).unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze(1)

        # Routing via cosine similarity
        routing_vec = self.routing_head(x)
        routing_vec = F.normalize(routing_vec, dim=-1)

        similarities = torch.mm(routing_vec, self.codebook_keys.T)

        # Top-K selection
        top_k_sim, top_k_idx = similarities.topk(self.top_k, dim=-1)

        # Normalize weights
        top_k_sim_pos = torch.clamp(top_k_sim, min=0.0)
        weight_sum = top_k_sim_pos.sum(dim=-1, keepdim=True) + 1e-8
        weights = top_k_sim_pos / weight_sum

        # Build full weight vector for Rodrigues
        full_weights = torch.zeros(batch_size, self.n_components, device=self.device)
        full_weights.scatter_(1, top_k_idx, weights)

        # Rodrigues rotation (vectorized)
        w_theta = full_weights * self.plane_theta.unsqueeze(0)
        sin_wt = torch.sin(w_theta)
        cos_wt = torch.cos(w_theta)

        # Project onto planes
        q_u = torch.mm(query, self.plane_u.T)  # (batch, n_components)
        q_v = torch.mm(query, self.plane_v.T)

        # Rotated components
        new_q_u = cos_wt * q_u - sin_wt * q_v
        new_q_v = sin_wt * q_u + cos_wt * q_v

        # Delta
        delta_u = new_q_u - q_u
        delta_v = new_q_v - q_v

        delta = torch.mm(delta_u, self.plane_u) + torch.mm(delta_v, self.plane_v)

        projected = query + delta

        # Scale
        scale = self.scale_head(x).squeeze(-1)
        scale = F.softplus(scale) + 0.5
        projected = projected * scale.unsqueeze(-1)

        return projected, weights, top_k_idx, scale

    def parameters(self):
        """Return trainable parameters."""
        params = list(self.input_proj.parameters())
        params += list(self.encoder.parameters())
        params += list(self.routing_head.parameters())
        params += list(self.scale_head.parameters())
        return params

    def train_mode(self):
        self.input_proj.train()
        self.encoder.train()
        self.routing_head.train()
        self.scale_head.train()

    def eval_mode(self):
        self.input_proj.eval()
        self.encoder.eval()
        self.routing_head.eval()
        self.scale_head.eval()


# =============================================================================
# Teacher for Training
# =============================================================================

class OrthogonalTeacher:
    """
    Teacher that projects using orthogonal codebook with Rodrigues formula.

    Much faster than BivectorTeacher because it avoids matrix exponential.
    """

    def __init__(
        self,
        federated_model_path: str,
        codebook: FastOrthogonalCodebook,
    ):
        """
        Args:
            federated_model_path: Path to federated .pkl model
            codebook: FastOrthogonalCodebook for rotation
        """
        self.codebook = codebook
        self.d = codebook.d
        self.n_components = codebook.n_components

        # Load federated model for routing
        self._load_federated_model(federated_model_path)

        log_info(
            f"OrthogonalTeacher: {self.num_clusters} clusters, "
            f"routing to {self.n_components} orthogonal planes"
        )

    def _load_federated_model(self, model_path: str):
        """Load federated model for routing."""
        model_path = Path(model_path)
        with open(model_path, 'rb') as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.temperature = self.meta.get("temperature", 0.1)
        self.num_clusters = len(self.cluster_ids)
        self.cluster_centroids = self.meta.get("cluster_centroids")

        # Load routing data
        cluster_dir = model_path.with_suffix('')
        routing_path = cluster_dir / "routing_data.npz"
        if routing_path.exists():
            routing_data = np.load(routing_path)
            self.query_embeddings = routing_data['query_embeddings']
        else:
            raise FileNotFoundError(f"Routing data not found: {routing_path}")

    def get_target_output(
        self,
        query_emb: np.ndarray,
        top_k: int = 8,
    ) -> np.ndarray:
        """
        Get target output using orthogonal codebook routing.

        Args:
            query_emb: Query embedding (d,) or (batch, d)

        Returns:
            output: Rotated output (d,) or (batch, d)
        """
        was_1d = query_emb.ndim == 1
        if was_1d:
            query_emb = query_emb[np.newaxis, :]

        output, _, _ = self.codebook.route_and_apply(query_emb, top_k=top_k)

        if was_1d:
            output = output[0]

        return output

    def compute_targets_batched(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 8,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Compute target outputs using Rodrigues formula (very fast).

        Args:
            query_embeddings: (N, d) query embeddings
            top_k: Number of codebook entries to blend
            batch_size: Batch size for processing

        Returns:
            outputs: (N, d) target outputs after rotation
        """
        n_samples = len(query_embeddings)
        all_outputs = np.zeros_like(query_embeddings)

        start_time = time.time()
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_queries = query_embeddings[batch_start:batch_end]

            batch_outputs, _, _ = self.codebook.route_and_apply(
                batch_queries, top_k=top_k
            )
            all_outputs[batch_start:batch_end] = batch_outputs

            if (batch_end) % 1000 == 0 or batch_end == n_samples:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed
                log_info(f"    Targets: {batch_end}/{n_samples} ({rate:.0f}/s)")

        elapsed = time.time() - start_time
        log_info(f"  Target computation: {elapsed:.2f}s ({n_samples/elapsed:.0f}/s)")

        return all_outputs


# =============================================================================
# Training Functions
# =============================================================================

def train_orthogonal_transformer(
    transformer: FastOrthogonalTransformer,
    teacher: OrthogonalTeacher,
    query_embeddings: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 10,
    top_k: int = 8,
) -> List[float]:
    """
    Train FastOrthogonalTransformer.

    Args:
        transformer: FastOrthogonalTransformer to train
        teacher: OrthogonalTeacher for target outputs
        query_embeddings: Training queries (N, d)
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_interval: Log every N epochs
        top_k: Top-K for teacher routing

    Returns:
        List of loss values per epoch
    """
    torch, nn, F = _import_torch()

    n_samples = len(query_embeddings)

    # Pre-compute target outputs (very fast with Rodrigues)
    log_info("Computing target outputs from teacher (Rodrigues)...")
    target_outputs = teacher.compute_targets_batched(
        query_embeddings, top_k=top_k, batch_size=256
    )

    # Convert to tensors
    queries_tensor = torch.from_numpy(query_embeddings).float().to(transformer.device)
    outputs_tensor = torch.from_numpy(target_outputs).float().to(transformer.device)

    # Optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    # Training loop
    transformer.train_mode()
    losses = []

    log_info(f"Training for {num_epochs} epochs, {n_samples} samples")

    training_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_cosine = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_queries = queries_tensor[idx]
            batch_targets = outputs_tensor[idx]

            # Forward pass
            predicted, weights, top_k_idx, scale = transformer.forward(batch_queries)

            # Loss: MSE + cosine
            mse_loss = F.mse_loss(predicted, batch_targets)

            pred_norm = F.normalize(predicted, dim=1)
            target_norm = F.normalize(batch_targets, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()

            loss = 0.5 * mse_loss + 0.5 * (1 - cosine_sim)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cosine += cosine_sim.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_cosine = epoch_cosine / n_batches
        losses.append(avg_loss)

        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - training_start
        avg_epoch_time = total_elapsed / (epoch + 1)
        remaining = avg_epoch_time * (num_epochs - epoch - 1)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            log_info(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_elapsed:.1f}s), "
                f"Loss: {avg_loss:.6f}, Cos: {avg_cosine:.4f} "
                f"[ETA: {remaining/60:.1f}m]"
            )

    total_time = time.time() - training_start
    transformer.eval_mode()
    log_info(f"Training complete: {total_time/60:.1f}m. Final loss: {losses[-1]:.6f}")

    return losses


def evaluate_orthogonal_transformer(
    transformer: FastOrthogonalTransformer,
    teacher: OrthogonalTeacher,
    test_queries: np.ndarray,
    batch_size: int = 128,
    top_k: int = 8,
) -> Dict:
    """Evaluate transformer against teacher on test set."""
    torch, nn, F = _import_torch()
    transformer.eval_mode()

    n_samples = len(test_queries)

    # Compute teacher outputs
    log_info(f"Computing teacher outputs for {n_samples} test samples...")
    teacher_outputs = teacher.compute_targets_batched(
        test_queries, top_k=top_k, batch_size=256
    )

    # Compute transformer outputs
    log_info("Computing transformer outputs...")
    trans_outputs = []

    queries_tensor = torch.from_numpy(test_queries).float().to(transformer.device)

    with torch.no_grad():
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_queries = queries_tensor[batch_start:batch_end]

            batch_out, _, _, _ = transformer.forward(batch_queries)
            trans_outputs.append(batch_out.cpu().numpy())

    trans_outputs = np.concatenate(trans_outputs, axis=0)

    # Compute metrics
    mse_values = np.mean((teacher_outputs - trans_outputs) ** 2, axis=1)

    # Cosine similarity
    teacher_norms = np.linalg.norm(teacher_outputs, axis=1, keepdims=True) + 1e-8
    trans_norms = np.linalg.norm(trans_outputs, axis=1, keepdims=True) + 1e-8
    teacher_normed = teacher_outputs / teacher_norms
    trans_normed = trans_outputs / trans_norms
    cosine_sims = np.sum(teacher_normed * trans_normed, axis=1)

    return {
        'mean_mse': np.mean(mse_values),
        'std_mse': np.std(mse_values),
        'mean_cosine_sim': np.mean(cosine_sims),
        'std_cosine_sim': np.std(cosine_sims),
        'min_cosine_sim': np.min(cosine_sims),
        'max_cosine_sim': np.max(cosine_sims),
        'n_samples': n_samples,
    }


# =============================================================================
# Full Rotation Teacher (for validation)
# =============================================================================

class FullRotationTeacher:
    """
    Teacher using full rotational manifold from federated model.

    Blends cluster bivectors and applies via matrix exponential.
    This is the "ground truth" for comparison with compressed codebook.
    """

    def __init__(self, federated_model_path: str):
        """Load federated model and compute cluster bivectors."""
        from scipy.linalg import logm

        model_path = Path(federated_model_path)
        with open(model_path, 'rb') as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.temperature = self.meta.get("temperature", 0.1)
        self.num_clusters = len(self.cluster_ids)
        self.cluster_centroids = self.meta.get("cluster_centroids")

        # Load routing data
        cluster_dir = model_path.with_suffix('')
        routing_path = cluster_dir / "routing_data.npz"
        if routing_path.exists():
            routing_data = np.load(routing_path)
            self.query_embeddings = routing_data['query_embeddings']
        else:
            raise FileNotFoundError(f"Routing data not found: {routing_path}")

        # Load cluster W matrices and compute bivectors
        self.cluster_bivectors = {}
        self.d = None

        for cid in self.cluster_ids:
            if cid.startswith("cluster_"):
                cluster_path = cluster_dir / f"{cid}.npz"
            else:
                cluster_path = cluster_dir / f"cluster_{cid}.npz"

            if cluster_path.exists():
                data = np.load(cluster_path)
                if "W" in data:
                    W = data["W"]
                elif "W_stack" in data:
                    W = data["W_stack"][0]
                else:
                    continue

                if self.d is None:
                    self.d = W.shape[0]

                # Compute bivector (log of rotation)
                try:
                    A = logm(W)
                    A = np.real(A)
                    A = (A - A.T) / 2  # Ensure antisymmetric
                    self.cluster_bivectors[cid] = A
                except Exception as e:
                    log_info(f"  Warning: logm failed for {cid}: {e}")

        log_info(f"FullRotationTeacher: {len(self.cluster_bivectors)} cluster bivectors, d={self.d}")

    def project(self, query_emb: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        Project query using full rotational blending.

        Blends top-K cluster bivectors and applies via matrix exponential.
        """
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Compute routing weights
        if self.cluster_centroids is not None:
            sims = q_norm @ self.cluster_centroids.T
        else:
            sims = q_norm @ self.query_embeddings.T

        sims_shifted = (sims - np.max(sims)) / self.temperature
        weights = np.exp(sims_shifted)
        weights /= weights.sum()

        top_indices = np.argsort(weights)[-top_k:]

        # Blend bivectors
        blended_bivector = np.zeros((self.d, self.d), dtype=np.float64)
        total_weight = 0.0

        for idx in top_indices:
            cid = self.cluster_ids[idx]
            if cid in self.cluster_bivectors:
                w = weights[idx]
                blended_bivector += w * self.cluster_bivectors[cid]
                total_weight += w

        if total_weight > 0:
            blended_bivector /= total_weight

        # Apply rotation via matrix exponential
        R = expm(blended_bivector)
        return (R @ query_emb).astype(np.float32)

    def compute_outputs_batched(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> np.ndarray:
        """Compute outputs for multiple queries (slow - uses matrix exp)."""
        n_samples = len(query_embeddings)
        outputs = np.zeros_like(query_embeddings)

        start_time = time.time()
        for i, query in enumerate(query_embeddings):
            outputs[i] = self.project(query, top_k)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_samples - i - 1) / rate
                log_info(f"    Full rotation: {i+1}/{n_samples} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

        elapsed = time.time() - start_time
        log_info(f"  Full rotation complete: {elapsed:.1f}s ({n_samples/elapsed:.1f}/s)")
        return outputs


class WeightedVectorBaseline:
    """
    Simple weighted vector baseline (no rotation).

    This is the standard approach: blend cluster outputs via weighted sum.
    """

    def __init__(self, federated_model_path: str):
        model_path = Path(federated_model_path)
        with open(model_path, 'rb') as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.temperature = self.meta.get("temperature", 0.1)
        self.cluster_centroids = self.meta.get("cluster_centroids")

        # Load routing data
        cluster_dir = model_path.with_suffix('')
        routing_data = np.load(cluster_dir / "routing_data.npz")
        self.query_embeddings = routing_data['query_embeddings']

        # Load cluster W matrices (for linear projection)
        self.cluster_W = {}
        for cid in self.cluster_ids:
            # Handle both "cluster_X" and "X" naming conventions
            if cid.startswith("cluster_"):
                cluster_path = cluster_dir / f"{cid}.npz"
            else:
                cluster_path = cluster_dir / f"cluster_{cid}.npz"

            if cluster_path.exists():
                data = np.load(cluster_path)
                if "W" in data:
                    self.cluster_W[cid] = data["W"]
                elif "W_stack" in data:
                    self.cluster_W[cid] = data["W_stack"][0]

        log_info(f"WeightedVectorBaseline: {len(self.cluster_W)} clusters")

    def project(self, query_emb: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Project using weighted vector blending (no rotation)."""
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Routing
        if self.cluster_centroids is not None:
            sims = q_norm @ self.cluster_centroids.T
        else:
            sims = q_norm @ self.query_embeddings.T

        sims_shifted = (sims - np.max(sims)) / self.temperature
        weights = np.exp(sims_shifted)
        weights /= weights.sum()

        top_indices = np.argsort(weights)[-top_k:]

        # Weighted vector blend
        output = np.zeros_like(query_emb)
        total_weight = 0.0

        for idx in top_indices:
            cid = self.cluster_ids[idx]
            if cid in self.cluster_W:
                w = weights[idx]
                # Apply cluster projection and blend
                output += w * (self.cluster_W[cid] @ query_emb)
                total_weight += w

        if total_weight > 0:
            output /= total_weight

        return output.astype(np.float32)

    def compute_outputs_batched(self, queries: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Batch computation."""
        outputs = np.zeros_like(queries)
        for i, q in enumerate(queries):
            outputs[i] = self.project(q, top_k)
        return outputs


def validate_against_full_rotation(
    codebook: FastOrthogonalCodebook,
    federated_model_path: str,
    n_samples: int = 500,
    top_k: int = 10,
) -> Dict:
    """
    Compare orthogonal codebook outputs to full rotational manifold.

    Args:
        codebook: FastOrthogonalCodebook to validate
        federated_model_path: Path to federated model
        n_samples: Number of samples to compare
        top_k: Top-K for routing

    Returns:
        Dict with comparison metrics
    """
    log_info(f"\nValidating orthogonal codebook against full rotation manifold...")

    # Load full rotation teacher
    full_teacher = FullRotationTeacher(federated_model_path)

    # Load query embeddings
    model_path = Path(federated_model_path)
    cluster_dir = model_path.with_suffix('')
    routing_data = np.load(cluster_dir / "routing_data.npz")
    all_queries = routing_data['query_embeddings'].astype(np.float32)

    # Sample queries
    if n_samples < len(all_queries):
        indices = np.random.choice(len(all_queries), n_samples, replace=False)
        queries = all_queries[indices]
    else:
        queries = all_queries
        n_samples = len(queries)

    log_info(f"  Comparing {n_samples} samples...")

    # Compute full rotation outputs (slow)
    log_info("  Computing full rotation outputs (matrix exp)...")
    full_outputs = full_teacher.compute_outputs_batched(queries, top_k)

    # Compute orthogonal codebook outputs (fast)
    log_info("  Computing orthogonal codebook outputs (Rodrigues)...")
    start = time.time()
    orth_outputs, _, _ = codebook.route_and_apply(queries, top_k=top_k)
    orth_time = time.time() - start
    log_info(f"    Orthogonal: {n_samples} samples in {orth_time:.2f}s ({n_samples/orth_time:.0f}/s)")

    # Compute metrics
    # Cosine similarity
    full_norms = np.linalg.norm(full_outputs, axis=1, keepdims=True) + 1e-8
    orth_norms = np.linalg.norm(orth_outputs, axis=1, keepdims=True) + 1e-8
    full_normed = full_outputs / full_norms
    orth_normed = orth_outputs / orth_norms
    cosine_sims = np.sum(full_normed * orth_normed, axis=1)

    # MSE
    mse_values = np.mean((full_outputs - orth_outputs) ** 2, axis=1)

    # Angle between outputs (in degrees)
    angles = np.arccos(np.clip(cosine_sims, -1, 1)) * 180 / np.pi

    results = {
        'n_samples': n_samples,
        'mean_cosine_sim': float(np.mean(cosine_sims)),
        'std_cosine_sim': float(np.std(cosine_sims)),
        'min_cosine_sim': float(np.min(cosine_sims)),
        'max_cosine_sim': float(np.max(cosine_sims)),
        'mean_mse': float(np.mean(mse_values)),
        'mean_angle_deg': float(np.mean(angles)),
        'max_angle_deg': float(np.max(angles)),
    }

    return results


def evaluate_hit_at_k(
    codebook: FastOrthogonalCodebook,
    federated_model_path: str,
    k_values: List[int] = [1, 5, 10, 20],
    top_k_routing: int = 10,
) -> Dict:
    """
    Evaluate hit@K for RAG retrieval.

    For each query, project query and all targets, find if correct target is in top-K.
    """
    log_info(f"\nEvaluating hit@K for RAG retrieval...")

    # Load data
    model_path = Path(federated_model_path)
    cluster_dir = model_path.with_suffix('')
    routing_data = np.load(cluster_dir / "routing_data.npz")

    queries = routing_data['query_embeddings'].astype(np.float32)
    targets = routing_data['target_embeddings'].astype(np.float32)
    n_samples = len(queries)

    log_info(f"  {n_samples} query-target pairs")

    # Load models
    full_teacher = FullRotationTeacher(federated_model_path)
    baseline = WeightedVectorBaseline(federated_model_path)

    results = {}

    # Evaluate each approach
    for approach_name, project_fn in [
        ("raw", lambda x: x),  # No projection (raw embeddings)
        ("orthogonal", lambda x: codebook.route_and_apply(x, top_k=top_k_routing)[0]),
        ("baseline", lambda x: baseline.compute_outputs_batched(x, top_k=top_k_routing)),
    ]:
        log_info(f"  Evaluating {approach_name}...")

        # Project queries and targets
        t0 = time.time()
        proj_queries = project_fn(queries)
        proj_targets = project_fn(targets)
        proj_time = time.time() - t0

        # Normalize for cosine similarity
        proj_queries = proj_queries / (np.linalg.norm(proj_queries, axis=1, keepdims=True) + 1e-8)
        proj_targets = proj_targets / (np.linalg.norm(proj_targets, axis=1, keepdims=True) + 1e-8)

        # Compute all pairwise similarities
        similarities = proj_queries @ proj_targets.T  # (n_samples, n_samples)

        # For each query, rank targets and check hit@K
        hits = {k: 0 for k in k_values}

        for i in range(n_samples):
            # Get similarity scores for this query
            sims = similarities[i]
            # Rank targets (descending)
            ranked_indices = np.argsort(sims)[::-1]
            # Find position of correct target (index i)
            correct_rank = np.where(ranked_indices == i)[0][0]

            for k in k_values:
                if correct_rank < k:
                    hits[k] += 1

        hit_rates = {k: hits[k] / n_samples for k in k_values}

        results[approach_name] = {
            'hit_rates': hit_rates,
            'time': proj_time,
            'samples_per_sec': n_samples / proj_time if proj_time > 0 else float('inf'),
        }

    return results


def compare_all_approaches(
    codebook: FastOrthogonalCodebook,
    federated_model_path: str,
    n_samples: int = 200,
    top_k: int = 10,
) -> Dict:
    """
    Compare orthogonal codebook, full rotation, and weighted baseline.

    Returns metrics for each approach compared to full rotation (ground truth).
    """
    log_info(f"\nComparing all approaches on {n_samples} samples...")

    # Load models
    full_teacher = FullRotationTeacher(federated_model_path)
    baseline = WeightedVectorBaseline(federated_model_path)

    # Load queries
    model_path = Path(federated_model_path)
    cluster_dir = model_path.with_suffix('')
    routing_data = np.load(cluster_dir / "routing_data.npz")
    all_queries = routing_data['query_embeddings'].astype(np.float32)

    if n_samples < len(all_queries):
        indices = np.random.choice(len(all_queries), n_samples, replace=False)
        queries = all_queries[indices]
    else:
        queries = all_queries
        n_samples = len(queries)

    # Compute outputs from each approach
    log_info("  Computing full rotation outputs...")
    t0 = time.time()
    full_outputs = full_teacher.compute_outputs_batched(queries, top_k)
    full_time = time.time() - t0

    log_info("  Computing orthogonal codebook outputs...")
    t0 = time.time()
    orth_outputs, _, _ = codebook.route_and_apply(queries, top_k=top_k)
    orth_time = time.time() - t0

    log_info("  Computing weighted baseline outputs...")
    t0 = time.time()
    baseline_outputs = baseline.compute_outputs_batched(queries, top_k)
    baseline_time = time.time() - t0

    def compute_metrics(pred, target):
        """Compute cosine similarity metrics."""
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
        cos_sims = np.sum(pred_norm * target_norm, axis=1)
        return {
            'mean_cos': float(np.mean(cos_sims)),
            'std_cos': float(np.std(cos_sims)),
            'min_cos': float(np.min(cos_sims)),
            'max_cos': float(np.max(cos_sims)),
        }

    # Compare each to full rotation
    orth_vs_full = compute_metrics(orth_outputs, full_outputs)
    baseline_vs_full = compute_metrics(baseline_outputs, full_outputs)

    # Also compare orthogonal to baseline directly
    orth_vs_baseline = compute_metrics(orth_outputs, baseline_outputs)

    return {
        'n_samples': n_samples,
        'top_k': top_k,
        'times': {
            'full_rotation': full_time,
            'orthogonal': orth_time,
            'baseline': baseline_time,
        },
        'orth_vs_full': orth_vs_full,
        'baseline_vs_full': baseline_vs_full,
        'orth_vs_baseline': orth_vs_baseline,
    }


# =============================================================================
# Codebook I/O
# =============================================================================

def save_orthogonal_codebook(codebook: FastOrthogonalCodebook, path: str):
    """Save orthogonal codebook to .npz file."""
    np.savez(
        path,
        bivectors=codebook.bivectors,
        plane_u=codebook.plane_u,
        plane_v=codebook.plane_v,
        plane_theta=codebook.plane_theta,
        codebook_keys=codebook.codebook_keys,
        d=codebook.d,
        n_components=codebook.n_components,
    )
    log_info(f"Saved orthogonal codebook to {path}")


def load_orthogonal_codebook(path: str) -> FastOrthogonalCodebook:
    """Load orthogonal codebook from .npz file."""
    data = np.load(path)

    # Reconstruct codebook
    codebook = FastOrthogonalCodebook.__new__(FastOrthogonalCodebook)
    codebook.bivectors = data['bivectors']
    codebook.plane_u = data['plane_u']
    codebook.plane_v = data['plane_v']
    codebook.plane_theta = data['plane_theta']
    codebook.codebook_keys = data['codebook_keys']
    codebook.d = int(data['d'])
    codebook.n_components = int(data['n_components'])
    codebook.planes = [(codebook.plane_u[i], codebook.plane_v[i], codebook.plane_theta[i])
                       for i in range(codebook.n_components)]

    log_info(f"Loaded orthogonal codebook: {codebook.n_components} planes, d={codebook.d}")
    return codebook


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast Orthogonal Bivector Codebook for rotation projection"
    )

    # Input
    parser.add_argument("--codebook", help="Path to existing bivector codebook (.npz)")
    parser.add_argument("--orthogonal-codebook", help="Path to orthogonal codebook (.npz)")
    parser.add_argument("--federated-model", help="Path to federated model (.pkl)")

    # Operations
    parser.add_argument("--orthogonalize", action="store_true",
                       help="Orthogonalize existing codebook")
    parser.add_argument("--build-canonical", action="store_true",
                       help="Build canonical orthogonal codebook from scratch")
    parser.add_argument("--test", action="store_true",
                       help="Test codebook with sample data")
    parser.add_argument("--train", action="store_true",
                       help="Train FastOrthogonalTransformer")
    parser.add_argument("--validate", action="store_true",
                       help="Validate orthogonal codebook against full rotation manifold")
    parser.add_argument("--compare", action="store_true",
                       help="Compare orthogonal vs full rotation vs weighted baseline")
    parser.add_argument("--hit-at-k", action="store_true",
                       help="Evaluate hit@K for RAG retrieval")
    parser.add_argument("--validate-samples", type=int, default=500,
                       help="Number of samples for validation/comparison")

    # Parameters
    parser.add_argument("--n-components", type=int, default=64,
                       help="Number of codebook entries")
    parser.add_argument("--top-k", type=int, default=8,
                       help="Top-K entries to blend")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")

    # Output
    parser.add_argument("--output", help="Path to save orthogonal codebook")

    args = parser.parse_args()

    print("=" * 60)
    print("Fast Orthogonal Bivector Codebook")
    print("=" * 60)

    if args.orthogonalize and args.codebook:
        # Load existing codebook and orthogonalize
        log_info(f"\nLoading codebook: {args.codebook}")
        data = np.load(args.codebook, allow_pickle=True)
        bivectors = data['basis']
        codebook_keys = data.get('codebook_keys')

        log_info(f"  Original shape: {bivectors.shape}")

        # Orthogonalize
        orth_bivectors, planes = orthogonalize_codebook(bivectors)

        # Create FastOrthogonalCodebook
        codebook = FastOrthogonalCodebook(orth_bivectors, codebook_keys)

        if args.output:
            save_orthogonal_codebook(codebook, args.output)

    elif args.build_canonical:
        # Build canonical codebook
        d = 768  # Default embedding dim
        if args.federated_model:
            with open(args.federated_model, 'rb') as f:
                meta = pickle.load(f)
            # Get dimension from model if available
            d = meta.get('embed_dim', 768)

        bivectors, planes = build_canonical_orthogonal_codebook(d, args.n_components)
        codebook = FastOrthogonalCodebook(bivectors)

        if args.output:
            save_orthogonal_codebook(codebook, args.output)

    elif args.test and args.codebook:
        # Test existing orthogonal codebook
        codebook = load_orthogonal_codebook(args.codebook)

        # Generate random test data
        np.random.seed(42)
        batch_size = 100
        x = np.random.randn(batch_size, codebook.d).astype(np.float32)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)

        # Test routing and rotation
        log_info(f"\nTesting with {batch_size} random vectors...")

        start = time.time()
        y, weights, indices = codebook.route_and_apply(x, top_k=args.top_k)
        elapsed = time.time() - start

        log_info(f"  Rodrigues rotation: {elapsed*1000:.2f}ms for {batch_size} vectors")
        log_info(f"  Output norm mean: {np.linalg.norm(y, axis=1).mean():.4f}")
        log_info(f"  Avg top-K weight: {weights.mean():.4f}")

        # Compare with matrix exponential (if small enough)
        if codebook.n_components <= 64:
            log_info("\nComparing with matrix exponential...")

            # Build blended bivector
            full_weights = np.zeros((batch_size, codebook.n_components))
            for b in range(batch_size):
                full_weights[b, indices[b]] = weights[b]

            start = time.time()
            for b in range(min(10, batch_size)):
                B_blend = np.tensordot(full_weights[b], codebook.bivectors, axes=([0], [0]))
                R = expm(B_blend)
                y_exp = R @ x[b]
            exp_elapsed = time.time() - start

            log_info(f"  Matrix exp (10 samples): {exp_elapsed*1000:.2f}ms")
            log_info(f"  Speedup estimate: {(exp_elapsed/10 * batch_size) / elapsed:.1f}x")

    elif args.train:
        # Full training with orthogonal codebook
        if not args.federated_model:
            log_info("ERROR: --federated-model required for training")
            return 1

        if not args.orthogonal_codebook and not args.codebook:
            log_info("ERROR: --orthogonal-codebook or --codebook required for training")
            return 1

        # Load or create orthogonal codebook
        if args.orthogonal_codebook:
            log_info(f"\nLoading orthogonal codebook: {args.orthogonal_codebook}")
            codebook = load_orthogonal_codebook(args.orthogonal_codebook)
        else:
            log_info(f"\nLoading codebook: {args.codebook}")
            data = np.load(args.codebook, allow_pickle=True)
            bivectors = data['basis']
            codebook_keys = data.get('codebook_keys')

            log_info(f"  Original shape: {bivectors.shape}")
            log_info("Orthogonalizing...")

            # Orthogonalize
            orth_bivectors, planes = orthogonalize_codebook(bivectors)
            codebook = FastOrthogonalCodebook(orth_bivectors, codebook_keys)

        # Load federated model routing data
        federated_path = Path(args.federated_model)
        cluster_dir = federated_path.with_suffix('')
        routing_path = cluster_dir / "routing_data.npz"

        if not routing_path.exists():
            log_info(f"ERROR: Routing data not found: {routing_path}")
            return 1

        routing_data = np.load(routing_path)
        query_embeddings = routing_data['query_embeddings'].astype(np.float32)
        log_info(f"  Loaded {len(query_embeddings)} query embeddings")

        # Create teacher
        log_info("\nCreating OrthogonalTeacher...")
        teacher = OrthogonalTeacher(args.federated_model, codebook)

        # Create transformer
        log_info("\nCreating FastOrthogonalTransformer...")
        transformer = FastOrthogonalTransformer(
            codebook=codebook,
            num_layers=args.layers,
            num_heads=args.heads,
            top_k=args.top_k,
        )

        # Count parameters
        total_params = sum(p.numel() for p in transformer.parameters())
        log_info(f"  Parameters: {total_params:,}")
        log_info(f"  Codebook size: {codebook.n_components}")
        log_info(f"  Top-K: {args.top_k}")
        log_info(f"  Device: {transformer.device}")

        # Split train/test
        n_test = int(len(query_embeddings) * args.test_split)
        indices = np.random.permutation(len(query_embeddings))
        train_idx = indices[:-n_test]
        test_idx = indices[-n_test:]

        train_queries = query_embeddings[train_idx]
        test_queries = query_embeddings[test_idx]

        log_info(f"\nTrain: {len(train_queries)}, Test: {len(test_queries)}")

        # Train
        log_info(f"\nTraining for {args.epochs} epochs...")
        losses = train_orthogonal_transformer(
            transformer=transformer,
            teacher=teacher,
            query_embeddings=train_queries,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            log_interval=10,
            top_k=args.top_k,
        )

        # Evaluate
        log_info("\nEvaluating on test set...")
        results = evaluate_orthogonal_transformer(
            transformer, teacher, test_queries,
            batch_size=128, top_k=args.top_k
        )

        print(f"\n{'=' * 40}")
        print("Results (vs orthogonal teacher):")
        print(f"  Mean Cosine Sim: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
        print(f"  Min/Max Cosine: [{results['min_cosine_sim']:.4f}, {results['max_cosine_sim']:.4f}]")
        print(f"  Mean MSE: {results['mean_mse']:.6f}")
        print(f"{'=' * 40}")

        if results['mean_cosine_sim'] > 0.90:
            print("\n✓ Excellent: Transformer closely matches orthogonal teacher")
        elif results['mean_cosine_sim'] > 0.80:
            print("\n✓ Good: Transformer reasonably matches teacher")
        elif results['mean_cosine_sim'] > 0.70:
            print("\n~ Fair: Room for improvement")
        else:
            print("\n~ Needs work: Consider more training or different architecture")

    elif args.validate:
        # Validate orthogonal codebook against full rotation manifold
        if not args.federated_model:
            log_info("ERROR: --federated-model required for validation")
            return 1

        if not args.orthogonal_codebook and not args.codebook:
            log_info("ERROR: --orthogonal-codebook or --codebook required for validation")
            return 1

        # Load or create orthogonal codebook
        if args.orthogonal_codebook:
            log_info(f"\nLoading orthogonal codebook: {args.orthogonal_codebook}")
            codebook = load_orthogonal_codebook(args.orthogonal_codebook)
        else:
            log_info(f"\nLoading codebook: {args.codebook}")
            data = np.load(args.codebook, allow_pickle=True)
            bivectors = data['basis']
            codebook_keys = data.get('codebook_keys')

            log_info(f"  Original shape: {bivectors.shape}")
            log_info("Orthogonalizing...")

            orth_bivectors, planes = orthogonalize_codebook(bivectors)
            codebook = FastOrthogonalCodebook(orth_bivectors, codebook_keys)

        # Run validation
        results = validate_against_full_rotation(
            codebook=codebook,
            federated_model_path=args.federated_model,
            n_samples=args.validate_samples,
            top_k=args.top_k,
        )

        print(f"\n{'=' * 60}")
        print("Validation: Orthogonal Codebook vs Full Rotation Manifold")
        print(f"{'=' * 60}")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Codebook: {codebook.n_components} orthogonal planes")
        print(f"  Full model: 124 cluster bivectors (matrix exp)")
        print()
        print(f"  Mean Cosine Sim: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
        print(f"  Min/Max Cosine: [{results['min_cosine_sim']:.4f}, {results['max_cosine_sim']:.4f}]")
        print(f"  Mean Angle: {results['mean_angle_deg']:.2f}° (max: {results['max_angle_deg']:.2f}°)")
        print(f"  Mean MSE: {results['mean_mse']:.6f}")
        print(f"{'=' * 60}")

        if results['mean_cosine_sim'] > 0.95:
            print("\n✓ Excellent: Orthogonal codebook closely approximates full rotation")
        elif results['mean_cosine_sim'] > 0.90:
            print("\n✓ Good: Reasonable approximation quality")
        elif results['mean_cosine_sim'] > 0.80:
            print("\n~ Fair: Some information loss, may need more planes")
        else:
            print("\n✗ Poor: Significant information loss, consider larger codebook")

    elif args.compare:
        # Compare all approaches
        if not args.federated_model:
            log_info("ERROR: --federated-model required for comparison")
            return 1

        if not args.orthogonal_codebook and not args.codebook:
            log_info("ERROR: --orthogonal-codebook or --codebook required")
            return 1

        # Load codebook
        if args.orthogonal_codebook:
            codebook = load_orthogonal_codebook(args.orthogonal_codebook)
        else:
            data = np.load(args.codebook, allow_pickle=True)
            bivectors = data['basis']
            orth_bivectors, _ = orthogonalize_codebook(bivectors)
            codebook = FastOrthogonalCodebook(orth_bivectors, data.get('codebook_keys'))

        # Run comparison
        results = compare_all_approaches(
            codebook=codebook,
            federated_model_path=args.federated_model,
            n_samples=args.validate_samples,
            top_k=args.top_k,
        )

        print(f"\n{'=' * 70}")
        print("Comparison: Orthogonal Codebook vs Full Rotation vs Weighted Baseline")
        print(f"{'=' * 70}")
        print(f"Samples: {results['n_samples']}, Top-K: {results['top_k']}")
        print(f"Orthogonal planes: {codebook.n_components}")
        print()

        print("Speed (samples/sec):")
        print(f"  Full rotation:  {results['n_samples']/results['times']['full_rotation']:>8.1f}/s")
        print(f"  Orthogonal:     {results['n_samples']/results['times']['orthogonal']:>8.1f}/s")
        print(f"  Baseline:       {results['n_samples']/results['times']['baseline']:>8.1f}/s")
        print()

        print("Cosine Similarity vs Full Rotation (ground truth):")
        print(f"  Orthogonal:     {results['orth_vs_full']['mean_cos']:.4f} ± {results['orth_vs_full']['std_cos']:.4f}")
        print(f"  Baseline:       {results['baseline_vs_full']['mean_cos']:.4f} ± {results['baseline_vs_full']['std_cos']:.4f}")
        print()

        print("Orthogonal vs Baseline (direct):")
        print(f"  Cosine:         {results['orth_vs_baseline']['mean_cos']:.4f} ± {results['orth_vs_baseline']['std_cos']:.4f}")
        print(f"{'=' * 70}")

        # Analysis
        orth_score = results['orth_vs_full']['mean_cos']
        base_score = results['baseline_vs_full']['mean_cos']

        if orth_score > base_score:
            print(f"\n✓ Orthogonal is closer to full rotation than baseline ({orth_score:.3f} > {base_score:.3f})")
        elif base_score > orth_score:
            print(f"\n✗ Baseline is closer to full rotation than orthogonal ({base_score:.3f} > {orth_score:.3f})")
        else:
            print(f"\n~ Both approaches similar distance from full rotation")

    elif args.hit_at_k:
        # Evaluate hit@K for RAG
        if not args.federated_model:
            log_info("ERROR: --federated-model required")
            return 1

        if not args.orthogonal_codebook and not args.codebook:
            log_info("ERROR: --orthogonal-codebook or --codebook required")
            return 1

        # Load codebook
        if args.orthogonal_codebook:
            codebook = load_orthogonal_codebook(args.orthogonal_codebook)
        else:
            data = np.load(args.codebook, allow_pickle=True)
            bivectors = data['basis']
            orth_bivectors, _ = orthogonalize_codebook(bivectors)
            codebook = FastOrthogonalCodebook(orth_bivectors, data.get('codebook_keys'))

        # Run hit@K evaluation
        results = evaluate_hit_at_k(
            codebook=codebook,
            federated_model_path=args.federated_model,
            k_values=[1, 5, 10, 20, 50],
            top_k_routing=args.top_k,
        )

        print(f"\n{'=' * 70}")
        print("Hit@K Evaluation for RAG Retrieval")
        print(f"{'=' * 70}")
        print(f"Orthogonal planes: {codebook.n_components}, Top-K routing: {args.top_k}")
        print()

        # Print header
        k_values = [1, 5, 10, 20, 50]
        header = f"{'Approach':<15} | " + " | ".join([f"Hit@{k:2d}" for k in k_values]) + " | Speed"
        print(header)
        print("-" * len(header))

        # Print results for each approach
        for approach in ["raw", "orthogonal", "baseline"]:
            r = results[approach]
            hits_str = " | ".join([f"{r['hit_rates'][k]*100:5.1f}%" for k in k_values])
            speed_str = f"{r['samples_per_sec']:,.0f}/s"
            print(f"{approach:<15} | {hits_str} | {speed_str}")

        print(f"{'=' * 70}")

        # Analysis
        raw_h1 = results['raw']['hit_rates'][1]
        orth_h1 = results['orthogonal']['hit_rates'][1]
        base_h1 = results['baseline']['hit_rates'][1]

        print(f"\nHit@1 comparison:")
        print(f"  Raw embeddings: {raw_h1*100:.1f}%")
        print(f"  Orthogonal:     {orth_h1*100:.1f}% ({'+' if orth_h1 > raw_h1 else ''}{(orth_h1-raw_h1)*100:.1f}%)")
        print(f"  Baseline:       {base_h1*100:.1f}% ({'+' if base_h1 > raw_h1 else ''}{(base_h1-raw_h1)*100:.1f}%)")

    else:
        parser.print_help()
        return 1

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
