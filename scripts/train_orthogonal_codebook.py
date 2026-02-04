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

    # Train with orthogonal teacher (fast)
    python scripts/train_orthogonal_codebook.py \
        --federated-model models/federated.pkl \
        --codebook models/orthogonal_codebook.npz \
        --epochs 50

    # Train with rotational teacher (accurate, distills logm/expm into transformer)
    python scripts/train_orthogonal_codebook.py \
        --train \
        --federated-model models/pearltrees_federated_nomic.pkl \
        --build-canonical \
        --n-components 64 \
        --layers 3 \
        --teacher rotational \
        --epochs 50 \
        --save-transformer models/orthogonal_transformer.pt

    # Train with JAX rotational teacher (uses JAX XLA for matrix_exp)
    python scripts/train_orthogonal_codebook.py \
        --train \
        --federated-model models/pearltrees_federated_nomic.pkl \
        --build-canonical \
        --n-components 64 \
        --layers 3 \
        --teacher jax \
        --epochs 50 \
        --save-transformer models/orthogonal_transformer_jax.pt
"""

import argparse
import sys
import logging
import time
import math
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
# Composed Rotation Transformer (Proper Rotation Composition)
# =============================================================================

class ComposedRotationTransformer:
    """
    Transformer that COMPOSES rotations instead of adding deltas.

    This preserves structural information better than FastOrthogonalTransformer
    because rotation composition is the correct operation for SO(n).

    For top-K selected planes with angles θ_1, θ_2, ..., θ_K:
        R_total = R_K @ R_{K-1} @ ... @ R_1
        output = R_total @ query

    This is still fast (just matrix multiplications) but correctly handles
    large rotation angles.
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
            raise ImportError("PyTorch required for ComposedRotationTransformer")

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

        log_info(f"ComposedRotationTransformer: {self.n_components} planes, "
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

        # Routing head - outputs per-plane rotation scales
        self.routing_head = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        # Angle scaling head - learns to scale the base angles per plane
        self.angle_head = nn.Sequential(
            nn.Linear(self.embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, self.n_components)
        ).to(self.device)

        # Scale head
        self.scale_head = nn.Sequential(
            nn.Linear(self.embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, 1)
        ).to(self.device)

    def _rodrigues_rotation_matrix(self, u, v, theta):
        """
        Build rotation matrix for a single 2D plane using Rodrigues formula.

        R = I + sin(θ)(vu^T - uv^T) + (cos(θ) - 1)(uu^T + vv^T)

        Args:
            u: (d,) first basis vector of rotation plane
            v: (d,) second basis vector of rotation plane
            theta: scalar rotation angle

        Returns:
            R: (d, d) rotation matrix
        """
        torch, _, _ = _import_torch()
        d = u.shape[0]

        # Outer products
        uuT = torch.outer(u, u)
        vvT = torch.outer(v, v)
        uvT = torch.outer(u, v)
        vuT = torch.outer(v, u)

        # Rodrigues formula
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        R = torch.eye(d, device=u.device) + sin_t * (vuT - uvT) + (cos_t - 1) * (uuT + vvT)
        return R

    def forward(self, query):
        """
        Forward pass with composed Rodrigues rotations.

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

        # Get angle scales from learned head
        angle_scales = torch.tanh(self.angle_head(x))  # (-1, 1) range

        # Compose rotations for each query in batch
        projected = torch.zeros_like(query)

        for b in range(batch_size):
            # Start with identity
            result = query[b].clone()

            # Apply rotations in sequence (compose them)
            for k in range(self.top_k):
                idx = top_k_idx[b, k].item()
                w = weights[b, k]

                # Scale the base angle by weight and learned scale
                theta = self.plane_theta[idx] * w * (1 + angle_scales[b, idx])

                # Get rotation plane
                u = self.plane_u[idx]
                v = self.plane_v[idx]

                # Build rotation matrix
                R = self._rodrigues_rotation_matrix(u, v, theta)

                # Apply rotation (compose)
                result = torch.mv(R, result)

            projected[b] = result

        # Scale
        scale = self.scale_head(x).squeeze(-1)
        scale = F.softplus(scale) + 0.5
        projected = projected * scale.unsqueeze(-1)

        return projected, weights, top_k_idx, scale

    def forward_batched(self, query):
        """
        Batched forward pass - more efficient for large batches.

        Uses a single composed rotation per query by pre-computing
        the blended rotation matrix.
        """
        torch, nn, F = _import_torch()
        batch_size = query.shape[0]

        # Encode
        x = self.input_proj(query).unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze(1)

        # Routing
        routing_vec = self.routing_head(x)
        routing_vec = F.normalize(routing_vec, dim=-1)
        similarities = torch.mm(routing_vec, self.codebook_keys.T)

        # Top-K selection
        top_k_sim, top_k_idx = similarities.topk(self.top_k, dim=-1)
        top_k_sim_pos = torch.clamp(top_k_sim, min=0.0)
        weight_sum = top_k_sim_pos.sum(dim=-1, keepdim=True) + 1e-8
        weights = top_k_sim_pos / weight_sum

        # Angle scales
        angle_scales = torch.tanh(self.angle_head(x))

        # Build rotation matrices for all top-k planes
        # This is the batched version that pre-computes skew matrices
        d = self.embed_dim

        # Pre-compute skew matrices for all planes: K = vu^T - uv^T
        # and projection matrices: P = uu^T + vv^T
        K_all = torch.einsum('ni,nj->nij', self.plane_v, self.plane_u) - \
                torch.einsum('ni,nj->nij', self.plane_u, self.plane_v)
        P_all = torch.einsum('ni,nj->nij', self.plane_u, self.plane_u) + \
                torch.einsum('ni,nj->nij', self.plane_v, self.plane_v)

        projected = torch.zeros_like(query)

        for b in range(batch_size):
            # Compose rotation matrix for this query
            R_total = torch.eye(d, device=self.device)

            for k in range(self.top_k):
                idx = top_k_idx[b, k].item()
                w = weights[b, k]
                theta = self.plane_theta[idx] * w * (1 + angle_scales[b, idx])

                sin_t = torch.sin(theta)
                cos_t = torch.cos(theta)

                R_k = torch.eye(d, device=self.device) + sin_t * K_all[idx] + (cos_t - 1) * P_all[idx]
                R_total = R_k @ R_total

            projected[b] = R_total @ query[b]

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
        params += list(self.angle_head.parameters())
        params += list(self.scale_head.parameters())
        return params

    def train_mode(self):
        self.input_proj.train()
        self.encoder.train()
        self.routing_head.train()
        self.angle_head.train()
        self.scale_head.train()

    def eval_mode(self):
        self.input_proj.eval()
        self.encoder.eval()
        self.routing_head.eval()
        self.angle_head.eval()
        self.scale_head.eval()


# =============================================================================
# Custom Autograd for Rodrigues Rotation
# =============================================================================

class RodriguesRotationFunction:
    """
    Custom autograd function for Rodrigues rotation.

    Forward:  R = I + α·B + β·B²  where α = sin(θ)/θ, β = (1-cos(θ))/θ², θ = ||B||_F/√2
    Backward: Analytical gradients, all O(N²) operations.

    This avoids autograd tracing and provides numerically stable gradients.
    """

    @staticmethod
    def apply(B, query):
        """
        Apply Rodrigues rotation.

        Args:
            B: (batch, d, d) blended skew-symmetric bivector
            query: (batch, d) input vectors

        Returns:
            output: (batch, d) rotated vectors
            cache: tuple of saved tensors for backward
        """
        torch, _, _ = _import_torch()

        batch_size = B.shape[0]
        d = B.shape[1]
        device = B.device

        # Compute theta = ||B||_F / sqrt(2) for each batch item
        theta = torch.norm(B.view(batch_size, -1), dim=1) / math.sqrt(2)  # (batch,)

        # Rodrigues coefficients with numerical stability
        alpha = torch.zeros(batch_size, device=device)
        beta = torch.zeros(batch_size, device=device)

        # Small angle approximation (Taylor series)
        small = theta < 1e-6
        alpha[small] = 1.0 - theta[small]**2 / 6  # sin(θ)/θ ≈ 1 - θ²/6
        beta[small] = 0.5 - theta[small]**2 / 24   # (1-cos(θ))/θ² ≈ 1/2 - θ²/24

        # Normal case
        large = ~small
        alpha[large] = torch.sin(theta[large]) / theta[large]
        beta[large] = (1 - torch.cos(theta[large])) / (theta[large] ** 2)

        # B² for each batch item
        B2 = torch.bmm(B, B)  # (batch, d, d)

        # R = I + α·B + β·B²
        I = torch.eye(d, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        R = I + alpha.view(-1, 1, 1) * B + beta.view(-1, 1, 1) * B2

        # Apply rotation: output = R @ query
        output = torch.bmm(R, query.unsqueeze(-1)).squeeze(-1)

        # Cache for backward
        cache = (B, B2, R, query, theta, alpha, beta)

        return output, cache

    @staticmethod
    def backward(grad_output, cache):
        """
        Compute gradients analytically.

        Args:
            grad_output: (batch, d) gradient w.r.t. output
            cache: saved tensors from forward

        Returns:
            grad_B: (batch, d, d) gradient w.r.t. blended bivector
            grad_query: (batch, d) gradient w.r.t. query
        """
        torch, _, _ = _import_torch()

        B, B2, R, query, theta, alpha, beta = cache
        batch_size = B.shape[0]
        d = B.shape[1]
        device = B.device

        # ∂L/∂R = grad_output ⊗ query (outer product)
        G = torch.bmm(grad_output.unsqueeze(-1), query.unsqueeze(1))  # (batch, d, d)

        # Scalar derivatives ∂α/∂θ and ∂β/∂θ
        d_alpha = torch.zeros(batch_size, device=device)
        d_beta = torch.zeros(batch_size, device=device)

        small = theta < 1e-6
        d_alpha[small] = -theta[small] / 3  # derivative of Taylor expansion
        d_beta[small] = -theta[small] / 12

        large = ~small
        t = theta[large]
        d_alpha[large] = (t * torch.cos(t) - torch.sin(t)) / (t ** 2)
        d_beta[large] = (t * torch.sin(t) - 2 * (1 - torch.cos(t))) / (t ** 3)

        # ∂L/∂B = α·G + β·(G@B + B@G) + θ-derivative terms
        grad_B = alpha.view(-1, 1, 1) * G
        grad_B = grad_B + beta.view(-1, 1, 1) * (torch.bmm(G, B) + torch.bmm(B, G))

        # θ-derivative contribution: (∂α/∂θ·⟨G,B⟩ + ∂β/∂θ·⟨G,B²⟩) · B / (θ·√2)
        G_dot_B = (G * B).sum(dim=(-2, -1))  # Frobenius inner product
        G_dot_B2 = (G * B2).sum(dim=(-2, -1))

        theta_grad = d_alpha * G_dot_B + d_beta * G_dot_B2
        theta_safe = theta + 1e-8  # avoid division by zero
        grad_B = grad_B + (theta_grad / (theta_safe * math.sqrt(2))).view(-1, 1, 1) * B

        # ∂L/∂query = R^T @ grad_output
        grad_query = torch.bmm(R.transpose(-2, -1), grad_output.unsqueeze(-1)).squeeze(-1)

        return grad_B, grad_query


def build_orthogonal_bivector_basis(federated_model_path: str, n_basis: int = 128):
    """
    Build an orthogonal basis of bivectors from all cluster bivectors.

    Uses SVD to extract principal rotation subspaces that are orthogonal.

    Args:
        federated_model_path: Path to federated model with bivectors
        n_basis: Number of basis bivectors to keep

    Returns:
        basis_bivectors: (n_basis, d, d) orthogonal bivectors
        basis_keys: (n_basis, d) key vectors for routing
    """
    # Load bivectors
    biv_path = Path(federated_model_path).with_suffix('.bivectors.npz')
    if not biv_path.exists():
        raise FileNotFoundError(f"Bivector cache not found: {biv_path}")

    biv_data = np.load(biv_path)
    bivectors = biv_data['bivectors']  # (n_clusters, d, d)
    n_clusters, d, _ = bivectors.shape

    log_info(f"Building orthogonal bivector basis from {n_clusters} cluster bivectors...")

    # Flatten bivectors to vectors for SVD
    # Each bivector is skew-symmetric, so we only need upper triangle
    # But for simplicity, we'll use full matrix flattening
    bivectors_flat = bivectors.reshape(n_clusters, d * d)

    # SVD to find principal components
    from scipy.linalg import svd
    U, s, Vh = svd(bivectors_flat, full_matrices=False)

    # Keep top n_basis components
    n_basis = min(n_basis, len(s))
    basis_flat = Vh[:n_basis]  # (n_basis, d*d)

    # Reshape back to matrices
    basis_bivectors = basis_flat.reshape(n_basis, d, d)

    # Ensure skew-symmetry (average with negative transpose)
    basis_bivectors = (basis_bivectors - basis_bivectors.transpose(0, 2, 1)) / 2

    # Normalize each basis bivector
    norms = np.linalg.norm(basis_bivectors.reshape(n_basis, -1), axis=1, keepdims=True)
    basis_bivectors = basis_bivectors / (norms.reshape(n_basis, 1, 1) + 1e-8)

    # Create key vectors (use first column of each bivector as a simple key)
    # Better: use the principal rotation vector
    basis_keys = np.zeros((n_basis, d), dtype=np.float32)
    for i in range(n_basis):
        # Extract rotation axis from bivector (eigenvector with largest imaginary eigenvalue)
        eigvals, eigvecs = np.linalg.eig(basis_bivectors[i])
        max_idx = np.argmax(np.abs(eigvals.imag))
        basis_keys[i] = eigvecs[:, max_idx].real

    # Normalize keys
    basis_keys = basis_keys / (np.linalg.norm(basis_keys, axis=1, keepdims=True) + 1e-8)

    # Check explained variance
    explained_var = s[:n_basis].sum() / s.sum() * 100
    log_info(f"  Kept {n_basis} basis bivectors, explaining {explained_var:.1f}% of variance")

    return basis_bivectors.astype(np.float32), basis_keys.astype(np.float32)


class ComposedBivectorTransformer:
    """
    Transformer that blends full bivectors and applies Rodrigues rotation.

    This properly preserves rotational structure by:
    1. Learning weights for an orthogonal basis of bivectors
    2. Blending bivectors: B = Σ wᵢ·Bᵢ (O(N²))
    3. Applying single Rodrigues rotation (O(N²))

    Because the basis is orthogonal, bivector blending is equivalent to
    rotation composition: exp(A+B) = exp(A)@exp(B) when [A,B]=0.
    """

    def __init__(
        self,
        basis_bivectors: np.ndarray,
        basis_keys: np.ndarray,
        cluster_biases: np.ndarray = None,
        bias_mode: str = "none",
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        device: str = "auto",
    ):
        """
        Args:
            basis_bivectors: (n_basis, d, d) orthogonal bivector basis
            basis_keys: (n_basis, d) key vectors for soft routing
            cluster_biases: (n_clusters, d) bias vectors for each cluster
            bias_mode: "none" (no bias), "shared" (same weights as rotation),
                      "separate" (independent attention for bias)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            device: Device to use
        """
        torch, nn, F = _import_torch()
        if torch is None:
            raise ImportError("PyTorch required for ComposedBivectorTransformer")

        self.n_basis = basis_bivectors.shape[0]
        self.embed_dim = basis_bivectors.shape[1]
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bias_mode = bias_mode

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Convert basis to tensors (not trainable - fixed basis)
        self.basis_bivectors = torch.from_numpy(basis_bivectors).float().to(self.device)
        self.basis_keys = torch.from_numpy(basis_keys).float().to(self.device)

        # Store cluster biases if provided
        if cluster_biases is not None:
            self.cluster_biases = torch.from_numpy(cluster_biases).float().to(self.device)
            self.n_clusters = cluster_biases.shape[0]
        else:
            self.cluster_biases = None
            self.n_clusters = 0

        # Build model
        self._build_model(nn, ff_dim, dropout)

        bias_info = f", bias_mode={bias_mode}" if bias_mode != "none" else ""
        log_info(f"ComposedBivectorTransformer: {self.n_basis} basis bivectors{bias_info}, "
                 f"layers={num_layers}, device={self.device}")

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

        # Weight head - outputs weight for EACH basis bivector
        self.weight_head = nn.Sequential(
            nn.Linear(self.embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, self.n_basis)
        ).to(self.device)

        # Optional: scale head for output magnitude
        self.scale_head = nn.Sequential(
            nn.Linear(self.embed_dim, ff_dim // 2),
            nn.GELU(),
            nn.Linear(ff_dim // 2, 1)
        ).to(self.device)

        # Bias attention (for "separate" mode)
        if self.bias_mode == "separate" and self.cluster_biases is not None:
            torch_mod, _, _ = _import_torch()
            # Query projection for bias attention
            self.bias_query_proj = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
            # Key projection (project cluster centroids to keys)
            self.bias_key_proj = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
            # Temperature for attention (create on device directly)
            self.bias_temperature = nn.Parameter(torch_mod.tensor(0.1, device=self.device))

    def forward(self, query, return_weights=False):
        """
        Forward pass with bivector blending and Rodrigues rotation.

        Args:
            query: (batch, embed_dim) input queries
            return_weights: If True, also return the bivector weights (deprecated, always returns 4 values)

        Returns:
            projected: (batch, embed_dim) rotated outputs
            weights: (batch, n_basis) bivector weights
            top_k_idx: (batch, k) top-k basis indices by weight magnitude
            scale: (batch,) scale factors
        """
        torch, nn, F = _import_torch()
        batch_size = query.shape[0]

        # Encode
        x = self.input_proj(query).unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze(1)

        # Predict weights for each basis bivector
        weights = self.weight_head(x)  # (batch, n_basis)

        # Get top-k indices by weight magnitude (for compatibility with training loop)
        top_k = min(8, self.n_basis)
        _, top_k_idx = torch.topk(torch.abs(weights), top_k, dim=-1)

        # Optionally use softmax or keep raw (allows negative weights for inverse rotation)
        # weights = F.softmax(weights, dim=-1)  # if we want positive weights only

        # Blend bivectors: B = Σ wᵢ·Bᵢ
        # Shape: (batch, n_basis) @ (n_basis, d, d) -> (batch, d, d)
        B_blend = torch.einsum('bn,nij->bij', weights, self.basis_bivectors)

        # Apply Rodrigues rotation using custom function
        projected, cache = RodriguesRotationFunction.apply(B_blend, query)

        # Store cache for backward (needed for custom backward)
        self._rodrigues_cache = cache
        self._weights = weights

        # Apply bias if enabled
        if self.bias_mode != "none" and self.cluster_biases is not None:
            if self.bias_mode == "shared":
                # Use same weights as rotation (but for clusters, not basis)
                # Need to map basis weights to cluster weights
                # For now, assume n_basis == n_clusters or use weight_head output directly
                if self.n_basis == self.n_clusters:
                    bias_weights = F.softmax(weights, dim=-1)
                    bias = torch.einsum('bn,nd->bd', bias_weights, self.cluster_biases)
                else:
                    # Fallback: use a simple projection
                    bias_weights = F.softmax(weights[:, :self.n_clusters], dim=-1)
                    bias = torch.einsum('bn,nd->bd', bias_weights, self.cluster_biases)
            elif self.bias_mode == "separate":
                # Independent attention for bias
                # Query: project encoded representation
                bias_query = self.bias_query_proj(x)  # (batch, d)
                # Keys: project cluster biases (or use them directly)
                bias_keys = self.bias_key_proj(self.cluster_biases)  # (n_clusters, d)
                # Attention scores
                attn_scores = torch.matmul(bias_query, bias_keys.T) / self.bias_temperature  # (batch, n_clusters)
                bias_weights = F.softmax(attn_scores, dim=-1)
                # Weighted sum of biases
                bias = torch.einsum('bn,nd->bd', bias_weights, self.cluster_biases)

            projected = projected + bias

        # Scale
        scale = self.scale_head(x).squeeze(-1)
        scale = F.softplus(scale) + 0.5
        projected = projected * scale.unsqueeze(-1)

        # Always return 4 values for compatibility with training loop
        return projected, weights, top_k_idx, scale

    def compute_loss_and_backward(self, query, target, loss_fn):
        """
        Compute loss and gradients with custom backward pass.

        This manually implements backprop through the Rodrigues rotation
        for efficiency and numerical stability.

        Args:
            query: (batch, embed_dim) input
            target: (batch, embed_dim) teacher target
            loss_fn: Loss function (e.g., MSE, cosine)

        Returns:
            loss: scalar loss value
        """
        torch, _, F = _import_torch()

        # Forward pass (now returns 4 values)
        projected, weights, top_k_idx, scale = self.forward(query)

        # Compute loss
        loss = loss_fn(projected, target)

        # Backward through loss
        loss.backward()

        return loss

    def parameters(self):
        """Return trainable parameters."""
        params = list(self.input_proj.parameters())
        params += list(self.encoder.parameters())
        params += list(self.weight_head.parameters())
        params += list(self.scale_head.parameters())
        if self.bias_mode == "separate" and hasattr(self, 'bias_query_proj'):
            params += list(self.bias_query_proj.parameters())
            params += list(self.bias_key_proj.parameters())
            params += [self.bias_temperature]
        return params

    def named_parameters(self):
        """Return named parameters for optimizer."""
        for name, param in self.input_proj.named_parameters():
            yield f"input_proj.{name}", param
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param
        for name, param in self.weight_head.named_parameters():
            yield f"weight_head.{name}", param
        for name, param in self.scale_head.named_parameters():
            yield f"scale_head.{name}", param
        if self.bias_mode == "separate" and hasattr(self, 'bias_query_proj'):
            for name, param in self.bias_query_proj.named_parameters():
                yield f"bias_query_proj.{name}", param
            for name, param in self.bias_key_proj.named_parameters():
                yield f"bias_key_proj.{name}", param
            yield "bias_temperature", self.bias_temperature

    def state_dict(self):
        """Return state dict for saving."""
        sd = {
            'input_proj': self.input_proj.state_dict(),
            'encoder': self.encoder.state_dict(),
            'weight_head': self.weight_head.state_dict(),
            'scale_head': self.scale_head.state_dict(),
        }
        if self.bias_mode == "separate" and hasattr(self, 'bias_query_proj'):
            sd['bias_query_proj'] = self.bias_query_proj.state_dict()
            sd['bias_key_proj'] = self.bias_key_proj.state_dict()
            sd['bias_temperature'] = self.bias_temperature
        return sd

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.input_proj.load_state_dict(state_dict['input_proj'])
        self.encoder.load_state_dict(state_dict['encoder'])
        self.weight_head.load_state_dict(state_dict['weight_head'])
        self.scale_head.load_state_dict(state_dict['scale_head'])
        if self.bias_mode == "separate" and 'bias_query_proj' in state_dict:
            self.bias_query_proj.load_state_dict(state_dict['bias_query_proj'])
            self.bias_key_proj.load_state_dict(state_dict['bias_key_proj'])
            self.bias_temperature = state_dict['bias_temperature']

    def train_mode(self):
        self.input_proj.train()
        self.encoder.train()
        self.weight_head.train()
        self.scale_head.train()
        if self.bias_mode == "separate" and hasattr(self, 'bias_query_proj'):
            self.bias_query_proj.train()
            self.bias_key_proj.train()

    def eval_mode(self):
        self.input_proj.eval()
        self.encoder.eval()
        self.weight_head.eval()
        self.scale_head.eval()
        if self.bias_mode == "separate" and hasattr(self, 'bias_query_proj'):
            self.bias_query_proj.eval()
            self.bias_key_proj.eval()

    def to(self, device):
        """Move model to device."""
        self.device = torch.device(device)
        self.input_proj = self.input_proj.to(device)
        self.encoder = self.encoder.to(device)
        self.weight_head = self.weight_head.to(device)
        self.scale_head = self.scale_head.to(device)
        self.basis_bivectors = self.basis_bivectors.to(device)
        self.basis_keys = self.basis_keys.to(device)
        if self.cluster_biases is not None:
            self.cluster_biases = self.cluster_biases.to(device)
        if self.bias_mode == "separate" and hasattr(self, 'bias_query_proj'):
            self.bias_query_proj = self.bias_query_proj.to(device)
            self.bias_key_proj = self.bias_key_proj.to(device)
            self.bias_temperature = self.bias_temperature.to(device)
        return self


def save_composed_bivector_transformer(
    transformer: ComposedBivectorTransformer,
    basis_bivectors: np.ndarray,
    basis_keys: np.ndarray,
    path: str,
    metadata: dict = None,
):
    """Save ComposedBivectorTransformer to file."""
    torch, _, _ = _import_torch()

    save_dict = {
        'transformer_type': 'composed_bivector',
        'basis_bivectors': basis_bivectors,
        'basis_keys': basis_keys,
        'n_basis': transformer.n_basis,
        'embed_dim': transformer.embed_dim,
        'num_layers': transformer.num_layers,
        'num_heads': transformer.num_heads,
        'state_dict': transformer.state_dict(),
        'metadata': metadata or {},
    }

    torch.save(save_dict, path)
    log_info(f"Saved ComposedBivectorTransformer to {path}")


def load_composed_bivector_transformer(path: str):
    """Load ComposedBivectorTransformer from file."""
    torch, _, _ = _import_torch()

    save_dict = torch.load(path, map_location='cpu', weights_only=False)

    if save_dict.get('transformer_type') != 'composed_bivector':
        raise ValueError(f"Not a ComposedBivectorTransformer checkpoint: {path}")

    transformer = ComposedBivectorTransformer(
        basis_bivectors=save_dict['basis_bivectors'],
        basis_keys=save_dict['basis_keys'],
        num_layers=save_dict['num_layers'],
        num_heads=save_dict['num_heads'],
    )

    transformer.load_state_dict(save_dict['state_dict'])
    log_info(f"Loaded ComposedBivectorTransformer: {save_dict['n_basis']} basis, "
             f"{save_dict['num_layers']} layers")

    return transformer


# =============================================================================
# Training Checkpoint Support
# =============================================================================

class TrainingCheckpoint:
    """Manages training checkpoints for resumable training."""

    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

    def save(
        self,
        transformer,
        optimizer,
        epoch: int,
        loss: float,
        history: dict,
        is_best: bool = False,
    ):
        """Save training checkpoint."""
        if self.checkpoint_path is None:
            return

        torch, _, _ = _import_torch()

        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'best_loss': self.best_loss,
            'history': history,
            'optimizer_state': optimizer.state_dict(),
        }

        # Handle different transformer types
        if hasattr(transformer, 'state_dict'):
            checkpoint['transformer_state'] = transformer.state_dict()
        else:
            # For FastOrthogonalTransformer etc.
            checkpoint['input_proj_state'] = transformer.input_proj.state_dict()
            checkpoint['encoder_state'] = transformer.encoder.state_dict()
            if hasattr(transformer, 'routing_head'):
                checkpoint['routing_head_state'] = transformer.routing_head.state_dict()
            if hasattr(transformer, 'weight_head'):
                checkpoint['weight_head_state'] = transformer.weight_head.state_dict()
            if hasattr(transformer, 'scale_head'):
                checkpoint['scale_head_state'] = transformer.scale_head.state_dict()
            if hasattr(transformer, 'angle_head'):
                checkpoint['angle_head_state'] = transformer.angle_head.state_dict()

        path = Path(self.checkpoint_path)
        torch.save(checkpoint, path)

        if is_best:
            best_path = path.with_suffix('.best.pt')
            torch.save(checkpoint, best_path)
            log_info(f"  Saved best checkpoint (loss={loss:.6f})")

    def load(self, transformer, optimizer):
        """
        Load training checkpoint.

        Returns:
            start_epoch: Epoch to resume from
            history: Training history dict
        """
        if self.checkpoint_path is None or not Path(self.checkpoint_path).exists():
            return 0, {'train_loss': [], 'val_loss': [], 'cosine_sim': []}

        torch, _, _ = _import_torch()

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        # Load transformer state
        if 'transformer_state' in checkpoint:
            transformer.load_state_dict(checkpoint['transformer_state'])
        else:
            if 'input_proj_state' in checkpoint:
                transformer.input_proj.load_state_dict(checkpoint['input_proj_state'])
            if 'encoder_state' in checkpoint:
                transformer.encoder.load_state_dict(checkpoint['encoder_state'])
            if 'routing_head_state' in checkpoint:
                transformer.routing_head.load_state_dict(checkpoint['routing_head_state'])
            if 'weight_head_state' in checkpoint:
                transformer.weight_head.load_state_dict(checkpoint['weight_head_state'])
            if 'scale_head_state' in checkpoint:
                transformer.scale_head.load_state_dict(checkpoint['scale_head_state'])
            if 'angle_head_state' in checkpoint:
                transformer.angle_head.load_state_dict(checkpoint['angle_head_state'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.best_loss = checkpoint.get('best_loss', float('inf'))

        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'cosine_sim': []})

        log_info(f"Resumed from checkpoint at epoch {checkpoint['epoch']} (loss={checkpoint['loss']:.6f})")

        return start_epoch, history

    def check_early_stopping(self, loss: float, patience: int = 10, min_delta: float = 0.001):
        """
        Check if training should stop early.

        Returns:
            should_stop: True if training should stop
            is_best: True if this is the best loss so far
        """
        is_best = loss < self.best_loss - min_delta

        if is_best:
            self.best_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = self.epochs_without_improvement >= patience

        return should_stop, is_best


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


class PairedDataTeacher:
    """
    Zero-compute teacher: uses pre-stored input/output embedding pairs.

    The federated model already has query_embeddings and target_embeddings
    stored in routing_data.npz. No computation needed at all!

    Also provides centroid pairs (centroid, W @ centroid) for replay.
    """

    def __init__(self, federated_model_path: str):
        """Load pre-stored pairs from federated model."""
        model_path = Path(federated_model_path)
        cluster_dir = model_path.with_suffix('')

        # Load routing data with pre-stored pairs
        routing_path = cluster_dir / "routing_data.npz"
        routing_data = np.load(routing_path)

        self.query_embeddings = routing_data['query_embeddings'].astype(np.float32)
        self.target_embeddings = routing_data['target_embeddings'].astype(np.float32)
        self.d = self.query_embeddings.shape[1]

        # Load cluster centroids and W matrices for centroid replay
        with open(model_path, 'rb') as f:
            meta = pickle.load(f)

        self.cluster_centroids = meta['cluster_centroids'].astype(np.float32)
        self.cluster_ids = meta['cluster_ids']
        self.n_clusters = len(self.cluster_ids)

        # Compute centroid targets as mean of target embeddings
        # Note: cluster_centroids = query_mean, so target should be target_mean
        self.centroid_targets = np.zeros_like(self.cluster_centroids)
        for i, cid in enumerate(self.cluster_ids):
            if cid.startswith("cluster_"):
                cluster_path = cluster_dir / f"{cid}.npz"
            else:
                cluster_path = cluster_dir / f"cluster_{cid}.npz"

            if cluster_path.exists():
                data = np.load(cluster_path)
                if "target_embeddings" in data:
                    # Use actual mean of target embeddings for this cluster
                    # This is correct: centroid_input = query_mean, centroid_target = target_mean
                    self.centroid_targets[i] = data["target_embeddings"].mean(axis=0)
                else:
                    # Fallback: if no target embeddings, use query centroid as target
                    self.centroid_targets[i] = self.cluster_centroids[i]

        log_info(f"PairedDataTeacher: {len(self.query_embeddings)} pre-stored pairs, "
                f"{self.n_clusters} centroids for replay")

    def get_all_pairs(self):
        """Return all pre-stored (input, output) pairs."""
        return self.query_embeddings, self.target_embeddings

    def get_centroid_pairs(self):
        """Return centroid (input, output) pairs for replay."""
        return self.cluster_centroids, self.centroid_targets

    def compute_targets_batched(self, query_embeddings, batch_size=256):
        """For compatibility - just returns stored targets."""
        # Assumes query_embeddings matches stored queries
        return self.target_embeddings[:len(query_embeddings)]


class DirectTeacher:
    """
    Simplest teacher: uses nearest cluster's W matrix directly.

    No interpolation, no logm/expm, no Rodrigues - just W @ query.
    This is O(N²) per query, the absolute fastest possible.
    """

    def __init__(self, federated_model_path: str, top_k: int = 1):
        """
        Args:
            federated_model_path: Path to federated .pkl model
            top_k: Number of clusters to blend (1 = hard routing)
        """
        model_path = Path(federated_model_path)
        with open(model_path, 'rb') as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.temperature = self.meta.get("temperature", 0.1)
        self.num_clusters = len(self.cluster_ids)
        self.cluster_centroids = self.meta.get("cluster_centroids")
        self.top_k = top_k

        # Load cluster W matrices
        cluster_dir = model_path.with_suffix('')
        self.W_matrices = {}
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

                self.W_matrices[cid] = W.astype(np.float32)

        # Build W tensor for fast indexing
        self.W_tensor = np.stack([self.W_matrices[cid] for cid in self.cluster_ids], axis=0)

        # Load routing data
        routing_path = cluster_dir / "routing_data.npz"
        if routing_path.exists():
            routing_data = np.load(routing_path)
            self.query_embeddings = routing_data['query_embeddings']
        else:
            self.query_embeddings = None

        log_info(f"DirectTeacher: {len(self.W_matrices)} clusters, top_k={top_k}, d={self.d}")

    def compute_targets_batched(
        self,
        query_embeddings: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Compute targets using direct W @ query (no interpolation).

        Args:
            query_embeddings: (N, d) query embeddings
            batch_size: Batch size for processing

        Returns:
            outputs: (N, d) target outputs
        """
        n_samples = len(query_embeddings)
        outputs = np.zeros_like(query_embeddings)

        # Normalize queries
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        q_normalized = query_embeddings / (q_norms + 1e-8)

        start_time = time.time()

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_q = q_normalized[batch_start:batch_end]
            batch_orig = query_embeddings[batch_start:batch_end]

            # Compute similarities to cluster centroids
            sims = batch_q @ self.cluster_centroids.T  # (batch, n_clusters)

            if self.top_k == 1:
                # Hard routing: just use nearest cluster
                nearest = np.argmax(sims, axis=1)
                for i, (q, idx) in enumerate(zip(batch_orig, nearest)):
                    outputs[batch_start + i] = self.W_tensor[idx] @ q
            else:
                # Soft routing: weighted average of top-k W matrices
                top_k_idx = np.argsort(sims, axis=1)[:, -self.top_k:]

                # Softmax weights
                top_k_sims = np.take_along_axis(sims, top_k_idx, axis=1)
                top_k_sims_shifted = (top_k_sims - top_k_sims.max(axis=1, keepdims=True)) / self.temperature
                weights = np.exp(top_k_sims_shifted)
                weights /= weights.sum(axis=1, keepdims=True)

                for i in range(len(batch_orig)):
                    q = batch_orig[i]
                    blended = np.zeros(self.d, dtype=np.float32)
                    for j, (idx, w) in enumerate(zip(top_k_idx[i], weights[i])):
                        blended += w * (self.W_tensor[idx] @ q)
                    outputs[batch_start + i] = blended

            if (batch_end) % 2000 == 0 or batch_end == n_samples:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                log_info(f"    DirectTeacher: {batch_end}/{n_samples} ({rate:.0f}/s)")

        elapsed = time.time() - start_time
        log_info(f"  DirectTeacher targets: {elapsed:.2f}s ({n_samples/elapsed:.0f}/s)")

        return outputs


class RotationalTeacher:
    """
    GPU-accelerated teacher using full rotational blending (logm/expm).

    Key optimization: Pre-compute logm(W) for all clusters at init (one-time cost),
    then use GPU matrix_exp for fast batched inference.

    Performance: ~1000+/s on GPU vs ~0.2/s without pre-computation.
    """

    def __init__(self, federated_model_path: str, top_k_routing: int = 10, use_gpu: bool = True):
        """
        Args:
            federated_model_path: Path to federated .pkl model
            top_k_routing: Number of training queries for routing
            use_gpu: Whether to use GPU acceleration (requires PyTorch)
        """
        from scipy.linalg import logm as scipy_logm
        self.scipy_logm = scipy_logm
        self.top_k_routing = top_k_routing
        self.use_gpu = use_gpu

        self.model_path = federated_model_path
        self._load_federated_model(federated_model_path)
        self._precompute_bivectors()  # Key optimization - caches to disk!
        self._setup_gpu()

        log_info(
            f"RotationalTeacher: {self.num_clusters} clusters, "
            f"top_k_routing={top_k_routing}, GPU={'enabled' if self.device != 'cpu' else 'disabled'}"
        )

    def _load_federated_model(self, model_path: str):
        """Load federated model with W matrices."""
        model_path = Path(model_path)
        with open(model_path, 'rb') as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.temperature = self.meta.get("temperature", 0.1)
        self.num_clusters = len(self.cluster_ids)

        # Load routing data
        cluster_dir = model_path.with_suffix('')
        routing_path = cluster_dir / "routing_data.npz"
        if routing_path.exists():
            routing_data = np.load(routing_path)
            self.query_embeddings = routing_data['query_embeddings']
            keys = routing_data['idx_to_cluster_keys']
            values = routing_data['idx_to_cluster_values']
            self.idx_to_cluster = {int(k): str(v) for k, v in zip(keys, values)}
        else:
            raise FileNotFoundError(f"Routing data not found: {routing_path}")

        # Load W matrices from each cluster
        self.clusters = {}
        for cid in self.cluster_ids:
            cluster_path = cluster_dir / f"{cid}.npz"
            if cluster_path.exists():
                data = np.load(cluster_path)
                self.clusters[cid] = {"W": data["W_stack"][0]}

        self.d = self.query_embeddings.shape[1]
        log_info(f"  Loaded {len(self.clusters)} cluster W matrices")

    def _precompute_bivectors(self):
        """Pre-compute logm(W) for all clusters - caches to disk for reuse!"""
        # Try to load cached bivectors first
        cache_path = Path(self.model_path).with_suffix('.bivectors.npz')
        if cache_path.exists():
            log_info(f"  Loading cached cluster bivectors from {cache_path}...")
            cached = np.load(cache_path, allow_pickle=True)
            self.bivectors = cached['bivectors'].astype(np.float32)
            self.cid_to_idx = dict(cached['cid_to_idx'].item())
            self.idx_to_cid = {int(k): v for k, v in cached['idx_to_cid'].item().items()}
            log_info(f"  Loaded {len(self.bivectors)} cached bivectors")

            # Create query mapping
            self.query_to_bivector_idx = np.zeros(len(self.query_embeddings), dtype=np.int32)
            for idx in range(len(self.query_embeddings)):
                cid = self.idx_to_cluster.get(idx)
                if cid and cid in self.cid_to_idx:
                    self.query_to_bivector_idx[idx] = self.cid_to_idx[cid]
            return

        log_info(f"  Pre-computing cluster bivectors (will cache to {cache_path.name})...")
        start = time.time()

        # Create ordered list of cluster IDs for indexing
        self.cid_to_idx = {}
        self.idx_to_cid = {}
        bivectors = []

        for i, cid in enumerate(self.cluster_ids):
            if cid not in self.clusters:
                continue
            self.cid_to_idx[cid] = len(bivectors)
            self.idx_to_cid[len(bivectors)] = cid

            W = self.clusters[cid]["W"].astype(np.float64)
            # Ensure W is a proper rotation (det = 1)
            if np.linalg.det(W) < 0:
                W = W.copy()
                W[:, -1] *= -1

            try:
                bivector = np.real(self.scipy_logm(W))
            except Exception:
                bivector = np.zeros((self.d, self.d), dtype=np.float64)

            bivectors.append(bivector)

            if (len(bivectors)) % 20 == 0:
                log_info(f"    logm: {len(bivectors)}/{self.num_clusters}")

        self.bivectors = np.array(bivectors, dtype=np.float32)  # (n_clusters, d, d)
        elapsed = time.time() - start
        log_info(f"  Pre-computed {len(bivectors)} bivectors in {elapsed:.1f}s")

        # Save to cache
        np.savez_compressed(
            cache_path,
            bivectors=self.bivectors,
            cid_to_idx=self.cid_to_idx,
            idx_to_cid=self.idx_to_cid,
        )
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        log_info(f"  Saved bivector cache to {cache_path} ({size_mb:.1f} MB)")

        # Create mapping from training query index to cluster bivector index
        self.query_to_bivector_idx = np.zeros(len(self.query_embeddings), dtype=np.int32)
        for idx in range(len(self.query_embeddings)):
            cid = self.idx_to_cluster.get(idx)
            if cid and cid in self.cid_to_idx:
                self.query_to_bivector_idx[idx] = self.cid_to_idx[cid]
            else:
                self.query_to_bivector_idx[idx] = 0  # Fallback

    def _setup_gpu(self):
        """Setup GPU tensors if available."""
        torch, _, _ = _import_torch()
        if torch is None or not self.use_gpu:
            self.device = 'cpu'
            self.torch = None
            return

        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Move data to GPU
        self.bivectors_gpu = torch.from_numpy(self.bivectors).to(self.device)
        self.query_embeddings_gpu = torch.from_numpy(
            self.query_embeddings.astype(np.float32)
        ).to(self.device)
        self.query_to_bivector_idx_gpu = torch.from_numpy(
            self.query_to_bivector_idx
        ).to(self.device)

    def compute_targets_batched(
        self,
        query_embeddings: np.ndarray,
        batch_size: int = 256,
        **kwargs
    ) -> np.ndarray:
        """
        Compute target outputs using GPU-accelerated rotational blending.

        Args:
            query_embeddings: (N, d) query embeddings
            batch_size: Batch size for GPU processing

        Returns:
            outputs: (N, d) target outputs after rotation
        """
        if self.torch is None or self.device == 'cpu':
            return self._compute_targets_cpu(query_embeddings)

        return self._compute_targets_gpu(query_embeddings, batch_size)

    def _compute_targets_gpu(self, query_embeddings: np.ndarray, batch_size: int) -> np.ndarray:
        """GPU-accelerated target computation."""
        torch = self.torch
        n_samples = len(query_embeddings)
        all_outputs = np.zeros_like(query_embeddings)

        queries_gpu = torch.from_numpy(query_embeddings.astype(np.float32)).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_queries = queries_gpu[batch_start:batch_end]  # (B, d)
                B = len(batch_queries)

                # Compute similarities to all training queries: (B, N_train)
                sims = batch_queries @ self.query_embeddings_gpu.T

                # Softmax with temperature
                sims_shifted = (sims - sims.max(dim=1, keepdim=True).values) / self.temperature
                weights = torch.softmax(sims_shifted, dim=1)  # (B, N_train)

                # Get top-k indices for each query
                _, top_indices = torch.topk(weights, self.top_k_routing, dim=1)  # (B, K)

                # Gather weights for top-k
                top_weights = torch.gather(weights, 1, top_indices)  # (B, K)
                top_weights = top_weights / top_weights.sum(dim=1, keepdim=True)  # Normalize

                # Get bivector indices for top-k training queries
                top_bivector_idx = self.query_to_bivector_idx_gpu[top_indices]  # (B, K)

                # Blend bivectors for each query
                blended = torch.zeros(B, self.d, self.d, device=self.device)
                for k in range(self.top_k_routing):
                    biv_idx = top_bivector_idx[:, k]  # (B,)
                    w = top_weights[:, k:k+1, None]  # (B, 1, 1)
                    blended += w * self.bivectors_gpu[biv_idx]  # (B, d, d)

                # Apply matrix exponential (GPU-accelerated)
                W_blended = torch.linalg.matrix_exp(blended)  # (B, d, d)

                # Apply rotation: q @ W
                batch_out = torch.bmm(batch_queries.unsqueeze(1), W_blended).squeeze(1)
                all_outputs[batch_start:batch_end] = batch_out.cpu().numpy()

                if (batch_end) % 1000 == 0 or batch_end == n_samples:
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed
                    eta = (n_samples - batch_end) / rate if rate > 0 else 0
                    log_info(f"    GPU rotational: {batch_end}/{n_samples} ({rate:.0f}/s, ETA: {eta:.0f}s)")

        elapsed = time.time() - start_time
        log_info(f"  GPU rotational computation: {elapsed:.1f}s ({n_samples/elapsed:.0f}/s)")
        return all_outputs

    def _compute_targets_cpu(self, query_embeddings: np.ndarray) -> np.ndarray:
        """CPU fallback (slower)."""
        from scipy.linalg import expm as scipy_expm

        n_samples = len(query_embeddings)
        all_outputs = np.zeros_like(query_embeddings)

        start_time = time.time()
        for i, q_emb in enumerate(query_embeddings):
            # Compute similarities
            sims = q_emb @ self.query_embeddings.T
            sims_shifted = (sims - np.max(sims)) / self.temperature
            weights = np.exp(sims_shifted)
            weights /= weights.sum()

            # Get top-k
            top_indices = np.argsort(weights)[-self.top_k_routing:]
            top_weights = weights[top_indices]
            top_weights /= top_weights.sum()

            # Blend pre-computed bivectors
            blended = np.zeros((self.d, self.d), dtype=np.float32)
            for k, idx in enumerate(top_indices):
                biv_idx = self.query_to_bivector_idx[idx]
                blended += top_weights[k] * self.bivectors[biv_idx]

            # Apply matrix exp
            W_blended = scipy_expm(blended.astype(np.float64)).astype(np.float32)
            all_outputs[i] = q_emb @ W_blended

            if (i + 1) % 100 == 0 or i + 1 == n_samples:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate if rate > 0 else 0
                log_info(f"    CPU rotational: {i+1}/{n_samples} ({rate:.1f}/s, ETA: {eta:.0f}s)")

        elapsed = time.time() - start_time
        log_info(f"  CPU rotational computation: {elapsed:.1f}s ({n_samples/elapsed:.1f}/s)")
        return all_outputs


# =============================================================================
# JAX-Accelerated Rotational Teacher
# =============================================================================

def _import_jax():
    """Import JAX lazily."""
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.linalg import expm as jax_expm
        return jax, jnp, jax_expm
    except ImportError:
        return None, None, None


class JaxRotationalTeacher:
    """
    JAX-accelerated teacher using full rotational blending (logm/expm).

    Uses JAX's JIT compilation and vmap for efficient batched matrix exponential.
    Can be significantly faster than PyTorch on both CPU and GPU due to XLA optimization.

    Key features:
    - JIT-compiled matrix exponential
    - Vectorized batch processing with vmap
    - Compatible with the same bivector cache as RotationalTeacher
    """

    def __init__(self, federated_model_path: str, top_k_routing: int = 10, use_gpu: bool = True):
        """
        Args:
            federated_model_path: Path to federated .pkl model
            top_k_routing: Number of training queries for routing
            use_gpu: Whether to try GPU (JAX will use available accelerators)
        """
        jax, jnp, jax_expm = _import_jax()
        if jax is None:
            raise ImportError("JAX is required for JaxRotationalTeacher. Install with: pip install jax jaxlib")

        from scipy.linalg import logm as scipy_logm
        self.scipy_logm = scipy_logm
        self.jax = jax
        self.jnp = jnp
        self.jax_expm = jax_expm
        self.top_k_routing = top_k_routing

        self.model_path = federated_model_path
        self._load_federated_model(federated_model_path)
        self._precompute_bivectors()  # Reuses same cache as RotationalTeacher
        self._setup_jax_functions()

        # Report available devices
        devices = jax.devices()
        device_types = [str(d.device_kind) for d in devices]
        log_info(
            f"JaxRotationalTeacher: {self.num_clusters} clusters, "
            f"top_k_routing={top_k_routing}, devices={device_types}"
        )

    def _load_federated_model(self, model_path: str):
        """Load federated model with W matrices."""
        model_path = Path(model_path)
        with open(model_path, 'rb') as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.temperature = self.meta.get("temperature", 0.1)
        self.num_clusters = len(self.cluster_ids)

        # Load routing data
        cluster_dir = model_path.with_suffix('')
        routing_path = cluster_dir / "routing_data.npz"
        if routing_path.exists():
            routing_data = np.load(routing_path)
            self.query_embeddings = routing_data['query_embeddings']
            keys = routing_data['idx_to_cluster_keys']
            values = routing_data['idx_to_cluster_values']
            self.idx_to_cluster = {int(k): str(v) for k, v in zip(keys, values)}
        else:
            raise FileNotFoundError(f"Routing data not found: {routing_path}")

        # Load W matrices from each cluster
        self.clusters = {}
        for cid in self.cluster_ids:
            cluster_path = cluster_dir / f"{cid}.npz"
            if cluster_path.exists():
                data = np.load(cluster_path)
                self.clusters[cid] = {"W": data["W_stack"][0]}

        self.d = self.query_embeddings.shape[1]
        log_info(f"  Loaded {len(self.clusters)} cluster W matrices")

    def _precompute_bivectors(self):
        """Pre-compute logm(W) for all clusters - reuses same cache as RotationalTeacher!"""
        # Try to load cached bivectors first
        cache_path = Path(self.model_path).with_suffix('.bivectors.npz')
        if cache_path.exists():
            log_info(f"  Loading cached cluster bivectors from {cache_path}...")
            cached = np.load(cache_path, allow_pickle=True)
            self.bivectors = cached['bivectors'].astype(np.float32)
            self.cid_to_idx = dict(cached['cid_to_idx'].item())
            self.idx_to_cid = {int(k): v for k, v in cached['idx_to_cid'].item().items()}
            log_info(f"  Loaded {len(self.bivectors)} cached bivectors")

            # Create query mapping
            self.query_to_bivector_idx = np.zeros(len(self.query_embeddings), dtype=np.int32)
            for idx in range(len(self.query_embeddings)):
                cid = self.idx_to_cluster.get(idx)
                if cid and cid in self.cid_to_idx:
                    self.query_to_bivector_idx[idx] = self.cid_to_idx[cid]
            return

        log_info(f"  Pre-computing cluster bivectors (will cache to {cache_path.name})...")
        start = time.time()

        # Create ordered list of cluster IDs for indexing
        self.cid_to_idx = {}
        self.idx_to_cid = {}
        bivectors = []

        for i, cid in enumerate(self.cluster_ids):
            if cid not in self.clusters:
                continue
            self.cid_to_idx[cid] = len(bivectors)
            self.idx_to_cid[len(bivectors)] = cid

            W = self.clusters[cid]["W"].astype(np.float64)
            # Ensure W is a proper rotation (det = 1)
            if np.linalg.det(W) < 0:
                W = W.copy()
                W[:, -1] *= -1

            try:
                bivector = np.real(self.scipy_logm(W))
            except Exception:
                bivector = np.zeros((self.d, self.d), dtype=np.float64)

            bivectors.append(bivector)

            if (len(bivectors)) % 20 == 0:
                log_info(f"    logm: {len(bivectors)}/{self.num_clusters}")

        self.bivectors = np.array(bivectors, dtype=np.float32)  # (n_clusters, d, d)
        elapsed = time.time() - start
        log_info(f"  Pre-computed {len(bivectors)} bivectors in {elapsed:.1f}s")

        # Save to cache
        np.savez_compressed(
            cache_path,
            bivectors=self.bivectors,
            cid_to_idx=self.cid_to_idx,
            idx_to_cid=self.idx_to_cid,
        )
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        log_info(f"  Saved bivector cache to {cache_path} ({size_mb:.1f} MB)")

        # Create mapping from training query index to cluster bivector index
        self.query_to_bivector_idx = np.zeros(len(self.query_embeddings), dtype=np.int32)
        for idx in range(len(self.query_embeddings)):
            cid = self.idx_to_cluster.get(idx)
            if cid and cid in self.cid_to_idx:
                self.query_to_bivector_idx[idx] = self.cid_to_idx[cid]
            else:
                self.query_to_bivector_idx[idx] = 0  # Fallback

    def _setup_jax_functions(self):
        """Setup JIT-compiled JAX functions."""
        jax = self.jax
        jnp = self.jnp
        jax_expm = self.jax_expm

        # Convert data to JAX arrays
        self.bivectors_jax = jnp.array(self.bivectors)
        self.query_embeddings_jax = jnp.array(self.query_embeddings.astype(np.float32))
        self.query_to_bivector_idx_jax = jnp.array(self.query_to_bivector_idx)

        # JIT-compiled single query processing
        @jax.jit
        def process_single_query(query, query_embs, bivectors, query_to_biv_idx, temp, top_k):
            """Process a single query with rotational blending."""
            # Compute similarities: (N_train,)
            sims = query @ query_embs.T

            # Softmax with temperature
            sims_shifted = (sims - jnp.max(sims)) / temp
            weights = jax.nn.softmax(sims_shifted)

            # Get top-k indices
            top_indices = jnp.argsort(weights)[-top_k:]
            top_weights = weights[top_indices]
            top_weights = top_weights / jnp.sum(top_weights)  # Normalize

            # Get bivector indices for top-k training queries
            top_biv_idx = query_to_biv_idx[top_indices]

            # Blend bivectors
            blended = jnp.zeros((query.shape[0], query.shape[0]), dtype=jnp.float32)
            for k in range(top_k):
                blended = blended + top_weights[k] * bivectors[top_biv_idx[k]]

            # Apply matrix exponential
            W_blended = jax_expm(blended)

            # Apply rotation: q @ W
            return query @ W_blended

        self._process_single = process_single_query

        # Vectorized version using vmap
        @jax.jit
        def process_batch(queries, query_embs, bivectors, query_to_biv_idx, temp, top_k):
            """Process a batch of queries."""
            # vmap over the batch dimension
            vmapped_fn = jax.vmap(
                lambda q: process_single_query(q, query_embs, bivectors, query_to_biv_idx, temp, top_k)
            )
            return vmapped_fn(queries)

        self._process_batch = process_batch

    def compute_targets_batched(
        self,
        query_embeddings: np.ndarray,
        batch_size: int = 256,
        **kwargs
    ) -> np.ndarray:
        """
        Compute target outputs using JAX-accelerated rotational blending.

        Args:
            query_embeddings: (N, d) query embeddings
            batch_size: Batch size for processing

        Returns:
            outputs: (N, d) target outputs after rotation
        """
        jnp = self.jnp
        n_samples = len(query_embeddings)
        all_outputs = np.zeros_like(query_embeddings)

        queries_jax = jnp.array(query_embeddings.astype(np.float32))

        start_time = time.time()

        # Warm up JIT compilation with first batch
        log_info("  JAX JIT compilation (first batch)...")
        first_batch = queries_jax[:min(batch_size, n_samples)]
        _ = self._process_batch(
            first_batch,
            self.query_embeddings_jax,
            self.bivectors_jax,
            self.query_to_bivector_idx_jax,
            self.temperature,
            self.top_k_routing
        )
        self.jax.block_until_ready(_)
        jit_time = time.time() - start_time
        log_info(f"  JIT compilation: {jit_time:.1f}s")

        start_time = time.time()
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_queries = queries_jax[batch_start:batch_end]

            batch_out = self._process_batch(
                batch_queries,
                self.query_embeddings_jax,
                self.bivectors_jax,
                self.query_to_bivector_idx_jax,
                self.temperature,
                self.top_k_routing
            )
            self.jax.block_until_ready(batch_out)
            all_outputs[batch_start:batch_end] = np.array(batch_out)

            if (batch_end) % 1000 == 0 or batch_end == n_samples:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                eta = (n_samples - batch_end) / rate if rate > 0 else 0
                log_info(f"    JAX rotational: {batch_end}/{n_samples} ({rate:.0f}/s, ETA: {eta:.0f}s)")

        elapsed = time.time() - start_time
        log_info(f"  JAX rotational computation: {elapsed:.1f}s ({n_samples/elapsed:.0f}/s)")
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
    precomputed_targets: Optional[np.ndarray] = None,
    centroid_queries: Optional[np.ndarray] = None,
    centroid_targets: Optional[np.ndarray] = None,
    centroid_warmup_epochs: int = 0,
) -> List[float]:
    """
    Train FastOrthogonalTransformer with optional curriculum learning.

    Args:
        transformer: FastOrthogonalTransformer to train
        teacher: OrthogonalTeacher for target outputs
        query_embeddings: Training queries (N, d)
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_interval: Log every N epochs
        top_k: Top-K for teacher routing
        precomputed_targets: Optional pre-computed teacher targets (skips slow computation)
        centroid_queries: Optional cluster centroids for curriculum warmup (n_clusters, d)
        centroid_targets: Optional W @ centroid targets for warmup (n_clusters, d)
        centroid_warmup_epochs: Number of epochs to train on centroids before full data

    Returns:
        List of loss values per epoch
    """
    torch, nn, F = _import_torch()

    n_samples = len(query_embeddings)

    # Use pre-computed targets or compute them
    if precomputed_targets is not None:
        log_info(f"Using {len(precomputed_targets)} pre-computed teacher targets")
        target_outputs = precomputed_targets
    else:
        log_info("Computing target outputs from teacher...")
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

    # Phase 1: Curriculum warmup on centroids (fast, clean signal)
    if centroid_queries is not None and centroid_targets is not None and centroid_warmup_epochs > 0:
        log_info(f"\nPhase 1: Centroid warmup ({centroid_warmup_epochs} epochs, {len(centroid_queries)} centroids)")
        centroid_queries_t = torch.from_numpy(centroid_queries).float().to(transformer.device)
        centroid_targets_t = torch.from_numpy(centroid_targets).float().to(transformer.device)
        n_centroids = len(centroid_queries)

        for warmup_epoch in range(centroid_warmup_epochs):
            perm = torch.randperm(n_centroids)
            epoch_loss = 0.0
            epoch_angle = 0.0
            n_batches = 0

            for i in range(0, n_centroids, batch_size):
                idx = perm[i:i+batch_size]
                batch_q = centroid_queries_t[idx]
                batch_t = centroid_targets_t[idx]

                predicted, _, _, _ = transformer.forward(batch_q)

                # Pure angle loss for warmup (get direction right)
                pred_norm = F.normalize(predicted, dim=1)
                target_norm = F.normalize(batch_t, dim=1)
                cosine_sim = (pred_norm * target_norm).sum(dim=1)
                angle_error = torch.acos(torch.clamp(cosine_sim, -1 + 1e-6, 1 - 1e-6))
                loss = angle_error.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_angle += (1 - angle_error.mean().item())  # Convert to similarity
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_angle_sim = epoch_angle / n_batches
            log_info(f"  Warmup {warmup_epoch+1}/{centroid_warmup_epochs}: "
                    f"angle_loss={avg_loss:.4f}, direction_sim={avg_angle_sim:.4f}")
            losses.append(avg_loss)

        log_info(f"Phase 1 complete. Starting Phase 2 on full data.\n")

    # Phase 2: Full training on all queries with centroid replay
    log_info(f"Training for {num_epochs} epochs, {n_samples} samples")

    # Prepare centroid tensors for replay (if available)
    centroid_replay_enabled = (centroid_queries is not None and centroid_targets is not None)
    if centroid_replay_enabled:
        centroid_queries_t = torch.from_numpy(centroid_queries).float().to(transformer.device)
        centroid_targets_t = torch.from_numpy(centroid_targets).float().to(transformer.device)
        n_centroids = len(centroid_queries)
        centroid_replay_interval = max(10, n_samples // (batch_size * 10))  # Replay every ~10% of epoch
        log_info(f"  Centroid replay enabled: {n_centroids} centroids every {centroid_replay_interval} batches")

    training_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_cosine = 0.0
        n_batches = 0
        centroid_perm_idx = 0  # Track position in centroid shuffle

        # Shuffle centroids at start of each epoch
        if centroid_replay_enabled:
            centroid_perm = torch.randperm(n_centroids)

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_queries = queries_tensor[idx]
            batch_targets = outputs_tensor[idx]

            # Forward pass
            predicted, weights, top_k_idx, scale = transformer.forward(batch_queries)

            # Loss: angle-focused early, then add MSE
            pred_norm = F.normalize(predicted, dim=1)
            target_norm = F.normalize(batch_targets, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1)

            # Angle loss: arccos(cos_sim) = actual angle in radians
            # Clamp to avoid NaN from numerical issues
            angle_error = torch.acos(torch.clamp(cosine_sim, -1 + 1e-6, 1 - 1e-6))
            angle_loss = angle_error.mean()

            # MSE loss for magnitude
            mse_loss = F.mse_loss(predicted, batch_targets)

            # Warmup schedule: pure angle loss early, blend in MSE later
            # First 20% of epochs: pure angle loss
            # After that: gradually blend in MSE
            warmup_epochs = max(1, num_epochs // 5)
            if epoch < warmup_epochs:
                mse_weight = 0.0
            else:
                # Linear ramp from 0 to 0.5 over remaining epochs
                progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
                mse_weight = 0.5 * progress

            loss = (1 - mse_weight) * angle_loss + mse_weight * mse_loss
            cosine_sim = cosine_sim.mean()  # for logging

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cosine += cosine_sim.item()
            n_batches += 1

            # Centroid replay: periodically train on centroid pairs
            if centroid_replay_enabled and n_batches % centroid_replay_interval == 0:
                # Get next batch of centroids (cycling through)
                c_start = centroid_perm_idx
                c_end = min(c_start + batch_size, n_centroids)
                c_idx = centroid_perm[c_start:c_end]
                centroid_perm_idx = c_end if c_end < n_centroids else 0

                if len(c_idx) > 0:
                    c_queries = centroid_queries_t[c_idx]
                    c_targets = centroid_targets_t[c_idx]

                    c_pred, _, _, _ = transformer.forward(c_queries)

                    # Pure angle loss for centroids (strong constraint)
                    c_pred_norm = F.normalize(c_pred, dim=1)
                    c_target_norm = F.normalize(c_targets, dim=1)
                    c_cosine = (c_pred_norm * c_target_norm).sum(dim=1)
                    c_angle_error = torch.acos(torch.clamp(c_cosine, -1 + 1e-6, 1 - 1e-6))
                    c_loss = c_angle_error.mean()

                    optimizer.zero_grad()
                    c_loss.backward()
                    optimizer.step()

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
    precomputed_targets: Optional[np.ndarray] = None,
) -> Dict:
    """Evaluate transformer against teacher on test set."""
    torch, nn, F = _import_torch()
    transformer.eval_mode()

    n_samples = len(test_queries)

    # Use pre-computed targets or compute them
    if precomputed_targets is not None:
        log_info(f"Using {len(precomputed_targets)} pre-computed teacher targets")
        teacher_outputs = precomputed_targets
    else:
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
# Multi-Source Orthogonal Teacher
# =============================================================================

class MultiSourceOrthogonalTeacher:
    """
    Teacher that combines multiple federated models for unified training.

    Supports training a single transformer that can project queries from
    multiple federated models (e.g., skills + Q/A pairs from books).
    """

    def __init__(
        self,
        federated_model_paths: List[str],
        codebook: FastOrthogonalCodebook,
    ):
        """
        Args:
            federated_model_paths: List of paths to federated .pkl models
            codebook: FastOrthogonalCodebook for rotation
        """
        self.codebook = codebook
        self.d = codebook.d
        self.n_components = codebook.n_components

        # Load all federated models
        self.sources = []
        self.all_query_embeddings = []
        self.source_indices = []  # Track which queries belong to which source

        for i, path in enumerate(federated_model_paths):
            source_data = self._load_federated_source(path, i)
            if source_data is not None:
                self.sources.append(source_data)

                # Track queries and their source
                n_queries = len(source_data['query_embeddings'])
                self.all_query_embeddings.append(source_data['query_embeddings'])
                self.source_indices.extend([i] * n_queries)

        # Concatenate all queries
        if self.all_query_embeddings:
            self.query_embeddings = np.concatenate(self.all_query_embeddings, axis=0)
        else:
            self.query_embeddings = np.zeros((0, self.d), dtype=np.float32)

        self.source_indices = np.array(self.source_indices)

        log_info(
            f"MultiSourceOrthogonalTeacher: {len(self.sources)} sources, "
            f"{len(self.query_embeddings)} total queries, "
            f"routing to {self.n_components} orthogonal planes"
        )

    def _load_federated_source(self, model_path: str, source_id: int) -> Optional[Dict]:
        """Load a single federated model source."""
        model_path = Path(model_path)

        try:
            with open(model_path, 'rb') as f:
                meta = pickle.load(f)

            cluster_ids = meta.get("cluster_ids", [])
            temperature = meta.get("temperature", 0.1)
            num_clusters = len(cluster_ids)
            cluster_centroids = meta.get("cluster_centroids")

            # Load routing data
            cluster_dir = model_path.with_suffix('')
            routing_path = cluster_dir / "routing_data.npz"

            if routing_path.exists():
                routing_data = np.load(routing_path)
                query_embeddings = routing_data['query_embeddings'].astype(np.float32)
            else:
                log_info(f"  Warning: No routing data for {model_path.name}")
                return None

            log_info(
                f"  Source {source_id}: {model_path.name} - "
                f"{num_clusters} clusters, {len(query_embeddings)} queries"
            )

            return {
                'path': model_path,
                'cluster_ids': cluster_ids,
                'temperature': temperature,
                'num_clusters': num_clusters,
                'cluster_centroids': cluster_centroids,
                'cluster_dir': cluster_dir,
                'query_embeddings': query_embeddings,
                'source_id': source_id,
            }

        except Exception as e:
            log_info(f"  Warning: Failed to load {model_path}: {e}")
            return None

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


def build_combined_orthogonal_codebook(
    federated_model_paths: List[str],
    n_components: int = 64,
    method: str = "pca",
    pca_algorithm: str = "auto",
) -> FastOrthogonalCodebook:
    """
    Build an orthogonal codebook from multiple federated models.

    Extracts cluster bivectors from all models and builds a combined
    orthogonal codebook.

    Args:
        federated_model_paths: List of paths to federated .pkl models
        n_components: Number of orthogonal planes in codebook
        method: "pca" for PCA-based extraction, "canonical" for axis-aligned

    Returns:
        FastOrthogonalCodebook combining all sources
    """
    from scipy.linalg import logm

    log_info(f"\nBuilding combined orthogonal codebook from {len(federated_model_paths)} sources...")

    all_bivectors = []
    source_info = []
    d = None

    for model_path in federated_model_paths:
        model_path = Path(model_path)

        try:
            with open(model_path, 'rb') as f:
                meta = pickle.load(f)

            cluster_dir = model_path.with_suffix('')

            # Handle two different model formats:
            # 1. New format: cluster_ids list + cluster_X.npz files
            # 2. Old format: clusters dict with W_file references
            cluster_ids = meta.get("cluster_ids", [])
            clusters_dict = meta.get("clusters", {})

            n_loaded = 0

            if cluster_ids:
                # New format with cluster_ids
                for cid in cluster_ids:
                    # Handle both naming conventions
                    if cid.startswith("cluster_"):
                        cluster_path = cluster_dir / f"{cid}.npz"
                    else:
                        cluster_path = cluster_dir / f"cluster_{cid}.npz"

                    if not cluster_path.exists():
                        continue

                    data = np.load(cluster_path)

                    if "W" in data:
                        W = data["W"]
                    elif "W_stack" in data:
                        W = data["W_stack"][0]
                    else:
                        continue

                    if d is None:
                        d = W.shape[0]

                    # Compute bivector (log of rotation)
                    try:
                        A = logm(W)
                        A = np.real(A)
                        A = (A - A.T) / 2  # Ensure antisymmetric
                        all_bivectors.append(A)
                        source_info.append({
                            'source': model_path.name,
                            'cluster': cid,
                        })
                        n_loaded += 1
                    except Exception:
                        pass

            elif clusters_dict:
                # Old format with clusters dict containing W_file references
                for cid, cluster_info in clusters_dict.items():
                    w_file = cluster_info.get("W_file", "")
                    if not w_file:
                        continue

                    cluster_path = cluster_dir / w_file
                    if not cluster_path.exists():
                        continue

                    data = np.load(cluster_path)

                    if "W" in data:
                        W = data["W"]
                    elif "W_stack" in data:
                        W = data["W_stack"][0]
                    else:
                        continue

                    if d is None:
                        d = W.shape[0]

                    # Compute bivector (log of rotation)
                    try:
                        A = logm(W)
                        A = np.real(A)
                        A = (A - A.T) / 2  # Ensure antisymmetric
                        all_bivectors.append(A)
                        source_info.append({
                            'source': model_path.name,
                            'cluster': cid,
                        })
                        n_loaded += 1
                    except Exception:
                        pass

            log_info(f"  {model_path.name}: {n_loaded} cluster bivectors")

        except Exception as e:
            log_info(f"  Warning: Failed to load {model_path}: {e}")

    if not all_bivectors:
        raise ValueError("No bivectors loaded from any source")

    all_bivectors = np.array(all_bivectors)
    log_info(f"  Total bivectors: {len(all_bivectors)}, d={d}")

    if method == "pca":
        # Use PCA to find dominant rotation directions
        log_info("  Extracting dominant planes via PCA...")

        # Flatten bivectors for PCA - use float32 to save memory
        n_bivectors = len(all_bivectors)
        flat_bivectors = all_bivectors.reshape(n_bivectors, -1).astype(np.float32)

        # Center
        mean_flat = flat_bivectors.mean(axis=0)
        centered = flat_bivectors - mean_flat

        n_pca = min(n_components * 2, n_bivectors)

        # Auto-select algorithm based on matrix dimensions
        # Key insight: For n_samples << n_features, numpy SVD with full_matrices=False
        # is efficient because it only computes n_samples singular values.
        # TruncatedSVD can be slower when n_components is close to n_samples.
        n_features = d * d  # 768² = 589,824 for nomic embeddings
        algo = pca_algorithm

        if algo == "auto":
            # When n_samples is small (<500), numpy SVD is efficient even with many features
            # because full_matrices=False only computes min(n_samples, n_features) singular values
            if n_bivectors < 500:
                algo = "full"
                log_info(f"  Auto: {n_bivectors} samples × {n_features} features, using full SVD (efficient for small n)")
            elif n_bivectors > 2000:
                algo = "incremental"
                log_info(f"  Auto: {n_bivectors} samples × {n_features} features, using incremental PCA")
            else:
                algo = "randomized"
                log_info(f"  Auto: {n_bivectors} samples × {n_features} features, using randomized SVD")

        if algo == "full":
            log_info(f"  Using full SVD for {n_pca} components...")
            from numpy.linalg import svd
            U, S, Vt = svd(centered, full_matrices=False)
            top_components = Vt[:n_pca]
            explained_var = np.sum(S[:n_pca]**2) / np.sum(S**2)
            log_info(f"  Explained variance ratio: {explained_var:.4f}")

        elif algo == "randomized":
            log_info(f"  Using randomized SVD for {n_pca} components...")
            try:
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=n_pca, random_state=42, algorithm='randomized')
                svd.fit(centered)
                top_components = svd.components_
                log_info(f"  Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
            except MemoryError:
                log_info("  Memory limit hit, falling back to incremental PCA...")
                algo = "incremental"

        if algo == "incremental":
            log_info(f"  Using incremental PCA for {n_pca} components...")
            from sklearn.decomposition import IncrementalPCA
            batch_size = max(10, min(50, n_bivectors // 4))
            ipca = IncrementalPCA(n_components=n_pca, batch_size=batch_size)
            ipca.fit(centered)
            top_components = ipca.components_
            log_info(f"  Explained variance ratio: {ipca.explained_variance_ratio_.sum():.4f}")

        # Free memory
        del centered, flat_bivectors

        # Reconstruct as bivectors
        pca_bivectors = []
        for i in range(len(top_components)):
            B = top_components[i].reshape(d, d)
            B = (B - B.T) / 2  # Ensure antisymmetric
            norm = np.linalg.norm(B)
            if norm > 1e-8:
                B = B / norm
                pca_bivectors.append(B)

        pca_bivectors = np.array(pca_bivectors[:n_components])
        log_info(f"  PCA components: {len(pca_bivectors)}")

        # Orthogonalize
        orth_bivectors, planes = orthogonalize_codebook(pca_bivectors)

    else:  # canonical
        log_info("  Building canonical orthogonal planes...")
        orth_bivectors, planes = build_canonical_orthogonal_codebook(d, n_components)

    # Create codebook
    codebook = FastOrthogonalCodebook(orth_bivectors)

    log_info(f"  Created codebook: {codebook.n_components} orthogonal planes")

    return codebook


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
    k_values: List[int] = [1, 5, 10, 20, 50],
    top_k_routing: int = 10,
) -> Dict:
    """
    Evaluate hit@K for RAG retrieval.

    Tests two modes:
    1. Symmetric: Project both queries and targets (what the old code did)
    2. Asymmetric: Project queries only, targets stay raw (actual RAG use case)

    The asymmetric mode is the realistic evaluation - queries are transformed
    to move closer to their answer embeddings in cosine space.
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

    def compute_hit_rates(proj_queries, targets_to_use, k_values, n_samples):
        """Compute hit@K rates for given projected queries and targets."""
        # Normalize for cosine similarity
        proj_queries = proj_queries / (np.linalg.norm(proj_queries, axis=1, keepdims=True) + 1e-8)
        targets_norm = targets_to_use / (np.linalg.norm(targets_to_use, axis=1, keepdims=True) + 1e-8)

        # Compute all pairwise similarities
        similarities = proj_queries @ targets_norm.T  # (n_samples, n_samples)

        # For each query, rank targets and check hit@K
        hits = {k: 0 for k in k_values}

        for i in range(n_samples):
            sims = similarities[i]
            ranked_indices = np.argsort(sims)[::-1]
            correct_rank = np.where(ranked_indices == i)[0][0]

            for k in k_values:
                if correct_rank < k:
                    hits[k] += 1

        return {k: hits[k] / n_samples for k in k_values}

    results = {}

    # Evaluate each approach in ASYMMETRIC mode (project queries only - realistic RAG)
    log_info("\n  === Asymmetric Mode (project queries only) ===")
    for approach_name, project_fn in [
        ("raw", lambda x: x),  # No projection (raw embeddings)
        ("orthogonal", lambda x: codebook.route_and_apply(x, top_k=top_k_routing)[0]),
        ("baseline", lambda x: baseline.compute_outputs_batched(x, top_k=top_k_routing)),
    ]:
        log_info(f"  Evaluating {approach_name}...")

        t0 = time.time()
        proj_queries = project_fn(queries)
        proj_time = time.time() - t0

        # Asymmetric: queries projected, targets raw
        hit_rates = compute_hit_rates(proj_queries, targets, k_values, n_samples)

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


def save_orthogonal_transformer(
    transformer: FastOrthogonalTransformer,
    codebook: FastOrthogonalCodebook,
    path: str,
    metadata: Optional[Dict] = None,
):
    """
    Save trained orthogonal transformer model.

    Saves both the transformer state dict and the codebook together.
    """
    torch, _, _ = _import_torch()
    if torch is None:
        raise ImportError("PyTorch required to save transformer")

    save_dict = {
        # Transformer state
        'input_proj_state': transformer.input_proj.state_dict(),
        'encoder_state': transformer.encoder.state_dict(),
        'routing_head_state': transformer.routing_head.state_dict(),
        'scale_head_state': transformer.scale_head.state_dict(),

        # Codebook data
        'codebook_bivectors': codebook.bivectors,
        'codebook_plane_u': codebook.plane_u,
        'codebook_plane_v': codebook.plane_v,
        'codebook_plane_theta': codebook.plane_theta,
        'codebook_keys': codebook.codebook_keys,

        # Architecture config
        'embed_dim': transformer.embed_dim,
        'n_components': transformer.n_components,
        'num_layers': transformer.num_layers,
        'num_heads': transformer.num_heads,
        'top_k': transformer.top_k,

        # Optional metadata
        'metadata': metadata or {},
    }

    torch.save(save_dict, path)
    log_info(f"Saved orthogonal transformer to {path}")


def load_orthogonal_transformer(path: str) -> Tuple[FastOrthogonalTransformer, FastOrthogonalCodebook]:
    """
    Load trained orthogonal transformer model.

    Returns both the transformer and the codebook.
    """
    torch, _, _ = _import_torch()
    if torch is None:
        raise ImportError("PyTorch required to load transformer")

    save_dict = torch.load(path, map_location='cpu')

    # Reconstruct codebook
    codebook = FastOrthogonalCodebook.__new__(FastOrthogonalCodebook)
    codebook.bivectors = save_dict['codebook_bivectors']
    codebook.plane_u = save_dict['codebook_plane_u']
    codebook.plane_v = save_dict['codebook_plane_v']
    codebook.plane_theta = save_dict['codebook_plane_theta']
    codebook.codebook_keys = save_dict['codebook_keys']
    codebook.d = save_dict['embed_dim']
    codebook.n_components = save_dict['n_components']
    codebook.planes = [(codebook.plane_u[i], codebook.plane_v[i], codebook.plane_theta[i])
                       for i in range(codebook.n_components)]

    # Reconstruct transformer
    transformer = FastOrthogonalTransformer(
        codebook=codebook,
        num_layers=save_dict['num_layers'],
        num_heads=save_dict['num_heads'],
        top_k=save_dict['top_k'],
    )

    # Load state dicts
    transformer.input_proj.load_state_dict(save_dict['input_proj_state'])
    transformer.encoder.load_state_dict(save_dict['encoder_state'])
    transformer.routing_head.load_state_dict(save_dict['routing_head_state'])
    transformer.scale_head.load_state_dict(save_dict['scale_head_state'])

    log_info(f"Loaded orthogonal transformer: {codebook.n_components} planes, {save_dict['num_layers']} layers")

    return transformer, codebook


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
    parser.add_argument("--federated-models", nargs="+",
                       help="Multiple federated models for multi-source training")

    # Operations
    parser.add_argument("--orthogonalize", action="store_true",
                       help="Orthogonalize existing codebook")
    parser.add_argument("--build-canonical", action="store_true",
                       help="Build canonical orthogonal codebook from scratch")
    parser.add_argument("--build-combined", action="store_true",
                       help="Build combined orthogonal codebook from multiple federated models")
    parser.add_argument("--test", action="store_true",
                       help="Test codebook with sample data")
    parser.add_argument("--train", action="store_true",
                       help="Train FastOrthogonalTransformer")
    parser.add_argument("--train-multisource", action="store_true",
                       help="Train on multiple federated models")
    parser.add_argument("--validate", action="store_true",
                       help="Validate orthogonal codebook against full rotation manifold")
    parser.add_argument("--compare", action="store_true",
                       help="Compare orthogonal vs full rotation vs weighted baseline")
    parser.add_argument("--hit-at-k", action="store_true",
                       help="Evaluate hit@K for RAG retrieval")
    parser.add_argument("--validate-samples", type=int, default=500,
                       help="Number of samples for validation/comparison")
    parser.add_argument("--codebook-method", choices=["pca", "canonical"], default="pca",
                       help="Method for building combined codebook")
    parser.add_argument("--pca-algorithm", choices=["auto", "full", "randomized", "incremental"],
                       default="auto",
                       help="PCA algorithm: auto (choose based on size), full (numpy SVD), "
                            "randomized (sklearn TruncatedSVD), incremental (sklearn IncrementalPCA)")

    # Parameters
    parser.add_argument("--n-components", type=int, default=64,
                       help="Number of codebook entries")
    parser.add_argument("--top-k", type=int, default=8,
                       help="Top-K entries to blend")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--centroid-warmup", type=int, default=0,
                       help="Epochs to warmup on cluster centroids before full training (curriculum learning)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--transformer-type", choices=["additive", "composed", "bivector-blend"],
                       default="additive",
                       help="Transformer type: additive (fast, good hit@k), composed (rotation composition), "
                            "bivector-blend (full bivector blending with Rodrigues)")
    parser.add_argument("--n-basis", type=int, default=128,
                       help="Number of basis bivectors for bivector-blend transformer")
    parser.add_argument("--bias-mode", choices=["none", "shared", "separate"], default="shared",
                       help="Bias mode: none (rotation only), shared (same weights as rotation, recommended), "
                            "separate (independent attention for bias)")
    parser.add_argument("--teacher", choices=["orthogonal", "rotational", "jax", "direct", "paired"], default="orthogonal",
                       help="Teacher type: orthogonal (fast Rodrigues), rotational (PyTorch logm/expm), "
                            "jax (JAX logm/expm), direct (raw W@query), paired (pre-stored pairs, fastest)")
    parser.add_argument("--top-k-routing", type=int, default=10,
                       help="Top-K routing queries for rotational teacher")
    parser.add_argument("--target-cache",
                       help="Path to cache/load teacher targets (~32MB). Saves hours on rotational teacher.")

    # Checkpoint and resumption
    parser.add_argument("--checkpoint-path",
                       help="Path to save training checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume",
                       help="Path to checkpoint to resume training from")

    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                       help="Stop training if no improvement for N epochs (0 = disabled)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001,
                       help="Minimum improvement threshold for early stopping")

    # Learning rate scheduling
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "plateau", "step"],
                       default="none",
                       help="Learning rate scheduler")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0,
                       help="Number of warmup epochs for LR scheduler")

    # Output
    parser.add_argument("--output", help="Path to save orthogonal codebook")
    parser.add_argument("--save-transformer", help="Path to save trained transformer model (.pt)")

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

    elif args.build_canonical and not args.train:
        # Build canonical codebook (standalone mode, not combined with --train)
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

    elif args.build_combined:
        # Build combined codebook from multiple federated models
        if not args.federated_models:
            log_info("ERROR: --federated-models required for --build-combined")
            return 1

        codebook = build_combined_orthogonal_codebook(
            federated_model_paths=args.federated_models,
            n_components=args.n_components,
            method=args.codebook_method,
            pca_algorithm=args.pca_algorithm,
        )

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

        if not args.orthogonal_codebook and not args.codebook and not args.build_canonical:
            log_info("ERROR: --orthogonal-codebook, --codebook, or --build-canonical required for training")
            return 1

        # Load or create orthogonal codebook
        if args.build_canonical:
            # Build canonical codebook on the fly
            d = 768  # Default embedding dim
            with open(args.federated_model, 'rb') as f:
                meta = pickle.load(f)
            d = meta.get('embed_dim', 768)

            log_info(f"\nBuilding canonical orthogonal codebook: {args.n_components} planes in {d}D")
            bivectors, planes = build_canonical_orthogonal_codebook(d, args.n_components)
            codebook = FastOrthogonalCodebook(bivectors)
        elif args.orthogonal_codebook:
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
        if args.teacher == "rotational":
            log_info("\nCreating RotationalTeacher (PyTorch logm/expm blending)...")
            teacher = RotationalTeacher(args.federated_model, top_k_routing=args.top_k_routing)
        elif args.teacher == "jax":
            log_info("\nCreating JaxRotationalTeacher (JAX logm/expm blending)...")
            teacher = JaxRotationalTeacher(args.federated_model, top_k_routing=args.top_k_routing)
        elif args.teacher == "direct":
            log_info("\nCreating DirectTeacher (raw W@query, no interpolation)...")
            teacher = DirectTeacher(args.federated_model, top_k=args.top_k_routing)
        elif args.teacher == "paired":
            log_info("\nCreating PairedDataTeacher (pre-stored pairs, zero computation)...")
            teacher = PairedDataTeacher(args.federated_model)
            # Override query_embeddings with the stored ones
            query_embeddings = teacher.query_embeddings
            log_info(f"  Using {len(query_embeddings)} pre-stored pairs")
        else:
            log_info("\nCreating OrthogonalTeacher...")
            teacher = OrthogonalTeacher(args.federated_model, codebook)

        # Create transformer
        basis_bivectors = None
        basis_keys = None

        if args.transformer_type == "bivector-blend":
            log_info(f"\nBuilding orthogonal bivector basis with {args.n_basis} components...")
            basis_bivectors, basis_keys = build_orthogonal_bivector_basis(
                args.federated_model, n_basis=args.n_basis
            )

            # Load cluster biases if bias mode is enabled
            cluster_biases = None
            if args.bias_mode != "none":
                bias_path = Path(args.federated_model).with_suffix('').parent / \
                           (Path(args.federated_model).stem + '_biases.npz')
                if bias_path.exists():
                    bias_data = np.load(bias_path)
                    cluster_biases = bias_data['biases']
                    log_info(f"  Loaded {len(cluster_biases)} cluster biases from {bias_path}")
                else:
                    log_info(f"  Warning: Bias file not found at {bias_path}, computing...")
                    # Compute biases on the fly
                    from scipy.linalg import orthogonal_procrustes
                    federated_path = Path(args.federated_model)
                    cluster_dir = federated_path.with_suffix('')
                    routing_data = np.load(cluster_dir / "routing_data.npz")

                    with open(federated_path, 'rb') as f:
                        meta = pickle.load(f)

                    biases_list = []
                    for cid in meta['cluster_ids']:
                        cpath = cluster_dir / (f"{cid}.npz" if cid.startswith("cluster_") else f"cluster_{cid}.npz")
                        if cpath.exists():
                            cdata = np.load(cpath)
                            indices = cdata['indices']
                            queries = routing_data['query_embeddings'][indices]
                            targets = cdata['target_embeddings']
                            q_mean, t_mean = queries.mean(0), targets.mean(0)
                            R, _ = orthogonal_procrustes(queries - q_mean, targets - t_mean)
                            biases_list.append(t_mean - q_mean @ R)
                        else:
                            biases_list.append(np.zeros(768))
                    cluster_biases = np.stack(biases_list, axis=0).astype(np.float32)
                    log_info(f"  Computed {len(cluster_biases)} cluster biases")

            log_info(f"\nCreating ComposedBivectorTransformer (bias_mode={args.bias_mode})...")
            transformer = ComposedBivectorTransformer(
                basis_bivectors=basis_bivectors,
                basis_keys=basis_keys,
                cluster_biases=cluster_biases,
                bias_mode=args.bias_mode,
                num_layers=args.layers,
                num_heads=args.heads,
            )
        elif args.transformer_type == "composed":
            log_info("\nCreating ComposedRotationTransformer (proper rotation composition)...")
            transformer = ComposedRotationTransformer(
                codebook=codebook,
                num_layers=args.layers,
                num_heads=args.heads,
                top_k=args.top_k,
            )
        else:
            log_info("\nCreating FastOrthogonalTransformer (additive, fast)...")
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

        # Pre-compute ALL teacher targets before splitting (can be cached to save hours)
        all_targets = None
        if args.target_cache:
            cache_path = Path(args.target_cache)
            if cache_path.exists():
                log_info(f"\nLoading cached teacher targets from {cache_path}...")
                cached = np.load(cache_path)
                if 'targets' in cached and len(cached['targets']) == len(query_embeddings):
                    all_targets = cached['targets'].astype(np.float32)
                    log_info(f"  Loaded {len(all_targets)} cached targets")
                else:
                    log_info(f"  Cache mismatch (expected {len(query_embeddings)}, "
                            f"got {len(cached.get('targets', []))}), recomputing...")

        if all_targets is None:
            if isinstance(teacher, PairedDataTeacher):
                # PairedDataTeacher has pre-stored targets - no computation!
                log_info(f"\nUsing pre-stored targets from PairedDataTeacher...")
                _, all_targets = teacher.get_all_pairs()
            elif isinstance(teacher, DirectTeacher):
                log_info(f"\nComputing teacher targets for {len(query_embeddings)} samples...")
                # DirectTeacher doesn't take top_k (it's set in __init__)
                all_targets = teacher.compute_targets_batched(
                    query_embeddings, batch_size=256
                )
            else:
                log_info(f"\nComputing teacher targets for {len(query_embeddings)} samples...")
                all_targets = teacher.compute_targets_batched(
                    query_embeddings, top_k=args.top_k, batch_size=256
                )

            # Save to cache
            if args.target_cache:
                cache_path = Path(args.target_cache)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, targets=all_targets)
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                log_info(f"  Saved targets to {cache_path} ({size_mb:.1f} MB)")

        # Split train/test (both queries AND targets together)
        n_test = int(len(query_embeddings) * args.test_split)
        indices = np.random.permutation(len(query_embeddings))
        train_idx = indices[:-n_test]
        test_idx = indices[-n_test:]

        train_queries = query_embeddings[train_idx]
        test_queries = query_embeddings[test_idx]
        train_targets = all_targets[train_idx]
        test_targets = all_targets[test_idx]

        log_info(f"\nTrain: {len(train_queries)}, Test: {len(test_queries)}")

        # Prepare centroid data for curriculum warmup and periodic replay
        centroid_queries = None
        centroid_targets = None
        if args.centroid_warmup > 0 or isinstance(teacher, PairedDataTeacher):
            log_info(f"\nPreparing centroid data...")
            if isinstance(teacher, PairedDataTeacher):
                # PairedDataTeacher already has pre-computed centroid pairs
                centroid_queries, centroid_targets = teacher.get_centroid_pairs()
                log_info(f"  {len(centroid_queries)} centroids from PairedDataTeacher")
            else:
                # Load centroids and W matrices
                direct_teacher = DirectTeacher(args.federated_model, top_k=1)
                centroid_queries = direct_teacher.cluster_centroids.astype(np.float32)
                # Compute W @ centroid for each cluster (direct, no interpolation)
                centroid_targets = np.zeros_like(centroid_queries)
                for i in range(len(centroid_queries)):
                    centroid_targets[i] = direct_teacher.W_tensor[i] @ centroid_queries[i]
                log_info(f"  {len(centroid_queries)} centroids computed")

        # Train with pre-computed targets
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
            precomputed_targets=train_targets,
            centroid_queries=centroid_queries,
            centroid_targets=centroid_targets,
            centroid_warmup_epochs=args.centroid_warmup,
        )

        # Evaluate with pre-computed targets
        log_info("\nEvaluating on test set...")
        results = evaluate_orthogonal_transformer(
            transformer, teacher, test_queries,
            batch_size=128, top_k=args.top_k,
            precomputed_targets=test_targets,
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

        # Save transformer model
        if args.save_transformer:
            metadata = {
                'federated_model': str(args.federated_model),
                'teacher_type': args.teacher,
                'top_k_routing': args.top_k_routing,
                'n_queries': len(query_embeddings),
                'mean_cosine_sim': results['mean_cosine_sim'],
            }
            if args.transformer_type == "bivector-blend":
                save_composed_bivector_transformer(
                    transformer, basis_bivectors, basis_keys,
                    args.save_transformer, metadata
                )
            else:
                save_orthogonal_transformer(transformer, codebook, args.save_transformer, metadata)

    elif args.train_multisource:
        # Multi-source training on multiple federated models
        if not args.federated_models:
            log_info("ERROR: --federated-models required for multi-source training")
            return 1

        log_info(f"\n{'=' * 60}")
        log_info("Multi-Source Orthogonal Codebook Training")
        log_info(f"{'=' * 60}")
        log_info(f"Sources: {len(args.federated_models)}")
        for i, path in enumerate(args.federated_models):
            log_info(f"  {i}: {Path(path).name}")

        # Load or build codebook
        if args.orthogonal_codebook:
            log_info(f"\nLoading orthogonal codebook: {args.orthogonal_codebook}")
            codebook = load_orthogonal_codebook(args.orthogonal_codebook)
        else:
            # Build combined codebook from all sources
            log_info("\nBuilding combined orthogonal codebook...")
            codebook = build_combined_orthogonal_codebook(
                federated_model_paths=args.federated_models,
                n_components=args.n_components,
                method=args.codebook_method,
                pca_algorithm=args.pca_algorithm,
            )

            if args.output:
                save_orthogonal_codebook(codebook, args.output)

        # Create multi-source teacher
        log_info("\nCreating MultiSourceOrthogonalTeacher...")
        teacher = MultiSourceOrthogonalTeacher(args.federated_models, codebook)

        # Create transformer
        if args.transformer_type == "composed":
            log_info("\nCreating ComposedRotationTransformer (proper rotation composition)...")
            transformer = ComposedRotationTransformer(
                codebook=codebook,
                num_layers=args.layers,
                num_heads=args.heads,
                top_k=args.top_k,
            )
        else:
            log_info("\nCreating FastOrthogonalTransformer (additive, fast)...")
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

        # Get combined queries from teacher
        query_embeddings = teacher.query_embeddings
        log_info(f"\nTotal queries from all sources: {len(query_embeddings)}")

        # Split train/test
        n_test = int(len(query_embeddings) * args.test_split)
        indices = np.random.permutation(len(query_embeddings))
        train_idx = indices[:-n_test]
        test_idx = indices[-n_test:]

        train_queries = query_embeddings[train_idx]
        test_queries = query_embeddings[test_idx]

        log_info(f"Train: {len(train_queries)}, Test: {len(test_queries)}")

        # Show source distribution in train/test
        train_sources = teacher.source_indices[train_idx]
        test_sources = teacher.source_indices[test_idx]
        for i, source in enumerate(teacher.sources):
            train_count = np.sum(train_sources == i)
            test_count = np.sum(test_sources == i)
            log_info(f"  Source {i} ({source['path'].name}): {train_count} train, {test_count} test")

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

        print(f"\n{'=' * 60}")
        print("Results (Multi-Source Training)")
        print(f"{'=' * 60}")
        print(f"  Sources: {len(args.federated_models)}")
        print(f"  Total queries: {len(query_embeddings)}")
        print(f"  Codebook: {codebook.n_components} orthogonal planes")
        print()
        print(f"  Mean Cosine Sim: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
        print(f"  Min/Max Cosine: [{results['min_cosine_sim']:.4f}, {results['max_cosine_sim']:.4f}]")
        print(f"  Mean MSE: {results['mean_mse']:.6f}")
        print(f"{'=' * 60}")

        if results['mean_cosine_sim'] > 0.90:
            print("\n✓ Excellent: Transformer learned to project across all sources")
        elif results['mean_cosine_sim'] > 0.80:
            print("\n✓ Good: Reasonable multi-source projection")
        elif results['mean_cosine_sim'] > 0.70:
            print("\n~ Fair: Room for improvement, consider more planes")
        else:
            print("\n~ Needs work: Try more components or different architecture")

        # Save transformer model
        if args.save_transformer:
            metadata = {
                'sources': [str(p) for p in args.federated_models],
                'n_sources': len(args.federated_models),
                'total_queries': len(query_embeddings),
                'mean_cosine_sim': results['mean_cosine_sim'],
                'codebook_method': args.codebook_method,
            }
            save_orthogonal_transformer(transformer, codebook, args.save_transformer, metadata)

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
