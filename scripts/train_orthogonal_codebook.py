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
    parser.add_argument("--federated-model", help="Path to federated model (.pkl)")

    # Operations
    parser.add_argument("--orthogonalize", action="store_true",
                       help="Orthogonalize existing codebook")
    parser.add_argument("--build-canonical", action="store_true",
                       help="Build canonical orthogonal codebook from scratch")
    parser.add_argument("--test", action="store_true",
                       help="Test codebook with sample data")

    # Parameters
    parser.add_argument("--n-components", type=int, default=64,
                       help="Number of codebook entries")
    parser.add_argument("--top-k", type=int, default=8,
                       help="Top-K entries to blend")

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

    else:
        parser.print_help()
        return 1

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
