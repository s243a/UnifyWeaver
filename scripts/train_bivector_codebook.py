#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Principal Bivector Codebook Transformer for semantic projection.

This implements the architecture from PR_principal_bivector_codebook.md:

1. Build a codebook of D principal bivectors from federated cluster rotations
2. Transformer predicts routing weights via cosine similarity to codebook keys
3. Blend top-K bivectors, apply rotation via matrix exponential
4. Both teacher and student operate in the same D-dimensional subspace

Key advantages over direct Givens angle prediction:
- Smaller output dimension (D routing weights vs k angles)
- Cosine similarity routing matches federated model approach
- Codebook constrains rotations to learned manifold
- Preserves geometric structure (isometry)

Usage:
    # Build codebook from federated model
    python scripts/train_bivector_codebook.py \
        --federated-model models/federated.pkl \
        --build-codebook \
        --codebook-size 64 \
        --output-codebook models/bivector_codebook.npz

    # Train transformer with codebook
    python scripts/train_bivector_codebook.py \
        --federated-model models/federated.pkl \
        --codebook models/bivector_codebook.npz \
        --epochs 100 \
        --layers 3
"""

import argparse
import sys
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle

import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Lazy imports for PyTorch
torch = None
nn = None
F = None


def _import_torch():
    """Lazy import torch."""
    global torch, nn, F
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch = _torch
        nn = _nn
        F = _F
    return torch, nn, F


# =============================================================================
# Codebook Construction
# =============================================================================

def matrix_log_safe(W: np.ndarray) -> np.ndarray:
    """
    Compute matrix logarithm safely, returning antisymmetric part.

    For a rotation matrix R, logm(R) should be antisymmetric.
    We enforce this for numerical stability.
    """
    from scipy.linalg import logm

    try:
        A = logm(W)
        # Ensure real (small imaginary parts can appear numerically)
        A = np.real(A)
        # Ensure antisymmetric: A = (A - A.T) / 2
        A = (A - A.T) / 2
        return A
    except Exception as e:
        logger.warning(f"Matrix log failed: {e}, returning zero matrix")
        return np.zeros_like(W)


def build_bivector_codebook(
    cluster_rotations: Dict[str, np.ndarray],
    n_components: int = 64,
    cluster_centroids: Optional[np.ndarray] = None,
) -> Dict:
    """
    Build a codebook of principal bivectors from cluster rotations.

    Args:
        cluster_rotations: Dict mapping cluster_id -> W matrix (d x d)
        n_components: Number of principal bivectors (codebook size)
        cluster_centroids: Optional (n_clusters, d) array for routing keys

    Returns:
        Dict with:
            'basis': (n_components, d, d) principal bivector basis
            'mean': (d, d) mean rotation generator
            'explained_variance': variance explained by each component
            'codebook_keys': (n_components, d) routing keys (if centroids provided)
            'cluster_ids': list of cluster IDs used
    """
    from sklearn.decomposition import PCA

    cluster_ids = list(cluster_rotations.keys())
    d = next(iter(cluster_rotations.values())).shape[0]

    logger.info(f"Building codebook from {len(cluster_ids)} clusters, dim={d}")

    # Extract rotation generators (bivectors as antisymmetric matrices)
    generators = []
    valid_cluster_ids = []

    for cid in cluster_ids:
        W = cluster_rotations[cid]
        A = matrix_log_safe(W)

        # Check if valid (non-zero)
        if np.abs(A).max() > 1e-10:
            generators.append(A)
            valid_cluster_ids.append(cid)
        else:
            logger.warning(f"Cluster {cid}: zero or invalid rotation generator, skipping")

    if len(generators) < n_components:
        logger.warning(
            f"Only {len(generators)} valid generators, reducing codebook size from {n_components}"
        )
        n_components = min(n_components, len(generators))

    logger.info(f"  Valid generators: {len(generators)}")

    # Flatten for PCA
    flat_generators = np.array([g.flatten() for g in generators])

    # PCA on rotation space
    pca = PCA(n_components=n_components)
    pca.fit(flat_generators)

    # Reshape principal components back to antisymmetric matrices
    basis = []
    for component in pca.components_:
        B = component.reshape(d, d)
        # Ensure antisymmetric
        B = (B - B.T) / 2
        # Normalize
        norm = np.linalg.norm(B)
        if norm > 1e-10:
            B = B / norm
        basis.append(B)

    basis = np.array(basis)
    mean = pca.mean_.reshape(d, d)
    mean = (mean - mean.T) / 2  # Ensure antisymmetric

    logger.info(f"  Codebook size: {n_components}")
    logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    result = {
        'basis': basis,
        'mean': mean,
        'explained_variance': pca.explained_variance_ratio_,
        'cluster_ids': valid_cluster_ids,
        'd': d,
        'n_components': n_components,
    }

    # Compute codebook keys from cluster centroids if provided
    if cluster_centroids is not None:
        # Project each cluster's rotation onto the basis to get routing keys
        # The key for basis component j is the mean projection of clusters that use it
        # For simplicity, use centroids directly as initial keys
        # (can be refined during training)
        logger.info("  Computing codebook keys from cluster centroids")

        # Use PCA on centroids to get n_components keys
        if len(cluster_centroids) >= n_components:
            centroid_pca = PCA(n_components=n_components)
            centroid_pca.fit(cluster_centroids)
            codebook_keys = centroid_pca.components_  # (n_components, d)
        else:
            # Not enough centroids, use what we have and pad
            codebook_keys = np.zeros((n_components, cluster_centroids.shape[1]))
            codebook_keys[:len(cluster_centroids)] = cluster_centroids[:n_components]

        # Normalize keys
        norms = np.linalg.norm(codebook_keys, axis=1, keepdims=True)
        codebook_keys = codebook_keys / (norms + 1e-8)

        result['codebook_keys'] = codebook_keys

    return result


def save_codebook(codebook: Dict, path: str):
    """Save codebook to .npz file."""
    np.savez(
        path,
        basis=codebook['basis'],
        mean=codebook['mean'],
        explained_variance=codebook['explained_variance'],
        codebook_keys=codebook.get('codebook_keys'),
        d=codebook['d'],
        n_components=codebook['n_components'],
    )
    logger.info(f"Saved codebook to {path}")


def load_codebook(path: str) -> Dict:
    """Load codebook from .npz file."""
    data = np.load(path, allow_pickle=True)
    codebook = {
        'basis': data['basis'],
        'mean': data['mean'],
        'explained_variance': data['explained_variance'],
        'd': int(data['d']),
        'n_components': int(data['n_components']),
    }
    if data['codebook_keys'] is not None:
        codebook['codebook_keys'] = data['codebook_keys']
    logger.info(f"Loaded codebook: {codebook['n_components']} components, dim={codebook['d']}")
    return codebook


# =============================================================================
# Bivector Codebook Transformer
# =============================================================================

class BivectorCodebookTransformer:
    """
    Transformer that routes to a codebook of principal bivectors.

    Architecture:
        Input: query embedding (batch, d)
        → Transformer encoder (L layers)
        → Routing projection → (batch, d)
        → Cosine similarity to codebook keys → (batch, n_components)
        → Top-K selection + normalize weights
        → Blend bivectors: B = Σ wᵢ × Bᵢ
        → Scale prediction
        → Output: exp(B) @ input × scale
    """

    def __init__(
        self,
        codebook: Dict,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 256,
        top_k: int = 8,
        dropout: float = 0.1,
        device: str = "auto",
    ):
        """
        Initialize BivectorCodebookTransformer.

        Args:
            codebook: Dict with 'basis', 'codebook_keys', etc.
            num_layers: Number of transformer layers
            num_heads: Attention heads per layer
            ff_dim: Feed-forward hidden dimension
            top_k: Number of codebook entries to blend
            dropout: Dropout rate
            device: Device ("auto", "cuda", "cpu")
        """
        _import_torch()

        self.embed_dim = codebook['d']
        self.n_components = codebook['n_components']
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.top_k = min(top_k, self.n_components)
        self.dropout_rate = dropout

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Register codebook as buffers (not trained by default)
        self.codebook_basis = torch.from_numpy(codebook['basis']).float().to(self.device)

        if 'codebook_keys' in codebook and codebook['codebook_keys'] is not None:
            self.codebook_keys = torch.from_numpy(codebook['codebook_keys']).float().to(self.device)
        else:
            # Initialize random keys if not provided
            self.codebook_keys = torch.randn(self.n_components, self.embed_dim).to(self.device)
            self.codebook_keys = F.normalize(self.codebook_keys, dim=1)

        # Build model
        self._build_model()

        logger.info(
            f"BivectorCodebookTransformer: {self.n_components} codebook entries, "
            f"top_k={self.top_k}, layers={num_layers}, device={self.device}"
        )

    def _build_model(self):
        """Build the transformer model."""
        # Input projection
        self.input_proj = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        ).to(self.device)

        # Routing head: projects to same dim as codebook keys
        self.routing_head = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        # Scale head
        self.scale_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, 1)
        ).to(self.device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.input_proj, self.routing_head]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for m in self.scale_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        query: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Forward pass.

        Args:
            query: Input query embeddings (batch, embed_dim)

        Returns:
            Tuple of:
                - projected: Transformed embeddings (batch, embed_dim)
                - weights: Routing weights for top-K (batch, top_k)
                - top_k_idx: Indices of top-K codebook entries (batch, top_k)
                - scale: Scale factors (batch,)
        """
        batch_size = query.shape[0]

        # Encode query
        x = self.input_proj(query).unsqueeze(1)  # (batch, 1, d)
        x = self.encoder(x)  # (batch, 1, d)
        x = x.squeeze(1)  # (batch, d)

        # Routing via cosine similarity
        routing_vec = self.routing_head(x)  # (batch, d)
        routing_vec = F.normalize(routing_vec, dim=-1)

        # Cosine similarity to codebook keys
        # codebook_keys: (n_components, d)
        similarities = torch.mm(routing_vec, self.codebook_keys.T)  # (batch, n_components)

        # Top-K selection
        top_k_sim, top_k_idx = similarities.topk(self.top_k, dim=-1)  # (batch, top_k)

        # Normalize weights over top-K (not softmax - direct cosine)
        # Ensure positive weights by clamping
        top_k_sim_pos = torch.clamp(top_k_sim, min=0.0)
        weight_sum = top_k_sim_pos.sum(dim=-1, keepdim=True) + 1e-8
        weights = top_k_sim_pos / weight_sum  # (batch, top_k)

        # Gather top-K bivectors and blend
        # codebook_basis: (n_components, d, d)
        top_k_bivectors = self.codebook_basis[top_k_idx]  # (batch, top_k, d, d)

        # Weighted sum: B = Σ wᵢ × Bᵢ
        # weights: (batch, top_k) -> (batch, top_k, 1, 1)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
        B = (weights_expanded * top_k_bivectors).sum(dim=1)  # (batch, d, d)

        # Scale prediction
        scale = self.scale_head(x).squeeze(-1)  # (batch,)
        scale = F.softplus(scale) + 0.5  # Ensure positive, centered around 1

        # Apply rotation via matrix exponential
        R = torch.matrix_exp(B)  # (batch, d, d)

        # Apply rotation and scale
        projected = torch.bmm(R, query.unsqueeze(-1)).squeeze(-1)  # (batch, d)
        projected = projected * scale.unsqueeze(-1)

        return projected, weights, top_k_idx, scale

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project query embedding (numpy interface)."""
        was_1d = query_emb.ndim == 1
        if was_1d:
            query_emb = query_emb[np.newaxis, :]

        with torch.no_grad():
            query_tensor = torch.from_numpy(query_emb).float().to(self.device)
            projected, _, _, _ = self.forward(query_tensor)
            result = projected.cpu().numpy()

        if was_1d:
            result = result[0]

        return result

    def get_routing_info(self, query_emb: np.ndarray) -> Dict:
        """Get routing information for a query."""
        was_1d = query_emb.ndim == 1
        if was_1d:
            query_emb = query_emb[np.newaxis, :]

        with torch.no_grad():
            query_tensor = torch.from_numpy(query_emb).float().to(self.device)
            _, weights, top_k_idx, scale = self.forward(query_tensor)

            result = {
                'weights': weights.cpu().numpy(),
                'top_k_indices': top_k_idx.cpu().numpy(),
                'scale': scale.cpu().numpy(),
            }

        if was_1d:
            result['weights'] = result['weights'][0]
            result['top_k_indices'] = result['top_k_indices'][0]
            result['scale'] = result['scale'][0]

        return result

    def parameters(self):
        """Return trainable parameters."""
        params = list(self.input_proj.parameters())
        params += list(self.encoder.parameters())
        params += list(self.routing_head.parameters())
        params += list(self.scale_head.parameters())
        return params

    def train_mode(self):
        """Set to training mode."""
        self.input_proj.train()
        self.encoder.train()
        self.routing_head.train()
        self.scale_head.train()

    def eval_mode(self):
        """Set to evaluation mode."""
        self.input_proj.eval()
        self.encoder.eval()
        self.routing_head.eval()
        self.scale_head.eval()

    def get_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'embed_dim': self.embed_dim,
            'n_components': self.n_components,
            'top_k': self.top_k,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'total_parameters': total_params,
            'device': str(self.device),
        }


# =============================================================================
# Teacher with Subspace Projection
# =============================================================================

class BivectorTeacher:
    """
    Federated teacher that projects to the same D-dimensional bivector subspace.

    This ensures teacher and student operate in the same manifold,
    making errors meaningful for model comparison.
    """

    def __init__(
        self,
        federated_model_path: str,
        codebook: Dict,
    ):
        """
        Args:
            federated_model_path: Path to federated .pkl model
            codebook: Codebook dict with 'basis' for projection
        """
        self.codebook = codebook
        self.basis = codebook['basis']  # (n_components, d, d)
        self.n_components = codebook['n_components']
        self.d = codebook['d']

        # Flatten basis for projection
        self.basis_flat = self.basis.reshape(self.n_components, -1)  # (n_components, d²)

        # Load federated model
        self._load_federated_model(federated_model_path)

        logger.info(
            f"BivectorTeacher: {self.num_clusters} clusters, "
            f"projecting to {self.n_components}-dim subspace"
        )

    def _load_federated_model(self, model_path: str):
        """Load federated model."""
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

        # Load cluster W matrices and compute bivectors
        self.cluster_bivectors = {}
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

                # Compute bivector (log of rotation)
                A = matrix_log_safe(W)
                self.cluster_bivectors[cid] = A

    def project_bivector_to_subspace(self, bivector: np.ndarray) -> np.ndarray:
        """
        Project a full bivector onto the D-dimensional subspace.

        Args:
            bivector: (d, d) antisymmetric matrix

        Returns:
            coefficients: (n_components,) projection onto each basis vector
        """
        biv_flat = bivector.flatten()
        coefficients = self.basis_flat @ biv_flat
        return coefficients

    def get_target_coefficients(
        self,
        query_emb: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Get target bivector coefficients for a query.

        This computes the blended bivector from the federated model,
        then projects it onto the codebook subspace.

        Args:
            query_emb: Query embedding (d,)
            top_k: Number of clusters to blend

        Returns:
            coefficients: (n_components,) target coefficients
            scale: Target scale factor
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

        # Project onto subspace
        coefficients = self.project_bivector_to_subspace(blended_bivector)

        # Compute scale (ratio of output to input norm after rotation)
        # For simplicity, use 1.0 or compute from actual projection
        scale = 1.0

        return coefficients.astype(np.float32), scale

    def project(self, query_emb: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        Project query using blended bivector in subspace.

        This is for evaluation - applies the subspace-projected rotation.
        """
        coefficients, scale = self.get_target_coefficients(query_emb, top_k)

        # Reconstruct bivector from coefficients
        # B = Σ cᵢ × Bᵢ
        bivector = np.tensordot(coefficients, self.basis, axes=([0], [0]))  # (d, d)

        # Apply rotation
        from scipy.linalg import expm
        R = expm(bivector)

        result = R @ query_emb * scale
        return result


# =============================================================================
# Training
# =============================================================================

def train_bivector_codebook_transformer(
    transformer: BivectorCodebookTransformer,
    teacher: BivectorTeacher,
    query_embeddings: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 10,
    coefficient_weight: float = 0.5,
    output_weight: float = 0.5,
) -> List[float]:
    """
    Train BivectorCodebookTransformer.

    Args:
        transformer: BivectorCodebookTransformer to train
        teacher: BivectorTeacher for target coefficients
        query_embeddings: Training queries (N, d)
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_interval: Log every N epochs
        coefficient_weight: Weight for coefficient loss
        output_weight: Weight for output reconstruction loss

    Returns:
        List of loss values per epoch
    """
    _import_torch()

    n_samples = len(query_embeddings)

    # Pre-compute target coefficients
    logger.info("Computing target coefficients from teacher...")
    target_coefficients = []
    target_outputs = []

    for i, query in enumerate(query_embeddings):
        coeffs, scale = teacher.get_target_coefficients(query)
        target_coefficients.append(coeffs)
        target_outputs.append(teacher.project(query))

        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{n_samples}")

    target_coefficients = np.stack(target_coefficients)
    target_outputs = np.stack(target_outputs)

    # Convert to tensors
    queries_tensor = torch.from_numpy(query_embeddings).float().to(transformer.device)
    coeffs_tensor = torch.from_numpy(target_coefficients).float().to(transformer.device)
    outputs_tensor = torch.from_numpy(target_outputs).float().to(transformer.device)

    # Optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    # Training loop
    transformer.train_mode()
    losses = []

    logger.info(
        f"Training for {num_epochs} epochs, {n_samples} samples, "
        f"coeff_weight={coefficient_weight}, output_weight={output_weight}"
    )

    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_coeff_loss = 0.0
        epoch_output_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_queries = queries_tensor[idx]
            batch_target_coeffs = coeffs_tensor[idx]
            batch_target_outputs = outputs_tensor[idx]

            # Forward
            predicted_output, weights, top_k_idx, scale = transformer.forward(batch_queries)

            # Coefficient loss: compare routing weights to target coefficients
            # We need to gather the target coefficients for the selected indices
            # This is approximate - we're comparing routing weights to projected coefficients
            batch_selected_coeffs = batch_target_coeffs.gather(
                1, top_k_idx
            )  # (batch, top_k)

            # Normalize target coefficients for comparison
            coeff_norm = batch_selected_coeffs.abs().sum(dim=1, keepdim=True) + 1e-8
            target_weights = batch_selected_coeffs.abs() / coeff_norm

            coeff_loss = F.mse_loss(weights, target_weights)

            # Output reconstruction loss
            mse_loss = F.mse_loss(predicted_output, batch_target_outputs)
            pred_norm = F.normalize(predicted_output, dim=1)
            target_norm = F.normalize(batch_target_outputs, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
            output_loss = 0.5 * mse_loss + 0.5 * (1 - cosine_sim)

            # Combined loss
            loss = coefficient_weight * coeff_loss + output_weight * output_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_coeff_loss += coeff_loss.item()
            epoch_output_loss += output_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, "
                f"Coeff: {epoch_coeff_loss/n_batches:.6f}, "
                f"Output: {epoch_output_loss/n_batches:.6f}, "
                f"Cos: {cosine_sim.item():.4f}"
            )

    transformer.eval_mode()
    logger.info(f"Training complete. Final loss: {losses[-1]:.6f}")

    return losses


def evaluate_transformer(
    transformer: BivectorCodebookTransformer,
    teacher: BivectorTeacher,
    test_queries: np.ndarray,
) -> Dict:
    """Evaluate transformer against teacher on test set."""
    transformer.eval_mode()

    mse_values = []
    cosine_sims = []

    for query in test_queries:
        teacher_out = teacher.project(query)
        trans_out = transformer.project(query)

        mse = np.mean((teacher_out - trans_out) ** 2)
        mse_values.append(mse)

        cos_sim = np.dot(teacher_out, trans_out) / (
            np.linalg.norm(teacher_out) * np.linalg.norm(trans_out) + 1e-8
        )
        cosine_sims.append(cos_sim)

    return {
        'mean_mse': np.mean(mse_values),
        'std_mse': np.std(mse_values),
        'mean_cosine_sim': np.mean(cosine_sims),
        'std_cosine_sim': np.std(cosine_sims),
        'min_cosine_sim': np.min(cosine_sims),
        'max_cosine_sim': np.max(cosine_sims),
        'n_samples': len(test_queries),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train BivectorCodebookTransformer for semantic projection"
    )

    # Input sources
    parser.add_argument("--federated-model", required=True,
                       help="Path to federated Procrustes model (.pkl)")
    parser.add_argument("--codebook", help="Path to pre-built codebook (.npz)")

    # Codebook building
    parser.add_argument("--build-codebook", action="store_true",
                       help="Build codebook from federated model")
    parser.add_argument("--codebook-size", type=int, default=64,
                       help="Number of principal bivectors in codebook")
    parser.add_argument("--output-codebook", help="Path to save built codebook")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")

    # Model architecture
    parser.add_argument("--layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--top-k", type=int, default=8, help="Top-K codebook entries to blend")

    # Output
    parser.add_argument("--save", help="Path to save trained model")

    args = parser.parse_args()

    print("=" * 60)
    print("Principal Bivector Codebook Transformer")
    print("=" * 60)

    # Load federated model
    federated_path = Path(args.federated_model)
    if not federated_path.exists():
        logger.error(f"Federated model not found: {federated_path}")
        return 1

    print(f"\nLoading federated model: {federated_path}")

    with open(federated_path, 'rb') as f:
        meta = pickle.load(f)

    cluster_ids = meta["cluster_ids"]
    cluster_centroids = meta.get("cluster_centroids")

    # Load cluster rotations
    cluster_dir = federated_path.with_suffix('')
    cluster_rotations = {}

    for cid in cluster_ids:
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
            cluster_rotations[cid] = W

    print(f"  Loaded {len(cluster_rotations)} cluster rotations")

    # Build or load codebook
    if args.build_codebook:
        print(f"\nBuilding codebook with {args.codebook_size} components...")
        codebook = build_bivector_codebook(
            cluster_rotations,
            n_components=args.codebook_size,
            cluster_centroids=cluster_centroids,
        )

        if args.output_codebook:
            save_codebook(codebook, args.output_codebook)

        if not args.epochs:
            print("\nCodebook built. Use --epochs to train transformer.")
            return 0

    elif args.codebook:
        print(f"\nLoading codebook: {args.codebook}")
        codebook = load_codebook(args.codebook)

    else:
        logger.error("Must provide --codebook or --build-codebook")
        return 1

    # Load query embeddings
    routing_path = cluster_dir / "routing_data.npz"
    routing_data = np.load(routing_path)
    query_embeddings = routing_data['query_embeddings'].astype(np.float32)
    print(f"  Loaded {len(query_embeddings)} query embeddings")

    # Create teacher
    print("\nCreating BivectorTeacher...")
    teacher = BivectorTeacher(args.federated_model, codebook)

    # Create transformer
    print("\nCreating BivectorCodebookTransformer...")
    transformer = BivectorCodebookTransformer(
        codebook=codebook,
        num_layers=args.layers,
        num_heads=args.heads,
        top_k=args.top_k,
    )

    info = transformer.get_info()
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Codebook size: {info['n_components']}")
    print(f"  Top-K: {info['top_k']}")
    print(f"  Device: {info['device']}")

    # Split train/test
    n_test = int(len(query_embeddings) * args.test_split)
    indices = np.random.permutation(len(query_embeddings))
    train_idx = indices[:-n_test]
    test_idx = indices[-n_test:]

    train_queries = query_embeddings[train_idx]
    test_queries = query_embeddings[test_idx]

    print(f"\nTrain: {len(train_queries)}, Test: {len(test_queries)}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    losses = train_bivector_codebook_transformer(
        transformer=transformer,
        teacher=teacher,
        query_embeddings=train_queries,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_interval=20,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate_transformer(transformer, teacher, test_queries)

    print(f"\n{'=' * 40}")
    print("Results (vs rotation-based teacher in same subspace):")
    print(f"  Mean Cosine Sim: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
    print(f"  Min/Max Cosine: [{results['min_cosine_sim']:.4f}, {results['max_cosine_sim']:.4f}]")
    print(f"  Mean MSE: {results['mean_mse']:.6f}")
    print(f"{'=' * 40}")

    if results['mean_cosine_sim'] > 0.90:
        print("\n✓ Excellent: Transformer closely matches teacher in subspace")
    elif results['mean_cosine_sim'] > 0.80:
        print("\n✓ Good: Transformer reasonably matches teacher")
    elif results['mean_cosine_sim'] > 0.70:
        print("\n~ Fair: Room for improvement")
    else:
        print("\n~ Needs work: Consider larger codebook or more training")

    if args.save:
        # TODO: Implement save/load for transformer
        print(f"\nNote: Model saving not yet implemented")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
