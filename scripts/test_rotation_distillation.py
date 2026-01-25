#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Rotation-based transformer distillation from LDA multi-head projection.

Instead of fitting output vectors directly, this constrains the transformer
to learn rotation angles and scaling factors - a minimal geometric transform.

Key idea: The transformation from query → projected is decomposed into:
    projected = scale * R(θ₁, θ₂, ..., θₖ) @ query

Where R is a rotation matrix composed from k Givens rotations on selected planes.

Two training modes:
1. --mode output: Supervise on output vectors (MSE + cosine loss)
2. --mode angle: Supervise on optimal rotation angles (default)
   - Computes the optimal angles for each input-target pair
   - Directly trains the network to predict those angles
   - Also includes output loss for end-to-end learning

This provides:
- More interpretable transformations
- Constrained parameter space (rotations preserve norms)
- Potentially better generalization from geometric priors

Usage:
    # Angle-supervised training (default, recommended)
    python scripts/test_rotation_distillation.py --db lda.db --mode angle

    # Output-supervised training (original approach)
    python scripts/test_rotation_distillation.py --db lda.db --mode output
"""

import argparse
import sys
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB

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


class GivensRotationLayer:
    """
    Apply Givens rotations to input vectors.

    A Givens rotation operates on a single 2D plane within the n-dimensional space.
    Multiple Givens rotations can be composed for more expressive rotations.

    For efficiency, we select k rotation planes and apply rotations in parallel
    where possible (non-overlapping planes).
    """

    def __init__(self, embed_dim: int, num_rotation_planes: int):
        """
        Args:
            embed_dim: Dimension of input/output vectors
            num_rotation_planes: Number of 2D rotation planes to use
        """
        self.embed_dim = embed_dim
        self.num_rotation_planes = num_rotation_planes

        # Pre-select rotation planes (pairs of dimensions)
        # We'll spread them across the space to cover different directions
        self.plane_indices = self._select_rotation_planes()

    def _select_rotation_planes(self) -> List[Tuple[int, int]]:
        """Select which 2D planes to rotate in."""
        planes = []
        # Strategy: use interleaved pairs to cover the space
        # E.g., (0,1), (2,3), ..., then (1,2), (3,4), ...

        used = set()
        stride = 1
        start = 0

        while len(planes) < self.num_rotation_planes:
            for i in range(start, self.embed_dim - stride, stride * 2):
                j = i + stride
                if j < self.embed_dim and (i, j) not in used:
                    planes.append((i, j))
                    used.add((i, j))
                    if len(planes) >= self.num_rotation_planes:
                        break

            start = (start + 1) % (stride * 2)
            if start == 0:
                stride += 1
                if stride >= self.embed_dim:
                    break

        return planes[:self.num_rotation_planes]

    def forward(self, x: "torch.Tensor", angles: "torch.Tensor") -> "torch.Tensor":
        """
        Apply Givens rotations to input vectors.

        Args:
            x: Input vectors (batch, embed_dim)
            angles: Rotation angles in radians (batch, num_rotation_planes)

        Returns:
            Rotated vectors (batch, embed_dim)
        """
        _import_torch()

        # Start with a copy, then apply rotations sequentially
        # Each rotation reads from result, computes new values, creates new tensor
        result = x

        for k, (i, j) in enumerate(self.plane_indices):
            theta = angles[:, k]  # (batch,)
            cos_t = torch.cos(theta)  # (batch,)
            sin_t = torch.sin(theta)  # (batch,)

            xi = result[:, i]  # (batch,)
            xj = result[:, j]  # (batch,)

            # Compute new values
            new_i = cos_t * xi - sin_t * xj  # (batch,)
            new_j = sin_t * xi + cos_t * xj  # (batch,)

            # Create new tensor with updated columns (no in-place modification)
            # Use scatter or manual construction
            cols = list(result.unbind(dim=1))  # list of (batch,) tensors
            cols[i] = new_i
            cols[j] = new_j
            result = torch.stack(cols, dim=1)

        return result


class RotationTransformer:
    """
    Transformer that predicts rotation angles and scaling for minimal transforms.

    Instead of directly predicting the output vector, this model predicts:
    - k rotation angles for Givens rotations on selected planes
    - Optional: a scaling factor (uniform or per-dimension)

    The final output is: scale * R(angles) @ input

    This constrains the transformation to be geometric, potentially
    leading to better generalization and interpretability.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_rotation_planes: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        use_scaling: bool = True,
        scaling_mode: str = "uniform",  # "uniform", "per_dim", or "none"
        dropout: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize RotationTransformer.

        Args:
            embed_dim: Embedding dimension (must match LDA)
            num_rotation_planes: Number of Givens rotation planes (k angles)
            num_heads: Attention heads per layer
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            use_scaling: Whether to predict scaling factors
            scaling_mode: "uniform" (1 scalar), "per_dim" (embed_dim scalars), "none"
            dropout: Dropout rate
            device: Device ("auto", "cuda", "cpu")
        """
        _import_torch()

        self.embed_dim = embed_dim
        self.num_rotation_planes = num_rotation_planes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.use_scaling = use_scaling
        self.scaling_mode = scaling_mode if use_scaling else "none"
        self.dropout_rate = dropout

        # Calculate output dimension
        self.num_angles = num_rotation_planes
        if self.scaling_mode == "uniform":
            self.num_scale_params = 1
        elif self.scaling_mode == "per_dim":
            self.num_scale_params = embed_dim
        else:
            self.num_scale_params = 0

        self.output_dim = self.num_angles + self.num_scale_params

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

        # Build model
        self._build_model()

        logger.info(
            f"RotationTransformer: {num_rotation_planes} planes, "
            f"scaling={self.scaling_mode}, output_dim={self.output_dim}, "
            f"device={self.device}"
        )

    def _build_model(self):
        """Build the transformer model that predicts rotation parameters."""
        # Input projection
        self.input_proj = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        # Transformer encoder layers
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

        # Output head: predicts rotation angles and optionally scaling
        self.param_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.output_dim)
        ).to(self.device)

        # Givens rotation layer (not a nn.Module, no .to() needed)
        self.rotation_layer = GivensRotationLayer(
            self.embed_dim,
            self.num_rotation_planes
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training."""
        for module in [self.input_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Initialize param_head to output small angles initially
        for m in self.param_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, query: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """
        Forward pass.

        Args:
            query: Input query embeddings (batch, embed_dim)

        Returns:
            Tuple of:
                - projected: Transformed embeddings (batch, embed_dim)
                - angles: Predicted rotation angles (batch, num_rotation_planes)
                - scale: Predicted scale factors (batch, num_scale_params) or None
        """
        # Process through transformer
        x = self.input_proj(query).unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze(1)

        # Predict rotation parameters
        params = self.param_head(x)  # (batch, output_dim)

        # Split into angles and scale
        angles = params[:, :self.num_angles]  # (batch, num_rotation_planes)

        # Constrain angles to reasonable range (-π, π) using tanh
        angles = torch.tanh(angles) * math.pi

        if self.num_scale_params > 0:
            raw_scale = params[:, self.num_angles:]  # (batch, num_scale_params)
            # Use softplus to ensure positive scaling, centered around 1
            scale = F.softplus(raw_scale) + 0.5
        else:
            scale = None

        # Apply rotation
        rotated = self.rotation_layer.forward(query, angles)

        # Apply scaling
        if scale is not None:
            if self.scaling_mode == "uniform":
                projected = rotated * scale  # broadcast (batch, 1) to (batch, embed_dim)
            else:  # per_dim
                projected = rotated * scale  # element-wise (batch, embed_dim)
        else:
            projected = rotated

        return projected, angles, scale

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Project a query embedding (numpy interface).

        Args:
            query_emb: Query embedding (embed_dim,) or (batch, embed_dim)

        Returns:
            Projected embedding(s)
        """
        was_1d = query_emb.ndim == 1
        if was_1d:
            query_emb = query_emb[np.newaxis, :]

        with torch.no_grad():
            query_tensor = torch.from_numpy(query_emb).float().to(self.device)
            projected, _, _ = self.forward(query_tensor)
            result = projected.cpu().numpy()

        if was_1d:
            result = result[0]

        return result

    def get_transform_params(self, query_emb: np.ndarray) -> dict:
        """
        Get the transformation parameters for a query.

        Args:
            query_emb: Query embedding (embed_dim,)

        Returns:
            Dict with 'angles' and optionally 'scale'
        """
        was_1d = query_emb.ndim == 1
        if was_1d:
            query_emb = query_emb[np.newaxis, :]

        with torch.no_grad():
            query_tensor = torch.from_numpy(query_emb).float().to(self.device)
            _, angles, scale = self.forward(query_tensor)

            result = {
                'angles': angles.cpu().numpy(),
                'rotation_planes': self.rotation_layer.plane_indices,
            }
            if scale is not None:
                result['scale'] = scale.cpu().numpy()

        if was_1d:
            result['angles'] = result['angles'][0]
            if 'scale' in result:
                result['scale'] = result['scale'][0]

        return result

    def parameters(self):
        """Return all trainable parameters."""
        params = list(self.input_proj.parameters())
        params += list(self.encoder.parameters())
        params += list(self.param_head.parameters())
        return params

    def train_mode(self):
        """Set to training mode."""
        self.input_proj.train()
        self.encoder.train()
        self.param_head.train()

    def eval_mode(self):
        """Set to evaluation mode."""
        self.input_proj.eval()
        self.encoder.eval()
        self.param_head.eval()

    def save(self, path: str):
        """Save model weights."""
        state = {
            'embed_dim': self.embed_dim,
            'num_rotation_planes': self.num_rotation_planes,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'use_scaling': self.use_scaling,
            'scaling_mode': self.scaling_mode,
            'dropout': self.dropout_rate,
            'plane_indices': self.rotation_layer.plane_indices,
            'input_proj': self.input_proj.state_dict(),
            'encoder': self.encoder.state_dict(),
            'param_head': self.param_head.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "RotationTransformer":
        """Load model from file."""
        _import_torch()
        state = torch.load(path, map_location='cpu')

        model = cls(
            embed_dim=state['embed_dim'],
            num_rotation_planes=state['num_rotation_planes'],
            num_heads=state['num_heads'],
            num_layers=state['num_layers'],
            ff_dim=state['ff_dim'],
            use_scaling=state['use_scaling'],
            scaling_mode=state['scaling_mode'],
            dropout=state['dropout'],
            device=device
        )
        model.input_proj.load_state_dict(state['input_proj'])
        model.encoder.load_state_dict(state['encoder'])
        model.param_head.load_state_dict(state['param_head'])
        model.rotation_layer.plane_indices = state['plane_indices']

        logger.info(f"Loaded model from {path}")
        return model

    def get_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'embed_dim': self.embed_dim,
            'num_rotation_planes': self.num_rotation_planes,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'scaling_mode': self.scaling_mode,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'device': str(self.device),
        }


def train_rotation_distillation(
    transformer: RotationTransformer,
    lda_projection,
    query_embeddings: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 10,
    cosine_weight: float = 0.5,
    angle_reg_weight: float = 0.01,
) -> List[float]:
    """
    Train rotation transformer via knowledge distillation from LDA projection.

    Args:
        transformer: RotationTransformer to train
        lda_projection: LDA multi-head projection (teacher)
        query_embeddings: Training query embeddings (N, embed_dim)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_interval: Log every N epochs
        cosine_weight: Weight for cosine similarity loss (0=MSE only, 1=cosine only)
        angle_reg_weight: Regularization weight for angle magnitudes (prefer small rotations)

    Returns:
        List of loss values per epoch
    """
    _import_torch()

    # Generate target projections from LDA (teacher)
    logger.info("Generating LDA target projections...")
    lda_outputs = []
    for i in range(len(query_embeddings)):
        proj = lda_projection.project(query_embeddings[i])
        if isinstance(proj, tuple):
            proj = proj[0]
        lda_outputs.append(proj)
    lda_outputs = np.stack(lda_outputs)

    # Convert to tensors
    queries_tensor = torch.from_numpy(query_embeddings).float().to(transformer.device)
    targets_tensor = torch.from_numpy(lda_outputs).float().to(transformer.device)

    # Optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    # Training loop
    transformer.train_mode()
    losses = []
    n_samples = len(query_embeddings)

    logger.info(f"Training for {num_epochs} epochs, {n_samples} samples, batch_size={batch_size}")

    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cos = 0.0
        epoch_angle = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_queries = queries_tensor[idx]
            batch_targets = targets_tensor[idx]

            # Forward
            predicted, angles, scale = transformer.forward(batch_queries)

            # MSE loss
            mse_loss = F.mse_loss(predicted, batch_targets)

            # Cosine loss
            pred_norm = F.normalize(predicted, p=2, dim=1)
            target_norm = F.normalize(batch_targets, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
            cosine_loss = 1 - cosine_sim

            # Angle regularization (prefer smaller rotations when possible)
            angle_reg = torch.mean(angles ** 2)

            # Combined loss
            loss = (
                (1 - cosine_weight) * mse_loss +
                cosine_weight * cosine_loss +
                angle_reg_weight * angle_reg
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_cos += cosine_sim.item()
            epoch_angle += angle_reg.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, "
                f"MSE: {epoch_mse/n_batches:.6f}, Cos: {epoch_cos/n_batches:.4f}, "
                f"AngleReg: {epoch_angle/n_batches:.6f}"
            )

    transformer.eval_mode()
    logger.info(f"Training complete. Final loss: {losses[-1]:.6f}")

    return losses


def compute_optimal_givens_angle(x_i: float, x_j: float, y_i: float, y_j: float) -> float:
    """Compute optimal Givens rotation angle for a single 2D plane."""
    x_angle = math.atan2(x_j, x_i)
    y_angle = math.atan2(y_j, y_i)
    theta = y_angle - x_angle
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def compute_optimal_rotation_params(
    input_vec: np.ndarray,
    target_vec: np.ndarray,
    plane_indices: List[Tuple[int, int]],
    compute_scale: bool = True
) -> Tuple[np.ndarray, Optional[float]]:
    """Compute optimal Givens rotation angles and scale to transform input to target."""
    input_norm = np.linalg.norm(input_vec)
    target_norm = np.linalg.norm(target_vec)

    if compute_scale and input_norm > 1e-8:
        scale = target_norm / input_norm
    else:
        scale = 1.0 if compute_scale else None

    if compute_scale and scale is not None:
        effective_target = target_vec / scale if scale > 1e-8 else target_vec
    else:
        effective_target = target_vec

    current = input_vec.copy()
    angles = []

    for (i, j) in plane_indices:
        theta = compute_optimal_givens_angle(
            current[i], current[j],
            effective_target[i], effective_target[j]
        )
        angles.append(theta)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        new_i = cos_t * current[i] - sin_t * current[j]
        new_j = sin_t * current[i] + cos_t * current[j]
        current[i] = new_i
        current[j] = new_j

    return np.array(angles, dtype=np.float32), scale


def compute_optimal_rotation_params_batch(
    input_vecs: np.ndarray,
    target_vecs: np.ndarray,
    plane_indices: List[Tuple[int, int]],
    compute_scale: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Batch version of compute_optimal_rotation_params."""
    batch_size = len(input_vecs)
    num_planes = len(plane_indices)

    all_angles = np.zeros((batch_size, num_planes), dtype=np.float32)
    all_scales = np.zeros(batch_size, dtype=np.float32) if compute_scale else None

    for b in range(batch_size):
        angles, scale = compute_optimal_rotation_params(
            input_vecs[b], target_vecs[b], plane_indices, compute_scale
        )
        all_angles[b] = angles
        if compute_scale:
            all_scales[b] = scale

    return all_angles, all_scales


def train_rotation_distillation_angle_supervised(
    transformer: RotationTransformer,
    lda_projection,
    query_embeddings: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 10,
    output_weight: float = 0.5,
    angle_weight: float = 0.5,
    scale_weight: float = 0.1,
) -> List[float]:
    """Train rotation transformer with direct angle supervision."""
    _import_torch()

    logger.info("Generating LDA target projections...")
    lda_outputs = []
    for i in range(len(query_embeddings)):
        proj = lda_projection.project(query_embeddings[i])
        if isinstance(proj, tuple):
            proj = proj[0]
        lda_outputs.append(proj)
    lda_outputs = np.stack(lda_outputs)

    logger.info("Computing optimal rotation parameters...")
    plane_indices = transformer.rotation_layer.plane_indices
    compute_scale = transformer.scaling_mode != "none"

    optimal_angles, optimal_scales = compute_optimal_rotation_params_batch(
        query_embeddings, lda_outputs, plane_indices, compute_scale
    )

    queries_tensor = torch.from_numpy(query_embeddings).float().to(transformer.device)
    targets_tensor = torch.from_numpy(lda_outputs).float().to(transformer.device)
    optimal_angles_tensor = torch.from_numpy(optimal_angles).float().to(transformer.device)

    if optimal_scales is not None:
        optimal_scales_tensor = torch.from_numpy(optimal_scales).float().to(transformer.device)
        if transformer.scaling_mode == "uniform":
            optimal_scales_tensor = optimal_scales_tensor.unsqueeze(1)
    else:
        optimal_scales_tensor = None

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    transformer.train_mode()
    losses = []
    n_samples = len(query_embeddings)

    logger.info(
        f"Training for {num_epochs} epochs, {n_samples} samples, "
        f"output_weight={output_weight}, angle_weight={angle_weight}"
    )

    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_output = 0.0
        epoch_angle = 0.0
        epoch_scale = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_queries = queries_tensor[idx]
            batch_targets = targets_tensor[idx]
            batch_opt_angles = optimal_angles_tensor[idx]

            predicted, pred_angles, pred_scale = transformer.forward(batch_queries)

            # Output loss
            mse_loss = F.mse_loss(predicted, batch_targets)
            pred_norm = F.normalize(predicted, p=2, dim=1)
            target_norm = F.normalize(batch_targets, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
            output_loss = 0.5 * mse_loss + 0.5 * (1 - cosine_sim)

            # Angle loss with circular wrapping
            angle_diff = pred_angles - batch_opt_angles
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            angle_loss = torch.mean(angle_diff ** 2)

            # Scale loss
            if optimal_scales_tensor is not None and pred_scale is not None:
                batch_opt_scales = optimal_scales_tensor[idx]
                scale_loss = F.mse_loss(pred_scale, batch_opt_scales)
            else:
                scale_loss = torch.tensor(0.0, device=transformer.device)

            loss = output_weight * output_loss + angle_weight * angle_loss + scale_weight * scale_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_output += output_loss.item()
            epoch_angle += angle_loss.item()
            epoch_scale += scale_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, "
                f"Output: {epoch_output/n_batches:.6f}, "
                f"Angle: {epoch_angle/n_batches:.6f}, "
                f"Cos: {cosine_sim.item():.4f}"
            )

    transformer.eval_mode()
    logger.info(f"Training complete. Final loss: {losses[-1]:.6f}")

    return losses


def evaluate_rotation_equivalence(
    transformer: RotationTransformer,
    lda_projection,
    test_embeddings: np.ndarray
) -> dict:
    """
    Evaluate how well rotation transformer approximates LDA projection.

    Args:
        transformer: Trained RotationTransformer
        lda_projection: LDA multi-head projection
        test_embeddings: Test query embeddings

    Returns:
        Dict with metrics including angle statistics
    """
    transformer.eval_mode()

    mse_values = []
    cosine_sims = []
    all_angles = []
    all_scales = []

    for query_emb in test_embeddings:
        # LDA projection
        lda_out = lda_projection.project(query_emb)
        if isinstance(lda_out, tuple):
            lda_out = lda_out[0]

        # Transformer projection
        trans_out = transformer.project(query_emb)
        params = transformer.get_transform_params(query_emb)

        all_angles.append(params['angles'])
        if 'scale' in params:
            all_scales.append(params['scale'])

        # MSE
        mse = np.mean((lda_out - trans_out) ** 2)
        mse_values.append(mse)

        # Cosine similarity
        cos_sim = np.dot(lda_out, trans_out) / (
            np.linalg.norm(lda_out) * np.linalg.norm(trans_out) + 1e-8
        )
        cosine_sims.append(cos_sim)

    all_angles = np.array(all_angles)

    result = {
        'mean_mse': np.mean(mse_values),
        'std_mse': np.std(mse_values),
        'mean_cosine_sim': np.mean(cosine_sims),
        'std_cosine_sim': np.std(cosine_sims),
        'min_cosine_sim': np.min(cosine_sims),
        'max_cosine_sim': np.max(cosine_sims),
        'n_samples': len(test_embeddings),
        'angle_stats': {
            'mean_abs': np.mean(np.abs(all_angles)),
            'max_abs': np.max(np.abs(all_angles)),
            'std': np.std(all_angles),
        }
    }

    if all_scales:
        all_scales = np.array(all_scales)
        result['scale_stats'] = {
            'mean': np.mean(all_scales),
            'std': np.std(all_scales),
            'min': np.min(all_scales),
            'max': np.max(all_scales),
        }

    return result


class LDAProjectionWrapper:
    """Wrapper to provide .project() interface for LDA multi-head."""

    def __init__(self, db: LDAProjectionDB, mh_projection_id: int):
        self.db = db
        self.mh_projection_id = mh_projection_id

        mh_proj = db.get_multi_head_projection(mh_projection_id)
        self.num_heads = mh_proj['num_heads']
        self.temperature = mh_proj['temperature']

        self.heads = []
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT cluster_id, centroid_path, answer_emb_path
            FROM cluster_heads
            WHERE mh_projection_id = ?
        """, (mh_projection_id,))

        for row in cursor.fetchall():
            centroid = np.load(row['centroid_path'])
            answer_emb = np.load(row['answer_emb_path'])
            self.heads.append({
                'cluster_id': row['cluster_id'],
                'centroid': centroid,
                'answer_emb': answer_emb
            })

        logger.info(f"Loaded LDA projection: {len(self.heads)} heads, temp={self.temperature}")

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project query using multi-head routing."""
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        similarities = []
        for head in self.heads:
            centroid_norm = head['centroid'] / (np.linalg.norm(head['centroid']) + 1e-8)
            sim = np.dot(query_norm, centroid_norm)
            similarities.append(sim)

        similarities = np.array(similarities)

        scaled = similarities / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        projected = np.zeros_like(query_emb)
        for i, head in enumerate(self.heads):
            projected += weights[i] * head['answer_emb']

        return projected


class MinimalTransformProjection:
    """
    Multi-head projection that interpolates rotation parameters instead of output vectors.

    Standard: output = Σ wᵢ × answer_embᵢ  (interpolates in output space)
    Minimal:  angles = Σ wᵢ × anglesᵢ, scale = Σ wᵢ × scaleᵢ
              output = scale × R(angles) @ input  (interpolates in transform space)

    This keeps the transformation minimal during blending - we stay in the
    rotation manifold rather than jumping to arbitrary output vectors.
    """

    def __init__(self, db: LDAProjectionDB, mh_projection_id: int, num_rotation_planes: int = 64):
        self.db = db
        self.mh_projection_id = mh_projection_id
        self.num_rotation_planes = num_rotation_planes

        mh_proj = db.get_multi_head_projection(mh_projection_id)
        self.num_heads = mh_proj['num_heads']
        self.temperature = mh_proj['temperature']

        # Initialize rotation layer for plane indices
        self.rotation_layer = GivensRotationLayer.__new__(GivensRotationLayer)
        self.rotation_layer.embed_dim = 384
        self.rotation_layer.num_rotation_planes = num_rotation_planes
        self.rotation_layer.plane_indices = self._select_rotation_planes(384, num_rotation_planes)

        self.heads = []
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT cluster_id, centroid_path, answer_emb_path
            FROM cluster_heads
            WHERE mh_projection_id = ?
        """, (mh_projection_id,))

        for row in cursor.fetchall():
            centroid = np.load(row['centroid_path'])
            answer_emb = np.load(row['answer_emb_path'])

            # Compute optimal rotation from centroid to answer_emb
            angles, scale = compute_optimal_rotation_params(
                centroid, answer_emb,
                self.rotation_layer.plane_indices,
                compute_scale=True
            )

            self.heads.append({
                'cluster_id': row['cluster_id'],
                'centroid': centroid,
                'answer_emb': answer_emb,
                'angles': angles,
                'scale': scale if scale is not None else 1.0,
            })

        logger.info(f"Loaded Rotation LDA: {len(self.heads)} heads, {num_rotation_planes} planes")

    def _select_rotation_planes(self, embed_dim: int, num_planes: int) -> List[Tuple[int, int]]:
        """Select rotation planes (same algorithm as GivensRotationLayer)."""
        planes = []
        used = set()
        stride = 1
        start = 0

        while len(planes) < num_planes:
            for i in range(start, embed_dim - stride, stride * 2):
                j = i + stride
                if j < embed_dim and (i, j) not in used:
                    planes.append((i, j))
                    used.add((i, j))
                    if len(planes) >= num_planes:
                        break
            start = (start + 1) % (stride * 2)
            if start == 0:
                stride += 1
                if stride >= embed_dim:
                    break

        return planes[:num_planes]

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project query using rotation-interpolated multi-head."""
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Compute routing weights
        similarities = []
        for head in self.heads:
            centroid_norm = head['centroid'] / (np.linalg.norm(head['centroid']) + 1e-8)
            sim = np.dot(query_norm, centroid_norm)
            similarities.append(sim)

        similarities = np.array(similarities)
        scaled = similarities / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        # Blend rotation parameters
        blended_angles = np.zeros(self.num_rotation_planes, dtype=np.float32)
        blended_scale = 0.0

        for i, head in enumerate(self.heads):
            blended_angles += weights[i] * head['angles']
            blended_scale += weights[i] * head['scale']

        # Apply blended rotation
        result = query_emb.copy()
        for k, (i, j) in enumerate(self.rotation_layer.plane_indices):
            theta = blended_angles[k]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            xi = result[i]
            xj = result[j]
            result[i] = cos_t * xi - sin_t * xj
            result[j] = sin_t * xi + cos_t * xj

        # Apply blended scale
        result = result * blended_scale

        return result


def compare_projection_methods(
    standard_proj: LDAProjectionWrapper,
    minimal_proj: MinimalTransformProjection,
    test_queries: np.ndarray
) -> dict:
    """Compare standard (vector-interpolated) vs minimal (rotation-interpolated) projection."""
    cos_sims = []
    mses = []
    norm_diffs = []

    for query in test_queries:
        std_out = standard_proj.project(query)
        rot_out = minimal_proj.project(query)

        cos_sim = np.dot(std_out, rot_out) / (
            np.linalg.norm(std_out) * np.linalg.norm(rot_out) + 1e-8
        )
        mse = np.mean((std_out - rot_out) ** 2)
        norm_diff = abs(np.linalg.norm(rot_out) - np.linalg.norm(std_out))

        cos_sims.append(cos_sim)
        mses.append(mse)
        norm_diffs.append(norm_diff)

    return {
        'mean_cosine_sim': np.mean(cos_sims),
        'std_cosine_sim': np.std(cos_sims),
        'mean_mse': np.mean(mses),
        'mean_norm_diff': np.mean(norm_diffs),
    }


class SyntheticProjection:
    """Synthetic multi-head projection for testing without a database."""

    def __init__(self, embed_dim: int = 384, num_heads: int = 8, temperature: float = 0.5):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature

        # Generate random cluster heads
        np.random.seed(42)
        self.heads = []
        for i in range(num_heads):
            centroid = np.random.randn(embed_dim).astype(np.float32)
            centroid = centroid / np.linalg.norm(centroid)
            # Answer embedding is a rotated + scaled version of centroid
            answer_emb = np.random.randn(embed_dim).astype(np.float32)
            answer_emb = answer_emb / np.linalg.norm(answer_emb) * (0.8 + 0.4 * np.random.rand())
            self.heads.append({
                'centroid': centroid,
                'answer_emb': answer_emb,
            })

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project query using multi-head routing (vector interpolation)."""
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        similarities = []
        for head in self.heads:
            sim = np.dot(query_norm, head['centroid'])
            similarities.append(sim)

        similarities = np.array(similarities)
        scaled = similarities / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        projected = np.zeros_like(query_emb)
        for i, head in enumerate(self.heads):
            projected += weights[i] * head['answer_emb']

        return projected


class SyntheticMinimalProjection:
    """Synthetic minimal transform projection for testing."""

    def __init__(self, synthetic_proj: SyntheticProjection, num_rotation_planes: int = 64):
        self.embed_dim = synthetic_proj.embed_dim
        self.num_heads = synthetic_proj.num_heads
        self.temperature = synthetic_proj.temperature
        self.num_rotation_planes = num_rotation_planes

        # Compute rotation planes
        self.plane_indices = self._select_rotation_planes()

        # Convert each head to rotation parameters
        self.heads = []
        for head in synthetic_proj.heads:
            angles, scale = compute_optimal_rotation_params(
                head['centroid'], head['answer_emb'],
                self.plane_indices, compute_scale=True
            )
            self.heads.append({
                'centroid': head['centroid'].copy(),
                'angles': angles,
                'scale': scale if scale is not None else 1.0,
            })

    def _select_rotation_planes(self) -> List[Tuple[int, int]]:
        planes = []
        used = set()
        stride = 1
        start = 0
        while len(planes) < self.num_rotation_planes:
            for i in range(start, self.embed_dim - stride, stride * 2):
                j = i + stride
                if j < self.embed_dim and (i, j) not in used:
                    planes.append((i, j))
                    used.add((i, j))
                    if len(planes) >= self.num_rotation_planes:
                        break
            start = (start + 1) % (stride * 2)
            if start == 0:
                stride += 1
                if stride >= self.embed_dim:
                    break
        return planes[:self.num_rotation_planes]

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project query using rotation-interpolated multi-head."""
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        similarities = []
        for head in self.heads:
            sim = np.dot(query_norm, head['centroid'])
            similarities.append(sim)

        similarities = np.array(similarities)
        scaled = similarities / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        # Blend rotation parameters
        blended_angles = np.zeros(self.num_rotation_planes, dtype=np.float32)
        blended_scale = 0.0
        for i, head in enumerate(self.heads):
            blended_angles += weights[i] * head['angles']
            blended_scale += weights[i] * head['scale']

        # Apply blended rotation
        result = query_emb.copy()
        for k, (i, j) in enumerate(self.plane_indices):
            theta = blended_angles[k]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            xi, xj = result[i], result[j]
            result[i] = cos_t * xi - sin_t * xj
            result[j] = sin_t * xi + cos_t * xj

        return result * blended_scale


def main():
    parser = argparse.ArgumentParser(description="Test rotation-based transformer distillation")
    parser.add_argument("--db", help="Path to LDA database (optional if --synthetic)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data for testing (no database required)")
    parser.add_argument("--mh-id", type=int, default=1, help="Multi-head projection ID")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--cosine-weight", type=float, default=0.5, help="Weight for cosine loss")
    parser.add_argument("--angle-reg", type=float, default=0.01, help="Angle regularization weight")
    parser.add_argument("--rotation-planes", type=int, default=64, help="Number of rotation planes")
    parser.add_argument("--scaling", choices=["uniform", "per_dim", "none"], default="uniform",
                       help="Scaling mode")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads per layer")
    parser.add_argument("--layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--save", help="Path to save trained transformer")
    parser.add_argument("--mode", choices=["output", "angle"], default="angle",
                       help="Training mode: 'output' supervises on output vectors, "
                            "'angle' supervises on optimal rotation angles (default)")
    parser.add_argument("--output-weight", type=float, default=0.5,
                       help="Weight for output loss (angle mode)")
    parser.add_argument("--angle-weight", type=float, default=0.5,
                       help="Weight for angle loss (angle mode)")
    parser.add_argument("--teacher", choices=["standard", "rotation"], default="rotation",
                       help="Teacher type: 'standard' (vector interpolation) or "
                            "'rotation' (rotation-parameter interpolation, default)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare standard vs rotation LDA methods before training")

    parser.add_argument("--synthetic-heads", type=int, default=8,
                       help="Number of heads for synthetic data")
    parser.add_argument("--synthetic-samples", type=int, default=500,
                       help="Number of samples for synthetic data")

    args = parser.parse_args()

    if not args.synthetic and (not args.db or not Path(args.db).exists()):
        logger.error(f"Database not found: {args.db}. Use --synthetic for testing without a database.")
        return 1

    print("=" * 60)
    print("Rotation-Based Transformer Distillation Test")
    print("=" * 60)

    # Load or create projections
    if args.synthetic:
        print("\nUsing SYNTHETIC data for testing")
        standard_proj = SyntheticProjection(
            embed_dim=384,
            num_heads=args.synthetic_heads,
            temperature=0.5
        )
        minimal_proj = SyntheticMinimalProjection(standard_proj, args.rotation_planes)
        db = None

        # Generate synthetic query embeddings
        np.random.seed(123)
        query_embeddings = np.random.randn(args.synthetic_samples, 384).astype(np.float32)
        # Normalize
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    else:
        db = LDAProjectionDB(args.db)
        standard_proj = LDAProjectionWrapper(db, args.mh_id)
        minimal_proj = MinimalTransformProjection(db, args.mh_id, args.rotation_planes)

    n_heads = standard_proj.num_heads
    print(f"\nMulti-head projection: {n_heads} heads")
    print(f"Teacher type: {args.teacher}")

    # Select teacher based on argument
    if args.teacher == "rotation":
        teacher = minimal_proj
        print("  Using Minimal Transform Projection (blends rotation parameters)")
    else:
        teacher = standard_proj
        print("  Using Standard Projection (blends output vectors)")

    # Get training data
    if not args.synthetic:
        print("\nCollecting training queries from database...")

        model = db.get_model('all-MiniLM-L6-v2')
        if not model:
            logger.error("Model 'all-MiniLM-L6-v2' not found in database")
            return 1

        model_id = model['model_id']
        cursor = db.conn.cursor()

        cursor.execute("""
            SELECT e.vector_path
            FROM embeddings e
            WHERE e.model_id = ? AND e.entity_type = 'question'
        """, (model_id,))

        query_embeddings = []
        for row in cursor.fetchall():
            emb = np.load(row['vector_path'])
            query_embeddings.append(emb)

        if not query_embeddings:
            logger.error("No question embeddings found in database")
            return 1

        query_embeddings = np.stack(query_embeddings).astype(np.float32)

    print(f"Using {len(query_embeddings)} query embeddings")

    # Split train/test
    n_test = int(len(query_embeddings) * args.test_split)
    n_train = len(query_embeddings) - n_test

    indices = np.random.permutation(len(query_embeddings))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_queries = query_embeddings[train_idx]
    test_queries = query_embeddings[test_idx]

    print(f"Train: {len(train_queries)}, Test: {len(test_queries)}")

    # Compare projection methods if requested
    if args.compare:
        print("\n" + "=" * 40)
        print("Comparing Standard vs Minimal Transform Projection")
        print("=" * 40)
        comparison = compare_projection_methods(standard_proj, minimal_proj, test_queries[:100])
        print(f"  Cosine similarity: {comparison['mean_cosine_sim']:.4f} +/- {comparison['std_cosine_sim']:.4f}")
        print(f"  MSE between outputs: {comparison['mean_mse']:.6f}")
        print(f"  Mean norm difference: {comparison['mean_norm_diff']:.4f}")
        if comparison['mean_cosine_sim'] > 0.95:
            print("  -> Methods produce very similar outputs")
        elif comparison['mean_cosine_sim'] > 0.85:
            print("  -> Methods produce moderately similar outputs")
        else:
            print("  -> Methods produce notably different outputs")

    # Create rotation transformer
    print(f"\nCreating RotationTransformer...")
    print(f"  Rotation planes: {args.rotation_planes}")
    print(f"  Scaling mode: {args.scaling}")
    print(f"  Architecture: H={args.heads}, L={args.layers}")

    transformer = RotationTransformer(
        embed_dim=384,
        num_rotation_planes=args.rotation_planes,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_dim=256,
        use_scaling=args.scaling != "none",
        scaling_mode=args.scaling,
        device="auto"
    )
    info = transformer.get_info()
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Output dim: {info['output_dim']} (angles + scale)")
    print(f"  Device: {info['device']}")

    # Train
    print(f"\nTraining for {args.epochs} epochs (mode: {args.mode})...")

    if args.mode == "angle":
        print(f"  output_weight={args.output_weight}, angle_weight={args.angle_weight}")
        losses = train_rotation_distillation_angle_supervised(
            transformer=transformer,
            lda_projection=teacher,
            query_embeddings=train_queries,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            log_interval=20,
            output_weight=args.output_weight,
            angle_weight=args.angle_weight
        )
    else:
        print(f"  cosine_weight={args.cosine_weight}, angle_reg={args.angle_reg}")
        losses = train_rotation_distillation(
            transformer=transformer,
            lda_projection=teacher,
            query_embeddings=train_queries,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            log_interval=20,
            cosine_weight=args.cosine_weight,
            angle_reg_weight=args.angle_reg
        )

    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate_rotation_equivalence(transformer, teacher, test_queries)

    print(f"\n{'=' * 40}")
    print("Results:")
    print(f"  Mean MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
    print(f"  Mean Cosine Sim: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
    print(f"  Min Cosine Sim: {results['min_cosine_sim']:.4f}")
    print(f"  Max Cosine Sim: {results['max_cosine_sim']:.4f}")
    print(f"\nRotation Statistics:")
    print(f"  Mean |angle|: {results['angle_stats']['mean_abs']:.4f} rad ({np.degrees(results['angle_stats']['mean_abs']):.2f}°)")
    print(f"  Max |angle|: {results['angle_stats']['max_abs']:.4f} rad ({np.degrees(results['angle_stats']['max_abs']):.2f}°)")
    print(f"  Angle std: {results['angle_stats']['std']:.4f}")

    if 'scale_stats' in results:
        print(f"\nScale Statistics:")
        print(f"  Mean: {results['scale_stats']['mean']:.4f}")
        print(f"  Range: [{results['scale_stats']['min']:.4f}, {results['scale_stats']['max']:.4f}]")

    print(f"{'=' * 40}")

    # Interpretation
    if results['mean_cosine_sim'] > 0.95:
        print("\n✓ Excellent: Rotation transform closely approximates LDA projection")
    elif results['mean_cosine_sim'] > 0.90:
        print("\n✓ Good: Rotation transform reasonably approximates LDA projection")
    elif results['mean_cosine_sim'] > 0.80:
        print("\n~ Fair: Rotation transform partially approximates LDA projection")
    else:
        print("\n✗ Poor: Rotation transform does not well approximate LDA projection")
        print("  Consider increasing rotation planes or using per-dim scaling")

    # Compare to direct output transformer
    print(f"\nTransform interpretability:")
    print(f"  This model predicts {info['output_dim']} geometric parameters")
    print(f"  vs. direct method which predicts {384} output dimensions")

    # Save if requested
    if args.save:
        transformer.save(args.save)
        print(f"\nSaved transformer to {args.save}")

    if db is not None:
        db.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
