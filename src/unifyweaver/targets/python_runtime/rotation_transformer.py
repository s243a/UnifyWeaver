# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Rotation-based transformer for minimal geometric distillation.

Instead of predicting output vectors directly, this transformer predicts
rotation angles and scaling factors that transform the input to the output.

Key insight: Many embedding transformations can be approximated as rotations
in high-dimensional space. By constraining the transformer to output rotation
parameters, we get:
- More interpretable transformations
- Geometric constraints (rotations preserve norms)
- Potentially better generalization

The transformation is: output = scale * R(θ₁, θ₂, ..., θₖ) @ input

Where R is composed of k Givens rotations on selected 2D planes.

Usage:
    from rotation_transformer import RotationTransformer, train_rotation_distillation

    # Create transformer with 64 rotation planes and uniform scaling
    transformer = RotationTransformer(
        embed_dim=384,
        num_rotation_planes=64,
        num_heads=4,
        num_layers=2,
        scaling_mode="uniform"
    )

    # Train via distillation from LDA
    train_rotation_distillation(transformer, lda_projection, training_queries)

    # Get transformation parameters for interpretation
    params = transformer.get_transform_params(query)
    print(f"Rotation angles: {params['angles']}")
    print(f"Scale factor: {params['scale']}")
"""

import math
import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_optimal_givens_angle(
    x_i: float, x_j: float,
    y_i: float, y_j: float
) -> float:
    """
    Compute the optimal Givens rotation angle for a single 2D plane.

    Given input components (x_i, x_j) and target components (y_i, y_j),
    find θ that minimizes:
        (cos(θ)x_i - sin(θ)x_j - y_i)² + (sin(θ)x_i + cos(θ)x_j - y_j)²

    The solution uses the fact that this is equivalent to finding the angle
    between (x_i, x_j) and (y_i, y_j) in the 2D plane.

    Args:
        x_i, x_j: Input vector components in the rotation plane
        y_i, y_j: Target vector components in the rotation plane

    Returns:
        Optimal rotation angle in radians
    """
    # The optimal angle aligns the input 2D vector with the target 2D vector
    # θ = angle(y) - angle(x) in the (i,j) plane

    x_angle = math.atan2(x_j, x_i)
    y_angle = math.atan2(y_j, y_i)

    theta = y_angle - x_angle

    # Normalize to [-π, π]
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi

    return theta


def compute_optimal_rotation_params(
    input_vec: np.ndarray,
    target_vec: np.ndarray,
    plane_indices: List[Tuple[int, int]],
    compute_scale: bool = True,
    method: str = "plane_projection"
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Compute optimal Givens rotation angles and scale to transform input to target.

    Methods:
    - "plane_projection" (default): Project overall rotation onto Givens planes.
      Fast O(k) - computes the rotation in input-target plane, then projects.
    - "iterative": Greedy sequential optimization of each plane.
      Slower O(k*d) but may be more accurate for complex transforms.

    Args:
        input_vec: Input vector (embed_dim,)
        target_vec: Target vector (embed_dim,)
        plane_indices: List of (i, j) dimension pairs for Givens rotations
        compute_scale: Whether to compute optimal scale factor
        method: "plane_projection" (fast) or "iterative" (accurate)

    Returns:
        Tuple of:
            - angles: Optimal rotation angles (num_planes,)
            - scale: Optimal scale factor, or None if compute_scale=False
    """
    # Compute scale first
    input_norm = np.linalg.norm(input_vec)
    target_norm = np.linalg.norm(target_vec)

    if compute_scale and input_norm > 1e-8:
        scale = target_norm / input_norm
    else:
        scale = 1.0 if compute_scale else None

    # Work with normalized vectors for angle computation
    if input_norm > 1e-8:
        input_normalized = input_vec / input_norm
    else:
        input_normalized = input_vec

    if target_norm > 1e-8:
        target_normalized = target_vec / target_norm
    else:
        target_normalized = target_vec

    if method == "plane_projection":
        # FAST: Compute overall rotation angle, then project to Givens planes
        # The rotation happens primarily in the plane spanned by input and target

        # Overall rotation angle
        cos_theta = np.clip(np.dot(input_normalized, target_normalized), -1, 1)
        overall_theta = math.acos(cos_theta)

        # For each Givens plane, compute how much rotation projects onto it
        # This is based on the 2D projection of input and target onto each plane
        angles = []
        for (i, j) in plane_indices:
            # 2D projections
            input_2d = np.array([input_normalized[i], input_normalized[j]])
            target_2d = np.array([target_normalized[i], target_normalized[j]])

            # Angle in this 2D plane
            input_angle = math.atan2(input_2d[1], input_2d[0])
            target_angle = math.atan2(target_2d[1], target_2d[0])

            theta = target_angle - input_angle

            # Normalize to [-π, π]
            while theta > math.pi:
                theta -= 2 * math.pi
            while theta < -math.pi:
                theta += 2 * math.pi

            angles.append(theta)

        return np.array(angles, dtype=np.float32), scale

    else:  # iterative method
        # SLOW but accurate: Sequential greedy optimization
        if compute_scale and scale is not None and scale > 1e-8:
            effective_target = target_vec / scale
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

            # Apply this rotation to current vector
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
    compute_scale: bool = True,
    method: str = "plane_projection"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Batch version of compute_optimal_rotation_params.

    Args:
        input_vecs: Input vectors (batch, embed_dim)
        target_vecs: Target vectors (batch, embed_dim)
        plane_indices: List of (i, j) dimension pairs
        compute_scale: Whether to compute scale factors
        method: "plane_projection" (fast, vectorized) or "iterative"

    Returns:
        Tuple of:
            - angles: (batch, num_planes)
            - scales: (batch,) or None
    """
    batch_size = len(input_vecs)
    num_planes = len(plane_indices)

    if method == "plane_projection":
        # FAST VECTORIZED: Compute all angles in parallel
        logger.info(f"  Computing rotations (vectorized): {batch_size} samples, {num_planes} planes...")

        # Compute scales
        input_norms = np.linalg.norm(input_vecs, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target_vecs, axis=1, keepdims=True)

        if compute_scale:
            all_scales = (target_norms / (input_norms + 1e-8)).squeeze()
        else:
            all_scales = None

        # Normalize
        input_normalized = input_vecs / (input_norms + 1e-8)
        target_normalized = target_vecs / (target_norms + 1e-8)

        # Compute angles for each plane (vectorized over batch)
        all_angles = np.zeros((batch_size, num_planes), dtype=np.float32)

        for k, (i, j) in enumerate(plane_indices):
            # Extract 2D components for all samples
            input_i = input_normalized[:, i]
            input_j = input_normalized[:, j]
            target_i = target_normalized[:, i]
            target_j = target_normalized[:, j]

            # Compute angles
            input_angle = np.arctan2(input_j, input_i)
            target_angle = np.arctan2(target_j, target_i)

            theta = target_angle - input_angle

            # Normalize to [-π, π]
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            all_angles[:, k] = theta

        logger.info(f"  Done computing {batch_size * num_planes:,} rotation angles")
        return all_angles, all_scales

    else:
        # Fallback to sequential (for iterative method)
        all_angles = np.zeros((batch_size, num_planes), dtype=np.float32)
        all_scales = np.zeros(batch_size, dtype=np.float32) if compute_scale else None

        log_interval = max(1, batch_size // 10)
        for b in range(batch_size):
            angles, scale = compute_optimal_rotation_params(
                input_vecs[b], target_vecs[b], plane_indices, compute_scale, method="iterative"
            )
            all_angles[b] = angles
            if compute_scale:
                all_scales[b] = scale

            if (b + 1) % log_interval == 0 or b == batch_size - 1:
                logger.info(f"  Computing rotations: {b+1}/{batch_size} ({100*(b+1)/batch_size:.0f}%)")

        return all_angles, all_scales

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

    A Givens rotation operates on a single 2D plane within the n-dimensional space,
    rotating vectors by angle θ in that plane while leaving all other dimensions fixed.

    Multiple Givens rotations can be composed for more expressive transformations.
    Any rotation matrix can be decomposed into at most n(n-1)/2 Givens rotations.

    For practical use, we select k << n(n-1)/2 rotation planes to keep the
    parameterization tractable while still capturing important transformations.
    """

    def __init__(self, embed_dim: int, num_rotation_planes: int, device="cpu"):
        """
        Args:
            embed_dim: Dimension of input/output vectors
            num_rotation_planes: Number of 2D rotation planes to use
            device: Torch device
        """
        _import_torch()
        self.embed_dim = embed_dim
        self.num_rotation_planes = num_rotation_planes
        self.device = device

        # Pre-select rotation planes (pairs of dimensions)
        self.plane_indices = self._select_rotation_planes()

    def _select_rotation_planes(self) -> List[Tuple[int, int]]:
        """
        Select which 2D planes to rotate in.

        Strategy: Cover the space by using interleaved dimension pairs.
        Start with adjacent pairs (0,1), (2,3), ...
        Then offset pairs (1,2), (3,4), ...
        Then larger strides to capture longer-range correlations.
        """
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

    def to(self, device):
        """Move to device."""
        self.device = device
        return self

    def forward(self, x: "torch.Tensor", angles: "torch.Tensor") -> "torch.Tensor":
        """
        Apply Givens rotations to input vectors.

        Args:
            x: Input vectors (batch, embed_dim)
            angles: Rotation angles in radians (batch, num_rotation_planes)

        Returns:
            Rotated vectors (batch, embed_dim)
        """
        result = x.clone()

        for k, (i, j) in enumerate(self.plane_indices):
            theta = angles[:, k]  # (batch,)
            cos_t = torch.cos(theta)  # (batch,)
            sin_t = torch.sin(theta)  # (batch,)

            xi = result[:, i]
            xj = result[:, j]

            # Givens rotation: [cos -sin; sin cos] applied to (xi, xj)
            result[:, i] = cos_t * xi - sin_t * xj
            result[:, j] = sin_t * xi + cos_t * xj

        return result

    def __call__(self, x, angles):
        return self.forward(x, angles)


class RotationTransformer:
    """
    Transformer that predicts rotation angles and scaling for minimal transforms.

    Instead of directly predicting the output vector, this model predicts:
    - k rotation angles for Givens rotations on selected planes
    - Optional: a scaling factor (uniform or per-dimension)

    The final output is: scale * R(angles) @ input

    This constrains the transformation to be geometric, potentially
    leading to better generalization and interpretability.

    Attributes:
        embed_dim: Input/output embedding dimension
        num_rotation_planes: Number of Givens rotation planes
        num_heads: Attention heads per transformer layer
        num_layers: Number of transformer layers
        scaling_mode: "uniform", "per_dim", or "none"
        output_dim: Number of parameters predicted (angles + scale)
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_rotation_planes: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        use_scaling: bool = True,
        scaling_mode: str = "uniform",
        dropout: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize RotationTransformer.

        Args:
            embed_dim: Embedding dimension (must match input data)
            num_rotation_planes: Number of Givens rotation planes (k angles)
            num_heads: Attention heads per layer
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            use_scaling: Whether to predict scaling factors
            scaling_mode: "uniform" (1 scalar), "per_dim" (embed_dim scalars), "none"
            dropout: Dropout rate
            device: Device ("auto", "cuda", "cpu")

        Example:
            # 64 rotation planes + uniform scaling = 65 output parameters
            transformer = RotationTransformer(
                embed_dim=384,
                num_rotation_planes=64,
                scaling_mode="uniform"
            )
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

        # Givens rotation layer
        self.rotation_layer = GivensRotationLayer(
            self.embed_dim,
            self.num_rotation_planes,
            device=self.device
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training."""
        for module in [self.input_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Initialize param_head to output small angles initially
        # This means starting close to identity transform
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

        # Constrain angles to (-π, π) using tanh
        angles = torch.tanh(angles) * math.pi

        if self.num_scale_params > 0:
            raw_scale = params[:, self.num_angles:]  # (batch, num_scale_params)
            # Use softplus to ensure positive scaling, centered around 1
            scale = F.softplus(raw_scale) + 0.5
        else:
            scale = None

        # Apply rotation
        rotated = self.rotation_layer(query, angles)

        # Apply scaling
        if scale is not None:
            if self.scaling_mode == "uniform":
                projected = rotated * scale  # broadcast
            else:  # per_dim
                projected = rotated * scale  # element-wise
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

        This is useful for interpreting what geometric transformation
        the model applies to a given input.

        Args:
            query_emb: Query embedding (embed_dim,) or (batch, embed_dim)

        Returns:
            Dict containing:
                - 'angles': Rotation angles in radians
                - 'rotation_planes': List of (i,j) dimension pairs
                - 'scale': Scale factor(s) if scaling enabled
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
        """Save model weights and configuration."""
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
    teacher_projection,
    query_embeddings: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 10,
    cosine_weight: float = 0.5,
    angle_reg_weight: float = 0.01,
) -> List[float]:
    """
    Train rotation transformer via knowledge distillation.

    Args:
        transformer: RotationTransformer to train
        teacher_projection: Teacher model with .project() method
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

    # Generate target projections from teacher
    n_samples = len(query_embeddings)
    logger.info(f"Generating teacher target projections for {n_samples} samples...")
    teacher_outputs = []
    log_interval = max(1, n_samples // 10)
    for i in range(n_samples):
        proj = teacher_projection.project(query_embeddings[i])
        if isinstance(proj, tuple):
            proj = proj[0]
        teacher_outputs.append(proj)
        if (i + 1) % log_interval == 0:
            logger.info(f"  Teacher projections: {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")
    teacher_outputs = np.stack(teacher_outputs)

    # Convert to tensors
    queries_tensor = torch.from_numpy(query_embeddings).float().to(transformer.device)
    targets_tensor = torch.from_numpy(teacher_outputs).float().to(transformer.device)

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


def train_rotation_distillation_angle_supervised(
    transformer: RotationTransformer,
    teacher_projection,
    query_embeddings: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 10,
    output_weight: float = 0.5,
    angle_weight: float = 0.5,
    scale_weight: float = 0.1,
) -> List[float]:
    """
    Train rotation transformer with direct angle supervision.

    Instead of only supervising on output vectors, this computes the optimal
    rotation angles for each input-target pair and uses those as training targets.

    Loss = output_weight * output_loss + angle_weight * angle_loss + scale_weight * scale_loss

    Where:
    - output_loss: MSE + cosine on final output vectors
    - angle_loss: MSE between predicted and optimal angles
    - scale_loss: MSE between predicted and optimal scale

    Args:
        transformer: RotationTransformer to train
        teacher_projection: Teacher model with .project() method
        query_embeddings: Training query embeddings (N, embed_dim)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_interval: Log every N epochs
        output_weight: Weight for output vector loss
        angle_weight: Weight for angle prediction loss
        scale_weight: Weight for scale prediction loss

    Returns:
        List of loss values per epoch
    """
    _import_torch()

    # Generate target projections from teacher
    n_samples = len(query_embeddings)
    logger.info(f"Generating teacher target projections for {n_samples} samples...")
    teacher_outputs = []
    log_interval = max(1, n_samples // 10)
    for i in range(n_samples):
        proj = teacher_projection.project(query_embeddings[i])
        if isinstance(proj, tuple):
            proj = proj[0]
        teacher_outputs.append(proj)
        if (i + 1) % log_interval == 0:
            logger.info(f"  Teacher projections: {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")
    teacher_outputs = np.stack(teacher_outputs)

    # Compute optimal rotation parameters for all samples
    logger.info("Computing optimal rotation parameters...")
    plane_indices = transformer.rotation_layer.plane_indices
    compute_scale = transformer.scaling_mode != "none"

    optimal_angles, optimal_scales = compute_optimal_rotation_params_batch(
        query_embeddings, teacher_outputs, plane_indices, compute_scale
    )

    # Convert to tensors
    queries_tensor = torch.from_numpy(query_embeddings).float().to(transformer.device)
    targets_tensor = torch.from_numpy(teacher_outputs).float().to(transformer.device)
    optimal_angles_tensor = torch.from_numpy(optimal_angles).float().to(transformer.device)

    if optimal_scales is not None:
        optimal_scales_tensor = torch.from_numpy(optimal_scales).float().to(transformer.device)
        if transformer.scaling_mode == "uniform":
            optimal_scales_tensor = optimal_scales_tensor.unsqueeze(1)  # (batch, 1)
    else:
        optimal_scales_tensor = None

    # Optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    # Training loop
    transformer.train_mode()
    losses = []
    n_samples = len(query_embeddings)

    logger.info(
        f"Training for {num_epochs} epochs, {n_samples} samples, "
        f"output_weight={output_weight}, angle_weight={angle_weight}, scale_weight={scale_weight}"
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

            # Forward
            predicted, pred_angles, pred_scale = transformer.forward(batch_queries)

            # Output loss (MSE + cosine)
            mse_loss = F.mse_loss(predicted, batch_targets)
            pred_norm = F.normalize(predicted, p=2, dim=1)
            target_norm = F.normalize(batch_targets, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
            output_loss = 0.5 * mse_loss + 0.5 * (1 - cosine_sim)

            # Angle loss - direct supervision on rotation angles
            # Use circular loss to handle angle wraparound
            angle_diff = pred_angles - batch_opt_angles
            # Wrap to [-π, π]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            angle_loss = torch.mean(angle_diff ** 2)

            # Scale loss
            if optimal_scales_tensor is not None and pred_scale is not None:
                batch_opt_scales = optimal_scales_tensor[idx]
                scale_loss = F.mse_loss(pred_scale, batch_opt_scales)
            else:
                scale_loss = torch.tensor(0.0, device=transformer.device)

            # Combined loss
            loss = (
                output_weight * output_loss +
                angle_weight * angle_loss +
                scale_weight * scale_loss
            )

            # Backward
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
                f"Scale: {epoch_scale/n_batches:.6f}, "
                f"Cos: {cosine_sim.item():.4f}"
            )

    transformer.eval_mode()
    logger.info(f"Training complete. Final loss: {losses[-1]:.6f}")

    return losses


def evaluate_rotation_equivalence(
    transformer: RotationTransformer,
    teacher_projection,
    test_embeddings: np.ndarray
) -> dict:
    """
    Evaluate how well rotation transformer approximates teacher.

    Args:
        transformer: Trained RotationTransformer
        teacher_projection: Teacher model with .project() method
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
        # Teacher projection
        teacher_out = teacher_projection.project(query_emb)
        if isinstance(teacher_out, tuple):
            teacher_out = teacher_out[0]

        # Transformer projection
        trans_out = transformer.project(query_emb)
        params = transformer.get_transform_params(query_emb)

        all_angles.append(params['angles'])
        if 'scale' in params:
            all_scales.append(params['scale'])

        # MSE
        mse = np.mean((teacher_out - trans_out) ** 2)
        mse_values.append(mse)

        # Cosine similarity
        cos_sim = np.dot(teacher_out, trans_out) / (
            np.linalg.norm(teacher_out) * np.linalg.norm(trans_out) + 1e-8
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


class MinimalTransformProjection:
    """
    Multi-head projection that interpolates rotation parameters instead of output vectors.

    Standard approach (vector interpolation):
        output = Σ wᵢ × answer_embᵢ

    Minimal transform approach (rotation interpolation):
        angles = Σ wᵢ × anglesᵢ
        scale = Σ wᵢ × scaleᵢ
        output = scale × R(angles) @ input

    This keeps the transformation minimal even when blending mappings from
    different clusters - we stay in the rotation manifold rather than
    jumping to arbitrary points in embedding space.

    Each cluster head stores:
    - centroid: for computing routing weights
    - angles: rotation angles for this cluster's transformation
    - scale: scale factor for this cluster's transformation

    This is no longer LDA - it's a geometric/minimal transform projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_rotation_planes: int,
        temperature: float = 1.0
    ):
        """
        Initialize MinimalTransformProjection.

        Args:
            embed_dim: Embedding dimension
            num_rotation_planes: Number of Givens rotation planes
            temperature: Softmax temperature for routing
        """
        self.embed_dim = embed_dim
        self.num_rotation_planes = num_rotation_planes
        self.temperature = temperature

        # Initialize rotation layer to get plane indices
        self.rotation_layer = GivensRotationLayer(embed_dim, num_rotation_planes)

        # Cluster heads: list of {centroid, angles, scale}
        self.heads = []

    def add_head(
        self,
        centroid: np.ndarray,
        target_emb: np.ndarray,
        reference_input: Optional[np.ndarray] = None
    ):
        """
        Add a cluster head by computing optimal rotation from input to target.

        Args:
            centroid: Cluster centroid for routing
            target_emb: Target embedding (answer embedding)
            reference_input: Input to compute rotation from. If None, uses centroid.
        """
        if reference_input is None:
            reference_input = centroid

        # Compute optimal rotation parameters
        angles, scale = compute_optimal_rotation_params(
            reference_input,
            target_emb,
            self.rotation_layer.plane_indices,
            compute_scale=True
        )

        self.heads.append({
            'centroid': centroid.copy(),
            'angles': angles,
            'scale': scale if scale is not None else 1.0,
            'target_emb': target_emb.copy(),  # Keep for reference/debugging
        })

    def add_head_direct(
        self,
        centroid: np.ndarray,
        angles: np.ndarray,
        scale: float = 1.0
    ):
        """
        Add a cluster head with pre-computed rotation parameters.

        Args:
            centroid: Cluster centroid for routing
            angles: Pre-computed rotation angles
            scale: Pre-computed scale factor
        """
        self.heads.append({
            'centroid': centroid.copy(),
            'angles': angles.copy(),
            'scale': scale,
            'target_emb': None,
        })

    def compute_weights(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Compute softmax routing weights for a query.

        Args:
            query_emb: Query embedding

        Returns:
            Routing weights (num_heads,)
        """
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

        return weights

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Project query using rotation-interpolated multi-head.

        Blends rotation parameters across heads, then applies the blended
        rotation to the input.

        Args:
            query_emb: Query embedding (embed_dim,)

        Returns:
            Projected embedding (embed_dim,)
        """
        weights = self.compute_weights(query_emb)

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

    def project_with_info(self, query_emb: np.ndarray) -> dict:
        """
        Project query and return detailed information.

        Args:
            query_emb: Query embedding

        Returns:
            Dict with projected, weights, blended_angles, blended_scale
        """
        weights = self.compute_weights(query_emb)

        blended_angles = np.zeros(self.num_rotation_planes, dtype=np.float32)
        blended_scale = 0.0

        for i, head in enumerate(self.heads):
            blended_angles += weights[i] * head['angles']
            blended_scale += weights[i] * head['scale']

        # Apply rotation
        result = query_emb.copy()
        for k, (i, j) in enumerate(self.rotation_layer.plane_indices):
            theta = blended_angles[k]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            xi = result[i]
            xj = result[j]
            result[i] = cos_t * xi - sin_t * xj
            result[j] = sin_t * xi + cos_t * xj

        result = result * blended_scale

        return {
            'projected': result,
            'weights': weights,
            'blended_angles': blended_angles,
            'blended_scale': blended_scale,
        }

    @classmethod
    def from_lda_heads(
        cls,
        centroids: List[np.ndarray],
        answer_embs: List[np.ndarray],
        num_rotation_planes: int,
        temperature: float = 1.0,
        reference_inputs: Optional[List[np.ndarray]] = None
    ) -> "MinimalTransformProjection":
        """
        Create from existing LDA cluster heads.

        Args:
            centroids: List of cluster centroids
            answer_embs: List of answer embeddings
            num_rotation_planes: Number of rotation planes
            temperature: Routing temperature
            reference_inputs: Optional reference inputs for rotation computation.
                            If None, uses centroids.

        Returns:
            MinimalTransformProjection instance
        """
        if len(centroids) == 0:
            raise ValueError("Need at least one cluster head")

        embed_dim = len(centroids[0])
        proj = cls(embed_dim, num_rotation_planes, temperature)

        for i, (centroid, answer_emb) in enumerate(zip(centroids, answer_embs)):
            ref_input = reference_inputs[i] if reference_inputs else None
            proj.add_head(centroid, answer_emb, ref_input)

        return proj

    def get_info(self) -> dict:
        """Get projection information."""
        return {
            'embed_dim': self.embed_dim,
            'num_rotation_planes': self.num_rotation_planes,
            'num_heads': len(self.heads),
            'temperature': self.temperature,
        }


def compare_projection_methods(
    query_emb: np.ndarray,
    minimal_proj: MinimalTransformProjection,
    standard_proj,  # Object with .project() that does vector interpolation
) -> dict:
    """
    Compare minimal transform vs standard vector-interpolated projection.

    Args:
        query_emb: Query to project
        minimal_proj: MinimalTransformProjection instance
        standard_proj: Standard projection with .project() method

    Returns:
        Comparison metrics
    """
    min_result = minimal_proj.project_with_info(query_emb)
    std_result = standard_proj.project(query_emb)
    if isinstance(std_result, tuple):
        std_result = std_result[0]

    min_out = min_result['projected']

    # Compare outputs
    cos_sim = np.dot(min_out, std_result) / (
        np.linalg.norm(min_out) * np.linalg.norm(std_result) + 1e-8
    )
    mse = np.mean((min_out - std_result) ** 2)

    # Check norm preservation
    input_norm = np.linalg.norm(query_emb)
    min_norm = np.linalg.norm(min_out)
    std_norm = np.linalg.norm(std_result)

    return {
        'cosine_similarity': cos_sim,
        'mse': mse,
        'input_norm': input_norm,
        'minimal_output_norm': min_norm,
        'standard_output_norm': std_norm,
        'blended_scale': min_result['blended_scale'],
        'weights': min_result['weights'],
        'blended_angles_mean': np.mean(np.abs(min_result['blended_angles'])),
    }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== RotationTransformer Test ===\n")

    # Create rotation transformer
    print("Creating RotationTransformer (64 planes, uniform scaling)...")
    transformer = RotationTransformer(
        embed_dim=384,
        num_rotation_planes=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        scaling_mode="uniform",
        device="cpu"
    )
    info = transformer.get_info()
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Output dim: {info['output_dim']} (64 angles + 1 scale)")
    print(f"  Rotation planes: {len(transformer.rotation_layer.plane_indices)}")
    print(f"  First 5 planes: {transformer.rotation_layer.plane_indices[:5]}")

    # Test forward pass
    print("\nTesting forward pass...")
    test_query = np.random.randn(384).astype(np.float32)
    projected = transformer.project(test_query)
    params = transformer.get_transform_params(test_query)

    print(f"  Input shape: {test_query.shape}")
    print(f"  Output shape: {projected.shape}")
    print(f"  Input norm: {np.linalg.norm(test_query):.4f}")
    print(f"  Output norm: {np.linalg.norm(projected):.4f}")
    print(f"  Angles range: [{params['angles'].min():.4f}, {params['angles'].max():.4f}]")
    print(f"  Scale: {params['scale']:.4f}")

    # Test batch
    print("\nTesting batch forward pass...")
    batch_queries = np.random.randn(8, 384).astype(np.float32)
    batch_projected = transformer.project(batch_queries)
    print(f"  Batch input shape: {batch_queries.shape}")
    print(f"  Batch output shape: {batch_projected.shape}")

    # Verify rotation preserves norms (approximately, before scaling)
    print("\nVerifying rotation properties...")
    transformer_no_scale = RotationTransformer(
        embed_dim=384,
        num_rotation_planes=64,
        scaling_mode="none",
        device="cpu"
    )
    proj_no_scale = transformer_no_scale.project(test_query)
    print(f"  Without scaling - Input norm: {np.linalg.norm(test_query):.4f}")
    print(f"  Without scaling - Output norm: {np.linalg.norm(proj_no_scale):.4f}")
    print(f"  Norm preserved: {abs(np.linalg.norm(test_query) - np.linalg.norm(proj_no_scale)) < 0.01}")

    print("\n=== Test Complete ===")
