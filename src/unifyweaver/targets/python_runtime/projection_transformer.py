# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Transformer-based projection for knowledge distillation from LDA multi-head.

This module implements a compact transformer that learns to approximate
the LDA multi-head projection function using fewer total parameters.

Key insight: H^L ≈ N_flat (heads per layer ^ layers ≈ flat LDA heads)

Usage:
    from projection_transformer import ProjectionTransformer, train_distillation

    # Create transformer to approximate 64 flat heads (H=4, L=3 → 4^3=64)
    transformer = ProjectionTransformer(
        embed_dim=384,
        num_heads=4,
        num_layers=3,
        ff_dim=512
    )

    # Train via distillation from LDA
    train_distillation(transformer, lda_projection, training_queries)
"""

import math
import logging
from typing import List, Optional, Tuple

import numpy as np

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


class ProjectionTransformer:
    """
    Transformer that learns to approximate LDA multi-head projection.

    Architecture:
    - Input projection
    - L transformer layers (each with H attention heads + FFN)
    - Output: projected embedding

    Capacity equivalence conjecture: H^L ≈ N_flat LDA heads
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize ProjectionTransformer.

        Args:
            embed_dim: Embedding dimension (must match LDA)
            num_heads: Attention heads per layer (H)
            num_layers: Number of transformer layers (L)
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout rate
            device: Device ("auto", "cuda", "cpu")

        Capacity: H^L flat LDA heads equivalent
        Example: num_heads=4, num_layers=3 → 64 flat heads
        """
        _import_torch()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        # Theoretical capacity
        self.equivalent_flat_heads = num_heads ** num_layers

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
            f"ProjectionTransformer: H={num_heads}, L={num_layers}, "
            f"equivalent_flat_heads={self.equivalent_flat_heads}, device={self.device}"
        )

    def _build_model(self):
        """Build the transformer model."""
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

        # Output projection (optional, for fine-tuning output space)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training."""
        for module in [self.input_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, query: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.

        Args:
            query: Input query embeddings (batch, embed_dim)

        Returns:
            Projected embeddings (batch, embed_dim)
        """
        # Add sequence dimension for transformer (batch, 1, embed_dim)
        x = self.input_proj(query).unsqueeze(1)

        # Pass through transformer encoder
        x = self.encoder(x)

        # Remove sequence dimension and project output
        x = x.squeeze(1)
        x = self.output_proj(x)

        return x

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
            projected = self.forward(query_tensor)
            result = projected.cpu().numpy()

        if was_1d:
            result = result[0]

        return result

    def parameters(self):
        """Return all trainable parameters."""
        params = list(self.input_proj.parameters())
        params += list(self.encoder.parameters())
        params += list(self.output_proj.parameters())
        return params

    def train_mode(self):
        """Set to training mode."""
        self.input_proj.train()
        self.encoder.train()
        self.output_proj.train()

    def eval_mode(self):
        """Set to evaluation mode."""
        self.input_proj.eval()
        self.encoder.eval()
        self.output_proj.eval()

    def save(self, path: str):
        """Save model weights."""
        state = {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate,
            'input_proj': self.input_proj.state_dict(),
            'encoder': self.encoder.state_dict(),
            'output_proj': self.output_proj.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "ProjectionTransformer":
        """Load model from file."""
        _import_torch()
        state = torch.load(path, map_location='cpu')

        model = cls(
            embed_dim=state['embed_dim'],
            num_heads=state['num_heads'],
            num_layers=state['num_layers'],
            ff_dim=state['ff_dim'],
            dropout=state['dropout'],
            device=device
        )
        model.input_proj.load_state_dict(state['input_proj'])
        model.encoder.load_state_dict(state['encoder'])
        model.output_proj.load_state_dict(state['output_proj'])

        logger.info(f"Loaded model from {path}")
        return model

    def get_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'equivalent_flat_heads': self.equivalent_flat_heads,
            'total_parameters': total_params,
            'device': str(self.device),
        }

    def export_onnx(self, path: str, opset_version: int = 14) -> dict:
        """
        Export model to ONNX format for browser deployment.

        Args:
            path: Output path for .onnx file
            opset_version: ONNX opset version (default: 14 for broad compatibility)

        Returns:
            dict with export info (path, size, input/output shapes)
        """
        _import_torch()

        self.eval_mode()

        # Create dummy input matching expected shape
        dummy_input = torch.randn(1, self.embed_dim).to(self.device)

        # Create a wrapper module that combines all components
        class ONNXWrapper(nn.Module):
            def __init__(self, input_proj, encoder, output_proj):
                super().__init__()
                self.input_proj = input_proj
                self.encoder = encoder
                self.output_proj = output_proj

            def forward(self, x):
                x = self.input_proj(x).unsqueeze(1)
                x = self.encoder(x)
                x = x.squeeze(1)
                x = self.output_proj(x)
                return x

        wrapper = ONNXWrapper(self.input_proj, self.encoder, self.output_proj)
        wrapper.eval()

        # Export to ONNX
        torch.onnx.export(
            wrapper,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['query_embedding'],
            output_names=['projected_embedding'],
            dynamic_axes={
                'query_embedding': {0: 'batch_size'},
                'projected_embedding': {0: 'batch_size'}
            }
        )

        # Get file size
        import os
        file_size = os.path.getsize(path)

        info = {
            'path': path,
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'opset_version': opset_version,
            'input_shape': ['batch_size', self.embed_dim],
            'output_shape': ['batch_size', self.embed_dim],
        }

        logger.info(f"Exported ONNX model to {path} ({info['size_mb']:.2f} MB)")
        return info

    def verify_onnx(self, onnx_path: str, test_input: Optional[np.ndarray] = None, rtol: float = 1e-4) -> dict:
        """
        Verify ONNX export matches PyTorch output.

        Args:
            onnx_path: Path to exported .onnx file
            test_input: Optional test input (default: random)
            rtol: Relative tolerance for comparison

        Returns:
            dict with verification results
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed, skipping verification")
            return {'verified': False, 'error': 'onnxruntime not installed'}

        _import_torch()

        # Create test input
        if test_input is None:
            test_input = np.random.randn(5, self.embed_dim).astype(np.float32)

        # Get PyTorch output
        self.eval_mode()
        with torch.no_grad():
            pt_input = torch.from_numpy(test_input).to(self.device)
            pt_output = self.forward(pt_input).cpu().numpy()

        # Get ONNX output
        session = ort.InferenceSession(onnx_path)
        onnx_output = session.run(None, {'query_embedding': test_input})[0]

        # Compare
        max_diff = np.max(np.abs(pt_output - onnx_output))
        mean_diff = np.mean(np.abs(pt_output - onnx_output))
        matches = np.allclose(pt_output, onnx_output, rtol=rtol)

        result = {
            'verified': matches,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'rtol': rtol,
            'test_samples': len(test_input)
        }

        if matches:
            logger.info(f"ONNX verification passed (max_diff={max_diff:.6f})")
        else:
            logger.warning(f"ONNX verification failed (max_diff={max_diff:.6f})")

        return result


def train_distillation(
    transformer: ProjectionTransformer,
    lda_projection,  # MultiHeadProjection or similar with .project() method
    query_embeddings: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 10,
    cosine_weight: float = 0.5
) -> List[float]:
    """
    Train transformer via knowledge distillation from LDA projection.

    Args:
        transformer: ProjectionTransformer to train
        lda_projection: LDA multi-head projection (teacher)
        query_embeddings: Training query embeddings (N, embed_dim)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_interval: Log every N epochs
        cosine_weight: Weight for cosine similarity loss (0=MSE only, 1=cosine only)

    Returns:
        List of loss values per epoch
    """
    _import_torch()

    # Generate target projections from teacher model
    logger.info("Generating teacher model projections...")
    lda_outputs = []
    for i in range(len(query_embeddings)):
        proj = lda_projection.project(query_embeddings[i])
        if isinstance(proj, tuple):
            proj = proj[0]  # Handle (projected, weights) return
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
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_queries = queries_tensor[idx]
            batch_targets = targets_tensor[idx]

            # Forward
            predicted = transformer.forward(batch_queries)

            # Combined MSE + cosine loss
            mse_loss = F.mse_loss(predicted, batch_targets)

            # Cosine loss (1 - cosine_sim)
            pred_norm = F.normalize(predicted, p=2, dim=1)
            target_norm = F.normalize(batch_targets, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
            cosine_loss = 1 - cosine_sim

            # Combined loss
            loss = (1 - cosine_weight) * mse_loss + cosine_weight * cosine_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    transformer.eval_mode()
    logger.info(f"Training complete. Final loss: {losses[-1]:.6f}")

    return losses


def evaluate_equivalence(
    transformer: ProjectionTransformer,
    lda_projection,
    test_embeddings: np.ndarray
) -> dict:
    """
    Evaluate how well transformer approximates LDA projection.

    Args:
        transformer: Trained ProjectionTransformer
        lda_projection: LDA multi-head projection
        test_embeddings: Test query embeddings

    Returns:
        Dict with MSE, cosine similarity, etc.
    """
    transformer.eval_mode()

    mse_values = []
    cosine_sims = []

    for query_emb in test_embeddings:
        # LDA projection
        lda_out = lda_projection.project(query_emb)
        if isinstance(lda_out, tuple):
            lda_out = lda_out[0]

        # Transformer projection
        trans_out = transformer.project(query_emb)

        # MSE
        mse = np.mean((lda_out - trans_out) ** 2)
        mse_values.append(mse)

        # Cosine similarity
        cos_sim = np.dot(lda_out, trans_out) / (
            np.linalg.norm(lda_out) * np.linalg.norm(trans_out) + 1e-8
        )
        cosine_sims.append(cos_sim)

    return {
        'mean_mse': np.mean(mse_values),
        'std_mse': np.std(mse_values),
        'mean_cosine_sim': np.mean(cosine_sims),
        'std_cosine_sim': np.std(cosine_sims),
        'min_cosine_sim': np.min(cosine_sims),
        'max_cosine_sim': np.max(cosine_sims),
        'n_samples': len(test_embeddings),
    }


def optimal_architecture(n_flat_heads: int, prefer_h: int = 4, embed_dim: int = 768) -> Tuple[int, int]:
    """
    Calculate optimal H (heads) and L (layers) for target flat head count.

    Uses H^L >= N constraint (must have sufficient capacity), minimizing
    total parameters while ensuring the architecture can represent N clusters.

    Considers H values that divide embed_dim (PyTorch requirement).
    E.g., for N=124 with embed_dim=768: 6^3=216 or 4^4=256.

    Args:
        n_flat_heads: Target equivalent flat LDA heads
        prefer_h: Preferred heads per layer (default 4, used as tiebreaker)
        embed_dim: Embedding dimension (H must divide this)

    Returns:
        (num_heads, num_layers) tuple
    """
    if n_flat_heads <= 1:
        return (1, 1)

    # Only consider H values that divide embed_dim (PyTorch MultiheadAttention requirement)
    valid_h = [h for h in [2, 3, 4, 6, 8, 12, 16] if embed_dim % h == 0]

    # Consider multiple H values to find optimal H^L >= N
    candidates = []
    for h in valid_h:
        # Find minimum L such that h^L >= N
        l = max(1, math.ceil(math.log(n_flat_heads) / math.log(h)))
        equiv = h ** l
        # Must have equiv >= N
        if equiv >= n_flat_heads:
            overhead = equiv - n_flat_heads  # How much over N
            # Prefer h == prefer_h as tiebreaker
            tiebreaker = 0 if h == prefer_h else 1
            candidates.append((l, overhead, tiebreaker, h, l))

    if not candidates:
        # Fallback: use prefer_h with ceil
        l = max(1, math.ceil(math.log(n_flat_heads) / math.log(prefer_h)))
        return (prefer_h, l)

    # Sort by: layers (fewer better), overhead (smaller better), tiebreaker
    candidates.sort()
    _, _, _, best_h, best_l = candidates[0]

    return (best_h, best_l)


def generate_candidate_architectures(
    n_clusters: int,
    embed_dim: int = 768,
    num_suggestions: int = 3
) -> List[dict]:
    """
    Generate candidate transformer architectures for distillation.

    Produces diverse architectures with different depth/width trade-offs.

    Args:
        n_clusters: Number of clusters in federated model (minimum capacity)
        embed_dim: Embedding dimension (H must divide this)
        num_suggestions: Number of candidates to return

    Returns:
        List of dicts with 'num_heads', 'num_layers', 'capacity', 'params_estimate'
    """
    # Valid H values that divide embed_dim
    valid_h = [h for h in [2, 3, 4, 6, 8, 12, 16] if embed_dim % h == 0]

    # Generate all valid (H, L) pairs with H^L >= n_clusters
    candidates = []
    for h in valid_h:
        # Find minimum L
        min_l = max(1, math.ceil(math.log(n_clusters) / math.log(h)))
        # Consider min_l through min_l + 2 for more diversity
        for l in range(min_l, min(min_l + 3, 9)):
            capacity = h ** l
            if capacity >= n_clusters:
                # Estimate parameters (rough approximation)
                # TransformerEncoderLayer: ~4 * d^2 per layer (attention + FFN)
                params = l * 4 * embed_dim * embed_dim + 2 * embed_dim * embed_dim
                overhead = capacity - n_clusters
                candidates.append({
                    'num_heads': h,
                    'num_layers': l,
                    'capacity': capacity,
                    'overhead': overhead,
                    'overhead_pct': 100.0 * overhead / n_clusters,
                    'params_estimate': params,
                })

    # Remove duplicates
    seen = set()
    unique = []
    for c in candidates:
        key = (c['num_heads'], c['num_layers'])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique

    # Sort by layers (prefer fewer), then by overhead
    candidates.sort(key=lambda c: (c['num_layers'], c['overhead']))

    # Select diverse candidates across different layer counts
    # Strategy: prioritize one candidate per layer count, then fill remaining
    selected = []
    by_layers = {}
    for c in candidates:
        l = c['num_layers']
        if l not in by_layers:
            by_layers[l] = []
        by_layers[l].append(c)

    # Pick best (lowest overhead) from each layer count
    for l in sorted(by_layers.keys()):
        if len(selected) >= num_suggestions:
            break
        selected.append(by_layers[l][0])

    # If we need more, add second-best from each layer count
    if len(selected) < num_suggestions:
        for l in sorted(by_layers.keys()):
            if len(selected) >= num_suggestions:
                break
            if len(by_layers[l]) > 1:
                selected.append(by_layers[l][1])

    return selected[:num_suggestions]


def compute_aic_student_t(
    residuals: np.ndarray,
    n_params: int,
    df: float = None
) -> dict:
    """
    Compute AIC using Student's t-distribution for residuals.

    Theory:
    -------
    Standard AIC assumes Gaussian errors: AIC = 2k - 2ln(L)

    For Gaussian with MLE variance σ² = MSE:
        -2ln(L) = n·ln(2πσ²) + n = n·ln(MSE) + const
        AIC_gaussian = n·ln(MSE) + 2k

    Student's t is more robust to outliers. The log-likelihood for
    standardized residuals z_i = (r_i - μ)/σ under t(ν) is:

        ln L = Σ ln(t_pdf(z_i; ν))

    where t_pdf includes the gamma function terms. The key insight
    is that the RELEVANCE of each error depends on σ:
    - Same absolute error is less significant if σ is large
    - Student's t has heavier tails, so outliers penalize less than Gaussian

    We estimate degrees of freedom ν from the data using the method of moments:
        ν̂ = 2σ⁴ / (σ⁴ - σ²) for excess kurtosis

    Args:
        residuals: Model residuals (predicted - actual), shape (n,) or (n, d)
        n_params: Number of model parameters (k)
        df: Degrees of freedom for t-distribution. If None, estimated from data.

    Returns:
        Dict with AIC scores and diagnostic info
    """
    from scipy import stats

    # Flatten if multi-dimensional
    if residuals.ndim > 1:
        residuals = residuals.flatten()

    n = len(residuals)
    mu = np.mean(residuals)
    sigma = np.std(residuals)

    # Standardize residuals
    z = (residuals - mu) / (sigma + 1e-10)

    # Estimate degrees of freedom if not provided
    if df is None:
        # Method of moments using kurtosis
        # For t(ν): kurtosis = 6/(ν-4) for ν > 4
        # So ν = 4 + 6/kurtosis
        kurtosis = stats.kurtosis(z)  # Excess kurtosis
        if kurtosis > 0.1:
            df = max(3, 4 + 6 / kurtosis)
        else:
            df = 30  # Effectively Gaussian for low kurtosis
        df = min(df, 100)  # Cap at 100

    # Compute log-likelihood under Student's t
    log_likelihood = np.sum(stats.t.logpdf(z, df=df))

    # Also compute Gaussian log-likelihood for comparison
    log_likelihood_gaussian = np.sum(stats.norm.logpdf(z))

    # AIC: 2k - 2ln(L)
    aic_t = 2 * n_params - 2 * log_likelihood
    aic_gaussian = 2 * n_params - 2 * log_likelihood_gaussian

    # BIC: k·ln(n) - 2ln(L)
    bic_t = n_params * np.log(n) - 2 * log_likelihood
    bic_gaussian = n_params * np.log(n) - 2 * log_likelihood_gaussian

    # Normalized versions (per-sample) for easier comparison
    aic_t_per_sample = aic_t / n
    aic_gaussian_per_sample = aic_gaussian / n

    return {
        'aic_t': aic_t,
        'aic_gaussian': aic_gaussian,
        'bic_t': bic_t,
        'bic_gaussian': bic_gaussian,
        'aic_t_per_sample': aic_t_per_sample,
        'aic_gaussian_per_sample': aic_gaussian_per_sample,
        'log_likelihood_t': log_likelihood,
        'log_likelihood_gaussian': log_likelihood_gaussian,
        'df_estimated': df,
        'residual_mean': mu,
        'residual_std': sigma,
        'n_samples': n,
        'n_params': n_params,
    }


def compute_aic_cosine(
    cosine_sims: np.ndarray,
    n_params: int,
    df: float = None
) -> dict:
    """
    Compute AIC using cosine similarity transformed to residuals.

    Transforms cosine similarity to a loss-like quantity for AIC:
        cosine_loss = 1 - cosine_sim  (range [0, 2])

    Then applies Student's t AIC to this loss.

    Args:
        cosine_sims: Cosine similarities between predicted and target, shape (n,)
        n_params: Number of model parameters
        df: Degrees of freedom (None = estimate from data)

    Returns:
        Dict with AIC scores and diagnostics
    """
    # Transform to loss (lower is better)
    cosine_loss = 1 - cosine_sims

    # Use as "residuals" for AIC computation
    result = compute_aic_student_t(cosine_loss, n_params, df=df)

    # Add cosine-specific stats
    result['mean_cosine_sim'] = float(np.mean(cosine_sims))
    result['std_cosine_sim'] = float(np.std(cosine_sims))
    result['mean_cosine_loss'] = float(np.mean(cosine_loss))

    return result


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== ProjectionTransformer Test ===\n")

    # Test architecture calculation
    print("Optimal architectures for various N:")
    for n in [16, 64, 256, 512, 1000]:
        h, l = optimal_architecture(n)
        actual = h ** l
        print(f"  N={n}: H={h}, L={l}, actual={actual}")

    print()

    # Create small transformer
    print("Creating transformer (H=4, L=2, equivalent=16 flat heads)...")
    transformer = ProjectionTransformer(
        embed_dim=384,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
        device="cpu"
    )
    info = transformer.get_info()
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Equivalent flat heads: {info['equivalent_flat_heads']}")

    # Test forward pass
    print("\nTesting forward pass...")
    test_query = np.random.randn(384).astype(np.float32)
    projected = transformer.project(test_query)
    print(f"  Input shape: {test_query.shape}")
    print(f"  Output shape: {projected.shape}")
    print(f"  Output norm: {np.linalg.norm(projected):.4f}")

    # Test batch
    print("\nTesting batch forward pass...")
    batch_queries = np.random.randn(8, 384).astype(np.float32)
    batch_projected = transformer.project(batch_queries)
    print(f"  Batch input shape: {batch_queries.shape}")
    print(f"  Batch output shape: {batch_projected.shape}")

    print("\n=== Test Complete ===")
