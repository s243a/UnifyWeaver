# PR: Rotational Teacher Distillation for Fast Orthogonal Inference

## Title
feat(training): Add rotational teacher with GPU acceleration and target caching

## Summary

This PR adds a rotational teacher for distilling the full federated model (logm/expm blending) into a fast orthogonal transformer. The key insight is that while the full rotational computation is expensive (~0.2/s on CPU), we can pre-compute targets once and cache them for instant retraining with different architectures.

## Key Features

### RotationalTeacher (PyTorch)
- **Bivector caching**: Pre-computes logm(W) for all clusters once, saves to disk (~300MB)
- **GPU acceleration**: Uses `torch.linalg.matrix_exp` for batched computation
- **Performance**: ~2/s on GPU (vs 0.2/s CPU without caching)

### JaxRotationalTeacher (JAX)
- **XLA compilation**: JIT-compiled matrix exponential
- **vmap vectorization**: Automatic batching across queries
- **Alternative backend**: Can be faster on some hardware configurations

### Target Caching
- **One-time computation**: Compute rotational targets once (~14 hours on CPU, ~1 hour GPU)
- **Instant retraining**: Subsequent runs with different architectures use cached targets
- **Cache size**: ~30MB for 10k samples

## Results

| Metric | Value |
|--------|-------|
| Hit@1 | 96.9% |
| Hit@5 | 100% |
| Hit@10 | 100% |
| Cosine Sim to Teacher | 0.49 |
| Training Time (with cache) | 18 seconds (5 epochs) |

The 0.49 cosine similarity to the rotational teacher may seem low, but the retrieval metrics (hit@k) are excellent, indicating the projection preserves what matters for retrieval.

## Usage Examples

```bash
# Train with rotational teacher (first run computes targets)
python scripts/train_orthogonal_codebook.py \
    --train \
    --federated-model models/pearltrees_federated_nomic.pkl \
    --build-canonical \
    --n-components 64 \
    --layers 3 \
    --teacher rotational \
    --epochs 50 \
    --target-cache models/rotational_targets_cache.npz \
    --save-transformer models/orthogonal_transformer_rotational.pt

# Subsequent runs are instant (uses cached targets)
python scripts/train_orthogonal_codebook.py \
    --train \
    --federated-model models/pearltrees_federated_nomic.pkl \
    --build-canonical \
    --n-components 128 \
    --layers 4 \
    --teacher rotational \
    --epochs 100 \
    --target-cache models/rotational_targets_cache.npz \
    --save-transformer models/orthogonal_transformer_v2.pt

# Use JAX backend (alternative)
python scripts/train_orthogonal_codebook.py \
    --train \
    --federated-model models/pearltrees_federated_nomic.pkl \
    --build-canonical \
    --teacher jax \
    --epochs 50
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Federated Model (.pkl)                                      │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Pre-compute     │    │ Bivector Cache  │                 │
│  │ logm(W) for all │───▶│ (.bivectors.npz)│                 │
│  │ 142 clusters    │    │ ~300MB          │                 │
│  └─────────────────┘    └─────────────────┘                 │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Compute targets │    │ Target Cache    │                 │
│  │ for all queries │───▶│ (.npz)          │                 │
│  │ using expm      │    │ ~30MB           │                 │
│  └─────────────────┘    └─────────────────┘                 │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                        │
│  │ Train Orthogonal│                                        │
│  │ Transformer     │ ← Fast (seconds with cache)            │
│  └─────────────────┘                                        │
│         │                                                    │
│         ▼                                                    │
│  Saved Model (.pt)                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Files Changed

- `scripts/train_orthogonal_codebook.py`
  - Added `RotationalTeacher` class with GPU acceleration
  - Added `JaxRotationalTeacher` class as alternative backend
  - Added `--target-cache` CLI argument
  - Added `--teacher jax` option
  - Updated docstrings with usage examples

## Cached Files (not in git)

- `models/pearltrees_federated_nomic.bivectors.npz` (~300MB) - Pre-computed logm(W)
- `models/rotational_targets_cache.npz` (~30MB) - Teacher target outputs

## Future Work / Proposals

### Checkpoint Saving During Target Computation
Currently, if target computation is interrupted, progress is lost. Could add:
```python
# Proposal: Save checkpoints every N samples
--checkpoint-interval 1000
--checkpoint-path models/targets_checkpoint.npz
```

### JAX GPU Optimization
The JAX implementation was only tested on CPU (GPU memory was occupied). Further testing needed:
- Compare JAX GPU vs PyTorch GPU performance
- Test with larger batch sizes
- Profile XLA compilation overhead vs execution speed

### AIC-Based Model Selection
As noted by the user: with hit@5 at 100%, adding more planes would violate AIC (adding complexity without improving the objective). However, more planes could help if:
- Fine-tuning on a different corpus
- Optimizing cross-entropy for specific downstream tasks
- Generalizing to out-of-distribution queries

## Co-authored-by

Co-authored-by: Claude <noreply@anthropic.com>
Co-authored-by: John William Creighton (s243a) <JohnCreighton_@hotmail.com>
