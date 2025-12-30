# Bookmark Filing - Phone Deployment

## Files Needed

### Option A: Dual-Objective Mode (Recommended for Phone)
- `models/dual_embeddings_full.npz` (262 MB) - Pre-computed target embeddings
- `models/all-MiniLM-L6-v2-onnx/` (88 MB) - ONNX model for query embedding
- `reports/pearltrees_targets_full_pearls.jsonl` (26 MB) - Paths for display
- `infer_phone.py` - Inference script

**Total: ~376 MB**

### Option B: Federated Hybrid Mode
- `models/pearltrees_federated_single.pkl` (400 KB)
- `models/pearltrees_federated_single/` (160 MB)
- `models/all-MiniLM-L6-v2-onnx/` (88 MB)
- `reports/pearltrees_targets_full_multi_account.jsonl` (3.8 MB)
- Nomic model not included (too large, would use API for Nomic queries)

**Total: ~252 MB** (but limited without Nomic)

## Usage

```bash
python infer_phone.py --query "Feynman Lectures" --top-k 5
```

## Dependencies

```
pip install numpy onnxruntime sentence-transformers
```
