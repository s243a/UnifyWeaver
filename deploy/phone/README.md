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
# Basic search (merged tree view is default)
python infer_phone.py --query "Feynman Lectures" --top-k 10

# Flat list output (last 3 path levels only)
python infer_phone.py --query "Feynman Lectures" --list

# JSON output
python infer_phone.py --query "Feynman Lectures" --json

# Adjust alpha (0=semantic, 1=structural, default=0.7)
python infer_phone.py --query "Feynman Lectures" --alpha 0.5

# Explicit temp directory (auto-detected if not specified)
python infer_phone.py --query "Feynman Lectures" --tmpdir $PREFIX/tmp
```

## Dependencies

```
pip install numpy onnxruntime sentence-transformers einops
```
