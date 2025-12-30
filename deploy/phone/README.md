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

## Fuzzy Logic Boosting

When the embedding model doesn't recognize a term (e.g., "bash-reduce"), use fuzzy
boosting to emphasize specific concepts:

```bash
# Boost results matching "bash" (AND: multiply scores)
python infer_phone.py --query "bash-reduce" --boost-and "bash"

# Weighted AND boost (term:weight format)
python infer_phone.py --query "bash-reduce" --boost-and "bash:0.8,unix:0.5"

# Distributed OR boost: score AND (term1 OR term2 OR ...)
# Formula: 1 - (1 - score*w1*t1)(1 - score*w2*t2)...
python infer_phone.py --query "bash-reduce" --boost-or "bash:0.9,shell:0.5,scripting:0.3"
```

### Fuzzy Logic Operations

| Operation | Formula | Use Case |
|-----------|---------|----------|
| `--boost-and "a,b"` | `score * a * b` | Require all terms |
| `--boost-and "a:0.8,b:0.5"` | `score * 0.8*a * 0.5*b` | Weighted requirement |
| `--boost-or "a,b"` | `1 - (1-score*a)(1-score*b)` | Match any term (distributed) |
| `--boost-or "a:0.9,b:0.3"` | `1 - (1-score*0.9*a)(1-score*0.3*b)` | Prioritized alternatives |

Note: `--boost-or` distributes the base score into each term before the OR operation.
This means the result is effectively `score AND (t1 OR t2 OR ...)`.

## Subtree Filtering

Restrict search to a specific folder subtree:

```bash
# Only search within BASH folders
python infer_phone.py --query "reduce" --subtree "BASH (Unix/Linux)"
```

## Dependencies

```
pip install numpy onnxruntime sentence-transformers einops
```
