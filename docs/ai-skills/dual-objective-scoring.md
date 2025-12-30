# Dual-Objective Scoring for Bookmark Filing

## Overview

The dual-objective scoring system combines two complementary signals for bookmark filing:

1. **Input Objective (Semantic)**: Pure title-to-title similarity using raw embeddings
2. **Output Objective (Structural)**: Functional query matching using trained embeddings

## How It Works

### Input Objective
- **Model**: MiniLM (or ModernBERT when available)
- **Target embeddings**: Raw folder titles (e.g., "Quantum Mechanics")
- **Query**: Raw bookmark title (e.g., "Introduction to Quantum Physics")
- **Purpose**: Catch semantic matches regardless of organizational structure

### Output Objective
- **Model**: Nomic v1.5
- **Target embeddings**: Functional queries (e.g., `locate_node('Quantum Mechanics')`)
- **Query**: Raw bookmark title
- **Purpose**: Match against learned organizational patterns

### Blending

Scores are blended using L1 probability normalization:

```
p_input = ReLU(input_scores) / sum(ReLU(input_scores))
p_output = ReLU(output_scores) / sum(ReLU(output_scores))
final = alpha * p_output + (1 - alpha) * p_input
```

Default `alpha = 0.7` (70% structural, 30% semantic).

## Usage

### Command Line
```bash
# Basic search
python3 scripts/test_dual_objective.py --query "Hyphanet" --top-k 10

# Adjust blending
python3 scripts/test_dual_objective.py --query "Hyphanet" --alpha 0.5  # Equal blend
python3 scripts/test_dual_objective.py --query "Hyphanet" --alpha 1.0  # Pure structural
python3 scripts/test_dual_objective.py --query "Hyphanet" --alpha 0.0  # Pure semantic
```

### Python API
```python
from scripts.bookmark_filing_assistant import get_dual_objective_candidates

tree_output, candidates = get_dual_objective_candidates(
    "Feynman Lectures",
    top_k=10,
    alpha=0.7
)

print(tree_output)  # Merged tree with ranks
for c in candidates:
    print(f"#{c['rank']} {c['title']} [Score: {c['score']:.6f}]")
```

### MCP Tool
The `get_dual_objective_candidates` tool is available via the MCP server:
```json
{
  "name": "get_dual_objective_candidates",
  "arguments": {
    "bookmark_title": "Introduction to Quantum Computing",
    "top_k": 10,
    "alpha": 0.7
  }
}
```

## When to Use

| Scenario | Recommended Method |
|----------|-------------------|
| Standard filing | Federated projection (`infer_pearltrees_federated.py`) |
| Suspect misfile | Dual-objective with lower alpha (0.5) |
| Quick semantic search | Dual-objective with alpha=0 |
| Ambiguous terms | Dual-objective (blends both signals) |

## Generating Embeddings

To regenerate dual embeddings after data changes:

```bash
python3 scripts/generate_dual_embeddings.py \
    --data reports/pearltrees_targets_full_pearls.jsonl \
    --output models/dual_embeddings_full.npz
```

This takes ~2 minutes for 38k items (vs ~30+ minutes for federated model training).

## Model Notes

**Input Objective**: Uses MiniLM (384-dim)
- Fallback for ModernBERT (requires Transformers 4.48+)
- Fast and sufficient since input dimension is smaller
- ~90MB model, ~5x faster than Nomic

**Output Objective**: Uses Nomic v1.5 (768-dim)
- Full capacity for structured functional queries like `locate_node(...)`
- ~400MB model

To upgrade to ModernBERT when available:
```bash
pip install transformers>=4.48
python3 scripts/generate_dual_embeddings.py \
    --alt-model nomic-ai/modernbert-embed-base \
    --data reports/pearltrees_targets_full_pearls.jsonl \
    --output models/dual_embeddings_full.npz
```

## Files Needed for Inference

**Scripts:**
- `scripts/test_dual_objective.py` - Main inference script

**Generated Data (from generate_dual_embeddings.py):**
- `models/dual_embeddings_full.npz` (262 MB) - Pre-computed embeddings
- `reports/pearltrees_targets_full_pearls.jsonl` (26 MB) - Paths for display

**Embedding Models (downloaded on first run):**
- `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) - For Input Objective
- `nomic-ai/nomic-embed-text-v1.5` (~400 MB) - For Output Objective

## Files

- `scripts/generate_dual_embeddings.py` - Generate embeddings
- `scripts/test_dual_objective.py` - Test and query
- `scripts/bookmark_filing_assistant.py` - Python API (includes `get_dual_objective_candidates`)
- `scripts/mcp_bookmark_filing_server.py` - MCP interface
- `models/dual_embeddings_full.npz` - Pre-computed embeddings
- `docs/theory/hybrid_scoring_theory.md` - Mathematical theory
