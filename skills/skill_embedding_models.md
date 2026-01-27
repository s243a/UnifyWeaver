# Skill: Embedding Model Selection

Choose and configure embedding models for semantic search, Q/A mapping, and entropy computation across UnifyWeaver targets.

## When to Use

- User asks "which embedding model should I use?"
- User needs to set up embeddings for a new project
- User asks about nomic, MiniLM, BERT, or modernBERT
- User wants to compute entropy/probability scores
- User is deploying to Go, Rust, or C# and needs embeddings
- User has an older Linux and needs help with modernBERT venv

## Model Recommendations by Use Case

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Q/A Mapping (Procrustes)** | nomic-embed-text-v1.5 | Best asymmetric search quality |
| **Lightweight/Fast** | all-MiniLM-L6-v2 | Smallest, fastest inference |
| **Entropy/Probability** | BERT or modernBERT | Produces calibrated logits |
| **Fine-tuning** | BERT-base | Mature tooling, well documented |
| **Long context (>512 tokens)** | nomic-embed-text-v1.5 | 8192 token context |
| **Mobile (Android/iOS)** | nomic-embed-text-v1.5 | Works on modern phones (tested S24+) |

## Model Specifications

### nomic-ai/nomic-embed-text-v1.5 (Recommended for Q/A)

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Context | 8192 tokens |
| Speed | Moderate |
| Asymmetric prefixes | Yes (`search_query:`, `search_document:`) |

**Why best for Q/A mapping:** Nomic produces well-separated embeddings for queries vs documents. The asymmetric prefixes help the Procrustes projection learn cleaner mappings.

**Mobile support:** Despite its 768 dimensions, nomic runs well on modern smartphones. Tested successfully on Samsung S24+ for on-device semantic search.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

# For queries (what user asks)
query_emb = model.encode("search_query: How do I authenticate?")

# For documents (what you're searching)
doc_emb = model.encode("search_document: Authentication requires valid credentials...")
```

### all-MiniLM-L6-v2 (Lightweight)

| Property | Value |
|----------|-------|
| Dimensions | 384 |
| Context | 256 tokens |
| Speed | Very fast |
| RAM | ~0.1-0.2 GB |

**Best for:** Resource-constrained environments, microservices, quick prototyping.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("How do I log in?")
```

### BERT-base (Entropy/Probability)

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Context | 512 tokens |
| Speed | Moderate |
| Fine-tunable | Yes |

**Best for:** Computing entropy/probability scores via logits. For short text, regular BERT and modernBERT produce similar entropy values. They may diverge for longer text where modernBERT's architecture handles context better.

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Sample text", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    # Use last_hidden_state for embeddings
    # Or load as BertForMaskedLM for logits/entropy
```

### answerdotai/ModernBERT-base (Newer Architecture)

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Context | 8192 tokens |
| Speed | Fast (with Flash Attention) |
| Requirements | Python 3.10+, transformers >= 4.48.0 |

**Note on older Linux:** ModernBERT may require a virtual environment on older Ubuntu/Debian due to Python and transformers version requirements.

```bash
# Create venv for modernBERT on older systems
python3.10 -m venv ~/.venvs/modernbert
source ~/.venvs/modernbert/bin/activate
pip install torch transformers>=4.48.0 flash-attn --no-build-isolation
```

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')
```

## Target-Specific Implementation

### Python Runtime

Full-featured with GPU support. All models available.

```python
# Using the unified provider (auto-selects best backend)
from modernbert_embedding import create_embedding_provider

# Default: nomic-embed-text-v1.5 with auto device selection
provider = create_embedding_provider()

# Or specify model
provider = create_embedding_provider(model_name="all-MiniLM-L6-v2")

# Get embedding
embedding = provider.get_embedding("How do I query a database?")
```

**File:** `src/unifyweaver/targets/python_runtime/modernbert_embedding.py`

### Go Runtime

ONNX-based inference via Candle FFI. Supports BERT-family models.

```go
// Uses Candle (Rust) FFI for high-performance inference
import "unifyweaver/embedder"

config := embedder.EmbedderConfig{
    ModelPath:  "models/all-MiniLM-L6-v2-safetensors/model.safetensors",
    Dimensions: 384,
    MaxLength:  512,
    UseGPU:     false,
}

embedder, _ := embedder.NewEmbedder(config)
vector := embedder.Embed("How do I log in?")
```

**File:** `src/unifyweaver/targets/go_runtime/embedder/embedder_candle.go`

**Supported models:** all-MiniLM-L6-v2, BERT-base (via safetensors format)

### Rust Runtime

Native Candle implementation. Supports BERT and ModernBERT architectures.

```rust
use embedding::{EmbeddingProvider, ModelType};

// BERT model
let provider = EmbeddingProvider::with_device(
    "models/all-MiniLM-L6-v2-safetensors/model.safetensors",
    "models/all-MiniLM-L6-v2-safetensors/tokenizer.json",
    ModelType::Bert,
    Device::Cpu,
)?;

// Or ModernBERT
let provider = EmbeddingProvider::with_device(
    model_path,
    tokenizer_path,
    ModelType::ModernBert,
    Device::Cuda(0),
)?;

let embedding = provider.embed("Sample query")?;
```

**File:** `src/unifyweaver/targets/rust_runtime/embedding.rs`

**Supported models:** all-MiniLM-L6-v2, BERT-base, ModernBERT (via MODEL_TYPE env var)

### C# Query Runtime

ONNX Runtime inference. Primarily supports MiniLM.

```csharp
using UnifyWeaver.QueryRuntime;

var provider = new OnnxEmbeddingProvider(
    modelPath: "models/all-MiniLM-L6-v2.onnx",
    vocabPath: "models/vocab.txt",
    dimensions: 384,
    maxLength: 512
);

double[] embedding = provider.GetEmbedding("How do I log in?");
```

**File:** `src/unifyweaver/targets/csharp_query_runtime/OnnxEmbeddingProvider.cs`

**Supported models:** all-MiniLM-L6-v2 (ONNX format)

## Performance Comparison

From experiments in `sandbox/docs/proposals/SEARCH_METHODS_COMPARISON.md`:

| Method | Recall@1 | MRR | Notes |
|--------|----------|-----|-------|
| Direct Search (MiniLM) | 70.0% | 0.8100 | Baseline |
| Multi-Head + Nomic | **76.7%** | **0.8648** | +6.7% with Procrustes projection |

The improvement comes from Nomic's asymmetric prefixes combined with learned Procrustes projection.

## Entropy Source Selection

For hierarchy objective (J = D/(1+H)), choose entropy source based on needs:

| Source | Model | Speed | When to Use |
|--------|-------|-------|-------------|
| `fisher` | Any embedding | Fast | Default, uses geometric proxy |
| `logits` | BERT/ModernBERT | Slow | Need true Shannon entropy |

```bash
# Fisher entropy (fast, embedding-based)
python3 scripts/mindmap/hierarchy_objective.py \
  --entropy-source fisher \
  --tree hierarchy.json

# Logits entropy (accurate, needs BERT)
python3 scripts/mindmap/hierarchy_objective.py \
  --entropy-source logits \
  --entropy-model answerdotai/ModernBERT-base \
  --tree hierarchy.json
```

## Storage Format

All targets use NumPy `.npy` or `.npz` format for cross-platform compatibility:

```python
import numpy as np

# Save single embedding
np.save("embedding.npy", vector)

# Save multiple embeddings
np.savez("embeddings.npz",
         input_embeddings=input_embs,
         output_embeddings=output_embs)

# Load (works in Python, Go, Rust via libraries)
data = np.load("embeddings.npz")
```

## Quick Reference: Model Selection

```
Need Q/A semantic search?
  └─► nomic-embed-text-v1.5 (with Procrustes projection)

Need fast inference / low memory?
  └─► all-MiniLM-L6-v2

Need entropy/probability scores?
  └─► BERT-base or ModernBERT

Need to fine-tune on your data?
  └─► BERT-base (best tooling support)

Deploying to Go/Rust/C#?
  └─► all-MiniLM-L6-v2 (widest ONNX support)

Processing long documents (>512 tokens)?
  └─► nomic-embed-text-v1.5 or ModernBERT (8K context)
```

## Model Registry

Models are managed via the Model Registry. Query available models:

```bash
# List all embedding models
python3 -m unifyweaver.config.model_registry --list --type embedding

# Get model for a task
python3 -m unifyweaver.config.model_registry --task bookmark_filing

# Check which environment to use (numpy 2.x compatibility)
python3 -m unifyweaver.config.model_registry --smart-envs
```

### Environment Requirements

Models saved with numpy 2.x require Python 3.9+ environments:

```bash
# Find compatible environment
python3 -m unifyweaver.config.model_registry --env-for pearltrees_federated_nomic

# Set up environment with required packages
python3 -m unifyweaver.config.model_registry --setup-env default --dry-run
```

See `skill_model_registry.md` for full registry documentation.

## Related

**Parent Skill:**
- `skill_ml_tools.md` - ML tools sub-master

**Sibling Skills:**
- `skill_model_registry.md` - Model discovery and selection
- `skill_train_model.md` - Training with embeddings
- `skill_hierarchy_objective.md` - Entropy computation
- `skill_semantic_inference.md` - Running inference
- `skill_density_explorer.md` - Visualization

**Documentation:**
- `docs/proposals/MINIMAL_TRANSFORMATION_PROJECTION.md` - Procrustes theory
- `sandbox/docs/proposals/SEARCH_METHODS_COMPARISON.md` - Method comparison

**Education (in `education/` subfolder):**
- `book-14-ai-training/02_embedding_providers.md` - Embedding fundamentals
- `book-14-ai-training/01_introduction.md` - Asymmetric semantics problem

**Code:**
- `src/unifyweaver/targets/python_runtime/modernbert_embedding.py` - Python provider
- `src/unifyweaver/targets/rust_runtime/embedding.rs` - Rust provider
- `src/unifyweaver/targets/go_runtime/embedder/` - Go provider
- `src/unifyweaver/targets/csharp_query_runtime/OnnxEmbeddingProvider.cs` - C# provider
