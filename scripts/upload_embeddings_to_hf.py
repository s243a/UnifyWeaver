#!/usr/bin/env python3
"""
Upload embeddings cache to Hugging Face Hub.

Naming convention:
    {dataset}_{model}_{dim}d_v{version}_{date}.npz

Example:
    tailored_all-minilm_384d_v1_2025-12-25.npz
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)

# Configuration
CACHE_DIR = Path("/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver/context/embeddings_cache")
REPO_ID = "s243a/unifyweaver-embeddings"
VERSION = "1"
DATE = datetime.now().strftime("%Y-%m-%d")

# Model metadata
MODEL_INFO = {
    "all-minilm": {"dim": 384, "full_name": "all-MiniLM-L6-v2"},
    "modernbert": {"dim": 768, "full_name": "nomic-embed-text-v1.5"},
}


def get_new_filename(old_name: str) -> str:
    """Convert cache filename to versioned filename.

    Old: embeddings_all-minilm_93580adb4bd7.npz
    New: tailored_all-minilm_384d_v1_2025-12-25.npz
    """
    # Parse old name
    parts = old_name.replace(".npz", "").split("_")
    # embeddings_all-minilm_hash -> model is parts[1]
    model = parts[1]

    info = MODEL_INFO.get(model, {"dim": "unknown"})
    dim = info["dim"]

    # Dataset is "tailored" based on how we generated the cache
    dataset = "tailored"

    return f"{dataset}_{model}_{dim}d_v{VERSION}_{DATE}.npz"


def create_readme() -> str:
    """Generate README for the HF repo."""
    return f"""---
license: mit
tags:
  - embeddings
  - semantic-search
  - question-answering
  - prolog
  - code-generation
  - rag
language:
  - en
size_categories:
  - n<1K
---

# UnifyWeaver Embeddings Cache

Pre-computed Q/A embeddings for semantic search and per-pair routing experiments.

## Project Repositories

UnifyWeaver is a declarative code generation system. The project spans three repositories:

| Repository | Description | Link |
|------------|-------------|------|
| **UnifyWeaver** | Main compiler and runtime (Prolog → Python/C#/Go/Rust/Bash) | [github.com/s243a/UnifyWeaver](https://github.com/s243a/UnifyWeaver) |
| **UnifyWeaver_Education** | Books, tutorials, and theory documentation | [github.com/s243a/UnifyWeaver_Education](https://github.com/s243a/UnifyWeaver_Education) |
| **UnifyWeaver_training-data** | Q/A pairs for semantic search training | [github.com/s243a/UnifyWeaver_training-data](https://github.com/s243a/UnifyWeaver_training-data) |

## Files

| File | Model | Dimensions | Dataset | Pairs |
|------|-------|------------|---------|-------|
| `tailored_all-minilm_384d_v{VERSION}_{DATE}.npz` | all-MiniLM-L6-v2 | 384 | tailored | 644 |
| `tailored_modernbert_768d_v{VERSION}_{DATE}.npz` | nomic-embed-text-v1.5 | 768 | tailored | 644 |

### File Contents

Each `.npz` file contains:
- `q_embeddings`: Question vectors (N × dim)
- `a_embeddings`: Answer vectors (N × dim)
- `cluster_ids`: Cluster labels for each pair
- `pair_ids`: Original pair IDs

## Downloading Embeddings

```python
import numpy as np
from huggingface_hub import hf_hub_download

# Download the MiniLM embeddings
path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="tailored_all-minilm_384d_v{VERSION}_{DATE}.npz"
)

# Load
data = np.load(path)
q_embeddings = data["q_embeddings"]  # (644, 384)
a_embeddings = data["a_embeddings"]  # (644, 384)
cluster_ids = data["cluster_ids"].tolist()
pair_ids = data["pair_ids"].tolist()

print(f"Loaded {{len(q_embeddings)}} Q/A pairs")
```

## Building the Data

### Prerequisites

```bash
pip install sentence-transformers numpy huggingface_hub
```

### Step 1: Clone the training data

```bash
git clone https://github.com/s243a/UnifyWeaver_training-data.git
```

### Step 2: Generate embeddings

```python
from training_data_loader import load_and_embed_with_cache

# Generate and cache embeddings
qa_embeddings, cluster_ids, pair_ids = load_and_embed_with_cache(
    data_dir="./UnifyWeaver_training-data",
    embedder_name="all-minilm",  # or "modernbert"
    subdirs=["tailored"],
    cache_dir="./embeddings_cache",
    force_recompute=False,  # Set True to regenerate
)
```

### Step 3: Upload to Hugging Face

```bash
# Login
huggingface-cli login

# Run upload script (from UnifyWeaver repo)
python scripts/upload_embeddings_to_hf.py
```

## Performance

Caching provides **200-1000x speedup** for iterative development:

| Model | First Run | Cached | Speedup |
|-------|-----------|--------|---------|
| all-MiniLM-L6-v2 (384d) | ~7s | 0.03s | ~230x |
| nomic-embed-text-v1.5 (768d) | ~36s | 0.03s | ~1200x |

## Per-Pair Routing Results

These embeddings were used to train per-pair Procrustes routing (Q→A transforms):

| Model | MRR | R@1 | R@5 | R@10 | Pool Size |
|-------|-----|-----|-----|------|-----------|
| all-MiniLM | 0.77 | 62% | 94% | 97% | 644 |
| ModernBERT | 0.90 | 81% | 99% | 100% | 644 |

See `per_pair_routing.py` in the main repo for implementation.

## Routing Method

These embeddings use **per-pair softmax routing** with minimal transformation:

1. **Train**: Learn an orthogonal transform $R_i$ for each Q/A cluster via Procrustes alignment
2. **Route**: Transform query $q$ through each cluster's transform, compute similarities
3. **Rank**: Softmax over similarities to rank candidate answers

$$\\hat{{a}} = R_i \\cdot q$$

This minimal approach (rotation only, no learned parameters) achieves strong results with limited data.
Alternative approaches like LDA topic models would require significantly more training data to match performance.

### Key Documents
- [Book 13: Semantic Search](https://github.com/s243a/UnifyWeaver_Education/tree/main/book-13-semantic-search)

## Training Data Structure

The training data repo contains:

```
UnifyWeaver_training-data/
├── tailored/           # 644 curated Q/A pairs (used here)
├── expanded/           # Additional generated pairs
├── tailored-gemini/    # Gemini-generated variants
├── book-01-foundations/  # Educational content
├── book-13-semantic-search/
└── ...
```

## Naming Convention

Files follow the pattern:
```
{{dataset}}_{{model}}_{{dim}}d_v{{version}}_{{date}}.npz
```

Example: `tailored_all-minilm_384d_v1_2025-12-25.npz`

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1 | {DATE} | Initial release with tailored dataset (644 pairs) |

## License

MIT - See [UnifyWeaver repository](https://github.com/s243a/UnifyWeaver) for details.

## Citation

```bibtex
@software{{unifyweaver2025,
  author = {{Creighton, John William}},
  title = {{UnifyWeaver: Declarative Data Integration with Semantic Search}},
  year = {{2025}},
  url = {{https://github.com/s243a/UnifyWeaver}}
}}
```
"""


def main():
    api = HfApi()

    # Create repo if needed
    print(f"Creating/verifying repo: {REPO_ID}")
    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Prepare files
    upload_dir = CACHE_DIR / "upload_staging"
    upload_dir.mkdir(exist_ok=True)

    files_to_upload = []

    for cache_file in CACHE_DIR.glob("embeddings_*.npz"):
        new_name = get_new_filename(cache_file.name)
        new_path = upload_dir / new_name

        print(f"  {cache_file.name} -> {new_name}")
        shutil.copy(cache_file, new_path)
        files_to_upload.append(new_path)

    # Create README
    readme_path = upload_dir / "README.md"
    readme_path.write_text(create_readme())
    files_to_upload.append(readme_path)

    # Upload
    print(f"\nUploading {len(files_to_upload)} files to {REPO_ID}...")

    for file_path in files_to_upload:
        print(f"  Uploading {file_path.name}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    print(f"\nDone! View at: https://huggingface.co/datasets/{REPO_ID}")

    # Cleanup staging
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    main()
