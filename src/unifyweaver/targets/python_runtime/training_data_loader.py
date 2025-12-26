"""
Load training data from JSONL files and convert to embeddings.

Supports caching embeddings to disk to avoid recomputation.

## Embeddings Cache

Embeddings are cached to `/context/embeddings_cache/` to avoid re-running
the embedding model on each execution.

### Cache files
```
embeddings_{model}_{hash}.npz
```
Where `hash` is derived from (data_dir, subdirs, max_pairs, embedder_name).

### Contents
- `q_embeddings`: Question vectors (N × dim)
- `a_embeddings`: Answer vectors (N × dim)
- `cluster_ids`: Cluster labels for each pair
- `pair_ids`: Original pair IDs

### Speed improvement

Caching provides **200-1000x speedup** for iterative development:

| Model | First run | Cached | Speedup |
|-------|-----------|--------|---------|
| all-MiniLM | ~7s | 0.03s | ~230x |
| ModernBERT | ~36s | 0.03s | ~1200x |

### Usage
```python
from training_data_loader import load_and_embed_with_cache

qa_embeddings, cluster_ids, pair_ids = load_and_embed_with_cache(
    data_dir="/path/to/training-data",
    embedder_name="all-minilm",
    subdirs=["tailored"],
    max_pairs=None,  # or limit
    force_recompute=False,  # set True to regenerate
)
```
"""

import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

# Default cache directory
DEFAULT_CACHE_DIR = "/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver/context/embeddings_cache"


@dataclass
class QAPair:
    """A question-answer pair with metadata."""
    pair_id: str
    cluster_id: str
    question: str
    answer: str
    question_type: str = "medium"
    topics: List[str] = None


def load_jsonl_pairs(
    data_dir: str,
    max_pairs: Optional[int] = None,
    subdirs: Optional[List[str]] = None,
) -> List[QAPair]:
    """
    Load Q/A pairs from JSONL files.

    Args:
        data_dir: Root directory containing JSONL files
        max_pairs: Maximum number of pairs to load
        subdirs: Specific subdirectories to include (e.g., ["expanded", "tailored"])

    Returns:
        List of QAPair objects
    """
    data_path = Path(data_dir)
    pairs = []

    # Find all JSONL files
    if subdirs:
        jsonl_files = []
        for subdir in subdirs:
            subpath = data_path / subdir
            if subpath.exists():
                jsonl_files.extend(subpath.rglob("*.jsonl"))
    else:
        jsonl_files = list(data_path.rglob("*.jsonl"))

    for jsonl_file in sorted(jsonl_files):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    pair = QAPair(
                        pair_id=data.get("pair_id", ""),
                        cluster_id=data.get("cluster_id", ""),
                        question=data.get("question", ""),
                        answer=data.get("answer", ""),
                        question_type=data.get("question_type", "medium"),
                        topics=data.get("topics", []),
                    )
                    pairs.append(pair)

                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
                except json.JSONDecodeError:
                    continue

    return pairs


def embed_pairs(
    pairs: List[QAPair],
    embedder_name: str = "all-minilm",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str]]:
    """
    Embed Q/A pairs using sentence-transformers.

    Args:
        pairs: List of QAPair objects
        embedder_name: Name of the embedding model

    Returns:
        (qa_embeddings, cluster_ids) where qa_embeddings is list of (q_emb, a_emb) tuples
    """
    # Import here to avoid dependency if not needed
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    # Model mapping
    model_map = {
        "all-minilm": "all-MiniLM-L6-v2",
        "e5-small": "intfloat/e5-small-v2",
        "modernbert": "nomic-ai/nomic-embed-text-v1.5",
    }

    model_id = model_map.get(embedder_name, embedder_name)
    print(f"Loading model: {model_id}")
    model = SentenceTransformer(model_id, trust_remote_code=True)

    # Extract texts
    questions = [p.question for p in pairs]
    answers = [p.answer for p in pairs]
    cluster_ids = [p.cluster_id for p in pairs]

    # Embed
    print(f"Embedding {len(questions)} questions...")
    q_embeddings = model.encode(questions, show_progress_bar=True)

    print(f"Embedding {len(answers)} answers...")
    a_embeddings = model.encode(answers, show_progress_bar=True)

    # Pair up
    qa_embeddings = [(q, a) for q, a in zip(q_embeddings, a_embeddings)]

    return qa_embeddings, cluster_ids


def get_cache_key(
    data_dir: str,
    subdirs: Optional[List[str]],
    max_pairs: Optional[int],
    embedder_name: str,
) -> str:
    """Generate a unique cache key based on data configuration."""
    key_parts = [
        data_dir,
        ",".join(sorted(subdirs)) if subdirs else "all",
        str(max_pairs) if max_pairs else "all",
        embedder_name,
    ]
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def save_embeddings_cache(
    cache_path: Path,
    qa_embeddings: List[Tuple[np.ndarray, np.ndarray]],
    cluster_ids: List[str],
    pair_ids: List[str],
):
    """Save embeddings to disk cache."""
    q_embs = np.stack([q for q, _ in qa_embeddings])
    a_embs = np.stack([a for _, a in qa_embeddings])

    np.savez_compressed(
        cache_path,
        q_embeddings=q_embs,
        a_embeddings=a_embs,
        cluster_ids=np.array(cluster_ids),
        pair_ids=np.array(pair_ids),
    )
    print(f"Saved embeddings cache to {cache_path}")


def load_embeddings_cache(
    cache_path: Path,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str], List[str]]:
    """Load embeddings from disk cache."""
    data = np.load(cache_path, allow_pickle=True)

    q_embs = data["q_embeddings"]
    a_embs = data["a_embeddings"]
    cluster_ids = data["cluster_ids"].tolist()
    pair_ids = data["pair_ids"].tolist()

    qa_embeddings = [(q, a) for q, a in zip(q_embs, a_embs)]

    print(f"Loaded {len(qa_embeddings)} embeddings from cache")
    return qa_embeddings, cluster_ids, pair_ids


def load_and_embed_with_cache(
    data_dir: str,
    embedder_name: str = "all-minilm",
    subdirs: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str], List[str]]:
    """
    Load Q/A pairs and embed them, using cache if available.

    Args:
        data_dir: Root directory containing JSONL files
        embedder_name: Name of the embedding model
        subdirs: Specific subdirectories to include
        max_pairs: Maximum number of pairs to load
        cache_dir: Directory to store embedding caches
        force_recompute: If True, ignore cache and recompute

    Returns:
        (qa_embeddings, cluster_ids, pair_ids)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_key = get_cache_key(data_dir, subdirs, max_pairs, embedder_name)
    cache_file = cache_path / f"embeddings_{embedder_name}_{cache_key}.npz"

    # Try to load from cache
    if cache_file.exists() and not force_recompute:
        try:
            return load_embeddings_cache(cache_file)
        except Exception as e:
            print(f"Cache load failed: {e}, recomputing...")

    # Load and embed
    print(f"Loading data from {data_dir}...")
    pairs = load_jsonl_pairs(data_dir, max_pairs=max_pairs, subdirs=subdirs)
    print(f"Loaded {len(pairs)} Q/A pairs")

    if len(pairs) == 0:
        raise ValueError("No pairs loaded")

    qa_embeddings, cluster_ids = embed_pairs(pairs, embedder_name)
    pair_ids = [p.pair_id for p in pairs]

    # Save to cache
    save_embeddings_cache(cache_file, qa_embeddings, cluster_ids, pair_ids)

    return qa_embeddings, cluster_ids, pair_ids


def get_unique_answers(
    pairs: List[QAPair],
) -> Dict[str, str]:
    """
    Get unique answers by cluster_id.

    Returns:
        Dict mapping cluster_id to answer text
    """
    answers = {}
    for pair in pairs:
        if pair.cluster_id not in answers:
            answers[pair.cluster_id] = pair.answer
    return answers


if __name__ == "__main__":
    # Test loading
    data_dir = "/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver/training-data"

    print("Loading from expanded/...")
    pairs = load_jsonl_pairs(data_dir, subdirs=["expanded"], max_pairs=100)
    print(f"Loaded {len(pairs)} pairs")

    if pairs:
        print(f"\nExample pair:")
        print(f"  ID: {pairs[0].pair_id}")
        print(f"  Cluster: {pairs[0].cluster_id}")
        print(f"  Q: {pairs[0].question[:80]}...")
        print(f"  A: {pairs[0].answer[:80]}...")

        # Count unique clusters
        clusters = set(p.cluster_id for p in pairs)
        print(f"\nUnique clusters: {len(clusters)}")
