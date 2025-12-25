"""
Load training data from JSONL files and convert to embeddings.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


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
