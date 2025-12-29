#!/usr/bin/env python3
"""
Federated Pearltrees Inference.

Given a query (bookmark title/URL), returns top-k candidate folders
for filing the bookmark. Uses query-level routing with cluster-shared
Procrustes transforms.

Usage:
    python3 scripts/infer_pearltrees_federated.py \
        --model models/pearltrees_federated_single.pkl \
        --query "New bookmark about quantum computing" \
        --top-k 5

    # Interactive mode:
    python3 scripts/infer_pearltrees_federated.py \
        --model models/pearltrees_federated_single.pkl \
        --interactive
        
    # JSON output for piping to other tools:
    python3 scripts/infer_pearltrees_federated.py \
        --model models/pearltrees_federated_single.pkl \
        --query "quantum physics bookmark" \
        --json
"""

import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A candidate folder for filing a bookmark."""
    rank: int
    score: float
    tree_id: str
    title: str
    path: str
    cluster_id: str
    dataset_index: int = -1


class FederatedInferenceEngine:
    """
    Inference engine for federated Pearltrees model.
    
    Uses query-level routing: finds most similar training queries,
    then uses their cluster's W matrix to project the new query.
    """
    
    def __init__(self, model_path: Path, embedder_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_path = Path(model_path)
        self.embedder_name = embedder_name
        
        # Load model metadata
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            self.meta = pickle.load(f)
        
        self.cluster_ids = self.meta["cluster_ids"]
        self.cluster_centroids = self.meta["cluster_centroids"]
        self.global_target_ids = self.meta.get("global_target_ids", [])
        self.global_target_titles = self.meta.get("global_target_titles", [])
        self.temperature = self.meta.get("temperature", 0.1)
        
        # Determine cluster directory
        if "cluster_dir" in self.meta:
            self.cluster_dir = Path(self.meta["cluster_dir"])
        else:
            self.cluster_dir = self.model_path.with_suffix('')
        
        logger.info(f"Model has {len(self.cluster_ids)} clusters")
        
        # Load routing data
        self._load_routing_data()
        
        # Load cluster W matrices
        self._load_clusters()
        
        # Load embedder (lazy)
        self._embedder = None
    
    def _load_routing_data(self):
        """Load query embeddings and index-to-cluster mapping for routing."""
        routing_path = self.cluster_dir / "routing_data.npz"
        
        if routing_path.exists():
            logger.info(f"Loading routing data from {routing_path}...")
            data = np.load(routing_path)
            self.query_embeddings = data["query_embeddings"]
            self.target_embeddings = data["target_embeddings"]
            
            # Reconstruct index-to-cluster mapping
            keys = data["idx_to_cluster_keys"]
            values = data["idx_to_cluster_values"]
            self.idx_to_cluster = {int(k): str(v) for k, v in zip(keys, values)}
            
            logger.info(f"Loaded {len(self.query_embeddings)} query embeddings for routing")
        else:
            logger.warning(f"Routing data not found at {routing_path}, using cluster centroids")
            self.query_embeddings = self.cluster_centroids
            self.target_embeddings = None
            self.idx_to_cluster = {i: cid for i, cid in enumerate(self.cluster_ids)}
    
    def _load_clusters(self):
        """Load W matrices from each cluster."""
        self.clusters = {}
        
        for cid in self.cluster_ids:
            cluster_path = self.cluster_dir / f"{cid}.npz"
            if cluster_path.exists():
                data = np.load(cluster_path)
                self.clusters[cid] = {
                    "W": data["W_stack"][0],  # Single W per cluster
                    "target_embeddings": data["target_embeddings"],
                    "indices": data["indices"]
                }
        
        logger.info(f"Loaded {len(self.clusters)} cluster W matrices")
    
    @property
    def embedder(self):
        """Lazy load embedder on first use."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedder: {self.embedder_name}")
            self._embedder = SentenceTransformer(self.embedder_name, trust_remote_code=True)
        return self._embedder
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embedder.encode([query], show_progress_bar=False)[0].astype(np.float32)
    
    def project_query(self, q_emb: np.ndarray, top_k_routing: int = 10) -> np.ndarray:
        """
        Project query using query-level routing.
        
        1. Find top-k most similar training queries
        2. For each, look up its cluster's W
        3. Compute weighted projection
        """
        # Compute similarities to all training queries
        sims = q_emb @ self.query_embeddings.T
        
        # Softmax weights
        sims_shifted = (sims - np.max(sims)) / self.temperature
        weights = np.exp(sims_shifted)
        weights /= weights.sum()
        
        # Get top-k training queries
        top_indices = np.argsort(weights)[-top_k_routing:]
        
        # Weighted projection using their clusters' W
        proj = np.zeros_like(q_emb)
        for idx in top_indices:
            cid = self.idx_to_cluster.get(int(idx))
            if cid and cid in self.clusters:
                W = self.clusters[cid]["W"]
                proj += weights[idx] * (q_emb @ W)
        
        return proj
    
    def search(self, query: str, top_k: int = 5, top_k_routing: int = 10, alpha: float = 1.0) -> List[Candidate]:
        """
        Search for best folders to file a bookmark.
        
        Args:
            query: Bookmark title, URL, or description
            top_k: Number of candidates to return
            top_k_routing: Number of training queries to use for routing
            alpha: Weight for projected score (1.0 = projection only, 0.0 = raw only)
            
        Returns:
            List of Candidate objects with scores
        """
        # Embed query
        q_emb = self.embed_query(query)
        
        # Project using federated model (Projection Score)
        q_proj = self.project_query(q_emb, top_k_routing)
        
        # Normalize
        q_proj_norm = q_proj / (np.linalg.norm(q_proj) + 1e-8)
        
        # Compare to all targets
        if self.target_embeddings is not None:
            A_norm = self.target_embeddings / (np.linalg.norm(self.target_embeddings, axis=1, keepdims=True) + 1e-8)
            
            # Projected Score
            scores_proj = q_proj_norm @ A_norm.T
            
            # Blend logic
            if 0.0 < alpha < 1.0:
                # Calculate Raw Score
                q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                scores_raw = q_emb_norm @ A_norm.T
                
                # ReLU + L1 Normalization (Treat as Probability Mass)
                # Projected
                p_proj = np.maximum(scores_proj, 0)
                p_proj /= (p_proj.sum() + 1e-10)
                
                # Raw
                p_raw = np.maximum(scores_raw, 0)
                p_raw /= (p_raw.sum() + 1e-10)
                
                # Blend
                scores = alpha * p_proj + (1.0 - alpha) * p_raw
                
            elif alpha <= 0.0:
                 # Pure raw (Cosine)
                q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                scores = q_emb_norm @ A_norm.T
            else:
                # Pure projected (Cosine) - Default behavior
                scores = scores_proj
        else:
            # Fall back to cluster centroids (no raw fallback easily available/meaningful here)
            scores = q_proj_norm @ self.cluster_centroids.T
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build candidates
        candidates = []
        for rank, idx in enumerate(top_indices, 1):
            idx = int(idx)
            
            # Get cluster ID for this target
            cluster_id = self.idx_to_cluster.get(idx, "unknown")
            
            # Get title from stored titles
            if idx < len(self.global_target_titles):
                title = self.global_target_titles[idx]
            else:
                title = f"Target {idx}"
            
            # Get tree ID
            if idx < len(self.global_target_ids):
                tree_id = self.global_target_ids[idx]
            else:
                tree_id = str(idx)
            
            # For now, path is same as title (could be enhanced)
            path = title
            
            candidates.append(Candidate(
                rank=rank,
                score=float(scores[idx]),
                tree_id=tree_id,
                title=title,
                path=path,
                cluster_id=cluster_id,
                dataset_index=idx
            ))
        
        return candidates
    
    def search_batch(self, queries: List[str], top_k: int = 5) -> List[List[Candidate]]:
        """Search for multiple queries."""
        return [self.search(q, top_k) for q in queries]


def format_candidates(candidates: List[Candidate], json_output: bool = False) -> str:
    """Format candidates for output."""
    if json_output:
        return json.dumps([{
            "rank": c.rank,
            "score": c.score,
            "tree_id": c.tree_id,
            "title": c.title,
            "path": c.path,
            "cluster_id": c.cluster_id
        } for c in candidates], indent=2)
    
    lines = []
    for c in candidates:
        lines.append(f"{c.rank}. [{c.score:.4f}] {c.title}")
        lines.append(f"   ID: {c.tree_id} | Cluster: {c.cluster_id}")
    return "\n".join(lines)


def build_merged_tree(candidates: List[Candidate], data: List[dict]) -> 'TreeNode':
    """Build a merged tree from candidates with full paths."""
    
    class TreeNode:
        def __init__(self, name):
            self.name = name
            self.children = {}
            self.is_result = False
            self.score = 0.0
            self.rank = 0
    
    root = TreeNode('ROOT')
    
    for c in candidates:
        # Find the data entry by dataset_index if available
        idx = None
        if c.dataset_index >= 0 and c.dataset_index < len(data):
            idx = c.dataset_index
        else:
            # Fallback matching
            for i, d in enumerate(data):
                if d.get('tree_id') == c.tree_id or d.get('uri', '').endswith(c.tree_id):
                    idx = i
                    break
        
        if idx is None:
            # Fallback: use title as single node
            if c.title not in root.children:
                root.children[c.title] = TreeNode(c.title)
            node = root.children[c.title]
            node.is_result = True
            node.score = c.score
            node.rank = c.rank
            continue
        
        d = data[idx]
        path_text = d.get('target_text', '')
        lines = path_text.split('\n')[1:]  # Skip ID line
        
        current = root
        for line in lines:
            stripped = line.lstrip('- ')
            if not stripped:
                continue
            if stripped not in current.children:
                current.children[stripped] = TreeNode(stripped)
            current = current.children[stripped]
        
        current.is_result = True
        current.score = c.score
        current.rank = c.rank
    
    return root


def format_tree(node, depth=0, prefix='') -> str:
    """Format tree as string with box-drawing characters."""
    lines = []
    items = sorted(node.children.items())
    
    for i, (name, child) in enumerate(items):
        is_last = i == len(items) - 1
        connector = '└── ' if is_last else '├── '
        
        if child.is_result:
            result_str = f' ★ #{child.rank} [{child.score:.3f}]'
        else:
            result_str = ''
        
        lines.append(f'{prefix}{connector}{name}{result_str}')
        
        new_prefix = prefix + ('    ' if is_last else '│   ')
        lines.append(format_tree(child, depth + 1, new_prefix))
    
    return '\n'.join(filter(None, lines))


def interactive_mode(engine: 'FederatedInferenceEngine', top_k: int = 5):
    """Run in interactive mode."""
    print("\n=== Pearltrees Bookmark Filing Assistant ===")
    print(f"Model: {engine.model_path}")
    print(f"Clusters: {len(engine.cluster_ids)}")
    print(f"Targets: {len(engine.query_embeddings)}")
    print("\nEnter bookmark titles/URLs to find best folders.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Query> ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            
            candidates = engine.search(query, top_k)
            print(f"\nTop {top_k} candidates:")
            print(format_candidates(candidates))
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inference for federated Pearltrees projection model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model", type=Path, required=True,
                       help="Path to model .pkl file")
    parser.add_argument("--query", type=str, default=None,
                       help="Query to search for")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of candidates to return")
    parser.add_argument("--top-k-routing", type=int, default=10,
                       help="Number of training queries for routing")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Weight for projected score (1.0=proj, 0.0=raw)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    parser.add_argument("--tree", action="store_true",
                       help="Output results as merged hierarchical tree")
    parser.add_argument("--data", type=Path, default=None,
                       help="Path to original JSONL data (for --tree mode)")
    parser.add_argument("--embedder", type=str, default="nomic-ai/nomic-embed-text-v1.5",
                       help="Embedding model to use")
    
    args = parser.parse_args()
    
    # Load engine
    engine = FederatedInferenceEngine(args.model, args.embedder)
    
    # Load data if tree mode requested
    data = None
    if args.tree:
        data_path = args.data or Path("reports/pearltrees_targets_full_multi_account.jsonl")
        if data_path.exists():
            with open(data_path) as f:
                data = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(data)} entries for tree display")
        else:
            logger.warning(f"Data file not found: {data_path}, tree mode disabled")
            args.tree = False
    
    if args.interactive:
        interactive_mode(engine, args.top_k)
    elif args.query:
        candidates = engine.search(args.query, args.top_k, args.top_k_routing, args.alpha)
        
        if args.tree and data:
            tree = build_merged_tree(candidates, data)
            print(f"Query: {args.query}")
            print(f"\nMerged tree of top {args.top_k} results:")
            print("=" * 60)
            print(format_tree(tree))
        else:
            print(format_candidates(candidates, args.json))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
