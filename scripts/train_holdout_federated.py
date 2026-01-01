#!/usr/bin/env python3
"""
Train Federated Model with Query-Level Holdout

This script trains on train queries only but keeps ALL targets searchable:
- Target embeddings: from FULL dataset (so test items can be found)
- W matrices: trained only on TRAIN queries
- Routing: based only on TRAIN query embeddings

Usage:
    python scripts/train_holdout_federated.py \
        --train datasets/pearltrees_public/train.jsonl \
        --full datasets/pearltrees_public/pearltrees_public.jsonl \
        --output models/holdout_federated_minilm.pkl \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --max-clusters 700
"""

import argparse
import hashlib
import json
import logging
import numpy as np
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))
sys.path.insert(0, str(Path(__file__).parent))

from minimal_transform import compute_minimal_transform
from train_pearltrees_federated import SimpleEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sanitize_cluster_name(name: str) -> str:
    """Convert cluster name to a valid filename."""
    # Use hash for URL-style cluster IDs
    if "/" in name or ":" in name:
        return f"cluster_{hashlib.md5(name.encode()).hexdigest()[:12]}"
    return name.replace(" ", "_").replace("/", "_")


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def cluster_by_path(data: List[Dict], cluster_field: str = "cluster_id") -> Dict[str, List[int]]:
    """Cluster by cluster_id field."""
    clusters = defaultdict(list)
    for i, d in enumerate(data):
        cid = d.get(cluster_field, "unknown")
        clusters[cid].append(i)
    return dict(clusters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="Training queries JSONL")
    parser.add_argument("--full", type=Path, required=True, help="Full dataset JSONL (for targets)")
    parser.add_argument("--output", type=Path, required=True, help="Output model .pkl")
    parser.add_argument("--model", type=str, default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--max-clusters", type=int, default=200)
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading train data: {args.train}")
    train_data = load_jsonl(args.train)
    logger.info(f"  Train items: {len(train_data)}")
    
    logger.info(f"Loading full data: {args.full}")
    full_data = load_jsonl(args.full)
    logger.info(f"  Full items: {len(full_data)}")
    
    # Build train tree_ids for filtering
    train_tree_ids = set(str(d.get("tree_id", "")) for d in train_data)
    logger.info(f"  Train tree_ids: {len(train_tree_ids)}")
    
    # Initialize embedder
    logger.info(f"Loading embedder: {args.model}")
    embedder = SimpleEmbedder(args.model)
    
    # Cluster training data
    logger.info("Clustering training data...")
    clusters = cluster_by_path(train_data)
    logger.info(f"  Found {len(clusters)} clusters")
    
    # Limit clusters if needed
    if len(clusters) > args.max_clusters:
        # Keep top clusters by size
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        clusters = dict(sorted_clusters[:args.max_clusters])
        logger.info(f"  Limited to {len(clusters)} clusters")
    
    # Embed ALL targets (full dataset)
    logger.info("Embedding all targets...")
    target_texts = [d.get("target_text", d.get("raw_title", "")) for d in full_data]
    target_embeddings = embedder.encode(target_texts)
    logger.info(f"  Target embeddings: {target_embeddings.shape}")
    
    # Build target lookup
    tree_id_to_idx = {str(d.get("tree_id", "")): i for i, d in enumerate(full_data)}
    
    # Output directory
    output_dir = args.output.parent / args.output.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train W matrices per cluster (using train data only)
    cluster_models = {}
    all_train_query_embeddings = []
    all_train_cluster_labels = []
    
    for cluster_name, indices in clusters.items():
        logger.info(f"Training cluster '{cluster_name}' with {len(indices)} items...")
        
        # Get train items for this cluster
        cluster_train = [train_data[i] for i in indices]
        
        # Prepare Q/A pairs
        queries = []
        target_idxs = []
        
        for item in cluster_train:
            q = item.get("raw_title", "")
            tree_id = str(item.get("tree_id", ""))
            
            if tree_id in tree_id_to_idx:
                queries.append(q)
                target_idxs.append(tree_id_to_idx[tree_id])
        
        if len(queries) < 1:
            continue
        
        # Embed queries
        query_embeddings = embedder.encode(queries)
        answer_embeddings = target_embeddings[target_idxs]
        
        # Store for routing
        for qe in query_embeddings:
            all_train_query_embeddings.append(qe)
            all_train_cluster_labels.append(cluster_name)
        
        # Compute W matrix via Procrustes
        # Use centroid of queries -> centroid of answers
        Q_centroid = query_embeddings.mean(axis=0, keepdims=True)
        A_centroid = answer_embeddings.mean(axis=0, keepdims=True)
        
        W, scale, info = compute_minimal_transform(Q_centroid, A_centroid)
        
        # Save cluster with sanitized name
        safe_name = sanitize_cluster_name(cluster_name)
        np.savez(
            output_dir / f"{safe_name}.npz",
            W=W,
            target_indices=target_idxs
        )
        cluster_models[cluster_name] = {
            "W_file": f"{safe_name}.npz",
            "n_items": len(queries)
        }
    
    # Save routing data (train queries only)
    logger.info("Saving routing data...")
    routing_embeddings = np.array(all_train_query_embeddings)
    np.savez(
        output_dir / "routing_data.npz",
        query_embeddings=routing_embeddings,
        cluster_labels=all_train_cluster_labels
    )
    
    # Save model metadata
    model_meta = {
        "clusters": cluster_models,
        "n_clusters": len(cluster_models),
        "target_tree_ids": [str(d.get("tree_id", "")) for d in full_data],
        "target_titles": [d.get("raw_title", "") for d in full_data],
        "n_targets": len(full_data),
        "n_train": len(train_data),
        "model": args.model
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(model_meta, f)
    
    # Also save target embeddings for inference
    np.savez(
        output_dir / "target_embeddings.npz",
        embeddings=target_embeddings
    )
    
    logger.info(f"Model saved to {args.output}")
    logger.info(f"Cluster files saved to {output_dir}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()
