#!/usr/bin/env python3
"""
Create Train/Test Split for Paper Experiments

Creates stratified holdout split:
- 90% training (for W matrix learning)
- 10% test (for evaluation)

Query-level holdout: test queries removed from training, targets remain.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items, path: Path):
    with open(path, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')


def stratified_split(items, test_ratio=0.1, seed=42):
    """
    Stratified split by cluster_id.
    Returns (train, test) lists.
    """
    random.seed(seed)
    
    # Group by cluster
    clusters = defaultdict(list)
    for item in items:
        cluster_id = item.get('cluster_id', 'unknown')
        clusters[cluster_id].append(item)
    
    train = []
    test = []
    
    for cluster_id, cluster_items in clusters.items():
        random.shuffle(cluster_items)
        
        # Take test_ratio from each cluster
        n_test = max(1, int(len(cluster_items) * test_ratio))
        
        test.extend(cluster_items[:n_test])
        train.extend(cluster_items[n_test:])
    
    # Shuffle final sets
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, 
                       default=Path("datasets/pearltrees_public/pearltrees_public.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                       default=Path("datasets/pearltrees_public"))
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    items = load_jsonl(args.input)
    print(f"  Loaded {len(items)} items")
    
    print(f"\nCreating stratified split (test={args.test_ratio*100:.0f}%)...")
    train, test = stratified_split(items, args.test_ratio, args.seed)
    
    print(f"  Train: {len(train)} items")
    print(f"  Test: {len(test)} items")
    
    # Save splits
    train_path = args.output_dir / "train.jsonl"
    test_path = args.output_dir / "test.jsonl"
    
    save_jsonl(train, train_path)
    save_jsonl(test, test_path)
    
    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    
    # Stats
    train_clusters = len(set(item.get('cluster_id', '') for item in train))
    test_clusters = len(set(item.get('cluster_id', '') for item in test))
    print(f"\nCluster coverage:")
    print(f"  Train: {train_clusters} clusters")
    print(f"  Test: {test_clusters} clusters")


if __name__ == "__main__":
    main()
