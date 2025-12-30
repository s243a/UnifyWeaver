#!/usr/bin/env python3
"""Test dual-objective scoring with merged tree output."""
import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List


@dataclass
class Candidate:
    rank: int
    score: float
    title: str
    item_type: str
    dataset_index: int


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.is_result = False
        self.score = 0.0
        self.rank = 0


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def build_merged_tree(candidates: List[Candidate], data: dict) -> TreeNode:
    """Build a merged tree from candidates with full paths."""
    root = TreeNode('ROOT')
    
    for c in candidates:
        d = data.get(c.dataset_index, {})
        path_text = d.get('target_text', c.title)
        lines = path_text.split('\n')
        
        # Skip ID line if present
        if lines and lines[0].startswith('/'):
            lines = lines[1:]
        
        current = root
        for line in lines:
            stripped = line.lstrip('- ').strip()
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
            result_str = f' ★ #{child.rank} [{child.score:.6f}]'
        else:
            result_str = ''
        
        lines.append(f'{prefix}{connector}{name}{result_str}')
        
        new_prefix = prefix + ('    ' if is_last else '│   ')
        lines.append(format_tree(child, depth + 1, new_prefix))
    
    return '\n'.join(filter(None, lines))


def main():
    parser = argparse.ArgumentParser(description="Test dual-objective scoring")
    parser.add_argument("--embeddings", type=str, default="models/dual_embeddings_full.npz",
                       help="Path to dual embeddings NPZ file")
    parser.add_argument("--data", type=str, default="reports/pearltrees_targets_full_pearls.jsonl",
                       help="Path to JSONL data file")
    parser.add_argument("--query", type=str, default="wave",
                       help="Query string")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Blend weight (0=Input only, 1=Output only, default=0.7)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of results")
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    emb_data = np.load(args.embeddings, allow_pickle=True)
    
    input_nomic = emb_data["input_nomic"]
    input_alt = emb_data["input_alt"]
    output_nomic = emb_data["output_nomic"]
    titles = emb_data["titles"]
    item_types = emb_data["item_types"]
    
    print(f"Loaded {len(titles)} items")
    
    # Load JSONL data for path display
    print(f"Loading data from {args.data}...")
    jsonl_data = {}
    with open(args.data) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                jsonl_data[i] = json.loads(line)
    
    # Load models
    print("Loading models...")
    nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)
    
    # Embed query
    q_nomic = nomic.encode(args.query)
    q_alt = minilm.encode(args.query)
    
    # Calculate scores
    input_scores = np.array([cosine(q_alt, e) for e in input_alt])
    output_scores = np.array([cosine(q_nomic, e) for e in output_nomic])
    
    # Normalize to probabilities (ReLU + L1)
    p_input = np.maximum(input_scores, 0)
    p_input /= p_input.sum() + 1e-10
    p_output = np.maximum(output_scores, 0)
    p_output /= p_output.sum() + 1e-10
    
    # Blend
    blended = args.alpha * p_output + (1 - args.alpha) * p_input
    top_indices = np.argsort(blended)[::-1][:args.top_k]
    
    # Build candidates
    candidates = []
    for rank, idx in enumerate(top_indices, 1):
        candidates.append(Candidate(
            rank=rank,
            score=blended[idx],
            title=str(titles[idx]),
            item_type=str(item_types[idx]),
            dataset_index=int(idx)
        ))
    
    # Display results
    print(f"\nQuery: {args.query}")
    print(f"Alpha: {args.alpha} (0=Input/Semantic, 1=Output/Structural)")
    print(f"\nMerged tree of top {args.top_k} results:")
    print("=" * 60)
    
    tree = build_merged_tree(candidates, jsonl_data)
    print(format_tree(tree))


if __name__ == "__main__":
    main()
