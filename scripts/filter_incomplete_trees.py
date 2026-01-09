#!/usr/bin/env python3
"""
Filter incomplete mindmaps using semantic embeddings.

Supports two filtering methods:
1. Lexical: Embed a query string directly (finds similar titles/paths)
2. Structural: Use an existing tree's output embedding (finds similar hierarchy positions)

Usage:
    # Lexical filtering - find trees with "science" in path/title
    python3 scripts/filter_incomplete_trees.py \
        --query "science" \
        --top-k 10 --format tree

    # Structural filtering - find trees near the "science" tree in hierarchy
    python3 scripts/filter_incomplete_trees.py \
        --tree-query "science" \
        --top-k 10 --format tree

    # Both methods combined (blend)
    python3 scripts/filter_incomplete_trees.py \
        --query "science" \
        --tree-query "science" \
        --alpha 0.5 \
        --top-k 10 --format tree
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FilteredTree:
    tree_id: str
    title: str
    uri: str
    path_text: str
    similarity: float
    lexical_score: float = 0.0
    structural_score: float = 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_incomplete_trees(scan_path: Path, jsonl_path: Path) -> List[Dict]:
    """Load incomplete trees with their hierarchical paths."""
    # Load path lookup
    path_lookup = {}
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line.strip())
            tree_id = record.get('tree_id', '')
            if tree_id:
                path_lookup[str(tree_id)] = record.get('target_text', '')

    # Load incomplete trees
    with open(scan_path) as f:
        incomplete = json.load(f)['maps']

    # Merge path data
    result = []
    for m in incomplete:
        tid = m['tree_id']
        path_text = path_lookup.get(tid, '')
        if path_text:
            lines = path_text.split('\n')
            path_lines = [l for l in lines if not l.startswith('/')]
            hierarchical_text = '\n'.join(path_lines)
        else:
            hierarchical_text = m['title']

        result.append({
            'tree_id': tid,
            'title': m['title'],
            'uri': m['uri'],
            'path_text': hierarchical_text
        })

    return result


def embed_trees(trees: List[Dict], cache_path: Optional[Path] = None) -> np.ndarray:
    """Embed tree paths, with optional caching."""
    if cache_path and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['embeddings']

    from sentence_transformers import SentenceTransformer
    print("Loading Nomic model...")
    nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    texts = [t['path_text'] for t in trees]
    print(f"Embedding {len(texts)} paths...")

    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = nomic.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embs)
        if (i // batch_size) % 10 == 0:
            print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")

    embeddings = np.array(all_embeddings)

    if cache_path:
        print(f"Caching embeddings to {cache_path}")
        np.savez_compressed(cache_path,
            embeddings=embeddings,
            tree_ids=np.array([t['tree_id'] for t in trees]),
            uris=np.array([t['uri'] for t in trees]),
            titles=np.array([t['title'] for t in trees])
        )

    return embeddings


def find_tree_embedding(tree_query: str, emb_path: Path) -> Optional[np.ndarray]:
    """Find output embedding for a tree by title."""
    if not emb_path.exists():
        print(f"Warning: {emb_path} not found")
        return None

    data = np.load(emb_path, allow_pickle=True)
    titles = data['titles']
    output_nomic = data['output_nomic']

    # Find matching tree
    query_lower = tree_query.lower()
    for i, title in enumerate(titles):
        if query_lower in str(title).lower():
            print(f"Found tree: '{title}' (idx={i})")
            return output_nomic[i]

    print(f"Warning: No tree matching '{tree_query}' found")
    return None


def filter_trees(
    trees: List[Dict],
    embeddings: np.ndarray,
    query: Optional[str] = None,
    tree_query_emb: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    top_k: int = 10
) -> List[FilteredTree]:
    """Filter trees by semantic similarity.

    Args:
        trees: List of tree dicts
        embeddings: Tree path embeddings
        query: Lexical query string
        tree_query_emb: Structural query (existing tree's output embedding)
        alpha: Blend weight (0=lexical only, 1=structural only)
        top_k: Number of results

    Returns:
        List of FilteredTree results
    """
    from sentence_transformers import SentenceTransformer

    lexical_scores = np.zeros(len(trees))
    structural_scores = np.zeros(len(trees))

    # Lexical scoring
    if query:
        nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        query_emb = nomic.encode([query])[0]
        for i in range(len(embeddings)):
            lexical_scores[i] = cosine_sim(query_emb, embeddings[i])

    # Structural scoring
    if tree_query_emb is not None:
        for i in range(len(embeddings)):
            structural_scores[i] = cosine_sim(tree_query_emb, embeddings[i])

    # Blend scores
    if query and tree_query_emb is not None:
        final_scores = (1 - alpha) * lexical_scores + alpha * structural_scores
    elif query:
        final_scores = lexical_scores
    elif tree_query_emb is not None:
        final_scores = structural_scores
    else:
        raise ValueError("Must provide either query or tree_query_emb")

    # Get top-k
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(FilteredTree(
            tree_id=trees[idx]['tree_id'],
            title=trees[idx]['title'],
            uri=trees[idx]['uri'],
            path_text=trees[idx]['path_text'],
            similarity=final_scores[idx],
            lexical_score=lexical_scores[idx],
            structural_score=structural_scores[idx]
        ))

    return results


def build_merged_tree(results: List[FilteredTree]) -> str:
    """Build merged tree view from filtered results."""

    class TreeNode:
        def __init__(self, name):
            self.name = name
            self.children = {}
            self.results = []
            self.count = 0

    def parse_path(path_text):
        lines = path_text.split('\n')
        path = []
        for line in lines:
            stripped = line.lstrip('- ').strip()
            if stripped:
                path.append(stripped)
        return path

    root = TreeNode('ROOT')

    for r in results:
        path = parse_path(r.path_text) if r.path_text else [r.title]
        current = root
        for part in path:
            if part not in current.children:
                current.children[part] = TreeNode(part)
            current = current.children[part]
        current.results.append(r)

    def compute_counts(node):
        node.count = len(node.results)
        for child in node.children.values():
            node.count += compute_counts(child)
        return node.count

    compute_counts(root)

    def format_tree(node, prefix=''):
        lines = []
        items = sorted(node.children.items(), key=lambda x: (-x[1].count, x[0]))

        for i, (name, child) in enumerate(items):
            is_last = i == len(items) - 1
            connector = '└── ' if is_last else '├── '
            count_str = f' ({child.count})' if child.count > 0 else ''
            lines.append(f'{prefix}{connector}{name}{count_str}')

            new_prefix = prefix + ('    ' if is_last else '│   ')

            for r in child.results:
                score_info = f'[{r.similarity:.3f}]'
                if r.lexical_score and r.structural_score:
                    score_info = f'[L:{r.lexical_score:.3f} S:{r.structural_score:.3f} = {r.similarity:.3f}]'
                lines.append(f'{new_prefix}★ {score_info} {r.uri}')

            child_output = format_tree(child, new_prefix)
            if child_output:
                lines.append(child_output)

        return '\n'.join(filter(None, lines))

    return format_tree(root)


def main():
    parser = argparse.ArgumentParser(
        description='Filter incomplete trees using semantic embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--scan', type=Path,
                       default=Path('.local/data/scans/incomplete_mindmaps.json'),
                       help='Path to incomplete mindmaps JSON')
    parser.add_argument('--data', type=Path,
                       default=Path('reports/pearltrees_targets_full_multi_account.jsonl'),
                       help='Path to JSONL with paths')
    parser.add_argument('--embeddings-cache', type=Path,
                       default=Path('.local/data/scans/incomplete_embeddings.npz'),
                       help='Cache path for embeddings')
    parser.add_argument('--output-embeddings', type=Path,
                       default=Path('models/dual_embeddings_full.npz'),
                       help='Path to pre-built output embeddings (for structural query)')

    parser.add_argument('--query', type=str, default=None,
                       help='Lexical query string')
    parser.add_argument('--tree-query', type=str, default=None,
                       help='Structural query - find tree by title and use its output embedding')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Blend weight: 0=lexical, 1=structural (default: 0.5)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results')

    parser.add_argument('--format', choices=['tree', 'list', 'urls', 'json'],
                       default='tree', help='Output format')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output file')

    args = parser.parse_args()

    if not args.query and not args.tree_query:
        parser.error("Must provide --query and/or --tree-query")

    # Load data
    print("Loading incomplete trees...")
    trees = load_incomplete_trees(args.scan, args.data)
    print(f"Loaded {len(trees)} incomplete trees")

    # Get/create embeddings
    embeddings = embed_trees(trees, args.embeddings_cache)

    # Get structural query embedding if requested
    tree_query_emb = None
    if args.tree_query:
        tree_query_emb = find_tree_embedding(args.tree_query, args.output_embeddings)

    # Filter
    print(f"\nFiltering with:")
    if args.query:
        print(f"  Lexical query: '{args.query}'")
    if args.tree_query:
        print(f"  Structural query: '{args.tree_query}'")
    if args.query and tree_query_emb is not None:
        print(f"  Alpha: {args.alpha} (0=lexical, 1=structural)")

    results = filter_trees(
        trees, embeddings,
        query=args.query,
        tree_query_emb=tree_query_emb,
        alpha=args.alpha,
        top_k=args.top_k
    )

    # Format output
    if args.format == 'tree':
        output = f"=== Top {args.top_k} results ===\n\n"
        output += build_merged_tree(results)
    elif args.format == 'list':
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r.similarity:.4f}] {r.title}")
            lines.append(f"   {r.uri}")
        output = '\n'.join(lines)
    elif args.format == 'urls':
        output = '\n'.join(r.uri for r in results)
    elif args.format == 'json':
        output = json.dumps([{
            'tree_id': r.tree_id,
            'title': r.title,
            'uri': r.uri,
            'similarity': r.similarity,
            'lexical_score': r.lexical_score,
            'structural_score': r.structural_score
        } for r in results], indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nWritten to {args.output}")
    else:
        print(f"\n{output}")


if __name__ == '__main__':
    main()
