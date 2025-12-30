#!/usr/bin/env python3
"""
Lightweight inference for phone deployment.
Uses ONNX runtime for embedding and numpy for scoring.
"""
import argparse
import json
import os
import sys
import tempfile
import numpy as np
from pathlib import Path


def get_temp_dir():
    """Get temp directory that works across environments (Termux, Linux, etc.)."""
    # tempfile.gettempdir() respects TMPDIR, TEMP, TMP env vars
    # Falls back to /tmp on Unix, which may not exist on Termux
    tmpdir = tempfile.gettempdir()

    # If default /tmp doesn't exist or isn't writable, try alternatives
    if not os.path.isdir(tmpdir) or not os.access(tmpdir, os.W_OK):
        # Termux: $PREFIX/tmp
        prefix_tmp = os.path.join(os.environ.get('PREFIX', ''), 'tmp')
        if os.path.isdir(prefix_tmp) and os.access(prefix_tmp, os.W_OK):
            return prefix_tmp
        # Android fallback: app cache dir
        android_tmp = '/data/local/tmp'
        if os.path.isdir(android_tmp) and os.access(android_tmp, os.W_OK):
            return android_tmp
        # Last resort: current directory
        return '.'

    return tmpdir

# Try ONNX runtime first (lighter), fall back to sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def load_embeddings(path):
    """Load pre-computed embeddings."""
    data = np.load(path, allow_pickle=True)
    return {
        "input_alt": data["input_alt"],
        "output_nomic": data["output_nomic"],
        "titles": data["titles"],
        "item_types": data["item_types"]
    }


def load_paths(jsonl_path):
    """Load paths from JSONL for display."""
    paths = {}
    if Path(jsonl_path).exists():
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    item = json.loads(line)
                    paths[i] = item.get("target_text", "")
    return paths


def build_tree(results, paths):
    """Build a nested tree structure from results."""
    tree = {}
    for r in results:
        idx = r["index"]
        path_text = paths.get(idx, "")
        lines = [l.strip() for l in path_text.split("\n") if l.strip() and not l.startswith("/")]

        current = tree
        for i, part in enumerate(lines):
            clean = part.lstrip("- ")
            if clean not in current:
                current[clean] = {"_children": {}, "_rank": None}
            if i == len(lines) - 1:
                current[clean]["_rank"] = r["rank"]
                current[clean]["_score"] = r["score"]
                current[clean]["_type"] = r["item_type"]
            current = current[clean]["_children"]
    return tree


def print_tree(node, prefix="", file=sys.stdout):
    """Print tree structure with box-drawing characters."""
    children = list(node.items())
    for i, (name, data) in enumerate(children):
        is_last = (i == len(children) - 1)
        connector = "└── " if is_last else "├── "

        rank_str = ""
        if data.get("_rank"):
            rank_str = f' ★ #{data["_rank"]} [{data["_score"]:.6f}] ({data["_type"]})'

        print(f"{prefix}{connector}{name}{rank_str}", file=file)

        next_prefix = prefix + ("    " if is_last else "│   ")
        if data.get("_children"):
            print_tree(data["_children"], next_prefix, file=file)


def embed_query(query, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Embed query using MiniLM."""
    if not HAS_ST:
        raise ImportError("sentence-transformers not installed")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model.encode(query)


def search(query_emb_minilm, query_emb_nomic, embeddings, alpha=0.7, top_k=10):
    """Search using dual-objective scoring."""
    input_alt = embeddings["input_alt"]  # 384-dim (MiniLM)
    output_nomic = embeddings["output_nomic"]  # 768-dim (Nomic)
    
    # Calculate scores using matching dimensions
    input_scores = np.array([cosine_similarity(query_emb_minilm, e) for e in input_alt])
    output_scores = np.array([cosine_similarity(query_emb_nomic, e) for e in output_nomic])
    
    # Normalize (ReLU + L1)
    p_input = np.maximum(input_scores, 0)
    p_input /= p_input.sum() + 1e-10
    p_output = np.maximum(output_scores, 0)
    p_output /= p_output.sum() + 1e-10
    
    # Blend
    blended = alpha * p_output + (1 - alpha) * p_input
    top_indices = np.argsort(blended)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "rank": rank,
            "score": float(blended[idx]),
            "title": str(embeddings["titles"][idx]),
            "item_type": str(embeddings["item_types"][idx]),
            "index": int(idx)
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phone-friendly bookmark search")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--embeddings", type=str, 
                       default="models/dual_embeddings_full.npz",
                       help="Path to embeddings file")
    parser.add_argument("--data", type=str,
                       default="reports/pearltrees_targets_full_pearls.jsonl",
                       help="Path to JSONL data")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Blend weight (0=semantic, 1=structural)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of results")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON instead of formatted results")
    parser.add_argument("--list", action="store_true",
                       help="Output flat list (last 3 path levels only)")
    parser.add_argument("--tmpdir", type=str, default=None,
                       help="Temp directory (auto-detected if not specified)")
    args = parser.parse_args()

    # Set up temp directory
    if args.tmpdir:
        tmpdir = args.tmpdir
    else:
        tmpdir = get_temp_dir()
    os.environ['TMPDIR'] = tmpdir  # Make available to subprocesses
    
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = load_embeddings(args.embeddings)
    print(f"Loaded {len(embeddings['titles'])} items")
    
    print(f"Embedding query with MiniLM: {args.query}")
    query_emb_minilm = embed_query(args.query, "sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"Embedding query with Nomic: {args.query}")
    query_emb_nomic = embed_query(args.query, "nomic-ai/nomic-embed-text-v1.5")
    
    print("Searching...")
    results = search(query_emb_minilm, query_emb_nomic, embeddings, alpha=args.alpha, top_k=args.top_k)
    
    paths = load_paths(args.data)

    if args.json:
        print(json.dumps(results, indent=2))
    elif args.list:
        # Flat list view (last 3 path levels only)
        print(f"\nResults for: {args.query}")
        print("=" * 60)
        for r in results:
            path = paths.get(r["index"], "")
            path_lines = path.split('\n')[-3:]  # Last 3 levels
            print(f"\n#{r['rank']} [{r['score']:.6f}] {r['title']} ({r['item_type']})")
            for line in path_lines:
                if line.strip():
                    print(f"    {line}")
    else:
        # Default: merged tree view
        print(f"\nResults for: {args.query}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        tree = build_tree(results, paths)
        print_tree(tree)


if __name__ == "__main__":
    main()
