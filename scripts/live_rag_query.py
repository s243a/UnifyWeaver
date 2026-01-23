#!/usr/bin/env python3
"""
Live RAG Query Tool

Hybrid approach that combines semantic routing with live document fetching:
1. Embed query → find relevant skill documents
2. Fetch current content from those skills
3. Generate answer from live source material
4. Optionally use cached Q/A as fast fallback

Usage:
    # Interactive mode
    python3 scripts/live_rag_query.py --interactive

    # Single query
    python3 scripts/live_rag_query.py "How do I compile Prolog to bash?"

    # Use cache fallback for fast responses
    python3 scripts/live_rag_query.py --cache-fallback "deployment options"

    # Just route (no generation, fast)
    python3 scripts/live_rag_query.py --route-only "authentication"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

project_root = Path(__file__).parent.parent
SKILLS_DIR = project_root / "skills"
DEFAULT_INDEX = project_root / "datasets" / "skills_qa" / "skill_index.npz"
DEFAULT_CACHE = project_root / "datasets" / "skills_qa" / "skills_qa.jsonl"


class SkillRouter:
    """Routes queries to relevant skill documents."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = None
        self.model = None

    def load(self):
        """Load index and model."""
        if self.index is not None:
            return

        from sentence_transformers import SentenceTransformer

        # Load index
        data = np.load(self.index_path)
        meta_path = self.index_path.with_suffix('.json')
        with open(meta_path) as f:
            meta = json.load(f)

        self.index = {
            "embeddings": data["embeddings"],
            "text_to_skill": data["text_to_skill"].tolist(),
            "metadata": meta["metadata"],
            "texts": meta["texts"]
        }

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def route(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant skills for a query."""
        self.load()

        query_emb = self.model.encode([query], convert_to_numpy=True)[0]

        # Cosine similarity
        embeddings = self.index["embeddings"]
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = query_emb / np.linalg.norm(query_emb)

        similarities = np.dot(embeddings_norm, query_norm)
        top_indices = np.argsort(similarities)[::-1]

        # Deduplicate by skill
        seen_skills = set()
        results = []
        for idx in top_indices:
            skill_idx = self.index["text_to_skill"][idx]
            if skill_idx not in seen_skills:
                seen_skills.add(skill_idx)
                results.append({
                    "skill": self.index["metadata"][skill_idx],
                    "matched_text": self.index["texts"][idx],
                    "score": float(similarities[idx])
                })
                if len(results) >= top_k:
                    break

        return results


class QACache:
    """Fast lookup cache for pre-computed Q/A pairs."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.pairs = None
        self.embeddings = None
        self.model = None

    def load(self):
        """Load cache and embeddings."""
        if self.pairs is not None:
            return

        from sentence_transformers import SentenceTransformer

        # Load Q/A pairs
        self.pairs = []
        with open(self.cache_path) as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line))

        # Load embeddings if available
        emb_path = self.cache_path.parent / "skills_embeddings_all-minilm.npz"
        if emb_path.exists():
            data = np.load(emb_path)
            self.embeddings = data["q_embeddings"]

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query: str, top_k: int = 1, threshold: float = 0.8) -> Optional[Dict]:
        """Search cache for matching Q/A pair."""
        self.load()

        if self.embeddings is None:
            return None

        query_emb = self.model.encode([query], convert_to_numpy=True)[0]

        # Cosine similarity
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        query_norm = query_emb / np.linalg.norm(query_emb)

        similarities = np.dot(embeddings_norm, query_norm)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= threshold:
            return {
                "pair": self.pairs[best_idx],
                "score": float(best_score),
                "source": "cache"
            }

        return None


def fetch_skill_content(skill_path: str, max_length: int = 6000) -> str:
    """Fetch current content from a skill document."""
    full_path = project_root / skill_path
    if full_path.exists():
        content = full_path.read_text(encoding='utf-8')
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[... truncated ...]"
        return content
    return ""


def generate_answer(
    query: str,
    skill_contents: List[Tuple[str, str]],  # [(skill_name, content), ...]
    provider: str = "claude",
    model: str = "sonnet"
) -> str:
    """Generate answer from live skill content."""

    context = "\n\n---\n\n".join([
        f"## {name}\n\n{content}"
        for name, content in skill_contents
    ])

    prompt = f'''You are a helpful assistant for UnifyWeaver, a Prolog-to-target-language compiler.

Based on the following skill documentation, answer the user's question.
Be concise and practical. Include code examples when relevant.
If the documentation doesn't cover the question, say so.

SKILL DOCUMENTATION:
{context}

USER QUESTION: {query}

Answer:'''

    try:
        if provider == "claude":
            result = subprocess.run(
                ["claude", "-p", "--model", model, prompt],
                capture_output=True, text=True, timeout=120
            )
        else:
            result = subprocess.run(
                ["gemini", "-p", prompt, "-m", model, "--output-format", "text"],
                capture_output=True, text=True, timeout=120
            )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error generating answer: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return "Error: Generation timed out"
    except Exception as e:
        return f"Error: {str(e)}"


def live_rag_query(
    query: str,
    router: SkillRouter,
    cache: Optional[QACache] = None,
    provider: str = "claude",
    model: str = "sonnet",
    cache_threshold: float = 0.9,
    top_k_skills: int = 2,
    route_only: bool = False
) -> Dict[str, Any]:
    """
    Full live RAG pipeline.

    1. Check cache for high-confidence match (fast path)
    2. Route to relevant skills
    3. Fetch current skill content
    4. Generate answer from live content
    """

    result = {
        "query": query,
        "source": None,
        "skills_used": [],
        "answer": None,
        "cache_score": None
    }

    # Fast path: check cache
    if cache is not None:
        cache_result = cache.search(query, threshold=cache_threshold)
        if cache_result:
            result["source"] = "cache"
            result["answer"] = cache_result["pair"]["answer"]
            result["cache_score"] = cache_result["score"]
            result["cached_question"] = cache_result["pair"]["question"]
            return result

    # Route to skills
    routes = router.route(query, top_k=top_k_skills)
    result["skills_used"] = [
        {"file": r["skill"]["file"], "score": r["score"], "matched": r["matched_text"]}
        for r in routes
    ]

    if route_only:
        result["source"] = "routing_only"
        return result

    # Fetch live content
    skill_contents = []
    for route in routes:
        content = fetch_skill_content(route["skill"]["path"])
        if content:
            skill_contents.append((route["skill"]["title"] or route["skill"]["file"], content))

    if not skill_contents:
        result["source"] = "no_content"
        result["answer"] = "Could not find relevant skill documentation."
        return result

    # Generate answer
    answer = generate_answer(query, skill_contents, provider, model)
    result["source"] = "live_generation"
    result["answer"] = answer

    return result


def interactive_mode(
    router: SkillRouter,
    cache: Optional[QACache],
    provider: str,
    model: str,
    cache_threshold: float
):
    """Interactive query loop."""
    print("Live RAG Query - Interactive Mode")
    print("=" * 50)
    print(f"Provider: {provider}/{model}")
    print(f"Cache threshold: {cache_threshold}")
    print("Type 'quit' to exit, 'route <query>' for routing only\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        route_only = False
        if query.lower().startswith('route '):
            query = query[6:].strip()
            route_only = True

        result = live_rag_query(
            query, router, cache,
            provider=provider, model=model,
            cache_threshold=cache_threshold,
            route_only=route_only
        )

        print(f"\n{'=' * 50}")
        print(f"Source: {result['source']}")

        if result.get('cache_score'):
            print(f"Cache score: {result['cache_score']:.3f}")
            print(f"Cached question: {result.get('cached_question', 'N/A')}")

        if result['skills_used']:
            print(f"\nSkills matched:")
            for skill in result['skills_used']:
                print(f"  [{skill['score']:.3f}] {skill['file']}")
                print(f"         → {skill['matched'][:50]}...")

        if result['answer']:
            print(f"\nAnswer:\n{result['answer']}")

        print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="Live RAG query tool")
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true", help="Disable cache fallback")
    parser.add_argument("--cache-threshold", type=float, default=0.9)
    parser.add_argument("--provider", default="claude", choices=["claude", "gemini"])
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--route-only", action="store_true", help="Only show routing, no generation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Check index exists
    if not args.index.exists():
        print(f"Index not found at {args.index}")
        print("Run: python3 scripts/index_skills.py --output datasets/skills_qa/skill_index.npz")
        sys.exit(1)

    # Initialize
    print("Loading...", file=sys.stderr)
    router = SkillRouter(args.index)

    cache = None
    if not args.no_cache and args.cache.exists():
        cache = QACache(args.cache)

    if args.interactive or not args.query:
        interactive_mode(router, cache, args.provider, args.model, args.cache_threshold)
    else:
        result = live_rag_query(
            args.query, router, cache,
            provider=args.provider,
            model=args.model,
            cache_threshold=args.cache_threshold,
            route_only=args.route_only
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Source: {result['source']}")
            if result['skills_used']:
                print(f"\nSkills: {', '.join(s['file'] for s in result['skills_used'])}")
            if result['answer']:
                print(f"\n{result['answer']}")


if __name__ == "__main__":
    main()
