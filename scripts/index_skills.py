#!/usr/bin/env python3
"""
Index Skill Documents for Semantic Routing

Extracts key information from skill documents and creates embeddings
for semantic retrieval. The index maps questions/topics to skill files.

Usage:
    python3 scripts/index_skills.py --output datasets/skills_qa/skill_index.npz
    python3 scripts/index_skills.py --output datasets/skills_qa/skill_index.npz --rebuild
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

project_root = Path(__file__).parent.parent
SKILLS_DIR = project_root / "skills"


def extract_skill_metadata(skill_path: Path) -> Dict[str, Any]:
    """Extract metadata from a skill document."""
    content = skill_path.read_text(encoding='utf-8')

    metadata = {
        "file": skill_path.name,
        "path": str(skill_path.relative_to(project_root)),
        "title": "",
        "description": "",
        "topics": [],
        "example_questions": [],
        "key_phrases": []
    }

    lines = content.split('\n')

    # Extract title (first H1)
    for line in lines:
        if line.startswith('# '):
            metadata["title"] = line[2:].strip()
            break

    # Extract description (first paragraph after title)
    in_description = False
    desc_lines = []
    for i, line in enumerate(lines):
        if line.startswith('# '):
            in_description = True
            continue
        if in_description:
            if line.strip() == '':
                if desc_lines:
                    break
                continue
            if line.startswith('#'):
                break
            desc_lines.append(line.strip())
    metadata["description"] = ' '.join(desc_lines)[:500]

    # Extract section headers as topics
    for line in lines:
        if line.startswith('## '):
            topic = line[3:].strip()
            if topic.lower() not in ['overview', 'introduction', 'usage', 'examples', 'see also']:
                metadata["topics"].append(topic)

    # Extract example questions (lines starting with "Q:" or in a questions section)
    in_questions = False
    for line in lines:
        if 'question' in line.lower() and line.startswith('#'):
            in_questions = True
            continue
        if in_questions and line.startswith('#'):
            in_questions = False
        if line.strip().startswith('Q:') or line.strip().startswith('- Q:'):
            q = re.sub(r'^[-\s]*Q:\s*', '', line.strip())
            metadata["example_questions"].append(q)
        elif in_questions and line.strip().startswith('- '):
            metadata["example_questions"].append(line.strip()[2:])

    # Extract key phrases (code references, predicates, commands)
    # Look for backtick-wrapped terms
    code_refs = re.findall(r'`([^`]+)`', content)
    # Filter to likely important ones (predicates, commands, paths)
    for ref in code_refs:
        if '/' in ref or ref.endswith('.pl') or ref.endswith('.py') or ref.endswith('.md'):
            if ref not in metadata["key_phrases"]:
                metadata["key_phrases"].append(ref)
        elif re.match(r'^[a-z_]+/\d+$', ref):  # Prolog predicate like foo/3
            if ref not in metadata["key_phrases"]:
                metadata["key_phrases"].append(ref)

    metadata["key_phrases"] = metadata["key_phrases"][:20]  # Limit

    return metadata


def create_routing_texts(metadata: Dict[str, Any]) -> List[str]:
    """Create text variations for embedding/routing."""
    texts = []

    # Title as a question
    if metadata["title"]:
        texts.append(f"How do I {metadata['title'].lower().replace('skill:', '').strip()}?")
        texts.append(metadata["title"])

    # Description
    if metadata["description"]:
        texts.append(metadata["description"][:200])

    # Topics as questions
    for topic in metadata["topics"][:5]:
        texts.append(f"How do I {topic.lower()}?")
        texts.append(topic)

    # Example questions
    for q in metadata["example_questions"][:5]:
        texts.append(q)

    # Key phrases
    if metadata["key_phrases"]:
        texts.append(f"Using {', '.join(metadata['key_phrases'][:5])}")

    return texts


def build_skill_index(
    skills_dir: Path,
    embedder_name: str = "all-minilm"
) -> Dict[str, Any]:
    """Build index of all skill documents."""

    skill_files = sorted(skills_dir.glob("skill_*.md"))
    print(f"Found {len(skill_files)} skill documents")

    all_metadata = []
    all_texts = []
    text_to_skill = []  # Maps text index to skill index

    for i, skill_path in enumerate(skill_files):
        print(f"  [{i+1}/{len(skill_files)}] {skill_path.name}")

        metadata = extract_skill_metadata(skill_path)
        all_metadata.append(metadata)

        texts = create_routing_texts(metadata)
        for text in texts:
            all_texts.append(text)
            text_to_skill.append(i)

    print(f"\nTotal routing texts: {len(all_texts)}")

    # Generate embeddings
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers required")
        sys.exit(1)

    model_map = {
        "all-minilm": "all-MiniLM-L6-v2",
        "e5-small": "intfloat/e5-small-v2",
    }
    model_id = model_map.get(embedder_name, embedder_name)

    print(f"\nLoading model: {model_id}")
    model = SentenceTransformer(model_id)

    print("Generating embeddings...")
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

    return {
        "metadata": all_metadata,
        "texts": all_texts,
        "text_to_skill": text_to_skill,
        "embeddings": embeddings
    }


def save_index(index: Dict[str, Any], output_path: Path):
    """Save index to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings as npz
    np.savez_compressed(
        output_path,
        embeddings=index["embeddings"],
        text_to_skill=np.array(index["text_to_skill"])
    )

    # Save metadata as JSON
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump({
            "metadata": index["metadata"],
            "texts": index["texts"]
        }, f, indent=2)

    print(f"\nSaved:")
    print(f"  Embeddings: {output_path}")
    print(f"  Metadata: {meta_path}")


def load_index(index_path: Path) -> Dict[str, Any]:
    """Load index from disk."""
    data = np.load(index_path)
    meta_path = index_path.with_suffix('.json')

    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "embeddings": data["embeddings"],
        "text_to_skill": data["text_to_skill"].tolist(),
        "metadata": meta["metadata"],
        "texts": meta["texts"]
    }


def test_search(index: Dict[str, Any], query: str, top_k: int = 3):
    """Test search on the index."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query], convert_to_numpy=True)[0]

    # Cosine similarity
    embeddings = index["embeddings"]
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_emb / np.linalg.norm(query_emb)

    similarities = np.dot(embeddings_norm, query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k * 3]

    # Deduplicate by skill
    seen_skills = set()
    results = []
    for idx in top_indices:
        skill_idx = index["text_to_skill"][idx]
        if skill_idx not in seen_skills:
            seen_skills.add(skill_idx)
            results.append({
                "skill": index["metadata"][skill_idx],
                "matched_text": index["texts"][idx],
                "score": float(similarities[idx])
            })
            if len(results) >= top_k:
                break

    return results


def main():
    parser = argparse.ArgumentParser(description="Index skill documents for semantic routing")
    parser.add_argument("--skills-dir", type=Path, default=SKILLS_DIR)
    parser.add_argument("--output", type=Path, default=project_root / "datasets" / "skills_qa" / "skill_index.npz")
    parser.add_argument("--embedder", default="all-minilm")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if exists")
    parser.add_argument("--test", type=str, help="Test search with a query")

    args = parser.parse_args()

    if args.test and args.output.exists():
        print(f"Loading index from {args.output}")
        index = load_index(args.output)
        print(f"\nSearching: {args.test}\n")
        results = test_search(index, args.test)
        for r in results:
            print(f"[{r['score']:.3f}] {r['skill']['file']}")
            print(f"  Title: {r['skill']['title']}")
            print(f"  Matched: {r['matched_text'][:60]}...")
            print()
        return

    if args.output.exists() and not args.rebuild:
        print(f"Index exists at {args.output}. Use --rebuild to regenerate.")
        return

    index = build_skill_index(args.skills_dir, args.embedder)
    save_index(index, args.output)

    print(f"\nIndex complete: {len(index['metadata'])} skills, {len(index['texts'])} routing texts")


if __name__ == "__main__":
    main()
