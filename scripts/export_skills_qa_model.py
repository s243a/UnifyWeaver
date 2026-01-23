#!/usr/bin/env python3
"""
Export Skills Training Data as Q/A Retrieval Model

Converts skills-generated.jsonl files to the standard format used by the
training data loader and optionally generates embeddings for Q/A retrieval.

Usage:
    python3 scripts/export_skills_qa_model.py --output datasets/skills_qa
    python3 scripts/export_skills_qa_model.py --output datasets/skills_qa --embed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))


def load_skills_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load skills training data from JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping invalid JSON in {path}: {e}")
    return items


def convert_skills_to_qa(skills_item: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
    """
    Convert a skills training data item to Q/A format.

    Each skill item may have multiple question variants, so we create
    multiple Q/A pairs that all reference the same cluster.
    """
    qa_pairs = []

    # Map skill id to cluster_id
    cluster_id = skills_item.get('id', 'unknown')
    answer = skills_item.get('answer', '')

    # Map level to question_type
    level = skills_item.get('level', 2)
    question_type = {1: 'short', 2: 'medium', 3: 'long'}.get(level, 'medium')

    # Use tree_path as topics
    topics = skills_item.get('tree_path', [])

    # Canonical question
    canonical_question = skills_item.get('question', '')
    if canonical_question:
        qa_pairs.append({
            'pair_id': f"{cluster_id}_q0",
            'cluster_id': cluster_id,
            'question': canonical_question,
            'answer': answer,
            'question_type': question_type,
            'topics': topics,
            'source_file': source_file,
            'tags': skills_item.get('tags', []),
            'related_skills': skills_item.get('related_skills', []),
            'related_docs': skills_item.get('related_docs', [])
        })

    # Question variants
    variants = skills_item.get('question_variants', [])
    for i, variant in enumerate(variants, start=1):
        if variant:
            qa_pairs.append({
                'pair_id': f"{cluster_id}_q{i}",
                'cluster_id': cluster_id,
                'question': variant,
                'answer': answer,
                'question_type': question_type,
                'topics': topics,
                'source_file': source_file,
                'tags': skills_item.get('tags', []),
                'related_skills': skills_item.get('related_skills', []),
                'related_docs': skills_item.get('related_docs', [])
            })

    return qa_pairs


def find_skills_files(training_data_dir: Path) -> List[Path]:
    """Find all skills-generated.jsonl files."""
    skills_files = []

    # Look in by-topic directory
    by_topic = training_data_dir / "by-topic"
    if by_topic.exists():
        skills_files.extend(by_topic.rglob("skills-generated.jsonl"))

    return sorted(skills_files)


def export_qa_pairs(output_dir: Path, qa_pairs: List[Dict[str, Any]]):
    """Export Q/A pairs to JSONL format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main dataset file
    output_file = output_dir / "skills_qa.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"  Exported {len(qa_pairs)} Q/A pairs to {output_file}")

    # Also export unique answers (for retrieval targets)
    answers_file = output_dir / "skills_answers.jsonl"
    seen_clusters = set()
    unique_answers = []
    for pair in qa_pairs:
        cluster_id = pair['cluster_id']
        if cluster_id not in seen_clusters:
            seen_clusters.add(cluster_id)
            unique_answers.append({
                'cluster_id': cluster_id,
                'answer': pair['answer'],
                'topics': pair['topics'],
                'related_skills': pair.get('related_skills', []),
                'related_docs': pair.get('related_docs', []),
                'tags': pair.get('tags', [])
            })

    with open(answers_file, 'w', encoding='utf-8') as f:
        for answer in unique_answers:
            f.write(json.dumps(answer) + '\n')

    print(f"  Exported {len(unique_answers)} unique answers to {answers_file}")

    return output_file


def generate_embeddings(
    qa_pairs: List[Dict[str, Any]],
    output_dir: Path,
    embedder_name: str = "all-minilm"
):
    """Generate embeddings for Q/A pairs."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("Error: sentence-transformers required. Install with: pip install sentence-transformers")
        return None

    # Model mapping
    model_map = {
        "all-minilm": "all-MiniLM-L6-v2",
        "e5-small": "intfloat/e5-small-v2",
        "modernbert": "nomic-ai/nomic-embed-text-v1.5",
    }

    model_id = model_map.get(embedder_name, embedder_name)
    print(f"\nLoading embedding model: {model_id}")
    model = SentenceTransformer(model_id, trust_remote_code=True)

    # Extract questions and answers
    questions = [p['question'] for p in qa_pairs]
    answers = [p['answer'] for p in qa_pairs]
    pair_ids = [p['pair_id'] for p in qa_pairs]
    cluster_ids = [p['cluster_id'] for p in qa_pairs]

    # Generate embeddings
    print(f"Embedding {len(questions)} questions...")
    q_embeddings = model.encode(questions, show_progress_bar=True, convert_to_numpy=True)

    print(f"Embedding {len(answers)} answers...")
    a_embeddings = model.encode(answers, show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings
    embeddings_file = output_dir / f"skills_embeddings_{embedder_name}.npz"
    np.savez_compressed(
        embeddings_file,
        q_embeddings=q_embeddings,
        a_embeddings=a_embeddings,
        pair_ids=np.array(pair_ids),
        cluster_ids=np.array(cluster_ids)
    )

    print(f"  Saved embeddings to {embeddings_file}")
    print(f"  Shape: questions={q_embeddings.shape}, answers={a_embeddings.shape}")

    return embeddings_file


def print_stats(qa_pairs: List[Dict[str, Any]]):
    """Print statistics about the Q/A dataset."""
    print("\n=== Dataset Statistics ===")
    print(f"Total Q/A pairs: {len(qa_pairs)}")

    # Unique clusters (answers)
    clusters = set(p['cluster_id'] for p in qa_pairs)
    print(f"Unique answers (clusters): {len(clusters)}")

    # Question type distribution
    type_counts = defaultdict(int)
    for p in qa_pairs:
        type_counts[p['question_type']] += 1
    print(f"Question types: {dict(type_counts)}")

    # Topic distribution (top 10)
    topic_counts = defaultdict(int)
    for p in qa_pairs:
        for topic in p.get('topics', []):
            topic_counts[topic] += 1
    top_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:10]
    print(f"Top topics: {top_topics}")

    # Source files
    sources = defaultdict(int)
    for p in qa_pairs:
        sources[p.get('source_file', 'unknown')] += 1
    print(f"Source files: {len(sources)}")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count} pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Export skills training data as Q/A retrieval model"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "training-data",
        help="Training data root directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "datasets" / "skills_qa",
        help="Output directory"
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate embeddings (requires sentence-transformers)"
    )
    parser.add_argument(
        "--embedder",
        default="all-minilm",
        choices=["all-minilm", "e5-small", "modernbert"],
        help="Embedding model to use"
    )

    args = parser.parse_args()

    print(f"Looking for skills training data in {args.input}...")
    skills_files = find_skills_files(args.input)

    if not skills_files:
        print("No skills-generated.jsonl files found!")
        return 1

    print(f"Found {len(skills_files)} skills files:")
    for f in skills_files:
        print(f"  {f.relative_to(args.input)}")

    # Convert all files
    all_qa_pairs = []
    for skills_file in skills_files:
        print(f"\nProcessing {skills_file.name}...")
        items = load_skills_jsonl(skills_file)
        print(f"  Loaded {len(items)} skill entries")

        source_file = str(skills_file.relative_to(args.input))
        for item in items:
            qa_pairs = convert_skills_to_qa(item, source_file)
            all_qa_pairs.extend(qa_pairs)

    print(f"\nTotal Q/A pairs: {len(all_qa_pairs)}")

    # Print stats
    print_stats(all_qa_pairs)

    # Export
    print(f"\nExporting to {args.output}...")
    output_file = export_qa_pairs(args.output, all_qa_pairs)

    # Generate embeddings if requested
    if args.embed:
        generate_embeddings(all_qa_pairs, args.output, args.embedder)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
