#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Knowledge Graph Tool for Q/A Playbook Relations.

Computes content hashes for answers, manages relations, and identifies gaps.

Usage:
    # List all answers with hashes
    python scripts/knowledge_graph_tool.py list playbooks/lda-training-data/raw/

    # Show gaps (answers missing relations)
    python scripts/knowledge_graph_tool.py gaps playbooks/lda-training-data/raw/

    # Add a relation
    python scripts/knowledge_graph_tool.py add-relation \
        --from abc123 --to def456 --type foundational

    # Show relations for an answer
    python scripts/knowledge_graph_tool.py show abc123
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


RELATION_TYPES = ['foundational', 'preliminary', 'compositional', 'transitional']


def compute_answer_hash(answer_text: str, length: int = 12) -> str:
    """Compute truncated SHA256 hash of answer text."""
    full_hash = hashlib.sha256(answer_text.encode('utf-8')).hexdigest()
    return full_hash[:length]


@dataclass
class Answer:
    """Represents an answer/example with its metadata."""
    cluster_id: str
    answer_hash: str
    answer_text: str
    source_file: str
    answer_source: str
    relations: Dict[str, List[str]] = field(default_factory=lambda: {
        rt: [] for rt in RELATION_TYPES
    })


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph for Q/A relations."""
    answers: Dict[str, Answer] = field(default_factory=dict)  # hash -> Answer
    id_to_hash: Dict[str, str] = field(default_factory=dict)  # cluster_id -> hash

    def add_answer(self, answer: Answer):
        self.answers[answer.answer_hash] = answer
        self.id_to_hash[answer.cluster_id] = answer.answer_hash

    def get_by_hash(self, hash_prefix: str) -> Optional[Answer]:
        """Get answer by hash or hash prefix."""
        # Exact match
        if hash_prefix in self.answers:
            return self.answers[hash_prefix]
        # Prefix match
        matches = [h for h in self.answers if h.startswith(hash_prefix)]
        if len(matches) == 1:
            return self.answers[matches[0]]
        elif len(matches) > 1:
            print(f"Ambiguous hash prefix '{hash_prefix}', matches: {matches}")
        return None

    def get_by_id(self, cluster_id: str) -> Optional[Answer]:
        """Get answer by cluster ID."""
        if cluster_id in self.id_to_hash:
            return self.answers[self.id_to_hash[cluster_id]]
        return None

    def find_gaps(self) -> Dict[str, List[str]]:
        """Find answers missing relations by category."""
        gaps = {rt: [] for rt in RELATION_TYPES}
        for answer in self.answers.values():
            for rt in RELATION_TYPES:
                if not answer.relations.get(rt):
                    gaps[rt].append(answer.cluster_id)
        return gaps


def load_qa_files(data_dir: Path) -> KnowledgeGraph:
    """Load all Q/A JSON files from directory."""
    graph = KnowledgeGraph()

    for json_file in sorted(data_dir.glob("qa_pairs*.json")):
        with open(json_file) as f:
            data = json.load(f)

        for cluster in data.get('clusters', []):
            cluster_id = cluster.get('id', '')
            answers = cluster.get('answers', {})
            answer_text = answers.get('default', '')

            if not answer_text:
                continue

            answer_hash = compute_answer_hash(answer_text)

            # Load existing relations if present
            relations = cluster.get('relations', {})
            normalized_relations = {rt: relations.get(rt, []) for rt in RELATION_TYPES}

            answer = Answer(
                cluster_id=cluster_id,
                answer_hash=answer_hash,
                answer_text=answer_text[:100] + '...' if len(answer_text) > 100 else answer_text,
                source_file=str(json_file.name),
                answer_source=cluster.get('answer_source', ''),
                relations=normalized_relations
            )
            graph.add_answer(answer)

    return graph


def load_relations_file(relations_path: Path, graph: KnowledgeGraph):
    """Load relations from a separate relations JSON file."""
    if not relations_path.exists():
        return

    with open(relations_path) as f:
        data = json.load(f)

    for rel in data.get('relations', []):
        from_ref = rel.get('from', '')
        to_ref = rel.get('to', '')
        rel_type = rel.get('type', '')

        if rel_type not in RELATION_TYPES:
            print(f"Warning: Unknown relation type '{rel_type}'")
            continue

        # Resolve references (can be hash or cluster_id)
        from_answer = graph.get_by_hash(from_ref) or graph.get_by_id(from_ref)
        to_answer = graph.get_by_hash(to_ref) or graph.get_by_id(to_ref)

        if from_answer and to_answer:
            if to_answer.answer_hash not in from_answer.relations[rel_type]:
                from_answer.relations[rel_type].append(to_answer.answer_hash)


def cmd_list(args):
    """List all answers with their hashes."""
    graph = load_qa_files(Path(args.data_dir))

    if args.relations_file:
        load_relations_file(Path(args.relations_file), graph)

    print(f"{'Hash':<14} {'Cluster ID':<35} {'Source':<20}")
    print("-" * 70)

    for answer in sorted(graph.answers.values(), key=lambda a: a.cluster_id):
        print(f"{answer.answer_hash:<14} {answer.cluster_id:<35} {answer.source_file:<20}")

    print(f"\nTotal: {len(graph.answers)} answers")


def cmd_gaps(args):
    """Show answers missing relations by category."""
    graph = load_qa_files(Path(args.data_dir))

    if args.relations_file:
        load_relations_file(Path(args.relations_file), graph)

    gaps = graph.find_gaps()

    print("=" * 60)
    print("Knowledge Graph Gap Analysis")
    print("=" * 60)

    total_answers = len(graph.answers)

    for rel_type in RELATION_TYPES:
        missing = gaps[rel_type]
        coverage = (total_answers - len(missing)) / total_answers * 100 if total_answers > 0 else 0

        print(f"\n{rel_type.upper()}: {len(missing)} missing ({coverage:.0f}% coverage)")
        if missing and args.verbose:
            for cluster_id in missing[:10]:  # Show first 10
                print(f"  - {cluster_id}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

    # Summary
    print("\n" + "=" * 60)
    fully_connected = sum(1 for a in graph.answers.values()
                          if all(a.relations.get(rt) for rt in RELATION_TYPES))
    print(f"Fully connected: {fully_connected}/{total_answers} answers")
    print(f"Orphaned (no relations at all): {sum(1 for a in graph.answers.values() if not any(a.relations.get(rt) for rt in RELATION_TYPES))}")


def cmd_show(args):
    """Show details for a specific answer."""
    graph = load_qa_files(Path(args.data_dir))

    if args.relations_file:
        load_relations_file(Path(args.relations_file), graph)

    answer = graph.get_by_hash(args.ref) or graph.get_by_id(args.ref)

    if not answer:
        print(f"Answer not found: {args.ref}")
        return 1

    print(f"Hash:      {answer.answer_hash}")
    print(f"ID:        {answer.cluster_id}")
    print(f"Source:    {answer.answer_source}")
    print(f"Text:      {answer.answer_text}")
    print()
    print("Relations:")
    for rel_type in RELATION_TYPES:
        targets = answer.relations.get(rel_type, [])
        if targets:
            print(f"  {rel_type}:")
            for target_hash in targets:
                target = graph.get_by_hash(target_hash)
                if target:
                    print(f"    - {target_hash} ({target.cluster_id})")
                else:
                    print(f"    - {target_hash} (not found)")
        else:
            print(f"  {rel_type}: (none)")


def cmd_init_relations(args):
    """Initialize a relations file with all answers."""
    graph = load_qa_files(Path(args.data_dir))

    output = {
        "description": "Knowledge graph relations for Q/A playbook examples",
        "note": "References can be hash (abc123...) or cluster_id (json_litedb_streaming)",
        "relation_types": {
            "foundational": "Concepts this example depends on",
            "preliminary": "Setup steps required before this example",
            "compositional": "Examples that extend/build upon this one",
            "transitional": "Natural next steps after this example"
        },
        "answers": [
            {"hash": a.answer_hash, "id": a.cluster_id, "source": a.answer_source}
            for a in sorted(graph.answers.values(), key=lambda x: x.cluster_id)
        ],
        "relations": []
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Initialized relations file: {output_path}")
    print(f"Contains {len(output['answers'])} answers ready for relations")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Tool for Q/A Playbook Relations"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # List command
    list_parser = subparsers.add_parser('list', help='List all answers with hashes')
    list_parser.add_argument('data_dir', help='Directory containing qa_pairs*.json files')
    list_parser.add_argument('--relations-file', '-r', help='Relations JSON file')

    # Gaps command
    gaps_parser = subparsers.add_parser('gaps', help='Show knowledge gaps')
    gaps_parser.add_argument('data_dir', help='Directory containing qa_pairs*.json files')
    gaps_parser.add_argument('--relations-file', '-r', help='Relations JSON file')
    gaps_parser.add_argument('--verbose', '-v', action='store_true', help='Show details')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show answer details')
    show_parser.add_argument('data_dir', help='Directory containing qa_pairs*.json files')
    show_parser.add_argument('ref', help='Answer hash or cluster ID')
    show_parser.add_argument('--relations-file', '-r', help='Relations JSON file')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize relations file')
    init_parser.add_argument('data_dir', help='Directory containing qa_pairs*.json files')
    init_parser.add_argument('--output', '-o', default='relations.json', help='Output file')

    args = parser.parse_args()

    if args.command == 'list':
        return cmd_list(args)
    elif args.command == 'gaps':
        return cmd_gaps(args)
    elif args.command == 'show':
        return cmd_show(args)
    elif args.command == 'init':
        return cmd_init_relations(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
