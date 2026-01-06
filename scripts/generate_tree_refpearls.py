#!/usr/bin/env python3
"""
Generate RefPearls for all parent-child tree relationships.

The RDF export expresses hierarchy through cluster_id fields, but the mindmap
generator needs explicit RefPearls to discover child trees for recursive
generation. This script creates RefPearls from the existing tree hierarchy.

Usage:
    python3 scripts/generate_tree_refpearls.py \
        --input reports/pearltrees_targets.jsonl \
        --output reports/pearltrees_targets_with_refs.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_refpearls(input_path: Path, output_path: Path, accounts: Set[str] = None):
    """Generate RefPearls for parent-child tree relationships.

    Args:
        input_path: Input JSONL with trees
        output_path: Output JSONL with trees + generated RefPearls
        accounts: Optional set of accounts to process (None = all)
    """
    # First pass: index all trees by URI and collect parent-child relationships
    trees_by_uri = {}
    children_by_parent = {}  # parent_uri -> list of child uris
    existing_refpearls = set()  # URIs of existing RefPearls

    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)

            entry_type = entry.get('type', '')

            if entry_type == 'Tree':
                uri = entry.get('uri', '')
                if uri:
                    trees_by_uri[uri] = entry

                    # Record parent-child relationship
                    cluster_id = entry.get('cluster_id', '')
                    if cluster_id and cluster_id != uri:
                        if cluster_id not in children_by_parent:
                            children_by_parent[cluster_id] = []
                        children_by_parent[cluster_id].append(uri)

            elif entry_type == 'RefPearl':
                # Track existing RefPearls to avoid duplicates
                target = entry.get('alias_target_uri', '')
                parent = entry.get('parent_tree_uri', '') or entry.get('cluster_id', '')
                existing_refpearls.add(f"{parent}#{target}")

    logger.info(f"Loaded {len(trees_by_uri)} trees")
    logger.info(f"Found {len(children_by_parent)} parent trees with children")
    logger.info(f"Existing RefPearls: {len(existing_refpearls)}")

    # Generate RefPearls for each parent-child relationship
    generated_refs = []
    for parent_uri, child_uris in children_by_parent.items():
        parent_tree = trees_by_uri.get(parent_uri)
        if not parent_tree:
            continue

        parent_account = parent_tree.get('account', '')
        if accounts and parent_account not in accounts:
            continue

        for child_uri in child_uris:
            child_tree = trees_by_uri.get(child_uri)
            if not child_tree:
                continue

            child_account = child_tree.get('account', '')
            if accounts and child_account not in accounts:
                continue

            # Check if RefPearl already exists
            key = f"{parent_uri}#{child_uri}"
            if key in existing_refpearls:
                continue

            # Extract tree_id from child URI
            import re
            id_match = re.search(r'/id(\d+)$', child_uri)
            child_tree_id = id_match.group(1) if id_match else child_tree.get('tree_id', '')

            # Create RefPearl
            ref_pearl = {
                'type': 'RefPearl',
                'raw_title': child_tree.get('raw_title', ''),
                'query': child_tree.get('raw_title', ''),
                'cluster_id': parent_uri,
                'pearl_id': f"ref_{child_tree_id}" if child_tree_id else f"ref_{len(generated_refs)}",
                'pearl_uri': f"{parent_uri}#ref_{child_tree_id}" if child_tree_id else f"{parent_uri}#ref_{len(generated_refs)}",
                'parent_tree_uri': parent_uri,
                'alias_target_uri': child_uri,
                'account': parent_account,
                '_source': 'hierarchy_inference',
            }
            generated_refs.append(ref_pearl)

    logger.info(f"Generated {len(generated_refs)} new RefPearls")

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        for ref in generated_refs:
            f.write(json.dumps(ref, ensure_ascii=False) + '\n')

    logger.info(f"Wrote {len(entries) + len(generated_refs)} entries to {output_path}")

    # Summary
    print(f"\n=== RefPearl Generation Summary ===")
    print(f"Input: {input_path}")
    print(f"Trees: {len(trees_by_uri)}")
    print(f"Parent-child relationships: {sum(len(v) for v in children_by_parent.values())}")
    print(f"Existing RefPearls: {len(existing_refpearls)}")
    print(f"Generated RefPearls: {len(generated_refs)}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RefPearls for parent-child tree relationships")
    parser.add_argument("--input", type=Path, required=True,
                       help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output JSONL file")
    parser.add_argument("--accounts", nargs='*', default=None,
                       help="Only process these accounts (default: all)")

    args = parser.parse_args()

    accounts = set(args.accounts) if args.accounts else None
    generate_refpearls(args.input, args.output, accounts)


if __name__ == "__main__":
    main()
