#!/usr/bin/env python3
"""
Auto-suggest hierarchy repairs for orphaned trees.

Finds orphaned trees (those with account root as cluster_id) that have
same-account AliasPearls pointing to them. These AliasPearls indicate
where the tree should logically be placed in the hierarchy.

Usage:
    python3 scripts/suggest_hierarchy_repairs.py \
        --input reports/pearltrees_targets_combined.jsonl \
        --output data/hierarchy_repairs_suggested.json \
        --account s243a
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_account_from_uri(uri: str) -> Optional[str]:
    """Extract account name from Pearltrees URI."""
    # Handle both /s243a/ and /t/s243a/ formats
    match = re.search(r'pearltrees\.com/(?:t/)?([^/]+)/', uri)
    return match.group(1) if match else None


def is_orphan(entry: Dict, primary_account: str) -> bool:
    """Check if a tree is orphaned (cluster_id is account root)."""
    if entry.get('type') != 'Tree':
        return False

    cluster_id = entry.get('cluster_id', '')
    # Orphan if cluster_id is just the account root (no tree path)
    # e.g., "https://www.pearltrees.com/s243a" vs "https://www.pearltrees.com/s243a/folder/id123"

    # Check if cluster_id is account root (ends with account name, no further path)
    if cluster_id.rstrip('/').endswith(f'/{primary_account}'):
        return True
    if cluster_id.rstrip('/').endswith(f'.com/{primary_account}'):
        return True

    return False


def find_alias_links(jsonl_file: Path, primary_account: str) -> Dict[str, List[Dict]]:
    """
    Find all AliasPearls that link to trees, grouped by target URI.

    Returns: {target_tree_uri: [list of AliasPearl info]}
    """
    alias_links = defaultdict(list)

    # First pass: build set of trees owned by primary account
    primary_tree_uris = set()
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('type') == 'Tree' and entry.get('account') == primary_account:
                uri = entry.get('uri')
                if uri:
                    primary_tree_uris.add(uri)

    logger.info(f"Found {len(primary_tree_uris)} trees owned by {primary_account}")

    # Second pass: find AliasPearls pointing to primary account trees
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            if entry.get('type') != 'AliasPearl':
                continue

            target_uri = entry.get('alias_target_uri')
            if not target_uri:
                continue

            # Get the parent tree where this alias lives
            parent_tree_uri = entry.get('parent_tree_uri', entry.get('cluster_id', ''))
            alias_account = entry.get('account', '')

            # Only consider aliases owned by primary account pointing to primary account trees
            if alias_account == primary_account and target_uri in primary_tree_uris:
                alias_links[target_uri].append({
                    'alias_uri': entry.get('pearl_uri', ''),
                    'alias_title': entry.get('raw_title'),
                    'parent_tree_uri': parent_tree_uri,
                    'account': alias_account,
                    'pearl_type': 'AliasPearl'
                })

    return alias_links


def find_crossaccount_alias_links(jsonl_file: Path, primary_accounts: Set[str]) -> Dict[str, List[Dict]]:
    """
    Find AliasPearls from user accounts pointing to external account trees.

    This helps place external account trees in the user's hierarchy based on
    where the user has referenced them via AliasPearls. When multiple AliasPearls
    point to the same external tree, we can pick the best semantic fit.

    Returns: {target_tree_uri: [list of AliasPearl info]}
    """
    alias_links = defaultdict(list)

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            if entry.get('type') != 'AliasPearl':
                continue

            target_uri = entry.get('alias_target_uri')
            if not target_uri:
                continue

            pearl_account = entry.get('account', '')
            target_account = extract_account_from_uri(target_uri)

            # Only consider AliasPearls from user's accounts pointing to external trees
            if pearl_account in primary_accounts and target_account and target_account not in primary_accounts:
                parent_tree_uri = entry.get('parent_tree_uri', entry.get('cluster_id', ''))
                alias_links[target_uri].append({
                    'pearl_uri': entry.get('pearl_uri', ''),
                    'pearl_title': entry.get('raw_title'),
                    'parent_tree_uri': parent_tree_uri,
                    'account': pearl_account,
                    'pearl_type': 'AliasPearl'
                })

    return alias_links


def suggest_repairs(jsonl_file: Path, primary_account: str,
                    existing_repairs: Optional[Path] = None,
                    additional_accounts: Optional[Set[str]] = None) -> Dict[str, Dict]:
    """
    Suggest hierarchy repairs for orphaned trees.

    For same-account orphans: Uses AliasPearls to find where they should be placed.
    For external account orphans: Uses RefPearls from user's accounts to find
    the best semantic fit in the user's hierarchy.

    Args:
        jsonl_file: Input JSONL with trees and pearls
        primary_account: Main user account
        existing_repairs: Path to existing repairs file to skip
        additional_accounts: Other user accounts (e.g., s243a_groups) to include

    Returns: {tree_uri: {parent_uri, source, confidence, notes}}
    """
    # Build set of all user accounts
    user_accounts = {primary_account}
    if additional_accounts:
        user_accounts.update(additional_accounts)

    # Load existing repairs to skip
    existing = set()
    if existing_repairs and existing_repairs.exists():
        with open(existing_repairs, 'r') as f:
            data = json.load(f)
            if 'repairs' in data:
                existing = set(data['repairs'].keys())
            else:
                existing = set(k for k in data.keys() if not k.startswith('_'))
        logger.info(f"Loaded {len(existing)} existing repairs to skip")

    # Find all alias links (same-account)
    alias_links = find_alias_links(jsonl_file, primary_account)
    logger.info(f"Found {len(alias_links)} trees with same-account AliasPearl links")

    # Find cross-account AliasPearl links
    crossaccount_links = find_crossaccount_alias_links(jsonl_file, user_accounts)
    logger.info(f"Found {len(crossaccount_links)} external trees with AliasPearl links from user accounts")

    # Build tree lookup
    trees = {}
    same_account_orphans = []
    external_orphans = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('type') == 'Tree':
                uri = entry.get('uri')
                trees[uri] = entry
                tree_account = entry.get('account', '')

                if uri in existing:
                    continue

                # Categorize orphans
                if tree_account in user_accounts:
                    if is_orphan(entry, primary_account):
                        same_account_orphans.append(entry)
                else:
                    # External tree - check if we have AliasPearl links to it
                    if uri in crossaccount_links:
                        external_orphans.append(entry)

    logger.info(f"Found {len(same_account_orphans)} same-account orphans")
    logger.info(f"Found {len(external_orphans)} external orphans with AliasPearl links")

    suggestions = {}

    # Process same-account orphans using AliasPearl links
    for orphan in same_account_orphans:
        uri = orphan.get('uri')

        if uri not in alias_links:
            continue

        links = alias_links[uri]
        best_parent = None
        best_confidence = 0

        for link in links:
            parent_uri = link['parent_tree_uri']

            if parent_uri not in trees:
                continue

            parent_entry = trees[parent_uri]

            # Higher confidence if parent has a real hierarchy
            if not is_orphan(parent_entry, primary_account):
                confidence = 0.9
            else:
                confidence = 0.5

            if confidence > best_confidence:
                best_confidence = confidence
                best_parent = {
                    'parent_uri': parent_uri,
                    'parent_title': parent_entry.get('raw_title'),
                    'pearl_title': link.get('alias_title') or link.get('pearl_title'),
                    'pearl_type': link.get('pearl_type', 'AliasPearl'),
                    'confidence': confidence
                }

        if best_parent:
            suggestions[uri] = {
                'parent_uri': best_parent['parent_uri'],
                'source': 'semantic',
                'date': str(date.today()),
                'confidence': best_parent['confidence'],
                'notes': f"{best_parent['pearl_type']} '{best_parent['pearl_title']}' in '{best_parent['parent_title']}' points to this tree"
            }

    # Process external orphans using AliasPearl links (crossing account boundaries)
    for orphan in external_orphans:
        uri = orphan.get('uri')
        orphan_account = orphan.get('account', '')

        links = crossaccount_links.get(uri, [])
        best_parent = None
        best_confidence = 0

        for link in links:
            parent_uri = link['parent_tree_uri']

            if parent_uri not in trees:
                continue

            parent_entry = trees[parent_uri]

            # For external trees, confidence based on parent's hierarchy status
            if not is_orphan(parent_entry, primary_account):
                confidence = 0.8  # Slightly lower than same-account
            else:
                confidence = 0.4

            if confidence > best_confidence:
                best_confidence = confidence
                best_parent = {
                    'parent_uri': parent_uri,
                    'parent_title': parent_entry.get('raw_title'),
                    'pearl_title': link.get('pearl_title'),
                    'pearl_type': 'AliasPearl',
                    'confidence': confidence
                }

        if best_parent:
            suggestions[uri] = {
                'parent_uri': best_parent['parent_uri'],
                'source': 'semantic-crossaccount',
                'date': str(date.today()),
                'confidence': best_parent['confidence'],
                'notes': f"AliasPearl '{best_parent['pearl_title']}' in '{best_parent['parent_title']}' references this {orphan_account} tree"
            }

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Suggest hierarchy repairs for orphaned trees")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output suggestions JSON file")
    parser.add_argument("--account", type=str, required=True, help="Primary account name")
    parser.add_argument("--additional-accounts", type=str, nargs='*', default=[],
                       help="Additional user accounts (e.g., s243a_groups)")
    parser.add_argument("--existing", type=Path, default=None,
                       help="Existing repairs file to skip")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                       help="Minimum confidence threshold (0-1)")
    parser.add_argument("--merge", action="store_true",
                       help="Merge suggestions into existing repairs file")

    args = parser.parse_args()

    additional = set(args.additional_accounts) if args.additional_accounts else None
    suggestions = suggest_repairs(args.input, args.account, args.existing, additional)

    # Filter by confidence
    if args.min_confidence > 0:
        suggestions = {k: v for k, v in suggestions.items()
                      if v.get('confidence', 0) >= args.min_confidence}

    logger.info(f"Generated {len(suggestions)} repair suggestions")

    # Count by source type
    same_account = sum(1 for s in suggestions.values() if s.get('source') == 'semantic')
    cross_account = sum(1 for s in suggestions.values() if s.get('source') == 'semantic-crossaccount')
    logger.info(f"  Same-account repairs: {same_account}")
    logger.info(f"  Cross-account repairs: {cross_account}")

    # Output format
    output_data = {
        "_meta": {
            "description": "Auto-suggested hierarchy repairs from AliasPearl and RefPearl analysis",
            "format_version": "1.1",
            "generated": str(date.today()),
            "source_file": str(args.input),
            "primary_account": args.account,
            "additional_accounts": args.additional_accounts or []
        },
        "repairs": suggestions
    }

    # Merge with existing if requested
    if args.merge and args.existing and args.existing.exists():
        with open(args.existing, 'r') as f:
            existing_data = json.load(f)

        if 'repairs' in existing_data:
            # Add new suggestions, don't overwrite existing
            for uri, repair in suggestions.items():
                if uri not in existing_data['repairs']:
                    existing_data['repairs'][uri] = repair
            output_data = existing_data
            logger.info(f"Merged with existing repairs")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote suggestions to {args.output}")

    # Print summary
    if suggestions:
        print(f"\n=== Top suggestions (by confidence) ===")
        sorted_suggestions = sorted(suggestions.items(),
                                   key=lambda x: x[1].get('confidence', 0),
                                   reverse=True)
        for uri, info in sorted_suggestions[:10]:
            tree_id = uri.split('/')[-1]
            parent_id = info['parent_uri'].split('/')[-1]
            print(f"  {tree_id} -> {parent_id} (conf={info['confidence']:.1f})")
            print(f"    {info['notes']}")


if __name__ == "__main__":
    main()
