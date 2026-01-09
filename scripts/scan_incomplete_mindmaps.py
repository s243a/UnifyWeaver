#!/usr/bin/env python3
"""
Scan mindmaps for incomplete/missing data.

Identifies mindmaps with only 1 node (root only), which typically indicates
the tree data wasn't properly fetched or the RDF export was incomplete.

Usage:
    python3 scripts/scan_incomplete_mindmaps.py output/mindmaps_curated/
    python3 scripts/scan_incomplete_mindmaps.py output/mindmaps_curated/ --account s243a
    python3 scripts/scan_incomplete_mindmaps.py output/mindmaps_curated/ --output incomplete.json
    python3 scripts/scan_incomplete_mindmaps.py output/mindmaps_curated/ --threshold 3
"""

import argparse
import json
import zipfile
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional


def count_topics(smmx_path: Path) -> tuple:
    """Count topics in a mindmap and extract metadata.

    Returns:
        (topic_count, title, tree_id, uri) or (None, None, None, None) on error
    """
    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
            topic_count = xml_content.count('<topic ')

            # Extract title
            title_match = re.search(r'<title text="([^"]*)"', xml_content)
            title = title_match.group(1) if title_match else "Unknown"

            # Extract tree ID from filename
            # Supports: id{number}, Title_id{number}, Title_{number}
            stem = smmx_path.stem
            tree_id = None

            # Try id{number} prefix
            if stem.startswith('id') and stem[2:].isdigit():
                tree_id = stem[2:]
            else:
                # Try Title_id{number} or Title_{number} suffix
                id_match = re.search(r'_id(\d+)$', stem)
                if id_match:
                    tree_id = id_match.group(1)
                else:
                    # Try just trailing digits (at least 6)
                    id_match = re.search(r'_(\d{6,})$', stem)
                    if id_match:
                        tree_id = id_match.group(1)

            # If still no tree_id, extract from urllink inside the mindmap
            if not tree_id:
                url_id_match = re.search(r'urllink="https://www\.pearltrees\.com/[^"]+/id(\d+)"', xml_content)
                if url_id_match:
                    tree_id = url_id_match.group(1)
                else:
                    # Last resort: fallback to stem
                    tree_id = stem

            # Extract actual URI from root topic's urllink
            uri_match = re.search(r'urllink="(https://www\.pearltrees\.com/[^"]+/id' + tree_id + r')"', xml_content)
            if uri_match:
                uri = uri_match.group(1)
            else:
                # Fallback: try to find any pearltrees URL with this tree_id
                uri_match = re.search(r'urllink="(https://www\.pearltrees\.com/[^"]*id' + tree_id + r'[^"]*)"', xml_content)
                uri = uri_match.group(1) if uri_match else None

            return topic_count, title, tree_id, uri
    except Exception as e:
        return None, None, None, None


def detect_account(path: Path, base_dir: Path) -> str:
    """Detect account from path structure."""
    try:
        rel_path = str(path.relative_to(base_dir))
    except ValueError:
        rel_path = str(path)

    # Check path patterns for known accounts
    if 's243a_groups' in rel_path or 's243a-groups' in rel_path:
        return 's243a_groups'
    elif 's243a' in rel_path:
        return 's243a'
    elif rel_path.startswith('Gods_of_Earth'):
        return 's243a'
    elif rel_path.startswith('Chomsky'):
        return 's243a'
    else:
        # Check base_dir name for account hints
        base_name = base_dir.name.lower()
        if 's243a_groups' in base_name or 's243a-groups' in base_name:
            return 's243a_groups'
        elif 'curated' in base_name or 's243a' in base_name:
            # mindmaps_curated is s243a's export
            return 's243a'
        return 'other'


def get_pearltrees_uri(tree_id: str, account: str) -> str:
    """Generate Pearltrees URI for a tree."""
    if account == 's243a_groups':
        return f"https://www.pearltrees.com/s243a-groups/id{tree_id}"
    elif account == 's243a':
        return f"https://www.pearltrees.com/s243a/id{tree_id}"
    else:
        # For unknown accounts, use generic format (user can look it up)
        return f"https://www.pearltrees.com/*/id{tree_id}"


def scan_mindmaps(
    base_dir: Path,
    threshold: int = 1,
    account_filter: Optional[str] = None,
    exclude_private: bool = False
) -> List[Dict]:
    """Scan mindmaps and find incomplete ones.

    Args:
        base_dir: Directory containing .smmx files
        threshold: Maximum topic count to consider incomplete (default: 1)
        account_filter: Only include maps from this account
        exclude_private: Exclude maps with "*private*" title

    Returns:
        List of dicts with path, title, tree_id, account, topic_count
    """
    smmx_files = list(base_dir.rglob("*.smmx"))
    incomplete = []

    for smmx_path in smmx_files:
        topic_count, title, tree_id, uri = count_topics(smmx_path)

        if topic_count is None:
            continue

        if topic_count > threshold:
            continue

        if exclude_private and title == "*private*":
            continue

        account = detect_account(smmx_path, base_dir)

        if account_filter and account != account_filter:
            continue

        try:
            rel_path = str(smmx_path.relative_to(base_dir))
        except ValueError:
            rel_path = str(smmx_path)

        # Use extracted URI from mindmap, fall back to constructed URI
        if not uri:
            uri = get_pearltrees_uri(tree_id, account)

        incomplete.append({
            'path': rel_path,
            'title': title,
            'tree_id': tree_id,
            'account': account,
            'topic_count': topic_count,
            'uri': uri
        })

    return incomplete


def main():
    parser = argparse.ArgumentParser(description='Scan for incomplete mindmaps')
    parser.add_argument('directory', type=Path, help='Directory to scan')
    parser.add_argument('--threshold', type=int, default=1,
                        help='Max topic count to consider incomplete (default: 1)')
    parser.add_argument('--account', type=str, default=None,
                        help='Filter by account (e.g., s243a, s243a_groups)')
    parser.add_argument('--exclude-private', action='store_true',
                        help='Exclude maps with "*private*" title')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--format', choices=['summary', 'list', 'json', 'urls'],
                        default='summary', help='Output format')
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        return 1

    print(f"Scanning {args.directory}...")
    incomplete = scan_mindmaps(
        args.directory,
        threshold=args.threshold,
        account_filter=args.account,
        exclude_private=args.exclude_private
    )

    # Sort by account, then title
    incomplete.sort(key=lambda x: (x['account'], x['title']))

    if args.format == 'json' or args.output:
        output_data = {
            'count': len(incomplete),
            'threshold': args.threshold,
            'account_filter': args.account,
            'maps': incomplete
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Written {len(incomplete)} entries to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))

    elif args.format == 'urls':
        # Output Pearltrees URLs for easy fetching
        for m in incomplete:
            print(m['uri'])

    elif args.format == 'list':
        for m in incomplete:
            print(f"{m['tree_id']}\t{m['title']}\t{m['account']}\t{m['path']}")

    else:  # summary
        print(f"\nFound {len(incomplete)} incomplete mindmaps (threshold: {args.threshold})\n")

        # Group by account
        by_account = defaultdict(list)
        for m in incomplete:
            by_account[m['account']].append(m)

        print("By account:")
        for acc, maps in sorted(by_account.items(), key=lambda x: -len(x[1])):
            print(f"  {acc}: {len(maps)}")

        # Show sample
        print(f"\nSample (first 20):")
        print("-" * 80)
        for m in incomplete[:20]:
            title_display = m['title'][:40] if len(m['title']) > 40 else m['title']
            print(f"  {m['tree_id']:>10} | {title_display:<40} | {m['account']}")

        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")

        print(f"\nUse --format=urls to get Pearltrees URLs for repair")
        print(f"Use --output=file.json to save full results")


if __name__ == '__main__':
    exit(main() or 0)
