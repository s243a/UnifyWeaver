#!/usr/bin/env python3
"""
Organize incomplete mindmaps into a hierarchical view.

Merges the incomplete tree URIs into a tree structure based on their
Pearltrees hierarchy paths, showing counts at each level.

This helps identify which topics have the most incomplete trees,
allowing you to prioritize API fetches by interest area.

Usage:
    python3 scripts/organize_incomplete_trees.py \
        --scan .local/data/scans/incomplete_mindmaps.json \
        --data reports/pearltrees_targets_full_multi_account.jsonl

    # Output as text tree:
    python3 scripts/organize_incomplete_trees.py --scan ... --format tree

    # Output as JSON for further processing:
    python3 scripts/organize_incomplete_trees.py --scan ... --format json

    # Save URLs grouped by top-level category:
    python3 scripts/organize_incomplete_trees.py --scan ... --format grouped
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """A node in the merged tree."""
    name: str
    children: Dict[str, 'TreeNode'] = field(default_factory=dict)
    incomplete_trees: List[dict] = field(default_factory=list)
    count: int = 0  # Total incomplete trees in this subtree


def load_path_lookup(data_path: Path) -> Dict[str, dict]:
    """Load tree_id -> full record mapping from JSONL."""
    lookup = {}
    with open(data_path) as f:
        for line in f:
            record = json.loads(line.strip())
            tree_id = record.get('tree_id', '')
            if tree_id:
                lookup[str(tree_id)] = record
    return lookup


def parse_path(target_text: str) -> List[str]:
    """Parse target_text into path components."""
    lines = target_text.split('\n')
    # Skip the ID line (starts with /)
    path_lines = [l for l in lines if not l.startswith('/')]

    path = []
    for line in path_lines:
        stripped = line.lstrip('- ').strip()
        if stripped:
            path.append(stripped)
    return path


def build_merged_tree(incomplete_trees: List[dict], path_lookup: Dict[str, dict]) -> TreeNode:
    """Build a merged tree from incomplete trees."""
    root = TreeNode('ROOT')

    for m in incomplete_trees:
        tree_id = m['tree_id']
        record = path_lookup.get(tree_id, {})
        target_text = record.get('target_text', '')

        if not target_text:
            # No path found, put under "Unknown"
            path = ['Unknown', m['title']]
        else:
            path = parse_path(target_text)

        # Navigate/create path
        current = root
        for part in path:
            if part not in current.children:
                current.children[part] = TreeNode(part)
            current = current.children[part]

        # Add the incomplete tree at the leaf
        current.incomplete_trees.append(m)

    # Compute counts
    def compute_counts(node: TreeNode) -> int:
        node.count = len(node.incomplete_trees)
        for child in node.children.values():
            node.count += compute_counts(child)
        return node.count

    compute_counts(root)
    return root


def format_tree(node: TreeNode, prefix: str = '', show_urls: bool = False, min_count: int = 1) -> str:
    """Format tree as string with box-drawing characters."""
    lines = []

    # Sort children by count (descending) then name
    items = sorted(node.children.items(), key=lambda x: (-x[1].count, x[0]))
    visible_items = [x for x in items if x[1].count >= min_count]

    for i, (name, child) in enumerate(visible_items):
        is_last = i == len(visible_items) - 1
        connector = '\u2514\u2500\u2500 ' if is_last else '\u251c\u2500\u2500 '

        # Show count
        count_str = f' ({child.count})' if child.count > 0 else ''

        lines.append(f'{prefix}{connector}{name}{count_str}')

        new_prefix = prefix + ('    ' if is_last else '\u2502   ')

        # Show URLs at this node (these are the incomplete trees filed here)
        if show_urls and child.incomplete_trees:
            for tree in child.incomplete_trees[:3]:  # Limit to 3 per folder
                lines.append(f'{new_prefix}\u2192 {tree["uri"]}')
            if len(child.incomplete_trees) > 3:
                lines.append(f'{new_prefix}... and {len(child.incomplete_trees) - 3} more')

        child_output = format_tree(child, new_prefix, show_urls, min_count)
        if child_output:
            lines.append(child_output)

    return '\n'.join(filter(None, lines))


def get_grouped_urls(node: TreeNode, depth: int = 1) -> Dict[str, List[str]]:
    """Get URLs grouped by category at specified depth."""
    groups = defaultdict(list)

    def collect(n: TreeNode, path: List[str], target_depth: int):
        if len(path) == target_depth:
            # Collect all URLs in this subtree
            def get_all_urls(node: TreeNode) -> List[str]:
                urls = [t['uri'] for t in node.incomplete_trees]
                for child in node.children.values():
                    urls.extend(get_all_urls(child))
                return urls

            category = ' > '.join(path) if path else 'Root'
            groups[category].extend(get_all_urls(n))
        else:
            for name, child in n.children.items():
                collect(child, path + [name], target_depth)

    collect(node, [], depth)
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(
        description='Organize incomplete mindmaps into hierarchical view',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--scan', type=Path, required=True,
                       help='Path to incomplete_mindmaps.json')
    parser.add_argument('--data', type=Path,
                       default=Path('reports/pearltrees_targets_full_multi_account.jsonl'),
                       help='Path to JSONL with path data')
    parser.add_argument('--format', choices=['tree', 'json', 'grouped', 'summary'],
                       default='tree', help='Output format')
    parser.add_argument('--min-count', type=int, default=1,
                       help='Minimum count to show in tree (default: 1)')
    parser.add_argument('--show-urls', action='store_true',
                       help='Show sample URLs in tree output')
    parser.add_argument('--group-depth', type=int, default=2,
                       help='Depth for grouped output (default: 2)')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output file (default: stdout)')

    args = parser.parse_args()

    # Load data
    print(f"Loading incomplete trees from {args.scan}...", file=__import__('sys').stderr)
    with open(args.scan) as f:
        scan_data = json.load(f)
    incomplete = scan_data['maps']
    print(f"Loaded {len(incomplete)} incomplete trees", file=__import__('sys').stderr)

    print(f"Loading path data from {args.data}...", file=__import__('sys').stderr)
    path_lookup = load_path_lookup(args.data)
    print(f"Loaded {len(path_lookup)} path entries", file=__import__('sys').stderr)

    # Build tree
    tree = build_merged_tree(incomplete, path_lookup)

    # Output
    if args.format == 'tree':
        output = format_tree(tree, min_count=args.min_count, show_urls=args.show_urls)
    elif args.format == 'json':
        def tree_to_dict(node: TreeNode) -> dict:
            return {
                'name': node.name,
                'count': node.count,
                'trees': [{'uri': t['uri'], 'title': t['title']} for t in node.incomplete_trees],
                'children': {k: tree_to_dict(v) for k, v in sorted(node.children.items())}
            }
        output = json.dumps(tree_to_dict(tree), indent=2)
    elif args.format == 'grouped':
        groups = get_grouped_urls(tree, args.group_depth)
        output = json.dumps(groups, indent=2)
    elif args.format == 'summary':
        # Show top-level categories with counts
        lines = ["Top-level categories by incomplete tree count:", ""]
        for name, child in sorted(tree.children.items(), key=lambda x: -x[1].count):
            lines.append(f"  {child.count:4d}  {name}")
        output = '\n'.join(lines)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Written to {args.output}", file=__import__('sys').stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
