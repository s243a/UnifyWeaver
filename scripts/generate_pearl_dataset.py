#!/usr/bin/env python3
"""
Pearltrees Pearl Dataset Generator

Generates training data that includes both trees (folders) AND pearls (bookmarks).
This enables semantic search across all content, not just folder structure.

For a pearl, the materialized path includes the pearl title as the final item:
    s243a
    - STEM
    - Physics
    - Quantum Mechanics
    - #item704016118: "Introduction to Quantum Entanglement"

Usage:
    # Full dataset (may be large!)
    python3 scripts/generate_pearl_dataset.py \
        --account s243a data/s243a.rdf \
        --account s243a_groups data/s243a_groups.rdf \
        --output reports/pearltrees_targets_with_pearls.jsonl

    # Physics subset
    python3 scripts/generate_pearl_dataset.py \
        --account s243a data/s243a.rdf \
        --filter-path "Physics" \
        --output reports/pearltrees_targets_physics_pearls.jsonl
"""

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from pearltrees_multi_account_generator import MultiAccountParser, TreeNode, Pearl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_item_id(uri: str) -> str:
    """Extract item ID from URI like http://www.pearltrees.com/s243a/item704016118"""
    if not uri:
        return "unknown"
    match = re.search(r'(item\d+|pearl\d+|\d{8,})', uri)
    return f"#{match.group(1)}" if match else f"#{uri.split('/')[-1]}"


def generate_pearl_targets(
    parser: MultiAccountParser,
    query_template: str = "{title}",
    filter_path: Optional[str] = None,
    include_trees: bool = True,
    include_pearls: bool = True
) -> List[dict]:
    """
    Generate targets for both trees and pearls.
    
    Pearl target format:
        target_text: 
            s243a
            - STEM
            - Physics
            - #item123456: "Article Title"
        
        query: "Article Title"
        type: "Pearl"
    """
    results = []
    
    # Build path cache for trees
    path_cache = {}
    
    def get_tree_path(uri: str) -> List[tuple]:
        """Get path as list of (title, account) tuples."""
        if uri in path_cache:
            return path_cache[uri]
        
        if uri not in parser.trees:
            return []
        
        tree = parser.trees[uri]
        path = [(tree.title, tree.account)]
        
        # Walk up parent chain
        current = tree
        visited = {uri}
        
        while True:
            parent_uri = current.parent_uri or current.section_based_parent_uri
            
            # Check cross-account entry
            if not parent_uri and current.cross_account_entry_uri:
                parent_uri = current.cross_account_entry_uri
            
            if not parent_uri or parent_uri in visited or parent_uri not in parser.trees:
                break
            
            visited.add(parent_uri)
            parent = parser.trees[parent_uri]
            path.insert(0, (parent.title, parent.account))
            current = parent
        
        path_cache[uri] = path
        return path
    
    def format_path_with_pearl(tree_path: List[tuple], pearl_title: str, pearl_id: str) -> str:
        """Format path including the pearl as final item."""
        if not tree_path:
            return f"{pearl_id}: \"{pearl_title}\""
        
        lines = []
        prev_account = None
        
        for title, account in tree_path:
            if prev_account and account != prev_account:
                lines.append(f"- {title} @{account}")
            else:
                lines.append(f"- {title}")
            prev_account = account
        
        # Add root account
        root_account = tree_path[0][1]
        lines.insert(0, root_account)
        
        # Add pearl as final item
        lines.append(f"- {pearl_id}: \"{pearl_title}\"")
        
        return "\n".join(lines)
    
    # Generate tree targets (existing logic)
    if include_trees:
        logger.info("Generating tree targets...")
        tree_results = parser.generate_targets(query_template, filter_path)
        for r in tree_results:
            r['data_type'] = 'tree'
        results.extend(tree_results)
        logger.info(f"  Generated {len(tree_results)} tree targets")
    
    # Generate pearl targets
    if include_pearls:
        logger.info("Generating pearl targets...")
        pearl_count = 0
        skipped = 0
        
        for pearl in parser.pearls:
            # Skip RefPearls and AliasPearls (they're references, not content)
            if pearl.type.lower() in ('refpearl', 'aliaspearl'):
                skipped += 1
                continue
            
            if not pearl.title or not pearl.title.strip():
                skipped += 1
                continue
            
            # Get parent tree path
            tree_path = get_tree_path(pearl.parent_tree_uri) if pearl.parent_tree_uri else []
            
            # Apply filter
            if filter_path:
                path_str = " ".join([t for t, a in tree_path])
                if filter_path.lower() not in path_str.lower():
                    skipped += 1
                    continue
            
            # Extract item ID from URI
            pearl_id = extract_item_id(pearl.uri)
            
            # Format target text
            target_text = format_path_with_pearl(tree_path, pearl.title, pearl_id)
            
            # Format query
            try:
                query_text = query_template.format(title=pearl.title)
            except KeyError:
                query_text = pearl.title
            
            # Cluster by parent tree
            cluster_id = pearl.parent_tree_uri or "root"
            
            results.append({
                "data_type": "pearl",
                "type": pearl.type,
                "target_text": target_text,
                "raw_title": pearl.title,
                "query": query_text,
                "cluster_id": cluster_id,
                "pearl_id": pearl_id,
                "pearl_uri": pearl.uri,
                "parent_tree_uri": pearl.parent_tree_uri,
                "account": pearl.account,
            })
            pearl_count += 1
        
        logger.info(f"  Generated {pearl_count} pearl targets ({skipped} skipped)")
    
    return results


def main():
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Generate Pearltrees dataset including pearls (bookmarks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    arg_parser.add_argument("--account", action="append", nargs=2, metavar=("NAME", "FILE"),
                           required=True,
                           help="Account name and RDF file path")
    arg_parser.add_argument("--output", "-o", type=Path, required=True,
                           help="Output JSONL file")
    arg_parser.add_argument("--primary", type=str, default=None,
                           help="Primary account name (default: first account)")
    arg_parser.add_argument("--query-template", type=str, default="{title}",
                           help="Template for query (use {title})")
    arg_parser.add_argument("--filter-path", type=str, default=None,
                           help="Only include items with this string in path")
    arg_parser.add_argument("--trees-only", action="store_true",
                           help="Only generate tree targets (no pearls)")
    arg_parser.add_argument("--pearls-only", action="store_true",
                           help="Only generate pearl targets (no trees)")
    
    args = arg_parser.parse_args()
    
    # Parse accounts
    accounts = [(name, Path(path)) for name, path in args.account]
    primary = args.primary or accounts[0][0]
    
    # Validate files
    for name, path in accounts:
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
    
    # Parse RDF files
    parser = MultiAccountParser(primary_account=primary)
    
    for name, path in accounts:
        logger.info(f"Parsing {name} from {path}...")
        parser.add_rdf_file(path, name)
    
    parser.parse_all()
    logger.info(f"Parsed {len(parser.trees)} trees, {len(parser.pearls)} pearls")
    
    # Generate targets
    results = generate_pearl_targets(
        parser,
        query_template=args.query_template,
        filter_path=args.filter_path,
        include_trees=not args.pearls_only,
        include_pearls=not args.trees_only
    )
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Stats
    tree_count = sum(1 for r in results if r.get('data_type') == 'tree')
    pearl_count = sum(1 for r in results if r.get('data_type') == 'pearl')
    
    logger.info(f"Wrote {len(results)} targets to {args.output}")
    logger.info(f"  Trees: {tree_count}")
    logger.info(f"  Pearls: {pearl_count}")


if __name__ == "__main__":
    main()
