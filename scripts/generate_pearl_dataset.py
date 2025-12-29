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


# Query style presets for different use cases
QUERY_STYLES = {
    "raw": {
        "tree": "{title}",
        "pearl": "{title}",
        "description": "Just the title, no wrapper"
    },
    "locate": {
        "tree": "locate_node({title})",
        "pearl": "locate_pearl({title})",
        "description": "Locate a specific item"
    },
    "locate_object": {
        "tree": "locate_object({title})",
        "pearl": "locate_object({title})",
        "description": "Generic locate (same for all types)"
    },
    "file": {
        "tree": "file_bookmark({title})",
        "pearl": "file_bookmark({title})",
        "description": "Filing a bookmark into a folder"
    },
    "similar": {
        "tree": "find_similar({title})",
        "pearl": "find_similar({title})",
        "description": "Find semantically similar items"
    },
    "browse": {
        "tree": "browse_folder({title})",
        "pearl": "view_bookmark({title})",
        "description": "Browsing/viewing items"
    }
}


def extract_item_id(uri: str) -> str:
    """Extract item ID from URI like http://www.pearltrees.com/s243a/item704016118"""
    if not uri:
        return "unknown"
    match = re.search(r'(item\d+|pearl\d+|\d{8,})', uri)
    return f"#{match.group(1)}" if match else f"#{uri.split('/')[-1]}"


def get_query_text(title: str, obj_type: str, style: str) -> str:
    """
    Format query text based on object type and style.
    obj_type: 'tree' or 'pearl'
    style: key in QUERY_STYLES
    """
    if style not in QUERY_STYLES:
        # Fallback to raw
        return title
        
    templates = QUERY_STYLES[style]
    template = templates.get(obj_type, "{title}")
    return template.format(title=title)


def generate_pearl_targets(
    parser: MultiAccountParser,
    query_style: str = "raw",
    filter_path: Optional[str] = None,
    include_trees: bool = True,
    include_pearls: bool = True
) -> List[dict]:
    """
    Generate targets for both trees and pearls.
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
    
    # Generate tree targets
    if include_trees:
        logger.info("Generating tree targets...")
        
        # We need to manually generate targets to apply styles, instead of using parser.generate_targets
        # because the parser only uses a simple template
        count = 0
        skipped = 0
        
        for uri, tree in parser.trees.items():
            if not tree.title:
                skipped += 1
                continue
                
            # Get path
            path = get_tree_path(uri)
            
            # Filter
            if filter_path:
                path_str = " ".join([t for t, a in path])
                if filter_path.lower() not in path_str.lower():
                    skipped += 1
                    continue
            
            # Format target text (same logic as parser.generate_targets but using our helpers)
            # Reconstruct format_path logic from parser for consistency or use what we refer to
            # Actually, let's just use the path we built
            lines = []
            prev_account = None
            for title, account in path:
                if prev_account and account != prev_account:
                    lines.append(f"- {title} @{account}")
                else:
                    lines.append(f"- {title}")
                prev_account = account
            
            # Add root if needed, but the path already has it from get_tree_path logic?
            # get_tree_path adds root.
            # parser.generate_targets Logic:
            #   lines = [root_account]
            #   ...
            # The get_tree_path returns full chain.
            # The root account is the first account in the chain.
            
            # Let's simple format:
            root_account = path[0][1]
            formatted_lines = [root_account]
            
            prev_account = None
            for title, account in path:
                if prev_account and account != prev_account:
                    formatted_lines.append(f"- {title} @{account}")
                else:
                    formatted_lines.append(f"- {title}")
                prev_account = account
            
            target_text = "\n".join(formatted_lines)
            
            query_text = get_query_text(tree.title, "tree", query_style)
            
            cluster_id = tree.parent_uri or tree.section_based_parent_uri or "root"
            
            results.append({
                "data_type": "tree",
                "type": "Tree",
                "target_text": target_text,
                "raw_title": tree.title,
                "query": query_text,
                "cluster_id": cluster_id,
                "tree_id": tree.tree_id,
                "account": tree.account,
                "uri": uri
            })
            count += 1
            
        logger.info(f"  Generated {count} tree targets ({skipped} skipped)")
    
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
            query_text = get_query_text(pearl.title, "pearl", query_style)
            
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
    
    # Replace query-template with query-style
    style_choices = list(QUERY_STYLES.keys())
    style_help = "Query style: " + ", ".join([f"{k} ({v['description']})" for k, v in QUERY_STYLES.items()])
    
    arg_parser.add_argument("--query-style", type=str, default="raw", choices=style_choices,
                           help=style_help)
                           
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
        query_style=args.query_style,
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
