#!/usr/bin/env python3
"""
Convert Pearltrees API database to tree data JSONL format.

Reads api_responses.db and builds path hierarchies by traversing
parent-child relationships via contentTree references.

Usage:
    python3 scripts/convert_api_to_tree_data.py \
        --db .local/data/pearltrees_api/api_responses.db \
        --output .local/data/tree_paths.jsonl
"""

import argparse
import json
import sqlite3
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_api_responses(db_path: Path) -> Dict[str, dict]:
    """Load all API responses from database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT tree_id, title, response_json FROM api_responses")

    trees = {}
    for tree_id, title, response_json in cur.fetchall():
        try:
            resp = json.loads(response_json)
            trees[str(tree_id)] = {
                'tree_id': str(tree_id),
                'title': title,
                'response': resp
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for tree {tree_id}: {e}")

    conn.close()
    logger.info(f"Loaded {len(trees)} trees from {db_path}")
    return trees


def load_tree_files(trees_dir: Path) -> Dict[str, dict]:
    """Load tree data from individual JSON files.

    Expects files with 'api_response' key (Pearltrees API format).
    """
    trees = {}
    skipped = 0

    if not trees_dir.exists():
        logger.warning(f"Trees directory not found: {trees_dir}")
        return trees

    for f in trees_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)

            # Validate structure - must be a dict with api_response
            if not isinstance(data, dict):
                skipped += 1
                continue
            if 'api_response' not in data:
                skipped += 1
                continue

            tree_id = str(data.get('tree_id', f.stem))
            resp = data.get('api_response', {})
            title = resp.get('tree', {}).get('title', '')
            trees[tree_id] = {
                'tree_id': tree_id,
                'title': title,
                'response': resp
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {f}: {e}")

    if skipped:
        logger.info(f"Skipped {skipped} files without api_response structure")
    logger.info(f"Loaded {len(trees)} trees from {trees_dir}")
    return trees


def parse_rdf_parents(rdf_path: Path) -> Tuple[Dict[str, tuple], Dict[str, str]]:
    """Parse RDF file for RefPearl/AliasPearl parent relationships, tree titles, and timestamps.

    Pearltrees RDF Schema (see context/PT/pearltrees_xmlns.rdf):
    - pt:Tree: A collection that contains pearls
    - pt:RefPearl: "A pearl of type ref references a tree INSIDE the parent tree"
      (internal sub-tree owned by same user)
    - pt:AliasPearl: "A pearl of type alias references a tree OUTSIDE the parent tree"
      (cross-link to another user's tree or different branch)

    Both RefPearl and AliasPearl have:
    - rdfs:seeAlso: URL of the referenced tree (the child)
    - pt:parentTree: URL of the tree containing this pearl (the parent)

    Example RDF:
        <pt:RefPearl rdf:about="...id11820376?show=item,101710908">
           <dcterms:title>Subjects of learning</dcterms:title>
           <rdfs:seeAlso rdf:resource=".../subjects-of-learning/id10444105" />
           <pt:parentTree rdf:resource=".../root-node-pearl-trees-version/id11820376" />
        </pt:RefPearl>

    This means: tree 10444105 ("Subjects of learning") is a child of tree 11820376.

    For account root level, parentTree may have no id:
        <pt:parentTree rdf:resource="https://www.pearltrees.com/s243a" />
    In this case, the child tree is directly under the account root.

    Returns:
        Tuple of:
        - Dict mapping child_tree_id -> (parent_tree_id, parent_title)
        - Dict mapping tree_id -> lastUpdate timestamp string (for precedence)
    """
    import re

    if not rdf_path.exists():
        logger.warning(f"RDF file not found: {rdf_path}")
        return {}, {}

    with open(rdf_path, 'r') as f:
        content = f.read()

    # Extract tree titles and lastUpdate timestamps from pt:Tree elements
    # lastUpdate is when the tree was last modified in Pearltrees
    tree_pattern = re.compile(
        r'<pt:Tree rdf:about="[^"]*id(\d+)">\s*'
        r'<dcterms:title><!\[CDATA\[([^\]]+)\]\]></dcterms:title>'
        r'.*?<pt:lastUpdate>([^<]*)</pt:lastUpdate>',
        re.MULTILINE | re.DOTALL
    )
    tree_titles = {}
    tree_timestamps = {}
    for tree_id, title, last_update in tree_pattern.findall(content):
        tree_titles[tree_id] = title
        if last_update:
            tree_timestamps[tree_id] = last_update
    logger.info(f"Parsed {len(tree_titles)} tree titles and {len(tree_timestamps)} timestamps from RDF")

    def extract_parent_from_url(parent_url: str) -> Optional[Tuple[str, str]]:
        """Extract parent_id and parent_title from a parentTree URL.

        Returns (parent_id, parent_title) or None if can't parse.
        """
        # Two cases: tree URL with id, or account root URL without id
        parent_id_match = re.search(r'id(\d+)$', parent_url)
        if parent_id_match:
            # Parent is a tree with numeric ID
            parent_id = parent_id_match.group(1)
            parent_title = tree_titles.get(parent_id, '')
            return (parent_id, parent_title)
        else:
            # Parent is account root (e.g., https://www.pearltrees.com/s243a)
            # Extract account name and use synthetic ID
            account_match = re.search(r'pearltrees\.com/([^/#]+)$', parent_url)
            if account_match:
                parent_id = f"account:{account_match.group(1)}"
                parent_title = account_match.group(1)
                return (parent_id, parent_title)
        return None

    # Pattern to extract parent-child relationships
    # We parse RefPearl and AliasPearl separately for precedence control
    def make_pearl_pattern(pearl_type: str):
        return re.compile(
            rf'<pt:{pearl_type}[^>]*>\s*'
            r'<dcterms:title><!\[CDATA\[([^\]]+)\]\]></dcterms:title>\s*'
            r'<rdfs:seeAlso rdf:resource="[^"]*id(\d+)"[^/]*/>\s*'
            r'<pt:parentTree rdf:resource="([^"]*)"',
            re.MULTILINE | re.DOTALL
        )

    parent_map = {}

    # First pass: RefPearl for canonical hierarchy
    # RefPearl = "references a tree INSIDE the parent tree" (owned sub-trees)
    ref_pattern = make_pearl_pattern('RefPearl')
    for title, child_id, parent_url in ref_pattern.findall(content):
        if child_id not in parent_map:
            result = extract_parent_from_url(parent_url)
            if result:
                parent_map[child_id] = result

    ref_count = len(parent_map)
    logger.info(f"Parsed {ref_count} RefPearl relationships (canonical hierarchy)")

    # Second pass: AliasPearl as fallback for bridging/cross-account links
    # AliasPearl = "references a tree OUTSIDE the parent tree" (cross-links)
    # Only use if no RefPearl relationship exists for this child
    # This handles: cross-account bridging, Navigate Up sections, etc.
    alias_pattern = make_pearl_pattern('AliasPearl')
    for title, child_id, parent_url in alias_pattern.findall(content):
        if child_id not in parent_map:
            result = extract_parent_from_url(parent_url)
            if result:
                parent_map[child_id] = result

    alias_count = len(parent_map) - ref_count
    logger.info(f"Parsed {alias_count} AliasPearl relationships (fallback/bridging)")

    # Also store parent titles in the map for path building
    for parent_id, title in tree_titles.items():
        if parent_id not in parent_map:
            parent_map[parent_id] = (None, title)

    total_with_parents = len([k for k, v in parent_map.items() if v[0]])
    logger.info(f"Total: {total_with_parents} trees with parent relationships from RDF")
    return parent_map, tree_timestamps


def build_parent_map(trees: Dict[str, dict],
                     rdf_parents: Optional[Dict[str, tuple]] = None,
                     rdf_timestamps: Optional[Dict[str, str]] = None) -> Dict[str, tuple]:
    """Build child_tree_id -> (parent_tree_id, parent_title) mapping.

    Merges parent information from two sources with timestamp-based precedence:

    1. API data (from trees dict):
       - Source: Pearltrees API via info.parentTree
       - Timestamp: fetched_at (when we downloaded the data)
       - Pros: Direct parent reference, authoritative for that tree
       - Cons: May be stale if tree was moved after fetch

    2. RDF data (from rdf_parents):
       - Source: Pearltrees RDF export via RefPearl/AliasPearl
       - Timestamp: lastUpdate (when tree was last modified in Pearltrees)
       - Pros: May be more recent, includes full account hierarchy
       - Cons: Aliases can create multiple parents (first match wins)

    Precedence logic:
    - If both sources have parent info, compare timestamps:
      - RDF lastUpdate > API fetched_at: RDF wins (tree modified after we fetched)
      - API fetched_at > RDF lastUpdate: API wins (our fetch is more recent)
    - If only one source has data, use it
    - If timestamps can't be compared, prefer API (has direct parentTree)

    This ensures we use the most current information available while
    not losing data when one source is missing information.
    """
    from datetime import datetime

    def parse_timestamp(ts: str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime."""
        if not ts:
            return None
        try:
            # Handle various ISO formats (with/without microseconds, Z suffix)
            ts = ts.replace('Z', '+00:00')
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    parent_map = {}
    rdf_timestamps = rdf_timestamps or {}

    # First pass: collect API parent info with fetch timestamps
    # API provides info.parentTree which is the authoritative parent for that tree
    api_parents = {}  # tree_id -> (parent_id, parent_title, fetched_at)

    for tree_id, tree_data in trees.items():
        resp = tree_data.get('response', {})
        info = resp.get('info', {})
        parent_tree = info.get('parentTree', {})

        if parent_tree and 'id' in parent_tree:
            parent_id = str(parent_tree['id'])
            parent_title = parent_tree.get('title', '')
            # fetched_at is when we downloaded this tree's data
            fetched_at = tree_data.get('fetched_at', '')
            api_parents[tree_id] = (parent_id, parent_title, fetched_at)

    # Merge with timestamp-based precedence
    all_tree_ids = set(api_parents.keys())
    if rdf_parents:
        all_tree_ids.update(k for k, v in rdf_parents.items() if v[0])  # Only IDs with actual parents

    rdf_used = 0
    api_used = 0

    for tree_id in all_tree_ids:
        has_api = tree_id in api_parents
        has_rdf = tree_id in rdf_parents and rdf_parents[tree_id][0]

        if has_api and has_rdf:
            # Both sources have data - compare timestamps
            api_parent_id, api_title, api_fetched = api_parents[tree_id]
            rdf_parent_id, rdf_title = rdf_parents[tree_id]
            rdf_updated = rdf_timestamps.get(tree_id, '')

            api_ts = parse_timestamp(api_fetched)
            rdf_ts = parse_timestamp(rdf_updated)

            if api_ts and rdf_ts:
                if rdf_ts > api_ts:
                    # RDF is more recent
                    parent_map[tree_id] = (rdf_parent_id, rdf_title)
                    rdf_used += 1
                else:
                    # API is more recent or same
                    parent_map[tree_id] = (api_parent_id, api_title)
                    api_used += 1
            else:
                # Can't compare timestamps - prefer API (has direct parentTree)
                parent_map[tree_id] = (api_parent_id, api_title)
                api_used += 1
        elif has_api:
            # Only API has data
            api_parent_id, api_title, _ = api_parents[tree_id]
            parent_map[tree_id] = (api_parent_id, api_title)
            api_used += 1
        elif has_rdf:
            # Only RDF has data
            rdf_parent_id, rdf_title = rdf_parents[tree_id]
            parent_map[tree_id] = (rdf_parent_id, rdf_title)
            rdf_used += 1

    # Also include RDF entries for title lookups (nodes without parent info)
    if rdf_parents:
        for tree_id, (parent_id, title) in rdf_parents.items():
            if tree_id not in parent_map and not parent_id:
                parent_map[tree_id] = (None, title)

    # Handle contentTree references from API (fallback for unmapped children)
    for tree_id, tree_data in trees.items():
        resp = tree_data.get('response', {})
        tree_info = resp.get('tree', {})
        pearls = tree_info.get('pearls', [])
        for pearl in pearls:
            content_tree = pearl.get('contentTree', {})
            if content_tree and 'id' in content_tree:
                child_id = str(content_tree['id'])
                if child_id not in parent_map:
                    parent_map[child_id] = (tree_id, tree_data['title'])

    logger.info(f"Built parent map: {api_used} from API, {rdf_used} from RDF, {len(parent_map)} total relationships")
    return parent_map


def get_path_to_root(tree_id: str, trees: Dict[str, dict], parent_map: Dict[str, tuple],
                     max_depth: int = 20) -> Tuple[List[str], List[str]]:
    """Get path from tree to root as list of titles and IDs.

    Uses parent_map which maps tree_id -> (parent_id, parent_title).
    This allows building paths even when parent trees weren't fetched,
    because we store parent titles in the parent_map.

    Returns:
        Tuple of (path_titles, path_ids) in root -> leaf order
    """
    path = []
    path_ids = []
    current_id = tree_id
    visited = set()

    while current_id and len(path) < max_depth:
        if current_id in visited:
            logger.warning(f"Cycle detected at {current_id}")
            break
        visited.add(current_id)

        # Get title and parent info for current node
        if current_id in trees:
            title = trees[current_id]['title']
            # Get parent from the tree's own info.parentTree
            resp = trees[current_id].get('response', {})
            info = resp.get('info', {})
            parent_tree = info.get('parentTree', {})
            if parent_tree and 'id' in parent_tree:
                next_id = str(parent_tree['id'])
                next_title = parent_tree.get('title', '')
                # Store in parent_map for future lookups
                if next_id not in parent_map:
                    parent_map[next_id] = (None, next_title)  # No parent of parent known
            else:
                next_id = None
        else:
            # Node not in trees - check if we have its title from parent_map
            title = None
            next_id = None
            # Look for this ID as a parent in parent_map entries
            for child_id, (parent_id, parent_title) in parent_map.items():
                if parent_id == current_id and parent_title:
                    title = parent_title
                    break
            if not title:
                # Check if current_id is stored as a key with title info
                if current_id in parent_map:
                    _, stored_title = parent_map[current_id]
                    if stored_title:
                        title = stored_title
            if not title:
                title = f"Unknown({current_id})"

        path.append(title)
        path_ids.append(current_id)

        # Move to parent - first check parent_map, then use next_id from tree
        parent_info = parent_map.get(current_id)
        if parent_info and parent_info[0]:
            current_id = parent_info[0]
        elif next_id:
            current_id = next_id
        else:
            current_id = None

    # Reverse to get root -> leaf order
    path.reverse()
    path_ids.reverse()
    return path, path_ids


def build_target_text(path: List[str], tree_id: str) -> str:
    """Build target_text in the expected format."""
    lines = [f"/{tree_id}"]  # ID line
    indent = ""
    for title in path:
        lines.append(f"{indent}- {title}")
        indent += "  "
    return "\n".join(lines)


def convert_to_jsonl(trees: Dict[str, dict], parent_map: Dict[str, str],
                     output_path: Path, account: str = "s243a"):
    """Convert trees to JSONL format with target_text paths and IDs."""
    count = 0

    with open(output_path, 'w') as f:
        for tree_id, tree_data in trees.items():
            path, path_ids = get_path_to_root(tree_id, trees, parent_map)

            if not path:
                # No path found, use just the title
                path = [tree_data['title']]
                path_ids = [tree_id]

            # If path doesn't start with account, prepend account as the root
            # Root-level trees in Pearltrees have the account as implicit parent
            if path and path[0] != account:
                path.insert(0, account)
                path_ids.insert(0, f"account:{account}")

            target_text = build_target_text(path, tree_id)

            # Get direct parent tree_id if available
            parent_tree_id = path_ids[-2] if len(path_ids) >= 2 else None

            entry = {
                'tree_id': tree_id,
                'title': tree_data['title'],
                'account': account,
                'target_text': target_text,
                'path_depth': len(path),
                'path_ids': path_ids,
                'parent_tree_id': parent_tree_id
            }

            f.write(json.dumps(entry) + '\n')
            count += 1

    logger.info(f"Wrote {count} entries to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pearltrees API data to tree data JSONL",
        epilog="""
Examples:
  # From database:
  python3 scripts/convert_api_to_tree_data.py \\
    --db .local/data/pearltrees_api/api_responses.db \\
    --output .local/data/api_tree_paths.jsonl

  # From tree files directory:
  python3 scripts/convert_api_to_tree_data.py \\
    --trees-dir .local/data/pearltrees_api/trees \\
    --output .local/data/api_tree_paths.jsonl

  # Combine both sources:
  python3 scripts/convert_api_to_tree_data.py \\
    --db .local/data/pearltrees_api/api_responses.db \\
    --trees-dir .local/data/pearltrees_api/trees \\
    --output .local/data/api_tree_paths.jsonl

  # With RDF for additional parent relationships:
  python3 scripts/convert_api_to_tree_data.py \\
    --trees-dir .local/data/pearltrees_api/trees \\
    --rdf "context/PT/pearltrees_export (8).rdf" \\
    --output .local/data/api_tree_paths.jsonl
        """
    )
    parser.add_argument("--db", type=Path, default=None,
                       help="Path to api_responses.db")
    parser.add_argument("--trees-dir", type=Path, default=None, dest="trees_dir",
                       help="Path to directory with tree JSON files")
    parser.add_argument("--rdf", type=Path, default=None,
                       help="Path to Pearltrees RDF export for RefPearl parent relationships")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--account", type=str, default="s243a",
                       help="Account name to use (default: s243a)")

    args = parser.parse_args()

    if not args.db and not args.trees_dir:
        parser.error("At least one of --db or --trees-dir is required")

    # Load data from all sources
    trees = {}

    if args.db and args.db.exists():
        db_trees = load_api_responses(args.db)
        trees.update(db_trees)
    elif args.db:
        logger.warning(f"Database not found: {args.db}")

    if args.trees_dir:
        file_trees = load_tree_files(args.trees_dir)
        # Merge, preferring file data (more likely to have info.parentTree)
        for tree_id, data in file_trees.items():
            if tree_id not in trees:
                trees[tree_id] = data
            else:
                # If file has info.parentTree and db doesn't, use file
                file_resp = data.get('response', {})
                db_resp = trees[tree_id].get('response', {})
                if file_resp.get('info', {}).get('parentTree') and not db_resp.get('info', {}).get('parentTree'):
                    trees[tree_id] = data

    if not trees:
        logger.error("No trees loaded from any source")
        return 1

    logger.info(f"Total trees loaded: {len(trees)}")

    # Parse RDF for additional parent relationships (from RefPearls)
    rdf_parents = None
    rdf_timestamps = None
    if args.rdf:
        rdf_parents, rdf_timestamps = parse_rdf_parents(args.rdf)

    # Build parent relationships with timestamp-based precedence
    parent_map = build_parent_map(trees, rdf_parents, rdf_timestamps)

    # Show some stats
    depths = defaultdict(int)
    for tree_id in trees:
        path, _ = get_path_to_root(tree_id, trees, parent_map)
        depths[len(path)] += 1

    logger.info("Path depth distribution:")
    for depth in sorted(depths.keys()):
        logger.info(f"  depth {depth}: {depths[depth]} trees")

    # Convert and write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert_to_jsonl(trees, parent_map, args.output, args.account)

    return 0


if __name__ == "__main__":
    exit(main())
