#!/usr/bin/env python3
"""
Import Pearltrees data (trees and pearls) into SQLite database.

Uses the multi-account parser to extract trees and pearls from RDF exports,
then stores them in the multi-account importer database.

Usage:
    python3 scripts/import_pearltrees_to_db.py \
        --account s243a data/s243a.rdf \
        --account s243a_groups data/s243a_groups.rdf \
        --output pearltrees.db
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))
sys.path.insert(0, str(Path(__file__).parent))

from importer import PtMultiAccountImporter
from pearltrees_multi_account_generator import MultiAccountParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_to_db(parser: MultiAccountParser, db_path: Path) -> dict:
    """
    Import parsed Pearltrees data into SQLite database.
    
    Returns stats about imported items.
    """
    importer = PtMultiAccountImporter(str(db_path))
    
    stats = {
        'trees': 0,
        'pearls': 0,
        'links': 0,
        'by_account': {}
    }
    
    # Import trees
    logger.info(f"Importing {len(parser.trees)} trees...")
    for uri, tree in parser.trees.items():
        importer.upsert_object(
            obj_id=tree.tree_id,
            obj_type='tree',
            data={
                'uri': uri,
                'tree_id': tree.tree_id,
                'title': tree.title,
                'parent_uri': tree.parent_uri,
                'section_based_parent_uri': tree.section_based_parent_uri,
                'cross_account_entry_uri': tree.cross_account_entry_uri
            },
            account=tree.account
        )
        stats['trees'] += 1
        stats['by_account'][tree.account] = stats['by_account'].get(tree.account, 0) + 1
        
        # Create parent link
        if tree.parent_uri and tree.parent_uri in parser.trees:
            parent_tree = parser.trees[tree.parent_uri]
            importer.upsert_link(
                source_id=parent_tree.tree_id,
                target_id=tree.tree_id,
                account=tree.account,
                link_type='contains_tree'
            )
            stats['links'] += 1
    
    # Import pearls
    logger.info(f"Importing {len(parser.pearls)} pearls...")
    for pearl in parser.pearls:
        # Extract pearl ID from URI
        pearl_id = pearl.uri.split('/')[-1] if pearl.uri else str(hash(pearl.uri))
        
        importer.upsert_object(
            obj_id=pearl_id,
            obj_type=pearl.type.lower(),  # 'refpearl', 'aliaspearl', 'pagepearl'
            data={
                'uri': pearl.uri,
                'title': pearl.title,
                'parent_tree_uri': pearl.parent_tree_uri,
                'pos_order': pearl.pos_order,
                'see_also_uri': pearl.see_also_uri
            },
            account=pearl.account
        )
        stats['pearls'] += 1
        
        # Link pearl to parent tree
        if pearl.parent_tree_uri and pearl.parent_tree_uri in parser.trees:
            parent_tree = parser.trees[pearl.parent_tree_uri]
            importer.upsert_link(
                source_id=parent_tree.tree_id,
                target_id=pearl_id,
                account=pearl.account,
                link_type='contains_pearl'
            )
            stats['links'] += 1
    
    importer.close()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Import Pearltrees RDF exports into SQLite database."
    )
    parser.add_argument(
        "--account", 
        nargs=2,
        action='append',
        metavar=('NAME', 'FILE'),
        required=True,
        help="Account name and RDF file path (can specify multiple)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("pearltrees.db"),
        help="Output SQLite database path"
    )
    parser.add_argument(
        "--primary",
        type=str,
        help="Primary account name (default: first account)"
    )
    
    args = parser.parse_args()
    
    # Parse accounts
    accounts = [(name, Path(path)) for name, path in args.account]
    primary = args.primary or accounts[0][0]
    
    # Validate files exist
    for name, path in accounts:
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
    
    # Parse RDF files
    multi_parser = MultiAccountParser(primary_account=primary)
    
    for name, path in accounts:
        logger.info(f"Parsing {name} from {path}...")
        multi_parser.add_rdf_file(path, name)
    
    multi_parser.parse_all()
    
    logger.info(f"Parsed {len(multi_parser.trees)} trees, {len(multi_parser.pearls)} pearls")
    
    # Import to database
    logger.info(f"Importing to {args.output}...")
    stats = import_to_db(multi_parser, args.output)
    
    logger.info("Import complete!")
    logger.info(f"  Trees: {stats['trees']}")
    logger.info(f"  Pearls: {stats['pearls']}")
    logger.info(f"  Links: {stats['links']}")
    logger.info(f"  By account: {stats['by_account']}")


if __name__ == "__main__":
    main()
