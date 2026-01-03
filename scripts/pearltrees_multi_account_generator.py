#!/usr/bin/env python3
"""
Multi-Account Pearltrees Target Generator

Parses multiple Pearltrees RDF exports (e.g., s243a and s243a_groups) and builds
unified materialized paths that can span accounts.

Key features:
- Cross-account linking via AliasPearls
- Section-based relationship detection (Subcategories, Navigate Up, Super Categories)
- Account boundary notation: @account_name appended when crossing accounts
"""

import sys
import xml.etree.ElementTree as ET
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# XML Namespaces in Pearltrees RDF
NAMESPACES = {
    'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    'rdfs': "http://www.w3.org/2000/01/rdf-schema#",
    'dcterms': "http://purl.org/dc/elements/1.1/",
    'pt': "http://www.pearltrees.com/rdf/0.1/#",
    'foaf': "http://xmlns.com/foaf/0.1/",
    'sioc': "http://rdfs.org/sioc/ns#",
}

# Section type patterns
CHILD_SECTION_PATTERNS = [
    r'subcategor',
    r'subtopic',
]

PARENT_SECTION_PATTERNS = [
    r'super.categor',
    r'navigate.up',
]


@dataclass
class SectionPearl:
    """Represents a section divider in a tree."""
    uri: str
    title: str
    description: str
    pos_order: int
    parent_tree_uri: str
    section_type: str = "unknown"  # "child", "parent", or "unknown"


@dataclass
class TreeNode:
    """Represents a tree/folder."""
    uri: str
    tree_id: str
    title: str
    account: str  # e.g., "s243a" or "s243a_groups"
    parent_uri: Optional[str] = None
    section_based_parent_uri: Optional[str] = None  # From "Navigate Up" sections
    cross_account_entry_uri: Optional[str] = None  # Where this tree is linked from another account


@dataclass
class Pearl:
    """Represents a pearl (link item)."""
    uri: str
    type: str  # PagePearl, AliasPearl, RefPearl, etc.
    title: str
    parent_tree_uri: str
    pos_order: int
    see_also_uri: Optional[str] = None  # For RefPearl/AliasPearl - points to another tree
    account: str = ""


class MultiAccountParser:
    """Parser for multiple Pearltrees RDF exports."""

    def __init__(self, primary_account: str = "s243a"):
        self.primary_account = primary_account
        self.trees: Dict[str, TreeNode] = {}  # uri -> TreeNode
        self.pearls: List[Pearl] = []
        self.sections: Dict[str, List[SectionPearl]] = {}  # parent_tree_uri -> list of sections
        self.account_files: Dict[str, Path] = {}  # account_name -> file_path

    def add_rdf_file(self, rdf_file: Path, account_name: str):
        """Add an RDF file for a specific account."""
        self.account_files[account_name] = rdf_file
        logger.info(f"Added account '{account_name}' from {rdf_file}")

    def _classify_section(self, description: str) -> str:
        """Classify a section as child, parent, or unknown based on its description."""
        desc_lower = description.lower()
        
        for pattern in CHILD_SECTION_PATTERNS:
            if re.search(pattern, desc_lower):
                return "child"
        
        for pattern in PARENT_SECTION_PATTERNS:
            if re.search(pattern, desc_lower):
                return "parent"
        
        return "unknown"

    def _extract_account_from_creator(self, creator_uri: str) -> str:
        """Extract account name from creator URI like https://www.pearltrees.com/s243a#sioc"""
        if not creator_uri:
            return "unknown"
        match = re.search(r'pearltrees\.com/([^/#]+)', creator_uri)
        return match.group(1) if match else "unknown"

    def _parse_single_file(self, rdf_file: Path, account_name: str):
        """Parse a single RDF file."""
        logger.info(f"Parsing {rdf_file} for account '{account_name}'...")
        
        try:
            tree = ET.parse(rdf_file)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return

        # 1. Parse Trees
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}Tree"):
            uri = elem.get(f"{{{NAMESPACES['rdf']}}}about")
            if not uri:
                continue
            
            tree_id_elem = elem.find(f"{{{NAMESPACES['pt']}}}treeId")
            tree_id = tree_id_elem.text if tree_id_elem is not None else "unknown"
            
            title_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None else "Untitled"
            
            creator_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}creator")
            if creator_elem is not None:
                creator_uri = creator_elem.get(f"{{{NAMESPACES['rdf']}}}resource", "")
                detected_account = self._extract_account_from_creator(creator_uri)
            else:
                detected_account = account_name
            
            self.trees[uri] = TreeNode(
                uri=uri,
                tree_id=tree_id,
                title=title,
                account=detected_account
            )

        # 2. Parse SectionPearls
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}SectionPearl"):
            uri = elem.get(f"{{{NAMESPACES['rdf']}}}about")
            if not uri:
                continue
            
            title_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None else ""
            
            desc_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}description")
            description = desc_elem.text if desc_elem is not None else ""
            
            pos_elem = elem.find(f"{{{NAMESPACES['pt']}}}posOrder")
            pos_order = int(pos_elem.text) if pos_elem is not None else 0
            
            parent_elem = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            parent_uri = parent_elem.get(f"{{{NAMESPACES['rdf']}}}resource") if parent_elem is not None else None
            
            if parent_uri:
                section = SectionPearl(
                    uri=uri,
                    title=title,
                    description=description,
                    pos_order=pos_order,
                    parent_tree_uri=parent_uri,
                    section_type=self._classify_section(description)
                )
                
                if parent_uri not in self.sections:
                    self.sections[parent_uri] = []
                self.sections[parent_uri].append(section)

        # 3. Parse RefPearls (links to subtrees)
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}RefPearl"):
            uri = elem.get(f"{{{NAMESPACES['rdf']}}}about")
            if not uri:
                continue
            
            parent_res = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            if parent_res is None:
                continue
            parent_uri = parent_res.get(f"{{{NAMESPACES['rdf']}}}resource")
            
            see_also = elem.find(f"{{{NAMESPACES['rdfs']}}}seeAlso")
            child_uri = see_also.get(f"{{{NAMESPACES['rdf']}}}resource") if see_also is not None else None
            
            pos_elem = elem.find(f"{{{NAMESPACES['pt']}}}posOrder")
            pos_order = int(pos_elem.text) if pos_elem is not None else 0
            
            title_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None else ""
            
            # Link child tree to parent
            if child_uri and child_uri in self.trees and parent_uri in self.trees:
                # Determine if this is in a "parent" section (Navigate Up, Super Categories)
                section_type = self._get_section_type_for_pos(parent_uri, pos_order)
                
                if section_type == "parent":
                    # This RefPearl is in a "Navigate Up" section - it's a parent link
                    self.trees[parent_uri].section_based_parent_uri = child_uri
                else:
                    # Normal child relationship
                    if self.trees[child_uri].parent_uri is None:
                        self.trees[child_uri].parent_uri = parent_uri
            
            self.pearls.append(Pearl(
                uri=uri,
                type="RefPearl",
                title=title,
                parent_tree_uri=parent_uri,
                pos_order=pos_order,
                see_also_uri=child_uri,
                account=account_name
            ))

        # 4. Parse AliasPearls (cross-account links)
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}AliasPearl"):
            uri = elem.get(f"{{{NAMESPACES['rdf']}}}about")
            if not uri:
                continue
            
            parent_res = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            if parent_res is None:
                continue
            parent_uri = parent_res.get(f"{{{NAMESPACES['rdf']}}}resource")
            
            see_also = elem.find(f"{{{NAMESPACES['rdfs']}}}seeAlso")
            target_uri = see_also.get(f"{{{NAMESPACES['rdf']}}}resource") if see_also is not None else None
            
            pos_elem = elem.find(f"{{{NAMESPACES['pt']}}}posOrder")
            pos_order = int(pos_elem.text) if pos_elem is not None else 0
            
            title_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None else ""
            
            # Track cross-account links: if parent is in s243a and target is in s243a_groups
            # we need to remember where the cross-account jump happens
            if target_uri and target_uri in self.trees and parent_uri and parent_uri in self.trees:
                parent_node = self.trees[parent_uri]
                target_node = self.trees[target_uri]
                
                # Cross-account link detected
                if parent_node.account != target_node.account:
                    # Store the linking point: target's "cross_account_entry" points to parent in primary account
                    if target_node.cross_account_entry_uri is None:
                        target_node.cross_account_entry_uri = parent_uri
            
            self.pearls.append(Pearl(
                uri=uri,
                type="AliasPearl",
                title=title,
                parent_tree_uri=parent_uri,
                pos_order=pos_order,
                see_also_uri=target_uri,
                account=account_name
            ))

        # 5. Parse PagePearls (actual bookmarks)
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}PagePearl"):
            uri = elem.get(f"{{{NAMESPACES['rdf']}}}about")
            if not uri:
                continue
            
            parent_res = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            # Items might not have a parent tree if they are in the root or detached
            parent_uri = parent_res.get(f"{{{NAMESPACES['rdf']}}}resource") if parent_res is not None else None
            
            pos_elem = elem.find(f"{{{NAMESPACES['pt']}}}posOrder")
            pos_order = int(pos_elem.text) if pos_elem is not None else 0
            
            title_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None else ""
            
            # Extract URL if available? RDF usually puts it in text or resource
            # But Pearl class only has uri (ID) and title. For now this is enough for the dataset.
            
            self.pearls.append(Pearl(
                uri=uri,
                type="PagePearl",
                title=title,
                parent_tree_uri=parent_uri,
                pos_order=pos_order,
                account=account_name
            ))

    def _get_section_type_for_pos(self, tree_uri: str, pos_order: int) -> str:
        """Determine which section a posOrder falls into for a given tree."""
        if tree_uri not in self.sections:
            return "unknown"
        
        sections = sorted(self.sections[tree_uri], key=lambda s: s.pos_order)
        
        # Find the section that contains this pos_order
        current_section_type = "unknown"  # Default for items before first section
        
        for section in sections:
            if pos_order >= section.pos_order:
                current_section_type = section.section_type
            else:
                break
        
        return current_section_type

    def parse_all(self, repair_missing: bool = True):
        """Parse all registered RDF files in two passes, optionally repairing missing trees.

        Args:
            repair_missing: If True, synthesize missing trees from RefPearl/AliasPearl references.
                           This works around a Pearltrees export bug where referenced trees
                           are sometimes not included in the export.
        """
        # Pass 1: Parse all Trees, Sections, RefPearls from all files
        for account_name, rdf_file in self.account_files.items():
            self._parse_single_file(rdf_file, account_name)

        logger.info(f"Pass 1 complete: {len(self.trees)} trees, {len(self.pearls)} pearls")

        # Pass 2: Re-scan AliasPearls to set cross-account links now that all trees are known
        for account_name, rdf_file in self.account_files.items():
            self._parse_alias_pearls_for_cross_account(rdf_file, account_name)

        # Pass 3: Repair missing trees (optional, enabled by default)
        if repair_missing:
            self.repair_missing_trees()

        logger.info(f"Total trees: {len(self.trees)}")
        logger.info(f"Total pearls: {len(self.pearls)}")
        logger.info(f"Total sections: {sum(len(s) for s in self.sections.values())}")

    def _parse_alias_pearls_for_cross_account(self, rdf_file: Path, account_name: str):
        """Second pass: set cross-account entry points on trees. Prefers RefPearls over AliasPearls."""
        try:
            tree = ET.parse(rdf_file)
            root = tree.getroot()
        except ET.ParseError:
            return
        
        # First, process RefPearls (these are actual subtree relationships - preferred)
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}RefPearl"):
            parent_res = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            if parent_res is None:
                continue
            parent_uri = parent_res.get(f"{{{NAMESPACES['rdf']}}}resource")
            
            see_also = elem.find(f"{{{NAMESPACES['rdfs']}}}seeAlso")
            target_uri = see_also.get(f"{{{NAMESPACES['rdf']}}}resource") if see_also is not None else None
            
            if target_uri and target_uri in self.trees and parent_uri and parent_uri in self.trees:
                parent_node = self.trees[parent_uri]
                target_node = self.trees[target_uri]
                
                # Cross-account link: parent in primary, target in secondary
                if parent_node.account == self.primary_account and target_node.account != self.primary_account:
                    # RefPearl always wins - overwrite any existing AliasPearl link
                    target_node.cross_account_entry_uri = parent_uri
                    logger.debug(f"Cross-account RefPearl: {parent_node.title} -> {target_node.title}")
        
        # Then, process AliasPearls (only if no RefPearl link exists)
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}AliasPearl"):
            parent_res = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            if parent_res is None:
                continue
            parent_uri = parent_res.get(f"{{{NAMESPACES['rdf']}}}resource")
            
            see_also = elem.find(f"{{{NAMESPACES['rdfs']}}}seeAlso")
            target_uri = see_also.get(f"{{{NAMESPACES['rdf']}}}resource") if see_also is not None else None
            
            if target_uri and target_uri in self.trees and parent_uri and parent_uri in self.trees:
                parent_node = self.trees[parent_uri]
                target_node = self.trees[target_uri]
                
                # Cross-account link: parent in primary, target in secondary
                # Only set if no RefPearl link exists (RefPearl takes priority)
                if parent_node.account == self.primary_account and target_node.account != self.primary_account:
                    if target_node.cross_account_entry_uri is None:
                        # Skip "drop zone" trees - they only contain temporary/misplaced items
                        if self._is_drop_zone(parent_node):
                            logger.debug(f"Skipping drop zone as entry point: {parent_node.title}")
                            continue
                        target_node.cross_account_entry_uri = parent_uri
                        logger.debug(f"Cross-account AliasPearl: {parent_node.title} -> {target_node.title}")

    def _is_drop_zone(self, tree_node: TreeNode) -> bool:
        """Check if a tree is a 'drop zone' (temporary holding area)."""
        title_lower = tree_node.title.lower()
        return "drop zone" in title_lower or "dropzone" in title_lower

    def repair_missing_trees(self) -> int:
        """
        Synthesize missing trees from RefPearl/AliasPearl references.

        Pearltrees export sometimes omits tree definitions but includes
        RefPearl/AliasPearl references to them. This method creates
        synthetic TreeNode entries for those missing trees.

        Key distinction:
        - RefPearl: References a tree INSIDE the parent tree (subfolder).
          The parent_tree_uri IS the actual parent of the missing tree.
        - AliasPearl: References a tree OUTSIDE the parent tree (cross-reference).
          The parent_tree_uri is NOT the actual parent, just where the alias appears.

        Returns:
            Number of trees synthesized.
        """
        # Collect all see_also_uri targets that don't have tree entries
        # Track separately by pearl type for correct parent assignment
        missing_refs: Dict[str, Pearl] = {}  # uri -> first pearl referencing it

        for pearl in self.pearls:
            if pearl.see_also_uri and pearl.see_also_uri not in self.trees:
                if pearl.see_also_uri not in missing_refs:
                    missing_refs[pearl.see_also_uri] = pearl
                elif pearl.type == "RefPearl" and missing_refs[pearl.see_also_uri].type != "RefPearl":
                    # Prefer RefPearl over AliasPearl for parent determination
                    missing_refs[pearl.see_also_uri] = pearl

        if not missing_refs:
            logger.info("No missing trees to repair")
            return 0

        logger.info(f"Found {len(missing_refs)} missing tree references, synthesizing...")

        synthesized = 0
        synthesized_from_ref = 0
        synthesized_from_alias = 0

        for target_uri, pearl in missing_refs.items():
            # Extract tree_id from URI (e.g., ".../fields-of-geometry/id53492143" -> "53492143")
            tree_id_match = re.search(r'/id(\d+)(?:\?|#|$)', target_uri)
            tree_id = tree_id_match.group(1) if tree_id_match else target_uri.split('/')[-1]

            # Extract account from URI (e.g., "pearltrees.com/s243a/..." -> "s243a")
            account_match = re.search(r'pearltrees\.com/([^/]+)/', target_uri)
            account = account_match.group(1) if account_match else self.primary_account

            # Determine parent_uri based on pearl type
            # RefPearl: parent_tree_uri is the actual parent (subfolder relationship)
            # AliasPearl: parent_tree_uri is just where alias appears, not actual parent
            if pearl.type == "RefPearl":
                parent_uri = pearl.parent_tree_uri if pearl.parent_tree_uri in self.trees else None
                synthesized_from_ref += 1
                logger.debug(f"Synthesized RefPearl->tree: {pearl.title} (parent: {self.trees[pearl.parent_tree_uri].title if parent_uri else 'unknown'})")
            else:
                # For AliasPearl, we don't know the actual parent
                # The tree exists somewhere else in the hierarchy
                parent_uri = None
                synthesized_from_alias += 1
                logger.debug(f"Synthesized AliasPearl->tree: {pearl.title} (parent: unknown - alias from {self.trees.get(pearl.parent_tree_uri, TreeNode('','','unknown','')).title})")

            # Create synthetic tree node
            synthetic_tree = TreeNode(
                uri=target_uri,
                tree_id=tree_id,
                title=pearl.title,
                account=account,
                parent_uri=parent_uri,
            )

            self.trees[target_uri] = synthetic_tree
            synthesized += 1

        logger.info(f"Synthesized {synthesized} missing trees: {synthesized_from_ref} from RefPearl (with parent), {synthesized_from_alias} from AliasPearl (orphaned)")
        return synthesized

    def get_path(self, tree_uri: str, visited: Optional[Set[str]] = None) -> Tuple[List[str], List[str], List[str]]:
        """
        Reconstruct the path of IDs, Titles, and Accounts back to root.
        For cross-account trees, prepends the path from primary account to the linking point.
        Returns ([id_root, ..., id_current], [title_root, ..., title_current], [account_root, ..., account_current])
        """
        if visited is None:
            visited = set()
        
        ids = []
        titles = []
        accounts = []
        
        curr = tree_uri
        cross_account_entry = None
        
        # First, trace path within the tree's own account hierarchy
        while curr and curr in self.trees:
            if curr in visited:
                break
            visited.add(curr)
            
            node = self.trees[curr]
            ids.insert(0, node.tree_id)
            titles.insert(0, node.title)
            accounts.insert(0, node.account)
            
            # Check if this node has a cross-account entry point
            if node.cross_account_entry_uri and node.account != self.primary_account:
                cross_account_entry = node.cross_account_entry_uri
            
            # Try primary parent first, fallback to section-based parent
            curr = node.parent_uri or node.section_based_parent_uri
        
        # If we found a cross-account entry, prepend the path from primary account
        if cross_account_entry and cross_account_entry in self.trees:
            primary_ids, primary_titles, primary_accounts = self.get_path(cross_account_entry, visited.copy())
            
            # Prepend primary account path before the cross-account path
            ids = primary_ids + ids
            titles = primary_titles + titles
            accounts = primary_accounts + accounts
        
        return ids, titles, accounts

    def generate_tree_targets(self, query_template: str = "{title}", filter_path: Optional[str] = None):
        """Generate target strings for trees with cross-account notation."""
        results = []
        
        for uri, tree_data in self.trees.items():
            # Get full path
            path_ids, path_titles, path_accounts = self.get_path(uri)
            
            # Filter logic
            if filter_path:
                if not any(filter_path.lower() in t.lower() for t in path_titles):
                    continue
            
            # Build ID path with account boundary markers
            id_parts = []
            prev_account = self.primary_account
            
            for i, (tree_id, account) in enumerate(zip(path_ids, path_accounts)):
                if account != prev_account:
                    # Account boundary crossed
                    id_parts.append(f"{tree_id}@{account}")
                else:
                    id_parts.append(tree_id)
                prev_account = account
            
            id_path_str = "/" + "/".join(id_parts)
            
            # Build hierarchical title list with account markers
            title_lines = []
            prev_account = self.primary_account
            indent_level = 0
            
            for title, account in zip(path_titles, path_accounts):
                indent = "  " * indent_level
                if account != prev_account:
                    title_lines.append(f"{indent}- {title} @{account}")
                else:
                    title_lines.append(f"{indent}- {title}")
                indent_level += 1
                prev_account = account
            
            formatted_text = f"{id_path_str}\n" + "\n".join(title_lines)
            
            # Apply query template
            try:
                query_text = query_template.format(title=tree_data.title)
            except KeyError:
                query_text = tree_data.title
            
            # Cluster by parent
            cluster_id = tree_data.parent_uri or tree_data.section_based_parent_uri or uri
            
            results.append({
                "type": "Tree",
                "target_text": formatted_text,
                "raw_title": tree_data.title,
                "query": query_text,
                "cluster_id": cluster_id,
                "tree_id": tree_data.tree_id,
                "account": tree_data.account,
                "uri": uri
            })
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate multi-account Pearltrees Q/A pairs.")
    parser.add_argument("--primary", type=str, default=None,
                       help="Primary account name (default: first --account)")
    parser.add_argument("--account", action="append", nargs=2, metavar=("NAME", "FILE"),
                       help="Add account: --account s243a file1.rdf --account s243a_groups file2.rdf")
    parser.add_argument("output_jsonl", type=Path, help="Output JSONL file")
    parser.add_argument("--query-template", type=str, default="{title}",
                       help="Template for the query. Use {title} as placeholder.")
    parser.add_argument("--filter-path", type=str, default=None,
                       help="Only include items with this string in their path")
    parser.add_argument("--item-type", type=str, default="tree", choices=["tree"],
                       help="Type of items to generate (currently only 'tree' supported)")
    
    args = parser.parse_args()
    
    if not args.account:
        logger.error("At least one --account must be specified")
        sys.exit(1)
    
    # Auto-detect primary account from first --account if not specified
    primary_account = args.primary if args.primary else args.account[0][0]
    logger.info(f"Primary account: {primary_account}")
    
    mp = MultiAccountParser(primary_account=primary_account)
    
    for name, filepath in args.account:
        mp.add_rdf_file(Path(filepath), name)
    
    mp.parse_all()
    
    targets = mp.generate_tree_targets(args.query_template, args.filter_path)
    logger.info(f"Generated {len(targets)} targets.")
    
    logger.info(f"Writing to {args.output_jsonl}...")
    with open(args.output_jsonl, 'w') as f:
        for t in targets:
            f.write(json.dumps(t) + "\n")
    
    # Preview
    if targets:
        print("\n--- Preview of first item ---")
        print(f"Query: {targets[0]['query']}")
        print(f"Account: {targets[0]['account']}")
        print("Target:")
        print(targets[0]['target_text'])
        print("-----------------------------\n")


if __name__ == "__main__":
    main()
