#!/usr/bin/env python3
"""
Pearltrees Target Generator

This script parses a Pearltrees RDF export and generates "Materialized Path" strings
to be used as targets (Answers) for the Minimal Transformation Projection.

Format Example:
    /2492215/2496226
    - Hacktivism
      - Hacktivism Political engagements
        - The Leaf Item Title
"""

import sys
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

def expand_ns(tag: str) -> str:
    """Expand namespace prefix to full URL."""
    for prefix, url in NAMESPACES.items():
        if tag.startswith(prefix + ":"):
            return tag.replace(prefix + ":", "{" + url + "}")
    return tag

class PearltreesParser:
    def __init__(self, rdf_file: Path):
        self.rdf_file = rdf_file
        self.trees: Dict[str, Dict] = {}  # uri -> {id, title, parent_uri}
        self.pearls: List[Dict] = []      # List of leaf items

    def parse(self):
        """Parse the RDF file and build the tree structure."""
        logger.info(f"Parsing {self.rdf_file}...")
        try:
            tree = ET.parse(self.rdf_file)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return

        # 1. First pass: Identify all Trees
        # Look for <pt:Tree>
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}Tree"):
            uri = elem.get(f"{{{NAMESPACES['rdf']}}}about")
            if not uri:
                continue
            
            # Extract ID
            tree_id_elem = elem.find(f"{{{NAMESPACES['pt']}}}treeId")
            tree_id = tree_id_elem.text if tree_id_elem is not None else "unknown"
            
            # Extract Title
            title_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None else "Untitled"
            
            self.trees[uri] = {
                "id": tree_id,
                "title": title,
                "parent_uri": None,  # To be filled
                "uri": uri
            }

        logger.info(f"Found {len(self.trees)} trees.")

        # 2. Second pass: Build Hierarchy (find RefPearls)
        # RefPearls link a parent tree to a child tree (via rdfs:seeAlso)
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}RefPearl"):
            
            # Parent Tree
            parent_res = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            if parent_res is None:
                continue
            parent_uri = parent_res.get(f"{{{NAMESPACES['rdf']}}}resource")
            
            # Child Tree (Referenced)
            see_also = elem.find(f"{{{NAMESPACES['rdfs']}}}seeAlso")
            if see_also is None:
                continue
            child_uri = see_also.get(f"{{{NAMESPACES['rdf']}}}resource")

            # Link them
            if child_uri in self.trees and parent_uri in self.trees:
                self.trees[child_uri]["parent_uri"] = parent_uri

        # 3. Third pass: Collect Leaf Pearls (PagePearl, NotePearl, etc.)
        # We process anything that has a parentTree and a Title, but is not a Tree or RefPearl (refpearls are structure)
        # Or maybe we want RefPearls too if they represent the category itself?
        # Minimally, let's catch PagePearl, AliasPearl, etc.
        
        # We'll just iterate over all children of root and check types
        for child in root:
            tag = child.tag
            # Skip Trees, Persons, UserAccounts
            if tag in [f"{{{NAMESPACES['pt']}}}Tree", f"{{{NAMESPACES['foaf']}}}Person",
                       f"{{{NAMESPACES['sioc']}}}UserAccount", f"{{{NAMESPACES['pt']}}}RefPearl",
                        f"{{{NAMESPACES['pt']}}}SectionPearl"]: # SectionPearl is usually just a separator?
                continue

            # Check if it's a pearl
            if not tag.startswith(f"{{{NAMESPACES['pt']}}}") or "Pearl" not in tag:
                continue

            uri = child.get(f"{{{NAMESPACES['rdf']}}}about")
            
            # Get Parent Tree
            parent_res = child.find(f"{{{NAMESPACES['pt']}}}parentTree")
            if parent_res is None:
                continue
            parent_uri = parent_res.get(f"{{{NAMESPACES['rdf']}}}resource")
            
            # Get Title
            title_elem = child.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None else "Untitled"

            self.pearls.append({
                "type": tag.split("}")[-1], # e.g. PagePearl
                "uri": uri,
                "title": title,
                "parent_uri": parent_uri
            })

        logger.info(f"Found {len(self.pearls)} leaf pearls.")

    def get_path(self, parent_uri: str) -> Tuple[List[str], List[str]]:
        """
        Reconstruct the path of IDs and Titles back to root.
        Returns ([id_root, ..., id_parent], [title_root, ..., title_parent])
        """
        ids = []
        titles = []
        curr = parent_uri
        
        # Prevent infinite loops with a set
        visited = set()

        while curr and curr in self.trees:
            if curr in visited:
                break
            visited.add(curr)
            
            node = self.trees[curr]
            ids.insert(0, node["id"])
            titles.insert(0, node["title"])
            
            curr = node["parent_uri"]
            
        return ids, titles

    def generate_targets(self, query_template: str = "{title}", filter_path: Optional[str] = None):
        """Generate the formatted target strings."""
        results = []
        
        for p in self.pearls:
            # Get ancestor path
            path_ids, path_titles = self.get_path(p["parent_uri"])
            
            # Filter logic: Check if filter_path string is in any of the ancestor titles or the item title itself
            if filter_path:
                full_path_titles = path_titles + [p["title"]]
                # Case-insensitive check
                if not any(filter_path.lower() in t.lower() for t in full_path_titles):
                    continue
            
            # Construct formatted string
            # 1. ID Path
            id_path_str = "/" + "/".join(path_ids)
            
            # 2. Hierarchical Titles
            title_lines = []
            indent_level = 0
            
            # Ancestors
            for t in path_titles:
                indent = "  " * indent_level
                title_lines.append(f"{indent}- {t}")
                indent_level += 1
            
            # The item itself
            indent = "  " * indent_level
            title_lines.append(f"{indent}- {p['title']}")
            
            formatted_text = f"{id_path_str}\n" + "\n".join(title_lines)
            
            # Apply template
            try:
                query_text = query_template.format(title=p["title"])
            except KeyError:
                # Fallback if template is invalid
                query_text = p["title"]
            
            results.append({
                "type": p["type"],
                "target_text": formatted_text,
                "raw_title": p["title"],
                "query": query_text
            })
            
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Pearltrees Q/A pairs.")
    parser.add_argument("input_rdf", type=Path, help="Input RDF file")
    parser.add_argument("output_jsonl", type=Path, help="Output JSONL file")
    parser.add_argument("--query-template", type=str, default="{title}", 
                       help="Template for the query. Use {title} as placeholder. E.g. 'locate({title})'")
    parser.add_argument("--filter-path", type=str, default=None,
                       help="Only include items that have this string in their ancestor path titles (e.g. 'Physics')")
    
    args = parser.parse_args()
    
    parser_obj = PearltreesParser(args.input_rdf)
    parser_obj.parse()
    
    # Update generate_targets to use template and filter
    targets = parser_obj.generate_targets(args.query_template, args.filter_path)
    
    logger.info(f"Writing {len(targets)} records to {args.output_jsonl}...")
    with open(args.output_jsonl, 'w') as f:
        for t in targets:
            f.write(json.dumps(t) + "\n")
            
    # Print a preview
    if targets:
        print("\n--- Preview of first item ---")
        print(f"Query: {targets[0]['query']}")
        print("Target:")
        print(targets[0]['target_text'])
        print("-----------------------------\n")

if __name__ == "__main__":
    main()
