#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# xml_to_prolog_facts.py - Transform XML elements to Prolog facts
#
# Purpose: Convert null-delimited XML chunks to Prolog fact representations.
#          Uses regex-based extraction to handle namespace-less fragments.
#
# Usage:
#   select_xml_elements.awk -v tag="pt:Tree" input.rdf | \
#       xml_to_prolog_facts.py --element-type=tree
#
#   select_xml_elements.awk -v tag="pt:.*Pearl" input.rdf | \
#       xml_to_prolog_facts.py --element-type=pearl
#
# Parameters:
#   --element-type TYPE   Type of element: tree, pearl (required)
#   --delimiter DELIM     Input delimiter (default: \0)
#   --debug               Enable debug output
#
# Element types:
#   tree    - Extract tree(ID, Title, Privacy, LastUpdate)
#   pearl   - Extract pearl(Type, TreeID, ParentID, PosOrder)
#             Also emits parent_tree(TreeID, ParentID)
#
# Examples:
#   # Extract tree facts
#   awk -f select_xml_elements.awk -v tag="pt:Tree" input.rdf | \
#       xml_to_prolog_facts.py --element-type=tree
#
#   # Extract pearl facts
#   awk -f select_xml_elements.awk -v tag="pt:.*Pearl" input.rdf | \
#       xml_to_prolog_facts.py --element-type=pearl

import sys
import re
import argparse

def escape_prolog_string(text):
    """
    Escape a string for use in Prolog.

    Handles single quotes and backslashes.

    Args:
        text: String to escape

    Returns:
        Escaped string safe for Prolog
    """
    if text is None:
        return ''

    # Escape backslashes first
    text = text.replace('\\', '\\\\')
    # Escape single quotes
    text = text.replace("'", "\\'")

    return text

def extract_text_content(xml_chunk, tag_pattern):
    """
    Extract text content from XML tag, handling CDATA.

    Args:
        xml_chunk: XML string
        tag_pattern: Tag name pattern (e.g., 'title', 'privacy')

    Returns:
        Text content or None if not found
    """
    # Try CDATA first: <tag><![CDATA[content]]></tag>
    pattern = f'<[^:]*:?{tag_pattern}[^>]*><!\\[CDATA\\[([^\\]]*?)\\]\\]><'
    match = re.search(pattern, xml_chunk)
    if match:
        return match.group(1).strip()

    # Try regular content: <tag>content</tag>
    pattern = f'<[^:]*:?{tag_pattern}[^>]*>([^<]*)<'
    match = re.search(pattern, xml_chunk)
    if match:
        return match.group(1).strip()

    return None

def extract_attribute(xml_chunk, tag_pattern, attr_name):
    """
    Extract attribute value from XML tag.

    Args:
        xml_chunk: XML string
        tag_pattern: Tag name pattern
        attr_name: Attribute name

    Returns:
        Attribute value or None if not found
    """
    pattern = f'<[^:]*:?{tag_pattern}[^>]+{attr_name}="([^"]*)"'
    match = re.search(pattern, xml_chunk)
    return match.group(1) if match else None

def extract_tree_facts(xml_chunk, debug=False):
    """
    Extract tree facts from XML chunk.

    Emits: tree(ID, Title, Privacy, LastUpdate).

    Args:
        xml_chunk: XML string
        debug: Enable debug output
    """
    # Extract tree ID from about URL
    about = extract_attribute(xml_chunk, r'\w+', 'rdf:about') or extract_attribute(xml_chunk, r'\w+', 'about')
    tree_id_match = re.search(r'id(\d+)', about) if about else None
    tree_id = tree_id_match.group(1) if tree_id_match else 'unknown'

    # Extract title (handle CDATA)
    title = extract_text_content(xml_chunk, 'title') or ''
    title_escaped = escape_prolog_string(title)

    # Extract other fields
    privacy = extract_text_content(xml_chunk, 'privacy') or 'null'
    last_update = extract_text_content(xml_chunk, 'lastUpdate') or 'null'

    # Emit Prolog fact
    if last_update != 'null':
        print(f"tree({tree_id}, '{title_escaped}', {privacy}, '{last_update}').")
    else:
        print(f"tree({tree_id}, '{title_escaped}', {privacy}, null).")

    if debug:
        print(f"% Debug: Extracted tree {tree_id}: {title[:50]}...", file=sys.stderr)

def extract_pearl_facts(xml_chunk, debug=False):
    """
    Extract pearl facts from XML chunk.

    Emits: pearl(Type, TreeID, ParentID, PosOrder).
           parent_tree(TreeID, ParentID).

    Args:
        xml_chunk: XML string
        debug: Enable debug output
    """
    # Extract pearl type from opening tag
    # Look for <pt:XxxPearl or <XxxPearl
    type_match = re.search(r'<[^:]*:?(\w*Pearl)', xml_chunk)
    if type_match:
        pearl_type = type_match.group(1).replace('Pearl', '').lower()
    else:
        pearl_type = 'unknown'

    # Extract tree ID from about URL
    about = extract_attribute(xml_chunk, r'\w*Pearl', 'rdf:about') or extract_attribute(xml_chunk, r'\w*Pearl', 'about')
    tree_id_match = re.search(r'/id(\d+)', about) if about else None
    tree_id = tree_id_match.group(1) if tree_id_match else 'unknown'

    # Extract parent tree ID
    parent_pattern = r'<[^:]*:?parentTree[^>]+resource="[^"]*id(\d+)'
    parent_match = re.search(parent_pattern, xml_chunk)
    parent_id = parent_match.group(1) if parent_match else 'unknown'

    # Extract position order
    pos_order = extract_text_content(xml_chunk, 'posOrder') or 'null'

    # Emit Prolog facts
    print(f"pearl({pearl_type}, {tree_id}, {parent_id}, {pos_order}).")

    # Also emit parent relationship
    if parent_id != 'unknown' and tree_id != 'unknown':
        print(f"parent_tree({tree_id}, {parent_id}).")

    if debug:
        print(f"% Debug: Extracted {pearl_type} pearl {tree_id} → parent {parent_id}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description='Transform XML to Prolog facts',
        epilog='Example: select_xml_elements.awk -v tag="pt:Tree" input.rdf | xml_to_prolog_facts.py --element-type=tree'
    )
    parser.add_argument('--element-type', required=True, choices=['tree', 'pearl'],
                        help='Type of element to extract facts from')
    parser.add_argument('--delimiter', default='\0',
                        help='Input delimiter (default: null character)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()

    if args.debug:
        print(f"# Extracting {args.element_type} facts", file=sys.stderr)

    # Read null-delimited chunks from stdin
    buffer = sys.stdin.read()
    chunks = buffer.split(args.delimiter)

    fact_count = 0

    for chunk in chunks:
        # Skip empty chunks
        chunk = chunk.strip()
        if not chunk:
            continue

        if args.element_type == 'tree':
            extract_tree_facts(chunk, debug=args.debug)
            fact_count += 1
        elif args.element_type == 'pearl':
            extract_pearl_facts(chunk, debug=args.debug)
            fact_count += 1

    if args.debug:
        print(f"# Extracted {fact_count} {args.element_type}(s)", file=sys.stderr)

if __name__ == '__main__':
    main()
