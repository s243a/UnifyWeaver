#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# filter_by_parent_tree.py - Filter XML elements by parent tree ID
#
# Purpose: Filter null-delimited XML chunks by parent tree reference.
#          Reads from stdin, outputs matching chunks to stdout.
#
# Usage:
#   select_xml_elements.awk -v tag="pt:.*Pearl" input.rdf | \
#       filter_by_parent_tree.py --tree-id=2492215
#
#   filter_by_parent_tree.py --tree-id=2492215 < pearls.xml0
#
# Parameters:
#   --tree-id ID        Parent tree ID to filter by (required)
#   --delimiter DELIM   Input/output delimiter (default: \0)
#   --debug             Enable debug output
#
# Examples:
#   # Extract pearls for specific tree
#   awk -f select_xml_elements.awk -v tag="pt:.*Pearl" pearltrees.rdf | \
#       filter_by_parent_tree.py --tree-id=2492215
#
#   # Extract with custom delimiter
#   awk -f select_xml_elements.awk -v tag="pt:.*Pearl" -v delimiter="\n---\n" input.rdf | \
#       filter_by_parent_tree.py --tree-id=123 --delimiter="\n---\n"

import sys
import re
import argparse
import xml.etree.ElementTree as ET

def extract_parent_tree_id(xml_chunk):
    """
    Extract parent tree ID from XML chunk.

    Looks for pt:parentTree element with rdf:resource attribute
    containing an ID pattern like "id12345".

    Uses regex-based extraction since namespace-less XML fragments
    can't be parsed by ElementTree without namespace declarations.

    Args:
        xml_chunk: XML string to parse

    Returns:
        Parent tree ID as string, or None if not found
    """
    # Method 1: Regex-based extraction (works with namespace-less fragments)
    # Look for pattern: <pt:parentTree rdf:resource="...id12345..."
    pattern = r'<[^:]*:?parentTree[^>]+resource="[^"]*id(\d+)'
    match = re.search(pattern, xml_chunk)

    if match:
        return match.group(1)

    # Method 2: Try XML parsing (if chunk has namespace declarations)
    try:
        root = ET.fromstring(xml_chunk)

        # Look for pt:parentTree element
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if 'parentTree' in tag_name:
                # Try different attribute formats
                resource = (
                    elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource') or
                    elem.get('rdf:resource') or
                    elem.get('resource')
                )

                if resource:
                    # Extract ID from URL like "https://.../id2492215"
                    match = re.search(r'id(\d+)', resource)
                    if match:
                        return match.group(1)

    except ET.ParseError:
        # Expected for namespace-less fragments, already tried regex
        pass

    return None

def main():
    parser = argparse.ArgumentParser(
        description='Filter XML elements by parent tree ID',
        epilog='Example: select_xml_elements.awk -v tag="pt:.*Pearl" input.rdf | filter_by_parent_tree.py --tree-id=2492215'
    )
    parser.add_argument('--tree-id', required=True,
                        help='Parent tree ID to filter by')
    parser.add_argument('--delimiter', default='\0',
                        help='Input/output delimiter (default: null character)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()

    if args.debug:
        print(f"# Filtering by parent tree ID: {args.tree_id}", file=sys.stderr)
        delimiter_str = '\\0 (null)' if args.delimiter == chr(0) else repr(args.delimiter)
        print(f"# Delimiter: {delimiter_str}", file=sys.stderr)

    # Read null-delimited chunks from stdin
    buffer = sys.stdin.read()
    chunks = buffer.split(args.delimiter)

    matched_count = 0
    total_count = 0

    for chunk in chunks:
        # Skip empty chunks
        chunk = chunk.strip()
        if not chunk:
            continue

        total_count += 1

        # Extract parent tree ID from this chunk
        parent_id = extract_parent_tree_id(chunk)

        if args.debug and parent_id:
            print(f"# Chunk {total_count}: parent_id={parent_id}", file=sys.stderr)

        # Filter: output only if parent matches
        if parent_id == args.tree_id:
            print(chunk, end=args.delimiter, flush=True)
            matched_count += 1

    if args.debug:
        print(f"# Matched {matched_count}/{total_count} chunks", file=sys.stderr)

if __name__ == '__main__':
    main()
