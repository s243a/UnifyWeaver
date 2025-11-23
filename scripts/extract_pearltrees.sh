#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# extract_pearltrees.sh - Complete Pearltrees RDF extraction pipeline
#
# Purpose: Extract tree and pearl data from Pearltrees RDF exports
#          and convert to Prolog facts for graph analysis.
#
# Usage:
#   extract_pearltrees.sh input.rdf output_dir
#   extract_pearltrees.sh input.rdf output_dir --tree-id=12345
#
# Parameters:
#   input.rdf      Pearltrees RDF export file
#   output_dir     Directory for output facts
#   --tree-id=ID   Optional: Extract only pearls for specific tree
#
# Outputs:
#   output_dir/trees.pl         - Tree facts
#   output_dir/pearls.pl        - Pearl facts
#   output_dir/all_facts.pl     - Combined facts
#
# Examples:
#   # Extract all trees and pearls
#   ./extract_pearltrees.sh pearltrees.rdf facts/
#
#   # Extract pearls for specific tree only
#   ./extract_pearltrees.sh pearltrees.rdf facts/ --tree-id=2492215

set -e  # Exit on error

# Color output for readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
error() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Parse arguments
INPUT_FILE=""
OUTPUT_DIR=""
TREE_ID=""

for arg in "$@"; do
    case $arg in
        --tree-id=*)
            TREE_ID="${arg#*=}"
            ;;
        --help|-h)
            cat << EOF
Pearltrees RDF Extraction Pipeline

Usage: $0 input.rdf output_dir [--tree-id=ID]

Extracts tree and pearl data from Pearltrees RDF exports and converts
to Prolog facts for graph analysis.

Parameters:
  input.rdf      Pearltrees RDF export file
  output_dir     Directory for output facts
  --tree-id=ID   Optional: Extract only pearls for specific tree

Examples:
  # Extract all trees and pearls
  $0 pearltrees.rdf facts/

  # Extract pearls for specific tree only
  $0 pearltrees.rdf facts/ --tree-id=2492215

Outputs:
  output_dir/trees.pl       - Tree facts
  output_dir/pearls.pl      - Pearl facts
  output_dir/all_facts.pl   - Combined facts

Pipeline:
  1. Select XML elements (awk)
  2. Filter by criteria (optional Python)
  3. Transform to Prolog facts (Python)
EOF
            exit 0
            ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$arg"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$arg"
            else
                error "Unknown argument: $arg"
            fi
            ;;
    esac
done

# Validate arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    error "Usage: $0 input.rdf output_dir [--tree-id=ID]"
fi

if [ ! -f "$INPUT_FILE" ]; then
    error "Input file not found: $INPUT_FILE"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get script directory to find utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$SCRIPT_DIR/utils"

# Validate utilities exist
if [ ! -f "$UTILS_DIR/select_xml_elements.awk" ]; then
    error "Utility not found: $UTILS_DIR/select_xml_elements.awk"
fi

info "Extracting from: $INPUT_FILE"
info "Output directory: $OUTPUT_DIR"

# ============================================
# Step 1: Extract Trees
# ============================================

info "Extracting tree facts..."

awk -f "$UTILS_DIR/select_xml_elements.awk" \
    -v tag="pt:Tree" \
    "$INPUT_FILE" | \
python3 "$UTILS_DIR/xml_to_prolog_facts.py" \
    --element-type=tree \
    > "$OUTPUT_DIR/trees.pl"

TREE_COUNT=$(grep -c "^tree(" "$OUTPUT_DIR/trees.pl" || echo "0")
success "Extracted $TREE_COUNT tree(s) → trees.pl"

# ============================================
# Step 2: Extract Pearls
# ============================================

info "Extracting pearl facts..."

if [ -n "$TREE_ID" ]; then
    # Filter pearls by specific tree ID
    info "Filtering pearls for tree ID: $TREE_ID"

    awk -f "$UTILS_DIR/select_xml_elements.awk" \
        -v tag="pt:.*Pearl" \
        "$INPUT_FILE" | \
    python3 "$UTILS_DIR/filter_by_parent_tree.py" \
        --tree-id="$TREE_ID" | \
    python3 "$UTILS_DIR/xml_to_prolog_facts.py" \
        --element-type=pearl \
        > "$OUTPUT_DIR/pearls.pl"
else
    # Extract all pearls
    awk -f "$UTILS_DIR/select_xml_elements.awk" \
        -v tag="pt:.*Pearl" \
        "$INPUT_FILE" | \
    python3 "$UTILS_DIR/xml_to_prolog_facts.py" \
        --element-type=pearl \
        > "$OUTPUT_DIR/pearls.pl"
fi

PEARL_COUNT=$(grep -c "^pearl(" "$OUTPUT_DIR/pearls.pl" || echo "0")
success "Extracted $PEARL_COUNT pearl(s) → pearls.pl"

# ============================================
# Step 3: Combine Facts
# ============================================

info "Combining facts..."

cat "$OUTPUT_DIR/trees.pl" "$OUTPUT_DIR/pearls.pl" > "$OUTPUT_DIR/all_facts.pl"

success "Combined facts → all_facts.pl"

# ============================================
# Summary
# ============================================

echo ""
echo "========================================="
echo "Extraction Complete"
echo "========================================="
echo "Trees:  $TREE_COUNT"
echo "Pearls: $PEARL_COUNT"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/trees.pl"
echo "  - $OUTPUT_DIR/pearls.pl"
echo "  - $OUTPUT_DIR/all_facts.pl"
echo ""
echo "Next steps:"
echo "  1. Load facts in Prolog:"
echo "     ?- ['"$OUTPUT_DIR/all_facts.pl"']."
echo ""
echo "  2. Query tree relationships:"
echo "     ?- parent_tree(ChildID, ParentID)."
echo ""
echo "  3. Find all pearls for a tree:"
echo "     ?- tree(ID, Title, _, _), pearl(_, ID, _, _)."
echo ""
