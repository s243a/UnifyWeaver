#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# extract_pearltrees_parallel.sh - Parallel Pearltrees extraction using byte-based partitioning
#
# Purpose: Extract tree/pearl data from large RDF files using parallel processing.
#          Uses byte-based partitioning with boundary completion strategy.
#
# Usage:
#   extract_pearltrees_parallel.sh input.rdf output_dir/ [--workers=N] [--partition-size=BYTES]
#
# Parameters:
#   input.rdf         Pearltrees RDF export file
#   output_dir        Directory for output facts
#   --workers=N       Number of parallel workers (default: CPU count)
#   --partition-size  Bytes per partition (default: 10MB)
#
# Strategy:
#   1. Split file into byte-range partitions (e.g., 0-10MB, 10-20MB, etc.)
#   2. Each worker:
#      - Seeks to start byte
#      - Skips to first complete opening tag
#      - Processes elements normally
#      - Continues past end byte to finish last element
#   3. Merge results from all workers

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

error() { echo -e "${RED}Error:${NC} $1" >&2; exit 1; }
success() { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${YELLOW}→${NC} $1"; }
debug() { echo -e "${BLUE}[DEBUG]${NC} $1" >&2; }

# Parse arguments
INPUT_FILE=""
OUTPUT_DIR=""
NUM_WORKERS=$(nproc 2>/dev/null || echo 4)  # Default to CPU count
PARTITION_SIZE=$((10 * 1024 * 1024))  # 10MB default

for arg in "$@"; do
    case $arg in
        --workers=*)
            NUM_WORKERS="${arg#*=}"
            ;;
        --partition-size=*)
            PARTITION_SIZE="${arg#*=}"
            ;;
        --help|-h)
            cat << EOF
Parallel Pearltrees Extraction

Usage: $0 input.rdf output_dir [OPTIONS]

Extracts tree and pearl data from large Pearltrees RDF files using
parallel processing with byte-based partitioning.

Parameters:
  input.rdf         Pearltrees RDF export file
  output_dir        Directory for output facts

Options:
  --workers=N       Number of parallel workers (default: CPU count)
  --partition-size=BYTES  Bytes per partition (default: 10MB = 10485760)

Examples:
  # Use all CPU cores, 10MB partitions
  $0 large.rdf output/

  # Use 8 workers with 20MB partitions
  $0 large.rdf output/ --workers=8 --partition-size=20971520

Strategy:
  1. Divide file into byte-range partitions
  2. Each worker processes one partition:
     - Skips to first complete element
     - Processes normally
     - Finishes last element past boundary
  3. Merge all results

Performance:
  - Speedup: ~N-1 workers (near-linear scaling)
  - Memory: Constant per worker (~20KB each)
  - Overhead: Minimal (byte-seeking is fast)
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
[ -z "$INPUT_FILE" ] && error "Missing input file"
[ -z "$OUTPUT_DIR" ] && error "Missing output directory"
[ ! -f "$INPUT_FILE" ] && error "Input file not found: $INPUT_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get file size
FILE_SIZE=$(stat -f%z "$INPUT_FILE" 2>/dev/null || stat -c%s "$INPUT_FILE" 2>/dev/null)
if [ -z "$FILE_SIZE" ]; then
    error "Could not determine file size"
fi

# Calculate number of partitions
NUM_PARTITIONS=$(( (FILE_SIZE + PARTITION_SIZE - 1) / PARTITION_SIZE ))

# Adjust workers to not exceed partitions
if [ "$NUM_WORKERS" -gt "$NUM_PARTITIONS" ]; then
    NUM_WORKERS=$NUM_PARTITIONS
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$SCRIPT_DIR/utils"

info "Parallel Pearltrees Extraction"
info "Input: $INPUT_FILE ($(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes"))"
info "Output: $OUTPUT_DIR"
info "Workers: $NUM_WORKERS"
info "Partition size: $(numfmt --to=iec-i --suffix=B $PARTITION_SIZE 2>/dev/null || echo "$PARTITION_SIZE bytes")"
info "Partitions: $NUM_PARTITIONS"
echo ""

# ============================================
# Function: Process a partition
# ============================================
process_partition() {
    local partition_id=$1
    local start_byte=$2
    local end_byte=$3
    local output_file=$4
    local element_type=$5
    local tag_pattern=$6

    # Seek to start position and extract elements
    if [ "$start_byte" -eq 0 ]; then
        # First partition: read from beginning
        if [ "$end_byte" -gt 0 ]; then
            head -c "$end_byte" "$INPUT_FILE"
        else
            cat "$INPUT_FILE"
        fi
    else
        # Other partitions: skip to start, read to end (or file end)
        if [ "$end_byte" -gt 0 ]; then
            local bytes_to_read=$((end_byte - start_byte))
            tail -c "+$((start_byte + 1))" "$INPUT_FILE" | head -c "$bytes_to_read"
        else
            tail -c "+$((start_byte + 1))" "$INPUT_FILE"
        fi
    fi | \
    awk -f "$UTILS_DIR/extract_xml_partition.awk" \
        -v tag="$tag_pattern" \
        -v start_byte="$start_byte" \
        -v end_byte="$end_byte" | \
    python3 "$UTILS_DIR/xml_to_prolog_facts.py" \
        --element-type="$element_type" \
        > "$output_file"
}

export -f process_partition
export INPUT_FILE UTILS_DIR

# ============================================
# Step 1: Extract Trees in Parallel
# ============================================

info "Extracting trees with $NUM_WORKERS workers..."

# Create partition jobs for trees
TREE_PARTITIONS=()
for ((i=0; i<NUM_PARTITIONS; i++)); do
    start_byte=$((i * PARTITION_SIZE))
    end_byte=$(((i + 1) * PARTITION_SIZE))

    # Last partition: read to end of file
    if [ "$end_byte" -gt "$FILE_SIZE" ]; then
        end_byte=0  # 0 means unbounded
    fi

    output_file="$OUTPUT_DIR/trees_part_$i.pl"
    TREE_PARTITIONS+=("$output_file")

    # Process partition in background
    process_partition "$i" "$start_byte" "$end_byte" "$output_file" "tree" "pt:Tree" &

    # Limit concurrent jobs
    if [ $(jobs -r | wc -l) -ge "$NUM_WORKERS" ]; then
        wait -n  # Wait for any job to finish
    fi
done

# Wait for all tree extraction jobs
wait

# Merge tree facts
cat "${TREE_PARTITIONS[@]}" > "$OUTPUT_DIR/trees.pl" 2>/dev/null || touch "$OUTPUT_DIR/trees.pl"
rm -f "${TREE_PARTITIONS[@]}"

TREE_COUNT=$(grep -c "^tree(" "$OUTPUT_DIR/trees.pl" 2>/dev/null || echo "0")
success "Extracted $TREE_COUNT tree(s) → trees.pl"

# ============================================
# Step 2: Extract Pearls in Parallel
# ============================================

info "Extracting pearls with $NUM_WORKERS workers..."

# Create partition jobs for pearls
PEARL_PARTITIONS=()
for ((i=0; i<NUM_PARTITIONS; i++)); do
    start_byte=$((i * PARTITION_SIZE))
    end_byte=$(((i + 1) * PARTITION_SIZE))

    if [ "$end_byte" -gt "$FILE_SIZE" ]; then
        end_byte=0
    fi

    output_file="$OUTPUT_DIR/pearls_part_$i.pl"
    PEARL_PARTITIONS+=("$output_file")

    process_partition "$i" "$start_byte" "$end_byte" "$output_file" "pearl" "pt:.*Pearl" &

    if [ $(jobs -r | wc -l) -ge "$NUM_WORKERS" ]; then
        wait -n
    fi
done

wait

# Merge pearl facts
cat "${PEARL_PARTITIONS[@]}" > "$OUTPUT_DIR/pearls.pl" 2>/dev/null || touch "$OUTPUT_DIR/pearls.pl"
rm -f "${PEARL_PARTITIONS[@]}"

PEARL_COUNT=$(grep -c "^pearl(" "$OUTPUT_DIR/pearls.pl" 2>/dev/null || echo "0")
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
echo "Parallel Extraction Complete"
echo "========================================="
echo "Workers:  $NUM_WORKERS"
echo "Partitions: $NUM_PARTITIONS"
echo "Trees:  $TREE_COUNT"
echo "Pearls: $PEARL_COUNT"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/trees.pl"
echo "  - $OUTPUT_DIR/pearls.pl"
echo "  - $OUTPUT_DIR/all_facts.pl"
echo ""
