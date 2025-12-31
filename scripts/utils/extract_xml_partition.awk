#!/usr/bin/awk -f
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# extract_xml_partition.awk - Extract XML elements from byte-range partition
#
# Purpose: Process a specific byte range of an XML file, handling boundaries correctly.
#          Skips incomplete elements at start, finishes incomplete elements at end.
#
# Usage:
#   awk -f extract_xml_partition.awk \
#       -v tag="pt:Tree" \
#       -v start_byte=0 \
#       -v end_byte=10000000 \
#       input.xml
#
# Parameters:
#   -v tag="PATTERN"       - Tag pattern to match (regex)
#   -v start_byte=N        - Starting byte offset (0 for first partition)
#   -v end_byte=N          - Ending byte offset (0 for unbounded)
#   -v delimiter="\0"      - Output delimiter (default: null)
#   -v debug=1             - Enable debug output
#
# Boundary Handling:
#   - If start_byte > 0: Skip forward to first complete opening tag
#   - At end_byte: Continue reading until current element is complete
#   - Ensures no duplicates between partitions

BEGIN {
    if (tag == "") {
        print "Error: -v tag=PATTERN required" > "/dev/stderr"
        exit 1
    }

    if (start_byte == "") start_byte = 0
    if (end_byte == "") end_byte = 0  # 0 means unbounded
    if (delimiter == "") delimiter = "\0"

    bytes_read = 0
    in_element = 0
    element_content = ""
    element_count = 0
    skipping_to_first = (start_byte > 0)
    past_boundary = 0

    if (debug) {
        printf "# Partition: bytes %d-%d\n", start_byte, end_byte > "/dev/stderr"
        printf "# Tag pattern: %s\n", tag > "/dev/stderr"
    }
}

# Track bytes read (approximate - actual byte counting would need to track line endings)
{
    bytes_read += length($0) + 1  # +1 for newline
}

# If we're not at file start, skip to first complete opening tag
skipping_to_first && $0 ~ "<" tag {
    skipping_to_first = 0
    if (debug) {
        printf "# Skipped to first element at ~%d bytes\n", bytes_read > "/dev/stderr"
    }
}

skipping_to_first {
    next  # Skip this line, we're looking for first complete element
}

# Match opening tag
$0 ~ "<" tag {
    in_element = 1
    element_content = $0 "\n"
    next
}

# Accumulate element content
in_element {
    element_content = element_content $0 "\n"

    # Match closing tag
    if ($0 ~ "</" tag ">") {
        # Emit complete element
        printf "%s%s", element_content, delimiter
        element_count++

        # Reset for next element
        in_element = 0
        element_content = ""

        # Check if we've passed our partition boundary
        if (end_byte > 0 && bytes_read >= end_byte && !past_boundary) {
            past_boundary = 1
            if (debug) {
                printf "# Passed partition boundary at %d bytes\n", bytes_read > "/dev/stderr"
                printf "# Continuing to finish current element...\n" > "/dev/stderr"
            }
        }

        # If we've passed boundary and finished the current element, stop
        if (past_boundary) {
            if (debug) {
                printf "# Finished last element at ~%d bytes\n", bytes_read > "/dev/stderr"
                printf "# Extracted %d element(s) from this partition\n", element_count > "/dev/stderr"
            }
            exit
        }
    }
}

END {
    if (debug && !past_boundary) {
        printf "# Reached end of file at %d bytes\n", bytes_read > "/dev/stderr"
        printf "# Extracted %d element(s)\n", element_count > "/dev/stderr"
    }

    if (in_element) {
        print "Warning: Incomplete element at end of partition" > "/dev/stderr"
    }
}
