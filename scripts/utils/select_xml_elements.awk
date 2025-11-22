#!/usr/bin/awk -f
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# select_xml_elements.awk - Extract XML elements by tag pattern
#
# Purpose: Generic XML element selector for streaming large XML files.
#          Outputs null-delimited XML chunks for pipeline processing.
#
# Usage:
#   awk -f select_xml_elements.awk -v tag="pt:Tree" input.rdf
#   awk -f select_xml_elements.awk -v tag="pt:.*Pearl" input.rdf
#   awk -f select_xml_elements.awk -v tag="product" catalog.xml
#
# Parameters:
#   -v tag="PATTERN"      - Tag pattern to match (regex)
#   -v delimiter="\0"     - Output delimiter (default: null character)
#   -v include_empty=0    - Include empty elements (default: no)
#
# Examples:
#   # Extract all Pearltrees trees
#   awk -f select_xml_elements.awk -v tag="pt:Tree" pearltrees.rdf
#
#   # Extract all pearl types (RootPearl, AliasPearl, etc.)
#   awk -f select_xml_elements.awk -v tag="pt:.*Pearl" pearltrees.rdf
#
#   # Extract with custom delimiter
#   awk -f select_xml_elements.awk -v tag="product" -v delimiter="\n---\n" catalog.xml
#
# Output: Null-delimited XML elements (suitable for piping to processors)

BEGIN {
    # Validate required parameters
    if (tag == "") {
        print "Error: -v tag=PATTERN required" > "/dev/stderr"
        print "Usage: awk -f select_xml_elements.awk -v tag=\"PATTERN\" input.xml" > "/dev/stderr"
        exit 1
    }

    # Set defaults
    if (delimiter == "") delimiter = "\0"
    if (include_empty == "") include_empty = 0

    # State variables
    in_element = 0
    element_content = ""
    element_count = 0

    # Debug info
    if (debug) {
        print "# Selecting elements matching: " tag > "/dev/stderr"
        print "# Delimiter: " (delimiter == "\0" ? "\\0 (null)" : delimiter) > "/dev/stderr"
    }
}

# Match opening tag
# Note: This regex handles both self-closing and regular opening tags
$0 ~ "<" tag {
    # Handle self-closing tags like <pt:Tree ... />
    if ($0 ~ "<" tag "[^>]*/>") {
        # Self-closing tag - emit immediately
        if (include_empty || $0 !~ /<[^>]*\/>[ \t]*$/) {
            printf "%s%s", $0 "\n", delimiter
            element_count++
        }
        next
    }

    # Regular opening tag
    in_element = 1
    element_content = $0 "\n"
    next
}

# Accumulate element content
in_element {
    element_content = element_content $0 "\n"

    # Match closing tag
    # Extract tag name from opening tag for precise matching
    if ($0 ~ "</" tag ">") {
        # Emit complete element
        printf "%s%s", element_content, delimiter
        element_count++

        # Reset for next element
        in_element = 0
        element_content = ""
    }
}

END {
    if (debug) {
        print "# Extracted " element_count " element(s)" > "/dev/stderr"
    }

    # Warn if we were still in an element (unclosed tag)
    if (in_element) {
        print "Warning: Unclosed element detected at end of file" > "/dev/stderr"
        print "Partial content: " substr(element_content, 1, 100) "..." > "/dev/stderr"
    }
}
