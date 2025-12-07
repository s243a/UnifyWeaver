# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# This file is part of UnifyWeaver.
# Licensed under either MIT or Apache-2.0 at your option.

#!/bin/bash

# test_all_examples.sh - Automated tester for all code examples in the education/ directory

# --- Configuration ---

# Default output directory (users can override with -o flag)
OUTPUT_DIR="education/output"
USE_TEMP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--temp)
            USE_TEMP=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -o, --output DIR    Output directory for generated files (default: education/output)"
            echo "  -t, --temp          Use temporary directory for output (safer for automated testing)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Use default education/output directory"
            echo "  $0 -o custom_dir    # Use custom directory"
            echo "  $0 -t               # Use temporary directory (safer for LLMs)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# --- Setup ---

set -e # Exit on first error

# Set up output directory
if [[ "$USE_TEMP" == true ]]; then
    OUTPUT_DIR=$(mktemp -d)
    echo "Using temporary output directory: $OUTPUT_DIR"
    # Cleanup trap for temp directory
    trap 'rm -rf -- "$OUTPUT_DIR"' EXIT
else
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    echo "Using output directory: $OUTPUT_DIR"
fi

# Copy necessary files (family_tree.pl, etc.) to a temp location for loading
TEMP_FILES_DIR=$(mktemp -d)
cp education/*.pl "$TEMP_FILES_DIR/"
trap 'rm -rf -- "$TEMP_FILES_DIR" $([[ "$USE_TEMP" == true ]] && echo "$OUTPUT_DIR")' EXIT

# DON'T change to temp directory - stay in project root for module resolution
# cd "$TEST_DIR"  # <-- This line removed to fix module loading

# --- Helper Functions ---

run_prolog_test() {
    local description="$1"
    local code="$2"
    echo -n "  Testing: $description ... "
    # Run from project root - this allows module resolution to work correctly
    if swipl -q -s "education/init.pl" -g "$code, halt." <<< ""; then
        echo "✅ PASS"
    else
        echo "❌ FAIL"
        exit 1
    fi
}

run_bash_test() {
    local description="$1"
    local code="$2"
    echo -n "  Testing: $description ... "
    # Run the bash code in a subshell
    if (eval "$code"); then
        echo "✅ PASS"
    else
        echo "❌ FAIL"
        exit 1
    fi
}

# --- Test Execution ---

echo "--- Running All Education Examples ---"

# Chapter 4: Your First UnifyWeaver Program
echo "
--- Chapter 4: Your First UnifyWeaver Program ---"
run_prolog_test "Initialize Environment" "true"
run_prolog_test "Full Chapter 4 Workflow" "
    use_module(unifyweaver(core/recursive_compiler)), 
    use_module(unifyweaver(core/stream_compiler)), 
    ['education/family_tree'],
    stream_compiler:compile_facts(parent, 2, [], _),
    compile_recursive(ancestor/2, [], _),
    stream_compiler:compile_facts(parent, 2, [], ParentCode), 
    open('$OUTPUT_DIR/parent.sh', write, ParentStream), 
    write(ParentStream, ParentCode), 
    close(ParentStream),
    compile_recursive(ancestor/2, [], AncestorCode), 
    open('$OUTPUT_DIR/ancestor.sh', write, AncestorStream), 
    write(AncestorStream, AncestorCode), 
    close(AncestorStream)"

run_bash_test "Source Scripts" "source '$OUTPUT_DIR/parent.sh' && source '$OUTPUT_DIR/ancestor.sh'"
run_bash_test "Run ancestor command" "source '$OUTPUT_DIR/parent.sh' && source '$OUTPUT_DIR/ancestor.sh' && ancestor abraham jacob | grep -q 'abraham:jacob'"

# Chapter 12: Seamless Compilation with the Compiler Driver
echo "
--- Chapter 12: Seamless Compilation with the Compiler Driver ---"
run_prolog_test "Chapter 12 Workflow" "
    use_module(unifyweaver(core/compiler_driver)),
    ['education/family_tree'],
    compiler_driver:compile(ancestor/2, [output_dir('$OUTPUT_DIR')], _)"
run_bash_test "Run Generated Scripts" "source '$OUTPUT_DIR/parent.sh' && source '$OUTPUT_DIR/ancestor.sh' && ancestor abraham jacob | grep -q 'abraham:jacob'"

echo "
--- All Education Examples Passed! ---"
