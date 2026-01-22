#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025-2026 John William Creighton (@s243a)
#
# generate.sh - Generate HTTP CLI Server from spec.pl
#
# This script runs the Prolog generators to produce TypeScript
# code from the declarative specification.
#
# Usage:
#   ./generate.sh
#
# Or from project root:
#   ./examples/http-cli-server/generate.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Generating HTTP CLI Server..."
echo "  Spec:   $SCRIPT_DIR/spec.pl"
echo "  Output: $SCRIPT_DIR/generated/"

# Ensure output directory exists
mkdir -p "$SCRIPT_DIR/generated"

# Run the Prolog generator
cd "$SCRIPT_DIR"
swipl -g "consult('spec.pl'), generate_all, halt" -t halt

echo ""
echo "Generation complete!"
echo "Files generated in: $SCRIPT_DIR/generated/"
