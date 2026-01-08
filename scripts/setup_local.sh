#!/bin/bash
# Setup the .local directory structure for UnifyWeaver
#
# This creates the local data and tool directories used by the project.
# The browser-automation tools should be cloned separately from pt-harvester.
#
# Usage:
#   ./scripts/setup_local.sh
#
# After running this script, clone pt-harvester:
#   git clone git@github.com:s243a/pt-harvester.git .local/tools/browser-automation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_DIR="$PROJECT_ROOT/.local"

echo "Setting up .local directory structure in $LOCAL_DIR"

# Create directory structure
mkdir -p "$LOCAL_DIR/bin"          # Executable scripts/tools
mkdir -p "$LOCAL_DIR/data/pearltrees_api"  # Cached API responses
mkdir -p "$LOCAL_DIR/data/scans"   # Scan results (incomplete mindmaps, embeddings)
mkdir -p "$LOCAL_DIR/lib"          # Shared libraries/modules
mkdir -p "$LOCAL_DIR/tools"        # External tool repos (e.g., browser-automation)

echo "Created directories:"
echo "  .local/bin/              - Executable scripts"
echo "  .local/data/pearltrees_api/ - Cached API responses"
echo "  .local/data/scans/       - Scan results and embeddings"
echo "  .local/lib/              - Shared libraries"
echo "  .local/tools/            - External tool repositories"

# Check if browser-automation exists
if [ -d "$LOCAL_DIR/tools/browser-automation" ]; then
    echo ""
    echo "browser-automation already exists at .local/tools/browser-automation"
else
    echo ""
    echo "Next step: Clone pt-harvester for browser automation tools:"
    echo "  git clone git@github.com:s243a/pt-harvester.git .local/tools/browser-automation"
fi

echo ""
echo "Setup complete!"
