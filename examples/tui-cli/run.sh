#!/bin/bash
# Runner script for http_cli

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for required commands
command -v tput >/dev/null 2>&1 || { echo "tput required but not installed."; exit 1; }

# Run the TUI application
exec bash "$SCRIPT_DIR/http_cli.sh"
