#!/bin/bash
# Runner script for dialog_cli (dialog-based)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for required commands
command -v dialog >/dev/null 2>&1 || { echo "dialog required but not installed."; echo "Install with: apt install dialog (Termux) or brew install dialog (macOS)"; exit 1; }

# Run the TUI application
exec bash "$SCRIPT_DIR/dialog_cli.sh"
