#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# test_from_bash.sh - Bash wrapper to test PowerShell compatibility layer from WSL/Bash
#
# Usage:
#   ./test_from_bash.sh wsl       # Test with WSL backend
#   ./test_from_bash.sh cygwin    # Test with Cygwin backend
#   ./test_from_bash.sh           # Test with default (Cygwin) backend

set -euo pipefail

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine backend (default to cygwin)
BACKEND="${1:-cygwin}"

case "$BACKEND" in
    wsl)
        WRAPPER_SCRIPT="test_compat_layer_wsl.ps1"
        ;;
    cygwin)
        WRAPPER_SCRIPT="test_compat_layer_cygwin.ps1"
        ;;
    *)
        echo "Usage: $0 [wsl|cygwin]"
        echo "  wsl    - Test with WSL backend"
        echo "  cygwin - Test with Cygwin backend"
        exit 1
        ;;
esac

# Find PowerShell executable
POWERSHELL=""
if command -v powershell.exe &> /dev/null; then
    POWERSHELL="powershell.exe"
elif command -v pwsh.exe &> /dev/null; then
    POWERSHELL="pwsh.exe"
elif command -v pwsh &> /dev/null; then
    POWERSHELL="pwsh"
elif [ -f "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" ]; then
    POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
else
    echo "Error: PowerShell not found"
    echo "Tried:"
    echo "  - powershell.exe (in PATH)"
    echo "  - pwsh.exe (in PATH)"
    echo "  - pwsh (in PATH)"
    echo "  - /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    exit 1
fi

echo "====================================="
echo "Testing PowerShell Compatibility Layer"
echo "Backend: $BACKEND"
echo "PowerShell: $POWERSHELL"
echo "====================================="

# Convert WSL path to Windows path for PowerShell
WRAPPER_PATH="$SCRIPT_DIR/$WRAPPER_SCRIPT"
if [[ "$WRAPPER_PATH" == /mnt/* ]]; then
    # Convert /mnt/c/... to C:\...
    DRIVE="${WRAPPER_PATH:5:1}"
    REST="${WRAPPER_PATH:7}"
    WRAPPER_PATH_WIN="${DRIVE^^}:${REST//\//\\}"
else
    WRAPPER_PATH_WIN="$WRAPPER_PATH"
fi

# Use -File parameter to avoid escaping issues
"$POWERSHELL" -File "$WRAPPER_PATH_WIN"
