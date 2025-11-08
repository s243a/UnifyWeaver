# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# test_compat_layer_wsl.ps1 - Wrapper script to test compatibility layer with WSL backend
# This script can be called from Bash without escaping issues:
#   powershell.exe -File test_compat_layer_wsl.ps1

# Set the backend to WSL
$env:UNIFYWEAVER_EXEC_MODE = 'wsl'

# Run the main test script
& "$PSScriptRoot\test_compat_layer.ps1"
