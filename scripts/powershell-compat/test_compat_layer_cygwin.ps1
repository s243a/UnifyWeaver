# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# test_compat_layer_cygwin.ps1 - Wrapper script to test compatibility layer with Cygwin backend
# This script can be called from Bash without escaping issues:
#   powershell.exe -File test_compat_layer_cygwin.ps1

# Set the backend to Cygwin
$env:UNIFYWEAVER_EXEC_MODE = 'cygwin'

# Run the main test script
& "$PSScriptRoot\test_compat_layer.ps1"
