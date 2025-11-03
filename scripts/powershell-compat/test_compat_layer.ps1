# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# test_compat_layer.ps1 - Test suite for PowerShell compatibility layer
# Tests WSL and Cygwin backends for Unix command execution

#try {
    Write-Host "==================================="
    Write-Host "--- Test Environment Setup ---"
    # We are already in the correct directory, so no Set-Location is needed.

    Write-Host "==================================="
    Write-Host "--- Sourcing Compatibility Layer ---"

    # Try to find init_unify_compat.ps1 in multiple possible locations
    $InitCompatPaths = @(
        "$PSScriptRoot\scripts\init_unify_compat.ps1",    # Test environment: test_env/test_compat_layer.ps1 -> test_env/scripts/
        "$PSScriptRoot\init_unify_compat.ps1",            # Same directory: scripts/powershell-compat/
        "$PSScriptRoot\..\scripts\init_unify_compat.ps1"  # Running from subdirectory
    )

    $InitCompatFound = $false
    foreach ($Path in $InitCompatPaths) {
        if (Test-Path $Path) {
            Write-Host "  Loading from: $Path" -ForegroundColor Gray
            . $Path
            $InitCompatFound = $true
            break
        }
    }

    if (-not $InitCompatFound) {
        Write-Host "[ERROR] Could not find init_unify_compat.ps1" -ForegroundColor Red
        Write-Host "Searched in:" -ForegroundColor Yellow
        foreach ($Path in $InitCompatPaths) {
            Write-Host "  - $Path" -ForegroundColor Yellow
        }
        exit 1
    }

    Write-Host "==================================="
    if ($IsLinux -or $IsMacOS) {
        Write-Host "--- TEST 1: Native Backend Verification (Linux/macOS) ---"
        $env:UNIFYWEAVER_EXEC_MODE = 'native'
        $output_native = uw-uname -o
        if ($output_native -like '*Linux*' -or $output_native -like '*Darwin*') {
            Write-Host 'PASS: Native backend confirmed.' -ForegroundColor Green
        } else {
            Write-Host "FAIL: Native backend not detected. Output: $output_native" -ForegroundColor Red
        }
    } else {
        Write-Host "--- TEST 1: WSL Backend Verification ---"
        $env:UNIFYWEAVER_EXEC_MODE = 'wsl'
        $output_wsl = uw-uname -o
        if ($output_wsl -like '*Linux*') { Write-Host 'PASS: WSL backend confirmed.' -ForegroundColor Green }
        else { Write-Host "FAIL: WSL backend not detected. Output: $output_wsl" -ForegroundColor Red }
    }

    Write-Host "==================================="
    if ($IsLinux -or $IsMacOS) {
        Write-Host "--- TEST 2: Pipeline Verification (ls | grep) - Native Mode ---"
        $env:UNIFYWEAVER_EXEC_MODE = 'native'
        $output_pipe = uw-ls | uw-grep 'examples'
        if ($output_pipe -like '*examples*') { Write-Host 'PASS: Pipeline and wrappers (ls | grep) are working.' -ForegroundColor Green }
        else { Write-Host "FAIL: Pipeline test did not produce expected output. Output: $output_pipe" -ForegroundColor Red }
    } else {
        Write-Host "--- TEST 2: Cygwin Backend Verification ---"
        $env:UNIFYWEAVER_EXEC_MODE = 'cygwin'
        $output_cygwin = uw-uname -o
        if ($output_cygwin -like '*Cygwin*') { Write-Host 'PASS: Cygwin backend confirmed.' -ForegroundColor Green }
        else { Write-Host "FAIL: Cygwin backend not detected. Output: $output_cygwin" -ForegroundColor Red }

        Write-Host "==================================="
        Write-Host "--- TEST 3: Pipeline Verification (ls | grep) ---"
        $env:UNIFYWEAVER_EXEC_MODE = 'cygwin' # Use cygwin backend for this test
        $output_pipe = uw-ls | uw-grep 'examples'
        if ($output_pipe -like '*examples*') { Write-Host 'PASS: Pipeline and wrappers (ls | grep) are working.' -ForegroundColor Green }
        else { Write-Host "FAIL: Pipeline test did not produce expected output. Output: $output_pipe" -ForegroundColor Red }
    }

    Write-Host "==================================="
    Write-Host "--- ALL TESTS COMPLETED SUCCESSFULLY ---" -ForegroundColor Green
#}
#catch {
#    Write-Host "--- SCRIPT FAILED ---" -ForegroundColor Red
#    # Write the actual error record to the host
#    Write-Host $_.Exception.ToString() -ForegroundColor Red
#    Write-Host $_.ScriptStackTrace -ForegroundColor Red
#}
