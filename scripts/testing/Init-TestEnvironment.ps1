# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# Init-TestEnvironment.ps1 - Initialize UnifyWeaver testing environment (PowerShell)
#
# Usage:
#   .\Init-TestEnvironment.ps1
#   .\Init-TestEnvironment.ps1 -TargetDir "C:\custom\path"
#
# Requirements:
#   - PowerShell 5.1 or later
#   - SWI-Prolog for Windows
#
# Note: If you get an execution policy error, run:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

[CmdletBinding()]
param(
    [Parameter(HelpMessage="Target directory for test environment")]
    [string]$TargetDir = "",

    [Parameter(HelpMessage="Show help message")]
    [switch]$Help
)

# Display help
if ($Help) {
    Write-Host @"
UnifyWeaver Testing Environment Setup (PowerShell)

Usage:
  .\Init-TestEnvironment.ps1 [-TargetDir <path>]

Parameters:
  -TargetDir    Custom target directory for test environment
                Default: .\test_env
  -Help         Show this help message

Examples:
  .\Init-TestEnvironment.ps1
  .\Init-TestEnvironment.ps1 -TargetDir "C:\UnifyWeaver\test"

Environment Variables:
  UNIFYWEAVER_ROOT   Alternative way to specify target directory
"@
    exit 0
}

# Script version
$ScriptVersion = "1.0.0"

Write-Host "===================================" -ForegroundColor Green
Write-Host "UnifyWeaver Testing Environment Setup" -ForegroundColor Green
Write-Host "PowerShell Version - v$ScriptVersion" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Write-Host "[INFO] Script directory: $ScriptDir" -ForegroundColor Cyan
Write-Host "[INFO] Project root: $ProjectRoot" -ForegroundColor Cyan

# Determine target directory
if ($TargetDir) {
    $TargetRoot = $TargetDir
    Write-Host "[INFO] Using specified target: $TargetRoot" -ForegroundColor Yellow
} elseif ($env:UNIFYWEAVER_ROOT) {
    $TargetRoot = $env:UNIFYWEAVER_ROOT
    Write-Host "[INFO] Using UNIFYWEAVER_ROOT: $TargetRoot" -ForegroundColor Yellow
} else {
    $TargetRoot = Join-Path $ScriptDir "test_env"
    Write-Host "[INFO] Using default target: $TargetRoot" -ForegroundColor Yellow
}

# Create target directory
if (!(Test-Path $TargetRoot)) {
    Write-Host "[INFO] Creating target directory..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null
}

# Check if target is empty
$ExistingFiles = Get-ChildItem -Path $TargetRoot -ErrorAction SilentlyContinue
if ($ExistingFiles) {
    Write-Host "[WARNING] Target directory is not empty!" -ForegroundColor Yellow
    Write-Host "[WARNING] Existing files will be overwritten" -ForegroundColor Yellow
    $Response = Read-Host "Continue? (y/N)"
    if ($Response -notmatch '^[Yy]') {
        Write-Host "[ABORT] User cancelled" -ForegroundColor Red
        exit 1
    }
}

# Verify SWI-Prolog is installed
Write-Host ""
Write-Host "[CHECK] Verifying SWI-Prolog installation..." -ForegroundColor Cyan

$SwiplExe = $null

# First check if swipl is on PATH
$SwiplPath = Get-Command swipl -ErrorAction SilentlyContinue
if ($SwiplPath) {
    $SwiplExe = "swipl"
    Write-Host "[OK] Found swipl on PATH" -ForegroundColor Green
} else {
    # Not on PATH - check common installation locations
    Write-Host "[INFO] swipl not on PATH, checking common locations..." -ForegroundColor Yellow

    $CommonLocations = @(
        "$env:ProgramFiles\swipl\bin\swipl.exe",
        "${env:ProgramFiles(x86)}\swipl\bin\swipl.exe",
        "C:\Program Files\swipl\bin\swipl.exe",
        "$env:LocalAppData\Programs\swipl\bin\swipl.exe"
    )

    foreach ($Location in $CommonLocations) {
        if (Test-Path $Location) {
            $SwiplExe = $Location
            Write-Host "[OK] Found: $Location" -ForegroundColor Green
            break
        }
    }

    if (!$SwiplExe) {
        Write-Host "[ERROR] SWI-Prolog not found!" -ForegroundColor Red
        Write-Host "[ERROR]" -ForegroundColor Red
        Write-Host "[ERROR] Checked locations:" -ForegroundColor Red
        Write-Host "[ERROR]   - System PATH" -ForegroundColor Red
        foreach ($Location in $CommonLocations) {
            Write-Host "[ERROR]   - $Location" -ForegroundColor Red
        }
        Write-Host "[ERROR]" -ForegroundColor Red
        Write-Host "[ERROR] Please install SWI-Prolog from:" -ForegroundColor Red
        Write-Host "[ERROR] https://www.swi-prolog.org/download/stable" -ForegroundColor Red
        Write-Host "[ERROR]" -ForegroundColor Red
        Write-Host "[ERROR] Or add swipl to your PATH if already installed." -ForegroundColor Red
        exit 1
    }
}

$SwiplVersion = & $SwiplExe --version 2>&1 | Select-String "SWI-Prolog"
Write-Host "[OK] Version: $SwiplVersion" -ForegroundColor Green

# Copy core modules
Write-Host ""
Write-Host "[COPY] Copying UnifyWeaver core modules..." -ForegroundColor Cyan

$SourceDirs = @(
    @{Source="src"; Dest="src"}
    @{Source="docs"; Dest="docs"}
    @{Source="examples"; Dest="examples"}
    @{Source="scripts"; Dest="scripts"}
)

foreach ($Dir in $SourceDirs) {
    $SourcePath = Join-Path $ProjectRoot $Dir.Source
    $DestPath = Join-Path $TargetRoot $Dir.Dest

    if (Test-Path $SourcePath) {
        Write-Host "  Copying $($Dir.Source)/ -> $($Dir.Dest)/" -ForegroundColor Gray

        # Remove destination if it exists
        if (Test-Path $DestPath) {
            Remove-Item -Path $DestPath -Recurse -Force
        }

        # Copy with robocopy for better performance
        # Exclude test_env* directories to prevent recursive copy
        if ($Dir.Source -eq "scripts") {
            $null = robocopy $SourcePath $DestPath /E /XD "test_env*" /NFL /NDL /NJH /NJS /NC /NS /NP
        } else {
            $null = robocopy $SourcePath $DestPath /E /NFL /NDL /NJH /NJS /NC /NS /NP
        }

        if ($LASTEXITCODE -ge 8) {
            Write-Host "[ERROR] Failed to copy $($Dir.Source)" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[WARNING] Source not found: $SourcePath" -ForegroundColor Yellow
    }
}

Write-Host "[OK] All modules copied" -ForegroundColor Green

# Create output directories
Write-Host ""
Write-Host "[CREATE] Creating output directories..." -ForegroundColor Cyan

$OutputDirs = @("output", "output/advanced")
foreach ($Dir in $OutputDirs) {
    $DirPath = Join-Path $TargetRoot $Dir
    if (!(Test-Path $DirPath)) {
        New-Item -ItemType Directory -Path $DirPath -Force | Out-Null
        Write-Host "  Created: $Dir/" -ForegroundColor Gray
    }
}

# Create launcher script
Write-Host ""
Write-Host "[CREATE] Creating test launcher..." -ForegroundColor Cyan

$LauncherPath = Join-Path $TargetRoot "start_test_env.bat"
$LauncherContent = @"
@echo off
REM UnifyWeaver Test Environment Launcher
REM Auto-generated by Init-TestEnvironment.ps1
REM SWI-Prolog: $SwiplExe

set "TEST_ROOT=%~dp0"
if "%TEST_ROOT:~-1%"=="\" set "TEST_ROOT=%TEST_ROOT:~0,-1%"

echo [UnifyWeaver] Test Environment
echo [UnifyWeaver] Location: %TEST_ROOT%
echo.

cd /d "%TEST_ROOT%"
"$SwiplExe" -q -g "asserta(library_directory('src')), asserta(file_search_path(unifyweaver, 'src/unifyweaver'))" -t prolog
"@

Set-Content -Path $LauncherPath -Value $LauncherContent -Encoding ASCII
Write-Host "  Created: start_test_env.bat" -ForegroundColor Gray

# Create README
$ReadmePath = Join-Path $TargetRoot "README.txt"
$ReadmeContent = @"
UnifyWeaver Test Environment
=============================

This directory contains a complete copy of UnifyWeaver for testing.

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
PowerShell Version: $PSVersionTable.PSVersion

Directory Structure:
  src/          - Core Prolog modules
  docs/         - Documentation
  examples/     - Example predicates
  scripts/      - Utility scripts
  output/       - Compiled bash scripts

To Start:
  Double-click: start_test_env.bat
  Or run: swipl -q

Quick Tests:
  1. Load stream compiler:
     ?- use_module('src/unifyweaver/core/stream_compiler').

  2. Run tests:
     ?- test_stream_compiler.

  3. Compile examples:
     ?- compile_predicate(grandparent/2, [], Code).

For more information, see docs/TESTING.md
"@

Set-Content -Path $ReadmePath -Value $ReadmeContent -Encoding UTF8
Write-Host "  Created: README.txt" -ForegroundColor Gray

# Summary
Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""
Write-Host "Test environment created at:" -ForegroundColor Cyan
Write-Host "  $TargetRoot" -ForegroundColor White
Write-Host ""
Write-Host "To start testing:" -ForegroundColor Cyan
Write-Host "  1. Navigate to: cd '$TargetRoot'" -ForegroundColor White
Write-Host "  2. Run: .\start_test_env.bat" -ForegroundColor White
Write-Host "  3. Or run: swipl" -ForegroundColor White
Write-Host ""
Write-Host "Run tests with:" -ForegroundColor Cyan
Write-Host "  swipl -g ""use_module('src/unifyweaver/core/stream_compiler'), test_stream_compiler, halt.""" -ForegroundColor White
Write-Host ""
