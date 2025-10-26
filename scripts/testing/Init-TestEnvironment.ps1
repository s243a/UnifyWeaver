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
    [Parameter(HelpMessage="Parent directory where test_env/ will be created")]
    [Alias("d")]
    [string]$ParentDir = "",

    [Parameter(HelpMessage="Full path to test environment (custom name allowed)")]
    [Alias("p")]
    [string]$TargetPath = "",

    [Parameter(HelpMessage="Show help message")]
    [switch]$Help
)

# Display help
if ($Help) {
    Write-Host @"
UnifyWeaver Testing Environment Setup (PowerShell)

Usage:
  .\Init-TestEnvironment.ps1 [-d <dir>] [-p <path>]

Parameters:
  -d, -ParentDir    Parent directory where test_env/ will be created
                    Example: -d C:\temp creates C:\temp\test_env
  -p, -TargetPath   Full path to test environment (allows custom name)
                    Example: -p C:\temp\my_test creates C:\temp\my_test
  -Help             Show this help message

Examples:
  .\Init-TestEnvironment.ps1                    # Creates test_env in scripts\testing\
  .\Init-TestEnvironment.ps1 -d test_env_ps     # Creates scripts\testing\test_env_ps\test_env
  .\Init-TestEnvironment.ps1 -p test_env_ps     # Creates scripts\testing\test_env_ps (custom name)

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

# Determine target directory (priority: -p > -d > UNIFYWEAVER_ROOT > default)
if ($TargetPath) {
    if ([System.IO.Path]::IsPathRooted($TargetPath)) {
        $TargetRoot = $TargetPath
    } else {
        $TargetRoot = Join-Path $ScriptDir $TargetPath
    }
    Write-Host "[INFO] Using custom full path: $TargetRoot" -ForegroundColor Yellow
} elseif ($ParentDir) {
    if ([System.IO.Path]::IsPathRooted($ParentDir)) {
        $TargetRoot = Join-Path $ParentDir "test_env"
    } else {
        $TargetRoot = Join-Path $ScriptDir (Join-Path $ParentDir "test_env")
    }
    Write-Host "[INFO] Using parent directory: $ParentDir (creating test_env/)" -ForegroundColor Yellow
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

# Copy PowerShell compatibility layer
Write-Host ""
Write-Host "[COPY] Setting up PowerShell compatibility layer..." -ForegroundColor Cyan

$PSCompatSource = Join-Path $ProjectRoot "scripts\powershell-compat"
if (Test-Path $PSCompatSource) {
    # Copy init_unify_compat.ps1 to scripts/
    $InitCompatSource = Join-Path $PSCompatSource "init_unify_compat.ps1"
    $InitCompatDest = Join-Path $TargetRoot "scripts\init_unify_compat.ps1"

    if (Test-Path $InitCompatSource) {
        Copy-Item $InitCompatSource $InitCompatDest -Force
        Write-Host "  Copied: init_unify_compat.ps1 -> scripts/" -ForegroundColor Gray
    } else {
        Write-Host "[WARNING] init_unify_compat.ps1 not found in powershell-compat/" -ForegroundColor Yellow
    }

    # Copy test_compat_layer.ps1 to root
    $TestCompatSource = Join-Path $PSCompatSource "test_compat_layer.ps1"
    $TestCompatDest = Join-Path $TargetRoot "test_compat_layer.ps1"

    if (Test-Path $TestCompatSource) {
        Copy-Item $TestCompatSource $TestCompatDest -Force
        Write-Host "  Copied: test_compat_layer.ps1 -> root/" -ForegroundColor Gray
    } else {
        Write-Host "[WARNING] test_compat_layer.ps1 not found in powershell-compat/" -ForegroundColor Yellow
    }

    Write-Host "[OK] PowerShell compatibility layer installed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] PowerShell compatibility layer not found at: $PSCompatSource" -ForegroundColor Yellow
    Write-Host "[INFO] Skipping PowerShell compatibility setup" -ForegroundColor Yellow
}

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

# Copy init.pl from template
Write-Host ""
Write-Host "[COPY] Setting up init.pl..." -ForegroundColor Cyan

$TemplatePath = Join-Path $ProjectRoot "templates\init_template.pl"
$InitPath = Join-Path $TargetRoot "init.pl"

if (Test-Path $TemplatePath) {
    Copy-Item $TemplatePath $InitPath -Force
    Write-Host "[OK] Copied init.pl from template" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Template not found: $TemplatePath" -ForegroundColor Yellow
    Write-Host "[INFO] Creating minimal init.pl..." -ForegroundColor Cyan

    # Fallback: create minimal init.pl
    $MinimalInit = @"
% Minimal init.pl (template not found)
:- dynamic user:library_directory/1.
:- dynamic user:file_search_path/2.

unifyweaver_init :-
    prolog_load_context(directory, Here),
    directory_file_path(Here, 'src', AbsSrcDir),
    directory_file_path(AbsSrcDir, 'unifyweaver', AbsUnifyweaverDir),
    asserta(user:library_directory(AbsSrcDir)),
    asserta(user:file_search_path(unifyweaver, AbsUnifyweaverDir)),
    format('[UnifyWeaver] Initialized (minimal mode)~n', []).

:- initialization(unifyweaver_init, now).
"@
    Set-Content -Path $InitPath -Value $MinimalInit -Encoding UTF8
    Write-Host "[OK] Created minimal init.pl" -ForegroundColor Yellow
}

# Check for ConEmu
Write-Host ""
Write-Host "[CHECK] Checking for ConEmu terminal..." -ForegroundColor Cyan

$ConEmuExe = $null
$ConEmuLocations = @(
    "$env:ProgramFiles\ConEmu\ConEmu64.exe",
    "${env:ProgramFiles(x86)}\ConEmu\ConEmu.exe",
    "$env:LocalAppData\ConEmu\ConEmu64.exe"
)

foreach ($Location in $ConEmuLocations) {
    if (Test-Path $Location) {
        $ConEmuExe = $Location
        Write-Host "[OK] Found ConEmu: $Location" -ForegroundColor Green
        break
    }
}

if (!$ConEmuExe) {
    Write-Host "[INFO] ConEmu not found - launcher will use standard console" -ForegroundColor Yellow
}

# Copy launcher scripts
Write-Host ""
Write-Host "[COPY] Installing launcher scripts..." -ForegroundColor Cyan

$LauncherSource = Join-Path $ProjectRoot "scripts\launchers"
if (Test-Path $LauncherSource) {
    # Copy main launchers
    $LauncherFiles = @(
        "Start-UnifyWeaverTerminal.ps1",
        "start_powershell_env.bat",
        "start_test_env.bat"
    )

    foreach ($File in $LauncherFiles) {
        $SourceFile = Join-Path $LauncherSource $File
        $DestFile = Join-Path $TargetRoot $File

        if (Test-Path $SourceFile) {
            Copy-Item $SourceFile $DestFile -Force
            Write-Host "  Copied: $File" -ForegroundColor Gray
        } else {
            Write-Host "[WARNING] Launcher not found: $File" -ForegroundColor Yellow
        }
    }

    Write-Host "[OK] Launcher scripts installed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Launcher directory not found: $LauncherSource" -ForegroundColor Yellow
    Write-Host "[INFO] Creating fallback launcher..." -ForegroundColor Cyan

    # Fallback: Create simple launcher if new scripts not found
    $LauncherPath = Join-Path $TargetRoot "start_test_env.bat"
    $FallbackLauncher = @"
@echo off
echo [UnifyWeaver] Starting test environment...
cd /d "%~dp0"
swipl -q -l init.pl -t prolog
"@
    Set-Content -Path $LauncherPath -Value $FallbackLauncher -Encoding ASCII
    Write-Host "[OK] Created fallback launcher" -ForegroundColor Yellow
}

# Optionally copy diagnostic tools
$DiagnosticsSource = Join-Path $ProjectRoot "scripts\diagnostics"
if (Test-Path $DiagnosticsSource) {
    Write-Host ""
    Write-Host "[COPY] Installing diagnostic tools..." -ForegroundColor Cyan

    $DiagnosticsDir = Join-Path $TargetRoot "diagnostics"
    if (!(Test-Path $DiagnosticsDir)) {
        New-Item -ItemType Directory -Path $DiagnosticsDir -Force | Out-Null
    }

    $DiagnosticFiles = Get-ChildItem -Path $DiagnosticsSource -Filter "*.ps1"
    foreach ($File in $DiagnosticFiles) {
        $DestFile = Join-Path $DiagnosticsDir $File.Name
        Copy-Item $File.FullName $DestFile -Force
        Write-Host "  Copied: $($File.Name) -> diagnostics/" -ForegroundColor Gray
    }

    Write-Host "[OK] Diagnostic tools installed" -ForegroundColor Green
}

# Copy init_swipl_env.ps1 to test environment
$SwiplEnvSource = Join-Path $ScriptDir "init_swipl_env.ps1"
if (Test-Path $SwiplEnvSource) {
    Write-Host ""
    Write-Host "[COPY] Installing SWI-Prolog environment script..." -ForegroundColor Cyan

    $SwiplEnvDest = Join-Path $TargetRoot "scripts\init_swipl_env.ps1"
    Copy-Item $SwiplEnvSource $SwiplEnvDest -Force
    Write-Host "[OK] Copied init_swipl_env.ps1 to scripts/" -ForegroundColor Green
}

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
