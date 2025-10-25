# init_swipl_env.ps1
# UnifyWeaver SWI-Prolog Environment Setup
# Source this script to add SWI-Prolog to PATH
#
# Usage:
#   . .\scripts\testing\init_swipl_env.ps1
#   . .\scripts\init_swipl_env.ps1  # From test environment root

param(
    [Parameter(Mandatory=$false)]
    [switch]$Quiet
)

function Write-Info {
    param([string]$Message)
    if (-not $Quiet) {
        Write-Host $Message -ForegroundColor Green
    }
}

function Write-Warn {
    param([string]$Message)
    if (-not $Quiet) {
        Write-Host $Message -ForegroundColor Yellow
    }
}

# --- Find SWI-Prolog ---
$SwiplLocations = @(
    "C:\Program Files\swipl\bin",
    "C:\Program Files (x86)\swipl\bin",
    "C:\swipl\bin",
    "$env:LocalAppData\swipl\bin",
    "$env:ProgramFiles\swipl\bin"
)

$SwiplBinDir = $null
foreach ($Location in $SwiplLocations) {
    if (Test-Path "$Location\swipl.exe") {
        $SwiplBinDir = $Location
        break
    }
}

if (-not $SwiplBinDir) {
    Write-Host "[ERROR] SWI-Prolog not found in common locations:" -ForegroundColor Red
    foreach ($Location in $SwiplLocations) {
        Write-Host "  - $Location" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "Please install SWI-Prolog from: https://www.swi-prolog.org/download/stable" -ForegroundColor Yellow
    return
}

# --- Check if already on PATH ---
$CurrentSwipl = Get-Command swipl.exe -ErrorAction SilentlyContinue
if ($CurrentSwipl -and $CurrentSwipl.Source -eq "$SwiplBinDir\swipl.exe") {
    Write-Info "[✓] SWI-Prolog already on PATH: $SwiplBinDir"
    return
}

# --- Add to PATH (session only) ---
if ($env:PATH -notlike "*$SwiplBinDir*") {
    $env:PATH = "$SwiplBinDir;$env:PATH"
    Write-Info "[✓] Added SWI-Prolog to PATH: $SwiplBinDir"
} else {
    Write-Info "[✓] SWI-Prolog already in PATH: $SwiplBinDir"
}

# --- UTF-8 encoding setup ---
try {
    chcp.com 65001 > $null 2>&1
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding  = [System.Text.Encoding]::UTF8
    $global:OutputEncoding    = [System.Text.Encoding]::UTF8
    Write-Info "[✓] Console encoding: UTF-8"
} catch {
    Write-Warn "[⚠] Could not set UTF-8 encoding (non-interactive shell?)"
}

# --- Set locale for SWI-Prolog ---
$env:LANG   = 'en_US.UTF-8'
$env:LC_ALL = 'en_US.UTF-8'

# --- Verify installation ---
try {
    $Version = & swipl --version 2>&1 | Select-Object -First 1
    Write-Info "[✓] $Version"
} catch {
    Write-Host "[ERROR] SWI-Prolog found but cannot execute: $SwiplBinDir\swipl.exe" -ForegroundColor Red
}
