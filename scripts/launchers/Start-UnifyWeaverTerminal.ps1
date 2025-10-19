# Start-UnifyWeaverTerminal.ps1
# UnifyWeaver Terminal Launcher with Auto-Detection
# Prefers Windows Terminal > ConEmu > Standard PowerShell
#
# Usage:
#   .\Start-UnifyWeaverTerminal.ps1                    # Auto-detect best terminal
#   .\Start-UnifyWeaverTerminal.ps1 -Terminal wt       # Force Windows Terminal
#   .\Start-UnifyWeaverTerminal.ps1 -Terminal conemu   # Force ConEmu
#   .\Start-UnifyWeaverTerminal.ps1 -Terminal powershell  # Force standard PowerShell
#   .\Start-UnifyWeaverTerminal.ps1 -ShowDetails       # Show verbose output

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('auto', 'wt', 'windows-terminal', 'conemu', 'powershell', 'console')]
    [string]$Terminal = 'auto',

    [Parameter(Mandatory=$false)]
    [switch]$ShowDetails
)

$TestRoot = $PSScriptRoot

function Write-Verbose-If {
    param([string]$Message)
    if ($ShowDetails) {
        Write-Host "[VERBOSE] $Message" -ForegroundColor DarkGray
    }
}

Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  UnifyWeaver Test Environment Launcher                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "[UnifyWeaver] Location: $TestRoot" -ForegroundColor Green
Write-Host ""

# --- SWI-Prolog location ---
$SwiplBinDir = "C:\Program Files\swipl\bin"
if (-not (Test-Path "$SwiplBinDir\swipl.exe")) {
    Write-Host "[ERROR] SWI-Prolog not found at: $SwiplBinDir" -ForegroundColor Red
    Write-Host "[ERROR] Please update the script with the correct path" -ForegroundColor Red
    exit 1
}
Write-Verbose-If "Found SWI-Prolog at: $SwiplBinDir"

# --- Detect available terminals ---
$WindowsTerminal = $null
$ConEmu = $null

# Check for Windows Terminal
$WTLocations = @(
    "$env:LocalAppData\Microsoft\WindowsApps\wt.exe",
    "${env:ProgramFiles}\WindowsApps\Microsoft.WindowsTerminal*\wt.exe"
)

foreach ($Location in $WTLocations) {
    $Resolved = Resolve-Path $Location -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($Resolved) {
        $WindowsTerminal = $Resolved.Path
        Write-Verbose-If "Found Windows Terminal: $WindowsTerminal"
        break
    }
}

if (-not $WindowsTerminal) {
    # Try command lookup
    $WTCmd = Get-Command wt.exe -ErrorAction SilentlyContinue
    if ($WTCmd) {
        $WindowsTerminal = $WTCmd.Source
        Write-Verbose-If "Found Windows Terminal via command: $WindowsTerminal"
    }
}

# Check for ConEmu
$ConEmuLocations = @(
    "C:\Program Files\ConEmu\ConEmu64.exe",
    "${env:ProgramFiles(x86)}\ConEmu\ConEmu.exe",
    "$env:LocalAppData\ConEmu\ConEmu64.exe"
)

foreach ($Location in $ConEmuLocations) {
    if (Test-Path $Location) {
        $ConEmu = $Location
        Write-Verbose-If "Found ConEmu: $ConEmu"
        break
    }
}

# --- Determine which terminal to use ---
$SelectedTerminal = $null
$TerminalType = $null

if ($Terminal -eq 'auto') {
    # Prefer Windows Terminal > ConEmu > PowerShell
    if ($WindowsTerminal) {
        $SelectedTerminal = $WindowsTerminal
        $TerminalType = 'windows-terminal'
        Write-Host "[✓] Auto-detected: Windows Terminal (best emoji support)" -ForegroundColor Green
    } elseif ($ConEmu) {
        $SelectedTerminal = $ConEmu
        $TerminalType = 'conemu'
        Write-Host "[ℹ] Auto-detected: ConEmu (BMP symbols only)" -ForegroundColor Yellow
    } else {
        $TerminalType = 'powershell'
        Write-Host "[ℹ] No enhanced terminal found - using standard PowerShell" -ForegroundColor Yellow
    }
} else {
    # User specified terminal
    switch ($Terminal) {
        { $_ -in 'wt', 'windows-terminal' } {
            if ($WindowsTerminal) {
                $SelectedTerminal = $WindowsTerminal
                $TerminalType = 'windows-terminal'
                Write-Host "[✓] Using Windows Terminal (forced)" -ForegroundColor Green
            } else {
                Write-Host "[ERROR] Windows Terminal not found" -ForegroundColor Red
                Write-Host "[INFO] Install from Microsoft Store or use -Terminal auto" -ForegroundColor Yellow
                exit 1
            }
        }
        'conemu' {
            if ($ConEmu) {
                $SelectedTerminal = $ConEmu
                $TerminalType = 'conemu'
                Write-Host "[✓] Using ConEmu (forced)" -ForegroundColor Yellow
            } else {
                Write-Host "[ERROR] ConEmu not found" -ForegroundColor Red
                Write-Host "[INFO] Install from https://conemu.github.io or use -Terminal auto" -ForegroundColor Yellow
                exit 1
            }
        }
        { $_ -in 'powershell', 'console' } {
            $TerminalType = 'powershell'
            Write-Host "[✓] Using standard PowerShell (forced)" -ForegroundColor Cyan
        }
    }
}

Write-Host ""

# --- Build PowerShell session initialization ---
$InitTemplate = @'
Set-Location '{0}'
$env:PATH = '{1};' + $env:PATH

# UTF-8 encoding setup
chcp.com 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# SWI-Prolog locale
$env:LANG  = 'en_US.UTF-8'
$env:LC_ALL = 'en_US.UTF-8'

Write-Host ''
Write-Host '╔════════════════════════════════════════════════════════╗' -ForegroundColor Cyan
Write-Host '║  UnifyWeaver Test Environment Ready                   ║' -ForegroundColor Cyan
Write-Host '╚════════════════════════════════════════════════════════╝' -ForegroundColor Cyan
Write-Host ''
Write-Host '[✓] SWI-Prolog added to PATH (session only)' -ForegroundColor Green
Write-Host '[✓] Console encoding: UTF-8' -ForegroundColor Green

$swiplVersion = swipl --version 2>&1 | Select-Object -First 1
Write-Host "[✓] SWI-Prolog: $swiplVersion" -ForegroundColor Green

Write-Host ''
Write-Host 'Quick Start:' -ForegroundColor Yellow
Write-Host '  swipl -l init.pl              # Load test environment' -ForegroundColor Gray
Write-Host '  swipl -g main -t halt init.pl # Run main/0 and exit' -ForegroundColor Gray
Write-Host ''
Write-Host 'PowerShell compatibility layer:' -ForegroundColor Yellow
Write-Host '  . .\scripts\init_unify_compat.ps1' -ForegroundColor Gray
Write-Host '  uw-ls, uw-grep, uw-awk, etc.' -ForegroundColor Gray
Write-Host ''
'@

$FormatArgs = @(
    ($TestRoot -replace "'", "''"),
    ($SwiplBinDir -replace "'", "''")
)

$InitScript = [string]::Format($InitTemplate, $FormatArgs)

# --- Launch terminal ---
switch ($TerminalType) {
    'windows-terminal' {
        Write-Host "[→] Launching Windows Terminal..." -ForegroundColor Cyan
        Write-Host "[ℹ] Emoji level: FULL (color emoji supported)" -ForegroundColor Green
        Write-Host ""

        # Use pwsh if available, otherwise powershell
        $PSExe = if (Get-Command pwsh.exe -ErrorAction SilentlyContinue) { 'pwsh.exe' } else { 'powershell.exe' }

        # Encode the script to avoid escaping issues (same approach as ConEmu)
        $ScriptBlockText = "& {`n$InitScript`n}"
        $EncodedCommand = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($ScriptBlockText))

        # Build Windows Terminal command line
        $WTCommand = "$PSExe -NoLogo -NoExit -ExecutionPolicy Bypass -EncodedCommand $EncodedCommand"

        # Launch Windows Terminal with encoded command
        Start-Process -FilePath $SelectedTerminal -ArgumentList "-d `"$TestRoot`" $WTCommand"
    }

    'conemu' {
        Write-Host "[→] Launching ConEmu..." -ForegroundColor Cyan
        Write-Host "[ℹ] Emoji level: BMP (symbols ✅ ❌ ⚠ ℹ ⚡ only)" -ForegroundColor Yellow
        Write-Host "[ℹ] For full emoji, use Windows Terminal (-Terminal wt)" -ForegroundColor Yellow
        Write-Host ""

        # Encode command for ConEmu
        $ScriptBlockText = "& {`n$InitScript`n}"
        $EncodedCommand = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($ScriptBlockText))

        $PSExe = if (Get-Command pwsh.exe -ErrorAction SilentlyContinue) { 'pwsh.exe' } else { 'powershell.exe' }
        $CmdInvocation = "cmd.exe /k ""chcp 65001 >nul & $PSExe -NoLogo -NoExit -ExecutionPolicy Bypass -EncodedCommand $EncodedCommand"""

        & $SelectedTerminal -Dir $TestRoot -run $CmdInvocation
    }

    'powershell' {
        Write-Host "[→] Launching standard PowerShell window..." -ForegroundColor Cyan
        Write-Host "[ℹ] Emoji level: Depends on terminal font configuration" -ForegroundColor Yellow
        Write-Host ""

        $ScriptBlockText = "& {`n$InitScript`n}"
        $EncodedCommand = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($ScriptBlockText))

        $PSExe = if (Get-Command pwsh.exe -ErrorAction SilentlyContinue) { 'pwsh.exe' } else { 'powershell.exe' }
        Start-Process cmd.exe -ArgumentList '/k', "chcp 65001 >nul & $PSExe -NoLogo -NoExit -ExecutionPolicy Bypass -EncodedCommand $EncodedCommand" -WorkingDirectory $TestRoot
    }
}

Write-Host "[✓] Terminal launched successfully" -ForegroundColor Green
Write-Host ""
