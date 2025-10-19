# start_powershell_env.ps1
# UnifyWeaver PowerShell Test Environment Launcher
# Adds SWI-Prolog to PATH (session-only) and launches PowerShell in ConEmu

$TestRoot = $PSScriptRoot

Write-Host "[UnifyWeaver] PowerShell Test Environment" -ForegroundColor Green
Write-Host "[UnifyWeaver] Location: $TestRoot" -ForegroundColor Green
Write-Host ""

# --- SWI-Prolog location ---
$SwiplBinDir = "C:\Program Files\swipl\bin"
if (-not (Test-Path "$SwiplBinDir\swipl.exe")) {
    Write-Host "[ERROR] SWI-Prolog not found at: $SwiplBinDir" -ForegroundColor Red
    Write-Host "[ERROR] Please update the script with the correct path" -ForegroundColor Red
    exit 1
}

# --- Check for ConEmu ---
$ConEmuExe = $null
$ConEmuLocations = @(
    "C:\Program Files\ConEmu\ConEmu64.exe",
    "${env:ProgramFiles(x86)}\ConEmu\ConEmu.exe",
    "$env:LocalAppData\ConEmu\ConEmu64.exe"
)

foreach ($Location in $ConEmuLocations) {
    if (Test-Path $Location) {
        $ConEmuExe = $Location
        Write-Host "[INFO] Found ConEmu: $Location" -ForegroundColor Cyan
        break
    }
}

# --- Build PowerShell session script ---
$CommandTemplate = @'
Set-Location '{0}'
$env:PATH = '{1};' + $env:PATH

# Ensure console stays in UTF-8 inside PowerShell
chcp.com 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# Set SWI-Prolog locale for Unicode
$env:LANG  = 'en_US.UTF-8'
$env:LC_ALL = 'en_US.UTF-8'

Write-Host '[UnifyWeaver] SWI-Prolog added to PATH (session only)' -ForegroundColor Green
Write-Host '[UnifyWeaver] Console encoding set to UTF-8' -ForegroundColor Green
Write-Host '[UnifyWeaver] Test swipl availability: ' -NoNewline -ForegroundColor Cyan
$swiplVersion = swipl --version 2>&1 | Select-Object -First 1
Write-Host $swiplVersion -ForegroundColor Green
Write-Host ''
Write-Host 'PowerShell compatibility layer available - use uw-* commands:' -ForegroundColor Yellow
Write-Host '  uw-ls, uw-grep, uw-sed, uw-awk, uw-cat, etc.' -ForegroundColor Gray
Write-Host ''
Write-Host 'To load compatibility layer:' -ForegroundColor Yellow
Write-Host '  . .\scripts\init_unify_compat.ps1' -ForegroundColor Gray
Write-Host ''
'@

$FormatArgs = @(
    ($TestRoot -replace "'", "''"),
    ($SwiplBinDir -replace "'", "''")
)

$CommandText = [string]::Format($CommandTemplate, $FormatArgs)
$ScriptBlockText = "& {`n$CommandText`n}"
$EncodedCommand  = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($ScriptBlockText))

# --- Construct common command line ---
# Use pwsh.exe (PowerShell 7+) if available, otherwise fall back to powershell.exe (5.1)
$PwshExe = Get-Command pwsh.exe -ErrorAction SilentlyContinue
$PSExe = if ($PwshExe) { 'pwsh.exe' } else { 'powershell.exe' }
$CmdInvocation = "cmd.exe /k ""chcp 65001 >nul & $PSExe -NoLogo -NoExit -ExecutionPolicy Bypass -EncodedCommand $EncodedCommand"""

# --- Launch with ConEmu if available ---
if ($ConEmuExe) {
    Write-Host "[UnifyWeaver] Launching PowerShell in ConEmu..." -ForegroundColor Cyan
    & $ConEmuExe -Dir $TestRoot -run $CmdInvocation
} else {
    Write-Host "[INFO] ConEmu not found - launching in standard PowerShell window" -ForegroundColor Yellow
    Write-Host ""
    # Use pwsh.exe (PowerShell 7+) if available, otherwise fall back to powershell.exe (5.1)
    $PwshExe = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    $PSExe = if ($PwshExe) { 'pwsh.exe' } else { 'powershell.exe' }
    Start-Process cmd.exe -ArgumentList '/k', "chcp 65001 >nul & $PSExe -NoLogo -NoExit -ExecutionPolicy Bypass -EncodedCommand $EncodedCommand" -WorkingDirectory $TestRoot
}
