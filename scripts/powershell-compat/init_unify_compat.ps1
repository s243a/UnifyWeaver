# init_unify_compat.ps1
# UnifyWeaver PowerShell Compatibility Layer (PowerShell 5.1-safe)
# - uw-* prefixed wrappers to avoid PowerShell alias/cmdlet conflicts
# - Streams stdin into WSL/Cygwin for Unix-like pipelines
# - Normalizes WSL output to UTF-8 for stable interop
# - Synchronizes backend working directory with the PowerShell cwd

Set-StrictMode -Version Latest

# --- Auto-load SWI-Prolog environment ---
# Look for init_swipl_env.ps1 in common locations
$SwiplEnvScript = $null
$SearchPaths = @(
    (Join-Path $PSScriptRoot '..' 'testing' 'init_swipl_env.ps1'),  # From scripts/powershell-compat/
    (Join-Path $PSScriptRoot 'init_swipl_env.ps1'),                  # From scripts/testing/
    (Join-Path $PSScriptRoot '..' 'init_swipl_env.ps1')              # From test environment scripts/
)

foreach ($Path in $SearchPaths) {
    if (Test-Path $Path) {
        $SwiplEnvScript = $Path
        break
    }
}

if ($SwiplEnvScript) {
    # Source SWI-Prolog environment (quiet mode to avoid noise)
    . $SwiplEnvScript -Quiet
} else {
    # Only warn if swipl is not already available
    if (-not (Get-Command swipl.exe -ErrorAction SilentlyContinue)) {
        Write-Host "[⚠] SWI-Prolog environment script not found. Run: . .\scripts\testing\init_swipl_env.ps1" -ForegroundColor Yellow
    }
}

function _PosixQuote {
    param([Parameter(Mandatory)][string]$s)
    return "'" + ($s -replace "'", "'\\''") + "'"
}

function _ToWslPath {
    param([Parameter(Mandatory)][string]$winPath)
    $drive = $winPath.Substring(0,1).ToLower()
    $rest  = $winPath.Substring(2) -replace '\\','/'
    return "/mnt/$drive$rest"
}

function _ToCygPath {
    param([Parameter(Mandatory)][string]$winPath)
    $drive = $winPath.Substring(0,1).ToLower()
    $rest  = $winPath.Substring(2) -replace '\\','/'
    return "/cygdrive/$drive$rest"
}

function Invoke-UnifyCommand {
    param(
        [Parameter(Mandatory=$true)][string]$Command,
        [string]$Stdin,
        [int[]]$AcceptExitCodes
    )

    $mode = $env:UNIFYWEAVER_EXEC_MODE
    if (-not $mode) { $mode = 'cygwin' }

    $pwdWin = (Get-Location).Path

    if ($mode -ieq 'wsl') {
        $env:WSL_UTF8 = '1'
        $cwd = _ToWslPath $pwdWin
        $cmdFull = "cd $(_PosixQuote $cwd) && $Command"

        if ($PSBoundParameters.ContainsKey('Stdin')) {
            $out = $Stdin | wsl.exe bash -lc $cmdFull 2>&1
        } else {
            $out = wsl.exe bash -lc $cmdFull 2>&1
        }
        $code = $LASTEXITCODE
        if ($code -ne 0 -and (-not $AcceptExitCodes -or -not ($AcceptExitCodes -contains $code))) {
            throw ("WSL exit {0}: {1}" -f $code, $out)
        }
        return $out
    }
    elseif ($mode -ieq 'cygwin') {
        $bash = $null
        if (Test-Path 'C:\cygwin64\bin\bash.exe') { $bash = 'C:\cygwin64\bin\bash.exe' }
        elseif (Test-Path 'C:\cygwin\bin\bash.exe') { $bash = 'C:\cygwin\bin\bash.exe' }
        else { throw 'Cygwin bash not found at C:\cygwin64\bin or C:\cygwin\bin' }

        $cwd = _ToCygPath $pwdWin
        $cmdFull = "cd $(_PosixQuote $cwd) && $Command"

        if ($PSBoundParameters.ContainsKey('Stdin')) {
            $out = $Stdin | & $bash --login -c $cmdFull 2>&1
        } else {
            $out = & $bash --login -c $cmdFull 2>&1
        }
        $code = $LASTEXITCODE
        if ($code -ne 0 -and (-not $AcceptExitCodes -or -not ($AcceptExitCodes -contains $code))) {
            throw ("Cygwin exit {0}: {1}" -f $code, $out)
        }
        return $out
    }
    else {
        throw ("Unknown UNIFYWEAVER_EXEC_MODE: {0}" -f $mode)
    }
}

# Publish invoker globally so wrappers can call it by name across scopes
New-Item -Path Function:\global:Invoke-UnifyCommand -Value (Get-Item Function:\Invoke-UnifyCommand).ScriptBlock -Force | Out-Null

# Generic one-off runner
function uw-run {
    param([Parameter(Mandatory)][string]$Cmd)
    & 'Invoke-UnifyCommand' -Command $Cmd
}

# Streaming filter wrapper (global registration)
function New-UWFilter {
    param(
        [Parameter(Mandatory)][string]$Name,
        [string]$CmdName,
        [int[]]$AcceptExitCodes
    )
    if (-not $CmdName) { $CmdName = $Name }
    $cmdNameLocal = $CmdName
    $acceptLocal  = @($AcceptExitCodes)

    $body = {
        [CmdletBinding()]
        param(
            [Parameter(Position=0, ValueFromRemainingArguments=$true)]
            [string[]]$ToolArgs,
            [Parameter(ValueFromPipeline=$true)]
            $InputObject
        )
        begin { $sb = New-Object System.Text.StringBuilder }
        process { if ($null -ne $InputObject) { [void]$sb.AppendLine([string]$InputObject) } }
        end {
            $stdin = $sb.ToString()
            $cmd = "$cmdNameLocal " + (($ToolArgs | ForEach-Object { $_ }) -join ' ')
            if ($stdin.Length -gt 0) {
                & 'Invoke-UnifyCommand' -Command $cmd -Stdin $stdin -AcceptExitCodes $acceptLocal
            } else {
                & 'Invoke-UnifyCommand' -Command $cmd -AcceptExitCodes $acceptLocal
            }
        }
    }.GetNewClosure()

    New-Item -Path ("Function:\global:{0}" -f $Name) -Value $body -Force | Out-Null
}

# Non-filter wrapper (global registration)
function New-UWCommand {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$CmdName,
        [int[]]$AcceptExitCodes
    )
    $cmdNameLocal = $CmdName
    $acceptLocal  = @($AcceptExitCodes)

    $body = {
        $cmd = "$cmdNameLocal " + ($args -join ' ')
        & 'Invoke-UnifyCommand' -Command $cmd -AcceptExitCodes $acceptLocal
    }.GetNewClosure()

    New-Item -Path ("Function:\global:{0}" -f $Name) -Value $body -Force | Out-Null
}

# Filters (stdin-driven)
New-UWFilter -Name 'uw-grep' -CmdName 'grep --color=never' -AcceptExitCodes @(0,1)
New-UWFilter -Name 'uw-sed'  -CmdName 'sed'
New-UWFilter -Name 'uw-awk'  -CmdName 'awk'
New-UWFilter -Name 'uw-sort' -CmdName 'sort'
New-UWFilter -Name 'uw-uniq' -CmdName 'uniq'
New-UWFilter -Name 'uw-cut'  -CmdName 'cut'
New-UWFilter -Name 'uw-tr'   -CmdName 'tr'
New-UWFilter -Name 'uw-head' -CmdName 'head'
New-UWFilter -Name 'uw-tail' -CmdName 'tail'
New-UWFilter -Name 'uw-wc'   -CmdName 'wc'
New-UWFilter -Name 'uw-jq'   -CmdName 'jq'
New-UWFilter -Name 'uw-cat'  -CmdName 'cat'

# Non-filters
New-UWCommand -Name 'uw-uname'   -CmdName 'uname'
New-UWCommand -Name 'uw-which'   -CmdName 'which'
New-UWCommand -Name 'uw-pwd'     -CmdName 'pwd'
New-UWCommand -Name 'uw-date'    -CmdName 'date'
New-UWCommand -Name 'uw-find'    -CmdName 'find'
New-UWCommand -Name 'uw-curl'    -CmdName 'curl'
New-UWCommand -Name 'uw-wget'    -CmdName 'wget'
New-UWCommand -Name 'uw-sqlite3' -CmdName 'sqlite3'
New-UWCommand -Name 'uw-cp'      -CmdName 'cp'
New-UWCommand -Name 'uw-mv'      -CmdName 'mv'
New-UWCommand -Name 'uw-rm'      -CmdName 'rm'
New-UWCommand -Name 'uw-mkdir'   -CmdName 'mkdir'
New-UWCommand -Name 'uw-rmdir'   -CmdName 'rmdir'

function uw-bash {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$CommandArgs
    )

    if ($CommandArgs.Length -gt 0 -and $CommandArgs[0] -eq '-c' -and $CommandArgs.Length -ge 2) {
        $scriptContent = $CommandArgs[1]
        $extraArgs = if ($CommandArgs.Length -gt 2) {
            ' ' + ($CommandArgs[2..($CommandArgs.Length - 1)] | ForEach-Object { _PosixQuote $_ } -join ' ')
        } else {
            ''
        }
        Invoke-UnifyCommand -Command ("bash$extraArgs") -Stdin $scriptContent
    } elseif ($CommandArgs.Length -gt 0) {
        $quotedArgs = $CommandArgs | ForEach-Object { _PosixQuote $_ }
        Invoke-UnifyCommand -Command ("bash " + ($quotedArgs -join ' '))
    } else {
        Invoke-UnifyCommand -Command 'bash'
    }
}

New-Item -Path Function:\global:uw-bash -Value (Get-Item Function:\uw-bash).ScriptBlock -Force | Out-Null

# Deterministic listing for pipelines
New-UWCommand -Name 'uw-ls' -CmdName 'ls -1'

# Fail-fast self-check using Function:\global
foreach ($n in 'uw-uname','uw-ls','uw-grep') {
    if (-not (Test-Path ("Function:\global:{0}" -f $n))) {
        throw "$n failed to load"
    }
}

Write-Host ("UnifyWeaver compatibility layer loaded. Backend mode: {0}" -f $env:UNIFYWEAVER_EXEC_MODE)
