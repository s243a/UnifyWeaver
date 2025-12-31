# Runs C# query-mode end-to-end execution without relying on SWI's process_create/3.
# It first generates per-plan C# console projects via the Prolog test suite in
# codegen-only mode (SKIP_CSHARP_EXECUTION=1), then builds and executes each
# generated project with dotnet and compares output to expected_rows.txt.
#
# Usage:
#   pwsh -File .\scripts\testing\run_csharp_query_runtime_smoke.ps1
#   pwsh -File .\scripts\testing\run_csharp_query_runtime_smoke.ps1 -KeepArtifacts
#   pwsh -File .\scripts\testing\run_csharp_query_runtime_smoke.ps1 -OutputDir tmp/csharp_query_smoke
#
# Notes:
# - Requires SWI-Prolog and dotnet (net9.0 SDK) on PATH (or in common SWI locations).
# - Generated bin/obj artifacts should be git-ignored.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "tmp/csharp_query_smoke",

    [Parameter(Mandatory = $false)]
    [string]$ProjectFilter = "csharp_query_*",

    [Parameter(Mandatory = $false)]
    [switch]$KeepArtifacts,

    [Parameter(Mandatory = $false)]
    [switch]$SkipCodegen
)

$ErrorActionPreference = "Stop"

function Join-PathMany {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Base,

        [Parameter(Mandatory = $true)]
        [string[]]$Parts
    )

    $path = $Base
    foreach ($part in $Parts) {
        $path = Join-Path $path $part
    }
    return $path
}

function Resolve-ProjectRoot {
    $root = Resolve-Path (Join-PathMany -Base $PSScriptRoot -Parts @("..", ".."))
    return $root.Path
}

function Find-SwiplExe {
    $cmd = Get-Command swipl -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $cmd = Get-Command swipl.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $locations = @(
        "C:\\Program Files\\swipl\\bin\\swipl.exe",
        "C:\\Program Files (x86)\\swipl\\bin\\swipl.exe",
        "C:\\swipl\\bin\\swipl.exe",
        "$env:LocalAppData\\swipl\\bin\\swipl.exe",
        "$env:ProgramFiles\\swipl\\bin\\swipl.exe"
    )

    foreach ($candidate in $locations) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    throw "swipl not found on PATH or in common install locations."
}

function Assert-DotnetAvailable {
    $cmd = Get-Command dotnet -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "dotnet not found on PATH."
    }
}

function Normalize-Rows {
    param([string[]]$Lines)
    return @(
        $Lines |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -ne "" } |
            Sort-Object
    )
}

function Invoke-NativeLogged {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $false)]
        [string[]]$ArgumentList = @(),

        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,

        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"

    foreach ($path in @($stdoutPath, $stderrPath, $LogPath)) {
        if (Test-Path $path) {
            Remove-Item -Force $path
        }
    }

    $proc = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $stdoutLines = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath } else { @() }
    $stderrLines = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath } else { @() }
    Set-Content -Path $LogPath -Value @($stdoutLines + $stderrLines)

    return $proc.ExitCode
}

$projectRoot = Resolve-ProjectRoot

$outputPath = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDir))
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

# Keep dotnet/NuGet writes inside the workspace to avoid permission issues.
$dotnetHome = Join-Path $outputPath ".dotnet_home"
$nugetRoot = Join-Path $outputPath ".nuget"
$nugetPackages = Join-Path $nugetRoot "packages"
$nugetHttpCache = Join-Path $nugetRoot "http-cache"

New-Item -ItemType Directory -Force -Path $dotnetHome | Out-Null
New-Item -ItemType Directory -Force -Path $nugetPackages | Out-Null
New-Item -ItemType Directory -Force -Path $nugetHttpCache | Out-Null

$env:DOTNET_CLI_HOME = $dotnetHome
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
$env:DOTNET_NOLOGO = "1"
$env:NUGET_PACKAGES = $nugetPackages
$env:NUGET_HTTP_CACHE_PATH = $nugetHttpCache

if (-not $SkipCodegen) {
    $generatedDirs = Get-ChildItem -Path $outputPath -Directory -Filter "csharp_query_*" -ErrorAction SilentlyContinue
    if ($generatedDirs) {
        $generatedDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    Assert-DotnetAvailable
    $swipl = Find-SwiplExe

    $env:SKIP_CSHARP_EXECUTION = "1"
    $env:CSHARP_QUERY_OUTPUT_DIR = $outputPath
    $env:CSHARP_QUERY_KEEP_ARTIFACTS = "1"

    $init = Join-Path $projectRoot "init.pl"
    $testFile = Join-PathMany -Base $projectRoot -Parts @("tests", "core", "test_csharp_query_target.pl")

    & $swipl -q -f $init -s $testFile -g "test_csharp_query_target:test_csharp_query_target" -t halt -- `
        --csharp-query-output $outputPath --csharp-query-keep
    if ($LASTEXITCODE -ne 0) {
        throw "SWI-Prolog C# query target test suite failed (exit code $LASTEXITCODE)."
    }
}

Assert-DotnetAvailable

$projects = Get-ChildItem -Path $outputPath -Directory -Filter $ProjectFilter | Where-Object {
    Test-Path (Join-Path $_.FullName "expected_rows.txt")
}

if (-not $projects) {
    throw "No generated query projects found in '$outputPath' (expected directories containing expected_rows.txt)."
}

$failures = 0
foreach ($project in $projects) {
    $dir = $project.FullName
    Write-Host "=== $($project.Name) ==="

    $buildLogName = "_dotnet_build.log"
    $buildLogPath = Join-Path $dir $buildLogName
    $buildExitCode = Invoke-NativeLogged -FilePath "dotnet" -ArgumentList @("build", "--nologo") -WorkingDirectory $dir -LogPath $buildLogPath
    if ($buildExitCode -ne 0) {
        $failures++
        Write-Host "FAIL (build)" -ForegroundColor Red
        if (Test-Path $buildLogPath) {
            Get-Content -Path $buildLogPath | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        }
        continue
    }

    $binDir = Join-PathMany -Base $dir -Parts @("bin", "Debug", "net9.0")
    if (-not (Test-Path $binDir)) {
        throw "Build output dir not found: $binDir"
    }

    $dll = Get-ChildItem -Path $binDir -File -Filter "*.dll" | Where-Object { $_.Name -notlike "*.deps.dll" } | Select-Object -First 1
    $exe = Get-ChildItem -Path $binDir -File -Filter "*.exe" | Select-Object -First 1

    $runLogName = "_query_output.log"
    $runLogPath = Join-Path $dir $runLogName
    # Prefer executing the project dll via `dotnet` so Windows MAX_PATH limitations don't break local runs.
    if ($dll) {
        $runExitCode = Invoke-NativeLogged -FilePath "dotnet" -ArgumentList @($dll.FullName) -WorkingDirectory $dir -LogPath $runLogPath
    } elseif ($exe) {
        $runExitCode = Invoke-NativeLogged -FilePath $exe.FullName -WorkingDirectory $dir -LogPath $runLogPath
    } else {
        throw "No executable or dll found in $binDir"
    }

    if ($runExitCode -ne 0) {
        $failures++
        Write-Host "FAIL (run)" -ForegroundColor Red
        if (Test-Path $runLogPath) {
            Get-Content -Path $runLogPath | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        }
        continue
    }

    $actual = Normalize-Rows -Lines (Get-Content -Path $runLogPath)
    $expectedPath = Join-Path $dir "expected_rows.txt"
    $expected = Normalize-Rows -Lines (Get-Content -Path $expectedPath)

    $match = ($actual.Count -eq $expected.Count)
    if ($match) {
        for ($i = 0; $i -lt $actual.Count; $i++) {
            if ($actual[$i] -ne $expected[$i]) {
                $match = $false
                break
            }
        }
    }

    if ($match) {
        Write-Host "PASS"
    } else {
        $failures++
        Write-Host "FAIL"
        Write-Host "  Expected:" -ForegroundColor Yellow
        $expected | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
        Write-Host "  Actual:" -ForegroundColor Yellow
        $actual | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    }
}

if (-not $KeepArtifacts) {
    Get-ChildItem -Path $outputPath -Directory -Filter "csharp_query_*" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) {
    throw "$failures project(s) failed."
}

Write-Host "All query runtime smoke tests passed."
