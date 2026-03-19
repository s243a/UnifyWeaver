# Runs the canonical cache-focused C# query runtime smoke sequence used by CI.
# It shares a single output directory so admission performs code generation,
# reuse reuses the generated projects, and lru performs the final cleanup pass.
#
# Usage:
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1 -OutputDir tmp/csharp_query_smoke_ci
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1 -KeepArtifacts
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1 -SummaryPath tmp/csharp_query_smoke_ci/cache_smoke_sequence_summary.md
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1 -NoSummaryOutput
#
# Notes:
# - Requires the same prerequisites as run_csharp_query_runtime_smoke.ps1.
# - The final lru slice removes generated projects unless -KeepArtifacts is set.
# - The generated markdown summary is printed to stdout unless -NoSummaryOutput is set.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "tmp/csharp_query_smoke_ci",

    [Parameter(Mandatory = $false)]
    [string]$ProjectFilter = "*",

    [Parameter(Mandatory = $false)]
    [string]$SummaryPath,

    [Parameter(Mandatory = $false)]
    [switch]$KeepArtifacts,

    [Parameter(Mandatory = $false)]
    [switch]$NoSummaryOutput
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runnerPath = Join-Path $PSScriptRoot "run_csharp_query_runtime_smoke.ps1"
$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDir))
}
$displaySummaryPath = if ([string]::IsNullOrWhiteSpace($SummaryPath)) {
    Join-Path $OutputDir "cache_smoke_sequence_summary.md"
} else {
    $SummaryPath
}
$resolvedSummaryPath = if ([System.IO.Path]::IsPathRooted($displaySummaryPath)) {
    [System.IO.Path]::GetFullPath($displaySummaryPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $displaySummaryPath))
}

function Format-Duration {
    param(
        [Parameter(Mandatory = $true)]
        [TimeSpan]$Duration
    )

    return ("{0:F1}s" -f $Duration.TotalSeconds)
}

function Write-CacheSmokeSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$DisplayOutputDir,

        [Parameter(Mandatory = $true)]
        [string]$DisplaySummaryPath,

        [Parameter(Mandatory = $true)]
        [string]$ProjectFilterValue,

        [Parameter(Mandatory = $true)]
        [bool]$KeepArtifactsAfterSequence,

        [Parameter(Mandatory = $true)]
        [object[]]$SliceResults,

        [Parameter(Mandatory = $true)]
        [bool]$HasFailure
    )

    $summaryDir = Split-Path -Parent $Path
    if ($summaryDir) {
        New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
    }

    $lines = @(
        "## C# Query Runtime Smoke",
        "",
        "- Overall result: $(if ($HasFailure) { "FAIL" } else { "PASS" })",
        "- Output dir: `"$DisplayOutputDir`"",
        "- Summary path: `"$DisplaySummaryPath`"",
        "- Project filter: `"$ProjectFilterValue`"",
        "- Keep artifacts after sequence: $(if ($KeepArtifactsAfterSequence) { "true" } else { "false" })",
        "",
        "| Slice | Status | Duration |",
        "| --- | --- | --- |"
    )

    foreach ($sliceResult in $SliceResults) {
        $lines += "| $($sliceResult.Slice) | $($sliceResult.Status) | $(Format-Duration -Duration $sliceResult.Duration) |"
    }

    $failedSlice = $SliceResults | Where-Object { $_.Status -eq "FAIL" } | Select-Object -First 1
    if ($failedSlice) {
        $lines += ""
        $lines += "Failure detail: `"$($failedSlice.Error)`""
    }

    Set-Content -Path $Path -Value $lines
}

function Invoke-CacheSmokeSlice {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Slice,

        [Parameter(Mandatory = $false)]
        [switch]$SkipCodegen,

        [Parameter(Mandatory = $false)]
        [switch]$KeepArtifactsForSlice
    )

    $runnerArgs = @{
        OutputDir = $resolvedOutputDir
        ProjectFilter = $ProjectFilter
        CacheSlice = $Slice
    }

    if ($SkipCodegen) {
        $runnerArgs.SkipCodegen = $true
    }

    if ($KeepArtifactsForSlice) {
        $runnerArgs.KeepArtifacts = $true
    }

    Write-Host "=== Running C# query cache smoke slice: $Slice ==="
    & $runnerPath @runnerArgs
}

New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null

$sliceSpecs = @(
    @{
        Slice = "admission"
        SkipCodegen = $false
        KeepArtifactsForSlice = $true
    },
    @{
        Slice = "reuse"
        SkipCodegen = $true
        KeepArtifactsForSlice = $true
    },
    @{
        Slice = "lru"
        SkipCodegen = $true
        KeepArtifactsForSlice = [bool]$KeepArtifacts
    }
)

$sliceResults = @()
$sequenceFailure = $null

foreach ($sliceSpec in $sliceSpecs) {
    $sliceStart = Get-Date
    $sliceStatus = "PASS"
    $sliceError = $null

    try {
        Invoke-CacheSmokeSlice -Slice $sliceSpec.Slice -SkipCodegen:$sliceSpec.SkipCodegen -KeepArtifactsForSlice:$sliceSpec.KeepArtifactsForSlice
    } catch {
        $sliceStatus = "FAIL"
        $sliceError = $_.Exception.Message
        $sequenceFailure = $_
    } finally {
        $sliceResults += [pscustomobject]@{
            Slice = $sliceSpec.Slice
            Status = $sliceStatus
            Duration = ((Get-Date) - $sliceStart)
            Error = $sliceError
        }
    }

    if ($sequenceFailure) {
        break
    }
}

Write-CacheSmokeSummary `
    -Path $resolvedSummaryPath `
    -DisplayOutputDir $OutputDir `
    -DisplaySummaryPath $displaySummaryPath `
    -ProjectFilterValue $ProjectFilter `
    -KeepArtifactsAfterSequence ([bool]$KeepArtifacts) `
    -SliceResults $sliceResults `
    -HasFailure ([bool]$sequenceFailure)

if (-not $NoSummaryOutput) {
    Write-Host ""
    Write-Host "=== C# query cache smoke summary ==="
    Get-Content -Path $resolvedSummaryPath | ForEach-Object {
        Write-Host $_
    }
    Write-Host ""
}

Write-Host "Wrote cache smoke summary: $resolvedSummaryPath"

if ($sequenceFailure) {
    throw $sequenceFailure.Exception
}
