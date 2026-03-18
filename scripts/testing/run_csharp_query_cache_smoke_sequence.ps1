# Runs the canonical cache-focused C# query runtime smoke sequence used by CI.
# It shares a single output directory so admission performs code generation,
# reuse reuses the generated projects, and lru performs the final cleanup pass.
#
# Usage:
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1 -OutputDir tmp/csharp_query_smoke_ci
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_sequence.ps1 -KeepArtifacts
#
# Notes:
# - Requires the same prerequisites as run_csharp_query_runtime_smoke.ps1.
# - The final lru slice removes generated projects unless -KeepArtifacts is set.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "tmp/csharp_query_smoke_ci",

    [Parameter(Mandatory = $false)]
    [string]$ProjectFilter = "*",

    [Parameter(Mandatory = $false)]
    [switch]$KeepArtifacts
)

$ErrorActionPreference = "Stop"

$runnerPath = Join-Path $PSScriptRoot "run_csharp_query_runtime_smoke.ps1"

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
        OutputDir = $OutputDir
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

Invoke-CacheSmokeSlice -Slice "admission" -KeepArtifactsForSlice
Invoke-CacheSmokeSlice -Slice "reuse" -SkipCodegen -KeepArtifactsForSlice
Invoke-CacheSmokeSlice -Slice "lru" -SkipCodegen -KeepArtifactsForSlice:$KeepArtifacts
