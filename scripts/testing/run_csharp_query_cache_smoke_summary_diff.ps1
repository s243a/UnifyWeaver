# Compares two cache smoke JSON summaries and reports total/per-slice deltas.
#
# Usage:
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_summary_diff.ps1 `
#     -BaselineSummaryPath tmp/csharp_query_smoke_ci/cache_smoke_sequence_summary.json `
#     -CompareSummaryPath tmp/csharp_query_smoke_summary_metadata/cache_smoke_sequence_summary.json

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineSummaryPath,

    [Parameter(Mandatory = $true)]
    [string]$CompareSummaryPath
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Format-DurationSeconds {
    param(
        [Parameter(Mandatory = $false)]
        $Seconds
    )

    if ($null -eq $Seconds) {
        return "n/a"
    }

    return ("{0:F1}s" -f ([double]$Seconds))
}

function Format-SignedDurationDelta {
    param(
        [Parameter(Mandatory = $false)]
        $Seconds
    )

    if ($null -eq $Seconds) {
        return "n/a"
    }

    $numericSeconds = [double]$Seconds
    $sign = if ($numericSeconds -ge 0) { "+" } else { "" }
    return ("{0}{1:F1}s" -f $sign, $numericSeconds)
}

function Get-SummaryValue {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Summary,

        [Parameter(Mandatory = $true)]
        [string]$PropertyName
    )

    $property = $Summary.PSObject.Properties[$PropertyName]
    if ($null -eq $property) {
        return $null
    }

    return $property.Value
}

function Format-UtcTimestamp {
    param(
        [Parameter(Mandatory = $false)]
        $Value
    )

    if ($null -eq $Value) {
        return $null
    }

    if ($Value -is [DateTime]) {
        return $Value.ToUniversalTime().ToString("o")
    }

    if ($Value -is [DateTimeOffset]) {
        return $Value.ToUniversalTime().ToString("o")
    }

    return [string]$Value
}

function Get-SliceDurationSeconds {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Slice
    )

    $durationSeconds = Get-SummaryValue -Summary $Slice -PropertyName "durationSeconds"
    if ($null -ne $durationSeconds) {
        return [double]$durationSeconds
    }

    $duration = Get-SummaryValue -Summary $Slice -PropertyName "duration"
    if (-not [string]::IsNullOrWhiteSpace($duration)) {
        $match = [regex]::Match($duration, "^(?<seconds>-?\d+(\.\d+)?)s$")
        if ($match.Success) {
            return [double]$match.Groups["seconds"].Value
        }
    }

    return $null
}

function Get-CacheSmokeSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolvedPath = Resolve-ProjectPath -Path $Path
    $summary = Get-Content -Path $resolvedPath -Raw | ConvertFrom-Json
    $slices = @(Get-SummaryValue -Summary $summary -PropertyName "slices")
    $totalDurationSeconds = Get-SummaryValue -Summary $summary -PropertyName "totalDurationSeconds"

    if ($null -eq $totalDurationSeconds) {
        $measuredSeconds = 0.0
        $hasMeasuredSeconds = $false
        foreach ($slice in $slices) {
            $sliceSeconds = Get-SliceDurationSeconds -Slice $slice
            if ($null -ne $sliceSeconds) {
                $measuredSeconds += $sliceSeconds
                $hasMeasuredSeconds = $true
            }
        }

        $totalDurationSeconds = if ($hasMeasuredSeconds) { $measuredSeconds } else { $null }
    } else {
        $totalDurationSeconds = [double]$totalDurationSeconds
    }

    $normalizedSlices = @{}
    foreach ($slice in $slices) {
        $sliceName = [string](Get-SummaryValue -Summary $slice -PropertyName "slice")
        if ([string]::IsNullOrWhiteSpace($sliceName)) {
            continue
        }

        $normalizedSlices[$sliceName] = [pscustomobject]@{
            Slice = $sliceName
            Status = [string](Get-SummaryValue -Summary $slice -PropertyName "status")
            DurationSeconds = Get-SliceDurationSeconds -Slice $slice
            Error = Get-SummaryValue -Summary $slice -PropertyName "error"
        }
    }

    return [pscustomobject]@{
        Path = $resolvedPath
        DisplayPath = $Path
        OverallResult = [string](Get-SummaryValue -Summary $summary -PropertyName "overallResult")
        GeneratedAtUtc = Format-UtcTimestamp -Value (Get-SummaryValue -Summary $summary -PropertyName "generatedAtUtc")
        ProjectFilter = [string](Get-SummaryValue -Summary $summary -PropertyName "projectFilter")
        OutputDir = [string](Get-SummaryValue -Summary $summary -PropertyName "outputDir")
        FailureDetail = Get-SummaryValue -Summary $summary -PropertyName "failureDetail"
        TotalDurationSeconds = $totalDurationSeconds
        Slices = $normalizedSlices
    }
}

$baseline = Get-CacheSmokeSummary -Path $BaselineSummaryPath
$compare = Get-CacheSmokeSummary -Path $CompareSummaryPath

$allSliceNames = @(
    ($baseline.Slices.Keys + $compare.Slices.Keys) |
        Sort-Object -Unique
)

$tableRows = foreach ($sliceName in $allSliceNames) {
    $baselineSlice = if ($baseline.Slices.ContainsKey($sliceName)) { $baseline.Slices[$sliceName] } else { $null }
    $compareSlice = if ($compare.Slices.ContainsKey($sliceName)) { $compare.Slices[$sliceName] } else { $null }

    $baselineStatus = if ($null -ne $baselineSlice) { $baselineSlice.Status } else { "MISSING" }
    $compareStatus = if ($null -ne $compareSlice) { $compareSlice.Status } else { "MISSING" }
    $baselineDuration = if ($null -ne $baselineSlice) { $baselineSlice.DurationSeconds } else { $null }
    $compareDuration = if ($null -ne $compareSlice) { $compareSlice.DurationSeconds } else { $null }
    $durationDelta = if (($null -ne $baselineDuration) -and ($null -ne $compareDuration)) {
        $compareDuration - $baselineDuration
    } else {
        $null
    }

    [pscustomobject]@{
        Slice = $sliceName
        BaselineStatus = $baselineStatus
        CompareStatus = $compareStatus
        StatusDelta = if ($baselineStatus -eq $compareStatus) { "same" } else { "$baselineStatus -> $compareStatus" }
        BaselineDuration = Format-DurationSeconds -Seconds $baselineDuration
        CompareDuration = Format-DurationSeconds -Seconds $compareDuration
        DurationDelta = Format-SignedDurationDelta -Seconds $durationDelta
    }
}

$totalDurationDelta = if (($null -ne $baseline.TotalDurationSeconds) -and ($null -ne $compare.TotalDurationSeconds)) {
    $compare.TotalDurationSeconds - $baseline.TotalDurationSeconds
} else {
    $null
}

Write-Host "=== C# query cache smoke summary diff ==="
Write-Host ("Baseline: `"{0}`"" -f $baseline.DisplayPath)
Write-Host ("Compare:  `"{0}`"" -f $compare.DisplayPath)
Write-Host ("Overall result: {0} -> {1}" -f $baseline.OverallResult, $compare.OverallResult)

if (-not [string]::IsNullOrWhiteSpace($baseline.GeneratedAtUtc) -or -not [string]::IsNullOrWhiteSpace($compare.GeneratedAtUtc)) {
    Write-Host ("Generated at (UTC): {0} -> {1}" -f `
        $(if ([string]::IsNullOrWhiteSpace($baseline.GeneratedAtUtc)) { "n/a" } else { $baseline.GeneratedAtUtc }), `
        $(if ([string]::IsNullOrWhiteSpace($compare.GeneratedAtUtc)) { "n/a" } else { $compare.GeneratedAtUtc }))
}

Write-Host ("Total duration: {0} -> {1} ({2})" -f `
    (Format-DurationSeconds -Seconds $baseline.TotalDurationSeconds), `
    (Format-DurationSeconds -Seconds $compare.TotalDurationSeconds), `
    (Format-SignedDurationDelta -Seconds $totalDurationDelta))

if ($baseline.ProjectFilter -ne $compare.ProjectFilter) {
    Write-Host ("Project filter: `"{0}`" -> `"{1}`"" -f $baseline.ProjectFilter, $compare.ProjectFilter)
}

if ($baseline.OutputDir -ne $compare.OutputDir) {
    Write-Host ("Output dir: `"{0}`" -> `"{1}`"" -f $baseline.OutputDir, $compare.OutputDir)
}

if ($baseline.FailureDetail -or $compare.FailureDetail) {
    Write-Host ("Failure detail: {0} -> {1}" -f `
        $(if ($null -eq $baseline.FailureDetail) { "n/a" } else { $baseline.FailureDetail }), `
        $(if ($null -eq $compare.FailureDetail) { "n/a" } else { $compare.FailureDetail }))
}

Write-Host ""
$tableRows | Format-Table -AutoSize
