# Compares two cache smoke JSON summaries and reports total/per-slice deltas.
#
# Usage:
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_summary_diff.ps1 `
#     -BaselineSummaryPath tmp/csharp_query_smoke_ci/cache_smoke_sequence_summary.json `
#     -CompareSummaryPath tmp/csharp_query_smoke_summary_metadata/cache_smoke_sequence_summary.json
#   pwsh -File .\scripts\testing\run_csharp_query_cache_smoke_summary_diff.ps1 `
#     -BaselineSummaryPath tmp/csharp_query_smoke_ci/cache_smoke_sequence_summary.json `
#     -CompareSummaryPath tmp/csharp_query_smoke_summary_metadata/cache_smoke_sequence_summary.json `
#     -JsonOutputPath tmp/csharp_query_smoke_summary_diff/cache_smoke_sequence_diff.json

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineSummaryPath,

    [Parameter(Mandatory = $true)]
    [string]$CompareSummaryPath,

    [Parameter(Mandatory = $false)]
    [string]$JsonOutputPath
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

function Round-Seconds {
    param(
        [Parameter(Mandatory = $false)]
        $Seconds
    )

    if ($null -eq $Seconds) {
        return $null
    }

    return [Math]::Round([double]$Seconds, 1)
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

function New-DiffField {
    param(
        [Parameter(Mandatory = $false)]
        $BaselineValue,

        [Parameter(Mandatory = $false)]
        $CompareValue,

        [Parameter(Mandatory = $false)]
        $DeltaValue = $null
    )

    return [pscustomobject]@{
        baseline = $BaselineValue
        compare = $CompareValue
        delta = $DeltaValue
    }
}

function Write-CacheSmokeDiffJson {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [psobject]$DiffData
    )

    $resolvedPath = Resolve-ProjectPath -Path $Path
    $parentDir = Split-Path -Parent $resolvedPath
    if ($parentDir) {
        New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
    }

    $DiffData | ConvertTo-Json -Depth 8 | Set-Content -Path $resolvedPath
    return $resolvedPath
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
        BaselineDurationSeconds = Round-Seconds -Seconds $baselineDuration
        CompareDurationSeconds = Round-Seconds -Seconds $compareDuration
        DurationDeltaSeconds = Round-Seconds -Seconds $durationDelta
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

$diffData = [pscustomobject][ordered]@{
    baselineSummaryPath = $baseline.DisplayPath
    compareSummaryPath = $compare.DisplayPath
    overallResult = New-DiffField `
        -BaselineValue $baseline.OverallResult `
        -CompareValue $compare.OverallResult `
        -DeltaValue $(if ($baseline.OverallResult -eq $compare.OverallResult) { "same" } else { "$($baseline.OverallResult) -> $($compare.OverallResult)" })
    generatedAtUtc = New-DiffField `
        -BaselineValue $(if ([string]::IsNullOrWhiteSpace($baseline.GeneratedAtUtc)) { $null } else { $baseline.GeneratedAtUtc }) `
        -CompareValue $(if ([string]::IsNullOrWhiteSpace($compare.GeneratedAtUtc)) { $null } else { $compare.GeneratedAtUtc })
    totalDuration = [pscustomobject]@{
        baseline = Format-DurationSeconds -Seconds $baseline.TotalDurationSeconds
        baselineSeconds = Round-Seconds -Seconds $baseline.TotalDurationSeconds
        compare = Format-DurationSeconds -Seconds $compare.TotalDurationSeconds
        compareSeconds = Round-Seconds -Seconds $compare.TotalDurationSeconds
        delta = Format-SignedDurationDelta -Seconds $totalDurationDelta
        deltaSeconds = Round-Seconds -Seconds $totalDurationDelta
    }
    projectFilter = New-DiffField `
        -BaselineValue $baseline.ProjectFilter `
        -CompareValue $compare.ProjectFilter `
        -DeltaValue $(if ($baseline.ProjectFilter -eq $compare.ProjectFilter) { "same" } else { "$($baseline.ProjectFilter) -> $($compare.ProjectFilter)" })
    outputDir = New-DiffField `
        -BaselineValue $baseline.OutputDir `
        -CompareValue $compare.OutputDir `
        -DeltaValue $(if ($baseline.OutputDir -eq $compare.OutputDir) { "same" } else { "$($baseline.OutputDir) -> $($compare.OutputDir)" })
    failureDetail = New-DiffField `
        -BaselineValue $baseline.FailureDetail `
        -CompareValue $compare.FailureDetail `
        -DeltaValue $(if (($baseline.FailureDetail ?? "") -eq ($compare.FailureDetail ?? "")) { "same" } else { "changed" })
    slices = @(
        foreach ($row in $tableRows) {
            [pscustomobject][ordered]@{
                slice = $row.Slice
                status = [pscustomobject]@{
                    baseline = $row.BaselineStatus
                    compare = $row.CompareStatus
                    delta = $row.StatusDelta
                }
                duration = [pscustomobject]@{
                    baseline = $row.BaselineDuration
                    baselineSeconds = $row.BaselineDurationSeconds
                    compare = $row.CompareDuration
                    compareSeconds = $row.CompareDurationSeconds
                    delta = $row.DurationDelta
                    deltaSeconds = $row.DurationDeltaSeconds
                }
            }
        }
    )
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
$tableRows | Format-Table Slice, BaselineStatus, CompareStatus, StatusDelta, BaselineDuration, CompareDuration, DurationDelta -AutoSize

if (-not [string]::IsNullOrWhiteSpace($JsonOutputPath)) {
    $resolvedJsonOutputPath = Write-CacheSmokeDiffJson -Path $JsonOutputPath -DiffData $diffData
    Write-Host ""
    Write-Host ("Wrote cache smoke diff JSON: {0}" -f $resolvedJsonOutputPath)
}
