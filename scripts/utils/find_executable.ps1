# find_executable.ps1
param (
    [string]$ExecutableName
)

$searchPaths = @(
    "C:\Strawberry\perl\bin",
    "C:\Program Files\swipl\bin",
    "C:\Program Files (x86)\swipl\bin"
    # Add other common paths here
)

$executableExtensions = @("", ".exe", ".bat", ".cmd")

foreach ($path in $searchPaths) {
    foreach ($ext in $executableExtensions) {
        $fullPath = Join-Path -Path $path -ChildPath ($ExecutableName + $ext)
        if (Test-Path -Path $fullPath -PathType Leaf) {
            Write-Output $fullPath
            exit 0
        }
    }
}

exit 1
