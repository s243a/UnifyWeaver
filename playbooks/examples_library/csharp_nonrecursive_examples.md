# C# Non-Recursive Examples for UnifyWeaver

## `unifyweaver.execution.csharp_grandparent_bash`

> [!example-record]
> id: unifyweaver.execution.csharp_grandparent_bash
> name: C# Grandparent Example Execution (Bash)
> platform: bash

This record contains a bash script to compile and run a non-recursive `grandparent/2` example using the `csharp_codegen` target.

```bash
#!/bin/bash
set -e

# Create a temporary directory for the project
TMP_DIR="tmp/csharp_grandparent_project"
mkdir -p $TMP_DIR

# Define and write the Prolog code to a file
echo ':- multifile parent/2.' > tmp/grandparent_csharp.pl
echo ':- dynamic parent/2.' >> tmp/grandparent_csharp.pl
echo '' >> tmp/grandparent_csharp.pl
echo 'parent(anne, bob).' >> tmp/grandparent_csharp.pl
echo 'parent(bob, charles).' >> tmp/grandparent_csharp.pl
echo 'parent(bob, diana).' >> tmp/grandparent_csharp.pl
echo '' >> tmp/grandparent_csharp.pl
echo 'grandparent(X, Z) :- parent(X, Y), parent(Y, Z).' >> tmp/grandparent_csharp.pl
echo '' >> tmp/grandparent_csharp.pl

# Compile the Prolog code to C#
echo "Compiling Prolog to C#..."
# Write the SWIPL goal to a temporary file
echo ":- asserta(user:file_search_path(library, 'src/unifyweaver/targets'))." > tmp/swipl_goal.pl
echo ":- asserta(user:file_search_path(library, 'src/unifyweaver/core'))." >> tmp/swipl_goal.pl
echo ":- consult('tmp/grandparent_csharp.pl')." >> tmp/swipl_goal.pl
echo ":- use_module(library(csharp_native_target))." >> tmp/swipl_goal.pl
echo ":- compile_predicate_to_csharp(grandparent/2, [unique(true)], CSharpCode), open('tmp/csharp_grandparent_project/grandparent.cs', write, Stream), write(Stream, CSharpCode), close(Stream)." >> tmp/swipl_goal.pl
echo ":- halt." >> tmp/swipl_goal.pl

swipl -l tmp/swipl_goal.pl

# Execute the C# program
echo "Executing C# program..."
(cd $TMP_DIR && \
  # The generated file is named after the predicate
  CS_FILE="grandparent.cs"

  # Detect latest available .NET runtime
  LATEST_SDK=$(dotnet --list-sdks | awk '{print $1}' | sort -V | tail -n 1)
  if [ -z "$LATEST_SDK" ]; then
    echo "ERROR: No .NET SDK found."
    exit 1
  fi
  DOTNET_VERSION="net${LATEST_SDK%%.*}.0"
  echo "Using .NET runtime: $DOTNET_VERSION (derived from SDK $LATEST_SDK)"

  # Create .csproj file using here-document
  cat > grandparent.csproj <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$DOTNET_VERSION</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
EOF

  # Build and run the C# project
  echo "Building C# project..."
  dotnet build

  echo "Executing C# program..."
  # Use dotnet run which works on both Linux and Windows
  dotnet run --no-build
)

echo "Success: C# program compiled and executed successfully."

# Clean up
# rm -rf $TMP_DIR
# rm tmp/grandparent_csharp.pl
```

## `unifyweaver.execution.csharp_grandparent_ps`

> [!example-record]
> id: unifyweaver.execution.csharp_grandparent_ps
> name: C# Grandparent Example Execution (PowerShell)
> platform: powershell

This record contains a PowerShell script to compile and run a non-recursive `grandparent/2` example using the `csharp_codegen` target.

```powershell
$ErrorActionPreference = "Stop"

# Create a temporary directory for the project
$tmpDir = "tmp/csharp_grandparent_project"
Write-Host "Checking for temporary directory: $tmpDir"
if (-not (Test-Path -Path $tmpDir)) {
    Write-Host "Creating temporary directory: $tmpDir"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
}

# Define the Prolog code
$prologCode = @'
:- multifile parent/2.
:- dynamic parent/2.

parent(anne, bob).
parent(bob, charles).
parent(bob, diana).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
'@

# Write the Prolog code to a file
$prologFile = "tmp/grandparent_csharp.pl"
Write-Host "Writing Prolog code to: $prologFile"
Set-Content -Path $prologFile -Value $prologCode

# Compile the Prolog code to C#
Write-Host "Compiling Prolog to C#..."
$csFile = "$tmpDir/grandparent.cs"

# Auto-detect swipl location
$swiplPath = $null
$swiplLocations = @(
    "C:\Program Files\swipl\bin\swipl.exe",
    "C:\Program Files (x86)\swipl\bin\swipl.exe",
    "$env:ProgramFiles\swipl\bin\swipl.exe",
    (Get-Command swipl -ErrorAction SilentlyContinue).Source
)

foreach ($loc in $swiplLocations) {
    if ($loc -and (Test-Path -Path $loc)) {
        $swiplPath = $loc
        break
    }
}

if (-not $swiplPath) {
    Write-Host "ERROR: Could not find swipl.exe. Please ensure SWI-Prolog is installed."
    Write-Host "Checked locations:"
    $swiplLocations | Where-Object { $_ } | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

Write-Host "Using SWI-Prolog at: $swiplPath"

$goal = "asserta(user:file_search_path(library, 'src/unifyweaver/targets')), asserta(user:file_search_path(library, 'src/unifyweaver/core')), consult('$prologFile'), use_module(library(csharp_native_target)), compile_predicate_to_csharp(grandparent/2, [unique(true)], CSharpCode), open('$csFile', write, Stream), write(Stream, CSharpCode), close(Stream)."
& $swiplPath -g $goal -t halt

# Check if the C# file was created
Write-Host "Checking for C# file: $csFile"
if (-not (Test-Path -Path $csFile)) {
    Write-Host "ERROR: C# file was not created."
    exit 1
}

# Execute the C# program
Write-Host "Executing C# program..."
Push-Location $tmpDir

# Detect latest available .NET runtime
$latestSdk = dotnet --list-sdks | ForEach-Object { $_.Split(' ')[0] } | Sort-Object -Descending | Select-Object -First 1
if (-not $latestSdk) {
    Write-Host "ERROR: No .NET SDK found."
    Pop-Location
    exit 1
}
$dotnetVersion = "net$($latestSdk.Split('.')[0]).0"
Write-Host "Using .NET runtime: $dotnetVersion (derived from SDK $latestSdk)"

# Create .csproj file
$csproj_content = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$dotnetVersion</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
"@
Set-Content -Path "grandparent.csproj" -Value $csproj_content

# Build and run the C# project
Write-Host "Building C# project..."
dotnet build

Write-Host "Executing C# program..."
# Use dotnet run which works on both Windows and Linux/WSL
dotnet run --no-build

Pop-Location

Write-Host "Success: C# program compiled and executed successfully."