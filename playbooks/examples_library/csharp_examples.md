---
file_type: UnifyWeaver Example Library
---
# C# Examples for UnifyWeaver

## `unifyweaver.execution.csharp_fibonacci`

> [!example-record]
> id: unifyweaver.execution.csharp_fibonacci
> name: C# Fibonacci Example Execution (Bash)
> platform: bash

This record contains a bash script to compile and run a recursive Fibonacci example using the `csharp_query` target.

```bash
#!/bin/bash
set -e

# Create a temporary directory for the project
TMP_DIR="tmp/csharp_fib_project"
mkdir -p $TMP_DIR

# Define and write the Prolog code for Fibonacci
cat > tmp/fib_csharp.pl <<'EOF'
:- multifile fib/2.
:- dynamic fib/2.

fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
EOF

# Compile the Prolog code to C#
echo "Compiling Prolog to C#..."

# Write the SWIPL goal to a temporary file
cat > tmp/swipl_fib_goal.pl <<'GOAL'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- consult('tmp/fib_csharp.pl').
:- use_module(library(csharp_stream_target)).
:- compile_predicate_to_csharp(fib/2, [unique(true)], CSharpCode),
   open('tmp/csharp_fib_project/fib.cs', write, Stream),
   write(Stream, CSharpCode),
   close(Stream).
:- halt.
GOAL

swipl -l tmp/swipl_fib_goal.pl

# Check if C# file was created
if [ ! -f "$TMP_DIR/fib.cs" ]; then
    echo "ERROR: C# file was not created."
    exit 1
fi

# Execute the C# program
echo "Executing C# program..."
(cd $TMP_DIR && \
  # Detect latest available .NET runtime
  LATEST_SDK=$(dotnet --list-sdks | awk '{print $1}' | sort -V | tail -n 1)
  if [ -z "$LATEST_SDK" ]; then
    echo "ERROR: No .NET SDK found."
    exit 1
  fi
  DOTNET_VERSION="net${LATEST_SDK%%.*}.0"
  echo "Using .NET runtime: $DOTNET_VERSION (derived from SDK $LATEST_SDK)"

  # Create .csproj file using here-document
  cat > fib.csproj <<CSPROJ
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$DOTNET_VERSION</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
CSPROJ

  # Build and run the C# project
  echo "Building C# project..."
  dotnet build

  echo "Executing C# program..."
  # Use dotnet run which works on both Linux and Windows
  dotnet run --no-build
)

echo "Success: C# program compiled and executed successfully."
```

## `unifyweaver.execution.csharp_fibonacci_ps`

> [!example-record]
> id: unifyweaver.execution.csharp_fibonacci_ps
> name: C# Fibonacci Example Execution (PowerShell)
> platform: powershell

This record contains a PowerShell script to compile and run a recursive Fibonacci example using the `csharp_query` target.

```powershell
$ErrorActionPreference = "Stop"

# Create a temporary directory for the project
$tmpDir = "tmp/csharp_fib_project"
if (-not (Test-Path -Path $tmpDir)) {
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
}

# Define the Prolog code for Fibonacci
$prologCode = @'
:- multifile fib/2.
:- dynamic fib/2.

fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
'@

# Write the Prolog code to a file
Set-Content -Path "tmp/fib_csharp.pl" -Value $prologCode

# Compile the Prolog code to C#
Write-Host "Compiling Prolog to C#..."
$csFile = "$tmpDir/fib.cs"

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
    exit 1
}

Write-Host "Using SWI-Prolog at: $swiplPath"

$goal = "asserta(user:file_search_path(library, 'src/unifyweaver/targets')), asserta(user:file_search_path(library, 'src/unifyweaver/core')), consult('tmp/fib_csharp.pl'), use_module(library(csharp_stream_target)), compile_predicate_to_csharp(fib/2, [unique(true)], CSharpCode), open('$csFile', write, Stream), write(Stream, CSharpCode), close(Stream)."
& $swiplPath -g $goal -t halt

# Check if the C# file was created
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
Set-Content -Path "fib.csproj" -Value $csproj_content

# Build and run the C# project
Write-Host "Building C# project..."
dotnet build

Write-Host "Executing C# program..."
dotnet run --no-build

Pop-Location

Write-Host "Success: C# program compiled and executed successfully."
```

## Additional C# Examples

The following examples are documented placeholders for future implementation:

- `unifyweaver.execution.csharp_json_schema` - Generate typed JSON queries from schema hints.
- `unifyweaver.execution.csharp_xml_fragments` - Read NUL/LF-delimited XML fragments via XmlStreamReader, projecting elements/attributes (prefix + qualified keys) into rows.
- `unifyweaver.execution.csharp_xml_fragments_playbook` - Playbook for streaming XML fragments (pearltrees preset, prefixes/CDATA defaults).
- `unifyweaver.execution.litedb_xml_fixedpoint` - LiteDB fixed-point crawl of XML fragments (pearltrees defaults), with POCO + Raw document and iterative child expansion.
