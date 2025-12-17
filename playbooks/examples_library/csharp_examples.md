---
file_type: UnifyWeaver Example Library
---
# C# Examples for UnifyWeaver

## `unifyweaver.execution.csharp_sum_pair`

> [!example-record]
> id: unifyweaver.execution.csharp_sum_pair
> name: C# Sum Pair Example Execution (Bash)
> platform: bash

This record contains a bash script demonstrating the `is/2` arithmetic support using the `csharp_target` query compiler.

```bash
#!/bin/bash
set -e

# Create a temporary directory for the project
TMP_DIR="tmp/csharp_sum_project"
mkdir -p "$TMP_DIR"

# Define and write the Prolog code
cat > tmp/sum_pair_csharp.pl <<'EOF'
% Facts: numeric pairs
:- multifile num_pair/2.
:- dynamic num_pair/2.

num_pair(1, 10).
num_pair(2, 20).
num_pair(3, 30).
num_pair(4, 40).

% Rule using is/2 to compute sum
sum_pair(X, Y, Sum) :-
    num_pair(X, Y),
    Sum is X + Y.
EOF

# Compile the Prolog code to C#
echo "Compiling Prolog to C#..."

# Write the SWIPL goal to a temporary file
cat > tmp/swipl_sum_goal.pl <<'GOAL'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- consult('tmp/sum_pair_csharp.pl').
:- use_module(library(csharp_target)).
:- compile_predicate_to_csharp(sum_pair/3, [unique(true)], CSharpCode),
   open('tmp/csharp_sum_project/sum_pair.cs', write, Stream),
   write(Stream, CSharpCode),
   close(Stream).
:- halt.
GOAL

swipl -l tmp/swipl_sum_goal.pl

# Check if C# file was created
if [ ! -f "$TMP_DIR/sum_pair.cs" ]; then
    echo "ERROR: C# file was not created."
    exit 1
fi

echo "C# code generated successfully."
cat "$TMP_DIR/sum_pair.cs"

echo "Success: C# program compiled successfully."
```

## `unifyweaver.execution.csharp_sum_pair_ps`

> [!example-record]
> id: unifyweaver.execution.csharp_sum_pair_ps
> name: C# Sum Pair Example Execution (PowerShell)
> platform: powershell

This record contains a PowerShell script demonstrating the `is/2` arithmetic support using the `csharp_target` query compiler.

```powershell
$ErrorActionPreference = "Stop"

# Create a temporary directory for the project
$tmpDir = "tmp/csharp_sum_project"
if (-not (Test-Path -Path $tmpDir)) {
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
}

# Define the Prolog code
$prologCode = @'
% Facts: numeric pairs
:- multifile num_pair/2.
:- dynamic num_pair/2.

num_pair(1, 10).
num_pair(2, 20).
num_pair(3, 30).
num_pair(4, 40).

% Rule using is/2 to compute sum
sum_pair(X, Y, Sum) :-
    num_pair(X, Y),
    Sum is X + Y.
'@

# Write the Prolog code to a file
Set-Content -Path "tmp/sum_pair_csharp.pl" -Value $prologCode

# Compile the Prolog code to C#
Write-Host "Compiling Prolog to C#..."
$csFile = "$tmpDir/sum_pair.cs"

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

$goal = "asserta(user:file_search_path(library, 'src/unifyweaver/targets')), asserta(user:file_search_path(library, 'src/unifyweaver/core')), consult('tmp/sum_pair_csharp.pl'), use_module(library(csharp_target)), compile_predicate_to_csharp(sum_pair/3, [unique(true)], CSharpCode), open('$csFile', write, Stream), write(Stream, CSharpCode), close(Stream)."
& $swiplPath -g $goal -t halt

# Check if the C# file was created
if (-not (Test-Path -Path $csFile)) {
    Write-Host "ERROR: C# file was not created."
    exit 1
}

Write-Host "C# code generated successfully."
Get-Content $csFile

Write-Host "Success: C# program compiled successfully."
```

## Query Mode Examples

### `unifyweaver.execution.csharp_fib_param_query`

> [!example-record]
> id: unifyweaver.execution.csharp_fib_param_query
> name: C# Parameterized Fibonacci (Query Mode) (Bash)
> platform: bash

This record demonstrates **parameterized query mode** by declaring an input mode for `fib/2` and compiling it to C# query runtime code.

```bash
#!/bin/bash
set -e

TMP_DIR="tmp/csharp_fib_query_project"
mkdir -p "$TMP_DIR"

cat > tmp/fib_query_csharp.pl <<'EOF'
:- dynamic fib/2.
:- dynamic mode/1.

% Parameterized query mode: first arg is input, second is output.
mode(fib(+, -)).

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

cat > tmp/swipl_fib_query_goal.pl <<'GOAL'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- consult('tmp/fib_query_csharp.pl').
:- use_module(library(csharp_target)).
:- compile_predicate_to_csharp(fib/2, [mode(query)], CSharpCode),
   open('tmp/csharp_fib_query_project/fib_query.cs', write, Stream),
   write(Stream, CSharpCode),
   close(Stream).
:- halt.
GOAL

swipl -l tmp/swipl_fib_query_goal.pl

if [ ! -f "$TMP_DIR/fib_query.cs" ]; then
    echo "ERROR: C# file was not created."
    exit 1
fi

echo "C# code generated successfully."
cat "$TMP_DIR/fib_query.cs"
```

### `unifyweaver.execution.csharp_fib_param_query_ps`

> [!example-record]
> id: unifyweaver.execution.csharp_fib_param_query_ps
> name: C# Parameterized Fibonacci (Query Mode) (PowerShell)
> platform: powershell

This record demonstrates **parameterized query mode** by declaring an input mode for `fib/2` and compiling it to C# query runtime code.

```powershell
$ErrorActionPreference = "Stop"

$tmpDir = "tmp/csharp_fib_query_project"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$prologCode = @'
:- dynamic fib/2.
:- dynamic mode/1.

% Parameterized query mode: first arg is input, second is output.
mode(fib(+, -)).

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

Set-Content -Path "tmp/fib_query_csharp.pl" -Value $prologCode

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

$csFile = "$tmpDir/fib_query.cs"
$goal = "asserta(user:file_search_path(library, 'src/unifyweaver/targets')), asserta(user:file_search_path(library, 'src/unifyweaver/core')), consult('tmp/fib_query_csharp.pl'), use_module(library(csharp_target)), compile_predicate_to_csharp(fib/2, [mode(query)], CSharpCode), open('$csFile', write, Stream), write(Stream, CSharpCode), close(Stream)."
& $swiplPath -g $goal -t halt

if (-not (Test-Path -Path $csFile)) {
    Write-Host "ERROR: C# file was not created."
    exit 1
}

Write-Host "C# code generated successfully."
Get-Content $csFile
```

## Generator Mode Examples

### `unifyweaver.execution.csharp_fib_generator`

> [!example-record]
> id: unifyweaver.execution.csharp_fib_generator
> name: C# Fibonacci Generator Mode (Bash)
> platform: bash

This record demonstrates generator mode with recursive arithmetic (Fibonacci). Generator mode supports recursive predicates that query mode cannot handle.

```bash
#!/bin/bash
set -e

TMP_DIR="tmp/csharp_fib_project"
mkdir -p "$TMP_DIR"

cat > tmp/fib_generator.pl <<'EOF'
% Fibonacci sequence - requires generator mode (recursive calls with computed args)
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

echo "Compiling Prolog to C# (generator mode)..."

cat > tmp/swipl_fib_goal.pl <<'GOAL'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- consult('tmp/fib_generator.pl').
:- use_module(library(csharp_target)).
:- compile_predicate_to_csharp(fib/2, [mode(generator)], CSharpCode),
   open('tmp/csharp_fib_project/fib.cs', write, Stream),
   write(Stream, CSharpCode),
   close(Stream).
:- halt.
GOAL

swipl -l tmp/swipl_fib_goal.pl

if [ ! -f "$TMP_DIR/fib.cs" ]; then
    echo "ERROR: C# file was not created."
    exit 1
fi

echo "C# code generated successfully (generator mode)."
echo "=== Generated C# code ==="
cat "$TMP_DIR/fib.cs"

echo ""
echo "Success: Fibonacci generator compiled to C#."
```

### `unifyweaver.execution.csharp_fib_generator_ps`

> [!example-record]
> id: unifyweaver.execution.csharp_fib_generator_ps
> name: C# Fibonacci Generator Mode (PowerShell)
> platform: powershell

PowerShell version of the Fibonacci generator mode example.

```powershell
$ErrorActionPreference = "Stop"

$tmpDir = "tmp/csharp_fib_project"
if (-not (Test-Path -Path $tmpDir)) {
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
}

$prologCode = @'
% Fibonacci sequence - requires generator mode (recursive calls with computed args)
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

Set-Content -Path "tmp/fib_generator.pl" -Value $prologCode

Write-Host "Compiling Prolog to C# (generator mode)..."
$csFile = "$tmpDir/fib.cs"

$swiplPath = $null
$swiplLocations = @(
    "/usr/bin/swipl",
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
    Write-Host "ERROR: Could not find swipl. Please ensure SWI-Prolog is installed."
    exit 1
}

Write-Host "Using SWI-Prolog at: $swiplPath"

$goal = "asserta(user:file_search_path(library, 'src/unifyweaver/targets')), asserta(user:file_search_path(library, 'src/unifyweaver/core')), consult('tmp/fib_generator.pl'), use_module(library(csharp_target)), compile_predicate_to_csharp(fib/2, [mode(generator)], CSharpCode), open('$csFile', write, Stream), write(Stream, CSharpCode), close(Stream)."
& $swiplPath -g $goal -t halt

if (-not (Test-Path -Path $csFile)) {
    Write-Host "ERROR: C# file was not created."
    exit 1
}

Write-Host "C# code generated successfully (generator mode)."
Write-Host "=== Generated C# code ==="
Get-Content $csFile

Write-Host "Success: Fibonacci generator compiled to C#."
```

## Additional C# Examples

The following examples are documented placeholders for future implementation:

- `unifyweaver.execution.csharp_json_schema` - Generate typed JSON queries from schema hints.
- `unifyweaver.execution.csharp_xml_fragments` - Read NUL/LF-delimited XML fragments via XmlStreamReader, projecting elements/attributes (prefix + qualified keys) into rows.
- `unifyweaver.execution.csharp_xml_fragments_playbook` - Playbook for streaming XML fragments (pearltrees preset, prefixes/CDATA defaults).
- `unifyweaver.execution.litedb_xml_fixedpoint` - LiteDB fixed-point crawl of XML fragments (pearltrees defaults), with POCO + Raw document and iterative child expansion.
