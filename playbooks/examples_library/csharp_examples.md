# C# Examples for UnifyWeaver

## `unifyweaver.execution.csharp_fibonacci`

> [!example-record]
> id: unifyweaver.execution.csharp_fibonacci_bash
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
cat <<'EOF' > tmp/fib_csharp.pl
:- use_module(library(unifyweaver)).

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

:- unify_definition(
    'fib_csharp',
    [
        target(csharp_query),
        entry_point(fib/2),
        output_dir('tmp/csharp_fib_project')
    ]).
EOF

# Compile the Prolog code to C#
echo "Compiling Prolog to C#..."
swipl -g "consult('init.pl'), consult('tmp/fib_csharp.pl'), build_unifyweaver_project." -t halt

# Execute the C# program
echo "Executing C# program..."
(cd $TMP_DIR && dotnet run -- 8)

echo "Success: C# program compiled and executed successfully."

# Clean up
# rm -rf $TMP_DIR
# rm tmp/fib_csharp.pl
```

## `unifyweaver.execution.csharp_fibonacci_ps`

> [!example-record]
> id: unifyweaver.execution.csharp_fibonacci_ps
> name: C# Fibonacci Example Execution (PowerShell)
> platform: powershell

This record contains a PowerShell script to compile and run a recursive Fibonacci example using the `csharp_query` target.

```powershell
# Create a temporary directory for the project
$tmpDir = "tmp/csharp_fib_project"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

# Define the Prolog code for Fibonacci
$prologCode = @'
:- use_module(library(unifyweaver)).

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

:- unify_definition(
    'fib_csharp',
    [
        target(csharp_query),
        entry_point(fib/2),
        output_dir('tmp/csharp_fib_project')
    ]).
'@

# Write the Prolog code to a file
Set-Content -Path "tmp/fib_csharp.pl" -Value $prologCode

# Compile the Prolog code to C#
Write-Host "Compiling Prolog to C#..."
swipl -g "consult('init.pl'), consult('tmp/fib_csharp.pl'), build_unifyweaver_project." -t halt

# Execute the C# program
Write-Host "Executing C# program..."
Push-Location $tmpDir
dotnet run -- 8
Pop-Location

Write-Host "Success: C# program compiled and executed successfully."

# Clean up
# Remove-Item -Recurse -Force $tmpDir
# Remove-Item -Force "tmp/fib_csharp.pl"
```
- `unifyweaver.execution.csharp_json_schema` — Generate typed JSON queries from schema hints.
- `unifyweaver.execution.csharp_xml_fragments` — Read NUL/LF-delimited XML fragments via XmlStreamReader, projecting elements/attributes (prefix + qualified keys) into rows.

- `unifyweaver.execution.csharp_xml_fragments_playbook` — Playbook for streaming XML fragments (pearltrees preset, prefixes/CDATa defaults).
-  — LiteDB fixed-point crawl of XML fragments (pearltrees defaults), with POCO + Raw document and iterative child expansion.
- `unifyweaver.execution.litedb_xml_fixedpoint` — LiteDB fixed-point crawl of XML fragments (pearltrees defaults), with POCO + Raw document and iterative child expansion.
