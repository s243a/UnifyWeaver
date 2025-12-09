---
file_type: UnifyWeaver Example Library
---
# PowerShell .NET Inline Examples for UnifyWeaver

## `unifyweaver.execution.dotnet_string_reverser_bash`

> [!example-record]
> id: unifyweaver.execution.dotnet_string_reverser_bash
> name: .NET String Reverser Example Execution (Bash)
> platform: bash

This record demonstrates compiling inline C# code to a PowerShell script using the `dotnet_source` plugin.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/dotnet_string_reverser"
mkdir -p $TMP_DIR

# Write the Prolog compilation script
cat > $TMP_DIR/compile_string_reverser.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/sources')).

:- use_module(library(dotnet_source)).

main :-
    CSharpCode = '
using System;

namespace UnifyWeaver.Generated.StringReverser {
    public class StringReverserHandler {
        public string ProcessStringReverser(string input) {
            if (string.IsNullOrEmpty(input)) {
                return input;
            }
            char[] charArray = input.ToCharArray();
            Array.Reverse(charArray);
            return new string(charArray);
        }
    }
}
',
    Config = [csharp_inline(CSharpCode)],
    Options = [],

    dotnet_source:compile_source(string_reverser/2, Config, Options, PowerShellCode),

    open('tmp/dotnet_string_reverser/string_reverser.ps1', write, Stream),
    write(Stream, PowerShellCode),
    close(Stream),

    format('PowerShell script generated successfully.~n').

:- main.
:- halt.
PROLOG

echo "Compiling Prolog to PowerShell..."
swipl -l $TMP_DIR/compile_string_reverser.pl

# Check if PowerShell file was created
if [ ! -f "$TMP_DIR/string_reverser.ps1" ]; then
    echo "ERROR: PowerShell file was not created."
    exit 1
fi

echo "PowerShell script generated successfully."
echo ""
echo "=== Generated PowerShell Script (first 50 lines) ==="
head -50 $TMP_DIR/string_reverser.ps1
echo ""
echo "=== End of Preview ==="
echo ""
echo "To test with PowerShell:"
echo "  pwsh $TMP_DIR/string_reverser.ps1"
echo "  echo 'Hello World' | pwsh $TMP_DIR/string_reverser.ps1"
```

## `unifyweaver.execution.dotnet_string_reverser_ps`

> [!example-record]
> id: unifyweaver.execution.dotnet_string_reverser_ps
> name: .NET String Reverser Example Execution (PowerShell)
> platform: powershell

PowerShell version of the string reverser example.

```powershell
$ErrorActionPreference = "Stop"

# Create temporary directory
$tmpDir = "tmp/dotnet_string_reverser"
if (-not (Test-Path -Path $tmpDir)) {
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
}

# Write the Prolog compilation script
$prologScript = @'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/sources')).

:- use_module(library(dotnet_source)).

main :-
    CSharpCode = '
using System;

namespace UnifyWeaver.Generated.StringReverser {
    public class StringReverserHandler {
        public string ProcessStringReverser(string input) {
            if (string.IsNullOrEmpty(input)) {
                return input;
            }
            char[] charArray = input.ToCharArray();
            Array.Reverse(charArray);
            return new string(charArray);
        }
    }
}
',
    Config = [csharp_inline(CSharpCode)],
    Options = [],

    dotnet_source:compile_source(string_reverser/2, Config, Options, PowerShellCode),

    open('tmp/dotnet_string_reverser/string_reverser.ps1', write, Stream),
    write(Stream, PowerShellCode),
    close(Stream),

    format('PowerShell script generated successfully.~n').

:- main.
:- halt.
'@

Set-Content -Path "$tmpDir/compile_string_reverser.pl" -Value $prologScript

Write-Host "Compiling Prolog to PowerShell..."
swipl -l "$tmpDir/compile_string_reverser.pl"

# Check if PowerShell file was created
if (-not (Test-Path -Path "$tmpDir/string_reverser.ps1")) {
    Write-Host "ERROR: PowerShell file was not created."
    exit 1
}

Write-Host "PowerShell script generated successfully."
Write-Host ""
Write-Host "=== Testing the generated script ==="
Write-Host "Input: 'Hello World'"
$result = "Hello World" | & "$tmpDir/string_reverser.ps1"
Write-Host "Output: $result"
Write-Host ""
Write-Host "Expected: 'dlroW olleH'"
```

## `unifyweaver.execution.dotnet_json_validator_bash`

> [!example-record]
> id: unifyweaver.execution.dotnet_json_validator_bash
> name: .NET JSON Validator Example Execution (Bash)
> platform: bash

This record demonstrates pre-compiled JSON validation using System.Text.Json.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/dotnet_json_validator"
mkdir -p $TMP_DIR

# Write the Prolog compilation script
cat > $TMP_DIR/compile_json_validator.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/sources')).

:- use_module(library(dotnet_source)).

main :-
    CSharpCode = '
using System;
using System.Text.Json;

namespace UnifyWeaver.Generated.JsonValidator {
    public class JsonValidatorHandler {
        public string ProcessJsonValidator(string jsonString) {
            try {
                using (JsonDocument doc = JsonDocument.Parse(jsonString)) {
                    return $"VALID: {doc.RootElement.ValueKind}";
                }
            } catch (JsonException ex) {
                return $"INVALID: {ex.Message}";
            } catch (Exception ex) {
                return $"ERROR: {ex.Message}";
            }
        }
    }
}
',
    Config = [csharp_inline(CSharpCode), pre_compile(true), references(['System.Text.Json'])],
    Options = [],

    dotnet_source:compile_source(json_validator/2, Config, Options, PowerShellCode),

    open('tmp/dotnet_json_validator/json_validator.ps1', write, Stream),
    write(Stream, PowerShellCode),
    close(Stream),

    format('PowerShell script generated successfully.~n').

:- main.
:- halt.
PROLOG

echo "Compiling Prolog to PowerShell..."
swipl -l $TMP_DIR/compile_json_validator.pl

# Check if PowerShell file was created
if [ ! -f "$TMP_DIR/json_validator.ps1" ]; then
    echo "ERROR: PowerShell file was not created."
    exit 1
fi

echo "PowerShell script generated successfully."
echo ""
echo "To test with PowerShell:"
echo "  echo '{\"name\": \"test\", \"value\": 123}' | pwsh $TMP_DIR/json_validator.ps1"
echo "  # Expected: VALID: Object"
echo ""
echo "  echo '{broken json}' | pwsh $TMP_DIR/json_validator.ps1"
echo "  # Expected: INVALID: ..."
```

## `unifyweaver.execution.dotnet_json_validator_ps`

> [!example-record]
> id: unifyweaver.execution.dotnet_json_validator_ps
> name: .NET JSON Validator Example Execution (PowerShell)
> platform: powershell

PowerShell version of the JSON validator with pre-compilation.

```powershell
$ErrorActionPreference = "Stop"

# Create temporary directory
$tmpDir = "tmp/dotnet_json_validator"
if (-not (Test-Path -Path $tmpDir)) {
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
}

# Write the Prolog compilation script
$prologScript = @'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/sources')).

:- use_module(library(dotnet_source)).

main :-
    CSharpCode = '
using System;
using System.Text.Json;

namespace UnifyWeaver.Generated.JsonValidator {
    public class JsonValidatorHandler {
        public string ProcessJsonValidator(string jsonString) {
            try {
                using (JsonDocument doc = JsonDocument.Parse(jsonString)) {
                    return $"VALID: {doc.RootElement.ValueKind}";
                }
            } catch (JsonException ex) {
                return $"INVALID: {ex.Message}";
            } catch (Exception ex) {
                return $"ERROR: {ex.Message}";
            }
        }
    }
}
',
    Config = [csharp_inline(CSharpCode), pre_compile(true), references([''System.Text.Json''])],
    Options = [],

    dotnet_source:compile_source(json_validator/2, Config, Options, PowerShellCode),

    open('tmp/dotnet_json_validator/json_validator.ps1', write, Stream),
    write(Stream, PowerShellCode),
    close(Stream),

    format('PowerShell script generated successfully.~n').

:- main.
:- halt.
'@

Set-Content -Path "$tmpDir/compile_json_validator.pl" -Value $prologScript

Write-Host "Compiling Prolog to PowerShell..."
swipl -l "$tmpDir/compile_json_validator.pl"

# Check if PowerShell file was created
if (-not (Test-Path -Path "$tmpDir/json_validator.ps1")) {
    Write-Host "ERROR: PowerShell file was not created."
    exit 1
}

Write-Host "PowerShell script generated successfully."
Write-Host ""
Write-Host "=== Testing the generated script ==="

Write-Host "Test 1 - Valid JSON:"
$validJson = '{"name": "test", "value": 123}'
Write-Host "  Input: $validJson"
$result1 = $validJson | & "$tmpDir/json_validator.ps1"
Write-Host "  Output: $result1"
Write-Host ""

Write-Host "Test 2 - Invalid JSON:"
$invalidJson = '{broken json}'
Write-Host "  Input: $invalidJson"
$result2 = $invalidJson | & "$tmpDir/json_validator.ps1"
Write-Host "  Output: $result2"
```

## `unifyweaver.execution.dotnet_csv_transformer_bash`

> [!example-record]
> id: unifyweaver.execution.dotnet_csv_transformer_bash
> name: .NET CSV Row Transformer Example Execution (Bash)
> platform: bash

This record demonstrates CSV row transformation using inline C#.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/dotnet_csv_transformer"
mkdir -p $TMP_DIR

# Write the Prolog compilation script
cat > $TMP_DIR/compile_csv_transformer.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/sources')).

:- use_module(library(dotnet_source)).

main :-
    CSharpCode = '
using System;
using System.Linq;

namespace UnifyWeaver.Generated.CsvRowTransformer {
    public class CsvRowTransformerHandler {
        public string ProcessCsvRowTransformer(string csvRow) {
            if (string.IsNullOrWhiteSpace(csvRow)) {
                return csvRow;
            }

            // Split by comma
            string[] fields = csvRow.Split('','');

            // Transform each field:
            // 1. Trim whitespace
            // 2. Convert to uppercase
            // 3. Add index prefix
            var transformed = fields
                .Select((field, index) => $"[{index}]{field.Trim().ToUpper()}")
                .ToArray();

            // Rejoin with pipe delimiter
            return string.Join("|", transformed);
        }
    }
}
',
    Config = [csharp_inline(CSharpCode), pre_compile(true)],
    Options = [],

    dotnet_source:compile_source(csv_row_transformer/2, Config, Options, PowerShellCode),

    open('tmp/dotnet_csv_transformer/csv_transformer.ps1', write, Stream),
    write(Stream, PowerShellCode),
    close(Stream),

    format('PowerShell script generated successfully.~n').

:- main.
:- halt.
PROLOG

echo "Compiling Prolog to PowerShell..."
swipl -l $TMP_DIR/compile_csv_transformer.pl

# Check if PowerShell file was created
if [ ! -f "$TMP_DIR/csv_transformer.ps1" ]; then
    echo "ERROR: PowerShell file was not created."
    exit 1
fi

echo "PowerShell script generated successfully."
echo ""
echo "To test with PowerShell:"
echo "  echo 'alice, 30, engineer' | pwsh $TMP_DIR/csv_transformer.ps1"
echo "  # Expected: [0]ALICE|[1]30|[2]ENGINEER"
```

## `unifyweaver.execution.dotnet_csv_transformer_ps`

> [!example-record]
> id: unifyweaver.execution.dotnet_csv_transformer_ps
> name: .NET CSV Row Transformer Example Execution (PowerShell)
> platform: powershell

PowerShell version of the CSV transformer.

```powershell
$ErrorActionPreference = "Stop"

# Create temporary directory
$tmpDir = "tmp/dotnet_csv_transformer"
if (-not (Test-Path -Path $tmpDir)) {
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
}

# Write the Prolog compilation script
$prologScript = @'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/sources')).

:- use_module(library(dotnet_source)).

main :-
    CSharpCode = '
using System;
using System.Linq;

namespace UnifyWeaver.Generated.CsvRowTransformer {
    public class CsvRowTransformerHandler {
        public string ProcessCsvRowTransformer(string csvRow) {
            if (string.IsNullOrWhiteSpace(csvRow)) {
                return csvRow;
            }

            // Split by comma
            string[] fields = csvRow.Split('','');

            // Transform each field:
            // 1. Trim whitespace
            // 2. Convert to uppercase
            // 3. Add index prefix
            var transformed = fields
                .Select((field, index) => $"[{index}]{field.Trim().ToUpper()}")
                .ToArray();

            // Rejoin with pipe delimiter
            return string.Join("|", transformed);
        }
    }
}
',
    Config = [csharp_inline(CSharpCode), pre_compile(true)],
    Options = [],

    dotnet_source:compile_source(csv_row_transformer/2, Config, Options, PowerShellCode),

    open('tmp/dotnet_csv_transformer/csv_transformer.ps1', write, Stream),
    write(Stream, PowerShellCode),
    close(Stream),

    format('PowerShell script generated successfully.~n').

:- main.
:- halt.
'@

Set-Content -Path "$tmpDir/compile_csv_transformer.pl" -Value $prologScript

Write-Host "Compiling Prolog to PowerShell..."
swipl -l "$tmpDir/compile_csv_transformer.pl"

# Check if PowerShell file was created
if (-not (Test-Path -Path "$tmpDir/csv_transformer.ps1")) {
    Write-Host "ERROR: PowerShell file was not created."
    exit 1
}

Write-Host "PowerShell script generated successfully."
Write-Host ""
Write-Host "=== Testing the generated script ==="
Write-Host "Input: 'alice, 30, engineer'"
$result = "alice, 30, engineer" | & "$tmpDir/csv_transformer.ps1"
Write-Host "Output: $result"
Write-Host ""
Write-Host "Expected: '[0]ALICE|[1]30|[2]ENGINEER'"
```

## Additional Notes

### API Usage

The examples use the direct `dotnet_source:compile_source/4` API:

```prolog
dotnet_source:compile_source(Predicate/Arity, Config, Options, PowerShellCode)
```

Where:
- `Predicate/Arity` - The predicate indicator (e.g., `string_reverser/2`)
- `Config` - List containing `csharp_inline(Code)` or `csharp_file(Path)`, plus optional settings
- `Options` - Additional runtime options (usually empty)
- `PowerShellCode` - Output variable bound to the generated PowerShell script

### Configuration Options

- `csharp_inline(Code)` - Inline C# code as an atom
- `csharp_file(Path)` - Path to external C# file
- `pre_compile(true|false)` - Enable DLL caching for performance
- `references([...])` - List of assembly references
- `namespace(Atom)` - Override auto-generated namespace
- `class_name(Atom)` - Override auto-generated class name
- `method_name(Atom)` - Override auto-generated method name

### Performance

| Mode | First Run | Cached Run |
|------|-----------|------------|
| Inline (no cache) | ~500ms | ~500ms |
| Pre-compiled | ~500ms | ~10ms |
| External compile (.NET SDK) | ~1s | ~50ms |
