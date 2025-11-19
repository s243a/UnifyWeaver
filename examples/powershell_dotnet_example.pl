:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% powershell_dotnet_example.pl - Examples of inline .NET code in PowerShell targets
%
% This file demonstrates:
% 1. Inline C# code compilation
% 2. Pre-compiled DLL caching for performance
% 3. Using .NET libraries for high-performance operations
% 4. Custom data transformations with C#

:- use_module('../src/unifyweaver/core/powershell_compiler').
:- use_module('../src/unifyweaver/core/dynamic_source_compiler').

%% ============================================
%% EXAMPLE 1: Simple String Processor
%% ============================================

%% string_reverser(+Input, -Reversed)
%  Reverse a string using inline C# code
%
%  This example shows:
%  - Basic inline C# code
%  - Simple string manipulation
%  - Inline compilation (no caching)
:- dynamic_source(
    string_reverser/2,
    [source_type(dotnet)],
    [csharp_inline('
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

        // Parameterless version for stream processing
        public string ProcessStringReverser() {
            return ProcessStringReverser(Console.ReadLine());
        }
    }
}
')]
).

%% ============================================
%% EXAMPLE 2: High-Performance Data Processor (Pre-Compiled)
%% ============================================

%% json_validator(+JsonString, -IsValid)
%  Validate JSON using .NET's Json libraries
%
%  This example shows:
%  - Pre-compilation for better performance
%  - Using .NET library (System.Text.Json)
%  - DLL caching to avoid recompilation
%  - Reference to additional assemblies
:- dynamic_source(
    json_validator/2,
    [source_type(dotnet)],
    [csharp_inline('
using System;
using System.Text.Json;

namespace UnifyWeaver.Generated.JsonValidator {
    public class JsonValidatorHandler {
        public string ProcessJsonValidator(string jsonString) {
            try {
                // Try to parse the JSON
                using (JsonDocument doc = JsonDocument.Parse(jsonString)) {
                    // If parsing succeeds, it\'s valid JSON
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
'),
    pre_compile(true),
    cache_dir('$env:TEMP/unifyweaver_examples'),
    references(['System.Text.Json'])
]).

%% ============================================
%% EXAMPLE 3: CSV Data Transformer
%% ============================================

%% csv_row_transformer(+CsvRow, -TransformedRow)
%  Transform CSV rows using custom C# logic
%
%  This example shows:
%  - Complex data transformation
%  - Multiple operations in one handler
%  - Pre-compilation for repeated use
:- dynamic_source(
    csv_row_transformer/2,
    [source_type(dotnet)],
    [csharp_inline('
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
'),
    pre_compile(true)
]).

%% ============================================
%% EXAMPLE 4: DateTime Parser with .NET
%% ============================================

%% datetime_parser(+DateString, -ParsedInfo)
%  Parse datetime strings using .NET's robust DateTime parsing
%
%  This example shows:
%  - Using .NET's DateTime capabilities
%  - Error handling in C#
%  - Formatted output
:- dynamic_source(
    datetime_parser/2,
    [source_type(dotnet)],
    [csharp_inline('
using System;
using System.Globalization;

namespace UnifyWeaver.Generated.DatetimeParser {
    public class DatetimeParserHandler {
        public string ProcessDatetimeParser(string dateString) {
            try {
                DateTime dt;
                if (DateTime.TryParse(dateString, out dt)) {
                    return $"Year={dt.Year},Month={dt.Month},Day={dt.Day},Hour={dt.Hour},Minute={dt.Minute},Second={dt.Second},DayOfWeek={dt.DayOfWeek}";
                } else {
                    return $"PARSE_FAILED: Unable to parse ''{dateString}''";
                }
            } catch (Exception ex) {
                return $"ERROR: {ex.Message}";
            }
        }
    }
}
'),
    pre_compile(true)
]).

%% ============================================
%% EXAMPLE 5: File Hash Calculator (Using System.Security)
%% ============================================

%% file_hash_calculator(+FilePath, -Sha256Hash)
%  Calculate SHA256 hash of a file using .NET crypto libraries
%
%  This example shows:
%  - File I/O in C#
%  - Cryptographic operations
%  - Using multiple .NET namespaces
:- dynamic_source(
    file_hash_calculator/2,
    [source_type(dotnet)],
    [csharp_inline('
using System;
using System.IO;
using System.Security.Cryptography;

namespace UnifyWeaver.Generated.FileHashCalculator {
    public class FileHashCalculatorHandler {
        public string ProcessFileHashCalculator(string filePath) {
            try {
                if (!File.Exists(filePath)) {
                    return $"ERROR: File not found: {filePath}";
                }

                using (var sha256 = SHA256.Create())
                using (var stream = File.OpenRead(filePath)) {
                    byte[] hash = sha256.ComputeHash(stream);
                    return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                }
            } catch (Exception ex) {
                return $"ERROR: {ex.Message}";
            }
        }
    }
}
'),
    pre_compile(true),
    references(['System.Security.Cryptography'])
]).

%% ============================================
%% TEST PREDICATES
%% ============================================

%% test_string_reverser
%  Test the string reverser example
test_string_reverser :-
    format('~n=== Testing String Reverser ===~n'),
    format('This will compile C# inline and reverse strings.~n~n'),
    % Note: Actual testing would require PowerShell execution
    format('To test manually:~n'),
    format('  1. Compile: compile_to_powershell(string_reverser/2, [source_type(dotnet)], Code)~n'),
    format('  2. Save to file: write_powershell_file(''string_reverser.ps1'', Code)~n'),
    format('  3. Run in PowerShell: ./string_reverser.ps1~n'),
    format('  4. Input: "Hello World"~n'),
    format('  5. Expected output: "dlroW olleH"~n').

%% test_json_validator
%  Test the JSON validator with pre-compilation
test_json_validator :-
    format('~n=== Testing JSON Validator (Pre-Compiled) ===~n'),
    format('This will pre-compile C# to a DLL and cache it.~n~n'),
    format('To test manually:~n'),
    format('  1. Compile: compile_to_powershell(json_validator/2, [source_type(dotnet)], Code)~n'),
    format('  2. Save to file: write_powershell_file(''json_validator.ps1'', Code)~n'),
    format('  3. Run in PowerShell: ./json_validator.ps1~n'),
    format('  4. Input valid JSON: {"name": "test", "value": 123}~n'),
    format('  5. Expected: VALID: Object~n'),
    format('  6. Input invalid JSON: {broken json}~n'),
    format('  7. Expected: INVALID: ...~n'),
    format('~nNote: Second run will use cached DLL for faster execution!~n').

%% test_all_examples
%  Run all example tests
test_all_examples :-
    format('~n╔════════════════════════════════════════════════╗~n'),
    format('║  PowerShell .NET Inline Code Examples         ║~n'),
    format('╚════════════════════════════════════════════════╝~n'),
    test_string_reverser,
    test_json_validator,
    format('~nAll examples documented. Compile and test in PowerShell!~n').

%% ============================================
%% USAGE NOTES
%% ============================================

/*

## Usage in PowerShell

### Basic Workflow:

1. **Compile the predicate to PowerShell:**
   ```prolog
   ?- compile_to_powershell(string_reverser/2, [source_type(dotnet)], Code).
   ```

2. **Save to file:**
   ```prolog
   ?- write_powershell_file('string_reverser.ps1', Code).
   ```

3. **Execute in PowerShell:**
   ```powershell
   # Inline mode - compiles on each run
   PS> "Hello World" | ./string_reverser.ps1
   dlroW olleH

   # Pre-compiled mode - compiles once, caches DLL
   PS> '{"test": 123}' | ./json_validator.ps1
   VALID: Object

   # Second run uses cached DLL (much faster!)
   PS> '{"another": "test"}' | ./json_validator.ps1
   VALID: Object
   ```

### Clear Cache:

```powershell
# For pre-compiled predicates, you can clear the cache:
PS> ./json_validator.ps1 -clear_cache
```

### Performance Comparison:

- **Inline mode**: ~500ms compilation + execution per run
- **Pre-compiled mode**:
  - First run: ~1s compilation + caching
  - Subsequent runs: ~10ms (just loading + execution)

### When to Use Pre-Compilation:

✅ **Use pre-compilation when:**
- The predicate will be called multiple times
- Compilation time is significant
- You want fast startup in production

❌ **Don't use pre-compilation when:**
- The code changes frequently during development
- The predicate is called only once
- Disk space for caching is a concern

*/
