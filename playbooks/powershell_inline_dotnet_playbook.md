# Playbook: Inline .NET in PowerShell Scripts

## Audience
This playbook guides coding agents in using UnifyWeaver to embed inline .NET code within PowerShell scripts for rapid prototyping and PowerShell workflow integration.

## Overview
This approach compiles C# code inline within PowerShell using `Add-Type`, enabling:
- .NET library access from PowerShell
- Strong typing and performance
- DLL caching for 138x performance improvement
- Seamless integration with PowerShell pipelines

## When to Use This Approach

### ✅ Use Inline PowerShell .NET When:
- Integrating with PowerShell workflows (automation, admin scripts)
- Need .NET libraries (JSON, XML, CSV, LiteDB, etc.)
- Rapid prototyping and iteration
- Performance-critical with caching (138x speedup)
- Windows-first, cross-platform secondary

### ❌ Use C# Code Generation Instead When:
- Need standalone executables
- Deploying to non-PowerShell environments
- Linux/Unix is the primary target
- Traditional compilation workflow preferred

See `playbooks/csharp_codegen_playbook.md` for the C# approach.

## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "powershell_inline_dotnet" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use powershell inline dotnet"
```

## Workflow

### Step 1: Define Inline .NET Source

```prolog
:- dynamic_source(
    string_reverser/2,
    [source_type(dotnet), target(powershell)],
    [csharp_inline('
using System;
using System.Linq;

namespace UnifyWeaver.Generated.StringReverser {
    public class StringReverserHandler {
        public string ProcessStringReverser(string input) {
            return new string(input.Reverse().ToArray());
        }
    }
}
'),
    pre_compile(true)  // Enable DLL caching
    ]
).
```

**Key Options**:
- `source_type(dotnet)` - Use inline .NET source
- `target(powershell)` - Generate PowerShell script
- `csharp_inline/1` - The C# code to compile
- `pre_compile(true)` - Cache compiled DLL for performance

### Step 2: Compile to PowerShell

```prolog
% Direct compilation using dotnet_source module
:- use_module(library(dotnet_source)).

% Define config and compile
:- CSharpCode = '...your code...',
   Config = [csharp_inline(CSharpCode), pre_compile(true)],
   dotnet_source:compile_source(string_reverser/2, Config, [], PowerShellCode),
   open('tmp/string_reverser.ps1', write, Stream),
   write(Stream, PowerShellCode),
   close(Stream).
```

Or using `compile_to_powershell/3` with the `output_file` option:

```prolog
:- use_module(library(powershell_compiler)).
:- compile_to_powershell(string_reverser/2,
                         [source_type(dotnet), output_file('tmp/string_reverser.ps1')],
                         _).
```

This generates a PowerShell script that:
1. Compiles the C# code using `Add-Type` (or `dotnet build` for external compile)
2. Calls the generated .NET class
3. Returns results to PowerShell

### Step 3: Execute

```bash
# First run: ~386ms (compiles C#)
pwsh tmp/string_reverser.ps1 -Key "Hello World"
# Output: dlroW olleH

# Cached run: ~2.8ms (uses cached DLL)
pwsh tmp/string_reverser.ps1 -Key "Hello World"
# Output: dlroW olleH  (138x faster!)
```

## Advanced Examples

### Example 1: JSON Validation with NuGet Packages

```prolog
:- dynamic_source(
    json_validator/2,
    [source_type(dotnet), target(powershell)],
    [csharp_inline('
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
            }
        }
    }
}
'),
    pre_compile(true),
    references(['System.Text.Json'])  // NuGet package
    ]
).
```

### Example 2: CSV Processing with LiteDB

See `playbooks/json_litedb_playbook.md` for a complete example of streaming data into a NoSQL database.

### Example 3: XML Transformation

```prolog
:- dynamic_source(
    xml_extractor/2,
    [source_type(dotnet), target(powershell)],
    [csharp_inline('
using System;
using System.Xml.Linq;
using System.Linq;

namespace UnifyWeaver.Generated.XmlExtractor {
    public class XmlExtractorHandler {
        public string ProcessXmlExtractor(string xpath) {
            var xml = XDocument.Load("data.xml");
            var elements = xml.XPathSelectElements(xpath);
            return string.Join("\\0",
                elements.Select(e => e.Value));
        }
    }
}
'),
    pre_compile(true),
    references(['System.Xml.XPath.XDocument'])
    ]
).
```

## Executable Examples

### Bash Execution

See `playbooks/examples_library/powershell_dotnet_examples.md` for:
- `unifyweaver.execution.dotnet_string_reverser_bash`
- `unifyweaver.execution.dotnet_json_validator_bash`
- `unifyweaver.execution.dotnet_csv_transformer_bash`

### PowerShell Execution

Same examples available as PowerShell executable records:
- `unifyweaver.execution.dotnet_string_reverser_ps`
- `unifyweaver.execution.dotnet_json_validator_ps`
- `unifyweaver.execution.dotnet_csv_transformer_ps`

## Performance Comparison

| Mode | First Run | Cached Run | Speedup |
|------|-----------|------------|---------|
| **Inline (no cache)** | ~386ms | ~386ms | 1x |
| **Inline (cached)** | ~386ms | **~2.8ms** | **138x** |
| **C# Compiled** | ~500ms* | ~50ms* | 10x |

*Includes `dotnet build` and `dotnet run` overhead

## Architecture

### Inline Mode (pre_compile: false)
```
Prolog → PowerShell Script → Add-Type (compile) → .NET Class → Result
         (every run compiles)
```

### Pre-compiled Mode (pre_compile: true)
```
Prolog → PowerShell Script → Add-Type (cache check) → Cached DLL → Result
         (first run compiles, subsequent runs use cache)
```

## Decision Guide

| Factor | Inline .NET | C# Codegen |
|--------|-------------|------------|
| **Deployment** | PowerShell script | Standalone .exe |
| **First Run** | 386ms | 500ms |
| **Cached Run** | 2.8ms | 50ms |
| **Dependencies** | PowerShell required | .NET runtime only |
| **Use Case** | Scripting, automation | Production apps |
| **Platform** | Windows-first | Cross-platform |

## Tips for Performance

1. **Always enable caching**: `pre_compile(true)`
2. **Batch processing**: Process multiple items per call
3. **Minimize Add-Type calls**: Reuse the same PowerShell session
4. **Profile first**: Test if .NET is needed vs pure PowerShell

## Troubleshooting

### "Add-Type: Cannot find type"
Ensure namespace matches the handler class:
```prolog
namespace UnifyWeaver.Generated.MyPredicate {
    public class MyPredicateHandler { ... }
}
```

### "Assembly already loaded"
Use a new PowerShell session or:
```powershell
Remove-Module MyModule -Force
```

### Performance Slower Than Expected
Check `pre_compile(true)` is set and DLL is being cached:
```powershell
# DLL cache location
$env:TEMP/unifyweaver_dotnet_cache/
```

## Integration with Other Playbooks

- **JSON → LiteDB**: `playbooks/json_litedb_playbook.md`
- **C# Codegen**: `playbooks/csharp_codegen_playbook.md`
- **XML Sources**: `docs/proposals/xml_source/SPECIFICATION.md`

## References

- **Implementation**: `src/unifyweaver/sources/dotnet_source.pl`
- **Examples**: `examples/powershell_dotnet_example.pl`
- **Test Results**: `TEST_RESULTS.md`
- **Codex's Guide**: `docs/guides/json_to_litedb_streaming.md`
