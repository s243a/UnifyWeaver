# Playbook: PowerShell Binding System

## Audience
This playbook guides coding agents in using UnifyWeaver's binding system to generate PowerShell code from declarative predicate-to-function mappings.

## Overview
The binding system provides a clean abstraction layer between Prolog predicates and PowerShell cmdlets/.NET methods. Instead of writing code generation rules manually, you declare bindings and let the code generator produce the appropriate calls.

Key features:
- **52 pre-registered PowerShell bindings** across 3 categories
- **Effect annotations** (pure, io, state, throws) for semantic analysis
- **Design patterns** (pipe_transform, cmdlet_output, stdout_return)
- **Automatic cmdlet wrapper generation** with CmdletBinding, parameters, and error handling

## When to Use This Approach

### Use Binding System When:
- Mapping Prolog predicates to PowerShell cmdlets
- Generating type-safe PowerShell function wrappers
- Need semantic effect tracking (pure vs side-effecting)
- Building pipelines from declarative specifications
- Want automatic parameter validation and verbose output

### Use Inline .NET Instead When:
- Need custom C# logic (see `powershell_inline_dotnet_playbook.md`)
- Complex algorithms that don't map to existing cmdlets
- Performance-critical code requiring compilation


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "powershell_binding" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
./unifyweaver search "how to use powershell binding"


## Agent Inputs
Reference the following artifacts:
1. **Bash Executable Records** in `playbooks/examples_library/powershell_binding_examples.md`:
   - `unifyweaver.execution.ps_binding_math_bash` - .NET Math operations
   - `unifyweaver.execution.ps_binding_cmdlet_bash` - Cmdlet wrapper generation
   - `unifyweaver.execution.ps_binding_file_ops_bash` - File I/O operations
   - `unifyweaver.execution.ps_binding_pipeline_bash` - Pipeline operations
   - `unifyweaver.execution.ps_binding_test_bash` - Run binding test suite
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

### Step-by-Step Instructions

#### For Linux/macOS (bash) environment:

**Step 1: Navigate to project root**
```bash
cd /path/to/UnifyWeaver
```

**Step 2: Extract the bash script for .NET Math operations example**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.ps_binding_math_bash" \
  playbooks/examples_library/powershell_binding_examples.md \
  > tmp/run_binding_math.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_binding_math.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_binding_math.sh
```

**Expected Output**:
```
PowerShell bindings initialized.

=== Generated PowerShell Code from Bindings ===

sqrt(16) generates: [Math]::Sqrt(16)
abs(-42) generates: [Math]::Abs(-42)
round(3.7) generates: [Math]::Round(3.7)

=== Binding Information ===

sqrt/2 binding:
  Target: [Math]::Sqrt
  Inputs: [type(double)]
  Outputs: [type(double)]
  Options: [pure,deterministic,total]

Success: Binding code generation complete.
```

### Alternative: Run Cmdlet Generation Example

**Step 2b: Extract and run cmdlet generation example**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.ps_binding_cmdlet_bash" \
  playbooks/examples_library/powershell_binding_examples.md \
  > tmp/run_binding_cmdlet.sh

chmod +x tmp/run_binding_cmdlet.sh
bash tmp/run_binding_cmdlet.sh
```

This generates a complete PowerShell cmdlet wrapper function from the `test_path/1` binding, including:
- `[CmdletBinding()]` attribute
- Parameter declarations with Position and Type
- Verbose output support
- Begin/Process/End blocks

### Alternative: Run Built-in Test Suite

**Step 2c: Run the binding system's test suite**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.ps_binding_test_bash" \
  playbooks/examples_library/powershell_binding_examples.md \
  > tmp/run_binding_tests.sh

chmod +x tmp/run_binding_tests.sh
bash tmp/run_binding_tests.sh
```

## Binding System Architecture

### Core Components

```
src/unifyweaver/
├── core/
│   └── binding_registry.pl    # binding/6 predicate, registry management
└── bindings/
    ├── powershell_bindings.pl # 52 PowerShell-specific bindings
    └── binding_codegen.pl     # Code generation utilities
```

### The `binding/6` Predicate

```prolog
binding(Target, Pred, TargetName, Inputs, Outputs, Options)
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| Target | Target language | `powershell` |
| Pred | Predicate indicator | `sqrt/2` |
| TargetName | Target function/cmdlet | `'[Math]::Sqrt'` |
| Inputs | Input type specifications | `[type(double)]` |
| Outputs | Output type specifications | `[type(double)]` |
| Options | Effect and pattern annotations | `[pure, deterministic]` |

### Binding Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Core Cmdlets** | 18 | `write_output/1`, `foreach_object/2`, `where_object/2` |
| **Windows Automation** | 22 | `get_service/1`, `get_child_item/2`, `get_cim_instance/2` |
| **.NET Integration** | 12 | `sqrt/2`, `file_exists/1`, `path_combine/3` |

### Effect Annotations

| Effect | Meaning | Example Predicates |
|--------|---------|-------------------|
| `pure` | No side effects | `sqrt/2`, `round/2`, `abs/2` |
| `effect(io)` | Performs I/O | `get_child_item/2`, `file_read_all_text/2` |
| `effect(state)` | Modifies state | `set_content/2`, `start_service/1` |
| `effect(throws)` | May throw | `file_read_all_text/2`, `to_int/2` |

### Design Patterns

| Pattern | Usage | Generated Code Style |
|---------|-------|---------------------|
| `pipe_transform` | Pipeline operations | `$input | Cmdlet` |
| `cmdlet_output` | Standard output | `Cmdlet -Param $value` |
| `stdout_return` | Console output | `Write-Host $msg` |
| `exit_code_bool` | Boolean via exit | `if (Test-Path $p) { ... }` |

## Code Generation APIs

### `generate_binding_call/4`
Generate a single function call:
```prolog
generate_binding_call(powershell, sqrt/2, [16], Code).
% Code = "[Math]::Sqrt(16)"
```

### `generate_cmdlet_from_binding/4`
Generate a complete cmdlet wrapper:
```prolog
generate_cmdlet_from_binding(powershell, test_path/1, [verbose_output(true)], Code).
% Code = "function Test-Path { [CmdletBinding()] param(...) ... }"
```

### `ps_binding/5`
Query bindings (convenience predicate):
```prolog
ps_binding(sqrt/2, TargetName, Inputs, Outputs, Options).
% TargetName = '[Math]::Sqrt'
% Options = [pure, deterministic, total]
```

## Expected Outcome
- Successful execution will demonstrate binding system functionality
- Generated PowerShell code will be displayed
- Binding metadata (inputs, outputs, effects) will be shown
- Exit code 0

## Comparison with Other Approaches

| Feature | Binding System | Inline .NET | C# Codegen |
|---------|----------------|-------------|------------|
| Setup complexity | Low | Medium | High |
| Custom logic | Limited | Full | Full |
| Type safety | Declarative | C# types | C# types |
| Effect tracking | Built-in | Manual | Manual |
| Code reuse | High | Medium | Low |

## Citations
[1] playbooks/examples_library/powershell_binding_examples.md
[2] src/unifyweaver/bindings/powershell_bindings.pl
[3] src/unifyweaver/bindings/binding_codegen.pl
[4] docs/POWERSHELL_PURE_IMPLEMENTATION.md
