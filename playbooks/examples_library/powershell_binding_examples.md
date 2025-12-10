---
file_type: UnifyWeaver Example Library
---
# PowerShell Binding System Examples for UnifyWeaver

## `unifyweaver.execution.ps_binding_math_bash`

> [!example-record]
> id: unifyweaver.execution.ps_binding_math_bash
> name: PowerShell Binding Math Example (Bash)
> platform: bash

This record demonstrates generating PowerShell code from bindings using the binding system's `generate_binding_call/4` predicate for .NET Math operations.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/ps_binding_math"
mkdir -p $TMP_DIR

# Write the Prolog script that uses bindings
cat > $TMP_DIR/binding_math.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/bindings')).

:- use_module(library(binding_registry)).
:- use_module(library(powershell_bindings)).
:- use_module(library(binding_codegen)).

main :-
    % Initialize all PowerShell bindings
    init_powershell_bindings,
    format('PowerShell bindings initialized.~n~n'),

    % Generate code for Math operations using bindings
    format('=== Generated PowerShell Code from Bindings ===~n~n'),

    % 1. Square root using [Math]::Sqrt binding
    generate_binding_call(powershell, sqrt/2, [16], SqrtCode),
    format('sqrt(16) generates: ~w~n', [SqrtCode]),

    % 2. Absolute value using [Math]::Abs binding
    generate_binding_call(powershell, abs/2, [-42], AbsCode),
    format('abs(-42) generates: ~w~n', [AbsCode]),

    % 3. Round using [Math]::Round binding
    generate_binding_call(powershell, round/2, [3.7], RoundCode),
    format('round(3.7) generates: ~w~n', [RoundCode]),

    format('~n=== Binding Information ===~n~n'),

    % Show binding details for sqrt
    (   binding(powershell, sqrt/2, TargetName, Inputs, Outputs, Options)
    ->  format('sqrt/2 binding:~n'),
        format('  Target: ~w~n', [TargetName]),
        format('  Inputs: ~w~n', [Inputs]),
        format('  Outputs: ~w~n', [Outputs]),
        format('  Options: ~w~n', [Options])
    ;   format('sqrt/2 binding not found~n')
    ),

    format('~nSuccess: Binding code generation complete.~n').

:- main.
:- halt.
PROLOG

echo "Running binding math example..."
swipl -l $TMP_DIR/binding_math.pl
```

## `unifyweaver.execution.ps_binding_cmdlet_bash`

> [!example-record]
> id: unifyweaver.execution.ps_binding_cmdlet_bash
> name: PowerShell Cmdlet Binding Example (Bash)
> platform: bash

This record demonstrates generating PowerShell cmdlet wrapper functions from bindings using `generate_cmdlet_from_binding/4`.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/ps_binding_cmdlet"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/binding_cmdlet.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/bindings')).

:- use_module(library(binding_registry)).
:- use_module(library(powershell_bindings)).
:- use_module(library(binding_codegen)).

main :-
    % Initialize bindings
    init_powershell_bindings,
    format('PowerShell bindings initialized.~n~n'),

    % Generate a full PowerShell cmdlet wrapper from the test_path binding
    format('=== Generated PowerShell Cmdlet from test_path/1 Binding ===~n~n'),
    generate_cmdlet_from_binding(powershell, test_path/1, [verbose_output(true)], CmdletCode),
    format('~w~n', [CmdletCode]),

    % Generate cmdlet call examples
    format('~n=== Cmdlet Call Examples ===~n~n'),

    generate_binding_call(powershell, get_child_item/2, ['/tmp'], GciCode),
    format('get_child_item("/tmp"): ~w~n', [GciCode]),

    generate_binding_call(powershell, get_service/2, ['BITS'], SvcCode),
    format('get_service("BITS"): ~w~n', [SvcCode]),

    generate_binding_call(powershell, write_output/1, ['Hello World'], WriteCode),
    format('write_output("Hello World"): ~w~n', [WriteCode]),

    format('~nSuccess: Cmdlet generation complete.~n').

:- main.
:- halt.
PROLOG

echo "Running cmdlet binding example..."
swipl -l $TMP_DIR/binding_cmdlet.pl

echo ""
echo "=== Generated PowerShell Script ==="
echo "The cmdlet above can be saved to a .ps1 file and used as:"
echo "  Test-Path -Path '/some/path'"
```

## `unifyweaver.execution.ps_binding_file_ops_bash`

> [!example-record]
> id: unifyweaver.execution.ps_binding_file_ops_bash
> name: PowerShell File Operations Binding Example (Bash)
> platform: bash

This record demonstrates generating PowerShell code for file operations using .NET System.IO bindings.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/ps_binding_file"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/binding_file.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/bindings')).

:- use_module(library(binding_registry)).
:- use_module(library(powershell_bindings)).
:- use_module(library(binding_codegen)).

main :-
    % Initialize bindings
    init_powershell_bindings,
    format('PowerShell bindings initialized.~n~n'),

    format('=== .NET File Operations via Bindings ===~n~n'),

    % File.Exists
    generate_binding_call(powershell, file_exists/1, ['/tmp/test.txt'], ExistsCode),
    format('file_exists("/tmp/test.txt"):~n  ~w~n~n', [ExistsCode]),

    % File.ReadAllText
    generate_binding_call(powershell, file_read_all_text/2, ['/tmp/input.txt'], ReadCode),
    format('file_read_all_text("/tmp/input.txt"):~n  ~w~n~n', [ReadCode]),

    % Path.Combine
    generate_binding_call(powershell, path_combine/3, ['/home', 'user'], CombineCode),
    format('path_combine("/home", "user"):~n  ~w~n~n', [CombineCode]),

    % Path.GetFullPath
    generate_binding_call(powershell, path_get_full/2, ['./relative/path'], FullPathCode),
    format('path_get_full("./relative/path"):~n  ~w~n~n', [FullPathCode]),

    format('=== Binding Effect Analysis ===~n~n'),

    % Show effects for file_exists
    (   binding(powershell, file_exists/1, _, _, _, Opts1),
        member(effect(E1), Opts1)
    ->  format('file_exists/1 effect: ~w~n', [E1])
    ;   format('file_exists/1: pure~n')
    ),

    % Show effects for file_write_all_text
    (   binding(powershell, file_write_all_text/2, _, _, _, Opts2),
        findall(E, member(effect(E), Opts2), Effects)
    ->  format('file_write_all_text/2 effects: ~w~n', [Effects])
    ;   true
    ),

    format('~nSuccess: File operations binding demo complete.~n').

:- main.
:- halt.
PROLOG

echo "Running file operations binding example..."
swipl -l $TMP_DIR/binding_file.pl
```

## `unifyweaver.execution.ps_binding_pipeline_bash`

> [!example-record]
> id: unifyweaver.execution.ps_binding_pipeline_bash
> name: PowerShell Pipeline Binding Example (Bash)
> platform: bash

This record demonstrates generating PowerShell pipeline operations (ForEach-Object, Where-Object, etc.) from bindings.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/ps_binding_pipeline"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/binding_pipeline.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/bindings')).

:- use_module(library(binding_registry)).
:- use_module(library(powershell_bindings)).
:- use_module(library(binding_codegen)).

main :-
    % Initialize bindings
    init_powershell_bindings,
    format('PowerShell bindings initialized.~n~n'),

    format('=== PowerShell Pipeline Operations via Bindings ===~n~n'),

    % ForEach-Object
    generate_binding_call(powershell, foreach_object/2, ['$input', '{ $_.Name }'], ForEachCode),
    format('foreach_object($input, { $_.Name }):~n  ~w~n~n', [ForEachCode]),

    % Where-Object
    generate_binding_call(powershell, where_object/2, ['$items', '{ $_.Status -eq "Running" }'], WhereCode),
    format('where_object($items, { $_.Status -eq "Running" }):~n  ~w~n~n', [WhereCode]),

    % Sort-Object
    generate_binding_call(powershell, sort_object/2, ['$data', 'Name'], SortCode),
    format('sort_object($data, Name):~n  ~w~n~n', [SortCode]),

    % Group-Object
    generate_binding_call(powershell, group_object/2, ['$items', 'Category'], GroupCode),
    format('group_object($items, Category):~n  ~w~n~n', [GroupCode]),

    % Measure-Object
    generate_binding_call(powershell, measure_object/2, ['$numbers', ['Sum', 'Average']], MeasureCode),
    format('measure_object($numbers, [Sum, Average]):~n  ~w~n~n', [MeasureCode]),

    format('=== Pattern Analysis ===~n~n'),

    % Check pipe_transform pattern
    findall(P, (
        ps_binding(P, _, _, _, Opts),
        member(pattern(pipe_transform), Opts)
    ), PipePredicates),
    format('Predicates with pipe_transform pattern:~n'),
    forall(member(PP, PipePredicates), format('  - ~w~n', [PP])),

    format('~nSuccess: Pipeline binding demo complete.~n').

:- main.
:- halt.
PROLOG

echo "Running pipeline binding example..."
swipl -l $TMP_DIR/binding_pipeline.pl
```

## `unifyweaver.execution.ps_binding_test_bash`

> [!example-record]
> id: unifyweaver.execution.ps_binding_test_bash
> name: PowerShell Binding System Test (Bash)
> platform: bash

This record runs the built-in test suites for the binding system.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/ps_binding_test"
mkdir -p $TMP_DIR

# Write the Prolog test runner
cat > $TMP_DIR/run_binding_tests.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/bindings')).

:- use_module(library(binding_registry)).
:- use_module(library(powershell_bindings)).
:- use_module(library(binding_codegen)).

main :-
    format('~n========================================~n'),
    format('   UnifyWeaver Binding System Tests~n'),
    format('========================================~n~n'),

    % Run PowerShell bindings tests
    test_powershell_bindings,

    format('~n'),

    % Run code generation tests
    test_binding_codegen,

    format('~n========================================~n'),
    format('   All Binding Tests Complete~n'),
    format('========================================~n').

:- main.
:- halt.
PROLOG

echo "Running binding system tests..."
swipl -l $TMP_DIR/run_binding_tests.pl
```

## Additional Notes

### Binding System Architecture

The binding system consists of three main components:

1. **`binding_registry.pl`** - Core `binding/6` predicate and registry management
2. **`powershell_bindings.pl`** - 52 pre-registered PowerShell bindings
3. **`binding_codegen.pl`** - Code generation from bindings

### Binding Categories

| Category | Count | Examples |
|----------|-------|----------|
| Core Cmdlets | 18 | Write-Output, ForEach-Object, Where-Object |
| Windows Automation | 22 | Get-Service, Get-ChildItem, Registry ops |
| .NET Integration | 12 | [Math]::Sqrt, [System.IO.File]::Exists |

### Effect Annotations

Bindings include effect annotations for semantic analysis:

- `pure` - No side effects, deterministic
- `effect(io)` - Performs I/O operations
- `effect(state)` - Modifies system state
- `effect(throws)` - May throw exceptions

### Design Patterns

Each binding specifies a design pattern for code generation:

- `pipe_transform` - Pipeline input/output transformation
- `cmdlet_output` - Standard cmdlet output
- `stdout_return` - Returns via stdout
- `exit_code_bool` - Boolean via exit code
