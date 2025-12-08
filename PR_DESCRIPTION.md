# Add Recursion and Fixpoint Mode Support to PowerShell Compiler

## Summary

Implements both procedural recursion and fixpoint (generator) mode for pure PowerShell.

## Features

### 1. Procedural Recursion
- Detects recursive predicates automatically
- Generates PowerShell recursive functions
- Handles factorial-like patterns

### 2. Generator/Fixpoint Mode
- `mode(generator)` option for fixpoint iteration
- PowerShell `Fact` class with `Key()` method
- Delta/total pattern for efficient iteration

## Usage

```prolog
% Procedural recursion
compile_to_powershell(factorial/2, [powershell_mode(pure)], Code)

% Fixpoint/generator mode
compile_to_powershell(ancestor/2, [powershell_mode(pure), mode(generator)], Code)
```

## Generated Code (Fixpoint Mode)

```powershell
class Fact {
    [string]$Pred
    [object[]]$Args
    
    [string] Key() {
        return "$($this.Pred):$($this.Args -join ':')"
    }
}

$script:total = @{}
$script:delta = @{}

function Solve-ancestor {
    Initialize-Facts
    
    do {
        $iteration++
        $script:delta = @{}
        Apply-ancestorRule $script:total
        
        foreach ($key in $script:delta.Keys) {
            $script:total[$key] = $script:delta[$key]
        }
    } while ($script:delta.Count -gt 0)
}
```

## New Predicates

| Predicate | Purpose |
|-----------|---------|
| `is_recursive_predicate_ps/2` | Detect recursive predicates |
| `compile_recursive_to_powershell/5` | Procedural recursion |
| `compile_generator_mode_powershell/5` | Fixpoint mode |
| `gather_dependencies_ps/3` | Find dependency predicates |
| `generate_fixpoint_loop_ps/2` | Generate Solve-* function |
