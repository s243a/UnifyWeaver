<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 2: Facts and Rules

This chapter covers how UnifyWeaver compiles Prolog facts and rules to PowerShell code. You'll learn how facts become PowerShell arrays with `PSCustomObject` entries, how rules become functions with joins, and how to integrate with PowerShell's object pipeline.

## Facts to PowerShell Arrays

In Prolog, facts are the foundation of knowledge representation. The PowerShell compiler transforms these facts into strongly-typed PowerShell arrays.

### Unary Facts

The simplest case is unary facts (facts with one argument):

```prolog
% Prolog facts
color(red).
color(green).
color(blue).
```

The PowerShell compiler generates:

```powershell
function color {
    param([string]$Key)

    $facts = @(
        'red'
        'green'
        'blue'
    )

    if ($Key) {
        $facts | Where-Object { $_ -eq $Key }
    } else {
        $facts
    }
}
```

**Usage:**
```powershell
# List all colors
color
# Output: red green blue

# Check if 'red' is a color
color red
# Output: red (if found, nothing if not)
```

### Binary Facts

Binary facts (two arguments) are represented using `PSCustomObject`:

```prolog
% Prolog facts
parent(anne, bob).
parent(bob, charles).
parent(bob, diana).
```

Generated PowerShell:

```powershell
function parent {
    param([string]$X, [string]$Y)

    $facts = @(
        [PSCustomObject]@{ X='anne'; Y='bob' },
        [PSCustomObject]@{ X='bob'; Y='charles' },
        [PSCustomObject]@{ X='bob'; Y='diana' }
    )

    if ($X -and $Y) {
        $facts | Where-Object { $_.X -eq $X -and $_.Y -eq $Y }
    } elseif ($X) {
        $facts | Where-Object { $_.X -eq $X } | ForEach-Object { $_.Y }
    } elseif ($Y) {
        $facts | Where-Object { $_.Y -eq $Y } | ForEach-Object { $_.X }
    } else {
        $facts | ForEach-Object { "$($_.X):$($_.Y)" }
    }
}
```

**Usage:**
```powershell
# All parent relationships
parent
# Output: anne:bob  bob:charles  bob:diana

# Who are bob's children?
parent -X bob
# Output: charles diana

# Who is charles's parent?
parent -Y charles
# Output: bob

# Is anne the parent of bob?
parent -X anne -Y bob
# Returns the matching object if true
```

### Why PSCustomObject?

PowerShell's `PSCustomObject` provides several advantages:

1. **Typed Properties**: Named access via `.X` and `.Y`
2. **Pipeline-Friendly**: Works with `Where-Object`, `Select-Object`, etc.
3. **Inspection**: `Format-Table`, `Format-List` for debugging
4. **JSON Serialization**: Easy export with `ConvertTo-Json`

## Rules to Functions

Prolog rules define relationships derived from other predicates. The PowerShell compiler translates these to functions with joins.

### Simple Join (Grandparent)

```prolog
% Rule: X is a grandparent of Z if X is a parent of Y and Y is a parent of Z
grandparent(X, Z) :-
    parent(X, Y),
    parent(Y, Z).
```

Generated PowerShell:

```powershell
function grandparent {
    param([string]$X, [string]$Z)

    # Get facts from both relations
    $rel1 = parent
    $rel2 = parent

    # Nested loop join: rel1.Y = rel2.X
    $results = foreach ($r1 in $rel1) {
        foreach ($r2 in $rel2) {
            if ($r1.Y -eq $r2.X) {
                [PSCustomObject]@{
                    X = $r1.X
                    Z = $r2.Y
                }
            }
        }
    }

    if ($X -and $Z) {
        $results | Where-Object { $_.X -eq $X -and $_.Z -eq $Z }
    } elseif ($X) {
        $results | Where-Object { $_.X -eq $X } | ForEach-Object { $_.Z }
    } elseif ($Z) {
        $results | Where-Object { $_.Z -eq $Z } | ForEach-Object { $_.X }
    } else {
        $results | ForEach-Object { "$($_.X):$($_.Z)" }
    }
}
```

### Understanding the Join Pattern

The nested loop join implements Prolog's unification:

```
parent(X, Y), parent(Y, Z)
         │          │
         └──────────┘
            Y must match
```

In PowerShell:
```powershell
if ($r1.Y -eq $r2.X)  # Y from first parent = X of second parent
```

This is the fundamental pattern for multi-goal rule bodies.

### Rules with Negation

The PowerShell compiler supports negation using `\+` or `not/1`:

```prolog
% People who are parents but not grandparents
parent_only(X) :-
    parent(X, _),
    \+ grandparent(X, _).
```

This generates a hashtable-based exclusion check for efficiency:

```powershell
function parent_only {
    param([string]$X)

    $facts = parent

    # Load negated facts into hashtable for O(1) lookup
    $grandparent_set = @{}
    foreach ($gp in grandparent) {
        $grandparent_set["$($gp.X):$($gp.Z)"] = $true
    }

    $results = foreach ($f in $facts) {
        $x = $f.X
        # Negation check: skip if X is a grandparent
        $negKey = "$x:*"
        if (-not ($grandparent_set.Keys | Where-Object { $_ -like $negKey })) {
            [PSCustomObject]@{ X = $x }
        }
    }

    # ... filtering logic
}
```

## Pipeline Integration

PowerShell's object pipeline is a powerful feature. Generated functions integrate seamlessly.

### Chaining Functions

```powershell
# Find all grandchildren of anne
parent -X anne | ForEach-Object { parent -X $_ }

# Or use the generated function directly
grandparent -X anne
```

### Pipeline Input

Functions accept pipeline input via `ValueFromPipeline`:

```powershell
function custom_query {
    param(
        [Parameter(ValueFromPipeline=$true)]
        $InputData
    )

    process {
        # Process each item from pipeline
        parent -X $InputData
    }
}

# Usage
@('anne', 'bob') | custom_query
```

### Output to Other Cmdlets

Results flow naturally to standard PowerShell cmdlets:

```powershell
# Export parent facts to CSV
parent | Export-Csv -Path parents.csv

# Count relationships
parent | Measure-Object

# Group by first argument
parent | Group-Object X

# Format as table
grandparent | Format-Table -AutoSize
```

## Step-by-Step: Compiling Facts and Rules

Let's walk through the complete process.

### Step 1: Define Facts in Prolog

```prolog
% In SWI-Prolog
?- assertz(employee(alice, engineering)).
?- assertz(employee(bob, sales)).
?- assertz(employee(carol, engineering)).

?- assertz(manager(dave, engineering)).
?- assertz(manager(eve, sales)).
```

### Step 2: Define a Rule

```prolog
% Rule: X reports to Y if X is an employee in department D and Y manages D
reports_to(X, Y) :-
    employee(X, D),
    manager(Y, D).
```

### Step 3: Compile to PowerShell

```prolog
?- ['education/init'].
?- use_module(unifyweaver(core/powershell_compiler)).

% Compile facts
?- compile_to_powershell(employee/2, [], EmployeeCode),
   open('output/employee.ps1', write, S1),
   write(S1, EmployeeCode),
   close(S1).

?- compile_to_powershell(manager/2, [], ManagerCode),
   open('output/manager.ps1', write, S2),
   write(S2, ManagerCode),
   close(S2).

% Compile rule
?- compile_to_powershell(reports_to/2, [], ReportsCode),
   open('output/reports_to.ps1', write, S3),
   write(S3, ReportsCode),
   close(S3).
```

### Step 4: Run in PowerShell

```powershell
# Load the generated scripts
. .\output\employee.ps1
. .\output\manager.ps1
. .\output\reports_to.ps1

# Query: Who reports to dave?
reports_to -Y dave
# Output: alice carol

# Query: Who does bob report to?
reports_to -X bob
# Output: eve
```

## Compilation Modes Revisited

The PowerShell compiler has two modes for facts and rules:

### Pure PowerShell Mode

Native PowerShell code with no external dependencies:

```prolog
?- compile_to_powershell(parent/2, [powershell_mode(pure)], Code).
```

**Pros:**
- No Bash required
- Native PowerShell objects
- Better IDE integration

**Cons:**
- Some complex patterns not yet supported

### BaaS Mode (Bash-as-a-Service)

Wraps Bash code in PowerShell:

```prolog
?- compile_to_powershell(parent/2, [powershell_mode(baas)], Code).
```

**Pros:**
- All Bash templates work immediately
- Battle-tested code generation

**Cons:**
- Requires Bash on the system
- Text-based rather than object-based

The compiler auto-selects the best mode, but you can override with the `powershell_mode` option.

## Performance Considerations

### Array Size

For large fact sets, consider:

1. **Lazy Loading**: Load facts from file at runtime
2. **Indexing**: Pre-compute hashtables for frequent lookups
3. **Streaming**: Process facts one at a time rather than loading all

### Join Optimization

The current nested loop join is O(n*m). For large datasets:

```powershell
# Pre-build index for faster joins
$parentByChild = @{}
foreach ($p in parent) {
    if (-not $parentByChild.ContainsKey($p.Y)) {
        $parentByChild[$p.Y] = @()
    }
    $parentByChild[$p.Y] += $p.X
}
```

Future versions may include automatic index generation.

## What's Next?

In Chapter 3, you'll learn about generating advanced PowerShell cmdlets with:
- Proper `[CmdletBinding()]` attributes
- Verbose and Debug output
- Parameter validation
- Error handling

## Quick Reference

### Compilation API

| Predicate | Description |
|-----------|-------------|
| `compile_to_powershell(Pred/Arity, [], Code)` | Compile with defaults |
| `compile_to_powershell(Pred/Arity, [powershell_mode(pure)], Code)` | Force pure PowerShell |
| `compile_to_powershell(Pred/Arity, [output_file(Path)], _)` | Write to file |

### Generated Function Patterns

| Prolog | PowerShell |
|--------|------------|
| Unary fact `p(x).` | Array of strings |
| Binary fact `p(x,y).` | Array of PSCustomObject |
| Rule with join | Nested foreach loops |
| Rule with negation | Hashtable exclusion |

### Pipeline Cmdlets

```powershell
parent | Where-Object { $_.X -eq 'bob' }  # Filter
parent | Select-Object Y                   # Project
parent | Sort-Object X                     # Order
parent | Group-Object X                    # Aggregate
parent | Export-Csv parents.csv            # Export
```

---

## Navigation

[← Previous: Chapter 1: Introduction](01_introduction) | [📖 Book 12: PowerShell Target](./) | [Next: Chapter 3: Cmdlet Generation →](03_cmdlet_generation)
