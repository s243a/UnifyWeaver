<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 3: Cmdlet Generation

This chapter covers generating PowerShell cmdlets from Prolog predicates. You'll learn how to create advanced functions with proper `[CmdletBinding()]` attributes, parameter validation, pipeline support, verbose output, and error handling.

## From Functions to Cmdlets

In Chapter 2, we generated basic PowerShell functions. Now we'll generate **advanced functions** (also called "script cmdlets") that behave like native PowerShell cmdlets.

### What Makes a Cmdlet?

A PowerShell cmdlet has:

1. **CmdletBinding** - Enables common parameters (`-Verbose`, `-Debug`, etc.)
2. **Parameter Attributes** - Validation, mandatory flags, pipeline binding
3. **Structured Output** - Proper object types for pipeline consumers
4. **Error Handling** - Using `Write-Error`, `$ErrorActionPreference`

### Basic vs Advanced Function

**Basic function (Chapter 2):**
```powershell
function parent {
    param([string]$X, [string]$Y)
    # ... implementation
}
```

**Advanced function (cmdlet):**
```powershell
function Get-Parent {
    [CmdletBinding()]
    param(
        [Parameter(Position=0)]
        [ValidateNotNullOrEmpty()]
        [string]$Parent,

        [Parameter(Position=1, ValueFromPipeline=$true)]
        [string]$Child
    )

    begin {
        Write-Verbose "Loading parent facts..."
        $facts = # ... load facts
    }

    process {
        Write-Verbose "Processing: Parent=$Parent, Child=$Child"
        # ... implementation
    }

    end {
        Write-Verbose "Query complete"
    }
}
```

## Generating Cmdlets

### Cmdlet Naming Convention

PowerShell uses Verb-Noun naming. The compiler maps predicates:

| Prolog Predicate | PowerShell Cmdlet |
|------------------|-------------------|
| `parent/2` | `Get-Parent` |
| `ancestor/2` | `Get-Ancestor` |
| `add_user/3` | `Add-User` |
| `remove_item/1` | `Remove-Item` |

### Compilation Options for Cmdlets

```prolog
?- compile_to_powershell(parent/2, [
       cmdlet_name('Get-Parent'),      % Override name
       cmdlet_binding(true),           % Add [CmdletBinding()]
       verbose_output(true),           % Add Write-Verbose calls
       parameter_sets(['ByParent', 'ByChild'])
   ], Code).
```

## Parameter Attributes

### Position and Mandatory

```prolog
% Prolog: query(search_term, max_results)
query(Term, Max) :- ...
```

Generates:

```powershell
function Search-Query {
    [CmdletBinding()]
    param(
        [Parameter(Position=0, Mandatory=$true)]
        [ValidateNotNullOrEmpty()]
        [string]$SearchTerm,

        [Parameter(Position=1)]
        [ValidateRange(1, 1000)]
        [int]$MaxResults = 100
    )

    # Implementation
}
```

### Pipeline Input

For predicates that accept input from a previous command:

```powershell
function Get-Ancestor {
    [CmdletBinding()]
    param(
        [Parameter(Position=0)]
        [string]$Ancestor,

        [Parameter(Position=1, ValueFromPipeline=$true, ValueFromPipelineByPropertyName=$true)]
        [Alias('Name', 'Person')]
        [string]$Descendant
    )

    process {
        # Called once per pipeline item
        if ($Descendant) {
            # Query for this specific descendant
        }
    }
}
```

**Usage:**
```powershell
# Direct call
Get-Ancestor -Ancestor 'abraham' -Descendant 'jacob'

# Pipeline input
'jacob', 'esau' | Get-Ancestor -Ancestor 'abraham'

# From another command
Get-Person | Get-Ancestor -Ancestor 'abraham'
```

### Parameter Validation

The compiler can generate validation attributes based on Prolog constraints:

```prolog
% Prolog with constraints
age(Person, Age) :-
    person(Person),
    Age > 0,
    Age < 150.
```

Generated:

```powershell
function Get-Age {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [ValidateScript({ Test-Person $_ })]
        [string]$Person,

        [ValidateRange(1, 149)]
        [int]$Age
    )

    # Implementation
}
```

### Common Validation Attributes

| Attribute | Purpose |
|-----------|---------|
| `[ValidateNotNullOrEmpty()]` | Reject null/empty strings |
| `[ValidateRange(min, max)]` | Numeric bounds |
| `[ValidateSet('a','b','c')]` | Allowed values |
| `[ValidatePattern('^[A-Z]')]` | Regex match |
| `[ValidateScript({...})]` | Custom validation |
| `[ValidateLength(min, max)]` | String length |

## Begin/Process/End Blocks

Advanced functions support three execution phases:

### Begin Block

Runs once before processing any pipeline input:

```powershell
begin {
    Write-Verbose "Initializing..."

    # Load facts once (expensive operation)
    $script:facts = Load-Facts

    # Build indexes
    $script:index = @{}
    foreach ($f in $script:facts) {
        $script:index[$f.Key] = $f
    }
}
```

### Process Block

Runs once per pipeline item:

```powershell
process {
    Write-Verbose "Processing: $_"

    # Use prebuilt index for fast lookup
    if ($script:index.ContainsKey($_)) {
        $script:index[$_]
    }
}
```

### End Block

Runs after all pipeline input is processed:

```powershell
end {
    Write-Verbose "Cleanup..."

    # Emit aggregated results
    # Clean up resources
}
```

### Generated Example

```prolog
% Prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

```powershell
function Get-Ancestor {
    [CmdletBinding()]
    param(
        [Parameter(Position=0)]
        [string]$Ancestor,

        [Parameter(Position=1, ValueFromPipeline=$true)]
        [string]$Descendant
    )

    begin {
        Write-Verbose "[Get-Ancestor] Loading parent facts..."
        $parentFacts = Get-Parent

        Write-Verbose "[Get-Ancestor] Building ancestor closure..."
        $ancestors = @{}

        # Base case: direct parents
        foreach ($p in $parentFacts) {
            $key = "$($p.X):$($p.Y)"
            $ancestors[$key] = [PSCustomObject]@{
                Ancestor = $p.X
                Descendant = $p.Y
            }
        }

        # Fixpoint: transitive closure
        do {
            $newCount = 0
            foreach ($a in @($ancestors.Values)) {
                foreach ($p in $parentFacts) {
                    if ($a.Descendant -eq $p.X) {
                        $key = "$($a.Ancestor):$($p.Y)"
                        if (-not $ancestors.ContainsKey($key)) {
                            $ancestors[$key] = [PSCustomObject]@{
                                Ancestor = $a.Ancestor
                                Descendant = $p.Y
                            }
                            $newCount++
                        }
                    }
                }
            }
            Write-Verbose "[Get-Ancestor] Added $newCount new relationships"
        } while ($newCount -gt 0)

        Write-Verbose "[Get-Ancestor] Total: $($ancestors.Count) ancestor relationships"
    }

    process {
        if ($Ancestor -and $Descendant) {
            $key = "$Ancestor`:$Descendant"
            if ($ancestors.ContainsKey($key)) {
                $ancestors[$key]
            }
        } elseif ($Ancestor) {
            $ancestors.Values | Where-Object { $_.Ancestor -eq $Ancestor }
        } elseif ($Descendant) {
            $ancestors.Values | Where-Object { $_.Descendant -eq $Descendant }
        } else {
            $ancestors.Values
        }
    }

    end {
        Write-Verbose "[Get-Ancestor] Query complete"
    }
}
```

## Verbose and Debug Output

### Write-Verbose

Add diagnostic messages visible with `-Verbose`:

```powershell
function Get-Parent {
    [CmdletBinding()]
    param([string]$Name)

    Write-Verbose "Querying parent for: $Name"
    # ...
    Write-Verbose "Found $count results"
}
```

```powershell
Get-Parent -Name 'bob' -Verbose
```

Output:
```
VERBOSE: Querying parent for: bob
VERBOSE: Found 2 results
charles
diana
```

### Write-Debug

For developer-level diagnostics:

```powershell
function Get-Parent {
    [CmdletBinding()]
    param([string]$Name)

    Write-Debug "Index lookup key: parent:$Name"
    Write-Debug "Hash: $($Name.GetHashCode())"
    # ...
}
```

```powershell
$DebugPreference = 'Continue'
Get-Parent -Name 'bob' -Debug
```

### Compilation Option

```prolog
?- compile_to_powershell(parent/2, [
       verbose_output(true),    % Add Write-Verbose
       debug_output(true)       % Add Write-Debug
   ], Code).
```

## Error Handling

### Write-Error vs Throw

**Write-Error** - Non-terminating error (continues execution):

```powershell
function Get-Parent {
    [CmdletBinding()]
    param([string]$Name)

    if (-not (Test-PersonExists $Name)) {
        Write-Error "Person not found: $Name"
        return
    }
    # ... continue
}
```

**Throw** - Terminating error (stops execution):

```powershell
function Get-Parent {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name
    )

    if (-not (Test-PersonExists $Name)) {
        throw "Person not found: $Name"
    }
    # ... never reached if person doesn't exist
}
```

### ErrorAction Parameter

`[CmdletBinding()]` automatically enables `-ErrorAction`:

```powershell
# Stop on first error
Get-Parent -Name 'unknown' -ErrorAction Stop

# Silently continue
Get-Parent -Name 'unknown' -ErrorAction SilentlyContinue

# Ask what to do
Get-Parent -Name 'unknown' -ErrorAction Inquire
```

### Try/Catch in Generated Code

```powershell
function Get-Parent {
    [CmdletBinding()]
    param([string]$Name)

    try {
        $result = Invoke-ExpensiveOperation -Name $Name
        $result
    }
    catch [System.IO.FileNotFoundException] {
        Write-Error "Data file not found: $_"
    }
    catch {
        Write-Error "Unexpected error: $_"
    }
}
```

## Output Formatting

### Type Names for Format.ps1xml

Add type names for custom formatting:

```powershell
function Get-Parent {
    [CmdletBinding()]
    [OutputType([UnifyWeaver.ParentRelation])]
    param([string]$Name)

    process {
        $result = [PSCustomObject]@{
            Parent = $p.X
            Child = $p.Y
        }
        $result.PSObject.TypeNames.Insert(0, 'UnifyWeaver.ParentRelation')
        $result
    }
}
```

### Custom Format File

Create `UnifyWeaver.format.ps1xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Configuration>
  <ViewDefinitions>
    <View>
      <Name>ParentRelation</Name>
      <ViewSelectedBy>
        <TypeName>UnifyWeaver.ParentRelation</TypeName>
      </ViewSelectedBy>
      <TableControl>
        <TableHeaders>
          <TableColumnHeader><Label>Parent</Label></TableColumnHeader>
          <TableColumnHeader><Label>Child</Label></TableColumnHeader>
        </TableHeaders>
        <TableRowEntries>
          <TableRowEntry>
            <TableColumnItems>
              <TableColumnItem><PropertyName>Parent</PropertyName></TableColumnItem>
              <TableColumnItem><PropertyName>Child</PropertyName></TableColumnItem>
            </TableColumnItems>
          </TableRowEntry>
        </TableRowEntries>
      </TableControl>
    </View>
  </ViewDefinitions>
</Configuration>
```

Load with:
```powershell
Update-FormatData -AppendPath UnifyWeaver.format.ps1xml
```

## Complete Example: Family Tree Cmdlet

Let's generate a complete cmdlet module from our family tree predicates.

### Step 1: Define the Predicates

```prolog
% parent facts
parent(abraham, ishmael).
parent(abraham, isaac).
parent(sarah, isaac).
parent(isaac, esau).
parent(isaac, jacob).

% grandparent rule
grandparent(GP, GC) :-
    parent(GP, P),
    parent(P, GC).

% ancestor rule
ancestor(A, D) :- parent(A, D).
ancestor(A, D) :- parent(A, P), ancestor(P, D).
```

### Step 2: Compile with Cmdlet Options

```prolog
?- compile_to_powershell(parent/2, [
       cmdlet_name('Get-FamilyParent'),
       cmdlet_binding(true),
       verbose_output(true),
       output_file('FamilyTree/Get-FamilyParent.ps1')
   ], _).

?- compile_to_powershell(grandparent/2, [
       cmdlet_name('Get-FamilyGrandparent'),
       cmdlet_binding(true),
       verbose_output(true),
       output_file('FamilyTree/Get-FamilyGrandparent.ps1')
   ], _).

?- compile_to_powershell(ancestor/2, [
       cmdlet_name('Get-FamilyAncestor'),
       cmdlet_binding(true),
       verbose_output(true),
       output_file('FamilyTree/Get-FamilyAncestor.ps1')
   ], _).
```

### Step 3: Create Module Manifest

```powershell
# FamilyTree/FamilyTree.psd1
@{
    ModuleVersion = '1.0.0'
    GUID = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    Author = 'UnifyWeaver'
    Description = 'Family tree queries generated from Prolog'
    FunctionsToExport = @(
        'Get-FamilyParent',
        'Get-FamilyGrandparent',
        'Get-FamilyAncestor'
    )
    NestedModules = @(
        'Get-FamilyParent.ps1',
        'Get-FamilyGrandparent.ps1',
        'Get-FamilyAncestor.ps1'
    )
}
```

### Step 4: Use the Module

```powershell
Import-Module ./FamilyTree

# Who are abraham's descendants?
Get-FamilyAncestor -Ancestor abraham -Verbose

# Pipeline: check if these people are descendants of isaac
'jacob', 'esau', 'ishmael' | Get-FamilyAncestor -Ancestor isaac

# Find all grandparents
Get-FamilyGrandparent | Format-Table
```

## What's Next?

In Chapter 4, you'll learn about .NET integration:
- Accessing .NET types from PowerShell
- Using NuGet packages
- Inline C# code via `Add-Type`
- The `dotnet_source` plugin

## Quick Reference

### CmdletBinding Options

```powershell
[CmdletBinding(
    SupportsShouldProcess=$true,    # Enable -WhatIf, -Confirm
    DefaultParameterSetName='Set1', # Default parameter set
    PositionalBinding=$false        # Require named parameters
)]
```

### Common Parameters (Automatic with CmdletBinding)

| Parameter | Purpose |
|-----------|---------|
| `-Verbose` | Show verbose messages |
| `-Debug` | Show debug messages |
| `-ErrorAction` | Control error behavior |
| `-WarningAction` | Control warning behavior |
| `-InformationAction` | Control info messages |
| `-ErrorVariable` | Store errors in variable |
| `-OutVariable` | Store output in variable |

### Parameter Attributes Cheat Sheet

```powershell
[Parameter(
    Mandatory=$true,              # Required
    Position=0,                   # Positional index
    ValueFromPipeline=$true,      # Accept pipeline input
    ValueFromPipelineByPropertyName=$true,  # By property name
    ParameterSetName='Set1',      # Belongs to set
    HelpMessage='Enter a name'    # Help text
)]
[Alias('N', 'PersonName')]        # Alternative names
[ValidateNotNullOrEmpty()]        # Validation
[string]$Name
```

### Compilation Options

| Option | Values | Description |
|--------|--------|-------------|
| `cmdlet_name(Name)` | String | Override function name |
| `cmdlet_binding(Bool)` | true/false | Add CmdletBinding |
| `verbose_output(Bool)` | true/false | Add Write-Verbose |
| `debug_output(Bool)` | true/false | Add Write-Debug |
| `output_type(Type)` | Type name | Add OutputType attribute |

---

## Navigation

[← Previous: Chapter 2: Facts and Rules](02_facts_rules) | [📖 Book 12: PowerShell Target](./) | [Next: Chapter 4: .NET Integration →](04_dotnet_integration)
