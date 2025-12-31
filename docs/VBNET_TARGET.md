# VB.NET Target

The VB.NET target compiles Prolog predicates to VB.NET programs for stream processing, data querying, and recursive algorithms.

## Overview

| Feature | Status |
|---------|--------|
| Fact Compilation | ✅ |
| Rule Compilation | ✅ |
| Bindings | ✅ 20+ |
| Tail Recursion | ✅ Do While |
| Linear Recursion | ✅ Dictionary |
| Mutual Recursion | ✅ |

## Quick Start

```prolog
% Load the target
?- use_module('src/unifyweaver/targets/vbnet_target').
?- init_vbnet_target.

% Compile facts
?- compile_predicate_to_vbnet(person/2, [], Code),
   write_vbnet_program(Code, 'person.vb').
```

## API Reference

### `compile_predicate_to_vbnet/3`

Compile a Prolog predicate to VB.NET.

```prolog
compile_predicate_to_vbnet(+Pred/Arity, +Options, -VBCode)
```

### `compile_facts_to_vbnet/3`

Compile facts to VB.NET with class generation.

```prolog
compile_facts_to_vbnet(+Pred, +Arity, -VBCode)
```

### `compile_tail_recursion_vbnet/3`

Compile tail-recursive predicates to iterative Do While loops.

```prolog
compile_tail_recursion_vbnet(+Pred/Arity, +Options, -VBCode)
```

### `compile_linear_recursion_vbnet/3`

Compile linear recursion with Dictionary memoization.

```prolog
compile_linear_recursion_vbnet(+Pred/Arity, +Options, -VBCode)
```

### `compile_mutual_recursion_vbnet/3`

Compile mutually recursive predicates (is_even/is_odd).

```prolog
compile_mutual_recursion_vbnet(+Predicates, +Options, -VBCode)
```

## Generated Code Patterns

### Facts → Class + List

```vb
Public Class PERSON
    Public Property Arg1 As String
    Public Property Arg2 As String
End Class

Module Facts
    Public Function GetAllPERSON() As List(Of PERSON)
        Return New List(Of PERSON) From { ... }
    End Function
End Module
```

### Tail Recursion → Do While

```vb
Do While current > 0
    accumulator += current
    current -= 1
Loop
```

### Linear Recursion → Dictionary

```vb
Private ReadOnly _memo As New Dictionary(Of Integer, Integer)

If _memo.ContainsKey(n) Then
    Return _memo(n)
End If
```

## Bindings

Initialize bindings:
```prolog
?- init_vbnet_target.
```

| Category | Examples |
|----------|----------|
| Core | `.Length`, `CInt`, `CDbl`, `CBool` |
| String | `.Split`, `String.Join`, `.Trim`, `.ToLower` |
| Math | `Math.Sqrt`, `Math.Abs`, `Math.Pow` |
| I/O | `File.Exists`, `File.ReadAllText` |

## See Also

- [CSHARP_TARGET](./targets/CSHARP_TARGET.md) - Similar C# target
- [POWERSHELL_TARGET](./POWERSHELL_TARGET.md) - PowerShell target
