# VB.NET Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: VB.NET code generation target testing

## Overview

This test plan covers the VB.NET target for UnifyWeaver, which generates Visual Basic .NET code for .NET framework integration and enterprise applications.

## Prerequisites

### System Requirements

- .NET SDK 8.0+
- VB.NET compiler (included with .NET SDK)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify .NET SDK
dotnet --version

# Check VB.NET availability
dotnet new console -lang VB -o /tmp/vbtest && rm -rf /tmp/vbtest

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_vbnet_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `class_generation` | Class/End Class | Class syntax |
| `sub_function` | Sub/Function | Method syntax |
| `property_generation` | Property | Property syntax |
| `dim_statements` | Dim declarations | Variable syntax |
| `if_then_else` | If/Then/Else | Conditional syntax |

#### 1.2 VB.NET Idioms

```bash
swipl -g "use_module('tests/core/test_vbnet_idioms'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `linq_queries` | From...Select | LINQ syntax |
| `with_blocks` | With/End With | With statement |
| `try_catch` | Try/Catch/Finally | Exception handling |
| `using_statement` | Using block | Resource management |

### 2. Compilation Tests

#### 2.1 .NET Build

```bash
./tests/integration/test_vbnet_dotnet.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `dotnet_build` | dotnet build | Build succeeds |
| `dotnet_run` | dotnet run | Execution works |
| `warnings_clean` | No warnings | Clean build |

### 3. Generated Code Structure

```vb
Imports System
Imports System.Collections.Generic
Imports System.Linq

Namespace UnifyWeaver.Generated
    Public Class Fact
        Public Property Relation As String
        Public Property Args As List(Of String)

        Public Overrides Function Equals(obj As Object) As Boolean
            Dim other = TryCast(obj, Fact)
            Return other IsNot Nothing AndAlso
                   Relation = other.Relation AndAlso
                   Args.SequenceEqual(other.Args)
        End Function

        Public Overrides Function GetHashCode() As Integer
            Return HashCode.Combine(Relation, String.Join(",", Args))
        End Function
    End Class

    Public Module GeneratedQuery
        Private Facts As New HashSet(Of Fact)
        Private Delta As New HashSet(Of Fact)

        Public Sub InitFacts()
            Facts.Add(New Fact With {.Relation = "parent", .Args = New List(Of String) From {"john", "mary"}})
            Delta = New HashSet(Of Fact)(Facts)
        End Sub

        Public Function Solve() As HashSet(Of Fact)
            InitFacts()
            While Delta.Count > 0
                Dim NewFacts = ApplyRules()
                Delta = New HashSet(Of Fact)(NewFacts.Except(Facts))
                Facts.UnionWith(Delta)
            End While
            Return Facts
        End Function

        Private Function ApplyRules() As HashSet(Of Fact)
            ' Rule implementations
            Return New HashSet(Of Fact)
        End Function

        Sub Main()
            For Each fact In Solve()
                Console.WriteLine($"{fact.Relation}({String.Join(", ", fact.Args)})")
            Next
        End Sub
    End Module
End Namespace
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/vbnet_target'),
    compile_to_vbnet(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_vbnet_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOTNET_ROOT` | .NET SDK path | (system) |
| `SKIP_VBNET_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Case insensitivity**: VB.NET is case-insensitive
2. **Line continuations**: _ character needed
3. **Modern features**: Some C# features not in VB
