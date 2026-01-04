# F# Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: F# code generation target testing

## Overview

This test plan covers the F# target for UnifyWeaver, which generates functional F# code with type inference, pattern matching, and .NET interop.

## Prerequisites

### System Requirements

- .NET SDK 8.0+
- F# compiler (included with .NET SDK)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify .NET SDK
dotnet --version
dotnet fsi --version

# Verify F# compiler
dotnet fsc --version 2>/dev/null || echo "Use 'dotnet build' for F#"

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_fsharp_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `let_bindings` | let/let rec | Binding syntax |
| `type_definitions` | type keyword | Record/DU types |
| `pattern_matching` | match/with | Match expressions |
| `pipe_operator` | \|> operator | Pipeline syntax |
| `lambda_expressions` | fun x -> | Lambda syntax |

#### 1.2 F# Type System

```bash
swipl -g "use_module('tests/core/test_fsharp_types'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `record_types` | { Field: T } | Record syntax |
| `discriminated_unions` | \| Case of T | DU syntax |
| `option_type` | Some/None | Option handling |
| `result_type` | Ok/Error | Result handling |

### 2. Compilation Tests

#### 2.1 .NET Build

```bash
./tests/integration/test_fsharp_dotnet.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `dotnet_build` | dotnet build | Build succeeds |
| `dotnet_run` | dotnet run | Execution works |
| `dotnet_publish` | dotnet publish | Publish succeeds |

### 3. Generated Code Structure

```fsharp
namespace UnifyWeaver.Generated

open System.Collections.Generic

type Fact = {
    Relation: string
    Args: string list
}

module GeneratedQuery =
    let initialFacts =
        set [
            { Relation = "parent"; Args = ["john"; "mary"] }
            { Relation = "parent"; Args = ["mary"; "susan"] }
        ]

    let applyRules (total: Set<Fact>) (delta: Set<Fact>) =
        seq {
            for d in delta do
                match d with
                | { Relation = "ancestor"; Args = [x; y] } ->
                    for t in total do
                        match t with
                        | { Relation = "parent"; Args = [y2; z] } when y = y2 ->
                            yield { Relation = "ancestor"; Args = [x; z] }
                        | _ -> ()
                | _ -> ()
        }
        |> Set.ofSeq

    let solve () =
        let rec loop facts delta =
            if Set.isEmpty delta then
                facts
            else
                let newFacts = applyRules facts delta - facts
                loop (facts + newFacts) newFacts
        loop initialFacts initialFacts

[<EntryPoint>]
let main _ =
    GeneratedQuery.solve()
    |> Set.iter (printfn "%A")
    0
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/fsharp_target'),
    compile_to_fsharp(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_fsharp_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOTNET_ROOT` | .NET SDK path | (system) |
| `SKIP_FSHARP_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Indentation sensitivity**: Whitespace matters
2. **C# interop**: Some friction with C# libraries
3. **Tooling**: IDE support varies
