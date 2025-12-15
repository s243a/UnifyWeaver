# F# Target

The F# target compiles Prolog predicates to F# programs using functional programming idioms like pattern matching, immutable data, and higher-order functions.

## Overview

| Feature | Status |
|---------|--------|
| Fact Compilation | ✅ |
| Rule Compilation | ✅ |
| Bindings | ✅ 25+ |
| Tail Recursion | ✅ let rec |
| Linear Recursion | ✅ Dictionary |
| Mutual Recursion | ✅ and keyword |

## Quick Start

```prolog
% Load the target
?- use_module('src/unifyweaver/targets/fsharp_target').
?- init_fsharp_target.

% Compile facts
?- compile_predicate_to_fsharp(person/2, [], Code),
   write_fsharp_program(Code, 'Person.fs').
```

## API Reference

### `compile_predicate_to_fsharp/3`

Compile a Prolog predicate to F#.

```prolog
compile_predicate_to_fsharp(+Pred/Arity, +Options, -FSharpCode)
```

### `compile_facts_to_fsharp/3`

Compile facts to F# with record type generation.

```prolog
compile_facts_to_fsharp(+Pred, +Arity, -FSharpCode)
```

### `compile_tail_recursion_fsharp/3`

Compile tail-recursive predicates using `let rec` with accumulator.

```prolog
compile_tail_recursion_fsharp(+Pred/Arity, +Options, -FSharpCode)
```

### `compile_linear_recursion_fsharp/3`

Compile linear recursion with Dictionary memoization.

```prolog
compile_linear_recursion_fsharp(+Pred/Arity, +Options, -FSharpCode)
```

### `compile_mutual_recursion_fsharp/3`

Compile mutually recursive predicates using F#'s `and` keyword.

```prolog
compile_mutual_recursion_fsharp(+Predicates, +Options, -FSharpCode)
```

## Generated Code Patterns

### Facts → Record + List

```fsharp
type PERSON = {
    Arg1: string
    Arg2: string
}

let getAllPERSON () = [ { Arg1 = "john"; Arg2 = "25" } ]
let streamPERSON () = getAllPERSON () |> Seq.ofList
```

### Tail Recursion → let rec

```fsharp
let sum n =
    let rec loop current acc =
        if current <= 0 then acc
        else loop (current - 1) (acc + current)
    loop n 0
```

### Mutual Recursion → and keyword

```fsharp
let rec is_even n =
    match n with
    | 0 -> true
    | n when n > 0 -> is_odd (n - 1)
    | _ -> false

and is_odd n =
    match n with
    | 0 -> false
    | n when n > 0 -> is_even (n - 1)
    | _ -> false
```

## Bindings

Initialize bindings:
```prolog
?- init_fsharp_target.
```

| Category | Examples |
|----------|----------|
| Core | `Array.length`, `List.length`, `string`, `int` |
| String | `.Split`, `String.concat`, `.Trim` |
| List/Seq | `List.map`, `List.filter`, `List.fold`, `Seq.ofList` |
| I/O | `File.Exists`, `File.ReadAllText`, `printfn` |

## Functional Idioms

The F# target leverages functional patterns:
- **Pipeline operator** `|>` for data flow
- **Pattern matching** with `match`
- **Immutable records** for data
- **Sequences** for lazy evaluation

## See Also

- [CSHARP_TARGET](./targets/CSHARP_TARGET.md) - C# target
- [RUST_TARGET](./RUST_TARGET.md) - Rust target (also functional)
