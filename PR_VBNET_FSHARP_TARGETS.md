# feat: Add VB.NET and F# targets for complete .NET language coverage

## Summary

This PR adds two new compilation targets for VB.NET and F# to UnifyWeaver, completing .NET language family support alongside the existing C# and PowerShell targets.

## New Features

### VB.NET Target (`vbnet_target.pl`)
- Compile Prolog predicates to VB.NET programs
- Fact export with typed `Public Class` and `List(Of T)`
- Tail recursion → `Do While` loops (O(1) stack)
- Linear recursion → `Dictionary(Of TKey, TValue)` memoization
- Mutual recursion → Shared `Dictionary` with string keys

### F# Target (`fsharp_target.pl`)
- Compile Prolog predicates to idiomatic F# code
- Fact export with record types and `Seq` streaming
- Tail recursion → `let rec` with accumulator pattern
- Linear recursion → `Dictionary` with pattern matching
- Mutual recursion → Native `and` keyword support

### Bindings
- `vbnet_bindings.pl` - 20+ bindings (CInt, CDbl, Math.*, String.*)
- `fsharp_bindings.pl` - 25+ bindings (List.map, List.filter, Seq.*, printfn)

## API Exports

| Target | Predicates |
|--------|------------|
| VB.NET | `compile_predicate_to_vbnet/3`, `compile_tail_recursion_vbnet/3`, `compile_linear_recursion_vbnet/3`, `compile_mutual_recursion_vbnet/3` |
| F# | `compile_predicate_to_fsharp/3`, `compile_tail_recursion_fsharp/3`, `compile_linear_recursion_fsharp/3`, `compile_mutual_recursion_fsharp/3` |

## Testing

- **8/8 tests pass** covering facts, tail, linear, and mutual recursion for both targets

## Documentation

### Main Docs
- `docs/VBNET_TARGET.md`
- `docs/FSHARP_TARGET.md`

### Education Books
- `book-vbnet-target/` (README, Ch1, Ch3)
- `book-fsharp-target/` (README, Ch1, Ch3)

## Files Changed

```
src/unifyweaver/targets/vbnet_target.pl        [NEW]
src/unifyweaver/targets/fsharp_target.pl       [NEW]
src/unifyweaver/bindings/vbnet_bindings.pl     [NEW]
src/unifyweaver/bindings/fsharp_bindings.pl    [NEW]
tests/test_vbnet_fsharp_targets.pl             [NEW]
docs/VBNET_TARGET.md                           [NEW]
docs/FSHARP_TARGET.md                          [NEW]
```

## Related

- Complements existing C# target (`csharp_target.pl`)
- Complements existing PowerShell target (`powershell_compiler.pl`)
