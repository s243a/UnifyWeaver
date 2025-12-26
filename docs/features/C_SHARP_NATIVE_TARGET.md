# C# Native Target

The C# native target (`csharp_native`) compiles Prolog predicates to standalone C# code using LINQ for rule bodies and procedural recursion for multiple clauses and recursive predicates.

## Features
- **Standalone Code**: Generates self-contained `.cs` files with no runtime engine dependency.
- **LINQ Integration**: Reuses efficient LINQ operators (`Join`, `Select`, `Concat`) for Datalog-style joins.
- **Procedural Recursion**: Supports recursive predicates using method-to-method calls and `yield return`.
- **Memoization**: Implements automatic memoization (using `Dictionary`) to ensure termination on cyclic data and improve performance.
- **Parameter Support**: Supports passing arguments to generated methods for efficient filtering.

## Evaluation Strategy: Procedural
Unlike the Query Runtime's semi-naive fixpoint solver, the Native target uses a procedural approach similar to the Python and Bash targets. Each clause is translated into a private method, and the main predicate method orchestrates these calls.

### Recursive Pattern
Recursive calls are translated into direct method calls. To prevent infinite loops on cyclic graphs (e.g., in transitive closure), the target wraps the execution in a memoized stream:

```csharp
public static IEnumerable<(string, string)> AncestorStream()
{
    if (_memo.TryGetValue("all", out var cached)) return cached;
    var results = new List<(string, string)>();
    foreach (var item in AncestorInternal()) {
        results.Add(item);
    }
    _memo["all"] = results;
    return results;
}
```

## Usage
To use the native target, specify `target(csharp_native)` or use the default `target(csharp)` with `mode(procedural)`:

```prolog
?- compile_predicate_to_csharp(ancestor/2, [mode(procedural)], Code).
```