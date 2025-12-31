# C# Native Target

The C# native target (`csharp_native`) compiles Prolog predicates to standalone C# code using LINQ for non-recursive rule bodies and semi-naive iteration for recursive predicates.

## Features

- **Standalone Code**: Generates self-contained `.cs` files with no runtime engine dependency.
- **LINQ Integration**: Uses efficient LINQ operators (`Select`, `Join`) for Datalog-style joins in non-recursive clauses.
- **Semi-Naive Recursion**: Supports recursive predicates using worklist-based semi-naive iteration.
- **Deduplication**: Uses `HashSet` to ensure termination on cyclic data and eliminate duplicate results.
- **Streaming Output**: Results are yielded incrementally via `IEnumerable<T>`.

## Evaluation Strategy

### Non-Recursive Predicates

Non-recursive predicates use LINQ pipelines for efficient joins:

```csharp
private static IEnumerable<(string, string)> _clause_1()
{
    return ParentStream()
        .Select(row0 => (row0.Item1, row0.Item2))
        .Select(res => (res.Item1, res.Item2));
}
```

### Recursive Predicates (Semi-Naive)

Recursive predicates use a worklist-based semi-naive algorithm that avoids stack overflow by only processing newly-discovered tuples (delta) in each iteration:

```csharp
public static IEnumerable<(string, string)> AncestorStream()
{
    var seen = new HashSet<(string, string)>();
    var delta = new List<(string, string)>();

    // Base case: ancestor(X,Y) :- parent(X,Y)
    foreach (var item in _clause_1())
    {
        if (seen.Add(item)) { delta.Add(item); yield return item; }
    }

    // Semi-naive iteration: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
    while (delta.Count > 0)
    {
        var newDelta = new List<(string, string)>();
        foreach (var d in delta)
        {
            foreach (var b in ParentStream())
            {
                if (b.Item2 == d.Item1)
                {
                    var newItem = (b.Item1, d.Item2);
                    if (seen.Add(newItem)) { newDelta.Add(newItem); yield return newItem; }
                }
            }
        }
        delta = newDelta;
    }
}
```

This approach:
1. Computes base case results first
2. Iteratively applies the recursive rule to only new results (delta)
3. Uses `HashSet.Add()` for O(1) deduplication and termination detection
4. Yields results incrementally for memory efficiency

## Usage

To use the native target, specify `target(csharp_native)` or use the default `target(csharp)` with `mode(procedural)`:

```prolog
?- compile_predicate_to_csharp(ancestor/2, [mode(procedural)], Code).
```

Or via the recursive compiler:

```prolog
?- compile_recursive(ancestor/2, [target(csharp), mode(procedural)], Code).
```

## Example Output

For the classic transitive closure example:

```prolog
parent(alice, bob).
parent(bob, charlie).
parent(charlie, diana).
parent(diana, eve).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

The generated C# produces:

```
alice:bob
bob:charlie
charlie:diana
diana:eve
alice:charlie
bob:diana
charlie:eve
alice:diana
bob:eve
alice:eve
```

## Comparison with Query Runtime

| Feature | Native Target | Query Runtime |
|---------|--------------|---------------|
| Evaluation | Semi-naive iteration | Semi-naive fixpoint |
| Output | Standalone .cs file | Requires runtime engine |
| Dependencies | None | Query runtime DLL |
| Best for | Simple recursive predicates | Complex Datalog programs |
