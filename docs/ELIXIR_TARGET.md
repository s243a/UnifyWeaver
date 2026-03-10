# UnifyWeaver Elixir Target

The Elixir target translates Prolog predicates into Elixir modules and scripts, leveraging BEAM's native tail-call optimization and pattern matching.

## Capabilities

| Feature | Support |
|---|---|
| Facts → module attributes | ✅ |
| Rules → pattern-matched `def` | ✅ multi-clause with ground arg matching |
| Tail recursion (TCO) | ✅ native BEAM |
| Linear recursion | ✅ multifile dispatch |
| Mutual recursion | ✅ single module with full clause generation |
| Transitive closure (BFS) | ✅ Map + MapSet |
| Pipeline mode (JSONL) | ✅ Stream + Jason |
| Generator mode (lazy) | ✅ Stream.unfold |
| Bindings | ✅ arithmetic, comparison, string, I/O |

## Prerequisites
- Elixir ≥ 1.14
- Optional: `Jason` (~> 1.4) for JSONL pipeline mode

## Usage

```bash
# Compile a predicate to Elixir
swipl -g "compile_incremental(my_pred/2, elixir, [], Code), writeln(Code)" -t halt

# Run a generated pipeline
elixir generated_pipeline.exs --run < data.jsonl
```

## Generated Code Structure

### Simple Mode — Pattern-Matched Clauses
```elixir
defmodule Generated.MyPred do
  def my_pred("hello", arg2) do
    try do
      # translated body
      {:ok, [arg1, arg2]}
    catch
      :fail -> :fail
    end
  end

  def my_pred("world", arg2) do
    # second clause with different pattern
  end
end
```

Ground arguments in clause heads become literal pattern matches in the
Elixir function heads (e.g., `"hello"` above). Variable arguments are
mapped to positional parameter names (`arg1`, `arg2`, ...).

### Pipeline Mode
```elixir
defmodule Generated.MyPredPipeline do
  def process(record), do: ...
  def run do
    IO.stream(:stdio, :line)
    |> Stream.map(&Jason.decode!/1)
    |> Stream.map(&process/1)
    |> Enum.each(&(IO.puts(Jason.encode!(&1))))
  end
end
```

### Mutual Recursion
```elixir
defmodule Generated.MutualGroup do
  def even(0), do: ...
  def odd(1), do: ...
  # All mutually-recursive predicates grouped in one module
end
```

### Module Naming

Predicate names are converted from `snake_case` to `CamelCase` for
idiomatic Elixir module names via `snake_to_camel/2`:

| Prolog Predicate | Elixir Module |
|---|---|
| `my_parent` | `Generated.MyParent` |
| `elix_greet` | `Generated.ElixGreet` |
| `ancestor` | `Generated.Ancestor` |

## Architecture

### Multifile Dispatch

The Elixir target registers multifile clauses for the recursion compiler
modules, following the same pattern established by the R target (PR #753):

- `tail_recursion:compile_tail_pattern/9` — BEAM native TCO
- `linear_recursion:compile_linear_pattern/8` — multi-clause defs
- `mutual_recursion:compile_mutual_pattern/5` — grouped module generation
- `compile_transitive_closure/6` — BFS with Map + MapSet (in recursive_compiler.pl)

### Variable Mapping

The simple mode compiler builds a `VarMap` during head argument analysis
that maps Prolog variables to their Elixir parameter names. This map is
threaded through body translation so that variables referenced in the
body resolve to the correct Elixir identifiers.

## Files
- `src/unifyweaver/targets/elixir_target.pl` — code generation
- `src/unifyweaver/bindings/elixir_bindings.pl` — built-in operation mappings
- `tests/test_elixir_target.pl` — unit tests (9 cases)
