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
| Mix project scaffolding | ✅ auto-dependency detection |

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

## Mix Project Generation

For projects that need dependency management (especially pipeline mode
which requires Jason), generate a complete Mix project:

```prolog
% Generate project with auto-detected Jason dependency
?- write_mix_project(my_pipeline, [pipeline_input(true)], '/tmp/output').
% Creates: /tmp/output/my_pipeline/{mix.exs, lib/, test/, config/, ...}

% Generate project with explicit deps
?- write_mix_project(my_app, [deps([jason, req])], '/tmp/output').

% Just generate the mix.exs content
?- generate_mix_exs('my_app', [pipeline_input(true)], Code).
```

The generated project includes:
- `mix.exs` — project definition with auto-detected dependencies
- `lib/<project>.ex` — placeholder module with correct CamelCase name
- `test/test_helper.exs` + `test/<project>_test.exs` — ExUnit test scaffold
- `config/config.exs` — logger configuration
- `.formatter.exs`, `.gitignore`, `README.md`

### Dependency Auto-Detection

| Option | Auto-Added Dep |
|--------|---------------|
| `pipeline_input(true)` | `jason ~> 1.4` |
| `generator_mode(true)` | `jason ~> 1.4` |
| `deps([plug, req])` | explicit deps with known versions |

Known dependency versions: jason ~> 1.4, plug ~> 1.14, plug_cowboy ~> 2.6,
ecto ~> 3.10, req ~> 0.4. Unknown deps default to ~> 0.1.

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
- `src/unifyweaver/targets/elixir_target.pl` — code generation + Mix project scaffolding
- `src/unifyweaver/bindings/elixir_bindings.pl` — built-in operation mappings
- `tests/test_elixir_target.pl` — unit tests (16 cases)
