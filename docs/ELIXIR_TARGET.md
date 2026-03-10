# UnifyWeaver Elixir Target

The Elixir target translates Prolog predicates into Elixir modules and scripts, leveraging BEAM's native tail-call optimization and pattern matching.

## Capabilities

| Feature | Support |
|---|---|
| Facts → module attributes | ✅ |
| Rules → pattern-matched `def` | ✅ |
| Tail recursion (TCO) | ✅ native |
| Linear recursion | ✅ |
| Mutual recursion | ✅ single module |
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

### Simple Mode
```elixir
defmodule Generated.MyPred do
  def my_pred(arg1, arg2) do
    # translated clause body
  end
end
```

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

## Files
- `src/unifyweaver/targets/elixir_target.pl` — code generation
- `src/unifyweaver/bindings/elixir_bindings.pl` — built-in operation mappings
- `tests/test_elixir_target.pl` — unit tests
