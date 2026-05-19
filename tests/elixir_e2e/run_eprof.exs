# Profiling driver: runs the bench query under :eprof and prints
# the analysis to stdout. Captures per-function call counts and
# cumulative time so we can see whether choice-point management,
# state-map updates, or unification dominates the bench.
#
# CLI: elixir run_eprof.exs <ModuleCamel> <PredKey> [<arg> ...]
#
# Args parsing mirrors run_classic.exs.

Code.require_file("lib/wam_runtime.ex", __DIR__)
Code.require_file("lib/wam_dispatcher.ex", __DIR__)

[mod_camel, pred_key | raw_args] = System.argv()

[pred_name, _arity] = String.split(pred_key, "/")
pred_camel = Macro.camelize(pred_name)
pred_module = Module.concat([mod_camel, pred_camel])

lib_dir = Path.join(__DIR__, "lib")
{:ok, lib_files} = File.ls(lib_dir)

lib_files
|> Enum.filter(&String.ends_with?(&1, ".ex"))
|> Enum.reject(&(&1 in ["wam_runtime.ex", "wam_dispatcher.ex"]))
|> Enum.each(fn fname ->
  Code.require_file("lib/#{fname}", __DIR__)
end)

defmodule ParseArg do
  def parse(s) do
    cond do
      s == "[]" ->
        []

      String.starts_with?(s, "[") and String.ends_with?(s, "]") ->
        s
        |> String.slice(1..-2//1)
        |> String.split(",")
        |> Enum.map(&String.trim/1)
        |> Enum.map(&parse/1)

      Regex.match?(~r/^-?\d+$/, s) ->
        String.to_integer(s)

      Regex.match?(~r/^-?\d+\.\d+$/, s) ->
        String.to_float(s)

      true ->
        s
    end
  end
end

parsed_args = Enum.map(raw_args, &ParseArg.parse/1)

# Warmup: untimed call so module loading and BEAM JIT settle.
_ = apply(pred_module, :run, [parsed_args])

# Profile a single invocation.
:eprof.start()
:eprof.start_profiling([self()])
_result = apply(pred_module, :run, [parsed_args])
:eprof.stop_profiling()

# Analyze prints to stdout (group_leader). Includes per-function
# call count and cumulative time (microseconds + percentage).
:eprof.analyze(:total)
