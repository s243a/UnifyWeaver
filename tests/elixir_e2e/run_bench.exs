# Bench driver for the WAM-Elixir atom-lookup baseline bench.
#
# Loads the project (same shape as run_classic.exs), then runs the
# query N times under :timer.tc/1. Prints two lines on stdout:
#   N=<reps> total_us=<total> mean_us=<mean>
#   result=<true|false|other>
#
# CLI: elixir run_bench.exs <ModuleCamel> <PredKey> <reps> [<arg> ...]
#
# Args parsing mirrors run_classic.exs (integers, floats, [a,b] lists,
# bare binaries).

Code.require_file("lib/wam_runtime.ex", __DIR__)
Code.require_file("lib/wam_dispatcher.ex", __DIR__)

[mod_camel, pred_key, reps_str | raw_args] = System.argv()
reps = String.to_integer(reps_str)

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

# Warmup: one untimed invocation so JIT / module loading is amortised.
_ = apply(pred_module, :run, [parsed_args])

# Timed: reps invocations under :timer.tc/1. Time returned in microseconds.
{total_us, last} =
  :timer.tc(fn ->
    Enum.reduce(1..reps, nil, fn _, _acc ->
      apply(pred_module, :run, [parsed_args])
    end)
  end)

mean_us = total_us / reps

IO.puts("N=#{reps} total_us=#{total_us} mean_us=#{Float.round(mean_us, 2)}")

result_str =
  case last do
    {:ok, _} -> "true"
    :fail -> "false"
    other -> "other:#{inspect(other)}"
  end

IO.puts("result=#{result_str}")
