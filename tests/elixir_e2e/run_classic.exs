# Shared driver for the WAM-Elixir classic-programs test suite.
#
# Mirrors the Scala test disciplines `verify_scala_args` pattern:
# `elixir run_classic.exs <ModuleName> <PredKey> <Arg1> [<Arg2> ...]`
# prints "true" if the predicate succeeds, "false" if it fails.
#
# Args are parsed loosely:
# - "5"     -> integer 5
# - "1.5"   -> float 1.5
# - "[a,b]" -> list ["a", "b"] (atom-list shorthand)
# - "[]"    -> empty list
# - other   -> binary string (atom-equivalent)
#
# This covers the ground-args query shapes used by the classic Prolog
# tests (fibonacci, Ackermann, list_reverse with simple atom lists).
# Compound terms (e.g. `1 + 2`) are not currently supported — extend
# parse_arg/1 below if a future test needs them.

Code.require_file("lib/wam_runtime.ex", __DIR__)
Code.require_file("lib/wam_dispatcher.ex", __DIR__)

# CLI: <module_camel> <pred_key> [<arg> ...]
[mod_camel, pred_key | raw_args] = System.argv()

# pred_key is "name/arity" — strip arity, camelise the name.
[pred_name, _arity] = String.split(pred_key, "/")
pred_camel = Macro.camelize(pred_name)
pred_module = Module.concat([mod_camel, pred_camel])

# Load the per-predicate Elixir module. Module name camelisation
# follows the same convention `wam_elixir_target.pl`s emitter uses.
Code.require_file("lib/#{pred_name}.ex", __DIR__)

defmodule ParseArg do
  def parse(s) do
    cond do
      s == "[]" ->
        []

      String.starts_with?(s, "[") and String.ends_with?(s, "]") ->
        # Simple flat atom-list "[a,b,c]" -> ["a","b","c"]. Not nested
        # lists; not list of integers; not lists with embedded commas.
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

# All args are ground in this test discipline (no `{:unbound, _}`).
# The pred modules `run/1` builds a fresh state, binds each arg into
# its A-register, and runs the WAM bytecode. Result:
#   {:ok, _final_state} -> all unifications succeeded -> "true"
#   :fail               -> some unification failed     -> "false"
case apply(pred_module, :run, [parsed_args]) do
  {:ok, _} -> IO.puts("true")
  :fail -> IO.puts("false")
  other -> IO.puts("error: #{inspect(other)}")
end
