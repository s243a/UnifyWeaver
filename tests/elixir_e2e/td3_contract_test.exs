defmodule Td3ContractTest do
  defp pair(state) do
    {WamRuntime.get_reg(state, 2), WamRuntime.get_reg(state, 3)}
  end

  defp collect({:ok, state}), do: collect_state(state, [])
  defp collect(:fail), do: []

  defp collect_state(state, acc) do
    next_acc = [pair(state) | acc]

    case WamRuntime.next_solution(state) do
      {:ok, next_state} -> collect_state(next_state, next_acc)
      :fail -> Enum.reverse(next_acc)
    end
  end

  defp assert_equal(name, actual, expected) do
    if actual != expected do
      raise "#{name}: expected #{inspect(expected)}, got #{inspect(actual)}"
    end
  end

  defp aliased_output_call do
    id = make_ref()
    shared = {:unbound, id}

    state = %WamRuntime.WamState{
      code: {},
      labels: %{},
      pc: 1,
      cp: &WamRuntime.terminal_cp/1,
      regs: %{1 => "a", 2 => shared, 3 => shared}
    }

    try do
      Probe.Kdtd.run(state)
    catch
      {:fail, _state} -> :fail
    end
  end

  defp aggregate_call(target, distance, value_reg) do
    aggregate_call(target, distance, value_reg, [])
  end

  defp aggregate_call(target, distance, value_reg, initial_accum) do
    state = %WamRuntime.WamState{
      code: {},
      labels: %{},
      pc: 1,
      cp: &WamRuntime.terminal_cp/1,
      regs: %{1 => "a", 2 => target, 3 => distance},
      choice_points: [
        %{
          agg_type: :findall,
          agg_value_reg: value_reg,
          agg_accum: initial_accum
        }
      ]
    }

    try do
      Probe.Kdtd.run(state)
      raise "aggregate dispatch returned instead of failing into its aggregate frame"
    catch
      {:fail, failed_state} ->
        [aggregate_cp] = failed_state.choice_points
        aggregate_cp.agg_accum
    end
  end

  defp aggregate_aliased_output_call(value_reg) do
    id = make_ref()
    shared = {:unbound, id}
    aggregate_call(shared, shared, value_reg)
  end

  def run do
    table = :ets.new(:td3_contract_edges, [:duplicate_bag, :public])

    Enum.each(
      [
        {"a", "b"},
        {"a", "b"},
        {"b", "c"},
        {"c", "a"},
        {"c", "d"},
        {"e", "e"},
        # This row proves the Source atom gate rather than merely relying
        # on a non-atom key being absent from the fact store.
        {1, "integer-key-destination"}
      ],
      &:ets.insert(table, &1)
    )

    source =
      WamRuntime.FactSource.Ets.open(%{table: table, arity: 2}, 2, nil)

    WamRuntime.FactSourceRegistry.register("kdedge/2", source)

    assert_equal(
      "unbound stream",
      collect(Probe.Kdtd.run(["a", {:unbound, :target}, {:unbound, :distance}]))
      |> Enum.sort(),
      [{"a", 3}, {"b", 1}, {"c", 2}, {"d", 3}]
    )

    assert_equal(
      "bound target",
      collect(Probe.Kdtd.run(["a", "c", {:unbound, :distance}])),
      [{"c", 2}]
    )

    assert_equal(
      "bound distance stream",
      collect(Probe.Kdtd.run(["a", {:unbound, :target}, 3])) |> Enum.sort(),
      [{"a", 3}, {"d", 3}]
    )

    assert_equal("both bound", collect(Probe.Kdtd.run(["a", "c", 2])), [{"c", 2}])
    assert_equal("bound mismatch", Probe.Kdtd.run(["a", "c", 1]), :fail)
    assert_equal(
      "aliased outputs",
      aliased_output_call(),
      :fail
    )
    assert_equal(
      "aggregate target slicing",
      aggregate_call({:unbound, make_ref()}, {:unbound, make_ref()}, 2)
      |> Enum.sort(),
      ["a", "b", "c", "d"]
    )
    assert_equal(
      "aggregate distance slicing",
      aggregate_call({:unbound, make_ref()}, {:unbound, make_ref()}, 3)
      |> Enum.sort(),
      [1, 2, 3, 3]
    )
    assert_equal(
      "aggregate bound target filters distance slice",
      aggregate_call("c", {:unbound, make_ref()}, 3),
      [2]
    )
    assert_equal(
      "aggregate bound distance filters target slice",
      aggregate_call({:unbound, make_ref()}, 3, 2) |> Enum.sort(),
      ["a", "d"]
    )
    assert_equal(
      "aggregate aliased outputs target slice",
      aggregate_aliased_output_call(2),
      []
    )
    assert_equal(
      "aggregate aliased outputs distance slice",
      aggregate_aliased_output_call(3),
      []
    )
    assert_equal("zero distance", Probe.Kdtd.run(["a", {:unbound, :target}, 0]), :fail)
    assert_equal("negative distance", Probe.Kdtd.run(["a", {:unbound, :target}, -1]), :fail)
    assert_equal("noninteger distance", Probe.Kdtd.run(["a", {:unbound, :target}, "1"]), :fail)
    assert_equal("nonatom source", Probe.Kdtd.run([1, {:unbound, :target}, {:unbound, :distance}]), :fail)
    assert_equal(
      "unbound source",
      Probe.Kdtd.run([{:unbound, :source}, {:unbound, :target}, {:unbound, :distance}]),
      :fail
    )
    assert_equal("sink", Probe.Kdtd.run(["d", {:unbound, :target}, {:unbound, :distance}]), :fail)
    assert_equal("unknown source", Probe.Kdtd.run(["z", {:unbound, :target}, {:unbound, :distance}]), :fail)
    assert_equal(
      "self loop",
      collect(Probe.Kdtd.run(["e", "e", {:unbound, :distance}])),
      [{"e", 1}]
    )

    WamRuntime.FactSourceRegistry.unregister("kdedge/2")
    :ets.delete(table)
    IO.puts("[PASS] TD3 Elixir bound modes, aggregate aliases, and stream retry")
  end
end

Td3ContractTest.run()
