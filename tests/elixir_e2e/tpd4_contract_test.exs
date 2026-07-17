defmodule Tpd4ContractTest do
  defp value(:binary, name), do: Atom.to_string(name)
  defp value(:interned, name), do: name

  defp free(label), do: {:unbound, label}
  defp fresh, do: {:unbound, make_ref()}

  defp triple(state) do
    {
      WamRuntime.get_reg(state, 2),
      WamRuntime.get_reg(state, 3),
      WamRuntime.get_reg(state, 4)
    }
  end

  # Reaching :fail is part of the assertion: this walks every generated
  # function-backed choice point and proves that retry terminates cleanly.
  defp collect({:ok, state}), do: collect_state(state, [])
  defp collect(:fail), do: []

  defp collect_state(state, acc) do
    next_acc = [triple(state) | acc]

    case WamRuntime.next_solution(state) do
      {:ok, next_state} -> collect_state(next_state, next_acc)
      :fail -> Enum.reverse(next_acc)
      other -> raise "retry returned unexpected result: #{inspect(other)}"
    end
  end

  defp assert_equal(name, actual, expected) do
    if actual != expected do
      raise "#{name}: expected #{inspect(expected)}, got #{inspect(actual)}"
    end
  end

  defp assert_solutions(name, result, expected) do
    assert_equal(name, collect(result) |> Enum.sort(), Enum.sort(expected))
  end

  defp direct_call(source, target, parent, distance) do
    state = %WamRuntime.WamState{
      code: {},
      labels: %{},
      pc: 1,
      cp: &WamRuntime.terminal_cp/1,
      regs: %{1 => source, 2 => target, 3 => parent, 4 => distance}
    }

    try do
      Probe.Kdpd.run(state)
    catch
      {:fail, _state} -> :fail
    end
  end

  defp aggregate_call(source, target, parent, distance, value_reg) do
    aggregate_call(source, target, parent, distance, value_reg, [])
  end

  defp aggregate_call(source, target, parent, distance, value_reg, initial_accum) do
    state = %WamRuntime.WamState{
      code: {},
      labels: %{},
      pc: 1,
      cp: &WamRuntime.terminal_cp/1,
      regs: %{1 => source, 2 => target, 3 => parent, 4 => distance},
      choice_points: [
        %{
          agg_type: :findall,
          agg_value_reg: value_reg,
          agg_accum: initial_accum
        }
      ]
    }

    try do
      Probe.Kdpd.run(state)
      raise "aggregate dispatch returned instead of failing into its aggregate frame"
    catch
      {:fail, failed_state} ->
        [aggregate_cp] = failed_state.choice_points
        aggregate_cp.agg_accum
    end
  end

  # TSPD5 shortest-positive correlated contract: unequal diamond keeps only
  # the shorter path to d (distance 2 via b); the longer a→c→e→d arm is
  # suppressed. A finite cycle must still terminate (source via cycle).
  defp assert_tspd5_correlated_shortest_positive do
    unequal_edges = %{
      "a" => [{"a", "b"}, {"a", "c"}],
      "b" => [{"b", "d"}],
      "c" => [{"c", "e"}],
      "e" => [{"e", "d"}]
    }

    unequal_neighbors = fn node -> Map.get(unequal_edges, node, []) end

    unequal_actual =
      WamRuntime.GraphKernel.TransitiveStepParentDistance.collect_quads(
        unequal_neighbors,
        "a"
      )
      |> Enum.sort()

    unequal_expected =
      [
        {"b", "b", "a", 1},
        {"c", "c", "a", 1},
        {"d", "b", "b", 2},
        {"e", "c", "c", 2}
      ]
      |> Enum.sort()

    assert_equal("TSPD5 correlated shortest-positive unequal", unequal_actual, unequal_expected)

    # Longer all-path distance must not appear.
    if Enum.any?(unequal_actual, &(&1 == {"d", "c", "e", 3})) do
      raise "TSPD5 correlated: unexpected longer all-path quad {d,c,e,3}"
    end

    cycle_edges = %{
      "a" => [{"a", "b"}],
      "b" => [{"b", "a"}]
    }

    cycle_neighbors = fn node -> Map.get(cycle_edges, node, []) end

    cycle_actual =
      WamRuntime.GraphKernel.TransitiveStepParentDistance.collect_quads(
        cycle_neighbors,
        "a"
      )
      |> Enum.sort()

    cycle_expected =
      [
        {"a", "b", "b", 2},
        {"b", "b", "a", 1}
      ]
      |> Enum.sort()

    assert_equal("TSPD5 correlated cycle terminates", cycle_actual, cycle_expected)
  end

  defp run_representation(representation) do
    a = value(representation, :a)
    b = value(representation, :b)
    c = value(representation, :c)
    d = value(representation, :d)
    z = value(representation, :z)
    seed = value(representation, :seed)
    label = Atom.to_string(representation)

    table = :ets.new(:tpd4_contract_edges, [:duplicate_bag, :public])

    Enum.each(
      [
        {a, b},
        {a, b},
        {a, c},
        {a, a},
        {b, d},
        {c, d},
        {d, a},
        # Mixed-domain rows are deliberately connected to the atom graph.
        # TPD4 must not emit the integer node or continue through it.
        {a, 1},
        {1, value(representation, :integer_key_destination)}
      ],
      &:ets.insert(table, &1)
    )

    source = WamRuntime.FactSource.Ets.open(%{table: table, arity: 2}, 2, nil)
    WamRuntime.FactSourceRegistry.register("kdedge/2", source)

    expected = [
      {a, a, 1},
      {b, a, 1},
      {c, a, 1},
      {d, b, 2},
      {d, c, 2}
    ]

    try do
      # All 2^3 Target/Parent/Distance bound/free modes.
      assert_solutions(
        "#{label} mode ---",
        Probe.Kdpd.run([a, free(:target), free(:parent), free(:distance)]),
        expected
      )

      assert_solutions(
        "#{label} mode T--",
        Probe.Kdpd.run([a, d, free(:parent), free(:distance)]),
        [{d, b, 2}, {d, c, 2}]
      )

      assert_solutions(
        "#{label} mode -P-",
        Probe.Kdpd.run([a, free(:target), a, free(:distance)]),
        [{a, a, 1}, {b, a, 1}, {c, a, 1}]
      )

      assert_solutions(
        "#{label} mode --D",
        Probe.Kdpd.run([a, free(:target), free(:parent), 2]),
        [{d, b, 2}, {d, c, 2}]
      )

      assert_solutions(
        "#{label} mode TP-",
        Probe.Kdpd.run([a, d, c, free(:distance)]),
        [{d, c, 2}]
      )

      assert_solutions(
        "#{label} mode T-D",
        Probe.Kdpd.run([a, d, free(:parent), 2]),
        [{d, b, 2}, {d, c, 2}]
      )

      assert_solutions(
        "#{label} mode -PD",
        Probe.Kdpd.run([a, free(:target), b, 2]),
        [{d, b, 2}]
      )

      assert_solutions(
        "#{label} mode TPD",
        Probe.Kdpd.run([a, d, c, 2]),
        [{d, c, 2}]
      )

      assert_equal("#{label} all-bound mismatch", Probe.Kdpd.run([a, d, c, 1]), :fail)

      assert_equal(
        "#{label} integer Target",
        Probe.Kdpd.run([a, 1, a, 1]),
        :fail
      )

      # Shared output variables must be unified jointly, not treated as
      # independent wildcards by the compatibility pre-filter.
      target_parent = fresh()

      assert_solutions(
        "#{label} alias Target=Parent",
        direct_call(a, target_parent, target_parent, fresh()),
        [{a, a, 1}]
      )

      target_distance = fresh()
      assert_equal(
        "#{label} alias Target=Distance",
        direct_call(a, target_distance, fresh(), target_distance),
        :fail
      )

      parent_distance = fresh()
      assert_equal(
        "#{label} alias Parent=Distance",
        direct_call(a, fresh(), parent_distance, parent_distance),
        :fail
      )

      all_outputs = fresh()
      assert_equal(
        "#{label} alias Target=Parent=Distance",
        direct_call(a, all_outputs, all_outputs, all_outputs),
        :fail
      )

      assert_solutions(
        "#{label} bound Source=Target",
        Probe.Kdpd.run([a, a, free(:parent), free(:distance)]),
        [{a, a, 1}]
      )

      # Aggregates project from the jointly filtered triple stream. Target
      # and distance retain multiplicity when equal-shortest parents differ.
      assert_equal(
        "#{label} aggregate Target",
        aggregate_call(a, fresh(), fresh(), fresh(), 2) |> Enum.sort(),
        [a, b, c, d, d] |> Enum.sort()
      )

      assert_equal(
        "#{label} aggregate Parent",
        aggregate_call(a, fresh(), fresh(), fresh(), 3) |> Enum.sort(),
        [a, a, a, b, c] |> Enum.sort()
      )

      assert_equal(
        "#{label} aggregate Distance",
        aggregate_call(a, fresh(), fresh(), fresh(), 4) |> Enum.sort(),
        [1, 1, 1, 2, 2]
      )

      assert_equal(
        "#{label} aggregate bound Target filters Parent",
        aggregate_call(a, d, fresh(), fresh(), 3) |> Enum.sort(),
        [b, c] |> Enum.sort()
      )

      aggregate_alias = fresh()
      assert_equal(
        "#{label} aggregate alias Target=Parent",
        aggregate_call(a, aggregate_alias, aggregate_alias, fresh(), 2),
        [a]
      )

      incompatible_aggregate_alias = fresh()
      assert_equal(
        "#{label} aggregate alias Target=Distance",
        aggregate_call(
          a,
          incompatible_aggregate_alias,
          fresh(),
          incompatible_aggregate_alias,
          3
        ),
        []
      )

      assert_equal(
        "#{label} aggregate preserves accumulator",
        aggregate_call(a, d, fresh(), fresh(), 3, [seed]),
        [seed, b, c]
      )

      # Domain and Source gates.
      assert_equal(
        "#{label} zero distance",
        Probe.Kdpd.run([a, free(:target), free(:parent), 0]),
        :fail
      )

      assert_equal(
        "#{label} negative distance",
        Probe.Kdpd.run([a, free(:target), free(:parent), -1]),
        :fail
      )

      assert_equal(
        "#{label} noninteger distance",
        Probe.Kdpd.run([a, free(:target), free(:parent), value(representation, :one)]),
        :fail
      )

      assert_equal(
        "#{label} integer Source",
        Probe.Kdpd.run([1, free(:target), free(:parent), free(:distance)]),
        :fail
      )

      assert_equal(
        "#{label} compound Source",
        Probe.Kdpd.run([{:compound, a}, free(:target), free(:parent), free(:distance)]),
        :fail
      )

      assert_equal(
        "#{label} unbound Source",
        Probe.Kdpd.run([free(:source), free(:target), free(:parent), free(:distance)]),
        :fail
      )

      assert_equal(
        "#{label} unknown Source",
        Probe.Kdpd.run([z, free(:target), free(:parent), free(:distance)]),
        :fail
      )
    after
      WamRuntime.FactSourceRegistry.unregister("kdedge/2")
      :ets.delete(table)
    end
  end

  def run do
    assert_tspd5_correlated_shortest_positive()
    Enum.each([:binary, :interned], &run_representation/1)

    IO.puts(
      "[PASS] TPD4 Elixir full binding, aggregate, alias, representation, and retry contract"
    )
  end
end

Tpd4ContractTest.run()
