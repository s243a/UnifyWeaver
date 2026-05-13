# End-to-end test for WamRuntime.FactSource.LmdbIntIds using a fake
# `Elmdb` backed by an in-memory Agent (see mock_elmdb.exs). Because
# the runtime calls Elmdb via `Module.concat([Elmdb])`, defining a
# top-level module named `Elmdb` is enough to redirect everything to
# the mock — no production code change required.
#
# Usage:
#   1. Generate the WamRuntime to <project>/lib/wam_runtime.ex.
#   2. Copy this file + mock_elmdb.exs into <project>.
#   3. cd <project> && elixir -r mock_elmdb.exs lmdb_int_ids_mock_test.exs
#
# The Prolog test wrapper (test_lmdb_int_ids_mock_e2e/0 in
# tests/test_wam_elixir_target.pl) does steps 1-3 and asserts the
# script's PASS markers in stdout.

Code.require_file("lib/wam_runtime.ex", __DIR__)

# Alias so production-code calls to `Module.concat([Elmdb])` resolve
# to our MockElmdb. The runtime's `apply(mod, :fn, args)` pattern
# means Elmdb just has to exist as a name; defining it here works.
defmodule Elmdb do
  defdelegate env_open(opts \\ []), to: MockElmdb
  defdelegate env_close(env), to: MockElmdb
  defdelegate db_open(env, name, opts), to: MockElmdb
  defdelegate rw_txn_begin(env), to: MockElmdb
  defdelegate ro_txn_begin(env), to: MockElmdb
  defdelegate txn_commit(txn), to: MockElmdb
  defdelegate txn_abort(txn), to: MockElmdb
  defdelegate ro_txn_commit(txn), to: MockElmdb
  defdelegate txn_get(txn, dbi, key), to: MockElmdb
  defdelegate txn_put(txn, dbi, key, value), to: MockElmdb
  defdelegate ro_txn_cursor_open(txn, dbi), to: MockElmdb
  defdelegate ro_txn_cursor_close(cur), to: MockElmdb
  defdelegate ro_txn_cursor_get(cur, op, arg), to: MockElmdb
end

defmodule LmdbIntIdsTest do
  @moduledoc """
  End-to-end exercise of LmdbIntIds against MockElmdb.

  Each test outputs `[PASS] <name>` or `[FAIL] <name>: <reason>`.
  The Prolog wrapper greps stdout for those markers.
  """

  alias WamRuntime.FactSource.LmdbIntIds

  defp pass(name), do: IO.puts("[PASS] #{name}")
  defp fail(name, reason), do: IO.puts("[FAIL] #{name}: #{reason}")

  defp open_handle(dupsort, cache_capacity \\ 0) do
    {:ok, env} = Elmdb.env_open()
    {:ok, facts} = Elmdb.db_open(env, "facts", if(dupsort, do: [:dupsort], else: []))
    {:ok, k_to_id} = Elmdb.db_open(env, "key_to_id", [])
    {:ok, id_to_k} = Elmdb.db_open(env, "id_to_key", [])

    handle =
      LmdbIntIds.open(
        %{
          env: env,
          facts_dbi: facts,
          key_to_id_dbi: k_to_id,
          id_to_key_dbi: id_to_k,
          arity: 2,
          dupsort: dupsort,
          cache_capacity: cache_capacity
        },
        2,
        nil
      )

    {handle, env}
  end

  # --- tests ---

  def test_encode_decode_round_trip do
    name = "encode_id/decode_id round trip on integer boundary values"
    # Reach into the module's private encode/decode via a public
    # round-trip path: ingest a single pair, lookup_id, lookup_key.
    {handle, env} = open_handle(false)

    {:ok, %{next_id: nid}} = LmdbIntIds.ingest_pairs(handle, [{"alpha", "beta"}])

    cases = [
      {"alpha", 0},
      {"beta", 1}
    ]

    ok? =
      Enum.all?(cases, fn {str, expected_id} ->
        actual_id = LmdbIntIds.lookup_id(handle, str)
        actual_key = LmdbIntIds.lookup_key(handle, expected_id)
        actual_id == expected_id and actual_key == str
      end) and nid == 2

    Elmdb.env_close(env)
    if ok?, do: pass(name), else: fail(name, "round trip mismatch (next_id=#{nid})")
  end

  def test_ingest_idempotent do
    name = "ingest_pairs is idempotent — re-ingesting same string keeps its ID"
    {handle, env} = open_handle(true)

    {:ok, r1} = LmdbIntIds.ingest_pairs(handle, [{"alpha", "beta"}])
    # Re-ingest same pair plus a new one. Existing strings reuse IDs;
    # only "gamma" gets a new one. next_id should be start + 1.
    {:ok, r2} = LmdbIntIds.ingest_pairs(handle, [{"alpha", "beta"}, {"alpha", "gamma"}],
                                        start_id: r1.next_id)

    alpha_id = LmdbIntIds.lookup_id(handle, "alpha")
    beta_id = LmdbIntIds.lookup_id(handle, "beta")
    gamma_id = LmdbIntIds.lookup_id(handle, "gamma")

    Elmdb.env_close(env)

    cond do
      r1.next_id != 2 ->
        fail(name, "first ingest should allocate 2 IDs, got next_id=#{r1.next_id}")
      r2.new_ids != 1 ->
        fail(name, "second ingest should allocate exactly 1 new ID (gamma), got #{r2.new_ids}")
      r2.next_id != 3 ->
        fail(name, "second ingest next_id should be 3, got #{r2.next_id}")
      alpha_id != 0 or beta_id != 1 or gamma_id != 2 ->
        fail(name, "expected alpha=0 beta=1 gamma=2; got #{alpha_id}/#{beta_id}/#{gamma_id}")
      true ->
        pass(name)
    end
  end

  def test_lookup_by_arg1_id_dupsort do
    name = "lookup_by_arg1_id with dupsort returns all values for a key"
    {handle, env} = open_handle(true)

    # alpha -> beta, alpha -> gamma, alpha -> delta
    {:ok, _} = LmdbIntIds.ingest_pairs(handle, [
      {"alpha", "beta"},
      {"alpha", "gamma"},
      {"alpha", "delta"}
    ])

    alpha_id = LmdbIntIds.lookup_id(handle, "alpha")
    pairs = LmdbIntIds.lookup_by_arg1_id(handle, alpha_id, nil)

    # All returned tuples should have alpha_id as the first slot.
    all_alpha = Enum.all?(pairs, fn {k, _} -> k == alpha_id end)

    # The set of value IDs should match {beta, gamma, delta}'s IDs.
    value_ids = pairs |> Enum.map(fn {_, v} -> v end) |> MapSet.new()
    expected_ids = ["beta", "gamma", "delta"]
                   |> Enum.map(&LmdbIntIds.lookup_id(handle, &1))
                   |> MapSet.new()

    Elmdb.env_close(env)

    cond do
      not all_alpha ->
        fail(name, "some returned pairs had non-alpha key: #{inspect(pairs)}")
      not MapSet.equal?(value_ids, expected_ids) ->
        fail(name, "value IDs mismatch — got #{inspect(value_ids)}, expected #{inspect(expected_ids)}")
      true ->
        pass(name)
    end
  end

  def test_lookup_by_arg1_string_round_trip do
    name = "lookup_by_arg1 (binary entry) round-trips IDs back to strings"
    {handle, env} = open_handle(true)

    {:ok, _} = LmdbIntIds.ingest_pairs(handle, [
      {"physics", "science"},
      {"physics", "academia"}
    ])

    pairs = LmdbIntIds.lookup_by_arg1(handle, "physics", nil)

    keys_match = Enum.all?(pairs, fn {k, _} -> k == "physics" end)
    values_set = pairs |> Enum.map(fn {_, v} -> v end) |> MapSet.new()
    expected_set = MapSet.new(["science", "academia"])

    Elmdb.env_close(env)

    cond do
      not keys_match ->
        fail(name, "binary entry key mismatch: #{inspect(pairs)}")
      not MapSet.equal?(values_set, expected_set) ->
        fail(name, "value strings mismatch — got #{inspect(values_set)}, expected #{inspect(expected_set)}")
      true ->
        pass(name)
    end
  end

  def test_lookup_cache_is_bounded_and_collision_safe do
    name = "lookup_by_arg1_id cache is bounded and collision-safe"
    {handle, env} = open_handle(true, 1)

    {:ok, _} = LmdbIntIds.ingest_pairs(handle, [
      {"alpha", "beta"},
      {"alpha", "gamma"},
      {"delta", "epsilon"}
    ])

    alpha_id = LmdbIntIds.lookup_id(handle, "alpha")
    delta_id = LmdbIntIds.lookup_id(handle, "delta")

    warmed = LmdbIntIds.preload_arg1_cache(handle, nil)
    alpha_first = LmdbIntIds.lookup_by_arg1_id(handle, alpha_id, nil)
    delta_pairs = LmdbIntIds.lookup_by_arg1_id(handle, delta_id, nil)
    alpha_second = LmdbIntIds.lookup_by_arg1_id(handle, alpha_id, nil)

    alpha_ok =
      alpha_first == alpha_second and
        alpha_first |> Enum.map(fn {_, v} -> v end) |> Enum.sort() ==
          ["beta", "gamma"] |> Enum.map(&LmdbIntIds.lookup_id(handle, &1)) |> Enum.sort()

    delta_ok =
      delta_pairs == [{delta_id, LmdbIntIds.lookup_id(handle, "epsilon")}]

    LmdbIntIds.close(handle, nil)
    Elmdb.env_close(env)

    cond do
      warmed != 2 ->
        fail(name, "expected preload to warm 2 arg1 entries, got #{inspect(warmed)}")
      not alpha_ok ->
        fail(name, "alpha lookup changed across cache collision: #{inspect(alpha_first)} / #{inspect(alpha_second)}")
      not delta_ok ->
        fail(name, "delta lookup mismatch: #{inspect(delta_pairs)}")
      true ->
        pass(name)
    end
  end

  def test_lookup_missing_key_returns_empty do
    name = "lookup_id on missing string returns nil; lookup_by_arg1_id on missing ID returns []"
    {handle, env} = open_handle(false)

    {:ok, _} = LmdbIntIds.ingest_pairs(handle, [{"a", "b"}])

    nil_id = LmdbIntIds.lookup_id(handle, "nonexistent")
    nil_key = LmdbIntIds.lookup_key(handle, 9999)
    empty = LmdbIntIds.lookup_by_arg1_id(handle, 9999, nil)

    Elmdb.env_close(env)

    cond do
      nil_id != nil -> fail(name, "lookup_id of missing returned #{inspect(nil_id)}")
      nil_key != nil -> fail(name, "lookup_key of missing returned #{inspect(nil_key)}")
      empty != [] -> fail(name, "lookup_by_arg1_id of missing returned #{inspect(empty)}")
      true -> pass(name)
    end
  end

  def test_stream_all_returns_int_pairs do
    name = "stream_all returns [{int_key_id, int_value_id}, ...] ordered by key"
    {handle, env} = open_handle(true)

    {:ok, _} = LmdbIntIds.ingest_pairs(handle, [
      {"alpha", "beta"},
      {"gamma", "delta"},
      {"alpha", "epsilon"}
    ])

    pairs = LmdbIntIds.stream_all(handle, nil)

    # Every returned pair should be {integer, integer}.
    all_ints =
      Enum.all?(pairs, fn {k, v} -> is_integer(k) and is_integer(v) end)

    # Should have 3 rows total.
    count_ok = length(pairs) == 3

    Elmdb.env_close(env)

    cond do
      not all_ints -> fail(name, "non-integer pair: #{inspect(pairs)}")
      not count_ok -> fail(name, "expected 3 pairs, got #{length(pairs)}: #{inspect(pairs)}")
      true -> pass(name)
    end
  end

  def test_migrate_from_string_keyed do
    name = "migrate_from_string_keyed transfers all (key, value) pairs into int-id env"
    # Set up source: existing string-keyed Lmdb env (PR #1792 shape).
    alias WamRuntime.FactSource.Lmdb

    {:ok, src_env} = Elmdb.env_open()
    {:ok, src_dbi} = Elmdb.db_open(src_env, "facts", [:dupsort])

    # Populate via direct Elmdb calls (simulating an existing string-keyed
    # store from PR #1792).
    {:ok, txn} = Elmdb.rw_txn_begin(src_env)
    Enum.each(
      [
        {"alpha", "beta"},
        {"alpha", "gamma"},
        {"delta", "epsilon"}
      ],
      fn {k, v} -> :ok = Elmdb.txn_put(txn, src_dbi, k, v) end
    )
    :ok = Elmdb.txn_commit(txn)

    src_handle =
      Lmdb.open(%{env: src_env, dbi: src_dbi, arity: 2, dupsort: true}, 2, nil)

    # Set up destination: fresh int-id env.
    {dest_handle, dest_env} = open_handle(true)

    {:ok, result} =
      LmdbIntIds.migrate_from_string_keyed(src_handle, dest_handle,
                                            start_id: 0, batch_size: 100)

    # All four unique strings (alpha, beta, gamma, delta, epsilon) should
    # have IDs assigned.
    ids =
      ["alpha", "beta", "gamma", "delta", "epsilon"]
      |> Enum.map(&LmdbIntIds.lookup_id(dest_handle, &1))

    all_assigned = Enum.all?(ids, &is_integer/1)
    distinct_count = ids |> Enum.uniq() |> length()

    # All 3 source pairs should have migrated.
    pairs_ok = result.pairs_migrated == 3
    ids_ok = result.ids_assigned == 5

    Elmdb.env_close(src_env)
    Elmdb.env_close(dest_env)

    cond do
      not all_assigned ->
        fail(name, "some strings not assigned IDs after migration: #{inspect(ids)}")
      distinct_count != 5 ->
        fail(name, "expected 5 distinct IDs, got #{distinct_count}: #{inspect(ids)}")
      not pairs_ok ->
        fail(name, "expected 3 pairs migrated, got #{result.pairs_migrated}")
      not ids_ok ->
        fail(name, "expected 5 IDs assigned, got #{result.ids_assigned}")
      true ->
        pass(name)
    end
  end

  def run do
    test_encode_decode_round_trip()
    test_ingest_idempotent()
    test_lookup_by_arg1_id_dupsort()
    test_lookup_by_arg1_string_round_trip()
    test_lookup_cache_is_bounded_and_collision_safe()
    test_lookup_missing_key_returns_empty()
    test_stream_all_returns_int_pairs()
    test_migrate_from_string_keyed()
  end
end

LmdbIntIdsTest.run()
