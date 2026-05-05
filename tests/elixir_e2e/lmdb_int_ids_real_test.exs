# End-to-end test for WamRuntime.FactSource.LmdbIntIds against a real
# LMDB environment via the Hex :elmdb package.
#
# The generated runtime intentionally resolves a module named `Elmdb`
# indirectly so it stays dependency-free for drivers that do not use
# LMDB. The Hex package exposes the Erlang module `:elmdb`, so this
# test supplies the small bridge module that a real driver would own.
#
# If :elmdb cannot be downloaded/compiled in the local environment,
# the script prints a [SKIP] marker and exits 0. If :elmdb is available,
# runtime behavior failures print [FAIL] and exit non-zero.

defmodule RealElmdbDependency do
  def ensure! do
    try do
      Mix.install([{:elmdb, "~> 0.4.1"}], consolidate_protocols: false)
      :ok
    rescue
      exception ->
        IO.puts("[SKIP] real Elmdb dependency unavailable: #{Exception.message(exception)}")
        System.halt(0)
    catch
      kind, reason ->
        IO.puts("[SKIP] real Elmdb dependency unavailable: #{inspect({kind, reason})}")
        System.halt(0)
    end
  end
end

RealElmdbDependency.ensure!()
Code.require_file("lib/wam_runtime.ex", __DIR__)

defmodule Elmdb do
  @moduledoc false

  def env_open(path, opts) when is_binary(path) do
    :elmdb.env_open(String.to_charlist(path), opts)
  end

  def env_close(env), do: :elmdb.env_close(env)

  def db_open(env, name, opts) do
    :elmdb.db_open(env, to_db_name(name), normalize_db_opts(opts))
  end

  def rw_txn_begin(env), do: :elmdb.txn_begin(env)
  def ro_txn_begin(env), do: :elmdb.ro_txn_begin(env)
  def txn_commit(txn), do: :elmdb.txn_commit(txn)
  def txn_abort(txn), do: :elmdb.txn_abort(txn)
  def ro_txn_commit(txn), do: :elmdb.ro_txn_commit(txn)
  def txn_get(txn, dbi, key) do
    bin_key = to_bin(key)

    try do
      :elmdb.txn_get(txn, dbi, bin_key)
    rescue
      ArgumentError -> :elmdb.ro_txn_get(txn, dbi, bin_key)
    end
  end
  def txn_put(txn, dbi, key, value), do: :elmdb.txn_put(txn, dbi, to_bin(key), to_bin(value))
  def ro_txn_cursor_open(txn, dbi), do: :elmdb.ro_txn_cursor_open(txn, dbi)
  def ro_txn_cursor_close(cur), do: :elmdb.ro_txn_cursor_close(cur)

  def ro_txn_cursor_get(cur, op, arg) do
    :elmdb.ro_txn_cursor_get(cur, normalize_cursor_op(op, arg))
  end

  defp to_db_name(name) when is_binary(name), do: name
  defp to_db_name(name) when is_atom(name), do: Atom.to_string(name)

  defp to_bin(value) when is_binary(value), do: value
  defp to_bin(value) when is_list(value), do: :erlang.iolist_to_binary(value)

  defp normalize_db_opts(opts) do
    opts
    |> Enum.map(fn
      :dupsort -> :dup_sort
      other -> other
    end)
    |> add_create()
    |> Enum.uniq()
  end

  defp add_create(opts), do: [:create | opts]

  defp normalize_cursor_op(:set, key), do: {:set, key}
  defp normalize_cursor_op(op, _arg), do: op
end

defmodule LmdbIntIdsRealTest do
  @moduledoc false

  alias WamRuntime.FactSource.Lmdb
  alias WamRuntime.FactSource.LmdbIntIds

  defp pass(name), do: IO.puts("[PASS] #{name}")

  defp fail(name, reason) do
    IO.puts("[FAIL] #{name}: #{reason}")
    System.halt(1)
  end

  defp with_envs(name, fun) do
    root = Path.join(System.tmp_dir!(), "uw_elixir_lmdb_real_#{System.unique_integer([:positive])}")
    File.rm_rf!(root)
    File.mkdir_p!(root)

    try do
      fun.(root)
    after
      File.rm_rf!(root)
    end
  rescue
    exception ->
      fail(name, Exception.format(:error, exception, __STACKTRACE__))
  catch
    kind, reason ->
      fail(name, inspect({kind, reason}))
  end

  defp open_int_handle(root, suffix, dupsort) do
    path = Path.join(root, suffix)
    {:ok, env} = Elmdb.env_open(path, [{:map_size, 16 * 1024 * 1024}, {:max_dbs, 8}])
    fact_opts = if dupsort, do: [:dupsort], else: []
    {:ok, facts} = Elmdb.db_open(env, "facts", fact_opts)
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
          dupsort: dupsort
        },
        2,
        nil
      )

    {handle, env}
  end

  defp open_string_handle(root, suffix, dupsort) do
    path = Path.join(root, suffix)
    {:ok, env} = Elmdb.env_open(path, [{:map_size, 16 * 1024 * 1024}, {:max_dbs, 4}])
    fact_opts = if dupsort, do: [:dupsort], else: []
    {:ok, facts} = Elmdb.db_open(env, "facts", fact_opts)
    handle = Lmdb.open(%{env: env, dbi: facts, arity: 2, dupsort: dupsort}, 2, nil)
    {handle, env, facts}
  end

  def test_unique_ingest_lookup_stream do
    name = "real :elmdb unique-key ingest, lookup, stream_all"

    with_envs(name, fn root ->
      {handle, env} = open_int_handle(root, "unique", false)
      {:ok, result} = LmdbIntIds.ingest_pairs(handle, [{"alpha", "beta"}, {"gamma", "delta"}])

      alpha_id = LmdbIntIds.lookup_id(handle, "alpha")
      beta_id = LmdbIntIds.lookup_id(handle, "beta")
      alpha_rows = LmdbIntIds.lookup_by_arg1_id(handle, alpha_id, nil)
      stream = LmdbIntIds.stream_all(handle, nil)
      Elmdb.env_close(env)

      cond do
        result.pairs_seen != 2 -> fail(name, "expected 2 pairs, got #{inspect(result)}")
        result.new_ids != 4 -> fail(name, "expected 4 new ids, got #{inspect(result)}")
        alpha_rows != [{alpha_id, beta_id}] -> fail(name, "lookup mismatch: #{inspect(alpha_rows)}")
        length(stream) != 2 -> fail(name, "stream_all length mismatch: #{inspect(stream)}")
        true -> pass(name)
      end
    end)
  end

  def test_dupsort_lookup do
    name = "real :elmdb dupsort lookup_by_arg1_id returns all values"

    with_envs(name, fn root ->
      {handle, env} = open_int_handle(root, "dupsort", true)

      {:ok, _} =
        LmdbIntIds.ingest_pairs(handle, [
          {"alpha", "beta"},
          {"alpha", "gamma"},
          {"alpha", "delta"}
        ])

      alpha_id = LmdbIntIds.lookup_id(handle, "alpha")
      expected =
        ["beta", "gamma", "delta"]
        |> Enum.map(&LmdbIntIds.lookup_id(handle, &1))
        |> MapSet.new()

      actual =
        handle
        |> LmdbIntIds.lookup_by_arg1_id(alpha_id, nil)
        |> Enum.map(fn {^alpha_id, value_id} -> value_id end)
        |> MapSet.new()

      Elmdb.env_close(env)

      if MapSet.equal?(actual, expected) do
        pass(name)
      else
        fail(name, "expected #{inspect(expected)}, got #{inspect(actual)}")
      end
    end)
  end

  def test_migrate_from_string_keyed do
    name = "real :elmdb migrate_from_string_keyed copies string LMDB into int-id LMDB"

    with_envs(name, fn root ->
      {source, src_env, src_dbi} = open_string_handle(root, "source", true)

      {:ok, txn} = Elmdb.rw_txn_begin(src_env)
      :ok = Elmdb.txn_put(txn, src_dbi, "alpha", "beta")
      :ok = Elmdb.txn_put(txn, src_dbi, "alpha", "gamma")
      :ok = Elmdb.txn_put(txn, src_dbi, "delta", "epsilon")
      :ok = Elmdb.txn_commit(txn)

      {dest, dest_env} = open_int_handle(root, "dest", true)
      {:ok, result} = LmdbIntIds.migrate_from_string_keyed(source, dest, batch_size: 2)

      migrated_alpha = LmdbIntIds.lookup_by_arg1(dest, "alpha", nil) |> MapSet.new()
      Elmdb.env_close(src_env)
      Elmdb.env_close(dest_env)

      cond do
        result.pairs_migrated != 3 -> fail(name, "expected 3 migrated pairs, got #{inspect(result)}")
        result.ids_assigned != 5 -> fail(name, "expected 5 ids assigned, got #{inspect(result)}")
        not MapSet.equal?(migrated_alpha, MapSet.new([{"alpha", "beta"}, {"alpha", "gamma"}])) ->
          fail(name, "alpha rows mismatch: #{inspect(migrated_alpha)}")
        true ->
          pass(name)
      end
    end)
  end

  def run do
    test_unique_ingest_lookup_stream()
    test_dupsort_lookup()
    test_migrate_from_string_keyed()
  end
end

LmdbIntIdsRealTest.run()
