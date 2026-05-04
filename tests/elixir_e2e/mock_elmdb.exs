defmodule MockElmdb do
  @moduledoc """
  In-memory fake of the `:elmdb` API surface that
  `WamRuntime.FactSource.Lmdb` and `WamRuntime.FactSource.LmdbIntIds`
  call through `Module.concat([Elmdb])`. Because of that indirect
  resolution, defining a module named `Elmdb` (no namespace) in this
  test file is enough to redirect all kernel-runtime calls to this
  mock — no production-code change required.

  Scope: covers the function surface the LmdbIntIds adaptor calls.

      env_open/1, env_close/1
      db_open/3
      rw_txn_begin/1, txn_commit/1, txn_abort/1
      ro_txn_begin/1, ro_txn_commit/1
      txn_get/3, txn_put/4
      ro_txn_cursor_open/2, ro_txn_cursor_close/1
      ro_txn_cursor_get/3   (with :first / :next / :set / :next_dup)

  Backed by an Agent that holds:

      %{
        databases: %{db_handle => %{dupsort: bool, data: [{key, val}, ...] sorted by key}},
        cursors:   %{cur_handle => %{db_handle, last_key, last_dup_idx}},
        next_db:   integer,
        next_cur:  integer
      }

  Sorted by binary memcmp on key (Elixir/Erlang's natural binary
  ordering) so cursor walks return entries in the same order LMDB
  would. For dupsort tables a single key maps to a list of values,
  which is the on-disk semantics LMDB uses with MDB_DUPSORT.

  Caveats vs real `:elmdb`:
  - Single-process, no real transaction isolation. ro_txn vs rw_txn
    is treated identically; reads always see writes.
  - No error returns for invalid handles, env capacity exceeded,
    page splits, etc. — those paths aren't exercised here.
  - dupsort comparator is the default memcmp. For 8-byte BE u64
    keys/values this matches integer order, which is what LmdbIntIds
    relies on. If you change the encoding, double-check this.

  Use solely as a test fixture. Real-world correctness of LmdbIntIds
  still requires a `:elmdb`-backed integration test (deferred until
  Hex.pm is reachable).
  """

  use Agent

  # ---- env management ----

  def env_open(_opts \\ []) do
    {:ok, agent} =
      Agent.start_link(fn ->
        %{databases: %{}, cursors: %{}, next_db: 0, next_cur: 0}
      end)
    {:ok, agent}
  end

  def env_close(env) do
    Agent.stop(env)
    :ok
  end

  # ---- database management ----

  def db_open(env, _name, opts) do
    dupsort? = Enum.member?(opts, :dupsort)
    {:ok,
     Agent.get_and_update(env, fn st ->
       db = st.next_db
       new_st =
         st
         |> Map.put(:next_db, db + 1)
         |> put_in([:databases, db], %{dupsort: dupsort?, data: []})
       {db, new_st}
     end)}
  end

  # ---- transactions (no isolation in the mock) ----

  def rw_txn_begin(env), do: {:ok, {:rw, env}}
  def ro_txn_begin(env), do: {:ok, {:ro, env}}
  def txn_commit(_txn), do: :ok
  def txn_abort(_txn), do: :ok
  def ro_txn_commit(_txn), do: :ok

  # ---- key/value API ----

  def txn_get({_kind, env}, dbi, key) when is_binary(key) do
    Agent.get(env, fn st ->
      case get_in(st, [:databases, dbi]) do
        nil -> :not_found
        %{dupsort: false, data: data} ->
          case List.keyfind(data, key, 0) do
            {^key, val} -> {:ok, val}
            nil -> :not_found
          end
        %{dupsort: true, data: data} ->
          # Real LMDB returns the FIRST value for the key on txn_get
          # under dupsort. We mimic that.
          case List.keyfind(data, key, 0) do
            {^key, [val | _]} -> {:ok, val}
            {^key, val} when is_binary(val) -> {:ok, val}
            nil -> :not_found
          end
      end
    end)
  end

  def txn_put({:rw, env}, dbi, key, value) when is_binary(key) and is_binary(value) do
    Agent.update(env, fn st ->
      update_in(st, [:databases, dbi], fn
        %{dupsort: false, data: data} = db ->
          new_data = data |> List.keystore(key, 0, {key, value}) |> Enum.sort()
          %{db | data: new_data}

        %{dupsort: true, data: data} = db ->
          new_data =
            case List.keyfind(data, key, 0) do
              nil ->
                List.keystore(data, key, 0, {key, [value]}) |> Enum.sort()
              {^key, vals} when is_list(vals) ->
                # Insert sorted, dedup. LMDB MDB_DUPSORT is set-like by
                # default (no duplicate values per key).
                new_vals = Enum.uniq([value | vals]) |> Enum.sort()
                List.keystore(data, key, 0, {key, new_vals}) |> Enum.sort()
              {^key, single} when is_binary(single) ->
                new_vals = if single == value, do: [single], else: Enum.sort([single, value])
                List.keystore(data, key, 0, {key, new_vals}) |> Enum.sort()
            end
          %{db | data: new_data}
      end)
    end)
    :ok
  end

  # ---- cursors ----

  def ro_txn_cursor_open({_kind, env}, dbi) do
    cur =
      Agent.get_and_update(env, fn st ->
        c = st.next_cur
        new_st =
          st
          |> Map.put(:next_cur, c + 1)
          |> put_in([:cursors, c], %{
            env: env,
            dbi: dbi,
            last_key: nil,
            last_dup_idx: 0
          })
        {c, new_st}
      end)
    {:ok, {env, cur}}
  end

  def ro_txn_cursor_close({env, cur}) do
    Agent.update(env, fn st -> update_in(st, [:cursors], &Map.delete(&1, cur)) end)
    :ok
  end

  def ro_txn_cursor_get({env, cur}, op, arg) do
    Agent.get_and_update(env, fn st ->
      cursor = get_in(st, [:cursors, cur])
      db = get_in(st, [:databases, cursor.dbi])
      do_cursor_get(st, cur, cursor, db, op, arg)
    end)
  end

  defp do_cursor_get(st, cur, _cursor, %{data: []}, _op, _arg), do: {:not_found, st}

  defp do_cursor_get(st, cur, _cursor, %{dupsort: false, data: data}, :first, _arg) do
    [{k, v} | _] = data
    new_st = put_in(st, [:cursors, cur, :last_key], k)
    {{:ok, k, v}, new_st}
  end

  defp do_cursor_get(st, cur, %{last_key: nil}, %{dupsort: false} = db, :next, arg),
    do: do_cursor_get(st, cur, %{last_key: nil}, db, :first, arg)

  defp do_cursor_get(st, cur, %{last_key: lk}, %{dupsort: false, data: data}, :next, _arg) do
    case Enum.drop_while(data, fn {k, _} -> k <= lk end) do
      [] -> {:not_found, st}
      [{k, v} | _] ->
        new_st = put_in(st, [:cursors, cur, :last_key], k)
        {{:ok, k, v}, new_st}
    end
  end

  # dupsort cursor: :first
  defp do_cursor_get(st, cur, _cursor, %{dupsort: true, data: data}, :first, _arg) do
    [{k, vals} | _] = data
    case vals do
      [v | _] ->
        new_st =
          st
          |> put_in([:cursors, cur, :last_key], k)
          |> put_in([:cursors, cur, :last_dup_idx], 0)
        {{:ok, k, v}, new_st}
      v when is_binary(v) ->
        new_st =
          st
          |> put_in([:cursors, cur, :last_key], k)
          |> put_in([:cursors, cur, :last_dup_idx], 0)
        {{:ok, k, v}, new_st}
    end
  end

  # dupsort cursor: :next - real LMDB MDB_NEXT walks ALL (key, value) pairs
  # in order, INCLUDING duplicates. So for a dupsort cursor we first try to
  # advance within the current key's dup list; only if exhausted do we
  # advance to the next distinct key.
  defp do_cursor_get(st, cur, %{last_key: nil}, %{dupsort: true} = db, :next, arg),
    do: do_cursor_get(st, cur, %{last_key: nil, last_dup_idx: 0}, db, :first, arg)

  defp do_cursor_get(st, cur, %{last_key: lk, last_dup_idx: idx},
                     %{dupsort: true, data: data}, :next, _arg) do
    case List.keyfind(data, lk, 0) do
      nil ->
        # Current key vanished — slide to next.
        advance_to_next_key(st, cur, lk, data)
      {^lk, vals} ->
        vals_list = if is_binary(vals), do: [vals], else: vals
        next_idx = idx + 1
        case Enum.at(vals_list, next_idx) do
          nil ->
            advance_to_next_key(st, cur, lk, data)
          v ->
            new_st = put_in(st, [:cursors, cur, :last_dup_idx], next_idx)
            {{:ok, lk, v}, new_st}
        end
    end
  end

  defp advance_to_next_key(st, cur, lk, data) do
    case Enum.drop_while(data, fn {k, _} -> k <= lk end) do
      [] -> {:not_found, st}
      [{k, vals} | _] ->
        v =
          case vals do
            [first | _] -> first
            single when is_binary(single) -> single
          end
        new_st =
          st
          |> put_in([:cursors, cur, :last_key], k)
          |> put_in([:cursors, cur, :last_dup_idx], 0)
        {{:ok, k, v}, new_st}
    end
  end

  # dupsort cursor: :set - position at exact key, return first dup
  defp do_cursor_get(st, cur, _cursor, %{dupsort: true, data: data}, :set, key) do
    case List.keyfind(data, key, 0) do
      nil -> {:not_found, st}
      {^key, vals} ->
        v =
          case vals do
            [first | _] -> first
            single when is_binary(single) -> single
          end
        new_st =
          st
          |> put_in([:cursors, cur, :last_key], key)
          |> put_in([:cursors, cur, :last_dup_idx], 0)
        {{:ok, key, v}, new_st}
    end
  end

  # dupsort cursor: :next_dup - advance to next value for current key
  defp do_cursor_get(st, cur, %{last_key: lk, last_dup_idx: idx},
                     %{dupsort: true, data: data}, :next_dup, _arg) do
    case List.keyfind(data, lk, 0) do
      nil -> {:not_found, st}
      {^lk, vals} ->
        vals_list = if is_binary(vals), do: [vals], else: vals
        next_idx = idx + 1
        case Enum.at(vals_list, next_idx) do
          nil -> {:not_found, st}
          v ->
            new_st = put_in(st, [:cursors, cur, :last_dup_idx], next_idx)
            {{:ok, lk, v}, new_st}
        end
    end
  end

  # Catch-all for unimplemented cursor ops — make the gap obvious.
  defp do_cursor_get(st, _cur, _cursor, _db, op, _arg) do
    raise "MockElmdb: cursor op #{inspect(op)} not implemented"
  end
end
