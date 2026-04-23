# Interaction probe: Task.async_stream × CPS throw/catch (Tier-2 precondition 4).
#
# Run:  elixir examples/debug_tier2_async_stream_throw_catch.exs
# Exits 0 on pass, 1 on any failure.
#
# Verifies the single load-bearing correctness assumption behind
# docs/design/WAM_TIERED_LOWERING.md — that CPS-lowered branches which
# signal failure via `throw({:fail, state})` can be fanned out through
# Task.async_stream without killing sibling tasks or crashing the parent.
#
# Finding (OTP 28 / Elixir 1.19): a NAKED throw from a task body escapes
# as `{:nocatch, thrown}` and bubbles up through Task.Supervised.invoke_mfa,
# exiting the parent. Task.async_stream does NOT catch thrown values by
# default. The fix is to wrap each branch call-site in a try/catch inside
# the task function so thrown values become return values (empty list
# for {:fail, _}, wrapped list for {:return, _}). With that wrapper the
# stream emits one `{:ok, list}` per branch and Enum.flat_map collects
# solutions cleanly.
#
# The Tier-2 wrapper emission template must therefore include the
# per-branch try/catch layer. A later PR will revise the design doc.

defmodule Tier2AsyncStreamProbe do
  @moduledoc """
  Reproduces the per-branch shape the Tier-2 wrapper will use. Each
  test asserts a specific property of the interaction.
  """

  # The wrapper a Tier-2-emitted body will apply around each branch.
  # Converts CPS throws into return values so Task.async_stream sees
  # a normal result, not an abnormal exit.
  def run_branch(branch_fn, branch_state) do
    try do
      branch_fn.(branch_state)
    catch
      {:fail, _state} -> []
      {:return, result} when is_list(result) -> result
      {:return, result} -> [result]
    end
  end

  def run_all(branches, branch_state, opts \\ []) do
    max_c = Keyword.get(opts, :max_concurrency, 4)

    branches
    |> Task.async_stream(
      fn branch -> run_branch(branch, branch_state) end,
      ordered: Keyword.get(opts, :ordered, true),
      on_timeout: :kill_task,
      max_concurrency: max_c
    )
    |> Enum.to_list()
  end
end

defmodule Tier2ProbeRunner do
  @moduledoc "PASS/FAIL runner. Accumulates failures in a process dict."

  def init, do: Process.put(:failures, [])

  def assert(cond, label) do
    if cond do
      IO.puts("[PASS] #{label}")
    else
      IO.puts("[FAIL] #{label}")
      Process.put(:failures, [label | Process.get(:failures, [])])
    end
  end

  def finish do
    case Process.get(:failures, []) do
      [] ->
        IO.puts("\n=== All interaction probes passed ===")
        System.halt(0)

      fs ->
        IO.puts("\n=== #{length(fs)} failure(s) ===")
        Enum.each(Enum.reverse(fs), fn l -> IO.puts("  - #{l}") end)
        System.halt(1)
    end
  end
end

# ----- probe body -----

Tier2ProbeRunner.init()

state = %{cut_point: [], parallel_depth: 1}

# Test 1: mixed success + {:fail, _} branches all drain, no parent crash.
branches_1 = [
  fn _s -> [:a1, :a2] end,
  fn s -> throw({:fail, s}) end,
  fn _s -> [:c1] end,
  fn s -> throw({:fail, %{s | cut_point: [:abc]}}) end
]

raw_1 = Tier2AsyncStreamProbe.run_all(branches_1, state)
Tier2ProbeRunner.assert(length(raw_1) == 4, "all 4 branches drain to stream results")
Tier2ProbeRunner.assert(
  Enum.all?(raw_1, fn {:ok, v} -> is_list(v); _ -> false end),
  "every stream entry is {:ok, list} — no {:exit, _} leaks"
)

succ_1 =
  Enum.flat_map(raw_1, fn
    {:ok, list} when is_list(list) -> list
    _ -> []
  end)

Tier2ProbeRunner.assert(
  succ_1 == [:a1, :a2, :c1],
  "Enum.flat_map collects 3 solutions in order (ordered: true)"
)

# Test 2: {:return, result} throws are captured as a single-element list.
branches_2 = [
  fn _s -> [:x1] end,
  fn _s -> throw({:return, :single_answer}) end,
  fn _s -> [:z1, :z2] end
]

raw_2 = Tier2AsyncStreamProbe.run_all(branches_2, state)
succ_2 =
  Enum.flat_map(raw_2, fn
    {:ok, list} when is_list(list) -> list
    _ -> []
  end)

Tier2ProbeRunner.assert(
  succ_2 == [:x1, :single_answer, :z1, :z2],
  "{:return, result} throws convert to [result] and merge in place"
)

# Test 3: slow branch is not killed early. 50 ms sleep on one branch,
# quick failure on another — both outcomes must land.
branches_3 = [
  fn _s ->
    Process.sleep(50)
    [:slow_answer]
  end,
  fn s -> throw({:fail, s}) end,
  fn _s -> [:fast_answer] end
]

t_start = System.monotonic_time(:millisecond)
raw_3 = Tier2AsyncStreamProbe.run_all(branches_3, state)
t_end = System.monotonic_time(:millisecond)

succ_3 =
  Enum.flat_map(raw_3, fn
    {:ok, list} when is_list(list) -> list
    _ -> []
  end)

Tier2ProbeRunner.assert(
  Enum.sort(succ_3) == [:fast_answer, :slow_answer],
  "slow branch runs to completion alongside fast + failing siblings"
)

Tier2ProbeRunner.assert(
  t_end - t_start >= 50,
  "stream waited for slow branch (>=50ms elapsed, actual: #{t_end - t_start}ms)"
)

# Test 4: one throw does not kill siblings that were already mid-computation.
# 4 branches, half throw, half sleep+return. Assert all 2 successes land.
branches_4 = [
  fn s -> throw({:fail, s}) end,
  fn _s ->
    Process.sleep(10)
    [:s4_a]
  end,
  fn s -> throw({:fail, s}) end,
  fn _s ->
    Process.sleep(20)
    [:s4_b]
  end
]

raw_4 = Tier2AsyncStreamProbe.run_all(branches_4, state, ordered: false)
succ_4 =
  Enum.flat_map(raw_4, fn
    {:ok, list} when is_list(list) -> list
    _ -> []
  end)

Tier2ProbeRunner.assert(
  Enum.sort(succ_4) == [:s4_a, :s4_b],
  "siblings mid-computation are not killed by peer throws (ordered: false)"
)

Tier2ProbeRunner.finish()
