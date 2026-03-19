defmodule AgentLoop.CostTrackerTest do
  use ExUnit.Case, async: true

  alias AgentLoop.CostTracker

  test "new struct has zero costs" do
    state = %CostTracker{}
    assert state.total_cost == 0.0
    assert state.total_input_tokens == 0
    assert state.total_output_tokens == 0
    assert state.records == []
  end

  test "is_over_budget returns false for zero budget (unlimited)" do
    state = %CostTracker{}
    refute CostTracker.is_over_budget(state, 0)
  end

  test "is_over_budget returns false when under budget" do
    state = %CostTracker{total_cost: 5.0}
    refute CostTracker.is_over_budget(state, 10.0)
  end

  test "is_over_budget returns true when over budget" do
    state = %CostTracker{total_cost: 15.0}
    assert CostTracker.is_over_budget(state, 10.0)
  end

  test "budget_remaining returns -1 for unlimited" do
    state = %CostTracker{}
    assert CostTracker.budget_remaining(state, 0) == -1.0
  end

  test "budget_remaining calculates correctly" do
    state = %CostTracker{total_cost: 3.0}
    assert CostTracker.budget_remaining(state, 10.0) == 7.0
  end

  test "reset clears all fields" do
    state = %CostTracker{total_cost: 5.0, total_input_tokens: 100, records: [%{}]}
    reset = CostTracker.reset(state)
    assert reset.total_cost == 0.0
    assert reset.total_input_tokens == 0
    assert reset.total_output_tokens == 0
    assert reset.records == []
  end

  test "cost_compute calculates per-million pricing" do
    # 1000 tokens at $15/million = $0.015
    cost = CostTracker.cost_compute(1000, 15.0)
    assert_in_delta cost, 0.015, 0.0001
  end
end
