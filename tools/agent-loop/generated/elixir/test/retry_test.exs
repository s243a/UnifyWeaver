defmodule AgentLoop.RetryTest do
  use ExUnit.Case, async: true

  alias AgentLoop.Retry

  test "is_retryable_status for 429" do
    assert Retry.is_retryable_status(429)
  end

  test "is_retryable_status for 500" do
    assert Retry.is_retryable_status(500)
  end

  test "is_retryable_status false for 200" do
    refute Retry.is_retryable_status(200)
  end

  test "is_retryable_status false for 404" do
    refute Retry.is_retryable_status(404)
  end

  test "compute_delay first attempt" do
    delay = Retry.compute_delay(1.0, 2.0, 1, 60.0)
    assert_in_delta delay, 1.0, 0.001
  end

  test "compute_delay exponential growth" do
    delay = Retry.compute_delay(1.0, 2.0, 4, 60.0)
    assert_in_delta delay, 8.0, 0.001
  end

  test "compute_delay caps at max_delay" do
    delay = Retry.compute_delay(1.0, 2.0, 10, 30.0)
    assert delay == 30.0
  end
end
