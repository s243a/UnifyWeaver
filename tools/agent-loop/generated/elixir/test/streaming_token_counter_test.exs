defmodule AgentLoop.StreamingTokenCounterTest do
  use ExUnit.Case, async: true

  alias AgentLoop.StreamingTokenCounter

  test "new counter starts at zero" do
    state = %StreamingTokenCounter{}
    assert state.token_count == 0
    assert state.char_count == 0
  end

  test "format_summary shows counts" do
    state = %StreamingTokenCounter{token_count: 42, char_count: 168}
    assert StreamingTokenCounter.format_summary(state) == "42 tokens, 168 chars"
  end
end
