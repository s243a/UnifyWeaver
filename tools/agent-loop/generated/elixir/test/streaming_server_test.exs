defmodule AgentLoop.StreamingServerTest do
  use ExUnit.Case, async: true

  alias AgentLoop.StreamingServer

  setup do
    {:ok, pid} = StreamingServer.start_link()
    %{server: pid}
  end

  test "initial summary is zero", %{server: s} do
    summary = StreamingServer.format_summary(s)
    assert summary =~ "0 tokens"
  end

  test "on_token increments counters", %{server: s} do
    StreamingServer.on_token(s, "hello")
    state = StreamingServer.get_state(s)
    assert state.token_count == 1
    assert state.char_count == 5
  end

  test "reset clears counters", %{server: s} do
    StreamingServer.on_token(s, "hi")
    StreamingServer.reset(s)
    state = StreamingServer.get_state(s)
    assert state.token_count == 0
  end
end
