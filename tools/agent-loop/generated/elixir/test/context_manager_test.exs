defmodule AgentLoop.ContextManagerTest do
  use ExUnit.Case, async: true

  alias AgentLoop.ContextManager

  test "new context is empty" do
    state = %ContextManager{}
    assert ContextManager.is_empty(state)
    assert ContextManager.len(state) == 0
  end

  test "clear resets messages" do
    state = %ContextManager{messages: [%{"role" => "user", "content" => "hi"}]}
    cleared = ContextManager.clear(state)
    assert cleared.messages == []
    assert ContextManager.is_empty(cleared)
  end

  test "len counts messages" do
    state = %ContextManager{messages: [%{}, %{}, %{}]}
    assert ContextManager.len(state) == 3
  end

  test "estimate_tokens uses char/4 heuristic" do
    msg = %{"content" => String.duplicate("a", 400)}
    state = %ContextManager{messages: [msg]}
    assert ContextManager.estimate_tokens(state) == 100
  end
end
