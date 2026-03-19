defmodule AgentLoop.ToolResultCacheTest do
  use ExUnit.Case, async: true

  alias AgentLoop.ToolResultCache

  test "new cache is empty" do
    state = %ToolResultCache{}
    assert Enum.count(state.cache) == 0
  end

  test "clear resets cache" do
    state = %ToolResultCache{cache: %{"key" => "val"}}
    cleared = ToolResultCache.clear(state)
    assert cleared.cache == %{}
  end

  test "len counts entries" do
    state = %ToolResultCache{cache: %{"a" => 1, "b" => 2}}
    assert ToolResultCache.len(state) == 2
  end
end
