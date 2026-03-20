defmodule AgentLoop.CacheServerTest do
  use ExUnit.Case, async: true

  alias AgentLoop.CacheServer

  setup do
    {:ok, pid} = CacheServer.start_link()
    %{server: pid}
  end

  test "get returns nil for missing key", %{server: s} do
    assert CacheServer.get(s, "missing") == nil
  end

  test "put and get round-trip", %{server: s} do
    CacheServer.put(s, "k1", "v1")
    assert CacheServer.get(s, "k1") == "v1"
  end

  test "has_key? after put", %{server: s} do
    refute CacheServer.has_key?(s, "k2")
    CacheServer.put(s, "k2", "v2")
    assert CacheServer.has_key?(s, "k2")
  end

  test "len tracks cache size", %{server: s} do
    assert CacheServer.len(s) == 0
    CacheServer.put(s, "a", "1")
    assert CacheServer.len(s) == 1
  end

  test "clear empties the cache", %{server: s} do
    CacheServer.put(s, "a", "1")
    CacheServer.clear(s)
    assert CacheServer.len(s) == 0
  end
end
