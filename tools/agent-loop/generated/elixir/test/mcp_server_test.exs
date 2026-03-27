defmodule AgentLoop.MCPServerTest do
  use ExUnit.Case, async: true

  alias AgentLoop.MCPServer

  setup do
    {:ok, pid} = MCPServer.start_link()
    %{server: pid}
  end

  test "next_request_id increments", %{server: s} do
    id1 = MCPServer.next_request_id(s)
    id2 = MCPServer.next_request_id(s)
    assert id2 > id1
  end

  test "get_state returns MCPClient struct", %{server: s} do
    state = MCPServer.get_state(s)
    assert %AgentLoop.MCPClient{} = state
  end
end
