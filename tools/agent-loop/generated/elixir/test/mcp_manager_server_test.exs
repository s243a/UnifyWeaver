defmodule AgentLoop.MCPManagerServerTest do
  use ExUnit.Case, async: true

  alias AgentLoop.MCPManagerServer

  setup do
    {:ok, pid} = MCPManagerServer.start_link()
    %{server: pid}
  end

  test "list_tools returns empty list initially", %{server: s} do
    assert MCPManagerServer.list_tools(s) == []
  end

  test "get_state returns MCPManager struct", %{server: s} do
    state = MCPManagerServer.get_state(s)
    assert %AgentLoop.MCPManager{} = state
    assert state.servers == %{}
    assert state.tools == %{}
  end
end
