defmodule AgentLoop.ToolsTest do
  use ExUnit.Case, async: true

  alias AgentLoop.Tools

  test "all returns non-empty list" do
    assert length(Tools.all()) > 0
  end

  test "lookup bash tool" do
    tool = Tools.lookup(:bash)
    assert tool != nil
    assert tool.name == :bash
    assert tool.confirm == true
  end

  test "lookup read tool is non-destructive" do
    tool = Tools.lookup(:read)
    assert tool != nil
    assert tool.confirm == false
  end

  test "destructive? checks destructive tools" do
    assert Tools.destructive?("bash")
    refute Tools.destructive?("read")
  end
end
