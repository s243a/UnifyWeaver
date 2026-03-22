defmodule AgentLoop.RouterTest do
  use ExUnit.Case, async: true
  use Plug.Test

  alias AgentLoop.Router

  @opts Router.init([])

  test "GET /api/health returns 200" do
    conn = conn(:get, "/api/health") |> Router.call(@opts)
    assert conn.status == 200
    body = Jason.decode!(conn.resp_body)
    assert body["status"] == "ok"
  end

  test "GET /api/tools returns tool list" do
    conn = conn(:get, "/api/tools") |> Router.call(@opts)
    assert conn.status == 200
    tools = Jason.decode!(conn.resp_body)
    assert is_list(tools)
    assert length(tools) == 4
  end

  test "GET /api/commands returns command list" do
    conn = conn(:get, "/api/commands") |> Router.call(@opts)
    assert conn.status == 200
    commands = Jason.decode!(conn.resp_body)
    assert is_list(commands)
  end

  test "POST /api/tools/read without params returns 400" do
    conn =
      conn(:post, "/api/tools/read", %{})
      |> put_req_header("content-type", "application/json")
      |> Router.call(@opts)
    assert conn.status == 400
    body = Jason.decode!(conn.resp_body)
    assert body["error"] == "missing_params"
  end

  test "unknown route returns 404" do
    conn = conn(:get, "/api/nonexistent") |> Router.call(@opts)
    assert conn.status == 404
  end
end
