defmodule AgentLoop.SessionsTest do
  use ExUnit.Case, async: true

  alias AgentLoop.Sessions

  setup do
    dir = Path.join(System.tmp_dir!(), "agent_loop_test_sessions_#{:rand.uniform(999999)}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    %{state: %Sessions{sessions_dir: dir}}
  end

  test "session_path builds correct path", %{state: state} do
    path = Sessions.session_path(state, "abc123")
    assert String.ends_with?(path, "abc123.json")
  end

  test "session_exists returns false for missing session", %{state: state} do
    refute Sessions.session_exists(state, "nonexistent")
  end

  test "save and load round-trip", %{state: state} do
    :ok = Sessions.save_session(state, "test1", "Test Session", [%{"role" => "user", "content" => "hello"}])
    assert Sessions.session_exists(state, "test1")
    {:ok, data} = Sessions.load_session(state, "test1")
    assert data["name"] == "Test Session"
    assert length(data["messages"]) == 1
  end

  test "list_sessions returns saved sessions", %{state: state} do
    :ok = Sessions.save_session(state, "s1", "Session 1", [])
    :ok = Sessions.save_session(state, "s2", "Session 2", [])
    sessions = Sessions.list_sessions(state)
    assert length(sessions) == 2
  end

  test "delete_session removes file", %{state: state} do
    :ok = Sessions.save_session(state, "del1", "To Delete", [])
    assert Sessions.session_exists(state, "del1")
    :ok = Sessions.delete_session(state, "del1")
    refute Sessions.session_exists(state, "del1")
  end
end
