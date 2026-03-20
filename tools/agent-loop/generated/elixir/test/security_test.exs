defmodule AgentLoop.SecurityTest do
  use ExUnit.Case, async: true

  alias AgentLoop.Security

  test "profile_names returns all profiles" do
    names = Security.profile_names()
    assert :open in names
    assert :cautious in names
    assert :guarded in names
    assert :paranoid in names
  end

  test "profile returns valid profile struct" do
    profile = Security.profile(:cautious)
    assert profile != nil
    assert profile.name == :cautious
  end

  test "profile returns nil for unknown" do
    assert Security.profile(:nonexistent) == nil
  end

  test "path_blocked? detects blocked paths" do
    assert Security.path_blocked?("/etc/shadow")
  end

  test "path_blocked? allows normal paths" do
    refute Security.path_blocked?("/home/user/code.py")
  end

  test "home_path_blocked? detects blocked home patterns" do
    assert Security.home_path_blocked?(".ssh/id_rsa")
  end
end
