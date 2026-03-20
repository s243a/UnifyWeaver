defmodule AgentLoop.ConfigTest do
  use ExUnit.Case, async: true

  alias AgentLoop.Config

  test "search_paths returns non-empty list" do
    paths = Config.search_paths()
    assert length(paths) > 0
  end

  test "defaults returns map with known keys" do
    defaults = Config.defaults()
    assert is_map(defaults)
    assert Map.has_key?(defaults, :max_iterations)
    assert Map.has_key?(defaults, :timeout)
  end

  test "api_key_env returns env var for known backend" do
    assert Config.api_key_env(:claude) == "ANTHROPIC_API_KEY"
  end

  test "api_key_env returns nil for unknown backend" do
    assert Config.api_key_env(:unknown) == nil
  end
end
