defmodule AgentLoop.PricingTest do
  use ExUnit.Case, async: true

  alias AgentLoop.Pricing

  test "all returns non-empty list" do
    assert length(Pricing.all()) > 0
  end

  test "lookup known model" do
    assert {15.0, 75.0} = Pricing.lookup("opus")
  end

  test "lookup unknown model" do
    assert :unknown = Pricing.lookup("nonexistent-model")
  end

  test "pricing table has 16 entries" do
    assert length(Pricing.all()) == 16
  end
end
