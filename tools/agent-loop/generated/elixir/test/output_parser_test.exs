defmodule AgentLoop.OutputParserTest do
  use ExUnit.Case, async: true

  alias AgentLoop.OutputParser

  test "extract_fenced parses fenced JSON blocks" do
    text = "Some text\n```json\n{\"key\": \"value\"}\n```\nMore text"
    result = OutputParser.extract_fenced(text)
    assert [%{"key" => "value"}] = result
  end

  test "extract_bare parses bare JSON objects" do
    text = "Here is {\"a\": 1} inline"
    result = OutputParser.extract_bare(text)
    assert [%{"a" => 1}] = result
  end

  test "parse_response prefers fenced blocks" do
    text = "```json\n{\"tool\": \"read\"}\n```"
    {blocks, _raw, errors} = OutputParser.parse_response(text)
    assert length(blocks) == 1
    assert errors == []
  end
end
