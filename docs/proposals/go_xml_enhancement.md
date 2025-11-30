# Proposal: Go Target XML Streaming & Flattening

## Overview
To align the Go target with C# and Python capabilities, we propose adding **native XML streaming and flattening** support. This allows the Go target to ingest XML files, flatten them into `map[string]interface{}` records, and process them (filter/store) efficiently.

## Architecture

We will extend `go_target.pl` to support a new compilation mode `compile_xml_input_mode`.

### 1. XML Input Mode
When `input_source(xml(File, Tags))` is present, the compiler will generate a Go program that:
1.  Opens the XML file.
2.  Uses `encoding/xml.Decoder` to stream tokens.
3.  Matches `StartElement` against the requested `Tags`.
4.  Decodes matching elements into a generic `Node` struct.
5.  Flattens the `Node` into a `map[string]interface{}` (JSON-like structure).
    - Attributes -> `@attr`
    - Text -> `text`
    - Children -> `tag` (text content) or nested map.

### 2. Flattening Logic in Go
Since Go is statically typed, we will decode into a recursive struct and then convert.

```go
type XmlNode struct {
    XMLName xml.Name
    Attrs   []xml.Attr `xml:",any,attr"`
    Content string     `xml:",chardata"`
    Nodes   []XmlNode  `xml:",any"`
}

func Flatten(n XmlNode) map[string]interface{} {
    m := make(map[string]interface{})
    for _, a := range n.Attrs {
        m["@"+a.Name.Local] = a.Value
    }
    if strings.TrimSpace(n.Content) != "" {
        m["text"] = strings.TrimSpace(n.Content)
    }
    for _, child := range n.Nodes {
        // Simplified: map child tag to child text or flatten child
        m[child.XMLName.Local] = Flatten(child) 
        // Note: Logic needs to handle lists of children with same tag
    }
    return m
}
```

### 3. Integration
The flattened map will be treated exactly like the `data` map in `compile_json_input_mode`. This means existing logic for:
- Field extraction (`field := data["key"]`)
- Database storage (`bbolt` Put)
- Output generation (JSONL)
will work seamlessly with XML input.

## Implementation Plan
1.  **Update `go_target.pl`**: Add `compile_xml_input_mode`.
2.  **Generate Go Code**: Implement `generate_xml_reader_go`.
3.  **Testing**: Add `tests/core/test_go_xml_integration.pl`.

## Future Work (Semantic)
After XML is working, we can explore **ONNX Runtime Go** bindings or `bbolt` vector storage to match the Python/C# semantic runtime.
