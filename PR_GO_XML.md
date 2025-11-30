# feat(go): Native XML Input with Flattening

## Summary
This PR brings the Go target closer to parity with Python and C# by adding native XML input support. It introduces a compilation mode that uses `encoding/xml` to stream XML files and flatten them into generic maps, enabling seamless integration with UnifyWeaver's existing field extraction and database logic.

## Changes
- **`go_target.pl`**:
    - Added `compile_xml_input_mode/4` to handle `xml_input(true)` option.
    - Implemented `generate_xml_helpers/1` which provides:
        - `XmlNode` struct for recursive decoding.
        - `FlattenXML` function to convert `XmlNode` to `map[string]interface{}` (attributes -> `@attr`, text -> `text`, children -> `tag`).
    - Updated `compile_predicate_to_go` to dispatch to XML mode.
    - **Added `bbolt` support:** XML mode now supports `db_backend(bbolt)` option to store flattened records in a local key-value store.
    - **Added Stdin support:** If `xml_file(stdin)` is used (or no file specified), reads XML from standard input.
- **Tests**:
    - Added `tests/core/test_go_xml_integration.pl` verifying end-to-end XML streaming (file and stdin), flattening, and field extraction in Go.

## Usage
```prolog
compile_predicate_to_go(my_pred/2, [
    xml_input(true),
    xml_file('data.xml'), % or xml_file(stdin)
    tags(['item']),
    db_backend(bbolt),  % Optional: Store to DB
    db_file('data.db')
], Code).
```

## Notes
- The flattened structure is compatible with the existing JSON processing logic, allowing reuse of field mapping code.
- Supports recursive structures (lists of children).
