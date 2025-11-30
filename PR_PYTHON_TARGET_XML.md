# feat(python): Integrate Native XML Reader (lxml) into Python Target

## Summary
This PR integrates a native XML reader directly into the `python_target` compiler, enabling "holistic" processing where XML reading, flattening, and logic execution happen within a single Python process. This aligns the Python target with the C# target's architecture and avoids the overhead of pipe-based serialization (JSONL) for XML sources.

## Changes
- **`python_target.pl`**:
    - Added `read_xml_lxml` helper function to the generated Python code.
    - Updated `compile_predicate_to_python` (and sub-predicates) to handle `input_source(xml(File, Tags))` option.
    - Generates a `main()` function that initializes the XML reader directly if an XML source is specified.
- **Tests**:
    - Added `tests/core/test_python_xml_integration.pl` to verify that a predicate compiled with an XML source correctly reads, flattens, and processes XML data.

## Usage
```prolog
compile_predicate_to_python(my_pred/1, [
    input_source(xml('data.xml', ['record'])),
    mode(procedural)
], Code).
```

## Benefits
- **Performance**: Avoids serialization/deserialization (XML -> JSONL -> Python Dict) between processes.
- **Simplicity**: Generates a self-contained Python script that reads XML directly.
- **Parity**: Matches C# target's `XmlStreamReader` capability.
