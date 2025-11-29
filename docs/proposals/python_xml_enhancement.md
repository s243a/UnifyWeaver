# Proposal: Enhanced Python XML Source with Flattening & SQLite Support

## Context
The C# target's `XmlStreamReader` offers powerful features like automatic flattening of XML fragments into dictionaries (handling attributes, namespaces, and CDATA) and optional nested projection. The current Python-based `xml_source.pl` primarily focuses on streaming raw XML fragments or extracting specific fields via Regex/AWK.

## Goal
Enhance the `xml_source` plugin (Python/lxml engine) to support:
1.  **XML Flattening:** Convert XML fragments into flat dictionaries (similar to C#).
2.  **SQLite Integration:** Directly insert these flattened records into a SQLite database.

## Proposed Features

### 1. Flattening Strategy
A new option `flatten(true)` or `mode(flatten)` will be added.
When enabled, the Python script (using `lxml`) will process each XML element:
- **Attributes:** Mapped to keys with `@` prefix (e.g., `<node id="1">` -> `{'@id': '1'}`).
- **Text Content:** Mapped to `text` key or element name if leaf.
- **Child Elements:** 
    - If leaf node (text only), mapped to key=tag, value=text.
    - If multiple children with same tag, mapped to list.
- **Namespaces:** Preserved as `prefix:local` keys where possible, or user-defined mapping.
- **CDATA:** Automatically stripped/extracted.

### 2. SQLite Export
A new option `sqlite_table(TableName)` and `database(DbPath)` (reusing existing `python_source` patterns) will enable direct insertion.
- **Auto-Schema:** If schema is not provided, the first record determines columns (or use JSON column for flexibility).
- **Explicit Schema:** Use `columns([...])` to map XML paths to DB columns.

### 3. Usage Example

```prolog
:- source(xml, products, [
    xml_file('data/products.xml'),
    tags(['product']),
    engine(iterparse),
    
    % New Options
    flatten(true),
    sqlite_table('products'),
    database('data.db'),
    
    % Optional: Explicit mapping
    columns([
        '@id' -> 'product_id',
        'name' -> 'product_name',
        'price' -> 'price'
    ])
]).
```

## Implementation Plan
1.  **Update `xml_source.pl`**:
    - Add validation for new options (`flatten`, `sqlite_table`).
    - Modify `generate_lxml_python_code` to include the flattening logic and SQLite insertion code.
2.  **Python Logic**:
    - Implement a `flatten_element(elem)` Python function.
    - Implement `sqlite3` batch insertion loop.
3.  **Testing**:
    - Add test case in `tests/core/test_xml_source.pl` with a sample XML and SQLite query verification.

## Benefits
- Parity with C# target capabilities.
- Efficient ETL from XML directly to SQL without intermediate CSV/JSON steps.
- Leverage robust `lxml` parsing for complex XML documents.
