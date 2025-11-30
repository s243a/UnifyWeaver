# feat(sources): XML Flattening and SQLite Export for Python Target

## Summary
This PR enhances the Python-based XML source plugin (`xml_source.pl`) to support **XML flattening** and **direct SQLite export**, bringing it closer to parity with the C# target's capabilities. This allows for efficient ETL pipelines where XML data is streamed, flattened into relational/JSON structures, and stored in a local database for further querying.

## Changes
- **New Options:**
    - `flatten(true)`: Converts XML elements into flat dictionaries (Attributes -> `@attr`, Text -> `text`, Children -> `tag`).
    - `sqlite_table('TableName')`: Specifies the target SQLite table.
    - `database('path.db')`: Specifies the SQLite database file.
- **Engine Logic:**
    - Forces `iterparse` engine (using `lxml`) when flattening is requested.
    - Implemented robust Python code generation that streams XML, flattens elements, and batch-inserts them as JSON into SQLite.
- **Validation:**
    - Added configuration validation for new options.
    - Ensures `lxml` is available before attempting execution.
- **Testing:**
    - Added `sqlite_flattening` test case in `tests/core/test_xml_source.pl` to verify the pipeline.

## Usage Example
```prolog
:- source(xml, products, [
    xml_file('data/products.xml'),
    tags(['product']),
    flatten(true),
    sqlite_table('products'),
    database('data.db')
]).
```

## Dependencies
- Requires `python3` and `lxml` (`pip install lxml`).
