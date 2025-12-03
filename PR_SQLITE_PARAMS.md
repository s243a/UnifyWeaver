# feat(sources): Safe SQLite Parameter Binding

## Summary
This PR enhances the `sqlite` source plugin to support parameterized queries, preventing SQL injection vulnerabilities when handling user input. It introduces a dual-mode compilation strategy:
1.  **CLI Mode**: Uses `sqlite3` binary for simple, static queries (fastest).
2.  **Python Mode**: Automatically switches to an embedded Python script (`sqlite3` module) when parameters are present, ensuring proper variable binding.

## Changes
- **`sqlite_source.pl`**:
    - Added `parameters(List)` option support.
    - Implemented `generate_sqlite_python_code` to create a Python wrapper that safely binds parameters.
    - Fixed template syntax errors.
- **Tests**:
    - Updated `tests/core/test_sqlite_source.pl` to verify parameter binding (e.g., `WHERE age > ?`).
- **Documentation**:
    - Updated `docs/EXTENDED_README.md` with parameter usage example.

## Usage
```prolog
:- source(sqlite, user_by_id, [
    sqlite_file('data.db'),
    query('SELECT name FROM users WHERE id = ?'),
    parameters(['$1'])  % Maps script argument $1 to ?
]).
```