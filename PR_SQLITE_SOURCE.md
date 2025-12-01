# feat(sources): Native SQLite Source Plugin

## Summary
This PR adds a new `source(sqlite, ...)` plugin that enables UnifyWeaver to query SQLite databases directly using the `sqlite3` command-line tool. This provides a performant, language-agnostic way to ingest data from SQLite into the UnifyWeaver pipeline without requiring a Python wrapper.

## Changes
- **`sqlite_source.pl`**: Implements the source compiler. Generates Bash code that invokes `sqlite3`.
- **`sources.pl`**: Registers the `sqlite` source type and validates options.
- **Tests**: Added `tests/core/test_sqlite_source.pl`.
- **Documentation**: Updated `README.md` and `docs/EXTENDED_README.md`.

## Usage
```prolog
:- source(sqlite, users, [
    sqlite_file('data.db'),
    query('SELECT name, age FROM users WHERE active = 1'),
    output_format(tsv)
]).
```

## Dependencies
- `sqlite3` (CLI tool) must be installed.