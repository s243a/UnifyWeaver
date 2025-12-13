# Playbook: Python Foreign Function Binding

## Audience
This playbook demonstrates how to use Python code as data sources through UnifyWeaver's foreign function binding system.

## Overview
The `python_source` plugin lets you declare Prolog predicates that execute Python code (inline or from files) and return results. Supports SQLite integration.

## When to Use This Playbook

✅ **Use python_source when:**
- You want to call Python code from Prolog
- You need Python's scientific/data libraries (pandas, numpy, etc.)
- You're integrating with Python-based systems
- You want embedded Python with SQLite support

## Agent Inputs

Reference the following artifacts:
1. **Executable Records** – `playbooks/examples_library/python_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/python_source.pl`

## Execution Guidance

### Example 1: Inline Python Code

```bash
cd /path/to/UnifyWeaver

perl scripts/extract_records.pl playbooks/examples_library/python_source_examples.md \
    python_inline_basic > tmp/python_inline.sh
chmod +x tmp/python_inline.sh
bash tmp/python_inline.sh
```

**Expected Output:**
```
Compiling Python source: fibonacci/1
Generated: tmp/fibonacci.sh
Testing fibonacci/1:
1
1
2
3
5
8
13
21
34
55
```

### Example 2: Python File

```bash
perl scripts/extract_records.pl playbooks/examples_library/python_source_examples.md \
    python_file_basic > tmp/python_file.sh
chmod +x tmp/python_file.sh
bash tmp/python_file.sh
```

**Expected Output:**
```
Compiling Python source: analyze_text/1
Generated: tmp/analyze_text.sh
Testing analyze_text/1:
words:5
chars:29
lines:1
```

### Example 3: Python with SQLite

```bash
perl scripts/extract_records.pl playbooks/examples_library/python_source_examples.md \
    python_sqlite > tmp/python_sqlite.sh
chmod +x tmp/python_sqlite.sh
bash tmp/python_sqlite.sh
```

**Expected Output:**
```
Compiling Python source: query_users/1
Generated: tmp/query_users.sh
Testing query_users/1:
alice:admin
bob:user
charlie:guest
```

## Configuration Options

- `python_inline(Code)` - Inline Python code to execute
- `python_file(File)` - Python script file to run
- `sqlite_query(Query)` - SQL query (requires `database(File)`)
- `database(File)` - SQLite database file
- `timeout(Seconds)` - Execution timeout (default: 30)
- `python_interpreter(Path)` - Python path (default: `python3`)

## See Also

- `playbooks/sqlite_source_playbook.md` - Direct SQLite access
- `playbooks/json_litedb_playbook.md` - .NET/Python interop
- `playbooks/awk_source_playbook.md` - AWK foreign function binding

## Summary

**Key Concepts:**
- ✅ Foreign function binding for Python
- ✅ Inline code or external files
- ✅ SQLite integration built-in
- ✅ Configurable timeout and interpreter
- ✅ Return structured data to Prolog
