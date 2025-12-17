# Playbook: SQLite Data Source

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents orchestrate UnifyWeaver to generate bash functions that query SQLite databases.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "sqlite_source" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use sqlite source"


## Workflow Overview
Use UnifyWeaver's sqlite_source plugin to:
1. Define SQLite queries as Prolog predicates
2. Compile predicates to bash functions using sqlite3 CLI or Python
3. Handle parameterized queries with safe binding
4. Generate streaming-capable data source functions

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** - `playbooks/examples_library/sqlite_source_examples.md`
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

### Step 1: Navigate to project root
```bash
cd /root/UnifyWeaver
```

### Step 2: Extract the basic SQLite source demo
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.sqlite_source_basic" \
  playbooks/examples_library/sqlite_source_examples.md \
  > tmp/run_sqlite_basic.sh
```

### Step 3: Make it executable and run
```bash
chmod +x tmp/run_sqlite_basic.sh
bash tmp/run_sqlite_basic.sh
```

**Expected Output**:
```
=== SQLite Source Demo: Basic Usage ===

Creating test SQLite database...
Database created with users and orders tables

Running Prolog to generate SQLite source scripts...

=== SQLite Source Configuration ===

Generating bash for user listing...
  Compiling SQLite source: list_users/1
Generated: tmp/sqlite_demo/list_users.sh

Generating bash for filtered query...
  Compiling SQLite source: users_over_27/1
Generated: tmp/sqlite_demo/users_over_27.sh

Generating bash for JOIN query...
  Compiling SQLite source: user_orders/1
Generated: tmp/sqlite_demo/user_orders.sh

=== All sources generated ===

...

Success: SQLite source demo complete
```

### Step 4: Test parameterized queries (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.sqlite_source_params" \
  playbooks/examples_library/sqlite_source_examples.md \
  > tmp/run_sqlite_params.sh
chmod +x tmp/run_sqlite_params.sh
bash tmp/run_sqlite_params.sh
```

### Step 5: View module info (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.sqlite_source_info" \
  playbooks/examples_library/sqlite_source_examples.md \
  > tmp/run_sqlite_info.sh
chmod +x tmp/run_sqlite_info.sh
bash tmp/run_sqlite_info.sh
```

## What This Playbook Demonstrates

1. **sqlite_source plugin** (`src/unifyweaver/sources/sqlite_source.pl`):
   - `compile_source/4` - Compile SQLite source to bash
   - `validate_config/1` - Validate configuration options
   - `source_info/1` - Plugin metadata

2. **Engine selection**:
   - **CLI Engine** (default): Uses `sqlite3` command-line tool
     - Fast and lightweight
     - No parameter binding
     - Use with static queries
   - **Python Engine** (when parameters specified):
     - Uses Python `sqlite3` module
     - Safe parameter binding
     - Prevents SQL injection

3. **Configuration options**:
   - `sqlite_file(Path)` - Path to SQLite database (required)
   - `query(SQL)` - SQL query to execute (required)
   - `output_format(tsv|csv|list)` - Output format (default: tsv)
   - `parameters([...])` - Query parameters (enables Python mode)

4. **Generated bash functions**:
   - `predicate()` - Main function
   - `predicate_stream()` - Streaming alias

## Example Configurations

### Simple SELECT:
```prolog
Config = [
    sqlite_file('data.db'),
    query('SELECT * FROM users')
].
```

### Query with WHERE clause:
```prolog
Config = [
    sqlite_file('data.db'),
    query('SELECT name, email FROM users WHERE active = 1'),
    output_format(csv)
].
```

### JOIN query:
```prolog
Config = [
    sqlite_file('data.db'),
    query('SELECT u.name, o.product FROM users u JOIN orders o ON u.id = o.user_id')
].
```

### Parameterized query (Python):
```prolog
Config = [
    sqlite_file('data.db'),
    query('SELECT * FROM users WHERE age > ?'),
    parameters(['$1'])  % $1 maps to first CLI argument
].
```

## Common Mistakes to Avoid

- **DO NOT** run extracted scripts with `swipl` - they are bash scripts
- **DO** ensure sqlite3 is installed before testing
- **DO** use parameterized queries to prevent SQL injection
- **DO** verify database file exists before querying

## Expected Outcome
- Generated bash scripts that query SQLite databases
- Scripts handle TSV/CSV/list output formats
- Parameterized queries use safe Python binding

## Citations
[1] playbooks/examples_library/sqlite_source_examples.md
[2] src/unifyweaver/sources/sqlite_source.pl
[3] src/unifyweaver/core/dynamic_source_compiler.pl
[4] skills/skill_unifyweaver_environment.md

