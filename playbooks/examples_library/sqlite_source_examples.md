---
file_type: UnifyWeaver Example Library
---
# SQLite Source Examples

This file contains executable records for the SQLite data source playbook.

## `unifyweaver.execution.sqlite_source_basic`

> [!example-record]
> id: unifyweaver.execution.sqlite_source_basic
> name: SQLite Source Basic Usage
> platform: bash

This record demonstrates querying SQLite databases with UnifyWeaver's sqlite_source plugin.

```bash
#!/bin/bash
# SQLite Source Demo - Basic Usage
# Demonstrates querying SQLite databases with UnifyWeaver

set -euo pipefail
cd /root/UnifyWeaver

echo "=== SQLite Source Demo: Basic Usage ==="

# Create test directory
mkdir -p tmp/sqlite_demo

# Create a test SQLite database
echo ""
echo "Creating test SQLite database..."
sqlite3 tmp/sqlite_demo/test.db <<'SQL'
DROP TABLE IF EXISTS users;
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    age INTEGER
);

INSERT INTO users (name, email, age) VALUES
    ('Alice', 'alice@example.com', 30),
    ('Bob', 'bob@example.com', 25),
    ('Charlie', 'charlie@example.com', 35),
    ('Diana', 'diana@example.com', 28);

DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    product TEXT,
    amount REAL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

INSERT INTO orders (user_id, product, amount) VALUES
    (1, 'Widget', 99.99),
    (2, 'Gadget', 149.99),
    (1, 'Gizmo', 49.99),
    (3, 'Widget', 99.99);
SQL

echo "Database created with users and orders tables"

# Create the Prolog script that uses sqlite_source
cat > tmp/sqlite_demo/generate_sqlite_source.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources/sqlite_source').

main :-
    format("~n=== SQLite Source Configuration ===~n"),

    % Example 1: Simple query - list all users
    format("~nGenerating bash for user listing...~n"),
    Config1 = [
        sqlite_file('tmp/sqlite_demo/test.db'),
        query('SELECT name, email, age FROM users'),
        output_format(tsv)
    ],

    (   sqlite_source:compile_source(list_users/1, Config1, [], BashCode1)
    ->  open('tmp/sqlite_demo/list_users.sh', write, S1),
        write(S1, BashCode1),
        close(S1),
        format("Generated: tmp/sqlite_demo/list_users.sh~n")
    ;   format("Failed to compile user listing source~n")
    ),

    % Example 2: Query with WHERE clause
    format("~nGenerating bash for filtered query...~n"),
    Config2 = [
        sqlite_file('tmp/sqlite_demo/test.db'),
        query('SELECT name, age FROM users WHERE age > 27'),
        output_format(tsv)
    ],

    (   sqlite_source:compile_source(users_over_27/1, Config2, [], BashCode2)
    ->  open('tmp/sqlite_demo/users_over_27.sh', write, S2),
        write(S2, BashCode2),
        close(S2),
        format("Generated: tmp/sqlite_demo/users_over_27.sh~n")
    ;   format("Failed to compile filtered query source~n")
    ),

    % Example 3: JOIN query
    format("~nGenerating bash for JOIN query...~n"),
    Config3 = [
        sqlite_file('tmp/sqlite_demo/test.db'),
        query('SELECT u.name, o.product, o.amount FROM users u JOIN orders o ON u.id = o.user_id'),
        output_format(tsv)
    ],

    (   sqlite_source:compile_source(user_orders/1, Config3, [], BashCode3)
    ->  open('tmp/sqlite_demo/user_orders.sh', write, S3),
        write(S3, BashCode3),
        close(S3),
        format("Generated: tmp/sqlite_demo/user_orders.sh~n")
    ;   format("Failed to compile JOIN query source~n")
    ),

    format("~n=== All sources generated ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate SQLite source scripts..."
swipl tmp/sqlite_demo/generate_sqlite_source.pl

echo ""
echo "=== Generated Scripts ==="

echo ""
echo "--- list_users.sh ---"
head -30 tmp/sqlite_demo/list_users.sh 2>/dev/null || echo "File not found"

echo ""
echo "=== Testing Generated Scripts ==="

chmod +x tmp/sqlite_demo/list_users.sh
chmod +x tmp/sqlite_demo/users_over_27.sh
chmod +x tmp/sqlite_demo/user_orders.sh

echo ""
echo "--- Output of list_users ---"
bash tmp/sqlite_demo/list_users.sh 2>/dev/null || echo "Script execution failed"

echo ""
echo "--- Output of users_over_27 ---"
bash tmp/sqlite_demo/users_over_27.sh 2>/dev/null || echo "Script execution failed"

echo ""
echo "--- Output of user_orders ---"
bash tmp/sqlite_demo/user_orders.sh 2>/dev/null || echo "Script execution failed"

echo ""
echo "Success: SQLite source demo complete"
```

## `unifyweaver.execution.sqlite_source_params`

> [!example-record]
> id: unifyweaver.execution.sqlite_source_params
> name: SQLite Source Parameterized Queries
> platform: bash

This record demonstrates safe parameter binding using Python wrapper for SQLite queries.

```bash
#!/bin/bash
# SQLite Source Demo - Parameterized Queries
# Demonstrates safe parameter binding using Python wrapper

set -euo pipefail
cd /root/UnifyWeaver

echo "=== SQLite Source Demo: Parameterized Queries ==="

mkdir -p tmp/sqlite_demo

# Ensure test database exists
if [ ! -f tmp/sqlite_demo/test.db ]; then
    echo "Creating test database..."
    sqlite3 tmp/sqlite_demo/test.db <<'SQL'
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        age INTEGER
    );
    INSERT OR REPLACE INTO users (id, name, email, age) VALUES
        (1, 'Alice', 'alice@example.com', 30),
        (2, 'Bob', 'bob@example.com', 25),
        (3, 'Charlie', 'charlie@example.com', 35);
SQL
fi

# Create Prolog script with parameterized queries
cat > tmp/sqlite_demo/demo_params.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources/sqlite_source').

main :-
    format("~n=== Generating Parameterized SQLite Source ===~n"),

    % Parameterized query - uses Python for safe binding
    Config = [
        sqlite_file('tmp/sqlite_demo/test.db'),
        query('SELECT name, email FROM users WHERE age > ?'),
        parameters(['$1']),  % $1 maps to first command line argument
        output_format(tsv)
    ],

    (   sqlite_source:compile_source(users_by_age/2, Config, [], BashCode)
    ->  open('tmp/sqlite_demo/users_by_age.sh', write, S),
        write(S, BashCode),
        close(S),
        format("Generated: tmp/sqlite_demo/users_by_age.sh~n"),
        format("~nThis script uses Python for safe parameter binding~n"),
        format("Usage: ./users_by_age.sh <min_age>~n")
    ;   format("Failed to compile parameterized source~n")
    ),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate parameterized SQLite source..."
swipl tmp/sqlite_demo/demo_params.pl

echo ""
echo "=== Parameterized Source Script ==="
head -40 tmp/sqlite_demo/users_by_age.sh

echo ""
echo "=== Testing Parameterized Query ==="
chmod +x tmp/sqlite_demo/users_by_age.sh

echo "Query: users with age > 26"
bash tmp/sqlite_demo/users_by_age.sh 26 2>/dev/null || echo "Script execution failed"

echo ""
echo "Query: users with age > 30"
bash tmp/sqlite_demo/users_by_age.sh 30 2>/dev/null || echo "Script execution failed"

echo ""
echo "Success: Parameterized SQLite source demo complete"
```

## `unifyweaver.execution.sqlite_source_info`

> [!example-record]
> id: unifyweaver.execution.sqlite_source_info
> name: SQLite Source Module Info
> platform: bash

This record displays SQLite source plugin capabilities and configuration options.

```bash
#!/bin/bash
# SQLite Source Demo - Module Information
# Shows SQLite source plugin capabilities

set -euo pipefail
cd /root/UnifyWeaver

echo "=== SQLite Source Plugin Information ==="

# Show source_info
swipl -g "
    use_module('src/unifyweaver/sources/sqlite_source'),
    sqlite_source:source_info(Info),
    format('~nSQLite Source Plugin Info:~n'),
    format('  ~w~n', [Info]),
    halt.
" 2>&1

echo ""
echo "=== Configuration Options ==="
echo ""
echo "Required:"
echo "  sqlite_file(Path)  - Path to SQLite database file"
echo "  query(SQL)         - SQL query to execute"
echo ""
echo "Optional:"
echo "  output_format(tsv|csv|list) - Output format (default: tsv)"
echo "  parameters([...])  - List of query parameters (enables Python mode)"
echo ""
echo "=== Engine Selection ==="
echo ""
echo "CLI Engine (default):"
echo "  - Uses sqlite3 command-line tool"
echo "  - Fast and lightweight"
echo "  - No parameter binding (use with static queries)"
echo ""
echo "Python Engine (when parameters specified):"
echo "  - Uses Python sqlite3 module"
echo "  - Safe parameter binding"
echo "  - Prevents SQL injection"

echo ""
echo "=== Example Configurations ==="
echo ""
echo "Simple query:"
echo '  [sqlite_file("data.db"),'
echo '   query("SELECT * FROM users")]'
echo ""
echo "With filter:"
echo '  [sqlite_file("data.db"),'
echo '   query("SELECT name, email FROM users WHERE active = 1"),'
echo '   output_format(csv)]'
echo ""
echo "Parameterized (Python):"
echo '  [sqlite_file("data.db"),'
echo '   query("SELECT * FROM users WHERE age > ?"),'
echo '   parameters(["$1"])]'

echo ""
echo "Success: SQLite source info displayed"
```

