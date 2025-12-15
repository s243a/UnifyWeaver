# Python Source Examples

This file contains executable records for the python_source playbook.

## Example 1: Inline Python

Query: `python_inline_basic`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Python Source Demo: Inline Code ==="

mkdir -p tmp
cat > tmp/python_inline.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(fibonacci/1, python, [
    python_inline('
a, b = 0, 1
for _ in range(10):
    a, b = b, a + b
    print(a)
')
]).

main :-
    compile_dynamic_source(fibonacci/1, [], 'tmp/fibonacci.sh'),
    format("Generated: tmp/fibonacci.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/python_inline.pl
echo "Testing fibonacci/1:"
bash tmp/fibonacci.sh fibonacci
echo "Success: Python inline code works"
```

## Example 2: Python File

Query: `python_file_basic`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Python Source Demo: Python File ==="

mkdir -p tmp
cat > tmp/analyze.py << 'PYTHON'
import sys
text = "Hello world from Python"
words = len(text.split())
chars = len(text)
lines = 1
print(f"words:{words}")
print(f"chars:{chars}")
print(f"lines:{lines}")
PYTHON

cat > tmp/python_file.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(analyze_text/1, python, [
    python_file('tmp/analyze.py')
]).

main :-
    compile_dynamic_source(analyze_text/1, [], 'tmp/analyze_text.sh'),
    format("Generated: tmp/analyze_text.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/python_file.pl
echo "Testing analyze_text/1:"
bash tmp/analyze_text.sh analyze_text
echo "Success: Python file works"
```

## Example 3: Python with SQLite

Query: `python_sqlite`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Python Source Demo: SQLite Integration ==="

mkdir -p tmp
# Create test database
sqlite3 tmp/users.db << 'SQL'
CREATE TABLE users (name TEXT, role TEXT);
INSERT INTO users VALUES ('alice', 'admin');
INSERT INTO users VALUES ('bob', 'user');
INSERT INTO users VALUES ('charlie', 'guest');
SQL

cat > tmp/python_sqlite.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(query_users/1, python, [
    sqlite_query('SELECT name, role FROM users'),
    database('tmp/users.db')
]).

main :-
    compile_dynamic_source(query_users/1, [], 'tmp/query_users.sh'),
    format("Generated: tmp/query_users.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/python_sqlite.pl
echo "Testing query_users/1:"
bash tmp/query_users.sh query_users
echo "Success: Python SQLite integration works"
```
