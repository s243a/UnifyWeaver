#!/bin/bash
# test_sql_setops.sh - SQLite integration test for INTERSECT/EXCEPT

set -e

echo "=== SQL Set Operations Test Suite ==="
echo

# Create test database
DB="output_sql_test/setops_test.db"
rm -f "$DB"

echo "Creating schema and test data..."

# Create tables
sqlite3 "$DB" <<EOF
CREATE TABLE person (name TEXT, age INTEGER, city TEXT);
CREATE TABLE special_members (name TEXT, member_type TEXT);
CREATE TABLE employees (name TEXT, dept TEXT);

-- Test data
-- Adults: Alice(25), Charlie(30), Eve(22), Frank(40), Grace(28)
-- Minors: Bob(17), Diana(16)
INSERT INTO person VALUES ('Alice', 25, 'NYC');
INSERT INTO person VALUES ('Bob', 17, 'LA');
INSERT INTO person VALUES ('Charlie', 30, 'Chicago');
INSERT INTO person VALUES ('Diana', 16, 'Boston');
INSERT INTO person VALUES ('Eve', 22, 'Seattle');
INSERT INTO person VALUES ('Frank', 40, 'Austin');
INSERT INTO person VALUES ('Grace', 28, 'Portland');

-- Members: Alice, Charlie, Henry
INSERT INTO special_members VALUES ('Alice', 'gold');
INSERT INTO special_members VALUES ('Charlie', 'silver');
INSERT INTO special_members VALUES ('Henry', 'bronze');

-- Employees: Alice, Eve, Isaac
INSERT INTO employees VALUES ('Alice', 'Engineering');
INSERT INTO employees VALUES ('Eve', 'Marketing');
INSERT INTO employees VALUES ('Isaac', 'Sales');
EOF

echo "✓ Schema and data created"
echo

# Generate SQL views
echo "Generating set operation views from Prolog..."
swipl test_sql_setops.pl 2>&1 | grep -v "^SQL table"
echo

# Load generated views
echo "Loading set operation views into SQLite..."
sqlite3 "$DB" < output_sql_test/intersect_test.sql
sqlite3 "$DB" < output_sql_test/except_test.sql
sqlite3 "$DB" < output_sql_test/intersect_three.sql
echo "✓ Views loaded"
echo

# Test 1: INTERSECT - adult_members (adults who are also members)
echo "Test 1: INTERSECT - adult_members"
echo "Expected: Alice, Charlie (2 people who are both adults and members)"
echo "Result:"
sqlite3 "$DB" "SELECT name FROM adult_members ORDER BY name;"
COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM adult_members;")
echo "Count: $COUNT"
if [ "$COUNT" -eq 2 ]; then
    echo "✓ Test 1 PASSED"
else
    echo "✗ Test 1 FAILED: Expected 2, got $COUNT"
    exit 1
fi
echo

# Test 2: EXCEPT - adults_only (adults who are NOT members)
echo "Test 2: EXCEPT - adults_only"
echo "Expected: Eve, Frank, Grace (3 adults who are not members)"
echo "Result:"
sqlite3 "$DB" "SELECT name FROM adults_only ORDER BY name;"
COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM adults_only;")
echo "Count: $COUNT"
if [ "$COUNT" -eq 3 ]; then
    echo "✓ Test 2 PASSED"
else
    echo "✗ Test 2 FAILED: Expected 3, got $COUNT"
    exit 1
fi
echo

# Test 3: Three-way INTERSECT - common_all (in all three groups)
echo "Test 3: Three-way INTERSECT - common_all"
echo "Expected: Alice (only person who is adult, member, AND employee)"
echo "Result:"
sqlite3 "$DB" "SELECT name FROM common_all ORDER BY name;"
COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM common_all;")
echo "Count: $COUNT"
if [ "$COUNT" -eq 1 ]; then
    echo "✓ Test 3 PASSED"
else
    echo "✗ Test 3 FAILED: Expected 1, got $COUNT"
    exit 1
fi
echo

# Test 4: Verify correctness
echo "Test 4: Verify correctness of set operations"
ALICE_IN_INTERSECT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM adult_members WHERE name = 'Alice';")
HENRY_NOT_IN_INTERSECT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM adult_members WHERE name = 'Henry';")
if [ "$ALICE_IN_INTERSECT" -eq 1 ] && [ "$HENRY_NOT_IN_INTERSECT" -eq 0 ]; then
    echo "✓ Test 4 PASSED: INTERSECT correctly filters"
else
    echo "✗ Test 4 FAILED: INTERSECT filtering incorrect"
    exit 1
fi
echo

echo "=== All Set Operations tests PASSED ==="
echo
echo "Database: $DB"
echo "Views: adult_members, adults_only, common_all"
