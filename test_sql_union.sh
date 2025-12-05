#!/bin/bash
# test_sql_union.sh - SQLite integration test for UNION

set -e

echo "=== SQL UNION Test Suite ==="
echo

# Create test database
DB="output_sql_test/union_test.db"
rm -f "$DB"

echo "Creating schema and test data..."

# Create tables
sqlite3 "$DB" <<EOF
CREATE TABLE person (name TEXT, age INTEGER, city TEXT);
CREATE TABLE special_members (name TEXT, member_type TEXT);
CREATE TABLE vip_list (name TEXT, level INTEGER);

-- Test data
INSERT INTO person VALUES ('Alice', 25, 'NYC');
INSERT INTO person VALUES ('Bob', 17, 'LA');
INSERT INTO person VALUES ('Charlie', 30, 'Chicago');
INSERT INTO person VALUES ('Diana', 16, 'Boston');
INSERT INTO person VALUES ('Eve', 22, 'Seattle');

INSERT INTO special_members VALUES ('Frank', 'regular');
INSERT INTO special_members VALUES ('Grace', 'gold');
INSERT INTO special_members VALUES ('Alice', 'regular');  -- Duplicate name

INSERT INTO vip_list VALUES ('Henry', 5);
INSERT INTO vip_list VALUES ('Grace', 10);  -- Also in special_members
EOF

echo "✓ Schema and data created"
echo

# Generate SQL views
echo "Generating UNION views from Prolog..."
swipl test_sql_union.pl 2>&1 | grep -v "^SQL table"
echo

# Load generated views
echo "Loading UNION views into SQLite..."
sqlite3 "$DB" < output_sql_test/adult_union.sql
sqlite3 "$DB" < output_sql_test/vip_union.sql
echo "✓ Views loaded"
echo

# Test 1: adult view (UNION of two clauses)
echo "Test 1: adult view (two-clause UNION)"
echo "Expected: Alice, Charlie, Eve, Frank, Grace (5 distinct adults)"
echo "Result:"
sqlite3 "$DB" "SELECT name FROM adult ORDER BY name;"
COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM adult;")
echo "Count: $COUNT"
if [ "$COUNT" -eq 5 ]; then
    echo "✓ Test 1 PASSED"
else
    echo "✗ Test 1 FAILED: Expected 5, got $COUNT"
    exit 1
fi
echo

# Test 2: vip view (UNION of three clauses)
echo "Test 2: vip view (three-clause UNION)"
echo "Expected: Alice, Charlie, Eve, Grace, Henry (5 distinct VIPs)"
echo "Result:"
sqlite3 "$DB" "SELECT name FROM vip ORDER BY name;"
COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM vip;")
echo "Count: $COUNT"
if [ "$COUNT" -eq 5 ]; then
    echo "✓ Test 2 PASSED"
else
    echo "✗ Test 2 FAILED: Expected 5, got $COUNT"
    exit 1
fi
echo

# Test 3: Verify UNION removes duplicates
echo "Test 3: UNION deduplication"
echo "Checking that Alice and Grace appear only once despite being in multiple sources..."
ALICE_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM adult WHERE name = 'Alice';")
GRACE_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM vip WHERE name = 'Grace';")
if [ "$ALICE_COUNT" -eq 1 ] && [ "$GRACE_COUNT" -eq 1 ]; then
    echo "✓ Test 3 PASSED: UNION correctly removes duplicates"
else
    echo "✗ Test 3 FAILED: Expected 1 occurrence each, got Alice=$ALICE_COUNT, Grace=$GRACE_COUNT"
    exit 1
fi
echo

echo "=== All UNION tests PASSED ==="
echo
echo "Database: $DB"
echo "Views: adult, vip"
