#!/bin/bash
# End-to-end SQLite test for SQL JOINs

echo "========================================"
echo "  SQL JOINs Integration Test"
echo "========================================"
echo ""

# Create test database
DB="output_sql_test/joins.db"
rm -f "$DB"

echo "1. Creating test database with customers and orders..."
sqlite3 "$DB" <<EOF
CREATE TABLE customers (
    customer_id INTEGER,
    name TEXT,
    city TEXT
);

CREATE TABLE orders (
    order_id INTEGER,
    customer_id INTEGER,
    product TEXT,
    amount INTEGER
);

-- Customers
INSERT INTO customers VALUES (1, 'Alice', 'NYC');
INSERT INTO customers VALUES (2, 'Bob', 'LA');
INSERT INTO customers VALUES (3, 'Charlie', 'NYC');

-- Orders
INSERT INTO orders VALUES (101, 1, 'Widget', 100);
INSERT INTO orders VALUES (102, 1, 'Gadget', 150);
INSERT INTO orders VALUES (103, 2, 'Tool', 200);
INSERT INTO orders VALUES (104, 3, 'Widget', 75);
INSERT INTO orders VALUES (105, 3, 'Gizmo', 125);
EOF

echo "✓ Database created with 3 customers and 5 orders"
echo ""

echo "2. Generating SQL JOIN views from Prolog..."
swipl test_sql_joins.pl 2>&1 | grep -v "^SQL table"
echo ""

echo "3. Loading JOIN views into SQLite..."
sqlite3 "$DB" < output_sql_test/customer_orders.sql 2>&1
sqlite3 "$DB" < output_sql_test/nyc_customer_orders.sql 2>&1
sqlite3 "$DB" < output_sql_test/customer_order_summary.sql 2>&1
echo "✓ Views loaded"
echo ""

echo "========================================"
echo "Testing JOIN Views"
echo "========================================"
echo ""

# Test 1: Simple JOIN
echo "Test 1: Customer Orders (INNER JOIN)"
echo "Expected: 5 rows with customer names and their orders"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_orders ORDER BY customer_orders.product;" 2>&1
COUNT1=$(sqlite3 "$DB" "SELECT COUNT(*) FROM customer_orders;" 2>&1)
if [ "$COUNT1" = "5" ]; then
    echo "✓ Simple INNER JOIN works (5 orders)"
else
    echo "✗ Simple INNER JOIN failed (expected 5, got $COUNT1)"
    echo "Debug: Checking view definition..."
    sqlite3 "$DB" ".schema customer_orders"
    exit 1
fi
echo ""

# Test 2: JOIN with WHERE
echo "Test 2: NYC Customer Orders (JOIN + WHERE)"
echo "Expected: 4 orders (Alice: 2, Charlie: 2)"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM nyc_customer_orders ORDER BY product;" 2>&1
COUNT2=$(sqlite3 "$DB" "SELECT COUNT(*) FROM nyc_customer_orders;" 2>&1)
if [ "$COUNT2" = "4" ]; then
    echo "✓ JOIN with WHERE works (4 NYC orders)"
else
    echo "✗ JOIN with WHERE failed (expected 4, got $COUNT2)"
    exit 1
fi
echo ""

# Test 3: Multi-column JOIN result
echo "Test 3: Customer Order Summary"
echo "Expected: All orders with customer name, city, and amount"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_order_summary ORDER BY name, total_amount;" 2>&1 | head -5
COUNT3=$(sqlite3 "$DB" "SELECT COUNT(*) FROM customer_order_summary;" 2>&1)
if [ "$COUNT3" = "5" ]; then
    echo "✓ Multi-column JOIN works (5 records)"
else
    echo "✗ Multi-column JOIN failed (expected 5, got $COUNT3)"
    exit 1
fi
echo ""

echo "========================================"
echo "  ✓ All JOIN tests PASSED!"
echo "========================================"
exit 0
