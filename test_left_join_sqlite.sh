#!/bin/bash
# test_left_join_sqlite.sh - SQLite integration test for LEFT JOIN

set -e

echo "=== LEFT JOIN SQLite Integration Test ==="
echo

# Create test database
DB="test_left_join.db"
rm -f "$DB"

# Create tables
sqlite3 "$DB" <<EOF
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product TEXT NOT NULL,
    amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- Insert test data
INSERT INTO customers (id, name, region) VALUES
    (1, 'Alice', 'US'),
    (2, 'Bob', 'EU'),
    (3, 'Charlie', 'US'),
    (4, 'Diana', 'EU');

INSERT INTO orders (id, customer_id, product, amount) VALUES
    (101, 1, 'Widget', 29.99),
    (102, 2, 'Gadget', 49.99),
    (103, 1, 'Doohickey', 19.99);
-- Note: Charlie and Diana have no orders

EOF

echo "✓ Created test database with sample data"
echo

# Test 1: Basic LEFT JOIN
echo "Test 1: Basic LEFT JOIN"
echo "Expected: All customers with their products (NULL for customers without orders)"
echo

sqlite3 "$DB" < output_sql_test/left_join_basic.sql

sqlite3 "$DB" "SELECT * FROM customer_orders ORDER BY name;"
echo

# Test 2: Multi-column LEFT JOIN
echo "Test 2: Multi-column LEFT JOIN"
echo "Expected: All customers with product and amount (NULL for both if no orders)"
echo

sqlite3 "$DB" < output_sql_test/left_join_multi.sql

sqlite3 "$DB" "SELECT * FROM customer_order_details ORDER BY name;"
echo

# Test 4: LEFT JOIN with WHERE
echo "Test 4: LEFT JOIN with WHERE (EU customers only)"
echo "Expected: Only EU customers (Bob, Diana) with their orders"
echo

sqlite3 "$DB" < output_sql_test/left_join_where.sql

sqlite3 "$DB" "SELECT * FROM eu_customer_orders ORDER BY name;"
echo

# Cleanup
rm -f "$DB"

echo "✓ All LEFT JOIN SQLite tests completed successfully!"
