#!/bin/bash
# test_mixed_joins_sqlite.sh - SQLite integration test for mixed INNER/LEFT JOINs

set -e

echo "=== Mixed INNER/LEFT JOIN SQLite Test ==="
echo

# Create test database
DB="test_mixed_joins.db"
rm -f "$DB"

# Create tables and data
sqlite3 "$DB" <<EOF
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    category TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    category_id INTEGER,
    product TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- Insert test data
INSERT INTO customers (id, name, region) VALUES
    (1, 'Alice', 'US'),
    (2, 'Bob', 'EU'),
    (3, 'Charlie', 'US');

INSERT INTO categories (id, customer_id, category) VALUES
    (10, 1, 'Electronics'),
    (11, 1, 'Books'),
    (12, 2, 'Electronics');
-- Note: Charlie has no categories

INSERT INTO orders (id, customer_id, category_id, product) VALUES
    (101, 1, 10, 'Laptop'),
    (102, 2, 12, 'Phone');
-- Note: Alice's Books category has no orders

EOF

echo "✓ Created test database with sample data"
echo

# Test 1: One INNER, One LEFT
echo "Test 1: One INNER (categories), One LEFT (orders)"
echo "Expected: All customer-category pairs, with product or NULL"
echo

sqlite3 "$DB" < output_sql_test/mixed_one_inner_one_left.sql

sqlite3 "$DB" "SELECT * FROM customer_category_orders ORDER BY name, category;"
echo

echo "Analysis:"
echo "- Alice/Electronics → Laptop (has order)"
echo "- Alice/Books → NULL (no order for this category)"
echo "- Bob/Electronics → Phone (has order)"
echo "- Charlie → Should NOT appear (no categories, INNER JOIN filters out)"
echo

# Cleanup
rm -f "$DB"

echo "✓ Mixed JOIN SQLite test completed successfully!"
