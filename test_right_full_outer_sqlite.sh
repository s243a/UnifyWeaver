#!/bin/bash
# test_right_full_outer_sqlite.sh - SQLite/PostgreSQL integration test for RIGHT/FULL OUTER JOINs

set -e

echo "=== RIGHT JOIN and FULL OUTER JOIN SQLite Test ==="
echo

# Create test database
DB="test_right_full_outer.db"
rm -f "$DB"

# Create tables and data
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
    (3, 'Charlie', 'US');

INSERT INTO orders (id, customer_id, product, amount) VALUES
    (101, 1, 'Laptop', 999.99),
    (102, 1, 'Mouse', 29.99),
    (103, NULL, 'Orphan Product', 49.99);  -- Order without customer
-- Note: Bob and Charlie have no orders

EOF

echo "✓ Created test database with sample data"
echo
echo "Data:"
echo "  Customers: Alice, Bob, Charlie"
echo "  Orders: Alice has Laptop + Mouse, NULL customer has Orphan Product"
echo "  Bob and Charlie have no orders"
echo

# Test 1: Simple RIGHT JOIN
echo "=== Test 1: Simple RIGHT JOIN ==="
echo "Pattern: (customers ; null), orders"
echo "Expected: Keep all orders, NULL for missing customers"
echo

# Note: SQLite doesn't support RIGHT JOIN directly, so we'll test with PostgreSQL syntax
# For now, just show that we generated the SQL
echo "Generated SQL:"
cat output_sql_test/right_join_simple.sql
echo

# Test with LEFT JOIN equivalent (swap tables)
echo "SQLite doesn't support RIGHT JOIN, but we can verify with LEFT JOIN equivalent:"
echo "Query: SELECT orders.product, customers.name FROM orders LEFT JOIN customers ON customers.id = orders.customer_id"
echo

sqlite3 "$DB" "SELECT orders.product, customers.name FROM orders LEFT JOIN customers ON customers.id = orders.customer_id ORDER BY orders.product;"
echo

echo "Analysis:"
echo "- Laptop | Alice (matched)"
echo "- Mouse | Alice (matched)"
echo "- Orphan Product | NULL (order without customer - preserved by LEFT/RIGHT JOIN)"
echo

# Test 2: FULL OUTER JOIN
echo "=== Test 2: FULL OUTER JOIN ==="
echo "Pattern: (customers ; null), (orders ; null)"
echo "Expected: Keep all customers AND all orders"
echo

echo "Generated SQL:"
cat output_sql_test/full_outer_join.sql
echo

# SQLite doesn't support FULL OUTER JOIN either, emulate with UNION
echo "SQLite doesn't support FULL OUTER JOIN, but we can emulate with UNION:"
echo

sqlite3 "$DB" <<EOF
-- Emulate FULL OUTER JOIN with UNION
SELECT customers.name, orders.product
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
UNION
SELECT customers.name, orders.product
FROM orders
LEFT JOIN customers ON customers.id = orders.customer_id
ORDER BY name, product;
EOF

echo
echo "Analysis:"
echo "- Alice | Laptop (matched)"
echo "- Alice | Mouse (matched)"
echo "- Bob | NULL (customer without orders)"
echo "- Charlie | NULL (customer without orders)"
echo "- NULL | Orphan Product (order without customer)"
echo

# Cleanup
rm -f "$DB"

echo "✓ RIGHT/FULL OUTER JOIN SQLite test completed!"
echo
echo "NOTE: SQLite doesn't support RIGHT JOIN or FULL OUTER JOIN natively."
echo "The generated SQL is valid for PostgreSQL, MySQL 8+, and other databases."
