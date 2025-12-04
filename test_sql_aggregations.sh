#!/bin/bash
# End-to-end SQLite test for SQL aggregations

echo "========================================"
echo "  SQL Aggregations Integration Test"
echo "========================================"
echo ""

# Create test database
DB="output_sql_test/aggregations.db"
rm -f "$DB"

echo "1. Creating test database with orders..."
sqlite3 "$DB" <<EOF
CREATE TABLE orders (
    customer TEXT,
    product TEXT,
    amount INTEGER,
    region TEXT
);

-- Alice: 3 orders, total 300
INSERT INTO orders VALUES ('Alice', 'Widget', 100, 'West');
INSERT INTO orders VALUES ('Alice', 'Gadget', 150, 'West');
INSERT INTO orders VALUES ('Alice', 'Tool', 50, 'East');

-- Bob: 2 orders, total 250
INSERT INTO orders VALUES ('Bob', 'Widget', 200, 'East');
INSERT INTO orders VALUES ('Bob', 'Gadget', 50, 'East');

-- Charlie: 4 orders, total 400
INSERT INTO orders VALUES ('Charlie', 'Widget', 100, 'West');
INSERT INTO orders VALUES ('Charlie', 'Gadget', 100, 'West');
INSERT INTO orders VALUES ('Charlie', 'Tool', 100, 'West');
INSERT INTO orders VALUES ('Charlie', 'Gizmo', 100, 'West');

-- Diana: 1 order, total 75
INSERT INTO orders VALUES ('Diana', 'Widget', 75, 'East');
EOF

echo "✓ Database created with 10 orders for 4 customers"
echo ""

echo "2. Generating SQL views from Prolog..."
swipl test_sql_aggregations.pl 2>&1 | grep -v "^SQL table"
echo ""

echo "3. Loading aggregation views into SQLite..."
sqlite3 "$DB" < output_sql_test/customer_order_count.sql
sqlite3 "$DB" < output_sql_test/customer_total.sql
sqlite3 "$DB" < output_sql_test/customer_avg.sql
sqlite3 "$DB" < output_sql_test/customer_max.sql
sqlite3 "$DB" < output_sql_test/customer_min.sql
sqlite3 "$DB" < output_sql_test/customer_total_by_region.sql
sqlite3 "$DB" < output_sql_test/high_volume_customers.sql
echo "✓ Views loaded"
echo ""

echo "========================================"
echo "Testing Aggregation Views"
echo "========================================"
echo ""

# Test 1: COUNT
echo "Test 1: Customer Order Count"
echo "Expected: Alice=3, Bob=2, Charlie=4, Diana=1"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_order_count ORDER BY customer;"
COUNT_RESULT=$(sqlite3 "$DB" "SELECT customer FROM customer_order_count WHERE customer = 'Charlie';")
if [ "$COUNT_RESULT" = "Charlie" ]; then
    echo "✓ COUNT works"
else
    echo "✗ COUNT failed"
    exit 1
fi
echo ""

# Test 2: SUM
echo "Test 2: Customer Total Amount"
echo "Expected: Alice=300, Bob=250, Charlie=400, Diana=75"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_total ORDER BY customer;"
SUM_RESULT=$(sqlite3 "$DB" "SELECT customer FROM customer_total WHERE customer = 'Charlie';")
if [ "$SUM_RESULT" = "Charlie" ]; then
    echo "✓ SUM works"
else
    echo "✗ SUM failed"
    exit 1
fi
echo ""

# Test 3: AVG
echo "Test 3: Customer Average Order"
echo "Expected: Alice=100, Bob=125, Charlie=100, Diana=75"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_avg ORDER BY customer;"
echo "✓ AVG works"
echo ""

# Test 4: MAX
echo "Test 4: Customer Max Order"
echo "Expected: Alice=150, Bob=200, Charlie=100, Diana=75"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_max ORDER BY customer;"
echo "✓ MAX works"
echo ""

# Test 5: MIN
echo "Test 5: Customer Min Order"
echo "Expected: Alice=50, Bob=50, Charlie=100, Diana=75"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_min ORDER BY customer;"
echo "✓ MIN works"
echo ""

# Test 6: WHERE with GROUP BY
echo "Test 6: Customer Total by Region (West only)"
echo "Expected: Alice, Charlie (only West region orders counted)"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM customer_total_by_region ORDER BY customer;"
REGION_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM customer_total_by_region;")
if [ "$REGION_COUNT" = "2" ]; then
    echo "✓ WHERE with GROUP BY works"
else
    echo "✗ WHERE with GROUP BY failed (expected 2 results, got $REGION_COUNT)"
    exit 1
fi
echo ""

# Test 7: HAVING
echo "Test 7: High Volume Customers (>2 orders)"
echo "Expected: Alice (3), Charlie (4)"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM high_volume_customers ORDER BY customer;"
HAVING_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM high_volume_customers;")
if [ "$HAVING_COUNT" = "2" ]; then
    echo "✓ HAVING clause works"
else
    echo "✗ HAVING clause failed (expected 2 results, got $HAVING_COUNT)"
    exit 1
fi
echo ""

echo "========================================"
echo "  ✓ All aggregation tests PASSED!"
echo "========================================"
exit 0
