#!/bin/bash
# End-to-end SQLite test for SQL target

echo "======================================"
echo "  SQL Target SQLite Integration Test"
echo "======================================"
echo ""

# Create test database
DB="output_sql_test/test.db"
rm -f "$DB"

echo "1. Creating test database..."
sqlite3 "$DB" <<EOF
CREATE TABLE person (
    name TEXT,
    age INTEGER,
    city TEXT
);

INSERT INTO person VALUES ('Alice', 30, 'NYC');
INSERT INTO person VALUES ('Bob', 17, 'LA');
INSERT INTO person VALUES ('Charlie', 25, 'NYC');
INSERT INTO person VALUES ('Diana', 45, 'NYC');
INSERT INTO person VALUES ('Eve', 19, 'LA');
EOF

echo "✓ Database created with 5 records"
echo ""

echo "2. Generating SQL from Prolog..."
swipl generate_sql_views.pl 2>&1 | grep -v "^SQL table"
echo "✓ SQL views generated"
echo ""

echo "3. Loading views into SQLite..."
sqlite3 "$DB" < output_sql_test/adult.sql
sqlite3 "$DB" < output_sql_test/nyc_adults.sql
echo "✓ Views loaded"
echo ""

echo "4. Testing adult view (age >= 18)..."
echo "Expected: Alice, Charlie, Diana, Eve"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM adult ORDER BY name;"
echo ""

echo "5. Testing nyc_adults view (NYC adults age >= 21)..."
echo "Expected: Alice, Charlie, Diana"
echo "Got:"
sqlite3 "$DB" "SELECT * FROM nyc_adults ORDER BY name;"
echo ""

echo "6. Verifying counts..."
ADULT_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM adult;")
NYC_ADULT_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM nyc_adults;")

if [ "$ADULT_COUNT" = "4" ] && [ "$NYC_ADULT_COUNT" = "3" ]; then
    echo "✓ All counts correct!"
    echo ""
    echo "======================================"
    echo "  ✓ SQLite integration test PASSED!"
    echo "======================================"
    exit 0
else
    echo "✗ Count mismatch!"
    echo "  adult count: $ADULT_COUNT (expected 4)"
    echo "  nyc_adults count: $NYC_ADULT_COUNT (expected 3)"
    echo ""
    echo "======================================"
    echo "  ✗ SQLite integration test FAILED!"
    echo "======================================"
    exit 1
fi
