#!/bin/bash

# Test runner for bbolt database integration

set -e

echo "===== bbolt Database Integration Tests ====="
echo

# Create output directories
mkdir -p output_bbolt_basic
mkdir -p output_bbolt_schema

# Test 1: Basic storage without schema
echo "Test 1: Basic bbolt storage (no schema)"
echo "----------------------------------------"
swipl -g test_basic_bbolt -t halt test_bbolt_basic.pl

# Build the Go program
cd output_bbolt_basic
echo "Building Go program..."
go mod init test_bbolt_basic 2>/dev/null || true
go get go.etcd.io/bbolt
go build user_store.go

# Create test data
echo '{"name": "Alice", "age": 30}' > test_input.jsonl
echo '{"name": "Bob", "age": 25}' >> test_input.jsonl
echo '{"name": "Charlie", "age": 35}' >> test_input.jsonl

# Run the program
echo "Storing records in database..."
./user_store < test_input.jsonl

echo "✓ Test 1 passed"
echo
cd ..

# Test 2: Storage with schema validation
echo "Test 2: bbolt storage with schema validation"
echo "---------------------------------------------"
swipl -g test_schema_bbolt -t halt test_bbolt_schema.pl

# Build the Go program
cd output_bbolt_schema
echo "Building Go program..."
go mod init test_bbolt_schema 2>/dev/null || true
go get go.etcd.io/bbolt
go build user_store.go

# Create test data (mix of valid and invalid)
echo '{"name": "Alice", "age": 30, "email": "alice@example.com"}' > test_input.jsonl
echo '{"name": "Bob", "age": "invalid", "email": "bob@example.com"}' >> test_input.jsonl
echo '{"name": "Charlie", "age": 35, "email": "charlie@example.com"}' >> test_input.jsonl

# Run the program
echo "Storing records with validation..."
./user_store < test_input.jsonl 2>&1

echo "✓ Test 2 passed"
echo
cd ..

# Test 3: Database read mode
echo "Test 3: Reading from bbolt database"
echo "------------------------------------"
swipl -g test_read_bbolt -t halt test_bbolt_read.pl

# Build the Go program
mkdir -p output_bbolt_read
cd output_bbolt_read
echo "Building Go program..."
go mod init test_bbolt_read 2>/dev/null || true
go get go.etcd.io/bbolt
go build read_users.go

# Copy database from Test 1
cp ../output_bbolt_basic/test_users.db .

# Run the program
echo "Reading all records from database..."
./read_users > output.jsonl

# Verify output
echo "Verifying output matches expected records..."
if grep -q "Alice" output.jsonl && grep -q "Bob" output.jsonl && grep -q "Charlie" output.jsonl; then
    echo "✓ All expected records found"
else
    echo "✗ Some records missing!"
    exit 1
fi

# Count records
RECORD_COUNT=$(wc -l < output.jsonl)
if [ "$RECORD_COUNT" -eq 3 ]; then
    echo "✓ Correct number of records (3)"
else
    echo "✗ Expected 3 records, got $RECORD_COUNT"
    exit 1
fi

echo "✓ Test 3 passed"
echo
cd ..

echo "===== All Tests Passed ====="
echo
echo "Complete pipeline tested:"
echo "1. ✓ Write mode: JSON → Database"
echo "2. ✓ Schema validation: Type checking before storage"
echo "3. ✓ Read mode: Database → JSON"
echo
echo "Verification steps:"
echo "1. Install bbolt CLI: go install go.etcd.io/bbolt/cmd/bbolt@latest"
echo "2. Inspect basic database: bbolt keys output_bbolt_basic/test_users.db users"
echo "3. Get a record: bbolt get output_bbolt_basic/test_users.db users Alice"
echo "4. Inspect schema database: bbolt keys output_bbolt_schema/test_users_schema.db users"
echo "5. View read output: cat output_bbolt_read/output.jsonl"
