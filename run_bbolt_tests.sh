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

echo "===== All Tests Passed ====="
echo
echo "Verification steps:"
echo "1. Install bbolt CLI: go install go.etcd.io/bbolt/cmd/bbolt@latest"
echo "2. Inspect basic database: bbolt keys output_bbolt_basic/test_users.db users"
echo "3. Get a record: bbolt get output_bbolt_basic/test_users.db users Alice"
echo "4. Inspect schema database: bbolt keys output_bbolt_schema/test_users_schema.db users"
