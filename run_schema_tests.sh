#!/bin/bash

set -e

echo "=== Testing JSON Schema Support ==="
echo

# Test 1: Basic schema (string, integer)
echo "Test 1: Basic Schema (string + integer)"
swipl -q -t "consult('test_schema_basic.pl')"

cat > test_schema_basic.jsonl << EOF
{"name": "Alice", "age": 25}
{"name": "Bob", "age": "thirty"}
{"name": "Charlie", "age": 35}
EOF

echo "Building..."
go build schema_basic.go

echo "Running with valid and invalid data..."
./schema_basic < test_schema_basic.jsonl
echo

# Test 2: All primitive types
echo "Test 2: All Primitive Types (string + float + integer + boolean)"
swipl -q -t "consult('test_schema_all_types.pl')"

cat > test_schema_all_types.jsonl << EOF
{"name": "Alice", "salary": 75000.50, "age": 30, "active": true}
{"name": "Bob", "salary": "high", "age": 25, "active": true}
{"name": "Charlie", "salary": 65000, "age": 35, "active": "yes"}
{"name": "Diana", "salary": 80000.0, "age": 28, "active": false}
EOF

echo "Building..."
go build schema_all_types.go

echo "Running with valid and invalid data..."
./schema_all_types < test_schema_all_types.jsonl
echo

# Test 3: Nested fields with schema
echo "Test 3: Nested Fields with Schema"
swipl -q -t "consult('test_schema_nested.pl')"

cat > test_schema_nested.jsonl << EOF
{"user": {"name": "Alice", "address": {"city": "NYC"}}}
{"user": {"name": 123, "address": {"city": "Boston"}}}
{"user": {"name": "Charlie", "address": {"city": 456}}}
EOF

echo "Building..."
go build schema_nested.go

echo "Running with valid and invalid data..."
./schema_nested < test_schema_nested.jsonl
echo

# Test 4: Mixed flat and nested with schema
echo "Test 4: Mixed Flat and Nested with Schema"
swipl -q -t "consult('test_schema_mixed.pl')"

cat > test_schema_mixed.jsonl << EOF
{"id": 1, "location": {"city": "NYC"}}
{"id": "two", "location": {"city": "Boston"}}
{"id": 3, "location": {"city": 789}}
EOF

echo "Building..."
go build schema_mixed.go

echo "Running with valid and invalid data..."
./schema_mixed < test_schema_mixed.jsonl
echo

echo "=== All schema tests completed ==="
