#!/bin/bash

set -e

echo "=== Testing Nested JSON Field Access ==="
echo

# Test 1: Simple nested (2 levels)
echo "Test 1: Simple nested (2 levels)"
cat > test_nested_simple.jsonl << EOF
{"user": {"city": "NYC"}}
{"user": {"city": "Boston"}}
{"user": {"city": "SF"}}
EOF
go build nested_simple.go
./nested_simple < test_nested_simple.jsonl
echo

# Test 2: Deep nested (3 levels)
echo "Test 2: Deep nested (3 levels)"
cat > test_nested_deep.jsonl << EOF
{"user": {"address": {"city": "NYC", "zip": "10001"}}}
{"user": {"address": {"city": "Boston", "zip": "02101"}}}
EOF
go build nested_deep.go
./nested_deep < test_nested_deep.jsonl
echo

# Test 3: Multiple nested fields
echo "Test 3: Multiple nested fields"
cat > test_nested_multiple.jsonl << EOF
{"user": {"name": "Alice", "address": {"city": "NYC"}}}
{"user": {"name": "Bob", "address": {"city": "Boston"}}}
EOF
go build nested_multiple.go
./nested_multiple < test_nested_multiple.jsonl
echo

# Test 4: Mixed flat and nested
echo "Test 4: Mixed flat and nested"
cat > test_nested_mixed.jsonl << EOF
{"id": 1, "location": {"city": "NYC"}}
{"id": 2, "location": {"city": "Boston"}}
EOF
go build nested_mixed.go
./nested_mixed < test_nested_mixed.jsonl
echo

# Test 5: Very deep nesting (4 levels)
echo "Test 5: Very deep nesting (4 levels)"
cat > test_nested_very_deep.jsonl << EOF
{"company": {"department": {"team": {"lead": "Alice"}}}}
{"company": {"department": {"team": {"lead": "Bob"}}}}
EOF
go build nested_very_deep.go
./nested_very_deep < test_nested_very_deep.jsonl
echo

echo "=== All nested field tests passed ==="
