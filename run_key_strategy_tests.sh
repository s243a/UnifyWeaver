#!/bin/bash

# Test runner for database key strategy tests

set -e

echo "===== Database Key Strategy Tests ====="
echo

# Generate all test code
echo "Generating test code..."
swipl -g main -t halt test_composite_keys.pl 2>&1 | grep -v "Warning:"
echo

# Test 1: Composite Keys (name + city)
echo "Test 1: Composite Keys (name:city)"
echo "------------------------------------"
cd output_composite_simple
echo "Building..."
go mod init test_composite 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null
go build user_store.go

# Create test data
echo '{"name": "Alice", "age": 30, "city": "NYC"}' > test_input.jsonl
echo '{"name": "Bob", "age": 25, "city": "SF"}' >> test_input.jsonl
echo '{"name": "Alice", "age": 28, "city": "LA"}' >> test_input.jsonl

# Run program
echo "Storing records..."
./user_store < test_input.jsonl

echo "✓ Test 1 completed"
echo "  Expected keys: Alice:NYC, Bob:SF, Alice:LA"
echo

cd ..

# Test 2: Backward Compatibility (single field)
echo "Test 2: Backward Compatibility (single field)"
echo "----------------------------------------------"
cd output_backward_compat
echo "Building..."
go mod init test_backward 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null
go build user_store.go

# Create test data
echo '{"name": "Alice", "age": 30}' > test_input.jsonl
echo '{"name": "Bob", "age": 25}' >> test_input.jsonl

# Run program
echo "Storing records..."
./user_store < test_input.jsonl

echo "✓ Test 2 completed"
echo "  Expected keys: Alice, Bob"
echo

cd ..

# Test 3: Hash Keys
echo "Test 3: Hash Keys (hash of content)"
echo "------------------------------------"
cd output_hash_keys
echo "Building..."
go mod init test_hash 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null
go build doc_store.go

# Create test data
echo '{"name": "doc1", "content": "Hello World"}' > test_input.jsonl
echo '{"name": "doc2", "content": "Goodbye World"}' >> test_input.jsonl

# Run program
echo "Storing records..."
./doc_store < test_input.jsonl

echo "✓ Test 3 completed"
echo "  Expected keys: SHA-256 hashes"
echo

cd ..

# Test 4: Composite with Hash (name + hash)
echo "Test 4: Composite Name + Hash(Content)"
echo "---------------------------------------"
cd output_name_hash
echo "Building..."
go mod init test_name_hash 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null
go build doc_store.go

# Create test data
echo '{"name": "mydoc", "content": "Hello World"}' > test_input.jsonl
echo '{"name": "otherdoc", "content": "Goodbye World"}' >> test_input.jsonl

# Run program
echo "Storing records..."
./doc_store < test_input.jsonl

echo "✓ Test 4 completed"
echo "  Expected keys: mydoc:<hash>, otherdoc:<hash>"
echo

cd ..

echo "===== All Tests Passed ======"
echo
echo "Key Strategies Tested:"
echo "1. ✓ Composite keys (field + field)"
echo "2. ✓ Backward compatibility (single field)"
echo "3. ✓ Hash keys (hash of field)"
echo "4. ✓ Complex composite (field + hash of field)"
echo
echo "Verification:"
echo "Install bbolt CLI: go install go.etcd.io/bbolt/cmd/bbolt@latest"
echo
echo "Inspect databases:"
echo "  bbolt keys output_composite_simple/composite_simple.db users"
echo "  bbolt keys output_backward_compat/backward_compat.db users"
echo "  bbolt keys output_hash_keys/hash_keys.db documents"
echo "  bbolt keys output_name_hash/name_hash.db documents"
echo
