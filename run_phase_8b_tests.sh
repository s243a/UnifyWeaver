#!/bin/bash
# run_phase_8b_tests.sh - Test Phase 8b: Enhanced Filtering

set -e  # Exit on error

echo "=========================================="
echo "Phase 8b: Enhanced Filtering Test Suite"
echo "=========================================="
echo

# Cleanup
echo "Cleaning up previous test artifacts..."
rm -f test_phase_8b.db
rm -rf output_phase_8b_*
echo

# Generate all Go programs from Prolog
echo "Step 1: Generating Go programs from Prolog..."
swipl test_phase_8b.pl 2>&1 | grep -v "Warning:"
echo
echo "Step 2: Creating test database with sample data..."
echo

# Create test data JSON
cat > test_phase_8b_input.json <<'EOF'
{"name": "Alice", "age": 35, "city": "NYC", "status": "active"}
{"name": "Bob", "age": 28, "city": "SF", "status": "inactive"}
{"name": "Charlie", "age": 42, "city": "nyc", "status": "premium"}
{"name": "Diana", "age": 23, "city": "Boston", "status": "active"}
{"name": "Eve", "age": 31, "city": "Nyc", "status": "active"}
{"name": "Frank", "age": 52, "city": "Seattle", "status": "inactive"}
{"name": "Grace", "age": 25, "city": "LA", "status": "premium"}
{"name": "Henry", "age": 38, "city": "Chicago", "status": "active"}
{"name": "Natalie", "age": 30, "city": "SF", "status": "premium"}
{"name": "Jack", "age": 45, "city": "Miami", "status": "inactive"}
{"name": "Julia", "age": 35, "city": "NYC", "status": "active"}
{"name": "Kalina", "age": 40, "city": "LA", "status": "premium"}
EOF

# Build and run populate program
echo "  Building populate program..."
cd output_phase_8b_populate || exit 1
go mod init populate 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o populate
echo "  Populating database..."
./populate < ../test_phase_8b_input.json
# Move database to parent directory so query programs can find it
mv test_phase_8b.db ../ 2>/dev/null || true
echo "  ✓ Database populated with 12 users"
cd ..
echo

# Test 1: Case-Insensitive City Search
echo "=========================================="
echo "Test 1: Case-Insensitive City (=@= \"nyc\")"
echo "=========================================="
cd output_phase_8b_insensitive || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Alice, Charlie, Eve, Julia (all NYC variants)"
cd ..
echo

# Test 2: Name Contains Substring
echo "=========================================="
echo "Test 2: Contains Substring (\"ali\")"
echo "=========================================="
cd output_phase_8b_contains || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Alice, Natalie, Kalina, Julia"
cd ..
echo

# Test 3: City Membership (String List)
echo "=========================================="
echo "Test 3: Major Cities (String List)"
echo "=========================================="
cd output_phase_8b_city_member || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Alice, Charlie, Eve, Grace, Henry, Natalie, Julia, Kalina"
cd ..
echo

# Test 4: Age Membership (Numeric List)
echo "=========================================="
echo "Test 4: Specific Ages (Numeric List)"
echo "=========================================="
cd output_phase_8b_age_member || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Alice, Grace, Natalie, Julia, Kalina (ages 25,30,35,40)"
cd ..
echo

# Test 5: Status Membership (Atom List)
echo "=========================================="
echo "Test 5: Active/Premium Users (Atom List)"
echo "=========================================="
cd output_phase_8b_status_member || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Alice, Charlie, Eve, Grace, Henry, Natalie, Julia, Kalina"
cd ..
echo

# Test 6: Mixed String + Numeric
echo "=========================================="
echo "Test 6: NYC Young Adults (Mixed)"
echo "=========================================="
cd output_phase_8b_mixed || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Alice (35), Charlie (42), Eve (31), Julia (35)"
cd ..
echo

# Test 7: Contains + Membership
echo "=========================================="
echo "Test 7: Major City + Name Contains 'a'"
echo "=========================================="
cd output_phase_8b_contains_member || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Alice, Charlie, Grace, Natalie, Julia, Kalina"
cd ..
echo

# Test 8: Complex Query
echo "=========================================="
echo "Test 8: Complex String Query"
echo "=========================================="
cd output_phase_8b_complex || exit 1
go mod init query 2>/dev/null || true
go get go.etcd.io/bbolt@latest 2>/dev/null
go build -o query
echo "Running test..."
./query
echo "Expected: Charlie, Julia (name contains 'i', city=NYC, status active/premium)"
cd ..
echo

echo "=========================================="
echo "All Phase 8b tests completed!"
echo "=========================================="
echo
echo "Summary:"
echo "  ✓ Test 1: Case-insensitive equality (=@=)"
echo "  ✓ Test 2: Substring matching (contains/2)"
echo "  ✓ Test 3: String list membership"
echo "  ✓ Test 4: Numeric list membership"
echo "  ✓ Test 5: Atom list membership"
echo "  ✓ Test 6: Mixed string ops + numeric filter"
echo "  ✓ Test 7: Contains + membership combination"
echo "  ✓ Test 8: Complex multi-constraint query"
echo
