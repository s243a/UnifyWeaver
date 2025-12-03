#!/bin/bash

set -e

echo "===== Database Filter Tests (Phase 8a) ====="
echo ""

# Step 1: Generate all test programs
echo "Step 1: Generating Go programs from Prolog..."
swipl -q -t "consult('test_db_filters.pl'), halt" 2>&1 | grep -v "Warning:"
echo ""

# Step 2: Create test data
echo "Step 2: Creating test data..."
cat > test_data_filters.json <<EOF
{"name": "Alice", "age": 35, "city": "NYC", "salary": 75000}
{"name": "Bob", "age": 28, "city": "SF", "salary": 90000}
{"name": "Charlie", "age": 42, "city": "NYC", "salary": 65000}
{"name": "Diana", "age": 25, "city": "LA", "salary": 45000}
{"name": "Eve", "age": 31, "city": "NYC", "salary": 55000}
{"name": "Frank", "age": 52, "city": "Chicago", "salary": 82000}
{"name": "Grace", "age": 29, "city": "SF", "salary": 72000}
{"name": "Henry", "age": 38, "city": "Boston", "salary": 68000}
{"name": "Iris", "age": 22, "city": "NYC", "salary": 28000}
{"name": "Jack", "age": 45, "city": "Seattle", "salary": 95000}
EOF
echo "Created test_data_filters.json with 10 users"
echo ""

# Step 3: Build and run populate program
echo "Step 3: Populating database..."
cd output_filters_populate
go mod init populate 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null || true
go build -o populate populate.go

# Remove old database
rm -f filters_test.db

# Populate database
cat ../test_data_filters.json | ./populate 2>&1
echo ""
cd ..

# Step 4: Run each filter test
echo "===== Running Filter Tests ====="
echo ""

# Test 1: Age filter (Age >= 30)
echo "--- Test 1: Age Filter (Age >= 30) ---"
cd output_filters_age
go mod init read_adults 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null || true
go build -o read_adults read_adults.go
cp ../output_filters_populate/filters_test.db .
echo "Output:"
./read_adults 2>&1 | head -20
echo ""
cd ..

# Test 2: Multi-field filter (Age > 25 AND City = "NYC")
echo "--- Test 2: Multi-Field Filter (Age > 25 AND City = NYC) ---"
cd output_filters_multi
go mod init read_nyc_young 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null || true
go build -o read_nyc_young read_nyc_young.go
cp ../output_filters_populate/filters_test.db .
echo "Output:"
./read_nyc_young 2>&1 | head -20
echo ""
cd ..

# Test 3: Salary range (30000 =< Salary =< 80000)
echo "--- Test 3: Salary Range (30000 =< Salary =< 80000) ---"
cd output_filters_salary
go mod init read_middle_income 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null || true
go build -o read_middle_income read_middle_income.go
cp ../output_filters_populate/filters_test.db .
echo "Output:"
./read_middle_income 2>&1 | head -20
echo ""
cd ..

# Test 4: Not equal filter (City \= "NYC")
echo "--- Test 4: Not Equal Filter (City != NYC) ---"
cd output_filters_not_equal
go mod init read_non_nyc 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null || true
go build -o read_non_nyc read_non_nyc.go
cp ../output_filters_populate/filters_test.db .
echo "Output:"
./read_non_nyc 2>&1 | head -20
echo ""
cd ..

# Test 5: All comparison operators
echo "--- Test 5: All Comparison Operators ---"
cd output_filters_all_ops
go mod init read_all_ops 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null || true
go build -o read_all_ops read_all_ops.go
cp ../output_filters_populate/filters_test.db .
echo "Output:"
./read_all_ops 2>&1 | head -20
echo ""
cd ..

# Test 6: No filter - read all (baseline)
echo "--- Test 6: No Filter (Read All) ---"
cd output_filters_all
go mod init read_all 2>/dev/null || true
go get go.etcd.io/bbolt 2>/dev/null || true
go build -o read_all read_all.go
cp ../output_filters_populate/filters_test.db .
echo "Output:"
./read_all 2>&1 | head -20
echo ""
cd ..

echo "===== All Filter Tests Complete ====="
echo ""
echo "Expected Results:"
echo "  Test 1: 7 users (Alice, Charlie, Eve, Frank, Grace, Henry, Jack)"
echo "  Test 2: 3 users (Alice, Charlie, Eve) - NYC users over 25"
echo "  Test 3: 6 users with salary 30000-80000"
echo "  Test 4: 5 users not from NYC"
echo "  Test 5: Users aged 20-60 with salary 25000-100000"
echo "  Test 6: All 10 users"
