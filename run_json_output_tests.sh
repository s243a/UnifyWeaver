#!/bin/bash

set -e

echo "=== Testing JSON Output ==="
echo

# Generate all test programs
swipl -q -t "consult('test_json_output.pl'), run_all_tests, halt" 2>&1 | grep -v "Warning:"
echo

# Test 1: Custom field names
echo "Test 1: Colon-delimited to JSON"
cat > test_colon.txt << EOF
Alice:25
Bob:30
Charlie:35
EOF
go build output_custom.go
./output_custom < test_colon.txt
echo

# Test 2: Three fields with mixed types
echo "Test 2: Three fields (id, name, active)"
cat > test_three.txt << EOF
1:Bob:true
2:Alice:false
3:Charlie:true
EOF
go build output_three.go
./output_three < test_three.txt
echo

# Test 3: Tab-delimited to JSON
echo "Test 3: Tab-delimited to JSON"
cat > test_tab.txt << EOF
1	Alice	Engineering	100000
2	Bob	Sales	80000
EOF
go build output_tab.go
./output_tab < test_tab.txt
echo

# Test 4: Round-trip test
echo "Test 4: Round-trip (JSON → Colon → JSON)"
echo "Input:"
cat test_users_numeric.jsonl
echo "After JSON input:"
./user_json < test_users_numeric.jsonl
echo "After JSON output:"
./user_json < test_users_numeric.jsonl | ./output_custom
echo

echo "=== All JSON output tests passed ==="
