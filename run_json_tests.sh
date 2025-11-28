#!/bin/bash

set -e

echo "=== Running JSON Input Comprehensive Tests ==="
echo

# Test 1: Two fields
echo "Test 1: Two fields (string and numeric)"
cat > test_input_1.jsonl << EOF
{"name": "Alice", "age": "25"}
{"name": "Bob", "age": 30}
{"name": "Charlie", "age": "35"}
EOF
go build test_two_fields.go
./test_two_fields < test_input_1.jsonl > test_output_1.txt
echo "Expected: Alice:25, Bob:30, Charlie:35"
echo "Got:      $(cat test_output_1.txt | tr '\n' ', ' | sed 's/, $//')"
echo

# Test 2: Three fields (mixed types)
echo "Test 2: Three fields (number, string, boolean)"
cat > test_input_2.jsonl << EOF
{"id": 1, "name": "Bob", "active": true}
{"id": 2, "name": "Alice", "active": false}
{"id": 3, "name": "Charlie", "active": true}
EOF
go build test_three_fields.go
./test_three_fields < test_input_2.jsonl > test_output_2.txt
echo "Expected: 1:Bob:true, 2:Alice:false, 3:Charlie:true"
echo "Got:      $(cat test_output_2.txt | tr '\n' ', ' | sed 's/, $//')"
echo

# Test 3: Single field
echo "Test 3: Single field"
cat > test_input_3.jsonl << EOF
{"product": "Laptop"}
{"product": "Mouse"}
{"product": "Keyboard"}
EOF
go build test_single_field.go
./test_single_field < test_input_3.jsonl > test_output_3.txt
echo "Expected: Laptop, Mouse, Keyboard"
echo "Got:      $(cat test_output_3.txt | tr '\n' ', ' | sed 's/, $//')"
echo

# Test 4: Four fields
echo "Test 4: Four fields"
cat > test_input_4.jsonl << EOF
{"id": 1, "name": "Alice", "department": "Engineering", "salary": 100000}
{"id": 2, "name": "Bob", "department": "Sales", "salary": 80000}
EOF
go build test_four_fields.go
./test_four_fields < test_input_4.jsonl > test_output_4.txt
echo "Expected: 1:Alice:Engineering:100000, 2:Bob:Sales:80000"
echo "Got:      $(cat test_output_4.txt | tr '\n' ', ' | sed 's/, $//')"
echo

# Test 5: Duplicates allowed
echo "Test 5: Duplicates allowed (unique=false)"
cat > test_input_5.jsonl << EOF
{"name": "Alice", "value": 10}
{"name": "Bob", "value": 20}
{"name": "Alice", "value": 10}
{"name": "Charlie", "value": 30}
EOF
go build test_duplicates.go
./test_duplicates < test_input_5.jsonl > test_output_5.txt
echo "Expected: 4 lines (duplicates allowed)"
echo "Got:      $(wc -l < test_output_5.txt) lines"
cat test_output_5.txt
echo

# Test 6: Tab delimiter
echo "Test 6: Tab delimiter"
cat > test_input_6.jsonl << EOF
{"field1": "A", "field2": "B"}
{"field1": "C", "field2": "D"}
EOF
go build test_tab.go
./test_tab < test_input_6.jsonl > test_output_6.txt
echo "Expected: Tab-separated values"
cat test_output_6.txt | cat -A  # Show tabs
echo

echo "=== All tests completed ==="
