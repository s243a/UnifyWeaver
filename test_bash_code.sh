#!/bin/bash

# Test the Bash example from Chapter 3: Facts as Associative Arrays

declare -A parent_data=(
    ["alice:bob"]=1
)

parent() {
    local key="$1:$2"
    # Check if the key exists in the array
    [[ -n "${parent_data[$key]}" ]] && echo "$key"
}

echo "Testing parent function:"
echo "parent alice bob should return: alice:bob"
parent alice bob
echo ""
echo "parent bob alice should return nothing (empty line below):"
parent bob alice
