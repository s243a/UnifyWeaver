# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# This file is part of UnifyWeaver.
# Licensed under either MIT or Apache-2.0 at your option.

#!/bin/bash

# Test runner for the family_tree example

# Source the necessary scripts
source /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/education/parent.sh
source /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/education/family_tree.sh

echo "--- Running family_tree tests ---"

# Test 1: Check for a known ancestor
echo -n "Test 1: Is abraham an ancestor of jacob? ... "
if [[ $(ancestor abraham jacob) == "abraham:jacob" ]]; then
    echo "PASS"
else
    echo "FAIL"
fi

# Test 2: Check for a known non-ancestor
echo -n "Test 2: Is sarah an ancestor of esau? ... "
# This is tricky, because parent(sarah,isaac) and parent(isaac,esau) makes sarah an ancestor.
# Let's pick a better test case. Let's check if ishmael is an ancestor of jacob.
if [[ -z $(ancestor ishmael jacob) ]]; then
    echo "PASS"
else
    echo "FAIL"
fi

# Test 3: List all descendants of isaac
echo "Test 3: Listing all descendants of isaac..."
descendants=$(ancestor isaac | sort)
expected_descendants="isaac:esau\nisaac:jacob\nisaac:reuben\nisaac:simeon\nisaac:levi\nisaac:judah"

# Sort the expected descendants as well
expected_descendants=$(echo -e "$expected_descendants" | sort)

if [[ "$descendants" == "$expected_descendants" ]]; then
    echo "PASS"
else
    echo "FAIL"
    echo "  Expected:"
    echo -e "$expected_descendants"
    echo "  Got:"
    echo "$descendants"
fi


echo "--- Tests finished ---"
