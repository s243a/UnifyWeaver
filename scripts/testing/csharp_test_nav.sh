#!/bin/bash
# C# Test Navigation Helper
# Source this file to get easy navigation to C# test directories
#
# Usage:
#   source csharp_test_nav.sh
#   cd "${test[even]}"           # Go to test_even directory
#   cd "${test[link]}"           # Go to test_link directory
#   csharp_run even              # Run test_even
#   csharp_list                  # List all available tests

# Declare associative array for test directories
declare -gA test

# Save the project root directory (where this script is sourced from)
export CSHARP_TEST_ROOT="$PWD"

# Function to populate the test array
csharp_scan_tests() {
    # Clear existing entries
    for key in "${!test[@]}"; do
        unset test["$key"]
    done

    # Search for csharp_query_* directories
    local search_dirs=("tmp" "/tmp")
    local found=0

    for base_dir in "${search_dirs[@]}"; do
        if [[ ! -d "$base_dir" ]]; then
            continue
        fi

        # Find all csharp_query_test_* directories
        while IFS= read -r -d '' dir; do
            # Extract test name from directory
            # Format: csharp_query_test_NAME_UUID or csharp_query_test_NAME
            local basename=$(basename "$dir")

            # Match pattern: csharp_query_test_NAME_* or csharp_query_NAME_*
            if [[ "$basename" =~ ^csharp_query_test_([^_]+)_.* ]]; then
                local testname="${BASH_REMATCH[1]}"
                # Get most recent directory for this test name
                if [[ -z "${test[$testname]:-}" ]] || [[ "$dir" -nt "${test[$testname]:-}" ]]; then
                    test["$testname"]="$dir"
                    ((found++))
                fi
            elif [[ "$basename" =~ ^csharp_query_([^_]+)_.* ]]; then
                # Legacy format without "test_" prefix
                local testname="${BASH_REMATCH[1]}"
                if [[ -z "${test[$testname]:-}" ]] || [[ "$dir" -nt "${test[$testname]:-}" ]]; then
                    test["$testname"]="$dir"
                    ((found++))
                fi
            fi
        done < <(find "$base_dir" -maxdepth 1 -type d -name "csharp_query_*" -print0 2>/dev/null)
    done

    if [[ $found -eq 0 ]]; then
        echo "Warning: No C# test directories found. Run tests with --csharp-query-keep first." >&2
        return 1
    fi

    return 0
}

# Function to list available tests
csharp_list() {
    if [[ ${#test[@]} -eq 0 ]]; then
        csharp_scan_tests
    fi

    if [[ ${#test[@]} -eq 0 ]]; then
        echo "No C# test directories found."
        return 1
    fi

    echo "Available C# tests:"
    for testname in $(echo "${!test[@]}" | tr ' ' '\n' | sort); do
        local dir="${test[$testname]}"
        local reldir="${dir#$PWD/}"
        printf "  %-15s → %s\n" "$testname" "$reldir"
    done
}

# Function to run a specific test
csharp_run() {
    local testname="$1"

    if [[ -z "$testname" ]]; then
        echo "Usage: csharp_run <testname>" >&2
        echo "Available tests:" >&2
        csharp_list >&2
        return 1
    fi

    if [[ ${#test[@]} -eq 0 ]]; then
        csharp_scan_tests
    fi

    if [[ -z "${test[$testname]:-}" ]]; then
        echo "Error: Test '$testname' not found." >&2
        echo "Available tests:" >&2
        csharp_list >&2
        return 1
    fi

    local dir="${test[$testname]}"
    echo "Running test: $testname (from $dir)"
    (cd "$dir" && dotnet run)
}

# Function to build a specific test
csharp_build() {
    local testname="$1"

    if [[ -z "$testname" ]]; then
        echo "Usage: csharp_build <testname>" >&2
        echo "Available tests:" >&2
        csharp_list >&2
        return 1
    fi

    if [[ ${#test[@]} -eq 0 ]]; then
        csharp_scan_tests
    fi

    if [[ -z "${test[$testname]:-}" ]]; then
        echo "Error: Test '$testname' not found." >&2
        echo "Available tests:" >&2
        csharp_list >&2
        return 1
    fi

    local dir="${test[$testname]}"
    echo "Building test: $testname (in $dir)"
    (cd "$dir" && dotnet build)
}

# Function to show C# code for a specific test
csharp_cat() {
    local testname="$1"

    if [[ -z "$testname" ]]; then
        echo "Usage: csharp_cat <testname> [file]" >&2
        echo "Available tests:" >&2
        csharp_list >&2
        return 1
    fi

    if [[ ${#test[@]} -eq 0 ]]; then
        csharp_scan_tests
    fi

    if [[ -z "${test[$testname]:-}" ]]; then
        echo "Error: Test '$testname' not found." >&2
        return 1
    fi

    local dir="${test[$testname]}"
    local file="${2:-Program.cs}"

    if [[ -f "$dir/$file" ]]; then
        cat "$dir/$file"
    else
        echo "Error: File '$file' not found in $dir" >&2
        echo "Available files:" >&2
        ls -1 "$dir"/*.cs 2>/dev/null || echo "  (no .cs files found)"
        return 1
    fi
}

# Function to go back to project root
csharp_root() {
    if [[ -z "$CSHARP_TEST_ROOT" ]]; then
        echo "Error: CSHARP_TEST_ROOT not set. Please source csharp_test_nav.sh first." >&2
        return 1
    fi
    cd "$CSHARP_TEST_ROOT"
}

# Auto-scan on source (ignore return value to prevent shell exit when sourcing)
csharp_scan_tests || true

# Print usage hint
if [[ ${#test[@]} -gt 0 ]]; then
    echo "C# test navigation loaded. Available commands:"
    echo "  csharp_list              - List all tests"
    echo "  csharp_run <test>        - Run a test (e.g., csharp_run even)"
    echo "  csharp_build <test>      - Build a test"
    echo "  csharp_cat <test> [file] - Show C# code"
    echo "  csharp_root              - Return to project root"
    echo "  cd \"\${test[<name>]}\"     - Navigate to test directory"
    echo "  cd \"\$CSHARP_TEST_ROOT\"   - Navigate to project root"
    echo ""
    echo "Available tests: ${!test[*]}"
    echo "Project root: $CSHARP_TEST_ROOT"
else
    echo "C# test navigation loaded (no tests found yet)."
    echo "Run tests with: SKIP_CSHARP_EXECUTION=1 swipl ... -- --csharp-query-keep"
    echo "Project root: $CSHARP_TEST_ROOT"
fi
