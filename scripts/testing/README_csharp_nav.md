# C# Test Navigation Helper

A bash helper script for easily navigating and working with generated C# test code.

## Quick Start

```bash
# Source the navigation script
source csharp_test_nav.sh

# List available tests
csharp_list

# Run a specific test
csharp_run even

# Navigate to a test directory
cd "${test[even]}"

# Return to project root
csharp_root
```

## Installation

### Automatic Installation

The script is automatically copied when you create a new test environment using `init_testing.sh`.

When you run:
```bash
cd scripts/testing/test_env10/scripts/testing
./init_testing.sh
```

The script will be copied to the root of your new test environment along with this README.

### Manual Installation

If you're in an existing test environment or want to use it elsewhere:

```bash
cd scripts/testing/test_env10
source csharp_test_nav.sh
```

To make it permanent, add to your `~/.bashrc`:
```bash
# If you frequently work in this test environment
alias csharp-nav='source /path/to/UnifyWeaver/scripts/testing/test_env10/csharp_test_nav.sh'
```

## Available Commands

### `csharp_list`
Lists all available C# test directories with their short names.

```bash
$ csharp_list
Available C# tests:
  even            → tmp/csharp_query_test_even_7d4f6198-bdf7-11f0-aeb5-00155db51a74
  factorial       → tmp/csharp_query_test_factorial_7d47787a-bdf7-11f0-8ffe-00155db51a74
  filtered        → tmp/csharp_query_test_filtered_7d412cc2-bdf7-11f0-bf91-00155db51a74
  increment       → tmp/csharp_query_test_increment_7d43e34a-bdf7-11f0-ac36-00155db51a74
  link            → tmp/csharp_query_test_link_7d3e6c4e-bdf7-11f0-a806-00155db51a74
  positive        → tmp/csharp_query_test_positive_7d4a0c84-bdf7-11f0-bcf1-00155db51a74
  reachable       → tmp/csharp_query_test_reachable_7d4c8a90-bdf7-11f0-85a6-00155db51a74
```

### `csharp_run <testname>`
Runs a specific test using `dotnet run`.

```bash
$ csharp_run even
Running test: even (from tmp/csharp_query_test_even_7d4f6198-bdf7-11f0-aeb5-00155db51a74)
0
2
4
```

### `csharp_build <testname>`
Builds a specific test using `dotnet build`.

```bash
$ csharp_build link
Building test: link (in tmp/csharp_query_test_link_7d3e6c4e-bdf7-11f0-a806-00155db51a74)
```

### `csharp_cat <testname> [file]`
Shows the C# code for a test. Defaults to `Program.cs`.

```bash
$ csharp_cat even
# Shows Program.cs for test_even

$ csharp_cat even QueryRuntime.cs
# Shows QueryRuntime.cs for test_even
```

### `csharp_root`
Returns to the project root directory (where the script was sourced).

```bash
$ cd "${test[even]}"
$ pwd
/mnt/.../tmp/csharp_query_test_even_...

$ csharp_root
$ pwd
/mnt/.../UnifyWeaver/scripts/testing/test_env10
```

## Using the `test` Array

The script creates an associative array called `test` that maps test names to directories:

```bash
# Navigate to a specific test
cd "${test[even]}"
cd "${test[link]}"
cd "${test[increment]}"

# Use in scripts
for testname in "${!test[@]}"; do
    echo "Test: $testname at ${test[$testname]}"
done

# Check if a test exists
if [[ -n "${test[factorial]}" ]]; then
    echo "Factorial test found at: ${test[factorial]}"
fi
```

## The `CSHARP_TEST_ROOT` Variable

The script exports `CSHARP_TEST_ROOT` which contains the directory where the script was sourced:

```bash
# Always return to project root
cd "$CSHARP_TEST_ROOT"

# Use in paths
ls "$CSHARP_TEST_ROOT/tmp"
cat "$CSHARP_TEST_ROOT/init.pl"
```

## Available Tests

The following test names are typically available after running the C# tests:

| Test Name | Description | Expected Output |
|-----------|-------------|-----------------|
| `link` | Join/grandparent query | `alice,charlie` |
| `filtered` | Selection test | `alice` |
| `increment` | Arithmetic test | `item1,6` / `item2,3` |
| `factorial` | Recursive arithmetic | Factorial results |
| `positive` | Comparison test | Positive numbers |
| `reachable` | Linear recursion | Reachability results |
| `even` | Mutual recursion | `0`, `2`, `4` |

## Workflow Example

```bash
# 1. Source the navigation script
source csharp_test_nav.sh

# 2. Run the tests (if not already run)
SKIP_CSHARP_EXECUTION=1 swipl -q -f init.pl -s tests/core/test_csharp_query_target.pl \
    -g 'test_csharp_query_target:test_csharp_query_target' -t halt -- --csharp-query-keep

# 3. List available tests
csharp_list

# 4. Run a specific test
csharp_run even

# 5. Navigate to inspect the code
cd "${test[even]}"
ls -la
cat Program.cs

# 6. Go back to project root
csharp_root

# 7. Run all tests
for t in "${!test[@]}"; do
    echo "=== Running $t ==="
    csharp_run "$t"
done
```

## Rescanning Tests

If you generate new test directories after sourcing the script, rescan with:

```bash
csharp_scan_tests
```

## Troubleshooting

**No tests found:**
```bash
$ csharp_list
Warning: No C# test directories found. Run tests with --csharp-query-keep first.
```

Solution: Run the tests first:
```bash
SKIP_CSHARP_EXECUTION=1 swipl -q -f init.pl -s tests/core/test_csharp_query_target.pl \
    -g 'test_csharp_query_target:test_csharp_query_target' -t halt -- --csharp-query-keep
```

**Test not found:**
```bash
$ csharp_run foo
Error: Test 'foo' not found.
```

Solution: Check available tests with `csharp_list`.

**CSHARP_TEST_ROOT not set:**
```bash
$ csharp_root
Error: CSHARP_TEST_ROOT not set. Please source csharp_test_nav.sh first.
```

Solution: Source the script again:
```bash
source csharp_test_nav.sh
```

## See Also

- `docs/TESTING_GUIDE.md` - General testing guide
- `docs/development/testing/v0_1_csharp_test_plan.md` - C# test plan
- `docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md` - dotnet run issues
