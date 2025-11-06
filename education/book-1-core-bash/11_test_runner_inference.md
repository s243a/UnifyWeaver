<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 11: Test Runner Inference

## Introduction

After compiling your Prolog predicates to Bash, you need to test them. But manually writing test scripts for every generated function is tedious and error-prone. What if the test suite could **generate itself automatically** by analyzing the generated code?

That's exactly what UnifyWeaver's **test runner inference** system does. It scans your output directory, analyzes the generated Bash scripts, infers appropriate test cases based on function signatures and patterns, and generates a complete test runner automatically.

This chapter explores how test runner inference works, how to use it, and how UnifyWeaver makes intelligent decisions about what tests to generate.

## 1. The Problem: Manual Testing is Tedious

Consider what you'd need to manually test your generated scripts:

```bash
# Manually written test script
source output/advanced/factorial.sh
source output/advanced/list_length.sh
source output/advanced/tree_sum.sh
source output/advanced/even_odd.sh

echo "Testing factorial..."
factorial 0 ""
factorial 5 ""

echo "Testing list_length..."
list_length "[]" ""
list_length "[a,b,c]" ""

echo "Testing tree_sum..."
tree_sum "[]"
tree_sum "[5,[3,[1,[],[]]],[2,[],[]]]"

echo "Testing is_even..."
is_even 0
is_even 4
is_even 3  # should fail

echo "Testing is_odd..."
is_odd 1
is_odd 3
```

**Problems:**
- Must manually track all generated scripts
- Need to know the arity of each function
- Must infer appropriate test inputs for each pattern type
- Forgetting to test a function means no coverage
- Adding new predicates requires updating the test script

## 2. The Solution: Automatic Test Discovery

UnifyWeaver's test runner inference automates this entire process:

```prolog
% One command generates complete test suite
?- generate_test_runner_inferred.
Generated test runner (inferred, explicit mode): output/advanced/test_runner.sh
```

The system:
1. **Scans** the output directory for `.sh` files
2. **Extracts** function signatures (name, arity, pattern type)
3. **Infers** appropriate test cases based on the signature
4. **Generates** a complete bash test runner script

## 3. How Test Runner Inference Works

### Step 1: Script Discovery

The inference system scans the output directory and classifies each script:

**Script Types:**

1. **function_library** - Callable functions (what we want to test)
2. **demo** - Demo scripts with inline execution
3. **test_wrapper** - Existing test scripts (exclude from inference)
4. **standalone** - Other scripts

**Implementation (from `src/unifyweaver/core/advanced/test_runner_inference.pl`):**

```prolog
classify_script_type(Content, demo) :-
    % Demo scripts have inline execution
    sub_string(Content, _, _, _, "Demo"),
    sub_string(Content, _, _, _, "echo \"╔"),
    !.

classify_script_type(Content, test_wrapper) :-
    % Test wrappers source other scripts
    sub_string(Content, _, _, _, "source"),
    sub_string(Content, _, _, _, "Testing"),
    !.

classify_script_type(Content, function_library) :-
    % Has callable functions - look for function_name() { pattern
    re_match("^\\w+\\(\\)\\s*\\{", Content, [multiline(true)]),
    !.
```

**Filtering logic:**

```prolog
should_include_script(FilePath, AllFiles) :-
    file_base_name(FilePath, FileName),

    % Exclude test runner scripts
    \+ atom_concat('test_runner', _, FileName),

    % Exclude test wrappers if production version exists
    \+ atom_concat('test_', _, FileName),

    % Check if it's a processable script type
    read_file_to_string(FilePath, Content, []),
    classify_script_type(Content, ScriptType),
    ScriptType = function_library.
```

### Step 2: Function Signature Extraction

For each script, the system extracts:
- **Function name(s)** - Main callable functions (excludes helpers like `_stream`, `_memo`, `parse_tree`)
- **Arity** - Number of arguments by counting `local arg$N` declarations
- **Pattern type** - Extracted from header comments
- **Description** - Human-readable description

**Header parsing:**

```prolog
% Extract header description - two formats supported:
% Format 1: "# predicate_name - description"
% Format 2: "# Mutually recursive group: names"

(   re_matchsub("^#\\s*(?<name>\\w+)\\s*-\\s*(?<desc>[^\\n]+)",
                 Content, Match, [multiline(true)]) ->
    get_dict(desc, Match, DescStr),
    atom_string(Description, DescStr)
;   re_matchsub("^#\\s*(?<desc>Mutually recursive[^\\n]+)",
                 Content, Match2, [multiline(true)]) ->
    get_dict(desc, Match2, DescStr2),
    atom_string(Description, DescStr2)
;   Description = unknown
).
```

**Function extraction:**

```prolog
% Extract ALL functions from script
extract_all_functions(Content, Functions) :-
    re_foldl(collect_function_name, "^(\\w+)\\(\\)\\s*\\{", Content, [], AllFuncs, [multiline(true)]),
    % Filter out helper functions
    include(is_main_function, AllFuncs, FilteredFuncs),
    list_to_set(FilteredFuncs, Functions).

% Check if function is main (not a helper)
is_main_function(FuncName) :-
    \+ sub_atom(FuncName, _, _, _, '_stream'),
    \+ sub_atom(FuncName, _, _, _, '_memo'),
    FuncName \= parse_tree.
```

**Arity detection:**

```prolog
extract_function_arity(Content, FuncName, Arity) :-
    % Find function definition and extract its body
    format(atom(Pattern), "^~w\\(\\)\\s*\\{([^}]*)", [FuncName]),
    (   re_matchsub(Pattern, Content, Match, [multiline(true), dotall(true)]) ->
        get_dict(1, Match, FuncBodyStr),
        % Count local parameters: local arg1="$1"
        re_foldl(count_match, "local\\s+\\w+=\"\\$\\d+\"", FuncBodyStr, 0, Arity, [])
    ;   Arity = 0
    ).
```

**Result structure:**

```prolog
Signature = function(
    FunctionName,
    Arity,
    metadata(
        pattern_type(PatternType),
        description(Description),
        file_path(FilePath)
    )
)
```

### Step 3: Test Case Inference

Based on the signature, the system infers appropriate test cases using **heuristic rules**:

**Rule 1: Arity 2, Linear recursion, List-related**

```prolog
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _, _)),
                 TestCases) :-
    (sub_atom(Name, _, _, _, length) ; sub_atom(Name, _, _, _, list)), !,
    TestCases = [
        test('Empty list', ['[]', '']),
        test('Single element list', ['[a]', '']),
        test('Three element list', ['[a,b,c]', ''])
    ].
```

**Rule 2: Arity 2, Linear recursion, Numeric**

```prolog
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _, _)),
                 TestCases) :-
    member(Name, [factorial, fib, power]), !,
    TestCases = [
        test('Base case 0', ['0', '']),
        test('Base case 1', ['1', '']),
        test('Larger value', ['5', ''])
    ].
```

**Rule 3: Arity 3, Tail recursive with accumulator**

```prolog
infer_test_cases(function(Name, 3, metadata(pattern_type(PatternType), _, _)),
                 TestCases) :-
    member(PatternType, [tail_recursive, accumulator]), !,
    (   (sub_atom(Name, _, _, _, sum) ; sub_atom(Name, _, _, _, add)) ->
        TestCases = [
            test('Empty list with accumulator 0', ['[]', '0', '']),
            test('Numeric list', ['[1,2,3]', '0', '']),
            test('Larger list', ['[5,10,15]', '0', ''])
        ]
    ;   % Default accumulator tests
        TestCases = [
            test('Empty list with accumulator 0', ['[]', '0', '']),
            test('List with elements', ['[a,b,c]', '0', ''])
        ]
    ).
```

**Rule 4: Arity 1, Mutual recursion (even/odd)**

```prolog
infer_test_cases(function(Name, 1, metadata(pattern_type(mutual_recursive), _, _)),
                 TestCases) :-
    (sub_atom(Name, 0, _, _, is_even) ; sub_atom(Name, 0, _, _, even)), !,
    TestCases = [
        test('Even: 0', ['0']),
        test('Even: 4', ['4']),
        test('Odd (should fail): 3', ['3'])
    ].
```

**Rule 5: Arity 1, Tree recursion**

```prolog
infer_test_cases(function(Name, 1, metadata(pattern_type(tree_recursive), _, _)),
                 TestCases) :-
    (sub_atom(Name, _, _, _, tree) ; sub_atom(Name, _, _, _, binary)), !,
    TestCases = [
        test('Empty tree', ['[]']),
        test('Single node', ['[5,[],[]]']),
        test('Small tree', ['[10,[5,[],[3,[],[]]],[7,[],[]]]'])
    ].
```

**Fallback: Generic test based on arity**

```prolog
infer_test_cases(function(_Name, Arity, _), TestCases) :-
    length(Args, Arity),
    maplist(=('test_value'), Args),
    TestCases = [test('Generic test', Args)].
```

### Step 4: Test Runner Generation

UnifyWeaver supports three output modes:

#### Mode 1: Explicit (Default)

One test block per file, clearly labeled and structured:

```bash
#!/bin/bash
# Test runner for generated advanced recursion scripts - explicit MODE

echo "=== Testing Generated Bash Scripts ==="

# Test factorial.sh
if [[ -f factorial.sh ]]; then
    echo "--- Testing factorial.sh ---"
    source factorial.sh

    echo "Test 1: Base case 0"
    factorial "0" ""

    echo ""
    echo "Test 2: Base case 1"
    factorial "1" ""

    echo ""
    echo "Test 3: Larger value"
    factorial "5" ""

    echo ""
fi
```

#### Mode 2: Concise

Loop-based test runner using associative arrays:

```bash
#!/bin/bash
# Test configurations (script:function:arity:test_args)
declare -A TEST_CONFIGS=(
    ["factorial.sh"]="factorial:2:0,:1,:5,"
    ["list_length.sh"]="list_length:2:[],:,[a],:,[a,b,c],"
)

for script in "${!TEST_CONFIGS[@]}"; do
    if [[ -f "$script" ]]; then
        echo "--- Testing $script ---"
        source "$script"

        IFS=':' read -r func arity tests <<< "${TEST_CONFIGS[$script]}"
        IFS=':' read -ra TEST_ARRAY <<< "$tests"

        test_num=1
        for test_args in "${TEST_ARRAY[@]}"; do
            echo "Test $test_num"
            IFS=',' read -ra args <<< "$test_args"
            "$func" "${args[@]}"
            echo ""
            ((test_num++))
        done
    fi
done
```

#### Mode 3: Hybrid

Smart selection based on complexity (currently uses explicit mode).

### Multi-Function Scripts (Mutual Recursion)

The inference system handles mutual recursion correctly by:
1. Detecting multiple functions in one file
2. Sourcing the file once
3. Testing all functions together
4. Annotating the output

**Example:**

```bash
# Test even_odd.sh (multi-function: 2 functions)
if [[ -f even_odd.sh ]]; then
    echo "--- Testing even_odd.sh ---"
    source even_odd.sh

    # Function: is_even
    echo "Test 1: Even: 0"
    is_even "0"

    echo ""
    echo "Test 2: Even: 4"
    is_even "4"

    echo ""

    # Function: is_odd
    echo "Test 1: Odd: 3"
    is_odd "3"

    echo ""
    echo "Test 2: Odd: 5"
    is_odd "5"

    echo ""
fi
```

## 4. Using Test Runner Inference

### Basic Usage

**From Prolog:**

```prolog
% Load the module
?- use_module(unifyweaver(core/advanced/test_runner_inference)).

% Generate test runner with default options (explicit mode)
?- generate_test_runner_inferred.
Generated test runner (inferred, explicit mode): output/advanced/test_runner.sh

% Specify output path
?- generate_test_runner_inferred('my_tests.sh').
Generated test runner (inferred, explicit mode): my_tests.sh

% Use concise mode
?- generate_test_runner_inferred('my_tests.sh', [mode(concise)]).
Generated test runner (inferred, concise mode): my_tests.sh

% Scan different directory
?- generate_test_runner_inferred('tests.sh', [output_dir('output/basic')]).
Generated test runner (inferred, explicit mode): tests.sh
```

**From Command Line:**

```bash
# Generate test runner
swipl -g "use_module(unifyweaver(core/advanced/test_runner_inference)), \
          generate_test_runner_inferred, \
          halt."

# Run the generated tests
bash output/advanced/test_runner.sh
```

### Integration with Chapter 4 Workflow

Recall from Chapter 4 that we manually compiled predicates. Now we can add automatic testing:

```prolog
% Step 1: Compile your predicates (from Chapter 4)
?- compile_recursive(factorial/2, [], BashCode),
   open('output/advanced/factorial.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).

?- compile_recursive(list_length/2, [], BashCode),
   open('output/advanced/list_length.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).

% Step 2: Generate test runner automatically
?- generate_test_runner_inferred('output/advanced/test_runner.sh').
Generated test runner (inferred, explicit mode): output/advanced/test_runner.sh

% Step 3: Exit Prolog and run tests
?- halt.
```

```bash
# Run the inferred tests
$ bash output/advanced/test_runner.sh

=== Testing Generated Bash Scripts ===

--- Testing factorial.sh ---
Test 1: Base case 0
0:1

Test 2: Base case 1
1:1

Test 3: Larger value
5:120

--- Testing list_length.sh ---
Test 1: Empty list
[]:0

Test 2: Single element list
[a]:1

Test 3: Three element list
[a,b,c]:3

=== All Tests Complete ===
```

## 5. Example: Complete Generated Test Runner

Here's what a real generated test runner looks like (from `output/advanced/inferred_test_runner.sh`):

```bash
#!/bin/bash
# Test runner for generated advanced recursion scripts - explicit MODE
# AUTO-GENERATED BY INFERENCE - DO NOT EDIT MANUALLY
#
# Generated by: test_runner_inference.pl
# To regenerate: swipl -g "use_module(unifyweaver(core/advanced/test_runner_inference)), generate_test_runner_inferred, halt."

echo "=== Testing Generated Bash Scripts ==="
echo ""

# Test count_items.sh (multi-function: 2 functions)
if [[ -f count_items.sh ]]; then
    echo "--- Testing count_items.sh ---"
    source count_items.sh

    # Function: count_items
    echo "Test 1: Empty list with accumulator 0"
    count_items "[]" "0" ""

    echo ""
    echo "Test 2: List with elements"
    count_items "[a,b,c]" "0" ""

    echo ""
fi

# Test factorial.sh
if [[ -f factorial.sh ]]; then
    echo "--- Testing factorial.sh ---"
    source factorial.sh

    echo "Test 1: Base case 0"
    factorial "0" ""

    echo ""
    echo "Test 2: Base case 1"
    factorial "1" ""

    echo ""
    echo "Test 3: Larger value"
    factorial "5" ""

    echo ""
fi

# Test tree_sum.sh
if [[ -f tree_sum.sh ]]; then
    echo "--- Testing tree_sum.sh ---"
    source tree_sum.sh

    echo "Test 1: Empty tree"
    tree_sum "[]"

    echo ""
    echo "Test 2: Single node"
    tree_sum "[5,[],[]]"

    echo ""
    echo "Test 3: Small tree"
    tree_sum "[10,[5,[],[3,[],[]]],[7,[],[]]]"

    echo ""
fi

echo "=== All Tests Complete ==="
```

## 6. Inference Heuristics: How It Decides

The inference system uses several heuristics to generate appropriate tests:

### Heuristic 1: Name-Based Inference

```prolog
% If function name contains "length" or "list"
(sub_atom(Name, _, _, _, length) ; sub_atom(Name, _, _, _, list))
→ Generate list test cases: [], [a], [a,b,c]

% If function name is factorial, fib, or power
member(Name, [factorial, fib, power])
→ Generate numeric test cases: 0, 1, 5

% If function name contains "sum" or "add"
(sub_atom(Name, _, _, _, sum) ; sub_atom(Name, _, _, _, add))
→ Generate numeric list test cases: [], [1,2,3], [5,10,15]

% If function name contains "tree" or "binary"
(sub_atom(Name, _, _, _, tree) ; sub_atom(Name, _, _, _, binary))
→ Generate tree test cases: [], [5,[],[]], [10,[5,[],[3,[],[]]],[7,[],[]]]

% If function name is is_even or even
(sub_atom(Name, 0, _, _, is_even) ; sub_atom(Name, 0, _, _, even))
→ Generate even test cases: 0, 4, 3 (should fail)

% If function name is is_odd or odd
(sub_atom(Name, 0, _, _, is_odd) ; sub_atom(Name, 0, _, _, odd))
→ Generate odd test cases: 3, 5, 6 (should fail)
```

### Heuristic 2: Pattern-Based Inference

```prolog
% Pattern type: tail_recursive or accumulator
pattern_type(tail_recursive) OR pattern_type(accumulator)
→ Generate accumulator tests with initial value 0

% Pattern type: linear_recursive
pattern_type(linear_recursive)
→ Check name for specific patterns (factorial, list, etc.)

% Pattern type: mutual_recursive
pattern_type(mutual_recursive)
→ Generate complementary test cases (even/odd)

% Pattern type: tree_recursive
pattern_type(tree_recursive)
→ Generate structural test cases (empty, single node, small tree)
```

### Heuristic 3: Arity-Based Inference

```prolog
% Arity 1: Typically unary predicates (is_even, tree_sum)
Arity = 1
→ Single argument tests based on pattern

% Arity 2: Typically binary relations (factorial, list_length)
Arity = 2
→ Input + output tests (second arg usually empty string for output)

% Arity 3: Typically accumulators (count_items, sum_list)
Arity = 3
→ Input + accumulator + output tests (acc starts at 0)
```

## 7. Advantages of Inference-Based Testing

### 1. Zero Manual Effort

You never write test scripts manually - they generate automatically.

### 2. Complete Coverage

Every generated function gets tested automatically.

### 3. Consistent Test Patterns

All similar functions get similar tests, ensuring uniformity.

### 4. Easy Regeneration

Change your Prolog code, recompile, regenerate tests - all automated.

### 5. Self-Documenting

Generated tests serve as examples of how to call each function.

### 6. Regression Detection

Rerun tests after changes to catch regressions immediately.

## 8. Exercises

### Exercise 1: Generate and Run Tests

1. Compile a few predicates from Chapter 9:
   ```prolog
   ?- compile_recursive(factorial/2, [], Code1),
      open('output/advanced/factorial.sh', write, S1),
      write(S1, Code1),
      close(S1).

   ?- compile_recursive(list_length/2, [], Code2),
      open('output/advanced/list_length.sh', write, S2),
      write(S2, Code2),
      close(S2).
   ```

2. Generate the test runner:
   ```prolog
   ?- generate_test_runner_inferred('output/advanced/my_tests.sh').
   ```

3. Run it:
   ```bash
   bash output/advanced/my_tests.sh
   ```

**Question:** What test cases were inferred for each function?

### Exercise 2: Understand Inference Rules

Given this function signature:
```prolog
function(sum_tree, 1, metadata(pattern_type(tree_recursive), description('Tree sum'), file_path('sum_tree.sh')))
```

**Question:** What test cases would be inferred? Why?

**Answer:** Tree test cases would be inferred:
- Empty tree: `[]`
- Single node: `[5,[],[]]`
- Small tree: `[10,[5,[],[3,[],[]]],[7,[],[]]]`

Because:
1. Name contains "tree" → name-based inference
2. Pattern type is `tree_recursive` → pattern-based inference
3. Arity is 1 → single argument tests

### Exercise 3: Add a Custom Inference Rule

Suppose you want to add support for a `max_list/2` predicate. Write the inference rule:

**Solution:**
```prolog
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _, _)),
                 TestCases) :-
    sub_atom(Name, _, _, _, max), !,
    TestCases = [
        test('Single element', ['[5]', '']),
        test('Ascending list', ['[1,2,3,4,5]', '']),
        test('Descending list', ['[5,4,3,2,1]', '']),
        test('Mixed list', ['[3,1,4,1,5]', ''])
    ].
```

### Exercise 4: Compare Output Modes

Generate test runners in all three modes:

```prolog
?- generate_test_runner_inferred('explicit.sh', [mode(explicit)]).
?- generate_test_runner_inferred('concise.sh', [mode(concise)]).
?- generate_test_runner_inferred('hybrid.sh', [mode(hybrid)]).
```

**Question:** What are the trade-offs between explicit and concise modes?

**Answer:**
- **Explicit:** More readable, easier to debug, larger file size
- **Concise:** More compact, loop-based, harder to debug specific tests
- **Hybrid:** Best of both (currently defaults to explicit)

## 9. Limitations and Future Work

### Current Limitations

1. **Heuristic-Based:** May not always infer the best test cases
2. **No Expected Results:** Tests don't verify correctness, just run functions
3. **Limited Pattern Coverage:** Some patterns may get generic tests
4. **No Custom Test Data:** Can't specify domain-specific test inputs

### Future Enhancements

1. **Expected Output Verification:** Generate assertions for correct output
2. **Property-Based Testing:** Generate random test cases based on properties
3. **Custom Test Hints:** Allow annotations in Prolog source to guide inference
4. **Coverage Analysis:** Report which functions were tested
5. **Performance Benchmarking:** Add timing measurements to tests

## 10. Integration with CI/CD

Test runner inference integrates seamlessly with continuous integration:

```bash
#!/bin/bash
# ci_test.sh - Run all tests in CI

# Compile all predicates
swipl -g "
    use_module(unifyweaver(core/recursive_compiler)),
    compile_all_predicates('src/predicates.pl', 'output/advanced/'),
    halt.
"

# Generate test runner
swipl -g "
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('output/advanced/test_runner.sh'),
    halt.
"

# Run tests
bash output/advanced/test_runner.sh

# Check exit code
if [ $? -eq 0 ]; then
    echo "✓ All tests passed"
    exit 0
else
    echo "✗ Tests failed"
    exit 1
fi
```

## Summary

Test runner inference is a powerful automation tool that eliminates the tedious work of writing test scripts manually. By analyzing generated Bash code and using intelligent heuristics, UnifyWeaver can automatically:

1. **Discover** all generated functions
2. **Classify** their patterns (tail, linear, tree, mutual recursion)
3. **Infer** appropriate test cases based on signatures
4. **Generate** complete, runnable test scripts

This closes the loop on UnifyWeaver's automated workflow: write declarative Prolog → compile to optimized Bash → test automatically.

### Key Takeaways

- **Automatic test discovery** eliminates manual test script writing
- **Signature extraction** uses regex parsing to understand function structure
- **Heuristic inference** generates appropriate test cases based on patterns
- **Multiple output modes** support different use cases (explicit, concise, hybrid)
- **Multi-function support** handles mutual recursion correctly
- **Zero-configuration** works out of the box for standard patterns

## Next Steps

- **Jupyter Notebooks:** Interactive exploration of UnifyWeaver features
- **Appendix A:** Detailed recursion pattern theory and classification algorithms
- **Complete Examples:** End-to-end projects using all UnifyWeaver features

## References

- `src/unifyweaver/core/advanced/test_runner_inference.pl` - Main inference engine
- `output/advanced/inferred_test_runner.sh` - Example generated test runner
- Chapter 9: Advanced Recursion Patterns - Understanding what patterns to test
- Chapter 10: Prolog Introspection - Understanding how code analysis works

---

## Navigation

**←** [Previous: Chapter 10: Prolog Introspection](10_prolog_introspection) | [📖 Back to Book 1](README) | [Next: Chapter 12: Recursive Compilation →](12_recursive_compilation)

