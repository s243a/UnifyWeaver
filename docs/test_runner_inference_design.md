# Test Runner Inference-Based Generation - Design Plan

## Overview
This document outlines the design for an inference-based test runner generator that automatically discovers bash scripts, extracts their signatures, infers appropriate test cases, and generates a test_runner.sh script without manual configuration.

## Current State (Configuration-Based)

### Advantages
- ✅ Explicit control over test cases
- ✅ Meaningful error line numbers (each test is explicit)
- ✅ Easy to customize individual test cases
- ✅ Clear, readable generated code

### Limitations
- ❌ Requires manual configuration for each new script
- ❌ Verbose output (repetitive code)
- ❌ Doesn't automatically discover new scripts

## Proposed Inference-Based Approach

### Goals
1. **Automatic Discovery**: Scan `output/advanced/` for all `.sh` files
2. **Signature Extraction**: Parse bash scripts to determine function signatures and arity
3. **Pattern Inference**: Infer test case patterns based on predicate characteristics
4. **Minimal Configuration**: Require no manual updates when new scripts are added

### Implementation Strategy

## 1. Signature Extraction

### Bash Script Analysis
Extract function signatures by parsing the generated bash scripts:

```prolog
%% extract_function_signature(+FilePath, -Signature)
%  Parse bash script to extract function name and arity
extract_function_signature(FilePath, function(Name, Arity, Metadata)) :-
    read_file_to_string(FilePath, Content, []),

    % Extract function name from file header comment
    % Example: "# list_length - linear recursive pattern with memoization"
    (   re_matchsub("^# (?<name>\\w+)\\s*-\\s*(?<desc>.*)",
                     Content, Match, [multiline(true)]) ->
        get_dict(name, Match, Name),
        get_dict(desc, Match, Description)
    ;   % Fallback: extract from filename
        file_base_name(FilePath, FileName),
        file_name_extension(Name, _, FileName),
        Description = unknown
    ),

    % Extract arity by counting function parameters
    % Pattern: function_name() { local arg1="$1" local arg2="$2" ...
    re_findall("local\\s+arg\\d+=\"\\$\\d+\"", Content, Matches),
    length(Matches, Arity),

    % Extract metadata from comments
    extract_pattern_type(Description, PatternType),
    Metadata = metadata(pattern_type(PatternType), description(Description)).

%% extract_pattern_type(+Description, -PatternType)
extract_pattern_type(Desc, tail_recursive) :-
    sub_string(Desc, _, _, _, "tail recursive"), !.
extract_pattern_type(Desc, linear_recursive) :-
    sub_string(Desc, _, _, _, "linear recursive"), !.
extract_pattern_type(Desc, mutual_recursive) :-
    sub_string(Desc, _, _, _, "mutual"), !.
extract_pattern_type(_, unknown).
```

### Signature Patterns Observed

From analyzing generated scripts:

1. **list_length/2**: `list_length(input, result)`
   - Arity: 2
   - Pattern: Linear recursion
   - Input type: List notation `[]`, `[a]`, `[a,b,c]`

2. **factorial/2**: `factorial(input, result)`
   - Arity: 2
   - Pattern: Linear recursion
   - Input type: Numeric

3. **count_items/3**: `count_items(input, accumulator, result_var)`
   - Arity: 3
   - Pattern: Tail recursive with accumulator
   - Input type: List notation

4. **is_even/1**, **is_odd/1**: Single argument
   - Arity: 1
   - Pattern: Mutual recursion
   - Input type: Numeric

## 2. Test Case Inference Rules

### Rule-Based Test Generation

```prolog
%% infer_test_cases(+Signature, -TestCases)
%  Infer appropriate test cases based on signature and pattern type

% Rule 1: Arity 2, Linear recursion, List pattern
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _)),
                 TestCases) :-
    atom_concat(_, length, Name), !,  % Name contains "length"
    TestCases = [
        test('Empty list', '[]', ''),
        test('Single element', '[a]', ''),
        test('Multiple elements', '[a,b,c]', '')
    ].

% Rule 2: Arity 2, Linear recursion, Numeric pattern
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _)),
                 TestCases) :-
    member(Name, [factorial, fib, power]), !,  % Known numeric predicates
    TestCases = [
        test('Base case 0', '0', ''),
        test('Base case 1', '1', ''),
        test('Larger value', '5', '')
    ].

% Rule 3: Arity 3, Tail recursive with accumulator
infer_test_cases(function(_Name, 3, metadata(pattern_type(tail_recursive), _)),
                 TestCases) :-
    TestCases = [
        test('Empty list, accumulator 0', '[]', '0', ''),
        test('List with elements', '[a,b,c]', '0', '')
    ].

% Rule 4: Arity 1, Mutual recursion, Numeric
infer_test_cases(function(Name, 1, metadata(pattern_type(mutual_recursive), _)),
                 TestCases) :-
    atom_concat(Prefix, _, Name),
    member(Prefix, [is_even, is_odd]), !,
    (   Prefix = is_even ->
        TestCases = [
            test('Even: 0', '0'),
            test('Even: 4', '4'),
            test('Odd: 1', '1')
        ]
    ;   TestCases = [
            test('Odd: 3', '3'),
            test('Odd: 5', '5'),
            test('Even: 2', '2')
        ]
    ).

% Fallback: Generic test cases based on arity
infer_test_cases(function(_Name, Arity, _), TestCases) :-
    length(Args, Arity),
    maplist(=('test_value'), Args),
    TestCases = [test('Generic test', Args)].
```

## 3. Test Runner Generation Modes

### Mode 1: Concise (Loop-based)

**Advantages**:
- Compact code
- Easy to maintain
- Scales well with many scripts

**Disadvantages**:
- Less meaningful error line numbers
- Harder to debug individual test failures
- Less customizable per-test

**Generated Structure**:
```bash
#!/bin/bash
# Test runner for generated scripts - CONCISE MODE

declare -A TEST_CONFIGS=(
    ["list_length.sh"]="list_length:2:[],'':single,[a],'':multi,[a,b,c],''"
    ["factorial.sh"]="factorial:2:0,'':1,'':5,''"
    ["count_items.sh"]="count_items:3:[],'0','':items,[a,b,c],'0',''"
)

for script in "${!TEST_CONFIGS[@]}"; do
    if [[ -f "$script" ]]; then
        echo "--- Testing $script ---"
        source "$script"

        IFS=':' read -r func arity tests <<< "${TEST_CONFIGS[$script]}"
        IFS=',' read -ra TEST_ARRAY <<< "$tests"

        test_num=1
        for ((i=0; i<${#TEST_ARRAY[@]}; i+=arity)); do
            echo "Test $test_num"
            args=("${TEST_ARRAY[@]:i:arity}")
            "$func" "${args[@]}"
            ((test_num++))
        done
    fi
done
```

### Mode 2: Explicit (Current approach)

**Advantages**:
- Clear error line numbers
- Easy to debug
- Highly customizable
- Readable test descriptions

**Disadvantages**:
- Verbose
- More code to maintain

**Implementation**: Already implemented in `test_runner_generator.pl`

### Mode 3: Hybrid

Combine both approaches:
- Use concise mode for standard predicates
- Use explicit mode for complex/special cases
- Allow per-script override

## 4. Implementation Plan

### Phase 1: Core Inference Engine
```prolog
% New module: test_runner_inference.pl

:- module(test_runner_inference, [
    generate_test_runner_inferred/0,
    generate_test_runner_inferred/1,
    generate_test_runner_inferred/2   % With mode option
]).

% Scan directory for scripts
scan_output_directory(-Scripts)

% Extract signatures from all scripts
extract_all_signatures(+Scripts, -Signatures)

% Infer test cases for each signature
infer_all_test_cases(+Signatures, -TestConfigs)

% Generate test runner in specified mode
generate_runner(+TestConfigs, +Mode, +OutputPath)
```

### Phase 2: Pattern Recognition
- Implement signature extraction predicates
- Build inference rules library
- Create pattern matching logic

### Phase 3: Code Generation
- Implement concise mode generator
- Implement hybrid mode generator
- Add mode selection parameter

### Phase 4: Integration
- Update `test_advanced.pl` to optionally use inference mode
- Add command-line flag to choose mode
- Provide migration path from config-based to inference-based

## 5. Configuration Options

### User-Controllable Parameters

```prolog
% In test_runner_generator.pl, add:

generate_test_runner :-
    generate_test_runner(default_options).

generate_test_runner(Options) :-
    option(mode(Mode), Options, explicit),        % explicit|concise|hybrid
    option(output(Output), Options, 'output/advanced/test_runner.sh'),
    option(method(Method), Options, config),      % config|inferred|hybrid

    % Choose generation method
    (   Method = config ->
        generate_config_based(Mode, Output)
    ;   Method = inferred ->
        generate_inference_based(Mode, Output)
    ;   generate_hybrid(Mode, Output)
    ).
```

### Example Usage

```prolog
% Concise mode, configuration-based
generate_test_runner([mode(concise), method(config)]).

% Explicit mode, inference-based
generate_test_runner([mode(explicit), method(inferred)]).

% Hybrid mode
generate_test_runner([mode(hybrid), method(hybrid)]).
```

## 6. Testing Strategy

### Validation
1. Generate test runner with inference
2. Compare output with manual configuration
3. Run both versions and verify identical results
4. Check edge cases (empty directory, single script, etc.)

### Test Cases
- Empty output directory
- Single script
- Multiple scripts of same pattern
- Mixed pattern types
- Scripts with unusual signatures

## 7. Future Enhancements

### Smart Test Value Selection
- Analyze script base cases to extract actual base values
- Parse recursive patterns to determine meaningful test inputs
- Use type inference from variable names

### Learning Mode
- Track which test cases find bugs
- Adjust inference rules based on success rate
- Build confidence scores for different inference rules

### Interactive Configuration
- Prompt user for test cases on first run
- Save learned configurations
- Allow manual override with fallback to inference

## 8. Migration Path

### Gradual Adoption
1. **Phase 1**: Keep config-based as default, add inference as option
2. **Phase 2**: Generate both, compare outputs
3. **Phase 3**: Make inference default, keep config as fallback
4. **Phase 4**: Deprecate pure config mode, recommend hybrid

### Backward Compatibility
- Preserve existing `generate_test_runner/0` interface
- Add new `generate_test_runner_inferred/0` alongside
- Allow mode selection via environment variable

## Summary

This inference-based approach will:
- ✅ Eliminate manual test configuration
- ✅ Automatically adapt to new scripts
- ✅ Provide flexible output modes (concise vs explicit)
- ✅ Maintain backward compatibility
- ✅ Enable future smart enhancements

The implementation will be modular, allowing users to choose:
- **Method**: Configuration vs Inference vs Hybrid
- **Mode**: Concise vs Explicit vs Hybrid
- **Customization**: Per-script overrides

This gives maximum flexibility while reducing maintenance burden.
