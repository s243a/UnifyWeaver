# Test Runner Generation - Remaining Issues and Solutions Plan

## Issues Identified

From testing the inference-based test runner, the following issues were identified:

### 1. Demo Scripts Without Callable Functions
**Issue**: Scripts like `demo.sh` are demonstration/documentation scripts, not callable functions
- `demo.sh` - Executes inline, sources other scripts
- Error: `demo: command not found`

**Root Cause**:
- Header extraction finds "# Demo of..." but there's no `demo()` function
- Inference tries to call `demo` as a function
- These are standalone scripts, not libraries with functions

### 2. Misidentified Script Names
**Issue**: Scripts named differently from their contained functions
- `even_odd.sh` contains `is_even()` and `is_odd()`, not `even_odd()`
- `test_even_odd.sh`, `test_group.sh`, `test_count.sh` are test wrappers
- Error: `even_odd: command not found`

**Root Cause**:
- Filename-based fallback assumes filename = function name
- Mutual recursion scripts use descriptive names like "even_odd" but export individual functions
- Test wrapper scripts have "test_" prefix but don't define functions with that name

### 3. Duplicate Script Detection
**Issue**: Multiple scripts compile the same predicate
- `count_items.sh` and `test_count.sh` both compile `count_items/3`
- `list_length.sh` and `test_length.sh` both compile `list_length/2`
- `even_odd.sh` and `test_even_odd.sh` both compile mutual recursion

**Root Cause**:
- Test compilation generates both production (`count_items.sh`) and test wrapper (`test_count.sh`)
- Inference treats them as separate scripts
- Results in duplicate/redundant tests

### 4. Multiple Functions Per Script
**Issue**: Mutual recursion scripts define multiple functions
- `even_odd.sh` defines `is_even()`, `is_odd()`, and helper `is_even_stream()`
- Currently extracting only one signature per script

**Root Cause**:
- `extract_function_signature/2` returns single function name from header
- Mutual recursion files contain multiple entry points
- Missing detection of all exported functions

### 5. Missing Arity for Some Scripts
**Issue**: Some scripts show arity 0 when they should have parameters
- `demo.sh` shows arity 0 (correct - it's a script)
- `even_odd.sh` shows arity 0 (should detect `is_even/1` and `is_odd/1`)

**Root Cause**:
- Header comment doesn't match function definition
- Regex looks for `local var="$N"` in first function only
- Doesn't handle multiple function definitions

## Proposed Solutions

### Solution 1: Script Type Classification
**Implement script classification to distinguish types**

```prolog
%% classify_script_type(+Content, -Type)
%  Classify script as: function_library, demo, test_wrapper, or standalone

classify_script_type(Content, demo) :-
    % Demo scripts have inline execution, no exported functions
    re_match("^echo.*Demo", Content, [multiline(true)]),
    \+ re_match("^\\w+\\(\\) \\{", Content, [multiline(true)]).

classify_script_type(Content, test_wrapper) :-
    % Test wrappers source other scripts and run tests
    re_match("^source.*\\.sh", Content, [multiline(true)]),
    re_match("^echo.*Test", Content, [multiline(true)]).

classify_script_type(Content, function_library) :-
    % Has callable functions
    re_match("^\\w+\\(\\) \\{", Content, [multiline(true)]).

classify_script_type(_Content, standalone).
```

**Action**:
- Skip `demo` and `test_wrapper` types during test generation
- Only generate tests for `function_library` scripts

### Solution 2: Extract All Function Names
**Parse script to find all function definitions**

```prolog
%% extract_all_functions(+Content, -Functions)
%  Extract all function definitions from bash script

extract_all_functions(Content, Functions) :-
    % Find all function definitions: name() {
    re_foldl(collect_function, "^(\\w+)\\(\\)\\s*\\{", Content, [], Functions, [multiline(true)]).

collect_function(Match, FuncsIn, FuncsOut) :-
    get_dict(1, Match, FuncName),
    atom_string(Func, FuncName),
    % Skip helper functions (containing _stream, _memo, etc.)
    \+ sub_atom(Func, _, _, _, '_stream'),
    \+ sub_atom(Func, _, _, _, '_memo'),
    FuncsOut = [Func|FuncsIn].
```

**Action**:
- Extract ALL functions from script
- Filter out helper functions (e.g., `is_even_stream`)
- Return list of callable functions

### Solution 3: Deduplicate Scripts
**Identify and skip duplicate/wrapper scripts**

```prolog
%% is_duplicate_script(+FileName, +AllScripts)
%  Check if script is a duplicate (test wrapper of another script)

is_duplicate_script(FileName, AllScripts) :-
    % test_foo.sh is duplicate if foo.sh exists
    atom_concat('test_', BaseName, FileName),
    member(BaseName, AllScripts), !.

is_duplicate_script(FileName, AllScripts) :-
    % foo_test.sh is duplicate if foo.sh exists
    atom_concat(BaseName, '_test.sh', FileName),
    atom_concat(BaseName, '.sh', ActualFile),
    member(ActualFile, AllScripts), !.
```

**Action**:
- Filter out test wrappers before generating tests
- Prefer production scripts over test wrappers

### Solution 4: Enhanced Signature Extraction
**Extract complete signature including all functions**

```prolog
%% extract_function_signature(+FilePath, -Signature)
%  Enhanced to extract ALL functions and proper metadata

extract_function_signature(FilePath, Signature) :-
    read_file_to_string(FilePath, Content, []),

    % Classify script type
    classify_script_type(Content, ScriptType),

    % Only process function libraries
    ScriptType = function_library, !,

    % Extract all functions
    extract_all_functions(Content, Functions),
    Functions \= [], % Must have functions

    % Get arity for each function
    maplist(extract_function_arity(Content), Functions, Arities),

    % Extract pattern type from header
    extract_pattern_type_from_header(Content, PatternType),

    % Return list of signatures (one per function)
    maplist(create_signature(FilePath, PatternType),
            Functions, Arities, Signatures),

    Signature = Signatures.

%% extract_function_arity(+Content, +FuncName, -Arity)
extract_function_arity(Content, FuncName, Arity) :-
    % Find function definition
    format(atom(Pattern), "^~w\\(\\)\\s*\\{([^}]*)", [FuncName]),
    re_matchsub(Pattern, Content, Match, [multiline(true), dotall(true)]),
    get_dict(1, Match, FuncBody),

    % Count local parameters in function body
    re_foldl(count_match, "local\\s+\\w+=\"\\$\\d+\"", FuncBody, 0, Arity, []).
```

**Changes**:
- Return list of function signatures per script (for mutual recursion)
- Extract arity per function, not just first match
- Skip scripts that aren't function libraries

### Solution 5: Smart Function-to-File Mapping
**Handle one script containing multiple functions**

```prolog
%% generate_explicit_tests_multi(+Stream, +Functions, +Metadata, +Tests)
%  Generate tests for scripts with multiple functions (e.g., mutual recursion)

write_explicit_tests(Stream, Functions, Metadata, Tests) when is_list(Functions) :-
    % Multiple functions from same file (mutual recursion)
    Metadata = metadata(_, _, file_path(FilePath)),
    file_base_name(FilePath, FileName),

    format(Stream, '# Test ~w (mutual recursion)~n', [FileName]),
    format(Stream, 'if [[ -f ~w ]]; then~n', [FileName]),
    format(Stream, '    echo "--- Testing ~w ---"~n', [FileName]),
    format(Stream, '    source ~w~n', [FileName]),
    format(Stream, '~n', []),

    % Write tests for each function
    forall(member(function(FuncName, TestCases), Tests),
           write_function_tests(Stream, FuncName, TestCases, 1)),

    format(Stream, 'fi~n~n', []).
```

**Changes**:
- Detect when multiple functions come from same file
- Group tests by file, but test each function separately
- Add "(mutual recursion)" annotation

### Solution 6: Improved Inference Rules
**Better pattern matching for multi-function scripts**

```prolog
%% infer_test_cases_multi(+Signatures, -TestCases)
%  Infer tests when script contains multiple functions

infer_test_cases_multi([function(is_even, 1, _), function(is_odd, 1, _)], Tests) :-
    !,
    % Mutual recursion: even/odd
    Tests = [
        function(is_even, [
            test('Even: 0', ['0']),
            test('Even: 4', ['4'])
        ]),
        function(is_odd, [
            test('Odd: 3', ['3']),
            test('Odd: 5', ['5'])
        ])
    ].

% Fallback: generate tests for each function independently
infer_test_cases_multi(Signatures, Tests) :-
    maplist(infer_test_cases_single, Signatures, Tests).
```

## Implementation Priority

### Phase 1: Quick Fixes (High Impact, Low Effort)
1. **Skip demo/test wrapper scripts** (Solution 1)
   - Classify script types
   - Filter before processing
   - **Impact**: Eliminates "command not found" errors
   - **Effort**: ~30 minutes

2. **Deduplicate scripts** (Solution 3)
   - Skip test_*.sh when *.sh exists
   - **Impact**: Reduces redundant tests
   - **Effort**: ~20 minutes

### Phase 2: Enhanced Extraction (Medium Impact, Medium Effort)
3. **Extract all functions per script** (Solution 2)
   - Parse all function definitions
   - Filter helpers
   - **Impact**: Properly handles mutual recursion
   - **Effort**: ~1 hour

4. **Per-function arity detection** (Solution 4)
   - Extract arity for each function
   - Better signature accuracy
   - **Impact**: Correct arity for all functions
   - **Effort**: ~45 minutes

### Phase 3: Multi-Function Support (High Impact, High Effort)
5. **Multi-function test generation** (Solution 5)
   - Handle multiple functions from one file
   - Group tests appropriately
   - **Impact**: Full mutual recursion support
   - **Effort**: ~1.5 hours

6. **Enhanced inference rules** (Solution 6)
   - Multi-function pattern detection
   - Smarter test case generation
   - **Impact**: Better test coverage
   - **Effort**: ~1 hour

## Testing Strategy

### After Each Phase
1. Run inference generator: `generate_test_runner_inferred`
2. Execute generated test_runner.sh
3. Verify no "command not found" errors
4. Compare test coverage vs config-based generator

### Success Criteria
- ✅ No "command not found" errors
- ✅ All function libraries have appropriate tests
- ✅ Demo/wrapper scripts excluded
- ✅ Mutual recursion scripts work correctly
- ✅ No duplicate tests
- ✅ Matches or exceeds config-based coverage

## Alternative: Hybrid Approach

For scripts that are difficult to infer:
1. Use inference as default
2. Allow manual override in configuration
3. Merge inferred + configured tests

```prolog
generate_test_runner_hybrid :-
    % Get inferred tests
    generate_inferred_tests(InferredTests),

    % Get manual config
    load_manual_test_config(ManualTests),

    % Merge (manual overrides inferred)
    merge_test_configs(InferredTests, ManualTests, FinalTests),

    % Generate
    generate_runner(FinalTests, explicit, 'output/advanced/test_runner.sh').
```

## Summary

**Immediate Actions** (Phase 1):
- Add script classification
- Skip demo/test wrapper scripts
- Deduplicate scripts

**Expected Improvement**:
- Eliminate all "command not found" errors
- Remove redundant tests
- Cleaner test runner output

**Future Enhancements** (Phases 2-3):
- Full multi-function support
- Enhanced inference rules
- Hybrid inference+config mode

This plan addresses all identified issues while maintaining backward compatibility with the config-based approach.
