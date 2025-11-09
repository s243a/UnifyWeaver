:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_runner_inference.pl - Inference-based test runner generation
% Automatically discovers scripts and infers test cases based on signatures

:- module(test_runner_inference, [
    generate_test_runner_inferred/0,
    generate_test_runner_inferred/1,
    generate_test_runner_inferred/2,
    extract_function_signature/2,
    infer_test_cases/3 % Updated arity
]).

:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).
:- use_module(library(option)).

%% generate_test_runner_inferred/0
%  Generate test runner using inference with default options
generate_test_runner_inferred :-
    generate_test_runner_inferred('output/advanced/test_runner.sh', [mode(explicit)]).

%% generate_test_runner_inferred(+OutputPath)
%  Generate test runner at specified path with default mode
generate_test_runner_inferred(OutputPath) :-
    generate_test_runner_inferred(OutputPath, [mode(explicit)]).

%% generate_test_runner_inferred(+OutputPath, +Options)
%  Generate test runner with specified options
%  Options:
%    - mode(explicit|concise|hybrid) - output format (default: explicit)
%    - output_dir(Dir) - directory to scan for scripts (default: 'output/advanced')
generate_test_runner_inferred(OutputPath, Options) :-
    option(mode(Mode), Options, explicit),
    option(output_dir(OutputDir), Options, 'output/advanced'),

    % Scan output directory for scripts
    scan_output_directory(OutputDir, Scripts),

    % Extract signatures from all scripts
    extract_all_signatures(Scripts, Signatures),

    % Infer test cases for each signature
    infer_all_test_cases(Signatures, TestConfigs),

    % Generate test runner based on mode
    generate_runner(TestConfigs, Mode, OutputPath),

    format('Generated test runner (inferred, ~w mode): ~w~n', [Mode, OutputPath]).

%% scan_output_directory(+OutputDir, -Scripts)
%  Find all .sh files in the specified directory
%  Filters out demo scripts, test wrappers, and duplicates
scan_output_directory(OutputDir, Scripts) :-
    (   exists_directory(OutputDir) ->
        atomic_list_concat([OutputDir, '/*.sh'], Pattern),
        expand_file_name(Pattern, AllFiles),
        findall(File,
                (   member(File, AllFiles),
                    should_include_script(File, AllFiles)
                ),
                Scripts)
    ;   Scripts = []
    ).

%% should_include_script(+FilePath, +AllFiles)
%  Determine if script should be included in test generation
should_include_script(FilePath, AllFiles) :-
    file_base_name(FilePath, FileName),

    % Exclude test runner scripts
    \+ atom_concat('test_runner', _, FileName),

    % Exclude test wrappers if production version exists
    \+ is_test_wrapper_duplicate(FileName, AllFiles),

    % Check if it's a processable script type
    read_file_to_string(FilePath, Content, []),
    classify_script_type(Content, ScriptType),
    ScriptType = function_library.

%% is_test_wrapper_duplicate(+FileName, +AllFiles)
%  Check if this is a test wrapper - exclude all test_* files
is_test_wrapper_duplicate(FileName, _AllFiles) :-
    % Simply exclude any file starting with test_
    % (they are generated test wrappers, not production scripts)
    atom_concat('test_', _, FileName),
    !.

%% classify_script_type(+Content, -ScriptType)
%  Classify bash script as: function_library, demo, test_wrapper, or standalone
classify_script_type(Content, demo) :-
    % Demo scripts have inline execution, typically echo statements for demo
    sub_string(Content, _, _, _, "Demo"),
    sub_string(Content, _, _, _, "echo \"╔"),
    !.

classify_script_type(Content, test_wrapper) :-
    % Test wrappers source other scripts and typically have "Test" in output
    sub_string(Content, _, _, _, "source"),
    sub_string(Content, _, _, _, "Testing"),
    !.

classify_script_type(Content, function_library) :-
    % Has callable functions - look for function_name() { pattern
    re_match("^\\w+\\(\\)\\s*\\{", Content, [multiline(true)]),
    !.

classify_script_type(_, standalone).

%% extract_all_signatures(+Scripts, -Signatures)
%  Extract function signatures from all script files
extract_all_signatures(Scripts, Signatures) :-
    findall(Sig,
            (   member(Script, Scripts),
                catch(extract_function_signature(Script, Sig), _, fail)
            ),
            Signatures).

%% extract_function_signature(+FilePath, -Signature)
%  Parse bash script to extract function name, arity, and metadata
%  Returns: function(Name, Arity, Metadata) OR list of functions for multi-function scripts
extract_function_signature(FilePath, Signature) :-
    read_file_to_string(FilePath, Content, []),

    % Extract header description - try both formats
    (   re_matchsub("^#\\s*(?<name>\\w+)\\s*-\\s*(?<desc>[^\\n]+)",
                     Content, Match, [multiline(true)]) ->
        get_dict(desc, Match, DescStr),
        atom_string(Description, DescStr)
    ;   % Try alternate format: "# Mutually recursive group: name"
        re_matchsub("^#\\s*(?<desc>Mutually recursive[^\\n]+)",
                     Content, Match2, [multiline(true)]) ->
        get_dict(desc, Match2, DescStr2),
        atom_string(Description, DescStr2)
    ;   Description = unknown
    ),

    % Extract ALL functions from the script
    extract_all_functions(Content, Functions),

    % Extract pattern type from description
    extract_pattern_type(Description, PatternType),

    % Build metadata
    Metadata = metadata(
        pattern_type(PatternType),
        description(Description),
        file_path(FilePath)
    ),

    % Return single function or list depending on count
    (   Functions = [SingleFunc] ->
        % Single function - return as before
        extract_function_arity(Content, SingleFunc, Arity),
        Signature = function(SingleFunc, Arity, Metadata)
    ;   % Multiple functions - return list with per-function arity
        maplist(build_function_signature(Content, Metadata), Functions, Signatures),
        Signature = Signatures
    ).

%% build_function_signature(+Content, +Metadata, +FuncName, -Signature)
build_function_signature(Content, Metadata, FuncName, function(FuncName, Arity, Metadata)) :-
    extract_function_arity(Content, FuncName, Arity).

%% extract_all_functions(+Content, -Functions)
%  Extract all function definitions from bash script, excluding helpers
extract_all_functions(Content, Functions) :-
    % Use re_foldl to find all function definitions
    re_foldl(collect_function_name, "^(\\w+)\\(\\)\\s*\\{", Content, [], AllFuncs, [multiline(true)]),
    % Filter out helper functions and remove duplicates
    include(is_main_function, AllFuncs, FilteredFuncs),
    list_to_set(FilteredFuncs, Functions).

%% collect_function_name(+Match, +ListIn, -ListOut)
%  Accumulator for re_foldl to collect function names
collect_function_name(Match, ListIn, ListOut) :-
    get_dict(1, Match, FuncNameStr),
    atom_string(FuncName, FuncNameStr),
    ListOut = [FuncName | ListIn].

%% is_main_function(+FuncName)
%  True if function is a main callable function (not a helper)
is_main_function(FuncName) :-
    \+ sub_atom(FuncName, _, _, _, '_stream'),
    \+ sub_atom(FuncName, _, _, _, '_memo'),
    FuncName \= parse_tree.  % Helper function for tree recursion

%% extract_function_arity(+Content, +FuncName, -Arity)
%  Extract arity for a specific function by finding its body and counting parameters

% Special case for known mutual recursive functions where arity inference is tricky
extract_function_arity(_Content, FuncName, 1) :-
    member(FuncName, [is_even, is_odd]), !.

extract_function_arity(Content, FuncName, Arity) :-
    % Find function definition and extract its body
    format(atom(Pattern), "^~w\\(\\)\\s*\\{([^}]*)", [FuncName]),
    (   re_matchsub(Pattern, Content, Match, [multiline(true), dotall(true)]) ->
        get_dict(1, Match, FuncBodyStr),
        % Count local parameters in function body (existing logic)
        re_foldl(count_match_local_param, "local\\s+\\w+=\"\\$\\d+\"", FuncBodyStr, 0, LocalArity, []),
        
        % Additionally, find highest $N used in the function body
        re_foldl(find_max_param_ref, "\\$(\\d+)", FuncBodyStr, 0, MaxParamRef, []),

        % The arity is the maximum of explicitly declared local params and highest param reference
        Arity is max(LocalArity, MaxParamRef)
    ;   % If can't extract body, default to 0
        Arity = 0
    ).

%% count_match_local_param(+Match, +CountIn, -CountOut)
%  Helper for re_foldl to count matches for local parameters
count_match_local_param(_, CountIn, CountOut) :-
    CountOut is CountIn + 1.

%% find_max_param_ref(+Match, +MaxIn, -MaxOut)
%  Helper for re_foldl to find the maximum parameter reference ($N)
find_max_param_ref(Match, MaxIn, MaxOut) :-
    get_dict(1, Match, ParamNumStr),
    atom_string(ParamNumAtom, ParamNumStr),
    atom_number(ParamNumAtom, ParamNum),
    MaxOut is max(MaxIn, ParamNum).

%% extract_pattern_type(+Description, -PatternType)
%  Infer pattern type from description string
extract_pattern_type(Desc, tail_recursive) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'tail recursive'), !.
extract_pattern_type(Desc, linear_recursive) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'linear recursive'), !.
extract_pattern_type(Desc, linear_recursive) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'linear recursion'), !.
extract_pattern_type(Desc, mutual_recursive) :-
    atom(Desc),
    downcase_atom(Desc, DescLower),
    sub_atom(DescLower, _, _, _, 'mutual'), !.
extract_pattern_type(Desc, accumulator) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'accumulator'), !.
extract_pattern_type(_, unknown).

%% infer_all_test_cases(+Signatures, -TestConfigs)
%  Infer test cases for all signatures, filtering out generic-only tests for helpers
infer_all_test_cases(Signatures, TestConfigs) :-
    findall(Config,
            (   member(Sig, Signatures),
                (   is_list(Sig) ->
                    % Multi-function script - generate configs for each function
                    member(FuncSig, Sig),
                    infer_test_cases(FuncSig, Tests, TestType),
                    (TestType = specific -> Config = config(FuncSig, Tests) ; fail) % Only include specific tests
                ;   % Single function
                    infer_test_cases(Sig, Tests, TestType),
                    (TestType = specific -> Config = config(Sig, Tests) ; fail) % Only include specific tests
                )
            ),
            TestConfigs).

%% infer_test_cases(+Signature, -TestCases, -TestType)
%  Infer appropriate test cases based on signature and pattern type
%  TestType will be 'specific' or 'generic'

% Rule 1: Arity 2, Linear recursion, List-related predicate
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _, _)),
                 TestCases, specific) :-
    (sub_atom(Name, _, _, _, length) ; sub_atom(Name, _, _, _, list)), !,
    TestCases = [
        test('Empty list', ['[]', '']),
        test('Single element list', ['[a]', '']),
        test('Three element list', ['[a,b,c]', ''])
    ].

% Rule 2: Arity 2, Linear recursion, Numeric predicate
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _, _)),
                 TestCases, specific) :-
    member(Name, [factorial, fib, power]), !,
    TestCases = [
        test('Base case 0', ['0', '']),
        test('Base case 1', ['1', '']),
        test('Larger value', ['5', ''])
    ].

% Rule 3: Arity 3, Tail recursive or accumulator pattern
infer_test_cases(function(Name, 3, metadata(pattern_type(PatternType), _, _)),
                 TestCases, specific) :-
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

% Rule 4: Arity 1, Mutual recursion (even/odd pattern)
infer_test_cases(function(Name, 1, _),
                 TestCases, specific) :-
    (sub_atom(Name, 0, _, _, is_even) ; sub_atom(Name, 0, _, _, even)), !,
    TestCases = [
        test('Even: 0', ['0']),
        test('Even: 4', ['4']),
        test('Odd (should fail): 3', ['3'])
    ].

infer_test_cases(function(Name, 1, _),
                 TestCases, specific) :-
    (sub_atom(Name, 0, _, _, is_odd) ; sub_atom(Name, 0, _, _, odd)), !,
    TestCases = [
        test('Odd: 3', ['3']),
        test('Odd: 5', ['5']),
        test('Even (should fail): 6', ['6'])
    ].

% Rule 5: Arity 1, Tree recursion pattern (tree_sum, tree_height, etc.)
infer_test_cases(function(Name, 1, metadata(pattern_type(tree_recursive), _, _)),
                 TestCases, specific) :-
    (sub_atom(Name, _, _, _, tree) ; sub_atom(Name, _, _, _, binary)), !,
    TestCases = [
        test('Empty tree', ['[]']),
        test('Single node', ['[5,[],[]]']),
        test('Small tree', ['[10,[5,[],[3,[],[]]],[7,[],[]]]'])
    ].

% Rule 6: Arity 1, Any tree-related function (inferred from name)
infer_test_cases(function(Name, 1, _),
                 TestCases, specific) :-
    (sub_atom(Name, _, _, _, tree) ; sub_atom(Name, _, _, _, binary)), !,
    TestCases = [
        test('Empty tree', ['[]']),
        test('Single node', ['[5,[],[]]']),
        test('Small tree', ['[10,[5,[],[3,[],[]]],[7,[],[]]]'])
    ].

% Rule 7: Transitive closure patterns (ancestor, reachable, etc.)
infer_test_cases(function(Name, Arity, metadata(pattern_type(PatternType), _, _)),
                 TestCases, specific) :-
    member(PatternType, [transitive_closure, unknown]),
    member(Arity, [1, 2]),
    (sub_atom(Name, _, _, _, ancestor) ; sub_atom(Name, _, _, _, reachable) ; sub_atom(Name, _, _, _, descendant)), !,
    (   Arity = 1 ->
        TestCases = [
            test('Find all descendants of abraham', ['abraham']),
            test('Find all descendants of isaac', ['isaac']),
            test('Find all descendants of jacob', ['jacob'])
        ]
    ;   % Arity = 2
        TestCases = [
            test('Check isaac is ancestor of judah', ['isaac', 'judah']),
            test('Check sarah is ancestor of esau', ['sarah', 'esau']),
            test('Check ishmael is NOT ancestor of jacob (should fail)', ['ishmael', 'jacob'])
        ]
    ).

% Fallback: Generic test cases based on arity
infer_test_cases(function(_Name, Arity, _), TestCases, generic) :-
    length(Args, Arity),
    maplist(=('test_value'), Args),
    TestCases = [test('Generic test', Args)].

%% generate_runner(+TestConfigs, +Mode, +OutputPath)
%  Generate test runner script based on mode
generate_runner(TestConfigs, Mode, OutputPath) :-
    (   Mode = concise ->
        generate_concise_runner(TestConfigs, OutputPath)
    ;   Mode = explicit ->
        generate_explicit_runner(TestConfigs, OutputPath)
    ;   Mode = hybrid ->
        generate_hybrid_runner(TestConfigs, OutputPath)
    ;   format('Warning: Unknown mode ~w, using explicit~n', [Mode]),
        generate_explicit_runner(TestConfigs, OutputPath)
    ).

%% generate_explicit_runner(+TestConfigs, +OutputPath)
%  Generate explicit test runner (one test per block)
%  Groups functions by file to avoid duplicate sourcing
generate_explicit_runner(TestConfigs, OutputPath) :-
    open(OutputPath, write, Stream),
    write_header(Stream, explicit),

    % Group test configs by file
    group_configs_by_file(TestConfigs, GroupedConfigs),

    % Preload all discovered scripts so shared dependencies are available
    extract_group_file_names(GroupedConfigs, FileNamesUnsorted),
    sort(FileNamesUnsorted, FileNames),
    write_global_source_block(Stream, FileNames),

    % Generate tests for each file (with all its functions)
    forall(member(file_group(FilePath, FunctionConfigs), GroupedConfigs),
           write_file_tests(Stream, FilePath, FunctionConfigs, FileNames)),

    write_footer(Stream),
    close(Stream).

%% group_configs_by_file(+TestConfigs, -GroupedConfigs)
%  Group test configurations by file path to avoid duplicate sourcing
%  Returns: list of file_group(FilePath, [config(...), config(...)])
group_configs_by_file(TestConfigs, GroupedConfigs) :-
    % Extract all unique file paths
    findall(FilePath,
            (   member(config(function(_, _, metadata(_, _, file_path(FilePath))), _), TestConfigs)
            ),
            AllPaths),
    list_to_set(AllPaths, UniquePaths),

    % For each unique path, collect all configs for that file
    findall(file_group(FilePath, Configs),
            (   member(FilePath, UniquePaths),
                findall(config(function(Name, Arity, Metadata), Tests),
                        (   member(config(function(Name, Arity, Metadata), Tests), TestConfigs),
                            Metadata = metadata(_, _, file_path(FilePath))
                        ),
                        Configs),
                Configs \= []
            ),
            GroupedConfigs).

%% extract_group_file_names(+GroupedConfigs, -FileNames)
extract_group_file_names(GroupedConfigs, FileNames) :-
    findall(FileName,
            (   member(file_group(FilePath, _), GroupedConfigs),
                file_base_name(FilePath, FileName)
            ),
            Names),
    list_to_set(Names, FileNames).

%% write_global_source_block(+Stream, +FileNames)
write_global_source_block(_, []) :- !.
write_global_source_block(Stream, FileNames) :-
    format(Stream, '# Preloading all discovered scripts to satisfy dependencies~n', []),
    forall(member(FileName, FileNames),
           (   format(Stream, 'if [[ -f "$SCRIPT_DIR/~w" ]]; then~n', [FileName]),
               format(Stream, '    source "$SCRIPT_DIR/~w"~n', [FileName]),
               format(Stream, 'fi~n', []),
               format(Stream, '~n', [])
           )),
    format(Stream, '~n', []).

%% extract_script_dependencies(+FilePath, -Dependencies)
extract_script_dependencies(FilePath, Dependencies) :-
    catch(read_file_to_string(FilePath, Content, []), _, Content = ""),
    re_foldl(collect_dependency_path,
             "^[ \\t]*(?:source|\\.)\\s+['\"]?(?<dep>[^'\"\\s;]+)['\"]?",
             Content,
             [],
             RawDeps,
             [multiline(true), capture_type(atom)]),
    maplist(normalize_dependency_path, RawDeps, NormalizedDeps),
    list_to_set(NormalizedDeps, Dependencies).

%% collect_dependency_path(+Match, +AccIn, -AccOut)
collect_dependency_path(Match, AccIn, [Dep|AccIn]) :-
    get_dict(dep, Match, Dep).

%% normalize_dependency_path(+Raw, -Normalized)
normalize_dependency_path(Raw, Normalized) :-
    (   atom(Raw) ->
        DepAtom = Raw
    ;   string(Raw) ->
        atom_string(DepAtom, Raw)
    ;   term_to_atom(Raw, DepAtom)
    ),
    file_base_name(DepAtom, BaseName),
    Normalized = BaseName.

%% exclude_self_dependency(+Dependencies, +FileName, -Filtered)
exclude_self_dependency([], _, []).
exclude_self_dependency([FileName|Rest], FileName, FilteredRest) :-
    !,
    exclude_self_dependency(Rest, FileName, FilteredRest).
exclude_self_dependency([Dep|Rest], FileName, [Dep|FilteredRest]) :-
    exclude_self_dependency(Rest, FileName, FilteredRest).

%% write_dependency_sources(+Stream, +Dependencies)
write_dependency_sources(Stream, Dependencies) :-
    forall(member(Dep, Dependencies),
           (   format(Stream, '    if [[ -f ~w ]]; then~n', [Dep]),
               format(Stream, '        source ~w~n', [Dep]),
               format(Stream, '    fi~n', [])
           )),
    (   Dependencies \= [] ->
        format(Stream, '~n', [])
    ;   true
    ).

%% write_file_tests(+Stream, +FilePath, +FunctionConfigs)
write_file_tests(Stream, FilePath, FunctionConfigs) :-
    write_file_tests(Stream, FilePath, FunctionConfigs, []).

%% write_file_tests(+Stream, +FilePath, +FunctionConfigs, +AllScripts)
%  Write tests for all functions in a file (source all other scripts first, then target last)
write_file_tests(Stream, FilePath, FunctionConfigs, _AllScripts) :- % AllScripts is no longer used directly for sourcing
    file_base_name(FilePath, FileName),
    
    % Extract actual dependencies for this script
    extract_script_dependencies(FilePath, RawDependencies),
    % Filter out self-dependency and ensure unique base names
    file_base_name(FilePath, SelfFileName),
    exclude_self_dependency(RawDependencies, SelfFileName, FilteredDependencies),
    list_to_set(FilteredDependencies, UniqueDependencies),

    format(Stream, '# Test ~w', [FileName]),

    % Add annotation if multiple functions (mutual recursion)
    (   length(FunctionConfigs, Count),
        Count > 1 ->
        format(Stream, ' (multi-function: ~w functions)~n', [Count])
    ;   format(Stream, '~n', [])
    ),

    format(Stream, 'if [[ -f "$SCRIPT_DIR/~w" ]]; then~n', [FileName]),
    format(Stream, '    echo "--- Testing ~w ---"~n', [FileName]),

    % Source actual dependencies
    (   UniqueDependencies \= [] ->
        forall(member(Dep, UniqueDependencies),
               (   format(Stream, '    if [[ -f "$SCRIPT_DIR/~w" ]]; then~n', [Dep]),
                   format(Stream, '        source "$SCRIPT_DIR/~w"~n', [Dep]),
                   format(Stream, '    fi~n', [])
               ))
    ;   true
    ),

    % Source target script last so it takes precedence
    format(Stream, '    source "$SCRIPT_DIR/~w"~n', [FileName]),
    format(Stream, '~n', []),

    % Write tests for each function in this file
    forall(member(config(function(FuncName, _, _), Tests), FunctionConfigs),
           (   length(FunctionConfigs, FuncCount),
               (   FuncCount > 1 ->
                   format(Stream, '    # Function: ~w~n', [FuncName])
               ;   true
               ),
               write_test_list(Stream, FuncName, Tests, 1)
           )),

    format(Stream, 'fi~n', []),
    format(Stream, '~n', []).

%% write_explicit_tests(+Stream, +FuncName, +Metadata, +Tests)
write_explicit_tests(Stream, FuncName, Metadata, Tests) :-
    Metadata = metadata(_, _, file_path(FilePath)),
    file_base_name(FilePath, FileName),

    format(Stream, '# Test ~w~n', [FileName]),
    format(Stream, 'if [[ -f ~w ]]; then~n', [FileName]),
    format(Stream, '    echo "--- Testing ~w ---"~n', [FileName]),
    format(Stream, '    source ~w~n', [FileName]),
    format(Stream, '~n', []),

    % Write each test
    write_test_list(Stream, FuncName, Tests, 1),

    format(Stream, 'fi~n', []),
    format(Stream, '~n', []).

%% write_test_list(+Stream, +FuncName, +Tests, +TestNum)
write_test_list(_, _, [], _).
write_test_list(Stream, FuncName, [test(Description, Args)|Rest], Num) :-
    format(Stream, '    echo "Test ~w: ~w"~n', [Num, Description]),

    % Build argument string
    maplist(quote_arg, Args, QuotedArgs),
    atomic_list_concat(QuotedArgs, ' ', ArgsStr),

    % Determine if this test expects failure based on description
    (   sub_atom(Description, _, _, _, 'should fail')
    ->  ExpectFailure = true
    ;   sub_atom(Description, _, _, _, 'NOT')
    ->  ExpectFailure = true
    ;   ExpectFailure = false
    ),

    % Check if function name ends with _check (these return exit codes, not output)
    (   sub_atom(FuncName, _, _, 0, '_check')
    ->  % Function returns exit code - test it directly
        format(Stream, '    if ~w ~w; then~n', [FuncName, ArgsStr]),
        (   ExpectFailure = true
        ->  format(Stream, '        echo "    Result: FAIL (expected failure but succeeded)"~n', [])
        ;   format(Stream, '        echo "    Result: PASS"~n', [])
        ),
        format(Stream, '    else~n', []),
        (   ExpectFailure = true
        ->  format(Stream, '        echo "    Result: PASS (correctly failed)"~n', [])
        ;   format(Stream, '        echo "    Result: FAIL"~n', [])
        ),
        format(Stream, '    fi~n', [])
    ;   % Regular function - capture output and check exit code
        format(Stream, '    output=$( ~w ~w 2>&1 )~n', [FuncName, ArgsStr]),
        format(Stream, '    exit_code=$?~n', []),
        format(Stream, '    if [[ -n "$output" ]]; then~n', []),
        format(Stream, '        echo "$output"~n', []),
        format(Stream, '    fi~n', []),
        format(Stream, '    if [[ $exit_code -eq 0 ]]; then~n', []),
        (   ExpectFailure = true
        ->  format(Stream, '        echo "    Result: FAIL (expected failure but succeeded)"~n', [])
        ;   format(Stream, '        echo "    Result: PASS"~n', [])
        ),
        format(Stream, '    else~n', []),
        (   ExpectFailure = true
        ->  format(Stream, '        echo "    Result: PASS (correctly failed)"~n', [])
        ;   format(Stream, '        echo "    Result: FAIL"~n', [])
        ),
        format(Stream, '    fi~n', [])
    ),
    format(Stream, '~n', []),
    format(Stream, '    echo ""~n', []),

    Num1 is Num + 1,
    write_test_list(Stream, FuncName, Rest, Num1).

%% quote_arg(+Arg, -QuotedArg)
quote_arg(Arg, Quoted) :-
    format(atom(Quoted), '"~w"', [Arg]).

%% generate_concise_runner(+TestConfigs, +OutputPath)
%  Generate concise loop-based test runner
generate_concise_runner(TestConfigs, OutputPath) :-
    open(OutputPath, write, Stream),
    write_header(Stream, concise),

    format(Stream, '# Test configurations (script:function:arity:test_args)~n', []),
    format(Stream, 'declare -A TEST_CONFIGS=(~n', []),

    % Write configuration for each script
    forall(member(config(function(Name, Arity, Metadata), Tests), TestConfigs),
           write_concise_config(Stream, Name, Arity, Metadata, Tests)),

    format(Stream, ')~n~n', []),

    % Write loop-based test executor
    write_concise_executor(Stream),

    write_footer(Stream),
    close(Stream).

%% write_concise_config(+Stream, +FuncName, +Arity, +Metadata, +Tests)
write_concise_config(Stream, FuncName, Arity, Metadata, Tests) :-
    Metadata = metadata(_, _, file_path(FilePath)),
    file_base_name(FilePath, FileName),

    % Format: "script.sh" = "function:arity:arg1,arg2,...:arg1,arg2,..."
    findall(ArgsStr,
            (   member(test(_, Args), Tests),
                atomic_list_concat(Args, ',', ArgsStr)
            ),
            ArgLists),
    atomic_list_concat(ArgLists, ':', AllArgs),

    format(Stream, '    ["~w"]="~w:~w:~w"~n', [FileName, FuncName, Arity, AllArgs]).

%% write_concise_executor(+Stream)
write_concise_executor(Stream) :-
    write(Stream, 'for script in "${!TEST_CONFIGS[@]}"; do\n'),
    write(Stream, '    if [[ -f "$script" ]]; then\n'),
    write(Stream, '        echo "--- Testing $script ---"\n'),
    write(Stream, '        source "$script"\n'),
    write(Stream, '\n'),
    write(Stream, '        IFS=\':\' read -r func arity tests <<< "${TEST_CONFIGS[$script]}"\n'),
    write(Stream, '        IFS=\':\' read -ra TEST_ARRAY <<< "$tests"\n'),
    write(Stream, '\n'),
    write(Stream, '        test_num=1\n'),
    write(Stream, '        for test_args in "${TEST_ARRAY[@]}"; do\n'),
    write(Stream, '            echo "Test $test_num"\n'),
    write(Stream, '            IFS=\',\' read -ra args <<< "$test_args"\n'),
    write(Stream, '            "$func" "${args[@]}"\n'),
    write(Stream, '            echo ""\n'),
    write(Stream, '            ((test_num++))\n'),
    write(Stream, '        done\n'),
    write(Stream, '    fi\n'),
    write(Stream, 'done\n\n').

%% generate_hybrid_runner(+TestConfigs, +OutputPath)
%  Generate hybrid runner (concise for simple, explicit for complex)
generate_hybrid_runner(TestConfigs, OutputPath) :-
    % For now, use explicit mode
    % TODO: Implement smart selection based on complexity
    generate_explicit_runner(TestConfigs, OutputPath).

%% write_header(+Stream, +Mode)
write_header(Stream, Mode) :-
    write(Stream, '#!/bin/bash\n'),
    format(Stream, '# Test runner for generated advanced recursion scripts - ~w MODE~n', [Mode]),
    write(Stream, '# AUTO-GENERATED BY INFERENCE - DO NOT EDIT MANUALLY\n'),
    write(Stream, '#\n'),
    write(Stream, '# Generated by: test_runner_inference.pl\n'),
    write(Stream, '# To regenerate: swipl -g "use_module(unifyweaver(core/advanced/test_runner_inference)), generate_test_runner_inferred, halt."\n'),
    write(Stream, '\n'),
    write(Stream, '# Get absolute path to script directory for robust sourcing\n'),
    write(Stream, 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'),
    write(Stream, '\n'),
    write(Stream, 'echo "=== Testing Generated Bash Scripts ==="\n'),
    write(Stream, 'echo ""\n'),
    write(Stream, '\n').

%% write_footer(+Stream)
write_footer(Stream) :-
    write(Stream, '\n'),
    write(Stream, 'echo "=== All Tests Complete ==="\n').
