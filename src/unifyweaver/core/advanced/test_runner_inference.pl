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
    infer_test_cases/2
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
generate_test_runner_inferred(OutputPath, Options) :-
    option(mode(Mode), Options, explicit),

    % Scan output directory for scripts
    scan_output_directory(Scripts),

    % Extract signatures from all scripts
    extract_all_signatures(Scripts, Signatures),

    % Infer test cases for each signature
    infer_all_test_cases(Signatures, TestConfigs),

    % Generate test runner based on mode
    generate_runner(TestConfigs, Mode, OutputPath),

    format('Generated test runner (inferred, ~w mode): ~w~n', [Mode, OutputPath]).

%% scan_output_directory(-Scripts)
%  Find all .sh files in output/advanced/ directory
%  Filters out demo scripts, test wrappers, and duplicates
scan_output_directory(Scripts) :-
    OutputDir = 'output/advanced',
    (   exists_directory(OutputDir) ->
        expand_file_name('output/advanced/*.sh', AllFiles),
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
%  Returns: function(Name, Arity, Metadata)
extract_function_signature(FilePath, function(Name, Arity, Metadata)) :-
    read_file_to_string(FilePath, Content, []),

    % Extract function name from file header comment
    % Example: "# list_length - linear recursive pattern with memoization"
    (   re_matchsub("^#\\s*(?<name>\\w+)\\s*-\\s*(?<desc>[^\\n]+)",
                     Content, Match, [multiline(true)]) ->
        get_dict(name, Match, NameAtom),
        atom_string(NameFromHeader, NameAtom),
        get_dict(desc, Match, DescStr),
        atom_string(Description, DescStr)
    ;   % Fallback: extract from filename
        file_base_name(FilePath, FileName),
        file_name_extension(NameFromHeader, _, FileName),
        Description = unknown
    ),

    % Verify the name matches an actual function, otherwise extract first function
    (   function_exists_in_content(Content, NameFromHeader) ->
        Name = NameFromHeader
    ;   % Extract first actual function name
        extract_first_function_name(Content, Name)
    ),

    % Extract arity by counting function parameters
    % Pattern: local <name>="$N" where N is a number
    re_foldl(count_match, "local\\s+\\w+=\"\\$\\d+\"", Content, 0, Arity, []),

    % Extract pattern type from description
    extract_pattern_type(Description, PatternType),

    Metadata = metadata(
        pattern_type(PatternType),
        description(Description),
        file_path(FilePath)
    ).

%% count_match(+Match, +CountIn, -CountOut)
%  Helper for re_foldl to count matches
count_match(_, CountIn, CountOut) :-
    CountOut is CountIn + 1.

%% function_exists_in_content(+Content, +FuncName)
%  Check if a function with the given name exists in the script
function_exists_in_content(Content, FuncName) :-
    atom(FuncName),
    format(atom(Pattern), "^~w\\(\\)\\s*\\{", [FuncName]),
    re_match(Pattern, Content, [multiline(true)]).

%% extract_first_function_name(+Content, -FuncName)
%  Extract the first function name from script content
extract_first_function_name(Content, FuncName) :-
    re_matchsub("^(\\w+)\\(\\)\\s*\\{", Content, Match, [multiline(true)]),
    get_dict(1, Match, FuncNameStr),
    atom_string(FuncName, FuncNameStr).

%% extract_pattern_type(+Description, -PatternType)
%  Infer pattern type from description string
extract_pattern_type(Desc, tail_recursive) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'tail recursive'), !.
extract_pattern_type(Desc, linear_recursive) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'linear recursive'), !.
extract_pattern_type(Desc, mutual_recursive) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'mutual'), !.
extract_pattern_type(Desc, accumulator) :-
    atom(Desc),
    sub_atom(Desc, _, _, _, 'accumulator'), !.
extract_pattern_type(_, unknown).

%% infer_all_test_cases(+Signatures, -TestConfigs)
%  Infer test cases for all signatures
infer_all_test_cases(Signatures, TestConfigs) :-
    findall(config(Sig, Tests),
            (   member(Sig, Signatures),
                infer_test_cases(Sig, Tests)
            ),
            TestConfigs).

%% infer_test_cases(+Signature, -TestCases)
%  Infer appropriate test cases based on signature and pattern type

% Rule 1: Arity 2, Linear recursion, List-related predicate
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _, _)),
                 TestCases) :-
    (sub_atom(Name, _, _, _, length) ; sub_atom(Name, _, _, _, list)), !,
    TestCases = [
        test('Empty list', ['[]', '']),
        test('Single element list', ['[a]', '']),
        test('Three element list', ['[a,b,c]', ''])
    ].

% Rule 2: Arity 2, Linear recursion, Numeric predicate
infer_test_cases(function(Name, 2, metadata(pattern_type(linear_recursive), _, _)),
                 TestCases) :-
    member(Name, [factorial, fib, power]), !,
    TestCases = [
        test('Base case 0', ['0', '']),
        test('Base case 1', ['1', '']),
        test('Larger value', ['5', ''])
    ].

% Rule 3: Arity 3, Tail recursive or accumulator pattern
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

% Rule 4: Arity 1, Mutual recursion (even/odd pattern)
infer_test_cases(function(Name, 1, metadata(pattern_type(mutual_recursive), _, _)),
                 TestCases) :-
    (sub_atom(Name, 0, _, _, is_even) ; sub_atom(Name, 0, _, _, even)), !,
    TestCases = [
        test('Even: 0', ['0']),
        test('Even: 4', ['4']),
        test('Odd (should fail): 3', ['3'])
    ].

infer_test_cases(function(Name, 1, metadata(pattern_type(mutual_recursive), _, _)),
                 TestCases) :-
    (sub_atom(Name, 0, _, _, is_odd) ; sub_atom(Name, 0, _, _, odd)), !,
    TestCases = [
        test('Odd: 3', ['3']),
        test('Odd: 5', ['5']),
        test('Even (should fail): 6', ['6'])
    ].

% Fallback: Generic test cases based on arity
infer_test_cases(function(_Name, Arity, _), TestCases) :-
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
generate_explicit_runner(TestConfigs, OutputPath) :-
    open(OutputPath, write, Stream),
    write_header(Stream, explicit),

    % Generate tests for each script
    forall(member(config(function(Name, _Arity, Metadata), Tests), TestConfigs),
           write_explicit_tests(Stream, Name, Metadata, Tests)),

    write_footer(Stream),
    close(Stream).

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

    format(Stream, '    ~w ~w~n', [FuncName, ArgsStr]),
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
    write(Stream, 'echo "=== Testing Generated Bash Scripts ==="\n'),
    write(Stream, 'echo ""\n'),
    write(Stream, '\n').

%% write_footer(+Stream)
write_footer(Stream) :-
    write(Stream, '\n'),
    write(Stream, 'echo "=== All Tests Complete ==="\n').
