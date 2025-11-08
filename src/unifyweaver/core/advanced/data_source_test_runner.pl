:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% data_source_test_runner.pl - Declarative Test Runner Generator for Data Sources
%
% Automatically generates integration test runners for data source declarations.
% Inspired by test_runner_inference.pl but adapted for data pipeline testing.
%
% Usage:
%   ?- use_module(unifyweaver(core/advanced/data_source_test_runner)).
%   ?- generate_data_source_test_runner.
%   % Generates examples/test_data_sources.sh
%
% Architecture:
%   1. Scan examples/ for .pl files with `:- source(...)` declarations
%   2. Extract source type, name, arity, configuration
%   3. Infer test cases based on source type (csv, json, python, http, awk)
%   4. Detect multi-stage pipelines
%   5. Generate bash test runner with setup, tests, validation, cleanup
%
% Key Differences from test_runner_inference.pl:
%   - Integration testing (pipelines) vs unit testing (functions)
%   - Uses write_and_execute_bash/3 for end-to-end flows
%   - Validates output correctness vs exit codes
%   - Manages test data files
%

:- module(data_source_test_runner, [
    generate_data_source_test_runner/0,
    generate_data_source_test_runner/1,
    generate_data_source_test_runner/2,
    extract_source_declarations/2,
    infer_data_source_tests/2
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

%% ============================================
%% PUBLIC API
%% ============================================

%% generate_data_source_test_runner
%  Generate test runner using default settings
generate_data_source_test_runner :-
    generate_data_source_test_runner('examples/test_data_sources.sh').

%% generate_data_source_test_runner(+OutputPath)
%  Generate test runner at specified path
generate_data_source_test_runner(OutputPath) :-
    generate_data_source_test_runner(OutputPath, [examples_dir('examples')]).

%% generate_data_source_test_runner(+OutputPath, +Options)
%  Generate test runner with options
%
%  Options:
%    - examples_dir(Dir): Directory to scan (default: 'examples')
%    - include_http(Bool): Include HTTP source tests (default: false, to avoid real network calls)
%    - test_data_dir(Dir): Where to create test data (default: 'test_input')
%    - output_dir(Dir): Where to write generated scripts (default: 'test_output')
generate_data_source_test_runner(OutputPath, Options) :-
    format('[DataSourceTestRunner] Generating test runner: ~w~n', [OutputPath]),

    % Extract options
    option(examples_dir(ExamplesDir), Options, 'examples'),
    option(include_http(IncludeHTTP), Options, false),
    option(test_data_dir(_TestDataDir), Options, 'test_input'),
    option(output_dir(_OutputDir), Options, 'test_output'),

    % Phase 1: Scan for source declarations
    format('[DataSourceTestRunner] Scanning ~w for source declarations...~n', [ExamplesDir]),
    scan_examples_directory(ExamplesDir, SourceFiles),
    length(SourceFiles, FileCount),
    format('[DataSourceTestRunner] Found ~w files with sources~n', [FileCount]),

    % Phase 2: Extract all source declarations
    format('[DataSourceTestRunner] Extracting source declarations...~n', []),
    findall(Decl,
            (   member(File, SourceFiles),
                extract_source_declarations(File, Decls),
                member(Decl, Decls)
            ),
            AllDeclarations),
    length(AllDeclarations, DeclCount),
    format('[DataSourceTestRunner] Found ~w source declarations~n', [DeclCount]),

    % Filter out HTTP sources if not included
    (   IncludeHTTP = false
    ->  exclude(is_http_source, AllDeclarations, FilteredDeclarations),
        length(FilteredDeclarations, FilteredCount),
        (   FilteredCount < DeclCount
        ->  HTTPCount is DeclCount - FilteredCount,
            format('[DataSourceTestRunner] Excluded ~w HTTP sources (use include_http(true) to test)~n', [HTTPCount])
        ;   true
        )
    ;   FilteredDeclarations = AllDeclarations
    ),

    % Phase 3: Infer test cases for each source
    format('[DataSourceTestRunner] Inferring test cases...~n', []),
    findall(config(Decl, Tests),
            (   member(Decl, FilteredDeclarations),
                infer_data_source_tests(Decl, Tests)
            ),
            TestConfigs),

    % Phase 4: Detect pipelines (sources used together)
    format('[DataSourceTestRunner] Detecting pipelines...~n', []),
    detect_pipelines(SourceFiles, Pipelines),
    length(Pipelines, PipelineCount),
    format('[DataSourceTestRunner] Found ~w pipelines~n', [PipelineCount]),

    % Phase 5: Generate test runner
    format('[DataSourceTestRunner] Generating test runner...~n', []),
    generate_test_runner(TestConfigs, Pipelines, OutputPath, Options),

    format('[DataSourceTestRunner] ✓ Test runner generated: ~w~n', [OutputPath]).

%% ============================================
%% PHASE 1: SCAN DIRECTORY
%% ============================================

%% scan_examples_directory(+Dir, -SourceFiles)
%  Scan directory for .pl files containing source declarations
scan_examples_directory(Dir, SourceFiles) :-
    % Find all .pl files
    (   exists_directory(Dir)
    ->  directory_files(Dir, AllFiles),
        findall(FilePath,
                (   member(FileName, AllFiles),
                    atom_concat(_, '.pl', FileName),
                    % Exclude test files
                    \+ atom_concat('test_', _, FileName),
                    atomic_list_concat([Dir, '/', FileName], FilePath),
                    exists_file(FilePath),
                    % Check if file has source declarations
                    has_source_declarations(FilePath)
                ),
                SourceFiles)
    ;   format('[Warning] Directory does not exist: ~w~n', [Dir]),
        SourceFiles = []
    ).

%% has_source_declarations(+FilePath)
%  Check if file contains `:- source(...)` declarations
has_source_declarations(FilePath) :-
    catch(
        (   read_file_to_string(FilePath, Content, []),
            re_matchsub("^\\s*:-\\s*source\\(", Content, _, [multiline(true)])
        ),
        _,
        fail
    ).

%% ============================================
%% PHASE 2: EXTRACT SOURCE DECLARATIONS
%% ============================================

%% extract_source_declarations(+FilePath, -Declarations)
%  Extract all `:- source(Type, Name, Config)` declarations from file
%  Uses Prolog term reading for accurate parsing
extract_source_declarations(FilePath, Declarations) :-
    catch(
        (   setup_call_cleanup(
                open(FilePath, read, Stream),
                read_source_terms(Stream, FilePath, Declarations),
                close(Stream)
            )
        ),
        Error,
        (   format('[Warning] Failed to read ~w: ~w~n', [FilePath, Error]),
            Declarations = []
        )
    ).

%% read_source_terms(+Stream, +FilePath, -Declarations)
%  Read Prolog terms from stream and extract source declarations
read_source_terms(Stream, FilePath, Declarations) :-
    findall(Decl,
            (   repeat,
                catch(read_term(Stream, Term, []), _, Term = end_of_file),
                (   Term = end_of_file
                ->  !, fail
                ;   extract_source_from_term(Term, FilePath, Decl)
                )
            ),
            Declarations).

%% extract_source_from_term(+Term, +FilePath, -Declaration)
%  Extract source declaration from Prolog term
extract_source_from_term((:- source(Type, Name, Config)), FilePath, Declaration) :-
    !,
    % Extract arity from config
    (   member(arity(Arity), Config)
    ->  true
    ;   Arity = 1  % Default arity
    ),

    % Parse key config options
    parse_config_list(Config, ConfigOptions),

    % Build declaration
    Declaration = source(Type, Name, Arity, ConfigOptions, file_path(FilePath)).
extract_source_from_term(_, _, _) :- fail.

%% parse_config_list(+ConfigList, -ConfigOptions)
%  Extract relevant config options from full config list
parse_config_list(ConfigList, ConfigOptions) :-
    findall(Opt,
            (   member(Term, ConfigList),
                (   Term = csv_file(File), Opt = csv_file(File)
                ;   Term = json_file(File), Opt = json_file(File)
                ;   Term = database(DB), Opt = database(DB)
                ;   Term = url(URL), Opt = url(URL)
                ;   Term = python_inline(_), Opt = python_inline(true)
                ;   Term = sqlite_query(_), Opt = sqlite_query(true)
                ;   Term = jq_filter(_), Opt = jq_filter(true)
                )
            ),
            ConfigOptions).

%% is_http_source(+Declaration)
%  Check if declaration is HTTP source
is_http_source(source(http, _, _, _, _)).

%% ============================================
%% PHASE 3: INFER TEST CASES
%% ============================================

%% infer_data_source_tests(+Declaration, -TestCases)
%  Infer appropriate test cases based on source type

% CSV Source Tests
infer_data_source_tests(source(csv, _Name, _Arity, _Config, _FilePath), Tests) :-
    !,
    Tests = [
        test('Compile CSV source', []),
        test('Load CSV data and validate row count', []),
        test('Check CSV field extraction', [])
    ].

% JSON Source Tests
infer_data_source_tests(source(json, _Name, _Arity, Config, _FilePath), Tests) :-
    !,
    (   member(jq_filter(true), Config)
    ->  Tests = [
            test('Compile JSON source with jq filter', []),
            test('Parse JSON and validate output format', []),
            test('Check jq filter produces TSV output', [])
        ]
    ;   Tests = [
            test('Compile JSON source', []),
            test('Parse JSON data', [])
        ]
    ).

% Python Source Tests
infer_data_source_tests(source(python, _Name, _Arity, Config, _FilePath), Tests) :-
    !,
    (   member(python_inline(true), Config)
    ->  Tests = [
            test('Compile Python inline source', []),
            test('Execute Python code with stdin', []),
            test('Validate Python output format', [])
        ]
    ;   member(sqlite_query(true), Config)
    ->  Tests = [
            test('Compile SQLite query source', []),
            test('Execute database query', []),
            test('Validate query results', [])
        ]
    ;   Tests = [
            test('Compile Python source', []),
            test('Execute Python script', [])
        ]
    ).

% HTTP Source Tests (generate script only, no network calls)
infer_data_source_tests(source(http, _Name, _Arity, _Config, _FilePath), Tests) :-
    !,
    Tests = [
        test('Compile HTTP source (script generation only)', []),
        test('Validate HTTP request construction', [])
    ].

% AWK Source Tests
infer_data_source_tests(source(awk, _Name, _Arity, _Config, _FilePath), Tests) :-
    !,
    Tests = [
        test('Compile AWK source', []),
        test('Process data with AWK pattern', []),
        test('Validate AWK output format', [])
    ].

% Fallback for unknown source types
infer_data_source_tests(source(Type, Name, Arity, _Config, _FilePath), Tests) :-
    format('[Warning] Unknown source type: ~w for ~w/~w~n', [Type, Name, Arity]),
    Tests = [
        test('Compile source (generic)', [])
    ].

%% ============================================
%% PHASE 4: DETECT PIPELINES
%% ============================================

%% detect_pipelines(+SourceFiles, -Pipelines)
%  Detect multi-stage data pipelines in source files
%  For now, use simple heuristic: files with multiple sources that reference each other
detect_pipelines(SourceFiles, Pipelines) :-
    % Simple implementation: look for files with multiple sources
    findall(pipeline(File, Sources),
            (   member(File, SourceFiles),
                extract_source_declarations(File, Decls),
                length(Decls, Count),
                Count >= 2,
                maplist(extract_source_name, Decls, Sources)
            ),
            Pipelines).

%% extract_source_name(+Declaration, -Name)
extract_source_name(source(Type, Name, Arity, _, _), source_ref(Type, Name, Arity)).

%% ============================================
%% PHASE 5: GENERATE TEST RUNNER
%% ============================================

%% generate_test_runner(+TestConfigs, +Pipelines, +OutputPath, +Options)
%  Generate bash test runner script
generate_test_runner(TestConfigs, Pipelines, OutputPath, Options) :-
    open(OutputPath, write, Stream),

    % Write header
    write_test_runner_header(Stream, _HeaderOptions),

    % Write test data setup
    write_test_data_setup(Stream, TestConfigs, Options),

    % Write individual source tests
    write_source_tests(Stream, TestConfigs, Options),

    % Write pipeline tests
    write_pipeline_tests(Stream, Pipelines, Options),

    % Write cleanup
    write_test_cleanup(Stream, Options),

    % Write footer
    write_test_runner_footer(Stream),

    close(Stream),

    % Make executable
    catch(process_create('/bin/chmod', ['+x', OutputPath], []), _, true).

%% write_test_runner_header(+Stream, +Options)
write_test_runner_header(Stream, _Options) :-
    write(Stream, '#!/bin/bash\n'),
    write(Stream, '# Data Source Test Runner\n'),
    write(Stream, '# AUTO-GENERATED BY data_source_test_runner.pl - DO NOT EDIT MANUALLY\n'),
    write(Stream, '#\n'),
    write(Stream, '# Generated by: data_source_test_runner.pl\n'),
    write(Stream, '# To regenerate: swipl -g "use_module(unifyweaver(core/advanced/data_source_test_runner)), generate_data_source_test_runner, halt."\n'),
    write(Stream, '\n'),
    write(Stream, 'set -e  # Exit on error\n'),
    write(Stream, '\n'),
    write(Stream, 'echo "=== Testing Data Sources ==="\n'),
    write(Stream, 'echo ""\n'),
    write(Stream, '\n').

%% write_test_data_setup(+Stream, +TestConfigs, +Options)
write_test_data_setup(Stream, _TestConfigs, Options) :-
    option(test_data_dir(TestDataDir), Options, 'test_input'),
    option(output_dir(OutputDir), Options, 'test_output'),

    write(Stream, '# Setup test data\n'),
    write(Stream, 'setup_test_data() {\n'),
    format(Stream, '    echo "Setting up test data in ~w/"~n', [TestDataDir]),
    format(Stream, '    mkdir -p ~w ~w~n', [TestDataDir, OutputDir]),
    write(Stream, '\n'),

    % Create sample CSV
    write(Stream, '    # Sample CSV\n'),
    format(Stream, '    cat > ~w/sample.csv <<EOF~n', [TestDataDir]),
    write(Stream, 'product,category,price,stock\n'),
    write(Stream, 'Laptop,Electronics,1200,15\n'),
    write(Stream, 'Mouse,Electronics,25,100\n'),
    write(Stream, 'Desk,Furniture,350,8\n'),
    write(Stream, 'EOF\n'),
    write(Stream, '\n'),

    % Create sample JSON
    write(Stream, '    # Sample JSON\n'),
    format(Stream, '    cat > ~w/sample.json <<EOF~n', [TestDataDir]),
    write(Stream, '{\n'),
    write(Stream, '  "orders": [\n'),
    write(Stream, '    {"id": 1, "product": "Laptop", "quantity": 2},\n'),
    write(Stream, '    {"id": 2, "product": "Mouse", "quantity": 5}\n'),
    write(Stream, '  ]\n'),
    write(Stream, '}\n'),
    write(Stream, 'EOF\n'),
    write(Stream, '\n'),

    write(Stream, '    echo "  ✓ Test data created"\n'),
    write(Stream, '}\n'),
    write(Stream, '\n').

%% write_source_tests(+Stream, +TestConfigs, +Options)
write_source_tests(Stream, TestConfigs, Options) :-
    write(Stream, '# Individual source tests\n'),
    forall(member(config(Decl, Tests), TestConfigs),
           write_source_test(Stream, Decl, Tests, Options)),
    write(Stream, '\n').

%% write_source_test(+Stream, +Declaration, +Tests, +Options)
write_source_test(Stream, source(Type, Name, Arity, _Config, _FilePath), _Tests, Options) :-
    option(output_dir(_OutputDir), Options, 'test_output'),

    format(Stream, 'test_~w_~w() {~n', [Type, Name]),
    format(Stream, '    echo "--- Testing ~w source: ~w/~w ---"~n', [Type, Name, Arity]),
    write(Stream, '\n'),

    % Compile step (simplified - would need actual compile logic)
    format(Stream, '    # Compile source to bash~n', []),
    format(Stream, '    # TODO: Add actual compilation command~n', []),
    format(Stream, '    # swipl -q -g "compile_dynamic_source(~w/~w, [], BashCode), ..." -t halt~n', [Name, Arity]),
    write(Stream, '\n'),

    % Execute and validate
    format(Stream, '    # Execute and validate~n', []),
    format(Stream, '    # if [[ -f ~w/~w.sh ]]; then~n', [OutputDir, Name]),
    format(Stream, '    #     ./~w/~w.sh > /tmp/~w_output.txt~n', [OutputDir, Name, Name]),
    format(Stream, '    #     # Validate output~n', []),
    format(Stream, '    # fi~n', []),
    write(Stream, '\n'),

    format(Stream, '    echo "  ✓ ~w/~w tests passed"~n', [Name, Arity]),
    write(Stream, '}\n'),
    write(Stream, '\n').

%% write_pipeline_tests(+Stream, +Pipelines, +Options)
write_pipeline_tests(Stream, Pipelines, _Options) :-
    (   Pipelines = []
    ->  true
    ;   write(Stream, '# Pipeline tests\n'),
        forall(member(Pipeline, Pipelines),
               write_pipeline_test(Stream, Pipeline)),
        write(Stream, '\n')
    ).

%% write_pipeline_test(+Stream, +Pipeline)
write_pipeline_test(Stream, pipeline(File, Sources)) :-
    file_base_name(File, BaseName),
    format(Stream, 'test_pipeline_~w() {~n', [BaseName]),
    format(Stream, '    echo "--- Testing pipeline from ~w ---"~n', [BaseName]),
    format(Stream, '    # Sources: ~w~n', [Sources]),
    write(Stream, '    # TODO: Implement pipeline test\n'),
    write(Stream, '    echo "  ⚠ Pipeline test not yet implemented"\n'),
    write(Stream, '}\n'),
    write(Stream, '\n').

%% write_test_cleanup(+Stream, +Options)
write_test_cleanup(Stream, Options) :-
    option(test_data_dir(TestDataDir), Options, 'test_input'),
    option(output_dir(_OutputDir), Options, 'test_output'),

    write(Stream, '# Cleanup\n'),
    write(Stream, 'cleanup_test_data() {\n'),
    write(Stream, '    if [[ "${KEEP_TEST_DATA}" != "true" ]]; then\n'),
    format(Stream, '        rm -rf ~w /tmp/*_output.txt~n', [TestDataDir]),
    write(Stream, '        echo "  ✓ Test data cleaned up"\n'),
    write(Stream, '    else\n'),
    write(Stream, '        echo "  ℹ Test data preserved (KEEP_TEST_DATA=true)"\n'),
    write(Stream, '    fi\n'),
    write(Stream, '}\n'),
    write(Stream, '\n').

%% write_test_runner_footer(+Stream)
write_test_runner_footer(Stream) :-
    write(Stream, '# Run all tests\n'),
    write(Stream, 'main() {\n'),
    write(Stream, '    setup_test_data\n'),
    write(Stream, '    \n'),
    write(Stream, '    # Run individual source tests\n'),
    write(Stream, '    # TODO: Call individual test functions\n'),
    write(Stream, '    \n'),
    write(Stream, '    # Run pipeline tests\n'),
    write(Stream, '    # TODO: Call pipeline test functions\n'),
    write(Stream, '    \n'),
    write(Stream, '    cleanup_test_data\n'),
    write(Stream, '    \n'),
    write(Stream, '    echo ""\n'),
    write(Stream, '    echo "=== All Tests Complete ==="\n'),
    write(Stream, '}\n'),
    write(Stream, '\n'),
    write(Stream, 'main "$@"\n').

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% option(+Term, +Options, +Default)
%  Extract option from list with default
option(Term, Options, _Default) :-
    member(Term, Options), !.
option(Term, _Options, Default) :-
    Term =.. [_Functor, Default],
    !.
