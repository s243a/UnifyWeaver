:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_all_strategies.pl - Test all three field extraction strategies
% Compares modular (AWK), inline (AWK), and prolog (SGML) implementations

:- initialization(main, main).

%% Load infrastructure
:- use_module('../src/unifyweaver/sources').
:- use_module('../src/unifyweaver/core/dynamic_source_compiler').
:- use_module('../src/unifyweaver/sources/xml_source').

%% ============================================
%% TEST DATA SOURCES
%% ============================================

% Strategy 1: Modular (Option B - AWK via xml_field_compiler)
:- source(xml, trees_modular, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ]),
    field_compiler(modular),
    output(dict)
]).

% Strategy 2: Inline (Option A - AWK in xml_source.pl)
:- source(xml, trees_inline, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ]),
    field_compiler(inline),
    output(dict)
]).

% Strategy 3: Pure Prolog (Option C - library(sgml))
:- source(xml, trees_prolog, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ]),
    field_compiler(prolog),
    output(dict)
]).

%% ============================================
%% TEST PREDICATES
%% ============================================

%% compile_and_execute(+SourceName, -Results)
compile_and_execute(SourceName, Results) :-
    % Compile the source
    compile_dynamic_source(SourceName/1, [], BashCode),

    % Get temp directory
    (   getenv('TMPDIR', TmpDir)
    ->  true
    ;   TmpDir = '/data/data/com.termux/files/usr/tmp'
    ),

    % Save to temp file
    format(atom(ScriptFile), '~w/~w_test.sh', [TmpDir, SourceName]),
    format(atom(OutputFile), '~w/~w_output.txt', [TmpDir, SourceName]),

    setup_call_cleanup(
        open(ScriptFile, write, Stream),
        write(Stream, BashCode),
        close(Stream)
    ),

    % Execute
    format(atom(Command), 'bash ~w > ~w 2>&1', [ScriptFile, OutputFile]),
    shell(Command),

    % Read results
    read_file_to_string(OutputFile, ResultsText, []),
    split_string(ResultsText, "\n", "\n\0", Lines),
    delete(Lines, "", Results).

%% test_strategy(+Name, +SourceName)
test_strategy(Name, SourceName) :-
    format('~nTesting ~w strategy...~n', [Name]),
    format('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━~n', []),

    % Measure compilation time
    statistics(walltime, [CompStart|_]),
    compile_dynamic_source(SourceName/1, [], BashCode),
    statistics(walltime, [CompEnd|_]),
    CompTime is CompEnd - CompStart,

    % Measure code size
    atom_length(BashCode, CodeSize),

    % Measure execution time
    statistics(walltime, [ExecStart|_]),
    compile_and_execute(SourceName, Results),
    statistics(walltime, [ExecEnd|_]),
    ExecTime is ExecEnd - ExecStart,

    % Count results
    length(Results, Count),

    % Show metrics
    format('  Compilation time: ~w ms~n', [CompTime]),
    format('  Generated code:   ~w bytes~n', [CodeSize]),
    format('  Execution time:   ~w ms~n', [ExecTime]),
    format('  Results found:    ~w~n', [Count]),

    % Show first 3 results
    (   Count > 0
    ->  format('~n  First 3 results:~n', []),
        length(First3, 3),
        append(First3, _, Results),
        maplist(show_result, First3)
    ;   format('  (No results found)~n', [])
    ),

    format('~n', []).

show_result(Result) :-
    format('    ~w~n', [Result]).

%% ============================================
%% MAIN PROGRAM
%% ============================================

main :-
    format('~n╔══════════════════════════════════════════════════╗~n', []),
    format('║                                                  ║~n', []),
    format('║  Field Extraction Strategy Comparison           ║~n', []),
    format('║  Testing: Modular, Inline, and Prolog           ║~n', []),
    format('║                                                  ║~n', []),
    format('╚══════════════════════════════════════════════════╝~n', []),

    % Check if data file exists
    (   exists_file('../context/PT/pearltrees_export.rdf')
    ->  format('~n✓ Data file found~n', [])
    ;   format('~n✗ Data file not found: ../context/PT/pearltrees_export.rdf~n', []),
        format('  Please ensure the file exists before running this test.~n', []),
        halt(1)
    ),

    % Run tests
    catch(
        (
            test_strategy('Modular (AWK)', trees_modular),
            test_strategy('Inline (AWK)', trees_inline),
            test_strategy('Prolog (SGML)', trees_prolog)
        ),
        Error,
        (
            format('~nError during test: ~w~n', [Error]),
            halt(1)
        )
    ),

    % Summary
    format('~n╔══════════════════════════════════════════════════╗~n', []),
    format('║  Test Complete!                                  ║~n', []),
    format('╚══════════════════════════════════════════════════╝~n~n', []),

    format('All three strategies have been tested.~n', []),
    format('Compare the metrics above to see differences.~n~n', []),

    halt(0).
