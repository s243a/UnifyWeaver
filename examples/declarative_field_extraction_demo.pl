:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% declarative_field_extraction_demo.pl - Full demonstration of field extraction
% Shows declarative field extraction with real Pearltrees data

:- initialization(main, main).

%% Load infrastructure
:- use_module('../src/unifyweaver/sources').
:- use_module('../src/unifyweaver/core/dynamic_source_compiler').

%% ============================================
%% DATA SOURCE DEFINITIONS
%% ============================================

% Option B: Modular approach (default)
:- source(xml, trees_modular, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title',
        privacy: 'pt:privacy'
    ]),
    field_compiler(modular),  % Explicit
    output(dict)
]).

% Option A: Inline approach
:- source(xml, trees_inline, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title',
        privacy: 'pt:privacy'
    ]),
    field_compiler(inline),  % Explicit
    output(dict)
]).

% Default (uses modular)
:- source(xml, trees_default, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ])
]).

% Different output format: List
:- source(xml, trees_list, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ]),
    output(list)
]).

% Different output format: Compound
:- source(xml, trees_compound, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ]),
    output(compound(tree))
]).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% is_physics_tree(+Tree)
%  Check if tree title contains "physics"
is_physics_tree(Tree) :-
    get_dict(title, Tree, Title),
    downcase_atom(Title, Lower),
    sub_atom(Lower, _, _, _, physics).

%% is_public_tree(+Tree)
%  Check if tree is public (privacy = 0)
is_public_tree(Tree) :-
    get_dict(privacy, Tree, Privacy),
    Privacy = '0'.

%% print_tree(+Tree)
%  Pretty print a tree
print_tree(Tree) :-
    format('  Tree ~w: ~w~n', [Tree.id, Tree.title]).

%% compile_and_execute(+SourceName, -Results)
%  Compile a source and execute it
compile_and_execute(SourceName, Results) :-
    % Compile the source
    compile_dynamic_source(SourceName/1, [], BashCode),

    % Save to temp file
    format(atom(ScriptFile), '/tmp/~w_demo.sh', [SourceName]),
    format(atom(OutputFile), '/tmp/~w_output.txt', [SourceName]),

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

%% ============================================
%% DEMONSTRATION TASKS
%% ============================================

demo_1_basic_extraction :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Demo 1: Basic Field Extraction       ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    format('Compiling trees_default source...~n', []),
    compile_and_execute(trees_default, Results),

    format('Extracting physics-related trees...~n~n', []),

    % Filter for physics
    include(contains_physics, Results, PhysicsTrees),

    length(PhysicsTrees, Count),
    format('Found ~w trees with "physics" in title:~n~n', [Count]),

    % Show first 5
    length(First5, 5),
    append(First5, _, PhysicsTrees),
    maplist(writeln, First5).

contains_physics(Line) :-
    downcase_atom(Line, Lower),
    sub_atom(Lower, _, _, _, physics).

demo_2_compare_implementations :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Demo 2: Compare Implementations      ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Modular
    format('Compiling with Option B (modular)...~n', []),
    compile_dynamic_source(trees_modular/1, [], ModularCode),
    atom_length(ModularCode, ModularLen),
    format('  Generated ~w bytes of bash code~n~n', [ModularLen]),

    % Inline
    format('Compiling with Option A (inline)...~n', []),
    compile_dynamic_source(trees_inline/1, [], InlineCode),
    atom_length(InlineCode, InlineLen),
    format('  Generated ~w bytes of bash code~n~n', [InlineLen]),

    % Compare
    format('Comparison:~n', []),
    format('  Modular: ~w bytes~n', [ModularLen]),
    format('  Inline:  ~w bytes~n', [InlineLen]),
    Diff is ModularLen - InlineLen,
    format('  Difference: ~w bytes~n~n', [Diff]).

demo_3_output_formats :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Demo 3: Different Output Formats     ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Dict format
    format('Dict format (default):~n', []),
    compile_and_execute(trees_default, DictResults),
    nth1(1, DictResults, FirstDict),
    format('  ~w~n~n', [FirstDict]),

    % List format
    format('List format:~n', []),
    compile_and_execute(trees_list, ListResults),
    nth1(1, ListResults, FirstList),
    format('  ~w~n~n', [FirstList]),

    % Compound format
    format('Compound format:~n', []),
    compile_and_execute(trees_compound, CompoundResults),
    nth1(1, CompoundResults, FirstCompound),
    format('  ~w~n~n', [FirstCompound]).

demo_4_performance :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Demo 4: Performance Measurement       ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Measure compilation time
    format('Measuring compilation time...~n', []),
    statistics(walltime, [Start|_]),
    compile_dynamic_source(trees_default/1, [], _Code),
    statistics(walltime, [End|_]),
    CompileTime is End - Start,
    format('  Compilation: ~w ms~n~n', [CompileTime]),

    % Measure execution time
    format('Measuring execution time...~n', []),
    statistics(walltime, [ExecStart|_]),
    compile_and_execute(trees_default, Results),
    statistics(walltime, [ExecEnd|_]),
    ExecTime is ExecEnd - ExecStart,
    length(Results, Count),
    format('  Execution: ~w ms~n', [ExecTime]),
    format('  Extracted ~w results~n', [Count]),

    % Calculate throughput
    (   Count > 0, ExecTime > 0
    ->  Throughput is Count / (ExecTime / 1000),
        format('  Throughput: ~2f items/second~n~n', [Throughput])
    ;   format('  (Throughput calculation skipped)~n~n', [])
    ).

%% ============================================
%% MAIN PROGRAM
%% ============================================

main :-
    format('~n', []),
    format('╔══════════════════════════════════════════════════╗~n', []),
    format('║                                                  ║~n', []),
    format('║  Declarative Field Extraction Demo              ║~n', []),
    format('║  Pearltrees Data Example                        ║~n', []),
    format('║                                                  ║~n', []),
    format('╚══════════════════════════════════════════════════╝~n', []),

    % Check if data file exists
    (   exists_file('../context/PT/pearltrees_export.rdf')
    ->  format('~n✓ Data file found: ../context/PT/pearltrees_export.rdf~n', [])
    ;   format('~n✗ Data file not found: ../context/PT/pearltrees_export.rdf~n', []),
        format('  Please ensure the file exists before running this demo.~n', []),
        halt(1)
    ),

    % Run demos
    catch(
        (
            demo_1_basic_extraction,
            demo_2_compare_implementations,
            demo_3_output_formats,
            demo_4_performance
        ),
        Error,
        (
            format('~nError during demo: ~w~n', [Error]),
            halt(1)
        )
    ),

    % Summary
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Demo Complete!                        ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    format('Key Takeaways:~n', []),
    format('  1. Declarative field extraction is simple and fast~n', []),
    format('  2. Both modular and inline approaches work identically~n', []),
    format('  3. Multiple output formats available (dict, list, compound)~n', []),
    format('  4. Streaming approach uses constant memory~n', []),
    format('  5. Perfect for large XML files on mobile devices~n~n', []),

    format('Try modifying the source definitions and re-running!~n~n', []),

    halt(0).
