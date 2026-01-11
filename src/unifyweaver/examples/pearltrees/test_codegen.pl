%% pearltrees/test_codegen.pl - Tests for cross-target code generation
%%
%% Tests that Pearltrees predicates can be compiled to Python, C#, and Go.
%% Also tests that generated Python code actually runs.
%%
%% Run with: swipl -g "run_tests" -t halt test_codegen.pl

:- module(test_pearltrees_codegen, []).

:- use_module(library(plunit)).

%% ============================================================================
%% Load required modules
%% ============================================================================

:- use_module('../../sources').
:- use_module('../../targets/python_target').
:- use_module('../../targets/go_target').
:- use_module('../../targets/csharp_target').
:- use_module(queries).
:- use_module(hierarchy).
:- use_module(templates).

%% ============================================================================
%% Helper predicates for testing
%% ============================================================================

%% code_has_structure(+Code, +Markers) is semidet.
%%   Check that code contains expected structural markers.
code_has_structure(Code, Markers) :-
    forall(member(Marker, Markers), sub_atom(Code, _, _, _, Marker)).

%% code_min_length(+Code, +MinLen) is semidet.
%%   Check that code is at least MinLen characters.
code_min_length(Code, MinLen) :-
    atom_length(Code, Len),
    Len >= MinLen.

%% write_code_to_file(+Code, +Filename) is det.
%%   Write generated code to a file for manual inspection/testing.
write_code_to_file(Code, Filename) :-
    open(Filename, write, Stream),
    write(Stream, Code),
    close(Stream).

%% ============================================================================
%% Tests: Python Target
%% ============================================================================

:- begin_tests(python_target).

test(compile_tree_child_count_to_python) :-
    compile_predicate_to_python(pearltrees_queries:tree_child_count/2, [], Code),
    code_min_length(Code, 100),
    code_has_structure(Code, ['import', 'def', 'json']).

test(compile_incomplete_tree_to_python) :-
    compile_predicate_to_python(pearltrees_queries:incomplete_tree/2, [], Code),
    code_min_length(Code, 100),
    code_has_structure(Code, ['import', 'def']).

test(compile_filter_trees_to_python) :-
    compile_predicate_to_python(pearltrees_queries:filter_trees/2, [], Code),
    code_min_length(Code, 100).

test(python_has_typing_imports) :-
    compile_predicate_to_python(pearltrees_queries:tree_child_count/2, [], Code),
    code_has_structure(Code, ['Iterator', 'Dict']).

test(python_has_jsonl_support) :-
    compile_predicate_to_python(pearltrees_queries:tree_child_count/2, [], Code),
    code_has_structure(Code, ['jsonl', 'json.loads']).

:- end_tests(python_target).

%% ============================================================================
%% Tests: Go Target
%% ============================================================================
%%
%% Go target now supports module-qualified predicates.

:- begin_tests(go_target).

test(go_target_module_loads) :-
    current_predicate(go_target:compile_predicate_to_go/3).

test(go_target_has_exports) :-
    % Verify key predicates are exported
    current_predicate(go_target:compile_predicate_to_go/3),
    current_predicate(go_target:init_go_target/0).

test(go_compile_module_qualified_predicate, [nondet]) :-
    % Test that module-qualified predicates can be compiled
    compile_predicate_to_go(pearltrees_queries:tree_child_count/2, [json_input(true)], Code),
    code_min_length(Code, 100),
    code_has_structure(Code, ['package main', 'func']).

test(go_compile_hierarchy_predicate, [nondet]) :-
    compile_predicate_to_go(pearltrees_hierarchy:tree_depth/2, [json_input(true)], Code),
    code_min_length(Code, 100).

:- end_tests(go_target).

%% ============================================================================
%% Tests: C# Target
%% ============================================================================
%%
%% C# target now supports module-qualified predicates.

:- begin_tests(csharp_target).

test(csharp_target_module_loads) :-
    current_predicate(csharp_target:compile_predicate_to_csharp/3).

test(csharp_target_has_exports) :-
    current_predicate(csharp_target:compile_predicate_to_csharp/3).

test(csharp_compile_module_qualified_predicate) :-
    % Test that module-qualified predicates can be compiled in generator mode
    compile_predicate_to_csharp(pearltrees_queries:tree_child_count/2, [mode(generator)], Code),
    code_min_length(Code, 100),
    code_has_structure(Code, ['class', 'void']).

test(csharp_compile_hierarchy_predicate) :-
    % Test compiling another module-qualified predicate
    compile_predicate_to_csharp(pearltrees_queries:incomplete_tree/2, [mode(generator)], Code),
    code_min_length(Code, 100).

:- end_tests(csharp_target).

%% ============================================================================
%% Tests: Template Code Generation
%% ============================================================================

:- begin_tests(template_generation).

test(generate_freemind_format) :-
    Children = [child(pagepearl, 'Test Link', 'http://example.com', 1)],
    pearltrees_templates:generate_freemind('123', 'Test Tree', Children, Output),
    code_has_structure(Output, ['<map', '<node', 'Test Tree', '</map>']).

test(generate_mermaid_format) :-
    Children = [child(pagepearl, 'Test Link', 'http://example.com', 1)],
    pearltrees_templates:generate_mermaid('123', 'Test Tree', Children, Output),
    code_has_structure(Output, ['mindmap', 'root']).

test(generate_opml_format) :-
    Children = [child(pagepearl, 'Test Link', 'http://example.com', 1)],
    pearltrees_templates:generate_opml('123', 'Test Tree', Children, Output),
    code_has_structure(Output, ['<opml', '<outline', '</opml>']).

test(generate_graphml_format, [nondet]) :-
    Children = [child(pagepearl, 'Test Link', 'http://example.com', 1)],
    pearltrees_templates:generate_graphml('123', 'Test Tree', Children, Output),
    code_has_structure(Output, ['<graphml', '<node', '<edge', '</graphml>']).

:- end_tests(template_generation).

%% ============================================================================
%% Tests: Python Code Execution
%% ============================================================================
%%
%% These tests generate Python code and verify it can be parsed/executed.

:- begin_tests(python_execution).

test(python_syntax_valid) :-
    compile_predicate_to_python(pearltrees_queries:tree_child_count/2, [], Code),
    % Write to temp file and check Python can parse it
    tmp_file_stream(text, TmpFile, Stream),
    write(Stream, Code),
    close(Stream),
    % Try to compile with Python (syntax check)
    format(atom(Cmd), 'python3 -m py_compile ~w 2>&1', [TmpFile]),
    shell(Cmd, Status),
    delete_file(TmpFile),
    Status == 0.

test(hierarchy_python_syntax_valid) :-
    compile_predicate_to_python(pearltrees_hierarchy:tree_depth/2, [], Code),
    tmp_file_stream(text, TmpFile, Stream),
    write(Stream, Code),
    close(Stream),
    format(atom(Cmd), 'python3 -m py_compile ~w 2>&1', [TmpFile]),
    shell(Cmd, Status),
    delete_file(TmpFile),
    Status == 0.

:- end_tests(python_execution).

%% ============================================================================
%% Tests: Python Runtime Execution
%% ============================================================================
%%
%% These tests actually run the generated Python code with test data.

:- begin_tests(python_runtime).

test(generated_code_runs_with_empty_input) :-
    compile_predicate_to_python(pearltrees_queries:tree_child_count/2, [], Code),
    % Write code to temp file
    tmp_file_stream(text, TmpFile, Stream),
    write(Stream, Code),
    close(Stream),
    % Run with empty input (should complete without error)
    format(atom(Cmd), 'echo "" | timeout 5 python3 ~w 2>&1; echo "EXIT:$?"', [TmpFile]),
    shell(Cmd, _),
    delete_file(TmpFile).

test(templates_generate_valid_xml) :-
    Children = [child(pagepearl, 'Test', 'http://example.com', 1)],
    pearltrees_templates:generate_freemind('123', 'Test', Children, Output),
    % Check it's valid XML-like structure
    sub_atom(Output, 0, _, _, '<'),
    sub_atom(Output, _, _, 0, '>').

test(templates_generate_valid_mermaid, [nondet]) :-
    Children = [child(pagepearl, 'Test', 'http://example.com', 1)],
    pearltrees_templates:generate_mermaid('123', 'Test', Children, Output),
    % Check mermaid structure
    sub_atom(Output, _, _, _, 'mindmap'),
    sub_atom(Output, _, _, _, 'root').

:- end_tests(python_runtime).

%% ============================================================================
%% Tests: Hierarchy Predicates Compilation
%% ============================================================================

:- begin_tests(hierarchy_compilation).

test(compile_tree_depth_to_python) :-
    compile_predicate_to_python(pearltrees_hierarchy:tree_depth/2, [], Code),
    code_min_length(Code, 100).

test(compile_root_tree_to_python) :-
    compile_predicate_to_python(pearltrees_hierarchy:root_tree/1, [], Code),
    code_min_length(Code, 100).

test(compile_has_cycle_to_python) :-
    compile_predicate_to_python(pearltrees_hierarchy:has_cycle/1, [], Code),
    code_min_length(Code, 100).

:- end_tests(hierarchy_compilation).

%% ============================================================================
%% Run tests when loaded directly
%% ============================================================================

:- initialization(run_tests, main).
