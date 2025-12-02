/**
 * C# Generator Target Tests (dependency groups) using Janus Bridge
 *
 * Compiles generator-mode C# for a dependency group and runs the produced
 * program via the Python helper. Requires dotnet CLI, Python, and Janus.
 */

:- module(test_csharp_generator_janus, [
    test_csharp_generator_dep_group/0
]).

:- use_module(library(plunit)).
:- use_module(library(janus)).

:- dynamic python_helper_ready/0.
%% test_file_dir(-Dir)
%  Directory of this test file.
test_file_dir(Dir) :-
    absolute_file_name('tests/core', Dir, [file_type(directory)]).

%% setup_paths_and_python/0
%  Ensure library paths and Python helper are available, then load csharp_target.
setup_paths_and_python :-
    test_file_dir(Dir),
    absolute_file_name('../../src/unifyweaver/targets', TargetDir,
                       [relative_to(Dir), file_type(directory)]),
    (   user:file_search_path(library, TargetDir)
    ->  true
    ;   asserta(user:file_search_path(library, TargetDir))
    ),
    absolute_file_name('../../src/unifyweaver/core', CoreDir,
                       [relative_to(Dir), file_type(directory)]),
    (   user:file_search_path(library, CoreDir)
    ->  true
    ;   asserta(user:file_search_path(library, CoreDir))
    ),
    ensure_python_helper(Dir),
    use_module(library(csharp_target)).

ensure_python_helper(_Dir) :-
    python_helper_ready, !.
ensure_python_helper(Dir) :-
    absolute_file_name('../helpers', HelpersDir,
                       [relative_to(Dir), file_type(directory)]),
    py_add_lib_dir(HelpersDir),
    py_call(importlib:import_module('csharp_test_helper'), _),
    asserta(python_helper_ready).

check_dotnet_available :-
    setup_paths_and_python,
    py_call(csharp_test_helper:get_dotnet_version(), Version),
    (   Version = @(none)
    ->  format('[CSharpGeneratorJanusTest] WARNING: dotnet CLI not found~n'),
        fail
    ;   format('[CSharpGeneratorJanusTest] dotnet version: ~w~n', [Version])
    ).

setup_test_data :-
    cleanup_test_data,
    assertz(user:cg_edge(a, b)),
    assertz(user:cg_edge(b, c)),
    assertz(user:cg_edge(b, d)),
    assertz(user:cg_blocked(a, b)),
    assertz(user:(cg_path(X, Y) :- cg_edge(X, Y), \+ cg_blocked(X, Y))),
    assertz(user:(cg_path(X, Y) :- cg_edge(X, Z), \+ cg_blocked(X, Z), cg_path(Z, Y))).

cleanup_test_data :-
    maplist(retractall, [
        user:cg_edge(_, _),
        user:cg_blocked(_, _),
        user:cg_path(_, _)
    ]),
    catch(abolish(user:cg_edge/2), _, true),
    catch(abolish(user:cg_blocked/2), _, true),
    catch(abolish(user:cg_path/2), _, true).

build_program_stub(Stub) :-
    Stub = "\npublic static class Program {\n    public static int Main() {\n        var result = UnifyWeaver.Generated.CgPath_Module.Solve();\n        foreach (var fact in result) {\n            var arg0 = fact.Args.ContainsKey(\"arg0\") ? fact.Args[\"arg0\"] : \"\";\n            var arg1 = fact.Args.ContainsKey(\"arg1\") ? fact.Args[\"arg1\"] : \"\";\n            System.Console.WriteLine($\"{fact.Relation}:{arg0}:{arg1}\");\n        }\n        return 0;\n    }\n}\n".

:- begin_tests(csharp_generator_janus, [
    setup(setup_paths_and_python),
    condition(check_dotnet_available)
]).

test(generator_dependency_group, [
    setup(setup_test_data),
    cleanup(cleanup_test_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(cg_path/2, [mode(generator)], GenCode),
    build_program_stub(Stub),
    atom_concat(GenCode, Stub, FullCode),
    py_call(csharp_test_helper:assert_output_contains(FullCode, "cg_path:b:c", 'csharp_dep_group_pos'),
            Result),
    (   Result.success == @(true),
        Result.assertion_passed == @(true)
    ->  true
    ;   format(user_error, '[CSharpGeneratorJanusTest] Helper result: ~w~n', [Result]),
        assertion(Result.success == @(true)),
        assertion(Result.assertion_passed == @(true))
    ).

test(generator_dependency_group_negation, [
    setup(setup_test_data),
    cleanup(cleanup_test_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(cg_path/2, [mode(generator)], GenCode),
    build_program_stub(Stub),
    atom_concat(GenCode, Stub, FullCode),
    py_call(csharp_test_helper:compile_and_run(FullCode, 'csharp_dep_group_neg'), Result),
    get_dict(success, Result, @(true)),
    get_dict(stdout, Result, Stdout),
    once(sub_string(Stdout, _, _, _, "cg_path:b:c")),
    once(sub_string(Stdout, _, _, _, "cg_path:b:d")),
    \+ sub_string(Stdout, _, _, _, "cg_path:a:b"),
    \+ sub_string(Stdout, _, _, _, "cg_path:a:c").

:- end_tests(csharp_generator_janus).

test_csharp_generator_dep_group :-
    format('~n=== Running C# Generator Dependency-Group Tests ===~n~n'),
    run_tests(csharp_generator_janus).
