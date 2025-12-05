:- module(test_csharp_generator_aggregates_run, [
    test_csharp_generator_aggregate_count/0
]).

:- use_module(library(plunit)).
:- use_module(library(janus)).
:- use_module('../../src/unifyweaver/targets/csharp_target.pl').

:- dynamic python_helper_ready/0.

setup_paths_and_python :-
    (   python_helper_ready -> true
    ;   source_file(setup_paths_and_python, File),
        file_directory_name(File, Dir),
        absolute_file_name('../helpers', HelpersDir,
                           [relative_to(Dir), file_type(directory)]),
        py_add_lib_dir(HelpersDir),
        py_call(importlib:import_module('csharp_test_helper'), _),
        asserta(python_helper_ready)
    ).

check_dotnet_available :-
    setup_paths_and_python,
    py_call(csharp_test_helper:get_dotnet_version(), Version),
    Version \= @(none).

setup_data :-
    cleanup_data,
    assertz(user:ac_edge(a, b)),
    assertz(user:ac_edge(a, c)),
    assertz(user:ac_edge(a, d)),
    assertz(user:(ac_count(N) :- aggregate_all(count, ac_edge(_X, _Y), N))).

cleanup_data :-
    maplist(retractall, [
        user:ac_edge(_, _),
        user:ac_count(_, _)
    ]),
    catch(abolish(user:ac_edge/2), _, true),
    catch(abolish(user:ac_count/2), _, true).

build_program_stub(Stub) :-
    Stub = "\npublic static class Program {\n    public static int Main() {\n        var result = UnifyWeaver.Generated.AcCount_Module.Solve();\n        foreach (var fact in result) {\n            var arg0 = fact.Args.ContainsKey(\"arg0\") ? fact.Args[\"arg0\"] : \"\";\n            System.Console.WriteLine($\"{fact.Relation}:{arg0}\");\n        }\n        return 0;\n    }\n}\n".

:- begin_tests(csharp_generator_aggregates_run, [
    condition(check_dotnet_available)
]).

test(aggregate_all_count_simple, [
    setup(setup_data),
    cleanup(cleanup_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(ac_count/1, [mode(generator)], GenCode),
    build_program_stub(Stub),
    atom_concat(GenCode, Stub, Full),
    py_call(csharp_test_helper:compile_and_run(Full, 'agg_count'), Result),
    get_dict(success, Result, @(true)),
    get_dict(stdout, Result, Stdout),
    sub_string(Stdout, _, _, _, "ac_count:3"),
    !.

:- end_tests(csharp_generator_aggregates_run).

test_csharp_generator_aggregate_count :-
    run_tests(csharp_generator_aggregates_run).
