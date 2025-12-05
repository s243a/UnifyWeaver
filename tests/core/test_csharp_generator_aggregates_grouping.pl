:- module(test_csharp_generator_aggregates_grouping, [
    test_csharp_generator_group_sum/0
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
    assertz(user:gs_item(a, 1)),
    assertz(user:gs_item(a, 2)),
    assertz(user:gs_item(b, 5)),
    assertz(user:gs_item(b, 7)),
    assertz(user:(gs_sum(Key, Sum) :- aggregate(sum(Val), gs_item(Key, Val), Key, Sum))).

cleanup_data :-
    maplist(retractall, [
        user:gs_item(_, _),
        user:gs_sum(_, _)
    ]),
    catch(abolish(user:gs_item/2), _, true),
    catch(abolish(user:gs_sum/2), _, true).

build_program_stub(Stub) :-
    Stub = "\npublic static class Program {\n    public static int Main() {\n        var result = UnifyWeaver.Generated.GsSum_Module.Solve();\n        foreach (var fact in result) {\n            var arg0 = fact.Args.ContainsKey(\"arg0\") ? fact.Args[\"arg0\"] : \"\";\n            var arg1 = fact.Args.ContainsKey(\"arg1\") ? fact.Args[\"arg1\"] : \"\";\n            System.Console.WriteLine($\"{fact.Relation}:{arg0}:{arg1}\");\n        }\n        return 0;\n    }\n}\n".

:- begin_tests(csharp_generator_aggregates_grouping, [
    condition(check_dotnet_available)
]).

test(aggregate_group_sum, [
    setup(setup_data),
    cleanup(cleanup_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(gs_sum/2, [mode(generator)], GenCode),
    build_program_stub(Stub),
    atom_concat(GenCode, Stub, Full),
    py_call(csharp_test_helper:compile_and_run(Full, 'agg_group_sum'), Result),
    get_dict(success, Result, @(true)),
    get_dict(stdout, Result, Stdout),
    sub_string(Stdout, _, _, _, "gs_sum:a:3"),
    sub_string(Stdout, _, _, _, "gs_sum:b:12"),
    !.

:- end_tests(csharp_generator_aggregates_grouping).

test_csharp_generator_group_sum :-
    run_tests(csharp_generator_aggregates_grouping).
