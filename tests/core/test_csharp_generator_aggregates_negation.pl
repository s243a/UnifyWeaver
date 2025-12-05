:- module(test_csharp_generator_aggregates_negation, [
    test_csharp_generator_aggregate_with_negation/0
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
    assertz(user:base(a)),
    assertz(user:base(b)),
    assertz(user:rel(a, 1)),
    assertz(user:rel(a, 2)),
    assertz(user:rel(b, 3)),
    assertz(user:blocked(b)),
    assertz(user:(an_sum(X, S) :- base(X), \+ blocked(X), aggregate_all(sum(V), rel(X, V), S))).

cleanup_data :-
    maplist(retractall, [
        user:base(_),
        user:rel(_, _),
        user:blocked(_),
        user:an_sum(_, _)
    ]),
    catch(abolish(user:base/1), _, true),
    catch(abolish(user:rel/2), _, true),
    catch(abolish(user:blocked/1), _, true),
    catch(abolish(user:an_sum/2), _, true).

build_program_stub(Stub) :-
    Stub = "\npublic static class Program {\n    public static int Main() {\n        var res = UnifyWeaver.Generated.AnSum_Module.Solve();\n        foreach (var fact in res) {\n            var arg0 = fact.Args.ContainsKey(\"arg0\") ? fact.Args[\"arg0\"] : \"\";\n            var arg1 = fact.Args.ContainsKey(\"arg1\") ? fact.Args[\"arg1\"] : \"\";\n            System.Console.WriteLine($\"{fact.Relation}:{arg0}:{arg1}\");\n        }\n        return 0;\n    }\n}\n".

:- begin_tests(csharp_generator_aggregates_negation, [
    condition(check_dotnet_available)
]).

test(aggregate_all_with_negation_filter, [
    setup(setup_data),
    cleanup(cleanup_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(an_sum/2, [mode(generator)], Code),
    build_program_stub(Stub),
    atom_concat(Code, Stub, Full),
    py_call(csharp_test_helper:compile_and_run(Full, 'agg_neg'), Result),
    get_dict(success, Result, @(true)),
    get_dict(stdout, Result, Stdout),
    sub_string(Stdout, _, _, _, "an_sum:a:3"),
    \+ sub_string(Stdout, _, _, _, "an_sum:b"),
    !.

:- end_tests(csharp_generator_aggregates_negation).

test_csharp_generator_aggregate_with_negation :-
    run_tests(csharp_generator_aggregates_negation).
