:- module(test_csharp_generator_aggregates_setbag, [
    test_csharp_generator_aggregate_setbag/0
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
    assertz(user:sb_val(a, 1)),
    assertz(user:sb_val(b, 1)),
    assertz(user:sb_val(a, 2)),
    assertz(user:(sb_set(S) :- aggregate_all(set(V), sb_val(V, _), S))),
    assertz(user:(sb_bag(B) :- aggregate_all(bag(V), sb_val(V, _), B))).

cleanup_data :-
    maplist(retractall, [
        user:sb_val(_, _),
        user:sb_set(_),
        user:sb_bag(_)
    ]),
    catch(abolish(user:sb_val/2), _, true),
    catch(abolish(user:sb_set/1), _, true),
    catch(abolish(user:sb_bag/1), _, true).

build_program_stub(Module, Stub) :-
    format(string(Stub),
"\npublic static class Program {\n    public static int Main() {\n        var res = UnifyWeaver.Generated.~w.Solve();\n        foreach (var fact in res) {\n            if (fact.Args.ContainsKey(\"arg0\") && fact.Args[\"arg0\"] is System.Collections.Generic.List<object> lst) {\n                lst.Sort();\n                System.Console.WriteLine($\"{fact.Relation}:{string.Join(\",\", lst)}\");\n            }\n        }\n        return 0;\n    }\n}\n", [Module]).

:- begin_tests(csharp_generator_aggregates_setbag, [
    condition(check_dotnet_available)
]).

test(aggregate_all_set_bag, [
    setup(setup_data),
    cleanup(cleanup_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(sb_set/1, [mode(generator)], SetCode),
    build_program_stub("SbSet_Module", SetStub),
    atom_concat(SetCode, SetStub, SetFull),
    py_call(csharp_test_helper:compile_and_run(SetFull, 'agg_set_only'), SetResult),
    get_dict(success, SetResult, @(true)),
    get_dict(stdout, SetResult, SetStdout),
    sub_string(SetStdout, _, _, _, "sb_set:a,b"),
    csharp_target:compile_predicate_to_csharp(sb_bag/1, [mode(generator)], BagCode),
    build_program_stub("SbBag_Module", BagStub),
    atom_concat(BagCode, BagStub, BagFull),
    py_call(csharp_test_helper:compile_and_run(BagFull, 'agg_bag_only'), BagResult),
    get_dict(success, BagResult, @(true)),
    get_dict(stdout, BagResult, BagStdout),
    sub_string(BagStdout, _, _, _, "sb_bag:a,a,b"),
    !.

:- end_tests(csharp_generator_aggregates_setbag).

test_csharp_generator_aggregate_setbag :-
    run_tests(csharp_generator_aggregates_setbag).
