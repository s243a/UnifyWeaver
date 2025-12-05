:- module(test_csharp_generator_aggregates_grouping_ext, [
    test_csharp_generator_group_minmax_setbag/0
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
    assertz(user:(gs_min(Key, Min) :- aggregate(min(Val), gs_item(Key, Val), Key, Min))),
    assertz(user:(gs_max(Key, Max) :- aggregate(max(Val), gs_item(Key, Val), Key, Max))),
    assertz(user:(gs_set(Key, Set) :- aggregate(set(Val), gs_item(Key, Val), Key, Set))),
    assertz(user:(gs_bag(Key, Bag) :- aggregate(bag(Val), gs_item(Key, Val), Key, Bag))).

cleanup_data :-
    maplist(retractall, [
        user:gs_item(_, _),
        user:gs_min(_, _),
        user:gs_max(_, _),
        user:gs_set(_, _),
        user:gs_bag(_, _)
    ]),
    catch(abolish(user:gs_item/2), _, true),
    catch(abolish(user:gs_min/2), _, true),
    catch(abolish(user:gs_max/2), _, true),
    catch(abolish(user:gs_set/2), _, true),
    catch(abolish(user:gs_bag/2), _, true).

build_program_stub(Module, Stub) :-
    format(string(Stub),
"\npublic static class Program {\n    public static int Main() {\n        var res = UnifyWeaver.Generated.~w.Solve();\n        foreach (var fact in res) {\n            var arg0 = fact.Args.ContainsKey(\"arg0\") ? fact.Args[\"arg0\"] : \"\";\n            var arg1 = fact.Args.ContainsKey(\"arg1\") ? fact.Args[\"arg1\"] : \"\";\n            if (arg1 is System.Collections.Generic.List<object> lst) {\n                lst.Sort();\n                System.Console.WriteLine($\"{fact.Relation}:{arg0}:{string.Join(\",\", lst)}\");\n            } else {\n                System.Console.WriteLine($\"{fact.Relation}:{arg0}:{arg1}\");\n            }\n        }\n        return 0;\n    }\n}\n", [Module]).

:- begin_tests(csharp_generator_aggregates_grouping_ext, [
    condition(check_dotnet_available)
]).

test(aggregate_group_minmax_setbag, [
    setup(setup_data),
    cleanup(cleanup_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(gs_min/2, [mode(generator)], MinCode),
    build_program_stub("GsMin_Module", MinStub),
    atom_concat(MinCode, MinStub, MinFull),
    py_call(csharp_test_helper:compile_and_run(MinFull, 'agg_group_min'), MinResult),
    get_dict(success, MinResult, @(true)),
    get_dict(stdout, MinResult, MinOut),
    sub_string(MinOut, _, _, _, "gs_min:a:1"),
    sub_string(MinOut, _, _, _, "gs_min:b:5"),

    csharp_target:compile_predicate_to_csharp(gs_max/2, [mode(generator)], MaxCode),
    build_program_stub("GsMax_Module", MaxStub),
    atom_concat(MaxCode, MaxStub, MaxFull),
    py_call(csharp_test_helper:compile_and_run(MaxFull, 'agg_group_max'), MaxResult),
    get_dict(success, MaxResult, @(true)),
    get_dict(stdout, MaxResult, MaxOut),
    sub_string(MaxOut, _, _, _, "gs_max:a:2"),
    sub_string(MaxOut, _, _, _, "gs_max:b:7"),

    csharp_target:compile_predicate_to_csharp(gs_set/2, [mode(generator)], SetCode),
    build_program_stub("GsSet_Module", SetStub),
    atom_concat(SetCode, SetStub, SetFull),
    py_call(csharp_test_helper:compile_and_run(SetFull, 'agg_group_set'), SetResult),
    get_dict(success, SetResult, @(true)),
    get_dict(stdout, SetResult, SetOut),
    sub_string(SetOut, _, _, _, "gs_set:a:1,2"),
    sub_string(SetOut, _, _, _, "gs_set:b:5,7"),

    csharp_target:compile_predicate_to_csharp(gs_bag/2, [mode(generator)], BagCode),
    build_program_stub("GsBag_Module", BagStub),
    atom_concat(BagCode, BagStub, BagFull),
    py_call(csharp_test_helper:compile_and_run(BagFull, 'agg_group_bag'), BagResult),
    get_dict(success, BagResult, @(true)),
    get_dict(stdout, BagResult, BagOut),
    sub_string(BagOut, _, _, _, "gs_bag:a:1,2"),
    sub_string(BagOut, _, _, _, "gs_bag:b:5,7"),
    !.

:- end_tests(csharp_generator_aggregates_grouping_ext).

test_csharp_generator_group_minmax_setbag :-
    run_tests(csharp_generator_aggregates_grouping_ext).
