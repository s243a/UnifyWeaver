:- module(test_csharp_generator_aggregates_minmax, [
    test_csharp_generator_aggregate_minmax/0
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
    assertz(user:mm_val(3)),
    assertz(user:mm_val(5)),
    assertz(user:mm_val(10)),
    assertz(user:(mm_min(M) :- aggregate_all(min(V), mm_val(V), M))),
    assertz(user:(mm_max(M) :- aggregate_all(max(V), mm_val(V), M))).

cleanup_data :-
    maplist(retractall, [
        user:mm_val(_),
        user:mm_min(_),
        user:mm_max(_)
    ]),
    catch(abolish(user:mm_val/1), _, true),
    catch(abolish(user:mm_min/1), _, true),
    catch(abolish(user:mm_max/1), _, true).

build_program_stub(Module, Stub) :-
    format(string(Stub),
"\npublic static class Program {\n    public static int Main() {\n        var res = UnifyWeaver.Generated.~w.Solve();\n        foreach (var fact in res) {\n            var arg0 = fact.Args.ContainsKey(\"arg0\") ? fact.Args[\"arg0\"] : \"\";\n            System.Console.WriteLine($\"{fact.Relation}:{arg0}\");\n        }\n        return 0;\n    }\n}\n", [Module]).

:- begin_tests(csharp_generator_aggregates_minmax, [
    condition(check_dotnet_available)
]).

test(aggregate_all_minmax_simple, [
    setup(setup_data),
    cleanup(cleanup_data),
    condition(check_dotnet_available)
]) :-
    csharp_target:compile_predicate_to_csharp(mm_min/1, [mode(generator)], MinCode),
    build_program_stub("MmMin_Module", MinStub),
    atom_concat(MinCode, MinStub, MinFull),
    py_call(csharp_test_helper:compile_and_run(MinFull, 'agg_min'), MinResult),
    get_dict(success, MinResult, @(true)),
    get_dict(stdout, MinResult, MinOut),
    sub_string(MinOut, _, _, _, "mm_min:3"),
    csharp_target:compile_predicate_to_csharp(mm_max/1, [mode(generator)], MaxCode),
    build_program_stub("MmMax_Module", MaxStub),
    atom_concat(MaxCode, MaxStub, MaxFull),
    py_call(csharp_test_helper:compile_and_run(MaxFull, 'agg_max'), MaxResult),
    get_dict(success, MaxResult, @(true)),
    get_dict(stdout, MaxResult, MaxOut),
    sub_string(MaxOut, _, _, _, "mm_max:10"),
    !.

:- end_tests(csharp_generator_aggregates_minmax).

test_csharp_generator_aggregate_minmax :-
    run_tests(csharp_generator_aggregates_minmax).
