:- module(test_cross_generator, [
    test_cross_generator/0
]).

:- use_module(library(lists)).
:- use_module(library(python_target)).
:- use_module(library(csharp_target)).

setup_test_data :-
    cleanup_test_data,
    assertz(user:cg_parent(alice, bob)),
    assertz(user:cg_parent(bob, charlie)).

cleanup_test_data :-
    maplist(retractall, [
        user:cg_parent(_, _)
    ]).

test_cross_generator :-
    writeln('=== Cross generator smoke test (Python ↔ C#) ==='),
    setup_test_data,
    test_python_generator,
    test_csharp_generator,
    cleanup_test_data,
    writeln('=== Cross generator smoke test complete ===').

test_python_generator :-
    python_target:compile_predicate_to_python(cg_parent/2, [], PyCode),
    string_length(PyCode, Len),
    (   Len > 0 ->
        writeln('  ✓ Python generator compiled')
    ;   writeln('  ✗ Python generator returned empty code'),
        fail
    ).

test_csharp_generator :-
    csharp_target:compile_predicate_to_csharp(cg_parent/2, [mode(generator)], CsCode),
    (   sub_string(CsCode, _, _, _, "class CgParent_Module"),
        sub_string(CsCode, _, _, _, "new Fact(\"cg_parent\""),
        sub_string(CsCode, _, _, _, "Solve()") ->
        writeln('  ✓ C# generator compiled')
    ;   writeln('  ✗ C# generator missing expected artifacts'),
        fail
    ).

:- begin_tests(cross_generator).
:- use_module(test_cross_generator).

test(run) :-
    test_cross_generator.

:- end_tests(cross_generator).
