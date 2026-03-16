:- module(type_declarations_test, [run_all_tests/0]).

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/type_declarations').

run_all_tests :-
    run_tests([type_declarations]).

setup_type_declarations :-
    clear_type_declarations.

cleanup_type_declarations :-
    clear_type_declarations.

:- begin_tests(type_declarations, [
    setup(setup_type_declarations),
    cleanup(cleanup_type_declarations)
]).

test(resolve_type_haskell_integer) :-
    clear_type_declarations,
    once(resolve_type(integer, haskell, "Int")).

test(resolve_type_java_atom) :-
    clear_type_declarations,
    once(resolve_type(atom, java, "String")).

test(resolve_typed_mode_precedence) :-
    clear_type_declarations,
    assertz(type_declarations:uw_typed_mode(edge/2, infer)),
    resolve_typed_mode(edge/2, [typed_mode(explicit)], off, Mode),
    assertion(Mode == infer).

test(build_type_context_declared_types) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(build_type_context(edge/2, haskell, Context)),
    once(member(node_type="String", Context)),
    once(member(edge_type="(String, String)", Context)),
    once(member(typed=true, Context)).

test(build_type_context_untyped) :-
    clear_type_declarations,
    once(build_type_context(edge/2, haskell, Context)),
    assertion(Context == [typed=false]).

:- end_tests(type_declarations).
