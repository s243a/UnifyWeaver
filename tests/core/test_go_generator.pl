:- module(test_go_generator, [
    run_tests/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/go_target.pl').

%% Setup test predicates
setup_ancestor_example :-
    cleanup_all,
    % Base facts
    assertz(user:parent(john, mary)),
    assertz(user:parent(mary, sue)),
    assertz(user:parent(sue, alice)),
    % Recursive rule
    assertz(user:(ancestor(X, Y) :- parent(X, Y))),
    assertz(user:(ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z))).

setup_negation_example :-
    cleanup_all,
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(user:blocked(a, b)),
    assertz(user:(path(X, Y) :- edge(X, Y), \+ blocked(X, Y))).

cleanup_all :-
    catch(abolish(user:parent/2), _, true),
    catch(abolish(user:ancestor/2), _, true),
    catch(abolish(user:edge/2), _, true),
    catch(abolish(user:blocked/2), _, true),
    catch(abolish(user:path/2), _, true).

:- begin_tests(go_generator).

test(compile_transitive_closure, [
    setup(setup_ancestor_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(ancestor/2, [mode(generator)], Code),
    % Check that code contains key elements
    sub_string(Code, _, _, _, "func Solve()"),
    sub_string(Code, _, _, _, "GetInitialFacts"),
    sub_string(Code, _, _, _, "ApplyRule_"),
    sub_string(Code, _, _, _, "Fact{Relation:"),
    format('~n=== Generated Go Code ===~n~w~n', [Code]).

test(compile_with_negation, [
    setup(setup_negation_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(path/2, [mode(generator)], Code),
    % Check negation handling
    sub_string(Code, _, _, _, "negFact"),
    sub_string(Code, _, _, _, "total[negFact.Key()]"),
    format('~n=== Generated Go Code (with negation) ===~n~w~n', [Code]).

test(contains_fixpoint_loop, [
    setup(setup_ancestor_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(ancestor/2, [mode(generator)], Code),
    % Verify fixpoint structure
    sub_string(Code, _, _, _, "changed := true"),
    sub_string(Code, _, _, _, "for changed"),
    sub_string(Code, _, _, _, "total[key] = nf").

:- end_tests(go_generator).

run_tests :-
    run_tests(go_generator).
