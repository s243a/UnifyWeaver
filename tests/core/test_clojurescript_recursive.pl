% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_clojurescript_recursive.pl - Codegen tests for compiling *recursive*
% predicates to ClojureScript through the recursive_compiler.
%
% Phase 3 of ClojureScript support: the single-predicate path lives in
% test_clojurescript_target.pl; this file covers the recursive_compiler wiring
% that lets `compile_recursive(.../target(clojurescript))` emit browser-runnable
% CLJS - the compiler half of the "Prolog generates ClojureScript" SciREPL demo.
%
% These assertions are deterministic (no runtime needed); the matching nbb
% execution check lives in test_clojurescript_runtime_smoke.pl.

:- module(test_clojurescript_recursive, [test_clojurescript_recursive/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/core/recursive_compiler').

test_clojurescript_recursive :-
    run_tests([clojurescript_recursive]).

:- begin_tests(clojurescript_recursive).

% Deterministic substring checks
has(Code, Substr)  :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

% Family-tree fixture: parent/2 facts + ancestor/2 transitive closure.
setup_family :-
    retractall(user:parent(_, _)),
    retractall(user:ancestor(_, _)),
    assertz(user:parent(alice, bob)),
    assertz(user:parent(bob, charlie)),
    assertz(user:parent(charlie, dave)),
    assertz(user:(ancestor(X, Y) :- parent(X, Y))),
    assertz(user:(ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y))).

teardown_family :-
    retractall(user:parent(_, _)),
    retractall(user:ancestor(_, _)).

% ============================================================================
% Transitive closure -> ClojureScript
% ============================================================================

% The TC path produces the CLJS banner, the BFS helpers, and embedded facts by
% default - i.e. code that runs as-is in a browser CLJS runtime (Scittle/SCI).
test(tc_emits_cljs_banner, [setup(setup_family), cleanup(teardown_family)]) :-
    recursive_compiler:compile_recursive(ancestor/2, [target(clojurescript)], Code),
    has(Code, "Target: ClojureScript").

test(tc_emits_bfs_helpers, [setup(setup_family), cleanup(teardown_family)]) :-
    recursive_compiler:compile_recursive(ancestor/2, [target(clojurescript)], Code),
    has(Code, "(defn find-all"),
    has(Code, "(defn check-path"),
    has(Code, "(defn add-fact").

% Default is embedded facts (no input(...) option given), so the Prolog facts
% are seeded inline rather than read from stdin/CLI.
test(tc_defaults_to_embedded_facts, [setup(setup_family), cleanup(teardown_family)]) :-
    recursive_compiler:compile_recursive(ancestor/2, [target(clojurescript)], Code),
    has(Code, "(add-fact \"alice\" \"bob\")"),
    has(Code, "(add-fact \"charlie\" \"dave\")").

% Critically: no JVM-only host interop leaks into the browser output.
test(tc_no_jvm_interop, [setup(setup_family), cleanup(teardown_family)]) :-
    recursive_compiler:compile_recursive(ancestor/2, [target(clojurescript)], Code),
    hasnt(Code, "java.io"),
    hasnt(Code, "System/exit"),
    hasnt(Code, "*command-line-args*"),
    hasnt(Code, "line-seq").

% An explicit input(Mode) is honored rather than overridden by the default.
test(tc_honors_explicit_input, [setup(setup_family), cleanup(teardown_family)]) :-
    recursive_compiler:compile_recursive(ancestor/2,
        [target(clojurescript), input(function)], Code),
    has(Code, "(defn ancestor-from-pairs").

% ============================================================================
% Non-recursive predicate -> ClojureScript (via recursive_compiler dispatch)
% ============================================================================

test(non_recursive_arithmetic, [setup(setup_arith), cleanup(teardown_arith)]) :-
    once(recursive_compiler:compile_recursive(triple/2, [target(clojurescript)], Code)),
    has(Code, "Target: ClojureScript"),
    has(Code, "(defn triple [arg1]"),
    has(Code, "(* arg1 3)").

setup_arith :-
    retractall(user:triple(_, _)),
    assertz(user:(triple(X, R) :- R is X * 3)).

teardown_arith :-
    retractall(user:triple(_, _)).

:- end_tests(clojurescript_recursive).
