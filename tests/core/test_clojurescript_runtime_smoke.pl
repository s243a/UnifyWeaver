% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_clojurescript_runtime_smoke.pl - End-to-end runtime smoke test for the
% ClojureScript target. Compiles predicates to ClojureScript, then *executes*
% the generated code with nbb (Node ClojureScript) and checks the real output.
%
% This is what proves the JVM->JS interop rewrites (js/parseInt, js/Math.abs,
% ...) actually run, not just that the strings were substituted.
%
% Runtime gate: the tests are skipped (not failed) when nbb is unavailable, so
% environments without a Node ClojureScript runtime still pass. Provide nbb on
% PATH, or point the NBB environment variable at the binary, to exercise them.
%   npm install nbb && NBB=./node_modules/.bin/nbb swipl ... test_...

:- module(test_clojurescript_runtime_smoke, [test_clojurescript_runtime_smoke/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(readutil)).
:- use_module('../../src/unifyweaver/targets/clojurescript_target').

test_clojurescript_runtime_smoke :-
    run_tests([clojurescript_runtime_smoke]).

% --- nbb discovery: NBB env var first, then PATH ---------------------------
nbb_path(Path) :-
    (   getenv('NBB', P), P \== ''
    ->  Path = P
    ;   absolute_file_name(path(nbb), Path, [access(execute), file_errors(fail)])
    ).

nbb_available :- catch(nbb_path(_), _, fail).

% --- compile a predicate to CLJS and run it under nbb with one arg ---------
compile_and_run(Pred/Arity, Arg, Output) :-
    clojurescript_target:compile_predicate_to_clojurescript(Pred/Arity, [], Code),
    tmp_file(cljs_smoke, Base),
    atom_concat(Base, '.cljs', File),
    setup_call_cleanup(
        ( open(File, write, S), write(S, Code), close(S) ),
        run_nbb(File, Arg, Output),
        catch(delete_file(File), _, true)
    ).

run_nbb(File, Arg, Output) :-
    nbb_path(Nbb),
    ( number(Arg) -> term_to_atom(Arg, ArgA) ; ArgA = Arg ),
    process_create(Nbb, [File, ArgA],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out), close(Err),
    process_wait(PID, _Status),
    ( ErrStr == "" -> true
    ; format(user_error, "nbb stderr: ~w~n", [ErrStr]) ),
    normalize_space(atom(Output), OutStr).

:- begin_tests(clojurescript_runtime_smoke).

% Sample predicates spanning arithmetic, guard chains, and Math interop.
setup_preds :-
    retractall(user:triple(_,_)),
    retractall(user:classify(_,_)),
    retractall(user:absval(_,_)),
    assertz(user:(triple(X,R) :- R is X*3)),
    assertz(user:(classify(X,small) :- X>0, X<10)),
    assertz(user:(classify(X,large) :- X>=10)),
    assertz(user:(absval(X,R) :- R is abs(X))).

test(triple_arithmetic, [condition(nbb_available), setup(setup_preds)]) :-
    compile_and_run(triple/2, 5, Out),
    assertion(Out == '15').

test(classify_small_branch, [condition(nbb_available), setup(setup_preds)]) :-
    compile_and_run(classify/2, 3, Out),
    assertion(Out == small).

test(classify_large_branch, [condition(nbb_available), setup(setup_preds)]) :-
    compile_and_run(classify/2, 42, Out),
    assertion(Out == large).

% Exercises the Math/abs -> js/Math.abs interop rewrite at runtime.
test(absval_math_interop, [condition(nbb_available), setup(setup_preds)]) :-
    compile_and_run(absval/2, 7, Out),
    assertion(Out == '7').

:- end_tests(clojurescript_runtime_smoke).
