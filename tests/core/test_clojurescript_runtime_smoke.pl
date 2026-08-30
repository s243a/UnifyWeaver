% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_clojurescript_runtime_smoke.pl - End-to-end runtime smoke test for the
% ClojureScript target. Compiles predicates to ClojureScript, then *executes*
% the generated code with a real sci runtime and checks the actual output.
%
% Two runtime variants are exercised:
%   - nbb : Node ClojureScript (sci). JS host interop -- proves the JVM->JS
%           rewrites (js/parseInt, js/Math.abs, ...) actually run, not just
%           that the strings were substituted. Generated with runtime(nbb).
%   - bb  : Babashka (sci, Clojure-on-JVM), run as an external binary with no
%           npm dependency. bb is Clojure, so it keeps the JVM host interop
%           (Integer/parseInt, Math/abs) -- proves runtime(bb) leaves the
%           interop UNrewritten and the script still runs.
%
% Runtime gate: each variant's tests are SKIPPED (not failed) when its binary
% is unavailable, so CI without a sci runtime still passes. Provide the binary
% on PATH, or point the env var at it, to exercise them:
%   npm install nbb && NBB=./node_modules/.bin/nbb swipl ... test_...
%   (bb on PATH, or BB=/path/to/bb) for the Babashka arms.

:- module(test_clojurescript_runtime_smoke, [test_clojurescript_runtime_smoke/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(readutil)).
:- use_module('../../src/unifyweaver/targets/clojurescript_target').
:- use_module('../../src/unifyweaver/core/recursive_compiler').

test_clojurescript_runtime_smoke :-
    run_tests([clojurescript_runtime_smoke]).

% --- nbb discovery: NBB env var first, then PATH ---------------------------
nbb_path(Path) :-
    (   getenv('NBB', P), P \== ''
    ->  Path = P
    ;   absolute_file_name(path(nbb), Path, [access(execute), file_errors(fail)])
    ).

nbb_available :- catch(nbb_path(_), _, fail).

% --- bb (Babashka) discovery: BB env var first, then PATH ------------------
bb_path(Path) :-
    (   getenv('BB', P), P \== ''
    ->  Path = P
    ;   absolute_file_name(path(bb), Path, [access(execute), file_errors(fail)])
    ).

bb_available :- catch(bb_path(_), _, fail).

% --- compile a predicate to CLJS and run it under nbb with one arg ---------
compile_and_run(Pred/Arity, Arg, Output) :-
    clojurescript_target:compile_predicate_to_clojurescript(Pred/Arity, [runtime(nbb)], Code),
    tmp_file(cljs_smoke, Base),
    atom_concat(Base, '.cljs', File),
    setup_call_cleanup(
        ( open(File, write, S), write(S, Code), close(S) ),
        run_nbb(File, Arg, Output),
        catch(delete_file(File), _, true)
    ).

% --- compile a predicate to the bb (Babashka) variant and run it under bb ---
% runtime(bb) keeps JVM host interop (Integer/parseInt, Math/abs) unchanged.
compile_and_run_bb(Pred/Arity, Arg, Output) :-
    clojurescript_target:compile_predicate_to_clojurescript(Pred/Arity, [runtime(bb)], Code),
    tmp_file(cljs_bb_smoke, Base),
    atom_concat(Base, '.clj', File),
    setup_call_cleanup(
        ( open(File, write, S), write(S, Code), close(S) ),
        run_bb(File, Arg, Output),
        catch(delete_file(File), _, true)
    ).

run_bb(File, Arg, Output) :-
    bb_path(Bb),
    ( number(Arg) -> term_to_atom(Arg, ArgA) ; ArgA = Arg ),
    process_create(Bb, [File, ArgA],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out), close(Err),
    process_wait(PID, _Status),
    ( ErrStr == "" -> true
    ; format(user_error, "bb stderr: ~w~n", [ErrStr]) ),
    normalize_space(atom(Output), OutStr).

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

% --- compile ancestor/2 (transitive closure) to CLJS, append REPL queries, ---
% --- and run the whole file under nbb (no CLI args: embedded facts + the   ---
% --- appended forms execute at load - same shape the SciREPL workbook runs). --
compile_tc_and_run(Output) :-
    recursive_compiler:compile_recursive(ancestor/2, [target(clojurescript)], Code),
    tmp_file(cljs_tc, Base),
    atom_concat(Base, '.cljs', File),
    Queries = "\n(println (sort (find-all \"alice\")))\n(println (check-path \"alice\" \"dave\"))\n(println (check-path \"alice\" \"zzz\"))\n",
    setup_call_cleanup(
        ( open(File, write, S), write(S, Code), write(S, Queries), close(S) ),
        run_nbb_noargs(File, Output),
        catch(delete_file(File), _, true)
    ).

run_nbb_noargs(File, Output) :-
    nbb_path(Nbb),
    process_create(Nbb, [File],
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

% Transitive closure compiled via recursive_compiler actually runs in CLJS:
% find-all returns the descendants, check-path answers reachability. This is
% the runtime half of the "Prolog generates ClojureScript" demo.
setup_family :-
    retractall(user:parent(_,_)),
    retractall(user:ancestor(_,_)),
    assertz(user:parent(alice, bob)),
    assertz(user:parent(bob, charlie)),
    assertz(user:parent(charlie, dave)),
    assertz(user:(ancestor(X,Y) :- parent(X,Y))),
    assertz(user:(ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y))).

test(transitive_closure_runtime, [condition(nbb_available), setup(setup_family)]) :-
    compile_tc_and_run(Out),
    assertion(Out == '(bob charlie dave) true false').

% --- Babashka (bb) arms: runtime(bb) keeps JVM host interop; the same -------
% --- predicates must produce the same results under the sci Clojure host. ---
% Skipped cleanly when bb is absent (condition(bb_available)).
test(bb_triple_arithmetic, [condition(bb_available), setup(setup_preds)]) :-
    compile_and_run_bb(triple/2, 5, Out),
    assertion(Out == '15').

test(bb_classify_small_branch, [condition(bb_available), setup(setup_preds)]) :-
    compile_and_run_bb(classify/2, 3, Out),
    assertion(Out == small).

test(bb_classify_large_branch, [condition(bb_available), setup(setup_preds)]) :-
    compile_and_run_bb(classify/2, 42, Out),
    assertion(Out == large).

% Exercises that Math/abs is retained (NOT rewritten to js/Math.abs) and runs.
test(bb_absval_jvm_interop, [condition(bb_available), setup(setup_preds)]) :-
    compile_and_run_bb(absval/2, 7, Out),
    assertion(Out == '7').

:- end_tests(clojurescript_runtime_smoke).
