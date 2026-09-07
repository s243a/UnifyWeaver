% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_clojurescript_wam_backtracking.pl - runtime probes for the ClojureScript
% WAM lane (wam_clojure_target's write_wam_clojurescript_files/3, whose output
% goes through clojurescript_target's JVM->JS interop rewrite).
%
% Each probe pins ONE defect that examples/pkg_resolver/cljs exposed when the
% first BACKTRACKING program was pointed at this lane -- the D40 argparser is
% deterministic, so none of these had ever been reached. Every probe is a
% minimal Prolog program, compiled here and EXECUTED under nbb; each one
% answered wrongly (or failed outright) before its fix:
%
%   nested_read_mode     a clause head that matches a compound INSIDE a list
%                        cell lost the list TAIL: `get_structure` overwrote the
%                        enclosing structure's read cursor instead of stacking
%                        it, so every list walk stopped after one element.
%   get_structure_write  an OUTPUT argument whose head is a compound could not
%                        be constructed: `get_structure` had no write mode and
%                        the clause head simply failed.
%   negation_cut_barrier `\+ G` lowers to try_me_else/call/!/fail, and `!`
%                        pruned against whatever barrier the last `allocate`
%                        left behind rather than this activation's. Every
%                        negation whose goal SUCCEEDED reported success.
%   ite_commits          `(Cond -> Then ; Else)` did not commit: cut_ite popped
%                        exactly one choice point, which was one Cond had left,
%                        so a failing Then fell through into Else.
%   call_then_ite        the structured-ITE lowering kept emitting straight-line
%                        steps AFTER a call, executing them against the callee's
%                        program counter and reporting a success the predicate
%                        never had, with its output argument still unbound.
%
% The lane is exercised end to end (compile -> nbb -> answer), because every one
% of these is invisible in the emitted text and only shows up when the code runs.
%
% nbb gate: SKIPPED (not failed) when nbb is unavailable, like
% test_clojurescript_runtime_smoke.pl.

:- module(test_clojurescript_wam_backtracking,
          [test_clojurescript_wam_backtracking/0]).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../../src/unifyweaver/targets/wam_clojure_target',
              [write_wam_clojurescript_files/3]).

test_clojurescript_wam_backtracking :-
    run_tests([clojurescript_wam_backtracking]).

nbb_path(Path) :-
    (   getenv('NBB', P), P \== ''
    ->  Path = P
    ;   absolute_file_name(path(nbb), Path, [access(execute), file_errors(fail)])
    ).

nbb_available :- catch(nbb_path(_), _, fail).

% --- the probe program ------------------------------------------------------

:- dynamic user:probe_names/3.
:- dynamic user:probe_wrap/2.
:- dynamic user:probe_lt/2.
:- dynamic user:probe_sat/2.
:- dynamic user:probe_base/2.
:- dynamic user:probe_pick/2.
:- dynamic user:probe_inner/2.
:- dynamic user:probe_outer/2.

probe_predicates([probe_names/3, probe_wrap/2, probe_lt/2, probe_sat/2,
                  probe_base/2, probe_pick/2, probe_inner/2, probe_outer/2]).

setup_probe_program :-
    forall(member(P/A, [probe_names/3, probe_wrap/2, probe_lt/2, probe_sat/2,
                        probe_base/2, probe_pick/2, probe_inner/2, probe_outer/2]),
           ( functor(H, P, A), retractall(user:H) )),
    % 1. a clause head matching a compound inside a list cell -- the TAIL is the
    %    enclosing cell's second argument and must survive the inner match.
    assertz(user:(probe_names([], Acc, Acc))),
    assertz(user:(probe_names([p(N, _V)|T], Acc, Out) :-
                      probe_names(T, [N|Acc], Out))),
    % 2. an output argument whose head is a compound.
    assertz(user:(probe_wrap(X, w(X, ok)))),
    % 3. negation over a goal that succeeds.
    assertz(user:(probe_lt(v(A), v(B)) :- ( A < B -> true ; fail ))),
    assertz(user:(probe_sat(V, gte(G)) :- \+ probe_lt(V, G))),
    % 4. an if-then-else whose condition leaves a choice point and whose THEN
    %    then fails: -> must commit, so the whole call fails.
    assertz(user:probe_base(x, 1)),
    assertz(user:probe_base(x, 2)),
    assertz(user:(probe_pick(N, R) :-
                      ( probe_base(x, V) -> V >= N, R = V ; R = none ))),
    % 5. a call followed by an if-then-else in the same clause.
    assertz(user:probe_inner(a, one)),
    assertz(user:(probe_outer(X, R) :-
                      probe_inner(X, Y), ( Y == one -> R = yes ; R = no ))).

% --- compile + run ----------------------------------------------------------

%% probe_output(-Lines)
%  Compile the probe program to ClojureScript, run the driver under nbb, and
%  return stdout split into lines.
probe_output(Lines) :-
    setup_probe_program,
    tmp_file(cljs_wam_probe, Base),
    atom_concat(Base, '_dir', Dir),
    atomic_list_concat([Dir, '/generated/probe'], NsDir),
    make_directory_path(NsDir),
    probe_predicates(Preds),
    setup_call_cleanup(
        ( write_wam_clojurescript_files(Preds, [namespace('generated.probe')], NsDir),
          atomic_list_concat([NsDir, '/.wamclj_tmp'], TmpProj),
          catch(delete_directory_and_contents(TmpProj), _, true),
          atomic_list_concat([Dir, '/driver.cljs'], Driver),
          probe_driver_source(Src),
          setup_call_cleanup(open(Driver, write, S), write(S, Src), close(S)) ),
        run_nbb(Dir, Driver, Out),
        catch(delete_directory_and_contents(Dir), _, true)),
    split_string(Out, "\n", " \r", Lines0),
    exclude(==(""), Lines0, Lines).

run_nbb(Classpath, Driver, Output) :-
    nbb_path(Nbb),
    process_create(Nbb, ['--classpath', Classpath, Driver],
                   [stdout(pipe(O)), stderr(pipe(E)), process(PID)]),
    read_string(O, _, Output),
    read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(PID, _),
    ( ErrStr == "" -> true ; format(user_error, "nbb stderr: ~w~n", [ErrStr]) ).

%% probe_driver_source(-Source)
%  The driver builds WAM terms, calls the generated `<pred>-state` wrappers
%  (which answer the SUCCEEDING state, or nil), and prints one `name=value`
%  line per probe. It contains no probe logic of its own.
probe_driver_source(
"(ns driver
  (:require [generated.probe.core :as c]
            [generated.probe.runtime :as rt]
            [clojure.string :as str]))

(defn st [f args] (rt/structure-term f (vec args)))
(defn lst [xs] (reduce (fn [t x] (st \"[|]/2\" [x t])) \"[]\" (reverse xs)))
(defn out [] {:var \"probe-out\"})

(defn show [state t]
  (let [t (rt/deref-value (:bindings state) t)
        ctx (:intern-context state)]
    (cond
      (rt/atom-term? t) (rt/deintern-atom ctx (:id t))
      (rt/structure-term? t)
      (let [n (str/replace (rt/deintern-atom ctx (:id (:functor t))) #\"/\\d+$\" \"\")
            a (mapv #(show state %) (:args t))]
        (if (= n \"[|]\")
          (str \"[\" (str/join \",\" (loop [cur t acc []]
                                    (let [cur (rt/deref-value (:bindings state) cur)]
                                      (if (and (rt/structure-term? cur)
                                               (= \"[|]\" (str/replace (rt/deintern-atom ctx (:id (:functor cur))) #\"/\\d+$\" \"\")))
                                        (recur (nth (:args cur) 1) (conj acc (show state (nth (:args cur) 0))))
                                        acc)))) \"]\")
          (str n \"(\" (str/join \",\" a) \")\")))
      :else (str t))))

(defn line [label state t]
  (println (str label \"=\" (if state (if t (show state t) \"true\") \"FAIL\"))))

;; 1. nested read mode -- the list TAIL must survive the inner p/2 match
(let [o (out)
      s (c/probe-names-state (lst [(st \"p/2\" [\"a\" 1]) (st \"p/2\" [\"b\" 2])]) \"[]\" o)]
  (line \"names\" s o))

;; 2. get_structure write mode -- construct a compound OUTPUT argument
(let [o (out) s (c/probe-wrap-state \"z\" o)] (line \"wrap\" s o))

;; 3. negation over a goal that succeeds must FAIL
(let [s (c/probe-sat-state (st \"v/1\" [1]) (st \"gte/1\" [(st \"v/1\" [2])]))]
  (line \"sat_false\" s nil))
(let [s (c/probe-sat-state (st \"v/1\" [2]) (st \"gte/1\" [(st \"v/1\" [1])]))]
  (line \"sat_true\" s nil))

;; 4. -> commits: a failing THEN must not fall through to ELSE
(let [o (out) s (c/probe-pick-state 2 o)] (line \"pick2\" s o))
(let [o (out) s (c/probe-pick-state 1 o)] (line \"pick1\" s o))

;; 5. a call followed by an if-then-else in the same clause
(let [o (out) s (c/probe-outer-state \"a\" o)] (line \"outer\" s o))
").

% --- the probes -------------------------------------------------------------

:- begin_tests(clojurescript_wam_backtracking).

probe_line(Lines, Key, Value) :-
    atomic_list_concat([Key, '='], Prefix),
    member(L, Lines),
    string_concat(Prefix, V, L),
    atom_string(Value, V),
    !.

test(nested_read_mode, [condition(nbb_available)]) :-
    probe_output(Lines),
    probe_line(Lines, names, V),
    assertion(V == '[b,a]').

test(get_structure_write, [condition(nbb_available)]) :-
    probe_output(Lines),
    probe_line(Lines, wrap, V),
    assertion(V == 'w(z,ok)').

test(negation_cut_barrier, [condition(nbb_available)]) :-
    probe_output(Lines),
    probe_line(Lines, sat_false, F),
    probe_line(Lines, sat_true, T),
    assertion(F == 'FAIL'),
    assertion(T == true).

test(ite_commits, [condition(nbb_available)]) :-
    probe_output(Lines),
    probe_line(Lines, pick2, P2),
    probe_line(Lines, pick1, P1),
    assertion(P2 == 'FAIL'),
    assertion(P1 == '1').

test(call_then_ite, [condition(nbb_available)]) :-
    probe_output(Lines),
    probe_line(Lines, outer, V),
    assertion(V == yes).

:- end_tests(clojurescript_wam_backtracking).
