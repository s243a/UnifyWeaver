% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_clojurescript_wam_indexing.pl - runtime probes for first-argument
% indexing in the ClojureScript WAM lane (wam_clojure_target's
% write_wam_clojurescript_files/3).
%
% wam_target.pl emits the standard-WAM switch family ahead of every
% multi-clause chain (switch_on_constant / switch_on_structure /
% switch_on_term, on A1 or A2, plus dedicated try / retry / trust dispatch
% chains). The Clojure runtime used to skip them all as unimplemented `:raw`
% hints and run every chain unindexed; it now executes them.
%
% Indexing is an OPTIMISATION: it must be invisible in the answers. These
% probes therefore pin exactly that -- the same predicate, compiled and run
% under nbb, must give SWI's answers in SWI's order:
%
%   order_preserved      idx_c/2 has clauses a,b,a and a trailing variable
%                        clause, so the switch table carries the key `a`
%                        TWICE. The entry list must be read first-match-wins
%                        (`a` -> fall through to the whole chain), exactly as
%                        a linear scan of it would: a last-wins map answers
%                        [3,9] where SWI answers [1,3,9].
%   fresh_atom_dispatch  a dispatch key reaches the runtime as a freshly built
%                        string, not as one of the switch table's own interned
%                        seeds; it must dispatch to the same clause. Includes a
%                        functor key ("g/1") built at run time.
%   unbound_first_arg    a query whose dispatch register is UNBOUND must fall
%                        through and enumerate every clause -- indexing may
%                        only ever remove work that could not have produced an
%                        answer.
%   dispatch_across_cut  an if-then-else that prunes choice points, run while
%                        an indexed enumeration of the same predicate is still
%                        being backtracked into. A dispatch that jumps straight
%                        at a clause body pushes NO choice point, so a pruned
%                        one must not leave the chain half-entered.
%   list_and_structure   switch_on_term's list label and switch_on_structure's
%                        functor table: the [] / [H|T] split every list walk in
%                        resolver.pl rests on, and a functor with no entry,
%                        which must still fail rather than dispatch anywhere.
%
% nbb gate: SKIPPED (not failed) when nbb is unavailable, like
% test_clojurescript_wam_backtracking.pl.

:- module(test_clojurescript_wam_indexing,
          [test_clojurescript_wam_indexing/0]).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../../src/unifyweaver/targets/wam_clojure_target',
              [write_wam_clojurescript_files/3]).

test_clojurescript_wam_indexing :-
    run_tests([clojurescript_wam_indexing]).

nbb_path(Path) :-
    (   getenv('NBB', P), P \== ''
    ->  Path = P
    ;   absolute_file_name(path(nbb), Path, [access(execute), file_errors(fail)])
    ).

nbb_available :- catch(nbb_path(_), _, fail).

% --- the probe program ------------------------------------------------------

:- dynamic user:idx_c/2.
:- dynamic user:idx_all/2.
:- dynamic user:idx_any/1.
:- dynamic user:idx_len/2.
:- dynamic user:idx_s/2.
:- dynamic user:idx_cut/2.
:- dynamic user:idx_bt/2.

idx_predicates([idx_c/2, idx_all/2, idx_any/1, idx_len/2, idx_s/2,
                idx_cut/2, idx_bt/2]).

setup_idx_program :-
    forall(member(P/A, [idx_c/2, idx_all/2, idx_any/1, idx_len/2, idx_s/2,
                        idx_cut/2, idx_bt/2]),
           ( functor(H, P, A), retractall(user:H) )),
    % Constant dispatch with a REPEATED key and a trailing variable clause:
    % wam_target emits switch_on_constant_fallthrough `a:default b:L_..._2
    % a:L_..._3`, and only a first-match-wins read of that list gives SWI's
    % answer order.
    assertz(user:idx_c(a, 1)),
    assertz(user:idx_c(b, 2)),
    assertz(user:idx_c(a, 3)),
    assertz(user:idx_c(_, 9)),
    assertz(user:(idx_all(K, L) :- findall(N, user:idx_c(K, N), L))),
    assertz(user:(idx_any(L) :- findall(N, user:idx_c(_, N), L))),
    % switch_on_term: the [] / [H|T] split of every list walk.
    assertz(user:idx_len([], 0)),
    assertz(user:(idx_len([_|T], N) :- idx_len(T, M), N is M + 1)),
    % switch_on_structure: dispatch on the functor, and fail on one with no
    % entry.
    assertz(user:idx_s(f(X), sf(X))),
    assertz(user:idx_s(g(X), sg(X))),
    % An if-then-else over the indexed predicate: the condition backtracks
    % (1 fails the test, 3 passes) and the commit prunes what it left.
    assertz(user:(idx_cut(K, R) :- ( user:idx_c(K, V), V > 1 -> R = V ; R = none ))),
    % ... and the same pruning ITE run INSIDE a backtracking enumeration of
    % the indexed predicate.
    assertz(user:(idx_bt(K, L) :-
                      findall(P, (user:idx_c(K, V), idx_cut(K, W), P = V-W), L))).

% --- compile + run ----------------------------------------------------------

idx_output(Lines) :-
    setup_idx_program,
    tmp_file(cljs_wam_index, Base),
    atom_concat(Base, '_dir', Dir),
    atomic_list_concat([Dir, '/generated/probe'], NsDir),
    make_directory_path(NsDir),
    idx_predicates(Preds),
    setup_call_cleanup(
        ( write_wam_clojurescript_files(Preds, [namespace('generated.probe')], NsDir),
          atomic_list_concat([NsDir, '/.wamclj_tmp'], TmpProj),
          catch(delete_directory_and_contents(TmpProj), _, true),
          atomic_list_concat([Dir, '/driver.cljs'], Driver),
          idx_driver_source(Src),
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

%% idx_driver_source(-Source)
%  Builds WAM terms, calls the generated `<pred>-state` wrappers and prints one
%  `name=value` line per probe. No probe logic of its own.
idx_driver_source(
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

;; order preservation: `a` is a REPEATED switch key, ahead of a variable clause
(let [o (out) s (c/idx-all-state \"a\" o)] (line \"all_a\" s o))
(let [o (out) s (c/idx-all-state \"b\" o)] (line \"all_b\" s o))
(let [o (out) s (c/idx-all-state \"z\" o)] (line \"all_z\" s o))

;; a key built at RUN TIME, never seeded from the switch table's own text
(let [o (out) s (c/idx-all-state (str \"a\" \"\") o)] (line \"all_fresh_a\" s o))
(let [o (out) s (c/idx-s-state (st (str \"g\" \"/1\") [\"q\"]) o)] (line \"s_fresh_g\" s o))

;; an UNBOUND dispatch register must enumerate every clause
(let [o (out) s (c/idx-all-state {:var \"k\"} o)] (line \"all_unbound\" s o))
(let [o (out) s (c/idx-any-state o)] (line \"any\" s o))

;; switch_on_term's list label, and switch_on_structure's functor table
(let [o (out) s (c/idx-len-state (lst [\"p\" \"q\" \"r\"]) o)] (line \"len3\" s o))
(let [o (out) s (c/idx-len-state \"[]\" o)] (line \"len0\" s o))
(let [o (out) s (c/idx-s-state (st \"f/1\" [\"q\"]) o)] (line \"s_f\" s o))
(let [o (out) s (c/idx-s-state (st \"g/1\" [\"q\"]) o)] (line \"s_g\" s o))
(let [o (out) s (c/idx-s-state (st \"h/1\" [\"q\"]) o)] (line \"s_h\" s o))

;; a pruning if-then-else over the indexed predicate, and the same one run
;; inside a backtracking enumeration of it
(let [o (out) s (c/idx-cut-state \"a\" o)] (line \"cut_a\" s o))
(let [o (out) s (c/idx-bt-state \"a\" o)] (line \"bt_a\" s o))
").

% --- the probes -------------------------------------------------------------

:- begin_tests(clojurescript_wam_indexing).

idx_line(Lines, Key, Value) :-
    atomic_list_concat([Key, '='], Prefix),
    member(L, Lines),
    string_concat(Prefix, V, L),
    atom_string(Value, V),
    !.

% SWI, on the same program:
%   findall(N, idx_c(a,N), L) -> [1,3,9]   (clause 1, clause 3, the var clause)
%   findall(N, idx_c(b,N), L) -> [2,9]
%   findall(N, idx_c(z,N), L) -> [9]
test(order_preserved, [condition(nbb_available)]) :-
    idx_output(Lines),
    idx_line(Lines, all_a, A),
    idx_line(Lines, all_b, B),
    idx_line(Lines, all_z, Z),
    assertion(A == '[1,3,9]'),
    assertion(B == '[2,9]'),
    assertion(Z == '[9]').

test(fresh_atom_dispatch, [condition(nbb_available)]) :-
    idx_output(Lines),
    idx_line(Lines, all_fresh_a, A),
    idx_line(Lines, s_fresh_g, G),
    assertion(A == '[1,3,9]'),
    assertion(G == 'sg(q)').

test(unbound_first_arg, [condition(nbb_available)]) :-
    idx_output(Lines),
    idx_line(Lines, all_unbound, U),
    idx_line(Lines, any, Any),
    assertion(U == '[1,2,3,9]'),
    assertion(Any == '[1,2,3,9]').

test(list_and_structure, [condition(nbb_available)]) :-
    idx_output(Lines),
    idx_line(Lines, len3, L3),
    idx_line(Lines, len0, L0),
    idx_line(Lines, s_f, F),
    idx_line(Lines, s_g, G),
    idx_line(Lines, s_h, H),
    assertion(L3 == '3'),
    assertion(L0 == '0'),
    assertion(F == 'sf(q)'),
    assertion(G == 'sg(q)'),
    assertion(H == 'FAIL').

test(dispatch_across_cut, [condition(nbb_available)]) :-
    idx_output(Lines),
    idx_line(Lines, cut_a, C),
    idx_line(Lines, bt_a, B),
    assertion(C == '3'),
    assertion(B == '[-(1,3),-(3,3),-(9,3)]').

:- end_tests(clojurescript_wam_indexing).
