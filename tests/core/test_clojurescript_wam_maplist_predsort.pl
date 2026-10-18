% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_clojurescript_wam_maplist_predsort.pl -- P3 helpers on the Clojure WAM
% lane: maplist meta-calls a USER predicate, predsort's user comparator must
% not smash caller registers (D61), string_codes empty <-> [], and
% UW_WAM_WARN_UNKNOWN is loud when set.
%
%   swipl -g 'test_clojurescript_wam_maplist_predsort,halt' \
%         tests/core/test_clojurescript_wam_maplist_predsort.pl

:- module(test_clojurescript_wam_maplist_predsort,
          [test_clojurescript_wam_maplist_predsort/0]).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../../src/unifyweaver/targets/wam_clojure_target',
              [write_wam_clojurescript_files/3]).

test_clojurescript_wam_maplist_predsort :-
    run_tests([clojurescript_wam_maplist_predsort]).

nbb_path(Path) :-
    (   getenv('NBB', P), P \== ''
    ->  Path = P
    ;   absolute_file_name(path(nbb), Path, [access(execute), file_errors(fail)])
    ).

nbb_available :- catch(nbb_path(_), _, fail).

:- dynamic user:is_v3/1.
:- dynamic user:inc1/2.
:- dynamic user:cmp_n/3.
:- dynamic user:empty_codes/1.
:- dynamic user:gmapv3/0.
:- dynamic user:gmapdeb/0.
:- dynamic user:gmapinc/0.
:- dynamic user:gscodes/0.
:- dynamic user:gpredsort/0.
:- dynamic user:gpreduser/0.
:- dynamic user:gunknown/0.
:- dynamic user:gdiv/0.

setup_probe_program :-
    retractall(user:is_v3(_)),
    retractall(user:inc1(_, _)),
    retractall(user:cmp_n(_, _, _)),
    retractall(user:empty_codes(_)),
    retractall(user:gmapv3),
    retractall(user:gmapdeb),
    retractall(user:gmapinc),
    retractall(user:gscodes),
    retractall(user:gpredsort),
    retractall(user:gpreduser),
    retractall(user:gunknown),
    retractall(user:gdiv),
    assertz(user:(is_v3(v(_, _, _)))),
    assertz(user:(inc1(X, Y) :- Y is X + 1)),
    assertz(user:(cmp_n(<, A, B) :- A < B, !)),
    assertz(user:(cmp_n(>, A, B) :- B < A, !)),
    assertz(user:(cmp_n(=, _, _))),
    assertz(user:(empty_codes([]))),
    assertz(user:(gmapv3 :- maplist(is_v3, [v(1, 0, 0), v(2, 0, 0)]))),
    assertz(user:(gmapdeb :- maplist(is_v3, [deb(0, [], [])]))),
    assertz(user:(gmapinc :- maplist(inc1, [1, 2, 3], [2, 3, 4]))),
    assertz(user:(gscodes :- string_codes('', L), empty_codes(L))),
    assertz(user:(gpredsort :- predsort(compare, [v(2, 0, 0), v(1, 0, 0)],
                                       [v(1, 0, 0), v(2, 0, 0)]))),
    assertz(user:(gpreduser :- predsort(cmp_n, [3, 1, 2], [1, 2, 3]))),
    assertz(user:(gunknown :- totally_missing_pred(x))),
    % D61/P3 index_catalog: `NL is (N-1)//2` emits put_structure ///2.
    % A first-slash arity split drops that instruction to :raw (backtrack).
    assertz(user:(gdiv :- N is (3-1)//2, N =:= 1)).

probe_predicates([is_v3/1, inc1/2, cmp_n/3, empty_codes/1,
                  gmapv3/0, gmapdeb/0, gmapinc/0, gscodes/0,
                  gpredsort/0, gpreduser/0, gunknown/0, gdiv/0]).

probe_driver_source(
"(ns driver
  (:require [generated.mlprobe.core :as c]))

(defn line [k s] (println (str k \"=\" (boolean s))))

(line \"MAPV3\" (c/gmapv3-state))
(line \"MAPDEB\" (c/gmapdeb-state))
(line \"MAPINC\" (c/gmapinc-state))
(line \"SCODES\" (c/gscodes-state))
(line \"PREDSORT\" (c/gpredsort-state))
(line \"PREDUSER\" (c/gpreduser-state))
(line \"UNKNOWN\" (c/gunknown-state))
(line \"DIV\" (c/gdiv-state))
").

run_nbb(Classpath, Driver, Output, ErrStr) :-
    nbb_path(Nbb),
    format(atom(Cmd),
           'UW_WAM_WARN_UNKNOWN=1 ~w --classpath ~w ~w',
           [Nbb, Classpath, Driver]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(O)), stderr(pipe(E)), process(PID)]),
    read_string(O, _, Output),
    read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(PID, _).

probe_output(Lines, ErrStr) :-
    setup_probe_program,
    tmp_file(cljs_wam_ml, Base),
    atom_concat(Base, '_dir', Dir),
    atomic_list_concat([Dir, '/generated/mlprobe'], NsDir),
    make_directory_path(NsDir),
    probe_predicates(Preds),
    setup_call_cleanup(
        ( write_wam_clojurescript_files(Preds, [namespace('generated.mlprobe')], NsDir),
          atomic_list_concat([NsDir, '/.wamclj_tmp'], TmpProj),
          catch(delete_directory_and_contents(TmpProj), _, true),
          atomic_list_concat([Dir, '/driver.cljs'], Driver),
          probe_driver_source(Src),
          setup_call_cleanup(open(Driver, write, S), write(S, Src), close(S)) ),
        run_nbb(Dir, Driver, Out, ErrStr),
        catch(delete_directory_and_contents(Dir), _, true)),
    split_string(Out, "\n", " \r", Lines0),
    exclude(==(""), Lines0, Lines).

:- begin_tests(clojurescript_wam_maplist_predsort).

probe_line(Lines, Key, Value) :-
    atomic_list_concat([Key, '='], Prefix),
    member(L, Lines),
    string_concat(Prefix, V, L),
    atom_string(Value, V),
    !.

test(maplist_predsort_string_codes_warn, [condition(nbb_available)]) :-
    probe_output(Lines, Err),
    probe_line(Lines, 'MAPV3', true),
    probe_line(Lines, 'MAPDEB', false),
    probe_line(Lines, 'MAPINC', true),
    probe_line(Lines, 'SCODES', true),
    probe_line(Lines, 'PREDSORT', true),
    probe_line(Lines, 'PREDUSER', true),
    probe_line(Lines, 'UNKNOWN', false),
    probe_line(Lines, 'DIV', true),
    assertion(sub_string(Err, _, _, _, "[wam_clojure]")),
    assertion(sub_string(Err, _, _, _, "unresolved goal")).

:- end_tests(clojurescript_wam_maplist_predsort).
