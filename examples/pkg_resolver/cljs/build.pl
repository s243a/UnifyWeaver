:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% build.pl -- compile the FROZEN examples/pkg_resolver/resolver.pl into ONE
% ClojureScript pair of namespaces (generated.resolver.core +
% generated.resolver.runtime) through UnifyWeaver's WAM lane of the
% ClojureScript target:
%
%   resolver.pl
%     -> wam_target            (WAM instructions, one shared instruction table)
%     -> wam_clojure_target    (Clojure instruction data + lowered prefixes)
%     -> clojurescript_target  (JVM host interop -> JS host interop)
%     -> generated/resolver/{core,runtime}.cljs, loadable by nbb
%
% The PATTERN lane (clojure_target's A3 whole-program lowering, the lane the
% D40 argparser uses) cannot carry this program: see README.md, "Why the
% pattern lane stops here". resolver.pl BACKTRACKS, and A3 emits one
% deterministic function per predicate.
%
% Why the clauses are re-asserted into `user` rather than the file consulted:
% the WAM compiler reads a predicate's clauses with user:clause/2, and
% resolver.pl is a module. Reading its terms and asserting the non-directive
% ones into `user` gives the compiler the same clause database SWI would run,
% without editing the frozen source file. Nothing is executed -- compiling a
% predicate never runs it.
%
% Run it with build.sh.

:- use_module('../../../src/unifyweaver/targets/wam_clojure_target',
              [write_wam_clojurescript_files/3]).

%% load_into_user(+File, -Preds)
load_into_user(File, Preds) :-
    setup_call_cleanup(open(File, read, S), load_terms(S, [], Acc), close(S)),
    sort(Acc, Preds).

load_terms(S, Acc, Preds) :-
    read_term(S, T, []),
    (   T == end_of_file
    ->  Preds = Acc
    ;   T = (:- _)
    ->  load_terms(S, Acc, Preds)
    ;   pred_of_term(T, PA),
        assertz(user:T),
        load_terms(S, [PA|Acc], Preds)
    ).

pred_of_term((Head :- _), P/A) :- !, functor(Head, P, A).
pred_of_term(Head, P/A) :- functor(Head, P, A).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [Src, OutDir|_]
    ->  true
    ;   Src = '../resolver.pl',
        OutDir = 'generated/resolver'
    ),
    load_into_user(Src, Preds),
    length(Preds, N),
    format("build.pl: compiling ~w predicates from ~w~n", [N, Src]),
    write_wam_clojurescript_files(
        Preds,
        [ namespace('generated.resolver'),
          emit_mode(mixed) ],
        OutDir),
    format("build.pl: wrote ~w/core.cljs + ~w/runtime.cljs~n", [OutDir, OutDir]).
