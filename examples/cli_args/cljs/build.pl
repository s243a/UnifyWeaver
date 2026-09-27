:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% build.pl -- compile examples/cli_args/cli_args.pl into ONE ClojureScript
% namespace through UnifyWeaver's PATTERN lane
% (clojure_target's A3 whole-program lowering -> clojurescript_target's JS
% interop rewrite).
%
% Run it with build.sh; the output is cli_args.cljs next to this file.
%
% Why the clauses are re-asserted into `user` rather than the file consulted:
% the pattern compilers read a predicate's clauses with user:clause/2, and
% cli_args.pl is a module. Reading its terms and asserting the non-directive
% ones into `user` gives the compilers the same clause database SWI would run,
% without editing the frozen source file. Nothing is executed -- compiling a
% predicate never runs it.

:- use_module('../../../src/unifyweaver/targets/clojure_target', []).
:- use_module('../../../src/unifyweaver/targets/clojurescript_target', []).

%% load_into_user(+File)
load_into_user(File) :-
    setup_call_cleanup(open(File, read, S), load_terms(S), close(S)).

load_terms(S) :-
    read_term(S, T, []),
    (   T == end_of_file
    ->  true
    ;   ( T = (:- _) -> true ; assertz(user:T) ),
        load_terms(S)
    ).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [Src, Out|_]
    ->  true
    ;   Src = 'cli_args.pl', Out = 'generated/cli_args.cljs'
    ),
    load_into_user(Src),
    clojurescript_target:compile_module(
        [pred(parse_args, 2, facts)],
        [ namespace('generated.cli-args'),
          include_dependencies(true),
          runtime(nbb) ],
        Code),
    setup_call_cleanup(open(Out, write, W), write(W, Code), close(W)),
    clj_report(Code, Out).

%% clj_report(+Code, +Out) -- say out loud what landed.
clj_report(Code, Out) :-
    split_string(Code, "\n", "", Lines),
    include(clj_is_defn_line, Lines, DefnLines),
    length(DefnLines, N),
    (   sub_string(Code, _, _, _, "TODO: Implement")
    ->  format("build.pl: WARNING -- a predicate fell back to the stub defn, see ~w~n", [Out])
    ;   true
    ),
    format("build.pl: wrote ~w (~w defn forms)~n", [Out, N]).

clj_is_defn_line(L) :- sub_string(L, 0, _, _, "(defn ").
