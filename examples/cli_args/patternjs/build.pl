:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% build.pl -- compile examples/cli_args/cli_args.pl into ONE JavaScript module
% through UnifyWeaver's PATTERN lane (typescript_target -> vanilla_js_target).
%
% Run it with build.sh; the output is cliArgs.generated.mjs next to this file.
%
% Why the clauses are re-asserted into `user` rather than the file consulted:
% the pattern compilers read a predicate's clauses with user:clause/2, and
% cli_args.pl is a module. Reading its terms and asserting the non-directive ones
% into `user` gives the compilers the same clause database SWI would run, without
% editing the frozen source file. Nothing is executed -- G-A3-8's rule is that
% compiling a predicate never runs it.

:- use_module('../../../src/unifyweaver/targets/typescript_target', []).
:- use_module('../../../src/unifyweaver/targets/vanilla_js_target', []).

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
    ;   Src = 'cli_args.pl', Out = 'cliArgs.generated.mjs'
    ),
    load_into_user(Src),
    vanilla_js_target:compile_module(
        [pred(parse_args, 2, facts)],
        [module_name('CliArgs'), include_dependencies(true)],
        Code),
    setup_call_cleanup(open(Out, write, W), write(W, Code), close(W)),
    ts_report(Code, Out).

%% ts_report(+Code, +Out) -- say out loud what landed.
ts_report(Code, Out) :-
    split_string(Code, "\n", "", Lines),
    include(ts_is_fn_line, Lines, FnLines),
    length(FnLines, N),
    (   sub_string(Code, _, _, _, "WARNING")
    ->  format("build.pl: WARNING -- some predicates were omitted, see ~w~n", [Out])
    ;   true
    ),
    (   sub_string(Code, _, _, _, "incomplete lowering")
    ->  format("build.pl: WARNING -- a goal was dropped, see ~w~n", [Out])
    ;   true
    ),
    format("build.pl: wrote ~w (~w function declarations)~n", [Out, N]).

ts_is_fn_line(L) :-
    ( sub_string(L, 0, _, _, "export function ") ; sub_string(L, 0, _, _, "function ") ).
