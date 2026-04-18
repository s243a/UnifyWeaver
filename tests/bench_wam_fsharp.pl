:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% bench_wam_fsharp.pl — Benchmark suite for WAM-to-F# compilation
%
% Compares compile_wam_predicate_to_fsharp/4 (interpreter mode) vs
% lowered emitter (wam_fsharp_lowered_emitter) compilation throughput
% on three standard WAM benchmarks: append/3, fib/2, member/2.
%
% Measures Prolog-side compilation time (how long it takes to generate
% the F# code), not F# runtime execution.
%
% Usage: swipl -g run_benchmarks -t halt tests/bench_wam_fsharp.pl

:- module(bench_wam_fsharp, [run_benchmarks/0]).

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').

% ============================================================================
% Top-level entry
% ============================================================================

run_benchmarks :-
	format("~n========================================~n"),
	format("  WAM-to-F# Compilation Benchmark~n"),
	format("========================================~n~n"),
	bench_append,
	bench_fibonacci,
	bench_member,
	format("~n========================================~n"),
	format("  Benchmark complete~n"),
	format("========================================~n~n").

% ============================================================================
% Benchmark 1: append/3
% ============================================================================
%
% append([], Y, Y).
% append([H|T], Y, [H|R]) :- append(T, Y, R).

bench_append :-
	format("=== append/3 benchmark ===~n"),
	N = 10000,

	% --- Interpreter mode (compile_wam_predicate_to_fsharp/4) ---
	AppendWam = 'append/3:\n  get_nil 1\n  get_value 3, 2\n  proceed',
	time_compilation(append/3, AppendWam, N, InterpMs),
	format("  compile_wam_predicate_to_fsharp (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Lowered mode (wam_fsharp_lowerable check) ---
	time_lowerable_check(append/3, AppendWam, N, LoweredMs),
	format("  wam_fsharp_lowerable check (~w iterations): ~w ms~n", [N, LoweredMs]),

	% --- Summary ---
	(   LoweredMs > 0
	->  Speedup is InterpMs / max(LoweredMs, 1),
	    format("  Speedup: ~2fx~n~n", [Speedup])
	;   format("  Speedup: N/A (lowered too fast to measure)~n~n")
	).

% ============================================================================
% Benchmark 2: fibonacci/2
% ============================================================================
%
% fib(0, 0).
% fib(1, 1).
% fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2.

bench_fibonacci :-
	format("=== fibonacci/2 benchmark ===~n"),
	N = 10000,

	% --- Interpreter mode ---
	FibWam = 'fib/2:\n  get_integer 0, 1\n  get_integer 0, 2\n  proceed',
	time_compilation(fib/2, FibWam, N, InterpMs),
	format("  compile_wam_predicate_to_fsharp (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Lowered mode ---
	time_lowerable_check(fib/2, FibWam, N, LoweredMs),
	format("  wam_fsharp_lowerable check (~w iterations): ~w ms~n", [N, LoweredMs]),

	% --- Summary ---
	(   LoweredMs > 0
	->  Speedup is InterpMs / max(LoweredMs, 1),
	    format("  Speedup: ~2fx~n~n", [Speedup])
	;   format("  Speedup: N/A~n~n")
	).

% ============================================================================
% Benchmark 3: member/2
% ============================================================================
%
% member(X, [X|_]).
% member(X, [_|T]) :- member(X, T).

bench_member :-
	format("=== member/2 benchmark ===~n"),
	N = 10000,

	% --- Interpreter mode ---
	MemberWam = 'member/2:\n  get_list 2\n  unify_value 1\n  unify_void 1\n  proceed',
	time_compilation(member/2, MemberWam, N, InterpMs),
	format("  compile_wam_predicate_to_fsharp (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Lowered mode ---
	time_lowerable_check(member/2, MemberWam, N, LoweredMs),
	format("  wam_fsharp_lowerable check (~w iterations): ~w ms~n", [N, LoweredMs]),

	% --- Summary ---
	(   LoweredMs > 0
	->  Speedup is InterpMs / max(LoweredMs, 1),
	    format("  Speedup: ~2fx~n~n", [Speedup])
	;   format("  Speedup: N/A (lowered too fast to measure)~n~n")
	).

% ============================================================================
% Timing helpers
% ============================================================================

%% time_compilation(+PredIndicator, +WamCode, +N, -Ms)
%  Time N iterations of compile_wam_predicate_to_fsharp/4.
time_compilation(PredIndicator, WamCode, N, Ms) :-
	get_time(T0),
	forall(
		between(1, N, _),
		(   compile_wam_predicate_to_fsharp(PredIndicator, WamCode, [], _Code)
		->  true
		;   true
		)
	),
	get_time(T1),
	Ms is round((T1 - T0) * 1000).

%% time_lowerable_check(+PredIndicator, +WamCode, +N, -Ms)
%  Time N iterations of wam_fsharp_lowerable/3 check.
time_lowerable_check(PredIndicator, WamCode, N, Ms) :-
	get_time(T0),
	forall(
		between(1, N, _),
		(   wam_fsharp_lowerable(PredIndicator, WamCode, _Reason)
		->  true
		;   true
		)
	),
	get_time(T1),
	Ms is round((T1 - T0) * 1000).
