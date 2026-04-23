:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% bench_wam_go.pl — Benchmark suite for WAM-to-Go compilation
%
% Compares compile_wam_predicate_to_go/4 (interpreter mode) vs
% direct wam_instruction_to_go_literal/2 (literal generation)
% compilation throughput on three standard WAM benchmarks:
% append/3, fib/2, member/2.
%
% Measures Prolog-side compilation time (how long it takes to generate
% the Go code), not Go runtime execution.
%
% Usage: swipl -g run_benchmarks -t halt tests/bench_wam_go.pl

:- module(bench_wam_go, [run_benchmarks/0]).

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

% ============================================================================
% Top-level entry
% ============================================================================

run_benchmarks :-
	format("~n========================================~n"),
	format("  WAM-to-Go Compilation Benchmark~n"),
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

	% --- Interpreter mode (compile_wam_predicate_to_go/4) ---
	AppendWam = 'append/3:\n  get_nil 1\n  get_value 3, 2\n  proceed',
	time_compilation(append/3, AppendWam, N, InterpMs),
	format("  compile_wam_predicate_to_go (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Literal mode (wam_instruction_to_go_literal/2) ---
	AppendInstrs = [
		get_constant(atom([]), a(1)),
		get_value(a(3), a(2)),
		proceed
	],
	time_literal(AppendInstrs, N, LiteralMs),
	format("  wam_instruction_to_go_literal (~w iterations): ~w ms~n", [N, LiteralMs]),

	% --- Summary ---
	(   LiteralMs > 0
	->  Speedup is InterpMs / max(LiteralMs, 1),
	    format("  Speedup: ~2fx~n~n", [Speedup])
	;   format("  Speedup: N/A (literal too fast to measure)~n~n")
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
	format("  compile_wam_predicate_to_go (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Literal mode (base case) ---
	FibInstrs = [
		get_constant(integer(0), a(1)),
		get_constant(integer(0), a(2)),
		proceed
	],
	time_literal(FibInstrs, N, LiteralMs),
	format("  wam_instruction_to_go_literal (~w iterations): ~w ms~n", [N, LiteralMs]),

	% --- Summary ---
	(   LiteralMs > 0
	->  Speedup is InterpMs / max(LiteralMs, 1),
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
	format("  compile_wam_predicate_to_go (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Literal mode (clause 1) ---
	MemberInstrs = [
		get_list(a(2)),
		unify_value(a(1)),
		proceed
	],
	time_literal(MemberInstrs, N, LiteralMs),
	format("  wam_instruction_to_go_literal (~w iterations): ~w ms~n", [N, LiteralMs]),

	% --- Summary ---
	(   LiteralMs > 0
	->  Speedup is InterpMs / max(LiteralMs, 1),
	    format("  Speedup: ~2fx~n~n", [Speedup])
	;   format("  Speedup: N/A (literal too fast to measure)~n~n")
	).

% ============================================================================
% Timing helpers
% ============================================================================

%% time_compilation(+PredIndicator, +WamCode, +N, -Ms)
%  Time N iterations of compile_wam_predicate_to_go/4.
time_compilation(PredIndicator, WamCode, N, Ms) :-
	get_time(T0),
	forall(
		between(1, N, _),
		(   compile_wam_predicate_to_go(PredIndicator, WamCode, [], _Code)
		->  true
		;   true
		)
	),
	get_time(T1),
	Ms is round((T1 - T0) * 1000).

%% time_literal(+Instrs, +N, -Ms)
%  Time N iterations of wam_instruction_to_go_literal/2.
time_literal(Instrs, N, Ms) :-
	get_time(T0),
	forall(
		between(1, N, _),
		forall(
			member(Instr, Instrs),
			(   wam_instruction_to_go_literal(Instr, _GoLit)
			->  true
			;   true
			)
		)
	),
	get_time(T1),
	Ms is round((T1 - T0) * 1000).
