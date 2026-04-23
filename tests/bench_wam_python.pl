:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% bench_wam_python.pl — Benchmark suite for WAM-to-Python compilation
%
% Compares interpreter mode (compile_wam_predicate_to_python/4) vs
% lowered mode (emit_lowered_python/4) compilation throughput on three
% standard WAM benchmarks: append/3, fib/2, member/2.
%
% Measures Prolog-side compilation time (how long it takes to generate
% the Python code), not Python runtime execution.
%
% Usage: swipl -g run_benchmarks -t halt tests/bench_wam_python.pl

:- module(bench_wam_python, [run_benchmarks/0]).

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_python_target').
:- use_module('../src/unifyweaver/targets/wam_python_lowered_emitter').

% ============================================================================
% Top-level entry
% ============================================================================

run_benchmarks :-
	format("~n========================================~n"),
	format("  WAM-to-Python Compilation Benchmark~n"),
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
%
% WAM instructions for clause 1 (base case):
%   get_nil 1
%   get_value 3, 2
%   proceed
%
% WAM instructions for clause 2 (recursive):
%   get_list 1
%   unify_variable 101
%   unify_variable 102
%   get_list 3
%   unify_value 101
%   unify_variable 103
%   put_value 102, 1
%   put_value 103, 3
%   execute append/3

bench_append :-
	format("=== append/3 benchmark ===~n"),
	N = 10000,

	% --- Interpreter mode ---
	AppendWam = 'append/3:\n  get_nil 1\n  get_value 3, 2\n  proceed',
	time_compilation(interpreter, append/3, AppendWam, N, InterpMs),
	format("  Interpreter mode (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Lowered mode (base case only — deterministic single clause) ---
	AppendLoweredInstrs = [get_nil("1"), get_value("3", "2"), proceed],
	time_lowered(append_base/3, AppendLoweredInstrs, N, LoweredMs),
	format("  Lowered mode (~w iterations):     ~w ms~n", [N, LoweredMs]),

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
%
% WAM for clause 1 (base case fib(0,0)):
%   get_integer 0, 1
%   get_integer 0, 2
%   proceed
%
% WAM for clause 2 (base case fib(1,1)):
%   get_integer 1, 1
%   get_integer 1, 2
%   proceed

bench_fibonacci :-
	format("=== fibonacci/2 benchmark ===~n"),
	N = 10000,

	% --- Interpreter mode ---
	FibWam = 'fib/2:\n  get_integer 0, 1\n  get_integer 0, 2\n  proceed',
	time_compilation(interpreter, fib/2, FibWam, N, InterpMs),
	format("  Interpreter mode (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Lowered mode (base case: deterministic) ---
	FibLoweredInstrs = [get_integer("0", "1"), get_integer("0", "2"), proceed],
	time_lowered(fib_base/2, FibLoweredInstrs, N, LoweredMs),
	format("  Lowered mode (~w iterations):     ~w ms~n", [N, LoweredMs]),

	% --- Lowered ITE mode (two base cases as ITE) ---
	FibITEInstrs = [
		get_integer("0", "1"), get_integer("0", "2"), proceed,
		get_integer("1", "1"), get_integer("1", "2"), proceed
	],
	time_lowered(fib_ite/2, FibITEInstrs, N, ITEMs),
	format("  Lowered ITE mode (~w iterations): ~w ms~n", [N, ITEMs]),

	% --- Summary ---
	(   LoweredMs > 0
	->  Speedup is InterpMs / max(LoweredMs, 1),
	    format("  Speedup (lowered vs interp): ~2fx~n", [Speedup])
	;   format("  Speedup (lowered vs interp): N/A~n")
	),
	(   ITEMs > 0
	->  SpeedupITE is InterpMs / max(ITEMs, 1),
	    format("  Speedup (ITE vs interp):     ~2fx~n~n", [SpeedupITE])
	;   format("  Speedup (ITE vs interp):     N/A~n~n")
	).

% ============================================================================
% Benchmark 3: member/2
% ============================================================================
%
% member(X, [X|_]).
% member(X, [_|T]) :- member(X, T).
%
% WAM for clause 1:
%   get_list 2
%   unify_value 1
%   unify_void 1
%   proceed
%
% WAM for clause 2 (recursive):
%   get_list 2
%   unify_void 1
%   unify_variable 102
%   put_value 102, 2
%   execute member/2

bench_member :-
	format("=== member/2 benchmark ===~n"),
	N = 10000,

	% --- Interpreter mode ---
	MemberWam = 'member/2:\n  get_list 2\n  unify_value 1\n  unify_void 1\n  proceed',
	time_compilation(interpreter, member/2, MemberWam, N, InterpMs),
	format("  Interpreter mode (~w iterations): ~w ms~n", [N, InterpMs]),

	% --- Lowered mode (clause 1 only — deterministic) ---
	MemberLoweredInstrs = [get_list("2"), unify_value("1"), unify_void("1"), proceed],
	time_lowered(member_base/2, MemberLoweredInstrs, N, LoweredMs),
	format("  Lowered mode (~w iterations):     ~w ms~n", [N, LoweredMs]),

	% --- Summary ---
	(   LoweredMs > 0
	->  Speedup is InterpMs / max(LoweredMs, 1),
	    format("  Speedup: ~2fx~n~n", [Speedup])
	;   format("  Speedup: N/A (lowered too fast to measure)~n~n")
	).

% ============================================================================
% Timing helpers
% ============================================================================

%% time_compilation(+Mode, +PredIndicator, +WamCode, +N, -Ms)
%  Time N iterations of interpreter-mode compilation.
time_compilation(interpreter, PredIndicator, WamCode, N, Ms) :-
	get_time(T0),
	forall(
		between(1, N, _),
		(   compile_wam_predicate_to_python(PredIndicator, WamCode, [], _Code)
		->  true
		;   true
		)
	),
	get_time(T1),
	Ms is round((T1 - T0) * 1000).

%% time_lowered(+PredIndicator, +Instrs, +N, -Ms)
%  Time N iterations of lowered-mode compilation.
time_lowered(PredIndicator, Instrs, N, Ms) :-
	get_time(T0),
	forall(
		between(1, N, _),
		(   emit_lowered_python(PredIndicator, Instrs, [], _Code)
		->  true
		;   true
		)
	),
	get_time(T1),
	Ms is round((T1 - T0) * 1000).
