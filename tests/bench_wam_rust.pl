:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% bench_wam_rust.pl — Benchmark suite for WAM-to-Rust compilation
%
% Compares compile_wam_predicate_to_rust/4 (interpreter mode) vs
% wam_rust_lowerable/3 check + lower_predicate_to_rust/4 (lowered mode)
% compilation throughput on three standard WAM benchmarks:
% append/3, fib/2, member/2.
%
% Measures Prolog-side compilation time (how long it takes to generate
% the Rust code), not Rust runtime execution.
%
% Usage: swipl -g run_benchmarks -t halt tests/bench_wam_rust.pl

:- module(bench_wam_rust, [run_benchmarks/0]).

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/wam_rust_lowered_emitter').

% ============================================================================
% Top-level entry
% ============================================================================

run_benchmarks :-
    format("~n========================================~n"),
    format("  WAM-to-Rust Compilation Benchmark~n"),
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

bench_append :-
    format("=== append/3 benchmark ===~n"),
    N = 10000,

    % --- Interpreter mode (compile_wam_predicate_to_rust/4) ---
    AppendWam = 'append/3:\n  get_constant [], A1\n  get_value A3, A2\n  proceed',
    time_compilation(append/3, AppendWam, N, InterpMs),
    format("  compile_wam_predicate_to_rust (~w iterations): ~w ms~n", [N, InterpMs]),

    % --- Lowered mode check ---
    AppendWamLowered = 'append/3:\n  get_constant [], A1\n  get_value A3, A2\n  proceed',
    time_lowerable_check(append/3, AppendWamLowered, N, LowerMs),
    format("  wam_rust_lowerable check (~w iterations): ~w ms~n", [N, LowerMs]),

    % --- Summary ---
    (   LowerMs > 0
    ->  Speedup is InterpMs / max(LowerMs, 1),
        format("  Speedup: ~2fx~n~n", [Speedup])
    ;   format("  Speedup: N/A (lowerable check too fast to measure)~n~n")
    ).

% ============================================================================
% Benchmark 2: fibonacci/2
% ============================================================================

bench_fibonacci :-
    format("=== fibonacci/2 benchmark ===~n"),
    N = 10000,

    % --- Interpreter mode ---
    FibWam = 'fib/2:\n  get_constant 0, A1\n  get_constant 0, A2\n  proceed',
    time_compilation(fib/2, FibWam, N, InterpMs),
    format("  compile_wam_predicate_to_rust (~w iterations): ~w ms~n", [N, InterpMs]),

    % --- Lowered mode ---
    time_lowerable_check(fib/2, FibWam, N, LowerMs),
    format("  wam_rust_lowerable check (~w iterations): ~w ms~n", [N, LowerMs]),

    % --- Summary ---
    (   LowerMs > 0
    ->  Speedup is InterpMs / max(LowerMs, 1),
        format("  Speedup: ~2fx~n~n", [Speedup])
    ;   format("  Speedup: N/A~n~n")
    ).

% ============================================================================
% Benchmark 3: member/2
% ============================================================================

bench_member :-
    format("=== member/2 benchmark ===~n"),
    N = 10000,

    % --- Interpreter mode ---
    MemberWam = 'member/2:\n  get_list A2\n  unify_value A1\n  proceed',
    time_compilation(member/2, MemberWam, N, InterpMs),
    format("  compile_wam_predicate_to_rust (~w iterations): ~w ms~n", [N, InterpMs]),

    % --- Lowered mode ---
    time_lowerable_check(member/2, MemberWam, N, LowerMs),
    format("  wam_rust_lowerable check (~w iterations): ~w ms~n", [N, LowerMs]),

    % --- Summary ---
    (   LowerMs > 0
    ->  Speedup is InterpMs / max(LowerMs, 1),
        format("  Speedup: ~2fx~n~n", [Speedup])
    ;   format("  Speedup: N/A (too fast to measure)~n~n")
    ).

% ============================================================================
% Timing helpers
% ============================================================================

%% time_compilation(+PredIndicator, +WamCode, +N, -Ms)
%  Time N iterations of compile_wam_predicate_to_rust/4.
time_compilation(PredIndicator, WamCode, N, Ms) :-
    get_time(T0),
    forall(
        between(1, N, _),
        (   compile_wam_predicate_to_rust(PredIndicator, WamCode, [], _Code)
        ->  true
        ;   true
        )
    ),
    get_time(T1),
    Ms is round((T1 - T0) * 1000).

%% time_lowerable_check(+PredIndicator, +WamCode, +N, -Ms)
%  Time N iterations of wam_rust_lowerable/3 + lower_predicate_to_rust/4.
time_lowerable_check(PredIndicator, WamCode, N, Ms) :-
    get_time(T0),
    forall(
        between(1, N, _),
        (   wam_rust_lowerable(PredIndicator, WamCode, _Reason)
        ->  lower_predicate_to_rust(PredIndicator, WamCode, [], _Lines)
        ;   true
        )
    ),
    get_time(T1),
    Ms is round((T1 - T0) * 1000).
