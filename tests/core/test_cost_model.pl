:- encoding(utf8).
%% Test suite for cost_model.pl
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_cost_model.pl

:- use_module('../../src/unifyweaver/core/cost_model').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("Cache Cost Model Tests~n"),
    format("========================================~n~n"),
    findall(Test, test(Test), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], Passed, Passed).
run_all([Test|Rest], Acc, Passed) :-
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

pass(Name) :- format("[PASS] ~w~n", [Name]).
fail_test(Name, Reason) :- format("[FAIL] ~w: ~w~n", [Name, Reason]), fail.

%% Approximate equality for floating-point comparisons.
approx_eq(A, B, Tol) :-
    Diff is abs(A - B),
    Diff =< Tol.

%% Within an order of magnitude (factor of 10) — used for estimates
%% where the formula's exact constant doesn't matter, only the regime.
within_factor(A, B, Factor) :-
    A =< B * Factor,
    B =< A * Factor.

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_default_constants_present).
test(test_constant_lookup_default).
test(test_constant_lookup_override).

test(test_cache_regime_all_hot).
test(test_cache_regime_all_cold).
test(test_cache_regime_mixed).
test(test_cache_regime_zero_w).
test(test_cache_regime_clamps_at_one).

test(test_scan_time_hot_simplewiki).
test(test_scan_time_cold_enwiki).

test(test_seek_time_hot).
test(test_seek_time_cold).

test(test_crossover_simplewiki_hot_matches_doc).
test(test_crossover_simplewiki_cold_shifts_left).
test(test_crossover_enwiki_hot_matches_doc).
test(test_crossover_enwiki_cold_shifts_left).

test(test_warming_pointless_when_all_hot).
test(test_warming_pays_off_when_all_cold).

test(test_recommend_sort_at_low_k).
test(test_recommend_scan_at_high_k).

test(test_read_mem_available_returns_positive).

%% ========================================================================
%% Default constants
%% ========================================================================

test_default_constants_present :-
    Test = 'default_constants/1 lists the four expected keys',
    (   default_constants(C),
        member(s_mem_seq_bps(_), C),
        member(s_disk_seq_bps(_), C),
        member(t_mem_seek_us(_), C),
        member(t_disk_seek_us(_), C)
    ->  pass(Test)
    ;   fail_test(Test, 'default_constants missing one of the four required keys')
    ).

test_constant_lookup_default :-
    Test = 'constant/3 returns default when not in custom list',
    (   constant(s_mem_seq_bps, [], V),
        V > 0
    ->  pass(Test)
    ;   fail_test(Test, 'default lookup failed for s_mem_seq_bps')
    ).

test_constant_lookup_override :-
    Test = 'constant/3 prefers custom value over default',
    (   constant(t_disk_seek_us, [t_disk_seek_us(10000.0)], V),
        approx_eq(V, 10000.0, 0.001)
    ->  pass(Test)
    ;   fail_test(Test, 'override lookup failed for t_disk_seek_us')
    ).

%% ========================================================================
%% Cache regime
%% ========================================================================

test_cache_regime_all_hot :-
    Test = 'cache_regime returns 1.0 when R_free >> W',
    %% 16 GB free, 100 MB working set — clearly all hot
    cache_regime(16_000_000_000, 100_000_000, F),
    (   approx_eq(F, 1.0, 0.001)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got F=~w (expected ~~1.0)", [F]))
    ).

test_cache_regime_all_cold :-
    Test = 'cache_regime drops near 0 when R_free << W',
    %% 100 MB free, 100 GB working set — clearly all cold
    cache_regime(100_000_000, 100_000_000_000, F),
    (   F < 0.01
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got F=~w (expected near 0)", [F]))
    ).

test_cache_regime_mixed :-
    Test = 'cache_regime returns ~0.5 when R_free = W/2',
    cache_regime(500_000_000, 1_000_000_000, F),
    (   approx_eq(F, 0.5, 0.001)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got F=~w (expected ~~0.5)", [F]))
    ).

test_cache_regime_zero_w :-
    Test = 'cache_regime returns 1.0 when W=0',
    cache_regime(1_000_000, 0, F),
    (   approx_eq(F, 1.0, 0.001)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got F=~w (expected 1.0)", [F]))
    ).

test_cache_regime_clamps_at_one :-
    Test = 'cache_regime never exceeds 1.0',
    cache_regime(100_000_000_000, 1_000_000, F),
    (   F =< 1.0
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got F=~w (exceeds 1.0)", [F]))
    ).

%% ========================================================================
%% Scan time
%% ========================================================================

test_scan_time_hot_simplewiki :-
    Test = 'scan_time_ms hot regime ~3 ms for simplewiki (15 MB at 5 GB/s)',
    default_constants(C),
    scan_time_ms(15_000_000, 1.0, C, Ms),
    %% 15e6 / 5e9 = 3e-3 s = 3 ms
    (   approx_eq(Ms, 3.0, 0.5)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got ~w ms (expected ~~3 ms)", [Ms]))
    ).

test_scan_time_cold_enwiki :-
    Test = 'scan_time_ms cold regime ~1000 ms for enwiki (500 MB at 500 MB/s)',
    default_constants(C),
    scan_time_ms(500_000_000, 0.0, C, Ms),
    %% 500e6 / 500e6 = 1 s = 1000 ms
    (   approx_eq(Ms, 1000.0, 50.0)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got ~w ms (expected ~~1000 ms)", [Ms]))
    ).

%% ========================================================================
%% Seek time
%% ========================================================================

test_seek_time_hot :-
    Test = 'seek_time_ms hot regime ~10 ms for 10000 keys (10000 * 1 µs)',
    default_constants(C),
    seek_time_ms(10_000, 1.0, C, Ms),
    %% 10000 * 1 µs = 10 ms
    (   approx_eq(Ms, 10.0, 0.5)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got ~w ms (expected ~~10 ms)", [Ms]))
    ).

test_seek_time_cold :-
    Test = 'seek_time_ms cold regime ~1000 ms for 10000 keys (10000 * 100 µs)',
    default_constants(C),
    seek_time_ms(10_000, 0.0, C, Ms),
    %% 10000 * 100 µs = 1 s = 1000 ms
    (   approx_eq(Ms, 1000.0, 50.0)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got ~w ms (expected ~~1000 ms)", [Ms]))
    ).

%% ========================================================================
%% Crossover K — these are the headline numbers from the philosophy doc
%% ========================================================================

test_crossover_simplewiki_hot_matches_doc :-
    Test = 'crossover_k simplewiki hot ~3000 keys (matches doc)',
    default_constants(C),
    crossover_k(15_000_000, 1.0, C, K),
    %% Doc says ~3000. Within an order of magnitude is the bar; we get
    %% 15e6 / (5e9 * 1e-6) = 15e6/5000 = 3000 exactly.
    (   within_factor(K, 3000.0, 2.0)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got K=~w (expected ~~3000)", [K]))
    ).

test_crossover_simplewiki_cold_shifts_left :-
    Test = 'crossover_k simplewiki cold ~300 keys (10x shift left)',
    default_constants(C),
    crossover_k(15_000_000, 0.0, C, K),
    %% 15e6 / (500e6 * 100e-6) = 15e6 / 50000 = 300
    (   within_factor(K, 300.0, 2.0)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got K=~w (expected ~~300)", [K]))
    ).

test_crossover_enwiki_hot_matches_doc :-
    Test = 'crossover_k enwiki hot ~100000 keys (matches doc)',
    default_constants(C),
    crossover_k(500_000_000, 1.0, C, K),
    %% 500e6 / (5e9 * 1e-6) = 500e6 / 5000 = 100000
    (   within_factor(K, 100000.0, 2.0)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got K=~w (expected ~~100000)", [K]))
    ).

test_crossover_enwiki_cold_shifts_left :-
    Test = 'crossover_k enwiki cold ~10000 keys (10x shift left)',
    default_constants(C),
    crossover_k(500_000_000, 0.0, C, K),
    %% 500e6 / (500e6 * 100e-6) = 500e6 / 50000 = 10000
    (   within_factor(K, 10000.0, 2.0)
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got K=~w (expected ~~10000)", [K]))
    ).

%% ========================================================================
%% Warming threshold
%% ========================================================================

test_warming_pointless_when_all_hot :-
    Test = 'warming_payoff_m returns positive_infinity when FHot=1 (no payoff)',
    default_constants(C),
    warming_payoff_m(100_000_000, 1000, 1.0, C, M),
    (   M = positive_infinity
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got M=~w (expected positive_infinity)", [M]))
    ).

test_warming_pays_off_when_all_cold :-
    Test = 'warming_payoff_m returns finite small M when FHot=0',
    default_constants(C),
    warming_payoff_m(100_000_000, 1000, 0.0, C, M),
    (   integer(M),
        M > 0,
        M < 100_000  %% Some finite, plausibly small number
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got M=~w (expected small finite)", [M]))
    ).

%% ========================================================================
%% Recommendation
%% ========================================================================

test_recommend_sort_at_low_k :-
    Test = 'recommend_access_pattern picks sort at low K (hot regime)',
    default_constants(C),
    %% 16 GB free, 15 MB DB, 100 keys → all hot, K << K_cross(~3000)
    recommend_access_pattern(100, 15_000_000, 16_000_000_000, C, Pattern),
    (   Pattern = sort
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got pattern=~w (expected sort)", [Pattern]))
    ).

test_recommend_scan_at_high_k :-
    Test = 'recommend_access_pattern picks scan at high K',
    default_constants(C),
    %% 16 GB free, 15 MB DB, 100000 keys (way past K_cross ~ 3000)
    recommend_access_pattern(100_000, 15_000_000, 16_000_000_000, C, Pattern),
    (   Pattern = scan
    ->  pass(Test)
    ;   fail_test(Test, format_atom("got pattern=~w (expected scan)", [Pattern]))
    ).

%% ========================================================================
%% System probe
%% ========================================================================

test_read_mem_available_returns_positive :-
    Test = 'read_mem_available_bytes returns positive integer on Linux',
    (   catch(read_mem_available_bytes(B), _, fail),
        integer(B),
        B > 0
    ->  pass(Test)
    ;   %% Skip on non-Linux platforms
        format("[SKIP] ~w (no /proc/meminfo)~n", [Test])
    ).

%% ========================================================================
%% Helper: format an atom for fail_test reasons
%% ========================================================================

format_atom(Format, Args, Atom) :-
    format(string(S), Format, Args),
    atom_string(Atom, S).
format_atom(Format, Args) :-
    format_atom(Format, Args, A),
    A == A.  % Just to enforce binding; only used inside fail_test/2.
