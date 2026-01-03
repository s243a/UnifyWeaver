:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% benchmark.pl - Performance benchmarks for incremental compilation
%
% Measures the speedup from cache hits vs fresh compilation.

:- module(benchmark, [
    run_benchmark/0,
    benchmark_cache_speedup/0,
    benchmark_multi_target/0
]).

:- use_module(library(lists)).

:- use_module(incremental_compiler, [
    compile_incremental/4,
    clear_incremental_cache/0
]).

:- use_module('../core/stream_compiler').
:- use_module('../targets/go_target').

% ============================================================================
% MAIN BENCHMARK RUNNER
% ============================================================================

run_benchmark :-
    writeln('=== INCREMENTAL COMPILATION BENCHMARKS ==='),
    writeln(''),

    % Setup
    setup_benchmark_predicates,

    % Run benchmarks
    benchmark_cache_speedup,
    benchmark_multi_target,

    % Cleanup
    cleanup_benchmark_predicates,
    clear_incremental_cache,

    writeln(''),
    writeln('=== BENCHMARKS COMPLETE ===').

% ============================================================================
% BENCHMARK SETUP
% ============================================================================

setup_benchmark_predicates :-
    % Create test predicates of varying complexity
    catch(abolish(user:bench_simple/2), _, true),
    catch(abolish(user:bench_medium/3), _, true),
    catch(abolish(user:bench_complex/2), _, true),

    % Simple facts
    forall(between(1, 50, I),
        (   atom_concat('item', I, Item),
            assertz(user:bench_simple(Item, I))
        )),

    % Medium complexity with rules
    assertz(user:(bench_medium(X, Y, Z) :- bench_simple(X, Y), Z is Y * 2)),
    assertz(user:(bench_medium(X, Y, Z) :- bench_simple(X, Y), Y > 25, Z is Y - 25)),

    % Complex with multiple body goals
    assertz(user:(bench_complex(X, Result) :-
        bench_simple(X, V1),
        bench_medium(X, V1, V2),
        Result is V1 + V2)).

cleanup_benchmark_predicates :-
    catch(abolish(user:bench_simple/2), _, true),
    catch(abolish(user:bench_medium/3), _, true),
    catch(abolish(user:bench_complex/2), _, true).

% ============================================================================
% BENCHMARK: CACHE SPEEDUP
% ============================================================================

benchmark_cache_speedup :-
    writeln('Benchmark 1: Cache Hit Speedup'),
    writeln('------------------------------'),

    clear_incremental_cache,

    % Measure cold compilation (cache miss)
    writeln('  Cold compilation (cache miss):'),
    get_time(Start1),
    compile_incremental(bench_simple/2, bash, [], _Code1),
    get_time(End1),
    ColdTime is (End1 - Start1) * 1000,
    format('    Time: ~3f ms~n', [ColdTime]),

    % Measure warm compilation (cache hit)
    writeln('  Warm compilation (cache hit):'),
    get_time(Start2),
    compile_incremental(bench_simple/2, bash, [], _Code2),
    get_time(End2),
    WarmTime is (End2 - Start2) * 1000,
    format('    Time: ~3f ms~n', [WarmTime]),

    % Calculate speedup
    (   WarmTime > 0.001
    ->  Speedup is ColdTime / WarmTime,
        format('  Speedup: ~2fx faster~n', [Speedup])
    ;   writeln('  Speedup: Cache hit effectively instant')
    ),

    writeln('').

% ============================================================================
% BENCHMARK: MULTI-TARGET COMPILATION
% ============================================================================

benchmark_multi_target :-
    writeln('Benchmark 2: Multi-Target Compilation'),
    writeln('-------------------------------------'),

    clear_incremental_cache,

    % Compile same predicate to multiple targets
    Targets = [bash, go],

    writeln('  First compilation (cold):'),
    get_time(Start1),
    forall(member(Target, Targets),
        compile_incremental(bench_simple/2, Target, [], _)),
    get_time(End1),
    ColdTime is (End1 - Start1) * 1000,
    format('    Total time for ~w targets: ~3f ms~n', [Targets, ColdTime]),

    writeln('  Second compilation (warm):'),
    get_time(Start2),
    forall(member(Target, Targets),
        compile_incremental(bench_simple/2, Target, [], _)),
    get_time(End2),
    WarmTime is (End2 - Start2) * 1000,
    format('    Total time for ~w targets: ~3f ms~n', [Targets, WarmTime]),

    (   WarmTime > 0.001
    ->  Speedup is ColdTime / WarmTime,
        format('  Speedup: ~2fx faster~n', [Speedup])
    ;   writeln('  Speedup: Cache hits effectively instant')
    ),

    writeln('').
