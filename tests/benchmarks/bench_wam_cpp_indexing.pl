:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% bench_wam_cpp_indexing.pl — Microbenchmark for the WAM-cpp first-arg
% indexing work (PRs #2296-#2303). Generates three N-clause predicates
% with different dispatch shapes, builds a tight C++ inner loop for
% each, and reports per-call dispatch cost.
%
% Shapes compared (head-unification work is identical; only dispatch
% shape differs):
%   bench_idx(K, V)    — N constant-A1 facts        → switch_on_constant
%   bench_chain(K, V)  — N variable-A1 clauses with body unification
%                        → NO A1/A2 indexing, plain try_me_else chain
%                        (the pre-indexing baseline)
%   bench_mma(K, V)    — N constant-A1 + 1 trailing var clause
%                        → switch_on_constant_fallthrough (PR #2301)
%
% For each shape, three positions:
%   first : query hits clause 1     (best case for chain too)
%   last  : query hits clause N     (worst case for chain)
%   miss  : query matches no clause (best case for indexed,
%           worst-case linear scan for chain)
%
% Usage:
%   swipl -g run -t halt tests/benchmarks/bench_wam_cpp_indexing.pl
%   swipl -g "run([n_clauses(500), n_iters(2_000_000)])" -t halt \
%         tests/benchmarks/bench_wam_cpp_indexing.pl

:- module(bench_wam_cpp_indexing, [run/0, run/1]).

:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(option)).
:- use_module('../../src/unifyweaver/targets/wam_cpp_target').

n_clauses_default(100).
% 100k keeps the slowest bench (chain_last on 100 clauses, ~65us/call)
% well under 10s — total run ~15s. Override with run([n_iters(N)]).
n_iters_default(100_000).

run :- run([]).
run(Opts) :-
    (   option(n_clauses(N), Opts) -> true ; n_clauses_default(N) ),
    (   option(n_iters(M),   Opts) -> true ; n_iters_default(M) ),
    BenchDir = '/tmp/bench_wam_cpp_indexing',
    retractall_bench,
    generate_bench_preds(N),
    generate_loop_preds(N),
    write_bench_project(BenchDir, M),
    build_bench(BenchDir),
    run_bench(BenchDir, M, Results),
    print_results(Results, N, M).

retractall_bench :-
    retractall(user:bench_idx(_, _)),
    retractall(user:bench_chain(_, _)),
    retractall(user:bench_mma(_, _)),
    retractall(user:bench_loop_idx_first(_)),
    retractall(user:bench_loop_idx_last(_)),
    retractall(user:bench_loop_idx_miss(_)),
    retractall(user:bench_loop_chain_first(_)),
    retractall(user:bench_loop_chain_last(_)),
    retractall(user:bench_loop_chain_miss(_)),
    retractall(user:bench_loop_mma_first(_)),
    retractall(user:bench_loop_mma_last(_)),
    retractall(user:bench_loop_mma_miss(_)).

%% Assert N clauses each for the three shapes.
generate_bench_preds(N) :-
    forall(between(1, N, I), (
        atom_concat(k, I, K),
        atom_concat(v, I, V),
        assertz(user:bench_idx(K, V)),
        assertz((user:bench_chain(KK, VV) :- KK = K, VV = V)),
        assertz(user:bench_mma(K, V))
    )),
    assertz(user:bench_mma(_, default_val)).

%% Assert loop predicates that drive iters calls of each bench query.
generate_loop_preds(N) :-
    KFirst = k1,
    atom_concat(k, N, KLast),
    KMiss = unknown_key,
    make_hit_loop(bench_loop_idx_first, bench_idx, KFirst),
    make_hit_loop(bench_loop_idx_last,  bench_idx, KLast),
    make_miss_loop(bench_loop_idx_miss, bench_idx, KMiss),
    make_hit_loop(bench_loop_chain_first, bench_chain, KFirst),
    make_hit_loop(bench_loop_chain_last,  bench_chain, KLast),
    make_miss_loop(bench_loop_chain_miss, bench_chain, KMiss),
    make_hit_loop(bench_loop_mma_first, bench_mma, KFirst),
    make_hit_loop(bench_loop_mma_last,  bench_mma, KLast),
    make_miss_loop(bench_loop_mma_miss, bench_mma, KMiss).

%% make_hit_loop(+LoopName, +BenchPred, +Key)
%  Asserts the canonical "loop M times, calling BenchPred(Key, _)"
%  template. once/1 ensures we only measure first-answer dispatch.
make_hit_loop(LoopName, BenchPred, Key) :-
    HZ =.. [LoopName, 0],
    HN =.. [LoopName, Nv],
    HN1 =.. [LoopName, Nv1],
    BG =.. [BenchPred, Key, _AnonVar],
    assertz(user:(HZ :- !)),
    assertz(user:(HN :- once(BG), Nv1 is Nv - 1, HN1)).

%% make_miss_loop(+LoopName, +BenchPred, +Key)
%  Same shape but expect BenchPred(Key, _) to fail. We swallow the
%  failure so the loop continues.
make_miss_loop(LoopName, BenchPred, Key) :-
    HZ =.. [LoopName, 0],
    HN =.. [LoopName, Nv],
    HN1 =.. [LoopName, Nv1],
    BG =.. [BenchPred, Key, _AnonVar],
    assertz(user:(HZ :- !)),
    assertz(user:(HN :- ( BG -> true ; true ), Nv1 is Nv - 1, HN1)).

%% Write a WAM-cpp project + custom bench_main.cpp that prints
%% BENCH_RESULT <name> <ns_per_iter> for each loop predicate.
write_bench_project(Dir, Iters) :-
    write_wam_cpp_project([
        user:bench_idx/2,
        user:bench_chain/2,
        user:bench_mma/2,
        user:bench_loop_idx_first/1,
        user:bench_loop_idx_last/1,
        user:bench_loop_idx_miss/1,
        user:bench_loop_chain_first/1,
        user:bench_loop_chain_last/1,
        user:bench_loop_chain_miss/1,
        user:bench_loop_mma_first/1,
        user:bench_loop_mma_last/1,
        user:bench_loop_mma_miss/1
    ], [emit_main(false)], Dir),
    bench_main_cpp(Iters, MainCode),
    directory_file_path(Dir, 'cpp/bench_main.cpp', MainPath),
    write_text_file(MainPath, MainCode).

bench_main_cpp(_Iters, Code) :-
    Code = '// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by bench_wam_cpp_indexing.pl. Do not edit by hand.
#include "wam_runtime.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

double run_one(WamState& vm, const char* pred, long iters) {
    std::vector<Value> args;
    args.push_back(Value::Integer(iters));
    auto t0 = Clock::now();
    bool ok = vm.query(pred, args);
    auto t1 = Clock::now();
    if (!ok) {
        std::fprintf(stderr, "WARN: %s returned false\\n", pred);
    }
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return double(ns) / double(iters);
}

} // anonymous

int main(int argc, char** argv) {
    long iters = (argc > 1) ? std::strtol(argv[1], nullptr, 10) : 1000000;
    WamState vm;
    Program::apply_setup(vm);
    const char* benches[] = {
        "bench_loop_idx_first/1",   "bench_loop_idx_last/1",   "bench_loop_idx_miss/1",
        "bench_loop_chain_first/1", "bench_loop_chain_last/1", "bench_loop_chain_miss/1",
        "bench_loop_mma_first/1",   "bench_loop_mma_last/1",   "bench_loop_mma_miss/1",
    };
    // Warmup pass so the OS/CPU settles before timed measurements.
    {
        std::vector<Value> args;
        args.push_back(Value::Integer(iters / 20 + 1));
        vm.query("bench_loop_idx_first/1", args);
    }
    for (auto* b : benches) {
        double ns = run_one(vm, b, iters);
        std::printf("BENCH_RESULT %s %.2f\\n", b, ns);
    }
    return 0;
}
'.

%% Compile the generated project. Split-build to keep total compile
%% time manageable: -O2 on wam_runtime.cpp (which contains the
%% step() loop we want optimized) and bench_main.cpp; -O0 on the
%% large auto-generated setup file (just a sequence of push_back
%% calls, runtime-irrelevant). Full -O2 takes ~75s vs ~30s split.
build_bench(Dir) :-
    directory_file_path(Dir, 'cpp', CppDir),
    cc(CppDir, ['-O2', '-c', 'wam_runtime.cpp',       '-o', 'wam_runtime.o']),
    cc(CppDir, ['-O0', '-c', 'generated_program.cpp', '-o', 'generated_program.o']),
    cc(CppDir, ['-O2', '-c', 'bench_main.cpp',        '-o', 'bench_main.o']),
    cc(CppDir, ['-o', 'bench',
                'wam_runtime.o', 'generated_program.o', 'bench_main.o']).

cc(CppDir, Args) :-
    process_create(path('g++'), ['-std=c++17' | Args],
        [cwd(CppDir), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutS), close(Out),
    read_string(Err, _, ErrS), close(Err),
    process_wait(Pid, Status),
    (   Status == exit(0) -> true
    ;   format(user_error, "BUILD FAILED (g++ ~w):~nstdout: ~w~nstderr: ~w~n",
               [Args, OutS, ErrS]),
        throw(bench_build_failed(Args, Status))
    ).

%% Run the binary, capture BENCH_RESULT lines into a list of
%% Name-NsPerIter pairs.
run_bench(Dir, Iters, Results) :-
    directory_file_path(Dir, 'cpp/bench', BinPath),
    atom_string(IterAtom, Iters),
    process_create(BinPath, [IterAtom],
        [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutS), close(Out),
    read_string(Err, _, ErrS), close(Err),
    process_wait(Pid, Status),
    (   Status == exit(0) -> true
    ;   format(user_error, "RUN FAILED: ~w~nstderr: ~w~n", [Status, ErrS]),
        throw(bench_run_failed(Status))
    ),
    split_string(OutS, "\n", "", Lines),
    findall(Name-Ns,
        ( member(Line, Lines),
          split_string(Line, " ", "", ["BENCH_RESULT", NameS, NsS]),
          atom_string(Name, NameS),
          number_string(Ns, NsS)
        ),
        Results),
    Results \= [].

%% Pretty-print as a table. Columns: first, last, miss.
print_results(Results, N, M) :-
    format("~n=============================================================~n"),
    format("  WAM-cpp indexing microbenchmark~n"),
    format("  N=~w clauses per predicate, ~w iters per cell~n", [N, M]),
    format("  Compiled with g++ -O2 (runtime) / -O0 (setup).~n"),
    format("  Values are ns/call (lower is better).~n"),
    format("=============================================================~n~n"),
    format("~w~t~25|~w~t~38|~w~t~51|~w~n",
           ['shape', 'first hit', 'last hit', 'miss']),
    format("~`-t~62|~n"),
    print_row('idx (constant)',  'bench_loop_idx_first/1',   'bench_loop_idx_last/1',   'bench_loop_idx_miss/1',   Results),
    print_row('chain (no index)','bench_loop_chain_first/1', 'bench_loop_chain_last/1', 'bench_loop_chain_miss/1', Results),
    print_row('mma (fallthrough)','bench_loop_mma_first/1',  'bench_loop_mma_last/1',   'bench_loop_mma_miss/1',   Results),
    format("~n"),
    print_speedups(Results),
    format("~n").

print_row(Label, F, L, Miss, Results) :-
    lookup(F,    Results, FNs),
    lookup(L,    Results, LNs),
    lookup(Miss, Results, MNs),
    format("~w~t~25|~3f~t~38|~3f~t~51|~3f~n",
           [Label, FNs, LNs, MNs]).

%% print_speedups(+Results)
%  The headline number for these PRs: how much faster does indexing
%  make the worst-case (last-clause hit and miss)?
print_speedups(R) :-
    lookup('bench_loop_chain_last/1', R, ChainLast),
    lookup('bench_loop_idx_last/1',   R, IdxLast),
    lookup('bench_loop_mma_last/1',   R, MmaLast),
    lookup('bench_loop_chain_miss/1', R, ChainMiss),
    lookup('bench_loop_idx_miss/1',   R, IdxMiss),
    lookup('bench_loop_mma_miss/1',   R, MmaMiss),
    SLastIdx is ChainLast / IdxLast,
    SLastMma is ChainLast / MmaLast,
    SMissIdx is ChainMiss / IdxMiss,
    SMissMma is ChainMiss / MmaMiss,
    format("speedup vs chain (no-index baseline):~n"),
    format("  last-hit  idx ~2fx,  mma ~2fx~n", [SLastIdx, SLastMma]),
    format("  miss      idx ~2fx,  mma ~2fx~n", [SMissIdx, SMissMma]).

lookup(K, Pairs, V) :- member(K-V, Pairs), !.
lookup(_, _, '?').

write_text_file(Path, Text) :-
    setup_call_cleanup(open(Path, write, S), write(S, Text), close(S)).
