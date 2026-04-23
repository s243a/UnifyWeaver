%% generate_llvm_bench.pl
%%
%% Phase 0 of the WAM-LLVM perf roadmap (docs/design/WAM_LLVM_LESSONS_FROM_WAT.md):
%% port the 13-workload benchmark suite from examples/wam_term_builtins_bench/
%% to the WAM-LLVM target so LLVM perf work has baseline measurement
%% infrastructure.
%%
%% Emits a NATIVE LLVM IR module (via write_wam_llvm_project/3). The WASM
%% variant was the originally planned "easiest path" but turns out to have
%% never been end-to-end validated — its state struct type mismatches the
%% shared state.ll.mustache helpers. Running native on Termux aarch64-android
%% sidesteps that and is also the better environment for Phase 1 profiling
%% (`perf` records instead of V8's sampling profiler).
%%
%% Usage (from project root):
%%   swipl examples/wam_llvm_term_builtins_bench/generate_llvm_bench.pl
%%
%% Output: examples/wam_llvm_term_builtins_bench/bench_suite.ll
%%         (+ bench_suite native binary after build_bench.sh)

:- use_module('../../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../wam_term_builtins_bench/bench_suite').
:- use_module('../wam_term_builtins_bench/bench_term_walk').

:- initialization(main, main).

main :-
    % Full 13-workload bench parity with the WAT suite — cut_ite/jump
    % and univ (=../2) landed in this branch, enabling bench_term_depth,
    % bench_fib10, and bench_univ_decomp. bench_copy_flat/nested remain
    % FAIL pending copy_term/2.
    Predicates = [
        bench_suite:bench_true/0,
        bench_suite:bench_is_arith/0,
        bench_suite:bench_unify/0,
        bench_suite:bench_functor_read/0,
        bench_suite:bench_arg_read/0,
        bench_suite:bench_univ_decomp/0,
        bench_suite:bench_copy_flat/0,
        bench_suite:bench_copy_nested/0,
        bench_suite:bench_sum_small/0,
        bench_suite:bench_sum_medium/0,
        bench_suite:bench_sum_big/0,
        bench_suite:bench_term_depth/0,
        bench_suite:bench_fib10/0,
        bench_term_walk:sum_ints/3,
        bench_term_walk:sum_ints_args/5,
        bench_suite:term_depth/2,
        bench_suite:term_depth_args/5,
        bench_suite:fib/3
    ],
    LLPath = 'examples/wam_llvm_term_builtins_bench/bench_suite.ll',
    write_wam_llvm_project(
        Predicates,
        [ module_name(bench_suite)
        , target_triple('aarch64-unknown-linux-android')
        , target_datalayout('e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128')
        ],
        LLPath),
    append_bench_wrappers(LLPath, Predicates),
    halt.

%% append_bench_wrappers(+LLPath, +Predicates)
%  Append small i32-returning wrappers to the generated LLVM module so a
%  native C driver can call each predicate via a stable ABI. Without
%  this, the predicate returns i1 — aarch64 ABI leaves bits 1..63 of the
%  return register unspecified, and linking against `int pred(void)` in
%  C is not safe. Wrappers also call @wam_cleanup so successive bench
%  iterations don't accumulate arena allocations.
append_bench_wrappers(LLPath, Predicates) :-
    findall(W, (
        member(_Mod:Pred/0, Predicates),
        atom_string(Pred, PS),
        format(atom(W),
'define i32 @run_~w() {
  %r = call i1 @~w()
  call void @wam_cleanup()
  %r32 = zext i1 %r to i32
  ret i32 %r32
}', [PS, PS])
    ), Wrappers),
    atomic_list_concat(Wrappers, '\n\n', WrappersStr),
    setup_call_cleanup(
        open(LLPath, append, Stream),
        format(Stream, "~n~n; === Bench C-ABI wrappers (Phase 0) ===~n~w~n", [WrappersStr]),
        close(Stream)
    ).
