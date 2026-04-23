/*
 * Phase 0 WAM-LLVM benchmark driver.
 *
 * Links against bench_suite.o (built from bench_suite.ll via llc + clang)
 * and loops each `run_bench_<name>` wrapper for a fixed iteration count,
 * reporting ns/call.
 *
 * Usage: ./bench_suite [iterations]
 *
 * Output JSON is written to bench_suite_results.json for programmatic
 * consumption — mirrors the WAT side's run_bench.js output so downstream
 * perf-comparison tooling can treat the two alike.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Wrappers emitted by append_bench_wrappers in generate_llvm_bench.pl.
 * Return 1 on success, 0 on fail. */
extern int run_bench_true(void);
extern int run_bench_is_arith(void);
extern int run_bench_unify(void);
extern int run_bench_functor_read(void);
extern int run_bench_arg_read(void);
extern int run_bench_univ_decomp(void);
extern int run_bench_copy_flat(void);
extern int run_bench_copy_nested(void);
extern int run_bench_sum_small(void);
extern int run_bench_sum_medium(void);
extern int run_bench_sum_big(void);
extern int run_bench_term_depth(void);
extern int run_bench_fib10(void);

typedef struct {
    const char *name;
    int (*fn)(void);
} bench_t;

static bench_t BENCHES[] = {
    {"bench_true",         run_bench_true},
    {"bench_is_arith",     run_bench_is_arith},
    {"bench_unify",        run_bench_unify},
    {"bench_functor_read", run_bench_functor_read},
    {"bench_arg_read",     run_bench_arg_read},
    {"bench_univ_decomp",  run_bench_univ_decomp},
    {"bench_copy_flat",    run_bench_copy_flat},
    {"bench_copy_nested",  run_bench_copy_nested},
    {"bench_sum_small",    run_bench_sum_small},
    {"bench_sum_medium",   run_bench_sum_medium},
    {"bench_sum_big",      run_bench_sum_big},
    {"bench_term_depth",   run_bench_term_depth},
    {"bench_fib10",        run_bench_fib10},
};
#define N_BENCHES (sizeof(BENCHES) / sizeof(BENCHES[0]))

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 10000;
    if (iters <= 0) iters = 10000;

    printf("WAM-LLVM Benchmark Suite (%d iterations per workload)\n", iters);
    puts("======================================================================");
    printf("%-30s %10s %12s %6s\n", "Workload", "ns/call", "calls/s", "ok");
    puts("----------------------------------------------------------------------");

    FILE *jf = fopen("examples/wam_llvm_term_builtins_bench/bench_suite_results.json", "w");
    if (jf) {
        fprintf(jf, "{\n  \"engine\": \"wam-llvm-native-aarch64\",\n");
        fprintf(jf, "  \"iterations\": %d,\n  \"results\": [\n", iters);
    }

    for (size_t bi = 0; bi < N_BENCHES; bi++) {
        bench_t *b = &BENCHES[bi];
        /* Warmup */
        for (int i = 0; i < 100; i++) b->fn();

        uint64_t t0 = ns_now();
        int ok_count = 0;
        for (int i = 0; i < iters; i++) {
            if (b->fn() == 1) ok_count++;
        }
        uint64_t t1 = ns_now();

        double ns_per_call = (double)(t1 - t0) / (double)iters;
        double calls_per_s = 1e9 / ns_per_call;
        int all_ok = (ok_count == iters);

        printf("%-30s %10.0f %12.0f %6s\n",
               b->name, ns_per_call, calls_per_s,
               all_ok ? "OK" : "FAIL");

        if (jf) {
            fprintf(jf, "    {\"workload\": \"%s\", \"ns_per_call\": %.2f, \"ok\": %s}%s\n",
                    b->name, ns_per_call, all_ok ? "true" : "false",
                    (bi + 1 < N_BENCHES) ? "," : "");
        }
    }

    puts("----------------------------------------------------------------------");
    if (jf) {
        fprintf(jf, "  ]\n}\n");
        fclose(jf);
        puts("\nResults written to examples/wam_llvm_term_builtins_bench/bench_suite_results.json");
    }
    return 0;
}
