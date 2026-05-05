#include <stdio.h>
#include <string.h>

#define DECL(name) extern int run_##name(void)
DECL(probe_sum_small_0); DECL(probe_sum_small_1); DECL(probe_sum_small_2);
DECL(probe_sum_small_3); DECL(probe_sum_small_4); DECL(probe_sum_small_5);
DECL(probe_sum_small_6); DECL(probe_sum_small_7);
DECL(probe_fib2_v0); DECL(probe_fib2_v1); DECL(probe_fib2_v2); DECL(probe_fib2_v3);
DECL(probe_fib3_v0); DECL(probe_fib3_v1); DECL(probe_fib3_v2); DECL(probe_fib3_v3);
DECL(probe_term_depth_v0); DECL(probe_term_depth_v1);
DECL(probe_term_depth_v2); DECL(probe_term_depth_v3);
DECL(probe_sum_leaf_v0); DECL(probe_sum_leaf_v1); DECL(probe_sum_leaf_v2);
DECL(probe_sum_leaf_acc5_v5); DECL(probe_sum_leaf_acc5_v6);
DECL(probe_sum_g1_v0); DECL(probe_sum_g1_v1); DECL(probe_sum_g1_v2);
DECL(probe_sum_args_v0); DECL(probe_sum_args_v1);
DECL(probe_sum_args_v2); DECL(probe_sum_args_v3);
DECL(probe_sum_args_2_1_v0); DECL(probe_sum_args_2_1_v1); DECL(probe_sum_args_2_1_v2);
DECL(probe_gt_2_1); DECL(probe_gt_1_1);
DECL(probe_eq_1_1); DECL(probe_eq_1_2);
DECL(probe_td_atom_a_v0); DECL(probe_td_atom_a_v1);
DECL(probe_td_g1_v0); DECL(probe_td_g1_v1); DECL(probe_td_g1_v2);
DECL(probe_td_g2_v0); DECL(probe_td_g2_v1); DECL(probe_td_g2_v2);
DECL(probe_sum_med_v10); DECL(probe_sum_med_v9); DECL(probe_sum_med_v0);
DECL(probe_nested_build); DECL(probe_dbg_sum_fg);

typedef struct { const char *name; int (*fn)(void); } probe_t;
static probe_t PROBES[] = {
    {"sum_ints(f(1,2,3),0,?) == 0", run_probe_sum_small_0},
    {"sum_ints(f(1,2,3),0,?) == 1", run_probe_sum_small_1},
    {"sum_ints(f(1,2,3),0,?) == 2", run_probe_sum_small_2},
    {"sum_ints(f(1,2,3),0,?) == 3", run_probe_sum_small_3},
    {"sum_ints(f(1,2,3),0,?) == 4", run_probe_sum_small_4},
    {"sum_ints(f(1,2,3),0,?) == 5", run_probe_sum_small_5},
    {"sum_ints(f(1,2,3),0,?) == 6", run_probe_sum_small_6},
    {"sum_ints(f(1,2,3),0,?) == 7", run_probe_sum_small_7},
    {"fib(2,0,?) == 0",  run_probe_fib2_v0},
    {"fib(2,0,?) == 1",  run_probe_fib2_v1},
    {"fib(2,0,?) == 2",  run_probe_fib2_v2},
    {"fib(2,0,?) == 3",  run_probe_fib2_v3},
    {"fib(3,0,?) == 0",  run_probe_fib3_v0},
    {"fib(3,0,?) == 1",  run_probe_fib3_v1},
    {"fib(3,0,?) == 2",  run_probe_fib3_v2},
    {"fib(3,0,?) == 3",  run_probe_fib3_v3},
    {"term_depth(f(a,g(b,c)),?) == 0", run_probe_term_depth_v0},
    {"term_depth(f(a,g(b,c)),?) == 1", run_probe_term_depth_v1},
    {"term_depth(f(a,g(b,c)),?) == 2", run_probe_term_depth_v2},
    {"term_depth(f(a,g(b,c)),?) == 3", run_probe_term_depth_v3},
    {"sum_ints(7,0,?) == 0",  run_probe_sum_leaf_v0},
    {"sum_ints(7,0,?) == 7",  run_probe_sum_leaf_v1},
    {"sum_ints(7,0,?) == 14", run_probe_sum_leaf_v2},
    {"sum_ints(7,5,?) == 5",  run_probe_sum_leaf_acc5_v5},
    {"sum_ints(7,5,?) == 12", run_probe_sum_leaf_acc5_v6},
    {"sum_ints(g(1),0,?) == 0", run_probe_sum_g1_v0},
    {"sum_ints(g(1),0,?) == 1", run_probe_sum_g1_v1},
    {"sum_ints(g(1),0,?) == 2", run_probe_sum_g1_v2},
    {"sum_ints_args(1,1,g(1),0,?) == 0", run_probe_sum_args_v0},
    {"sum_ints_args(1,1,g(1),0,?) == 1", run_probe_sum_args_v1},
    {"sum_ints_args(1,1,g(1),0,?) == 2", run_probe_sum_args_v2},
    {"sum_ints_args(1,1,g(1),0,?) == 3", run_probe_sum_args_v3},
    {"sum_ints_args(2,1,g(1),1,?) == 0 [clause1]", run_probe_sum_args_2_1_v0},
    {"sum_ints_args(2,1,g(1),1,?) == 1 [clause1]", run_probe_sum_args_2_1_v1},
    {"sum_ints_args(2,1,g(1),1,?) == 2 [clause1]", run_probe_sum_args_2_1_v2},
    {"2 > 1",         run_probe_gt_2_1},
    {"1 > 1 (false)", run_probe_gt_1_1},
    {"1 = 1",         run_probe_eq_1_1},
    {"1 = 2 (false)", run_probe_eq_1_2},
    {"term_depth(a,?)==0",       run_probe_td_atom_a_v0},
    {"term_depth(a,?)==1",       run_probe_td_atom_a_v1},
    {"term_depth(g(a),?)==0",    run_probe_td_g1_v0},
    {"term_depth(g(a),?)==1",    run_probe_td_g1_v1},
    {"term_depth(g(a),?)==2",    run_probe_td_g1_v2},
    {"term_depth(g(a,b),?)==0",  run_probe_td_g2_v0},
    {"term_depth(g(a,b),?)==1",  run_probe_td_g2_v1},
    {"term_depth(g(a,b),?)==2",  run_probe_td_g2_v2},
    {"sum_ints(f(1,g(2,3),4),0,?)==10", run_probe_sum_med_v10},
    {"sum_ints(f(1,g(2,3),4),0,?)==9",  run_probe_sum_med_v9},
    {"sum_ints(f(1,g(2,3),4),0,?)==0",  run_probe_sum_med_v0},
    {"probe_nested_build", run_probe_nested_build},
    {"probe_dbg_sum_fg", run_probe_dbg_sum_fg},
};

int main(void) {
    size_t n = sizeof(PROBES) / sizeof(PROBES[0]);
    for (size_t i = 0; i < n; i++) {
        int r = PROBES[i].fn();
        printf("%-40s -> %s\n", PROBES[i].name, r ? "TRUE " : "false");
    }
    return 0;
}
