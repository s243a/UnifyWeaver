% test_wam_c_fact_table_exec.pl
%
% T9 fact-table inline for the C target — end-to-end. Builds a project where an
% in-window fact predicate edge/2 lowers to the backtrackable fact-table handler
% (driving wam_fact_table_scan + the WAM_FACT_TABLE_RETRY choice point) and runs
% a generated C program that enumerates every solution per query arg mode by
% forcing backtracking (a collect/2 foreign that records the bindings and fails).
% Assertions are encoded as the process exit code (0 = ALL PASS).
%
% Skipped when gcc is not on PATH.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_c_target').

:- dynamic user:edge/2.
user:edge(a, 1). user:edge(a, 2). user:edge(b, 3).
user:edge(a, 4). user:edge(c, 5). user:edge(b, 6).

gcc_available :-
    catch(process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          _, fail),
    process_wait(Pid, exit(0)).

c_t9_compile(RuntimePath, PredPath, MainPath, ExePath) :-
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    format(atom(Cmd),
           'gcc -std=c11 -Wall -Wextra -I ~w ~w ~w ~w -o ~w',
           [IncludeDir, RuntimePath, PredPath, MainPath, ExePath]),
    shell(Cmd, Status),
    Status =:= 0.

:- begin_tests(wam_c_fact_table_exec, [condition(gcc_available)]).

test(query_mode_matrix_exec) :-
    % q/2 query stub: call edge (T9), then collect (records + fails -> redo).
    QStub = 'q/2:\n    call_foreign edge/2, 2\n    call_foreign collect/2, 2\n    proceed',
    compile_wam_predicate_to_c(user:q/2, QStub, [], QCode),
    wam_c_fact_table_helper_for_predicate(user:edge/2,
        [[a,1],[a,2],[b,3],[a,4],[c,5],[b,6]], _Key, EdgeCode, EdgeSetup),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now), Stamp is round(Now * 1000000),
    format(atom(Base), '/tmp/unifyweaver_wam_c_t9_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [Base]),
    format(atom(PredPath), '~w_pred.c', [Base]),
    format(atom(MainPath), '~w_main.c', [Base]),
    format(atom(ExePath), '~w_bin', [Base]),
    setup_call_cleanup(open(RuntimePath, write, R), write(R, RuntimeCode), close(R)),
    format(atom(PredTU),
        '#include "wam_runtime.h"~n~n~w~n~n~w~n~nvoid setup_edge_t9(WamState* state) {~n~w~n}~n',
        [QCode, EdgeCode, EdgeSetup]),
    setup_call_cleanup(open(PredPath, write, P), write(P, PredTU), close(P)),
    c_t9_exec_main(MainSrc),
    setup_call_cleanup(open(MainPath, write, M), write(M, MainSrc), close(M)),
    assertion(c_t9_compile(RuntimePath, PredPath, MainPath, ExePath)),
    format(atom(RunCmd), 'timeout 10 ~w', [ExePath]),
    shell(RunCmd, RunStatus),
    assertion(RunStatus =:= 0).

:- end_tests(wam_c_fact_table_exec).

% Generated C driver. enumerate() runs q/2 with seeded args; collect2 records
% each solution and returns false so wam_run backtracks into the fact-table
% choice point, yielding the next matching row until exhaustion.
c_t9_exec_main(
'#include "wam_runtime.h"
#include <stdio.h>
#include <string.h>

void setup_q_2(WamState* state);
void setup_edge_t9(WamState* state);

static int g_count;
static WamValue g_a0[64];
static WamValue g_a1[64];

static bool collect2(WamState *state, const char *pred, int arity) {
    (void)pred; (void)arity;
    if (g_count < 64) {
        g_a0[g_count] = *wam_deref_ptr(state, &state->A[0]);
        g_a1[g_count] = *wam_deref_ptr(state, &state->A[1]);
        g_count++;
    }
    return false;
}

static int enumerate(WamValue a0, WamValue a1) {
    WamState state;
    wam_state_init(&state);
    setup_q_2(&state);
    setup_edge_t9(&state);
    wam_register_foreign_predicate(&state, "collect/2", 2, collect2);
    g_count = 0;
    int entry = resolve_predicate_hash(&state, "q/2");
    if (entry < 0) { wam_free_state(&state); return -1; }
    state.A[0] = val_is_unbound(a0) ? wam_make_ref(&state) : a0;
    state.A[1] = val_is_unbound(a1) ? wam_make_ref(&state) : a1;
    state.CP = WAM_HALT;
    state.P = entry;
    wam_run(&state);
    int c = g_count;
    wam_free_state(&state);
    return c;
}

int main(void) {
    /* (+,-) edge(a,X) -> rows 1,2,4 in source order */
    if (enumerate(val_atom("a"), val_unbound("X")) != 3) return 11;
    if (!(g_a1[0].tag==VAL_INT && g_a1[0].data.integer==1 &&
          g_a1[1].tag==VAL_INT && g_a1[1].data.integer==2 &&
          g_a1[2].tag==VAL_INT && g_a1[2].data.integer==4)) return 12;
    /* (-,+) edge(K,3) -> 1 row: b */
    if (enumerate(val_unbound("K"), val_int(3)) != 1) return 21;
    if (!(g_a0[0].tag==VAL_ATOM && strcmp(g_a0[0].data.atom,"b")==0)) return 22;
    /* (-,-) edge(K,X) -> all 6 */
    if (enumerate(val_unbound("K"), val_unbound("X")) != 6) return 31;
    /* (+,+) membership */
    if (enumerate(val_atom("a"), val_int(2)) != 1) return 41;
    if (enumerate(val_atom("a"), val_int(3)) != 0) return 42;
    /* absent key -> 0 */
    if (enumerate(val_atom("z"), val_unbound("X")) != 0) return 51;
    printf("ALL 6 PASS\\n");
    return 0;
}
').
