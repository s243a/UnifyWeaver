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
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1]).
:- use_module('../src/unifyweaver/targets/wam_c_target').

:- dynamic user:edge/2.
user:edge(a, 1). user:edge(a, 2). user:edge(b, 3).
user:edge(a, 4). user:edge(c, 5). user:edge(b, 6).


:- dynamic user:c_t9_data/2, user:c_t9_project/1, user:c_t9_filter/1,
           user:c_t9_cut/1, user:c_t9_gate/0, user:c_t9_outer/1,
           user:c_t9_qproject/1, user:c_t9_qfilter/1,
           user:c_t9_qcut/1, user:c_t9_qouter/1.
user:c_t9_data(a,1). user:c_t9_data(a,2). user:c_t9_data(b,3).
user:c_t9_data(a,4). user:c_t9_data(a,4). user:c_t9_data(b,6).
user:c_t9_project(V) :- c_t9_data(a,V).
user:c_t9_filter(V) :- c_t9_data(a,V), V > 1.
user:c_t9_cut(V) :- c_t9_data(a,V), !.
user:c_t9_gate :- !.
user:c_t9_outer(V) :- c_t9_data(a,V), c_t9_gate.
user:c_t9_qproject(V) :- c_t9_project(V), c_t9_collector(V).
user:c_t9_qfilter(V) :- c_t9_filter(V), c_t9_collector(V).
user:c_t9_qcut(V) :- c_t9_cut(V), c_t9_collector(V).
user:c_t9_qouter(V) :- c_t9_outer(V), c_t9_collector(V).

gcc_available :-
    catch(process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          _, fail),
    process_wait(Pid, exit(0)).

c_t9_compile(RuntimePath, PredPath, MainPath, ExePath) :-
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc), ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
                               RuntimePath, PredPath, MainPath, 'tests/wam_c_fact_table_regression.c', '-lm', '-o', ExePath],
                   [process(Pid)]),
    process_wait(Pid, exit(0)).

:- begin_tests(wam_c_fact_table_exec, [condition(gcc_available)]).

test(query_mode_matrix_exec) :-
    % q/2 query stub: call edge (T9), then collect (records + fails -> redo).
    QStub = 'q/2:\n    call_foreign edge/2, 2\n    call_foreign collect/2, 2\n    proceed',
    compile_wam_predicate_to_c(user:q/2, QStub, [], QCode),
    wam_c_fact_table_helper_for_predicate(user:edge/2,
        [[a,1],[a,2],[b,3],[a,4],[c,5],[b,6]], _Key, EdgeCode, EdgeSetup),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now), Stamp is round(Now * 1000000),
    tmp_root(Root),
    format(atom(Base), '~w/unifyweaver_wam_c_t9_~w', [Root, Stamp]),
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
    c_t9_compile(RuntimePath, PredPath, MainPath, ExePath),
    process_create(ExePath, [], [process(Pid)]),
    process_wait(Pid, Status, [timeout(10)]),
    ( Status == timeout -> process_kill(Pid, kill), process_wait(Pid, _) ; true ),
    assertion(Status == exit(0)).

test(project_calls_cuts_and_fallback) :-
    forall(member(Options,
      [[lowered_helpers(true),t9_min_rows(1)],
       [lowered_helpers(true),t9_min_rows(1),t9_max_rows(5)]]),
      c_t9_project_exec(Options)).

:- end_tests(wam_c_fact_table_exec).

c_t9_project_exec(Options) :-
    tmp_root(Root), get_time(Now), Stamp is round(Now*1000000),
    format(atom(Dir), '~w/uw_c_t9_project_~w', [Root,Stamp]),
    Preds = [user:c_t9_data/2,user:c_t9_project/1,user:c_t9_filter/1,
      user:c_t9_cut/1,user:c_t9_gate/0,user:c_t9_outer/1,
      user:c_t9_qproject/1,user:c_t9_qfilter/1,user:c_t9_qcut/1,user:c_t9_qouter/1],
    write_wam_c_project(Preds, [no_kernels(true)|Options], Dir),
    compile_wam_predicate_to_c(user:c_t9_collector/1,
      'c_t9_collector/1:\n call_foreign record/1, 1\n proceed', [], Collector),
    directory_file_path(Dir, 'lib.c', Lib),
    setup_call_cleanup(open(Lib,append,L), format(L,'~n~w~n',[Collector]),close(L)),
    c_t9_project_driver(Driver),
    directory_file_path(Dir,'main.c', Main),
    setup_call_cleanup(open(Main,write,M),format(M,'~w',[Driver]),close(M)),
    directory_file_path(Dir,'wam_runtime.c', Runtime),
    directory_file_path(Dir,'t9_exec', Exe),
    c_t9_compile(Runtime,Lib,Main,Exe),
    process_create(Exe,[],[process(Pid)]),
    process_wait(Pid,Status,[timeout(10)]),
    ( Status == timeout -> process_kill(Pid,kill), process_wait(Pid,_) ; true ),
    assertion(Status == exit(0)).

c_t9_project_driver('
#include "wam_runtime.h"
void setup_c_t9_data_2(WamState*);
void setup_c_t9_project_1(WamState*);
void setup_c_t9_filter_1(WamState*);
void setup_c_t9_cut_1(WamState*);
void setup_c_t9_gate_0(WamState*);
void setup_c_t9_outer_1(WamState*);
void setup_c_t9_qproject_1(WamState*);
void setup_c_t9_qfilter_1(WamState*);
void setup_c_t9_qcut_1(WamState*);
void setup_c_t9_qouter_1(WamState*);
void setup_c_t9_collector_1(WamState*);
void setup_lowered_wam_c_helpers(WamState*);
static int count, values[16];
static bool record(WamState *s,const char *p,int a) {
 (void)p;(void)a;
 if(count<16) values[count++]=wam_deref_ptr(s,&s->A[0])->data.integer;
 return false;
}
static int query(const char *name,int expected,const int *want) {
 WamState *s=calloc(1,sizeof(*s)); wam_state_init(s);
 setup_c_t9_data_2(s); setup_c_t9_project_1(s); setup_c_t9_filter_1(s);
 setup_c_t9_cut_1(s); setup_c_t9_gate_0(s); setup_c_t9_outer_1(s);
 setup_c_t9_qproject_1(s);setup_c_t9_qfilter_1(s);setup_c_t9_qcut_1(s);setup_c_t9_qouter_1(s);
 setup_c_t9_collector_1(s); setup_lowered_wam_c_helpers(s);
 wam_register_foreign_predicate(s,"record/1",1,record);
 s->A[0]=wam_make_ref(s);s->CP=WAM_HALT;s->P=resolve_predicate_hash(s,name);count=0;
 int rc=wam_run(s);
 int ok=rc==WAM_HALT && count==expected && !s->B && !s->fact_table_top && !s->error;
 for(int i=0;i<count && i<expected;i++) ok &= values[i]==want[i];
 wam_free_state(s);free(s);return ok;
}
int main(int argc,char **argv) {
 (void)argc;(void)argv;
 const int all[]={1,2,4,4}, filtered[]={2,4,4}, first[]={1};
 if(!query("c_t9_qproject/1",4,all)) return 81;
 if(!query("c_t9_qfilter/1",3,filtered)) return 82;
 if(!query("c_t9_qcut/1",1,first)) return 83;
 if(!query("c_t9_qouter/1",4,all)) return 84;
 return 0;
}
').

% Generated C driver. enumerate() runs q/2 with seeded args; collect2 records
% each solution and returns false so wam_run backtracks into the fact-table
% choice point, yielding the next matching row until exhaustion.
c_t9_exec_main(
'#include "wam_runtime.h"
#include <stdio.h>
#include <string.h>

int fact_table_regressions(void);
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
    int regression = fact_table_regressions();
    if (regression) return regression;
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
