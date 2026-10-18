/* T9 lifetime and transactional-unification regressions, linked by the exec test. */
#include "wam_runtime.h"

static WamValue outer_root;
static int seen, do_cut, expected_outer, expected_inner, cut_base;
static const WamValue outer_rows[] = {
    {.tag=VAL_INT,.data.integer=1}, {.tag=VAL_INT,.data.integer=2},
    {.tag=VAL_INT,.data.integer=2} /* preserve duplicate rows */
};
static const WamValue inner_rows[] = {
    {.tag=VAL_INT,.data.integer=10}, {.tag=VAL_INT,.data.integer=20},
    {.tag=VAL_INT,.data.integer=30}
};
static bool outer(WamState *s, const char *p, int a) {
    (void)p; (void)a;
    return wam_fact_table_scan(s, outer_rows, 1, NULL, 3, 0, s->P+1);
}
static bool inner(WamState *s, const char *p, int a) {
    (void)p; (void)a;
    cut_base = s->B;
    s->A[0] = wam_make_ref(s);
    return wam_fact_table_scan(s, inner_rows, 1, NULL, 3, 0, s->P+1);
}
static bool collect(WamState *s, const char *p, int a) {
    (void)p; (void)a;
    int i = do_cut ? seen : seen / 3;
    int j = do_cut ? 0 : seen % 3;
    expected_outer &= wam_deref_ptr(s, &outer_root)->data.integer == (i ? 2 : 1);
    expected_inner &= wam_deref_ptr(s, &s->A[0])->data.integer == 10*(j+1);
    seen++;
    if (do_cut) wam_prune_choice_points(s, cut_base); /* keep outer, discard inner */
    return false;
}
static int nested(int cut) {
    WamState *s = calloc(1, sizeof(*s));
    wam_state_init(s);
    Instruction code[] = {
        {.tag=INSTR_CALL_FOREIGN,.as.pred={"outer/1",1}},
        {.tag=INSTR_CALL_FOREIGN,.as.pred={"inner/1",1}},
        {.tag=INSTR_CALL_FOREIGN,.as.pred={"collect/1",1}}
    };
    wam_register_foreign_predicate(s,"outer/1",1,outer);
    wam_register_foreign_predicate(s,"inner/1",1,inner);
    wam_register_foreign_predicate(s,"collect/1",1,collect);
    /* Runtime normally owns code, so retain its allocation for free_state. */
    s->code = realloc(s->code, sizeof(code));
    memcpy(s->code, code, sizeof(code)); s->code_size = 3;
    outer_root = wam_make_ref(s); s->A[0] = outer_root;
    s->P = 0; s->CP = WAM_HALT;
    seen = 0; expected_outer = expected_inner = 1; do_cut = cut;
    int rc = wam_run(s);
    int ok = rc == WAM_HALT && seen == (cut ? 3 : 9) &&
             expected_outer && expected_inner && !s->B && !s->fact_table_top && !s->error;
    wam_free_state(s); free(s);
    return ok;
}
int fact_table_regressions(void) {
    WamState *s = calloc(1, sizeof(*s));
    wam_state_init(s);
    WamValue rows[] = {val_int(1),val_int(2), val_int(3),val_int(3),
                       val_int(4),val_int(5)};
    /* The failed first and last rows partially bind a repeated query variable. */
    s->A[0] = wam_make_ref(s); s->A[1] = s->A[0];
    int tr = s->TR;
    if (!wam_fact_table_scan(s,rows,2,NULL,3,0,7) ||
        wam_deref_ptr(s,&s->A[0])->data.integer != 3 || s->B != 1) return 61;
    restore_choice_point(s,&s->B_array[0]);
    s->fact_table_top--; pop_choice_point(s);
    if (wam_fact_table_scan(s,rows,2,NULL,3,2,7) || s->TR != tr ||
        !val_is_unbound(*wam_deref_ptr(s,&s->A[0])) || s->fact_table_top) return 62;
    /* Raw register arguments must also be restored on retry. */
    s->A[0] = val_unbound("X");
    if (!wam_fact_table_scan(s,outer_rows,1,NULL,3,0,7)) return 63;
    restore_choice_point(s,&s->B_array[0]);
    if (!val_is_unbound(s->A[0])) return 64;
    wam_prune_choice_points(s,0);
    if (wam_fact_table_scan(s,rows,33,NULL,1,0,7) ||
        wam_fact_table_scan(s,rows,0,NULL,1,0,7) ||
        wam_fact_table_scan(s,rows,2,NULL,1,-1,7)) return 65;
    /* Capacity failure is an error, and pruning releases every live frame. */
    for (int i=0; i<WAM_META_GOAL_STACK_SIZE; i++) {
        s->A[0] = wam_make_ref(s);
        if (!wam_fact_table_scan(s,outer_rows,1,NULL,3,0,7)) return 66;
    }
    s->A[0] = wam_make_ref(s);
    if (wam_fact_table_scan(s,outer_rows,1,NULL,3,0,7) || s->error != WAM_ERR_OOB) return 67;
    wam_prune_choice_points(s,0);
    if (s->fact_table_top || s->B) return 68;
    wam_free_state(s); free(s);
    if (!nested(0)) return 69;
    if (!nested(1)) return 70;
    return 0;
}
