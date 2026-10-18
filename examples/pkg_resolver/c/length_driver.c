/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Driver for tests/test_wam_c_length.pl. Builds length/2 queries on the
 * heap, prints write_canonical-style ground lengths, and checks
 * construct mode, variable identity, continuation, rollback, and
 * unsupported shapes in C. Output variable heap handles are preserved
 * across calls rather than relying on argument registers surviving.
 */

#include "wam_runtime.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_wam_length_q_2(WamState *state);
void setup_wam_length_continuation_3(WamState *state);
void setup_wam_length_bind_control_1(WamState *state);
void setup_wam_length_backtrack_1(WamState *state);
void setup_wam_length_mismatch_positive_1(WamState *state);
void setup_wam_length_mismatch_rollback_1(WamState *state);
void setup_wam_length_construct_positive_1(WamState *state);
void setup_wam_length_construct_mismatch_1(WamState *state);

static int g_fail = 0;

static void fail_check(const char *id, const char *why) {
    fprintf(stderr, "length_driver fail %s: %s\n", id, why);
    g_fail = 1;
}

static void ensure_h(WamState *s, int n) {
    if (s->H + n < s->H_cap)
        return;
    int cap = s->H_cap ? s->H_cap : 64;
    while (s->H + n >= cap) {
        if (cap > 1 << 28)
            return;
        cap *= 2;
    }
    WamValue *heap = realloc(s->H_array, sizeof(WamValue) * (size_t)cap);
    if (!heap)
        return;
    s->H_array = heap;
    s->H_cap = cap;
}

static WamValue cons(WamState *s, WamValue h, WamValue t) {
    ensure_h(s, 2);
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = s->H;
    s->H_array[s->H++] = h;
    s->H_array[s->H++] = t;
    return list;
}

static WamValue mkstr(WamState *s, const char *functor, WamValue a, WamValue b) {
    ensure_h(s, 3);
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom(functor);
    s->H_array[s->H++] = a;
    s->H_array[s->H++] = b;
    return term;
}

static WamValue mkstr1(WamState *s, const char *functor, WamValue a) {
    ensure_h(s, 2);
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom(functor);
    s->H_array[s->H++] = a;
    return term;
}

static WamValue nil_atom(void) {
    return val_atom("[]");
}

static int is_nil_cell(WamValue *d) {
    return d->tag == VAL_ATOM && d->data.atom && strcmp(d->data.atom, "[]") == 0;
}

static int list_heads(WamState *s, WamValue v, WamValue **cells, int max, int *n) {
    WamValue *d = wam_deref_ptr(s, &v);
    *n = 0;
    while (1) {
        int c = wam_cons_head_addr(s, d);
        if (c >= 0) {
            if (*n >= max)
                return 0;
            cells[*n] = &s->H_array[c];
            (*n)++;
            d = wam_deref_ptr(s, &s->H_array[c + 1]);
            continue;
        }
        return is_nil_cell(d);
    }
}

static int is_plain_atom(const char *str) {
    if (str == NULL || str[0] == '\0')
        return 0;
    if (!(str[0] >= 'a' && str[0] <= 'z'))
        return 0;
    for (const char *p = str + 1; *p; p++) {
        if (!(isalnum((unsigned char)*p) || *p == '_'))
            return 0;
    }
    return 1;
}

static void print_atom(const char *str) {
    if (str == NULL) {
        fputs("''", stdout);
        return;
    }
    if (strcmp(str, "[]") == 0) {
        fputs("[]", stdout);
        return;
    }
    if (is_plain_atom(str)) {
        fputs(str, stdout);
        return;
    }
    putchar('\'');
    for (const char *p = str; *p; p++) {
        if (*p == '\\' || *p == '\'')
            putchar('\\');
        putchar(*p);
    }
    putchar('\'');
}

static int split_functor(const char *qualified, char *name, size_t name_sz, int *arity) {
    const char *slash = strrchr(qualified, '/');
    if (slash == NULL || slash == qualified)
        return 0;
    size_t nlen = (size_t)(slash - qualified);
    if (nlen + 1 > name_sz)
        return 0;
    memcpy(name, qualified, nlen);
    name[nlen] = '\0';
    *arity = atoi(slash + 1);
    return 1;
}

static void print_term(WamState *state, WamValue v, int depth);

static int is_list_functor(const char *name) {
    return strcmp(name, ".") == 0 || strcmp(name, "[|]") == 0;
}

static void print_list_from_cells(WamState *state, WamValue head, WamValue tail, int depth) {
    putchar('[');
    print_term(state, head, depth + 1);
    for (;;) {
        WamValue *td = wam_deref_ptr(state, &tail);
        if (is_nil_cell(td))
            break;
        int c = wam_cons_head_addr(state, td);
        if (c >= 0) {
            putchar(',');
            print_term(state, state->H_array[c], depth + 1);
            tail = state->H_array[c + 1];
            continue;
        }
        fputs("|", stdout);
        print_term(state, tail, depth + 1);
        break;
    }
    putchar(']');
}

static void print_term(WamState *state, WamValue v, int depth) {
    if (depth > 128) {
        fputs("...", stdout);
        return;
    }
    WamValue *d = wam_deref_ptr(state, &v);
    switch (d->tag) {
    case VAL_ATOM:
        print_atom(d->data.atom);
        return;
    case VAL_INT:
        printf("%d", d->data.integer);
        return;
    case VAL_FLOAT:
        printf("%g", d->data.floating);
        return;
    case VAL_UNBOUND:
        fputs("_", stdout);
        return;
    case VAL_LIST:
        print_list_from_cells(state,
                              state->H_array[d->data.ref_addr],
                              state->H_array[d->data.ref_addr + 1],
                              depth);
        return;
    case VAL_STR: {
        WamValue *fn = &state->H_array[d->data.ref_addr];
        char name[128];
        int arity = 0;
        if (fn->tag != VAL_ATOM || fn->data.atom == NULL ||
            !split_functor(fn->data.atom, name, sizeof name, &arity)) {
            fputs("'<struct>'", stdout);
            return;
        }
        if (arity == 2 && is_list_functor(name)) {
            print_list_from_cells(state,
                                  state->H_array[d->data.ref_addr + 1],
                                  state->H_array[d->data.ref_addr + 2],
                                  depth);
            return;
        }
        print_atom(name);
        if (arity <= 0)
            return;
        putchar('(');
        for (int i = 0; i < arity; i++) {
            if (i > 0)
                putchar(',');
            print_term(state, state->H_array[d->data.ref_addr + 1 + i], depth + 1);
        }
        putchar(')');
        return;
    }
    case VAL_REF:
        fputs("_", stdout);
        return;
    default:
        fputs("'<unknown>'", stdout);
        return;
    }
}

static void emit_case(const char *id, const char *status, WamState *state, WamValue *term) {
    printf("CASE %s\n", id);
    printf("STATUS %s\n", status);
    if (term) {
        fputs("TERM ", stdout);
        print_term(state, *term, 0);
        putchar('\n');
    }
}

static void emit_token(const char *id, const char *status, const char *token) {
    printf("CASE %s\n", id);
    printf("STATUS %s\n", status);
    printf("TERM %s\n", token);
}

static int run_length(WamState *s, WamValue list, WamValue n) {
    WamValue args[2] = { list, n };
    return wam_run_predicate(s, "wam_length_q/2", args, 2);
}

static WamValue make_out_ref(WamState *s, const char *name) {
    ensure_h(s, 1);
    int addr = s->H++;
    s->H_array[addr] = val_unbound(name);
    WamValue ref;
    ref.tag = VAL_REF;
    ref.data.ref_addr = addr;
    return ref;
}

static void run_ground(WamState *s, const char *id, WamValue in) {
    WamValue n_ref = make_out_ref(s, "N");
    int rc = run_length(s, in, n_ref);
    if (rc == 0 && s->error == 0) {
        emit_case(id, "ok", s, &n_ref);
        return;
    }
    if (rc == WAM_HALT && s->error == 0) {
        emit_case(id, "fail", s, NULL);
        return;
    }
    emit_case(id, "runtime_error", s, NULL);
}

static int same_int(WamValue *cell, int n) {
    return cell->tag == VAL_INT && cell->data.integer == n;
}

static int same_atom(WamValue *cell, const char *name) {
    return cell->tag == VAL_ATOM && cell->data.atom &&
           strcmp(cell->data.atom, name) == 0;
}

static int unsupported_length(int rc, WamState *s) {
    return rc == WAM_ERR_UNSUPPORTED &&
           s->error == WAM_ERR_UNSUPPORTED &&
           s->error_op && strcmp(s->error_op, "length/2") == 0 &&
           s->error_arity == 2;
}

static int constructed_fresh_vars(WamState *s, WamValue list, int expect) {
    WamValue *heads[2048];
    int n = 0;
    if (expect == 0)
        return is_nil_cell(wam_deref_ptr(s, &list));
    if (expect > 2048)
        return 0;
    if (!list_heads(s, list, heads, expect + 1, &n) || n != expect)
        return 0;
    for (int i = 0; i < n; i++) {
        WamValue *hi = wam_deref_ptr(s, heads[i]);
        if (!val_is_unbound(*hi))
            return 0;
        for (int j = i + 1; j < n; j++) {
            if (hi == wam_deref_ptr(s, heads[j]))
                return 0;
        }
    }
    return 1;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_length_q_2(&state);
    setup_wam_length_continuation_3(&state);
    setup_wam_length_bind_control_1(&state);
    setup_wam_length_backtrack_1(&state);
    setup_wam_length_mismatch_positive_1(&state);
    setup_wam_length_mismatch_rollback_1(&state);
    setup_wam_length_construct_positive_1(&state);
    setup_wam_length_construct_mismatch_1(&state);

    WamValue nil = nil_atom();

    /* 1. Ground measure cases matching SWI length/2 */
    run_ground(&state, "empty", nil);
    run_ground(&state, "singleton", cons(&state, val_atom("a"), nil));
    run_ground(&state, "duplicates",
               cons(&state, val_atom("b"),
               cons(&state, val_atom("a"),
               cons(&state, val_atom("b"),
               cons(&state, val_atom("a"),
               cons(&state, val_atom("c"), nil))))));
    run_ground(&state, "unordered_ints",
               cons(&state, val_int(3),
               cons(&state, val_int(1),
               cons(&state, val_int(2), nil))));
    run_ground(&state, "unordered_atoms",
               cons(&state, val_atom("c"),
               cons(&state, val_atom("a"),
               cons(&state, val_atom("b"), nil))));
    run_ground(&state, "negatives",
               cons(&state, val_int(3),
               cons(&state, val_int(-1),
               cons(&state, val_int(0),
               cons(&state, val_int(-5), nil)))));
    run_ground(&state, "ints_and_atoms",
               cons(&state, val_atom("b"),
               cons(&state, val_int(2),
               cons(&state, val_atom("a"),
               cons(&state, val_int(1), nil)))));
    run_ground(&state, "pairs",
               cons(&state, mkstr(&state, "-/2", val_atom("c"), val_int(1)),
               cons(&state, mkstr(&state, "-/2", val_atom("a"), val_int(1)),
               cons(&state, mkstr(&state, "-/2", val_atom("b"), val_int(1)), nil))));
    run_ground(&state, "compounds",
               cons(&state, mkstr(&state, "g/2", val_atom("a"), val_atom("b")),
               cons(&state, mkstr1(&state, "f/1", val_atom("b")),
               cons(&state, mkstr1(&state, "f/1", val_atom("a")), nil))));
    run_ground(&state, "lists_and_compounds",
               cons(&state, mkstr1(&state, "f/1", val_atom("a")),
               cons(&state, cons(&state, val_atom("a"), nil),
               cons(&state, val_atom("a"), nil))));
    run_ground(&state, "nested_lists",
               cons(&state,
                    cons(&state, val_int(3), cons(&state, val_int(1), nil)),
               cons(&state,
                    cons(&state, val_int(1), cons(&state, val_int(2), nil)),
               cons(&state,
                    cons(&state, val_int(3), cons(&state, val_int(1), nil)),
                    nil))));
    run_ground(&state, "five_elements",
               cons(&state, val_int(1),
               cons(&state, val_int(2),
               cons(&state, val_int(3),
               cons(&state, val_int(4),
               cons(&state, val_int(5), nil))))));

    /* 2. Prebound matching length */
    {
        int rc = run_length(&state,
                            cons(&state, val_atom("a"),
                            cons(&state, val_atom("b"), nil)),
                            val_int(2));
        if (rc == 0 && state.error == 0)
            emit_token("prebound_match", "ok", "prebound_ok");
        else
            emit_token("prebound_match", rc == WAM_HALT ? "fail" : "runtime_error",
                       "prebound_bad");
    }

    /* 3. Prebound mismatching length (logical failure, no heap/trail growth) */
    {
        WamValue lst = cons(&state, val_atom("a"),
                       cons(&state, val_atom("b"), nil));
        int h0 = state.H;
        int tr0 = state.TR;
        int rc = run_length(&state, lst, val_int(3));
        int rolled = state.H == h0 && state.TR == tr0 && state.error == 0;
        if (rc == WAM_HALT && rolled)
            emit_token("prebound_mismatch", "fail", "prebound_mismatch_ok");
        else
            emit_token("prebound_mismatch", rc == 0 ? "ok" : "runtime_error",
                       "prebound_mismatch_bad");
    }

    /* 4. Preserve input cells */
    {
        WamValue in = cons(&state, val_int(3),
                      cons(&state, val_int(1),
                      cons(&state, val_int(2), nil)));
        int base = in.data.ref_addr;
        int h0 = state.H_array[base].data.integer;
        int h1 = state.H_array[state.H_array[base + 1].data.ref_addr].data.integer;
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, in, n_ref);
        int still0 = state.H_array[base].tag == VAL_INT &&
                     state.H_array[base].data.integer == h0;
        int taddr = state.H_array[base + 1].data.ref_addr;
        int still1 = state.H_array[taddr].tag == VAL_INT &&
                     state.H_array[taddr].data.integer == h1;
        int n_ok = same_int(wam_deref_ptr(&state, &n_ref), 3);
        if (rc == 0 && still0 && still1 && n_ok && h0 == 3 && h1 == 1)
            emit_token("preserve_input", "ok", "preserve_ok");
        else {
            fail_check("preserve_input", "input cons cells changed");
            emit_token("preserve_input", "fail", "preserve_bad");
        }
    }

    /* 5. Heap growth: construct 1024 fresh variables across reallocation */
    {
        WamState growing;
        wam_state_init(&growing);
        setup_wam_length_q_2(&growing);
        WamValue l_ref = make_out_ref(&growing, "L");
        int old_cap = growing.H_cap;
        int rc = run_length(&growing, l_ref, val_int(1024));
        int ok = rc == 0 && growing.error == 0 && growing.H_cap > old_cap &&
                 constructed_fresh_vars(&growing, l_ref, 1024);
        emit_token("heap_growth", ok ? "ok" : "fail",
                   ok ? "heap_growth_ok" : "heap_growth_bad");
        wam_free_state(&growing);
    }

    /* 6. Shared variable identity while measuring */
    {
        WamValue x = wam_make_ref(&state);
        WamValue lst = cons(&state, x,
                       cons(&state, val_atom("a"),
                       cons(&state, x, nil)));
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, lst, n_ref);
        WamValue *heads[8];
        int n = 0;
        int ok = 0;
        if (rc == 0 && same_int(wam_deref_ptr(&state, &n_ref), 3) &&
            list_heads(&state, lst, heads, 8, &n) && n == 3) {
            WamValue *h0 = wam_deref_ptr(&state, heads[0]);
            WamValue *h1 = wam_deref_ptr(&state, heads[1]);
            WamValue *h2 = wam_deref_ptr(&state, heads[2]);
            WamValue *xcell = wam_deref_ptr(&state, &x);
            ok = val_is_unbound(*h0) && val_is_unbound(*h2) &&
                 h0 == xcell && h2 == xcell && same_atom(h1, "a");
        }
        if (ok)
            emit_token("shared_var", "ok", "shared_ok");
        else {
            fail_check("shared_var", "shared variable identity lost");
            emit_token("shared_var", "fail", "shared_bad");
        }
    }

    /* 7. Distinct variables preserved */
    {
        WamValue x = wam_make_ref(&state);
        WamValue y = wam_make_ref(&state);
        WamValue lst = cons(&state, y,
                       cons(&state, x,
                       cons(&state, y, nil)));
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, lst, n_ref);
        WamValue *heads[8];
        int n = 0;
        int ok = 0;
        if (rc == 0 && same_int(wam_deref_ptr(&state, &n_ref), 3) &&
            list_heads(&state, lst, heads, 8, &n) && n == 3) {
            WamValue *h0 = wam_deref_ptr(&state, heads[0]);
            WamValue *h1 = wam_deref_ptr(&state, heads[1]);
            WamValue *h2 = wam_deref_ptr(&state, heads[2]);
            WamValue *xcell = wam_deref_ptr(&state, &x);
            WamValue *ycell = wam_deref_ptr(&state, &y);
            ok = val_is_unbound(*h0) && val_is_unbound(*h1) && val_is_unbound(*h2) &&
                 h0 == ycell && h2 == ycell && h1 == xcell && xcell != ycell;
        }
        if (ok)
            emit_token("distinct_vars", "ok", "distinct_ok");
        else {
            fail_check("distinct_vars", "distinct variables collapsed");
            emit_token("distinct_vars", "fail", "distinct_bad");
        }
    }

    /* 8. Cell vars remain independently bindable after measure */
    {
        WamValue in = cons(&state, val_unbound("X"),
                      cons(&state, val_unbound("Y"), nil));
        int xaddr = in.data.ref_addr;
        int yaddr = state.H_array[xaddr + 1].data.ref_addr;
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, in, n_ref);
        WamValue *x = &state.H_array[xaddr];
        WamValue *y = &state.H_array[yaddr];
        int ok = rc == 0 && same_int(wam_deref_ptr(&state, &n_ref), 2) &&
                 val_is_unbound(*wam_deref_ptr(&state, x)) &&
                 val_is_unbound(*wam_deref_ptr(&state, y)) &&
                 wam_deref_ptr(&state, x) != wam_deref_ptr(&state, y);
        if (ok) {
            WamValue bound = val_atom("bound_val");
            ok = wam_unify(&state, x, &bound) &&
                 same_atom(wam_deref_ptr(&state, x), "bound_val") &&
                 val_is_unbound(*wam_deref_ptr(&state, y));
        }
        emit_token("cell_vars", ok ? "ok" : "fail",
                   ok ? "cell_vars_ok" : "cell_vars_bad");
    }

    /* 9. Compound terms sharing variables */
    {
        WamValue x = wam_make_ref(&state);
        WamValue lst = cons(&state, mkstr1(&state, "f/1", x),
                       cons(&state, mkstr1(&state, "g/1", x), nil));
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, lst, n_ref);
        WamValue *heads[2];
        int n = 0;
        int ok = rc == 0 && same_int(wam_deref_ptr(&state, &n_ref), 2) &&
                 list_heads(&state, lst, heads, 2, &n) && n == 2;
        if (ok) {
            WamValue *f_term = wam_deref_ptr(&state, heads[0]);
            WamValue *g_term = wam_deref_ptr(&state, heads[1]);
            ok = f_term->tag == VAL_STR && g_term->tag == VAL_STR;
            if (ok) {
                WamValue *f_arg = &state.H_array[f_term->data.ref_addr + 1];
                WamValue *g_arg = &state.H_array[g_term->data.ref_addr + 1];
                WamValue *fx = wam_deref_ptr(&state, f_arg);
                WamValue *gx = wam_deref_ptr(&state, g_arg);
                WamValue *xcell = wam_deref_ptr(&state, &x);
                ok = fx == xcell && gx == xcell && val_is_unbound(*xcell);
            }
        }
        emit_token("compound_shared_var", ok ? "ok" : "fail",
                   ok ? "compound_shared_ok" : "compound_shared_bad");
    }

    /* 10a. Positive control: length([X], X) binds X = 1 */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_length_bind_control/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_int(wam_deref_ptr(&state, &out_ref), 1);
        emit_token("bind_control", ok ? "ok" : "fail",
                   ok ? "bind_control_ok" : "bind_control_bad");
    }

    /* 10b. Backtracking rollback restores the shared length variable */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_length_backtrack/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("backtrack_rollback", ok ? "ok" : "fail",
                   ok ? "backtrack_ok" : "backtrack_bad");
    }

    /* 11a. Positive mismatch control: length([a, X], 2) succeeds */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_length_mismatch_positive/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "unbound");
        emit_token("mismatch_positive", ok ? "ok" : "fail",
                   ok ? "mismatch_positive_ok" : "mismatch_positive_bad");
    }

    /* 11b. Mismatch rollback: length([a, X], 1) fails; X stays unbound */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_length_mismatch_rollback/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "mismatch_rollback_ok" : "mismatch_rollback_bad");
    }

    /* 11c. Direct C unifier rollback on measure mismatch */
    {
        WamValue x = wam_make_ref(&state);
        WamValue in = cons(&state, x, nil);
        int tr_before = state.TR;
        int rc_pos = run_length(&state, in, x);
        int c_bind_ok = rc_pos == 0 && same_int(wam_deref_ptr(&state, &x), 1);
        unwind_trail(&state, tr_before);

        WamValue y = wam_make_ref(&state);
        WamValue mis = cons(&state, val_atom("a"), cons(&state, y, nil));
        int h_before = state.H;
        int tr_mis = state.TR;
        int rc_mis = run_length(&state, mis, val_int(1));
        int c_unwind_ok = rc_mis == WAM_HALT && state.error == 0 &&
                          val_is_unbound(*wam_deref_ptr(&state, &y)) &&
                          state.H == h_before && state.TR == tr_mis;
        int ok = c_bind_ok && c_unwind_ok;
        emit_token("c_unifier_rollback", ok ? "ok" : "fail",
                   ok ? "c_rollback_ok" : "c_rollback_bad");
    }

    /* 12. Caller continuation */
    {
        WamValue l = cons(&state, val_int(1), cons(&state, val_int(2), nil));
        WamValue s1_ref = make_out_ref(&state, "S1");
        WamValue s2_ref = make_out_ref(&state, "S2");
        WamValue args[3] = { l, s1_ref, s2_ref };
        int rc = wam_run_predicate(&state, "wam_length_continuation/3", args, 3);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &s1_ref), "start") &&
                 same_atom(wam_deref_ptr(&state, &s2_ref), "done");
        emit_token("caller_continuation", ok ? "ok" : "fail",
                   ok ? "continuation_ok" : "continuation_bad");
    }

    /* 13. Construct empty list */
    {
        WamValue l_ref = make_out_ref(&state, "L");
        int rc = run_length(&state, l_ref, val_int(0));
        int ok = rc == 0 && state.error == 0 &&
                 is_nil_cell(wam_deref_ptr(&state, &l_ref));
        emit_token("construct_zero", ok ? "ok" : "fail",
                   ok ? "construct_zero_ok" : "construct_zero_bad");
    }

    /* 14. Construct three distinct fresh variables */
    {
        WamValue l_ref = make_out_ref(&state, "L");
        int rc = run_length(&state, l_ref, val_int(3));
        int ok = rc == 0 && state.error == 0 &&
                 constructed_fresh_vars(&state, l_ref, 3);
        if (ok) {
            WamValue *heads[3];
            int n = 0;
            ok = list_heads(&state, l_ref, heads, 3, &n) && n == 3;
            if (ok) {
                WamValue bound = val_atom("bound_val");
                ok = wam_unify(&state, heads[0], &bound) &&
                     same_atom(wam_deref_ptr(&state, heads[0]), "bound_val") &&
                     val_is_unbound(*wam_deref_ptr(&state, heads[1])) &&
                     val_is_unbound(*wam_deref_ptr(&state, heads[2]));
            }
        }
        emit_token("construct_three", ok ? "ok" : "fail",
                   ok ? "construct_three_ok" : "construct_three_bad");
    }

    /* 15. Aliased nonnegative length constructs a list */
    {
        WamValue n = wam_make_ref(&state);
        WamValue two = val_int(2);
        int alias_ok = wam_unify(&state, &n, &two);
        WamValue l_ref = make_out_ref(&state, "L");
        int rc = run_length(&state, l_ref, n);
        int ok = alias_ok && rc == 0 && state.error == 0 &&
                 constructed_fresh_vars(&state, l_ref, 2) &&
                 same_int(wam_deref_ptr(&state, &n), 2);
        emit_token("construct_aliased", ok ? "ok" : "fail",
                   ok ? "construct_aliased_ok" : "construct_aliased_bad");
    }

    /* 16. Constructed list unifies with a matching bound list */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_length_construct_positive/1", args, 1);
        int ok = rc == 0 && state.error == 0;
        if (ok) {
            WamValue *heads[2];
            int n = 0;
            ok = list_heads(&state, out_ref, heads, 2, &n) && n == 2 &&
                 same_atom(wam_deref_ptr(&state, heads[0]), "a") &&
                 same_atom(wam_deref_ptr(&state, heads[1]), "b");
        }
        emit_token("construct_positive", ok ? "ok" : "fail",
                   ok ? "construct_positive_ok" : "construct_positive_bad");
    }

    /* 17. Construct then failed unify rolls back the list binding */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_length_construct_mismatch/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("construct_mismatch", ok ? "ok" : "fail",
                   ok ? "construct_mismatch_ok" : "construct_mismatch_bad");
    }

    /* 18. Repeated measure-mismatch rollback on one WamState */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_length_q_2(&st);
        WamValue x = wam_make_ref(&st);
        WamValue lst = cons(&st, val_atom("a"), cons(&st, x, nil));
        int baseline_H = st.H;
        int baseline_TR = st.TR;
        int all_failed = 1;
        for (int iter = 0; iter < 5; iter++) {
            int rc = run_length(&st, lst, val_int(1));
            if (rc != WAM_HALT || st.error != 0 ||
                !val_is_unbound(*wam_deref_ptr(&st, &x)) ||
                st.H != baseline_H || st.TR != baseline_TR) {
                all_failed = 0;
                break;
            }
        }
        int rc_good = run_length(&st, lst, val_int(2));
        int good_ok = rc_good == 0 && st.error == 0 &&
                      val_is_unbound(*wam_deref_ptr(&st, &x));
        int ok = all_failed && good_ok;
        emit_token("repeated_mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "repeated_rollback_ok" : "repeated_rollback_bad");
        wam_free_state(&st);
    }

    /* 19. Unsupported shapes */
    {
        WamValue l_ref = make_out_ref(&state, "L");
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, l_ref, n_ref);
        if (unsupported_length(rc, &state))
            emit_token("both_unbound", "runtime_error", "both_unbound_ok");
        else {
            fail_check("both_unbound", "both unbound not diagnosed");
            emit_token("both_unbound", rc == 0 ? "ok" : "fail", "both_unbound_bad");
        }
        wam_clear_error(&state);
    }

    {
        WamValue l_ref = make_out_ref(&state, "L");
        int rc = run_length(&state, l_ref, val_int(-1));
        if (unsupported_length(rc, &state))
            emit_token("negative_n", "runtime_error", "negative_ok");
        else {
            fail_check("negative_n", "negative N not diagnosed");
            emit_token("negative_n", rc == 0 ? "ok" : "fail", "negative_bad");
        }
        wam_clear_error(&state);
    }

    {
        int rc = run_length(&state, cons(&state, val_atom("a"), nil), val_int(-1));
        if (unsupported_length(rc, &state))
            emit_token("negative_n_with_list", "runtime_error", "negative_list_ok");
        else {
            fail_check("negative_n_with_list", "negative N with list not diagnosed");
            emit_token("negative_n_with_list", rc == 0 ? "ok" : "fail",
                       "negative_list_bad");
        }
        wam_clear_error(&state);
    }

    {
        WamValue l_ref = make_out_ref(&state, "L");
        int rc = run_length(&state, l_ref, val_float(2.0));
        if (unsupported_length(rc, &state))
            emit_token("non_integer_n", "runtime_error", "non_integer_ok");
        else {
            fail_check("non_integer_n", "non-integer N not diagnosed");
            emit_token("non_integer_n", rc == 0 ? "ok" : "fail", "non_integer_bad");
        }
        wam_clear_error(&state);
    }

    {
        ensure_h(&state, 3);
        int base = state.H;
        state.H_array[state.H++] = val_int(1);
        WamValue cyclic;
        cyclic.tag = VAL_LIST;
        cyclic.data.ref_addr = base;
        state.H_array[state.H++] = cyclic;
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, cyclic, n_ref);
        if (unsupported_length(rc, &state))
            emit_token("cyclic", "runtime_error", "cyclic_ok");
        else {
            fail_check("cyclic", "cyclic list not diagnosed");
            emit_token("cyclic", rc == 0 ? "ok" : "fail", "cyclic_bad");
        }
        wam_clear_error(&state);
    }

    {
        WamValue open = cons(&state, val_int(1), val_unbound("Tail"));
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, open, n_ref);
        if (unsupported_length(rc, &state))
            emit_token("open_list", "runtime_error", "open_ok");
        else {
            fail_check("open_list", "open list not diagnosed");
            emit_token("open_list", rc == 0 ? "ok" : "fail", "open_bad");
        }
        wam_clear_error(&state);
    }

    {
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, val_atom("not_a_list"), n_ref);
        if (unsupported_length(rc, &state))
            emit_token("non_list", "runtime_error", "non_list_ok");
        else {
            fail_check("non_list", "non-list not diagnosed");
            emit_token("non_list", rc == 0 ? "ok" : "fail", "non_list_bad");
        }
        wam_clear_error(&state);
    }

    {
        WamValue improper = cons(&state, val_int(1), val_atom("end"));
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_length(&state, improper, n_ref);
        if (unsupported_length(rc, &state))
            emit_token("improper", "runtime_error", "improper_ok");
        else {
            fail_check("improper", "improper list not diagnosed");
            emit_token("improper", rc == 0 ? "ok" : "fail", "improper_bad");
        }
        wam_clear_error(&state);
    }

    wam_free_state(&state);
    return g_fail ? 20 : 0;
}
