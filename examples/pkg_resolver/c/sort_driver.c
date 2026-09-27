/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Driver for tests/test_wam_c_sort.pl. Builds sort/2 queries on the
 * heap, prints write_canonical-style ground answers, and checks
 * variable identity, input preservation, and invalid lists in C.
 */

#include "wam_runtime.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_wam_sort_q_2(WamState *state);

static int g_fail = 0;

static void fail_check(const char *id, const char *why) {
    fprintf(stderr, "sort_driver fail %s: %s\n", id, why);
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

static int run_sort(WamState *s, WamValue in, WamValue out) {
    WamValue args[2] = { in, out };
    return wam_run_predicate(s, "wam_sort_q/2", args, 2);
}

static void run_ground(WamState *s, const char *id, WamValue in) {
    int rc = run_sort(s, in, val_unbound("S"));
    if (rc == 0 && s->error == 0) {
        emit_case(id, "ok", s, &s->A[1]);
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

static int same_float(WamValue *cell, double n) {
    return cell->tag == VAL_FLOAT && cell->data.floating == n;
}

static int same_atom(WamValue *cell, const char *name) {
    return cell->tag == VAL_ATOM && cell->data.atom &&
           strcmp(cell->data.atom, name) == 0;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_sort_q_2(&state);

    WamValue nil = nil_atom();

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
    run_ground(&state, "list_binary_order",
               cons(&state, cons(&state, val_atom("a"), nil),
               cons(&state, mkstr(&state, "a/2", val_atom("a"), val_atom("b")),
               cons(&state, mkstr1(&state, "z/1", val_atom("a")),
               cons(&state, mkstr(&state, "z/2", val_atom("a"), val_atom("b")), nil)))));
    run_ground(&state, "nested_lists",
               cons(&state,
                    cons(&state, val_int(3), cons(&state, val_int(1), nil)),
               cons(&state,
                    cons(&state, val_int(1), cons(&state, val_int(2), nil)),
               cons(&state,
                    cons(&state, val_int(3), cons(&state, val_int(1), nil)),
                    nil))));

    {
        WamValue want = cons(&state, val_int(1),
                        cons(&state, val_int(2),
                        cons(&state, val_int(3), nil)));
        int rc = run_sort(&state,
                          cons(&state, val_int(3),
                          cons(&state, val_int(1),
                          cons(&state, val_int(2), nil))),
                          want);
        if (rc == 0 && state.error == 0)
            emit_token("prebound_match", "ok", "prebound_ok");
        else
            emit_token("prebound_match", rc == WAM_HALT ? "fail" : "runtime_error",
                       "prebound_bad");
    }
    {
        WamValue want = cons(&state, val_int(3),
                        cons(&state, val_int(2),
                        cons(&state, val_int(1), nil)));
        int rc = run_sort(&state,
                          cons(&state, val_int(3),
                          cons(&state, val_int(1),
                          cons(&state, val_int(2), nil))),
                          want);
        if (rc == WAM_HALT && state.error == 0)
            emit_token("prebound_mismatch", "fail", "prebound_mismatch_ok");
        else
            emit_token("prebound_mismatch", rc == 0 ? "ok" : "runtime_error",
                       "prebound_mismatch_bad");
    }

    {
        WamValue in = cons(&state, val_int(3),
                      cons(&state, val_int(1),
                      cons(&state, val_int(2), nil)));
        int base = in.data.ref_addr;
        int h0 = state.H_array[base].data.integer;
        int h1 = state.H_array[state.H_array[base + 1].data.ref_addr].data.integer;
        int rc = run_sort(&state, in, val_unbound("S"));
        int still0 = state.H_array[base].tag == VAL_INT &&
                     state.H_array[base].data.integer == h0;
        int taddr = state.H_array[base + 1].data.ref_addr;
        int still1 = state.H_array[taddr].tag == VAL_INT &&
                     state.H_array[taddr].data.integer == h1;
        if (rc == 0 && still0 && still1 && h0 == 3 && h1 == 1)
            emit_token("preserve_input", "ok", "preserve_ok");
        else {
            fail_check("preserve_input", "input cons cells changed");
            emit_token("preserve_input", "fail", "preserve_bad");
        }
    }

    {
        WamState growing;
        wam_state_init(&growing);
        setup_wam_sort_q_2(&growing);
        WamValue input = nil_atom();
        for (int i = 1; i <= 1024; i++) input = cons(&growing, val_int(i), input);
        int old_cap = growing.H_cap;
        int rc = run_sort(&growing, input, val_unbound("S"));
        int ok = rc == 0 && growing.H_cap > old_cap;
        WamValue cursor = growing.A[1];
        for (int i = 1; ok && i <= 1024; i++) {
            int head = wam_cons_head_addr(&growing, wam_deref_ptr(&growing, &cursor));
            ok = head >= 0 && same_int(wam_deref_ptr(&growing, &growing.H_array[head]), i);
            if (ok) cursor = growing.H_array[head + 1];
        }
        ok = ok && is_nil_cell(wam_deref_ptr(&growing, &cursor));
        cursor = input;
        for (int i = 1024; ok && i >= 1; i--) {
            int head = wam_cons_head_addr(&growing, wam_deref_ptr(&growing, &cursor));
            ok = head >= 0 && same_int(wam_deref_ptr(&growing, &growing.H_array[head]), i);
            if (ok) cursor = growing.H_array[head + 1];
        }
        ok = ok && is_nil_cell(wam_deref_ptr(&growing, &cursor));
        emit_token("heap_growth", ok ? "ok" : "fail", ok ? "heap_growth_ok" : "heap_growth_bad");
        wam_free_state(&growing);
    }

    {
        WamValue lst = cons(&state, val_int(1),
                       cons(&state, val_float(1.0),
                       cons(&state, val_int(2),
                       cons(&state, val_int(0), nil))));
        int rc = run_sort(&state, lst, val_unbound("S"));
        WamValue *heads[8];
        int n = 0;
        int ok = rc == 0 && list_heads(&state, state.A[1], heads, 8, &n) && n == 4 &&
                 same_int(wam_deref_ptr(&state, heads[0]), 0) &&
                 same_float(wam_deref_ptr(&state, heads[1]), 1.0) &&
                 same_int(wam_deref_ptr(&state, heads[2]), 1) &&
                 same_int(wam_deref_ptr(&state, heads[3]), 2);
        if (ok)
            emit_token("mixed_numeric", "ok", "mixed_ok");
        else {
            fail_check("mixed_numeric", "int/float order or tie-break mismatch");
            emit_token("mixed_numeric", "fail", "mixed_bad");
        }
    }

    {
        WamValue in = cons(&state, val_unbound("X"),
                      cons(&state, val_unbound("Y"), nil));
        int xaddr = in.data.ref_addr;
        int yaddr = state.H_array[xaddr + 1].data.ref_addr;
        int rc = run_sort(&state, in, val_unbound("S"));
        WamValue *heads[2];
        int n = 0;
        int ok = rc == 0 && list_heads(&state, state.A[1], heads, 2, &n) && n == 2;
        if (ok) {
            WamValue *a = wam_deref_ptr(&state, heads[0]);
            WamValue *b = wam_deref_ptr(&state, heads[1]);
            WamValue *x = &state.H_array[xaddr];
            WamValue *y = &state.H_array[yaddr];
            ok = a != b && ((a == x && b == y) || (a == y && b == x));
            if (ok) {
                WamValue bound = val_atom("bound");
                ok = wam_unify(&state, a, &bound) &&
                     same_atom(wam_deref_ptr(&state, a), "bound");
            }
        }
        emit_token("cell_vars", ok ? "ok" : "fail", ok ? "cell_vars_ok" : "cell_vars_bad");
    }

    {
        WamValue x = wam_make_ref(&state);
        WamValue lst = cons(&state, x,
                       cons(&state, val_atom("a"),
                       cons(&state, x, nil)));
        int rc = run_sort(&state, lst, val_unbound("S"));
        WamValue *heads[8];
        int n = 0;
        int ok = 0;
        if (rc == 0 && list_heads(&state, state.A[1], heads, 8, &n) && n == 2) {
            WamValue *h0 = wam_deref_ptr(&state, heads[0]);
            WamValue *h1 = wam_deref_ptr(&state, heads[1]);
            WamValue *xcell = wam_deref_ptr(&state, &x);
            ok = val_is_unbound(*h0) && h0 == xcell && same_atom(h1, "a");
        }
        if (ok)
            emit_token("shared_var", "ok", "shared_ok");
        else {
            fail_check("shared_var", "shared variable identity lost");
            emit_token("shared_var", "fail", "shared_bad");
        }
    }

    {
        WamValue x = wam_make_ref(&state);
        WamValue y = wam_make_ref(&state);
        WamValue lst = cons(&state, y,
                       cons(&state, x,
                       cons(&state, y, nil)));
        int rc = run_sort(&state, lst, val_unbound("S"));
        WamValue *heads[8];
        int n = 0;
        int ok = 0;
        if (rc == 0 && list_heads(&state, state.A[1], heads, 8, &n) && n == 2) {
            WamValue *h0 = wam_deref_ptr(&state, heads[0]);
            WamValue *h1 = wam_deref_ptr(&state, heads[1]);
            WamValue *xcell = wam_deref_ptr(&state, &x);
            WamValue *ycell = wam_deref_ptr(&state, &y);
            int saw_x = (h0 == xcell) || (h1 == xcell);
            int saw_y = (h0 == ycell) || (h1 == ycell);
            ok = val_is_unbound(*h0) && val_is_unbound(*h1) &&
                 h0 != h1 && saw_x && saw_y;
        }
        if (ok)
            emit_token("distinct_vars", "ok", "distinct_ok");
        else {
            fail_check("distinct_vars", "distinct variables collapsed");
            emit_token("distinct_vars", "fail", "distinct_bad");
        }
    }

    {
        /* Behavioral regression for sort([2,1],[X,3]): logical failure,
         * X restored to unbound, temporary heap reclaimed. Repeated failure
         * on the SAME state and subsequent successful compatible query. */
        WamValue x = wam_make_ref(&state);
        WamValue in = cons(&state, val_int(2), cons(&state, val_int(1), nil));
        WamValue mismatch_out = cons(&state, x, cons(&state, val_int(3), nil));
        int h_baseline = state.H;
        int tr_baseline = state.TR;

        /* First attempt: sort([2, 1], [X, 3]) fails */
        int rc1 = run_sort(&state, in, mismatch_out);
        int fail1_ok = (rc1 == WAM_HALT) && (state.error == 0) &&
                       val_is_unbound(*wam_deref_ptr(&state, &x)) &&
                       (state.H == h_baseline) &&
                       (state.TR == tr_baseline);

        /* Repeated failure on the SAME state: sort([2, 1], [X, 3]) fails again */
        int rc2 = run_sort(&state, in, mismatch_out);
        int fail2_ok = (rc2 == WAM_HALT) && (state.error == 0) &&
                       val_is_unbound(*wam_deref_ptr(&state, &x)) &&
                       (state.H == h_baseline) &&
                       (state.TR == tr_baseline);

        /* Subsequent successful compatible query on the SAME state:
         * sort([2, 1], [X, 2]) succeeds and binds X = 1. */
        WamValue compat_out = cons(&state, x, cons(&state, val_int(2), nil));
        int h_before_compat = state.H;
        int rc_compat = run_sort(&state, in, compat_out);
        WamValue *heads[2];
        int n = 0;
        int list_ok = list_heads(&state, state.A[1], heads, 2, &n) && (n == 2) &&
                      same_int(wam_deref_ptr(&state, heads[0]), 1) &&
                      same_int(wam_deref_ptr(&state, heads[1]), 2);
        int compat_ok = (rc_compat == 0) && (state.error == 0) &&
                        same_int(wam_deref_ptr(&state, &x), 1) &&
                        (state.H >= h_before_compat + 4) &&
                        list_ok;

        int ok = fail1_ok && fail2_ok && compat_ok;
        if (!ok) {
            fail_check("mismatch_rollback",
                       !fail1_ok ? "first failure did not restore unbound X or reclaim heap" :
                       !fail2_ok ? "repeated failure on same state corrupted state" :
                       "subsequent compatible query failed or did not bind X");
        }
        emit_token("mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "mismatch_rollback_ok" : "mismatch_rollback_bad");
    }

    {
        ensure_h(&state, 2);
        int base = state.H;
        state.H_array[state.H++] = val_int(1);
        WamValue cyclic;
        cyclic.tag = VAL_LIST;
        cyclic.data.ref_addr = base;
        state.H_array[state.H++] = cyclic;
        int rc = run_sort(&state, cyclic, val_unbound("S"));
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "sort/2") == 0)
            emit_token("cyclic", "runtime_error", "cyclic_ok");
        else {
            fail_check("cyclic", "cyclic list not diagnosed");
            emit_token("cyclic", rc == 0 ? "ok" : "fail", "cyclic_bad");
        }
    }

    {
        WamValue open = cons(&state, val_int(1), val_unbound("Tail"));
        int rc = run_sort(&state, open, val_unbound("S"));
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "sort/2") == 0)
            emit_token("open_list", "runtime_error", "open_ok");
        else {
            fail_check("open_list", "open list not diagnosed");
            emit_token("open_list", rc == 0 ? "ok" : "fail", "open_bad");
        }
    }

    {
        int rc = run_sort(&state, val_atom("not_a_list"), val_unbound("S"));
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "sort/2") == 0)
            emit_token("non_list", "runtime_error", "non_list_ok");
        else {
            fail_check("non_list", "non-list not diagnosed");
            emit_token("non_list", rc == 0 ? "ok" : "fail", "non_list_bad");
        }
    }

    {
        WamValue improper = cons(&state, val_int(1), val_atom("end"));
        int rc = run_sort(&state, improper, val_unbound("S"));
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "sort/2") == 0)
            emit_token("improper", "runtime_error", "improper_ok");
        else {
            fail_check("improper", "improper list not diagnosed");
            emit_token("improper", rc == 0 ? "ok" : "fail", "improper_bad");
        }
    }

    wam_free_state(&state);
    return g_fail ? 20 : 0;
}
