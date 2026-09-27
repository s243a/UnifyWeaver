/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Driver for tests/test_wam_c_reverse.pl. Builds reverse/2 queries on the
 * heap, prints write_canonical-style ground answers, and checks
 * variable identity, continuation, rollback, and invalid lists in C.
 * Output variable heap handles are preserved across calls rather than
 * relying on argument registers surviving.
 */

#include "wam_runtime.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_wam_reverse_q_2(WamState *state);
void setup_wam_reverse_continuation_3(WamState *state);
void setup_wam_reverse_bind_control_1(WamState *state);
void setup_wam_reverse_backtrack_1(WamState *state);
void setup_wam_reverse_mismatch_positive_1(WamState *state);
void setup_wam_reverse_mismatch_rollback_1(WamState *state);

static int g_fail = 0;

static void fail_check(const char *id, const char *why) {
    fprintf(stderr, "reverse_driver fail %s: %s\n", id, why);
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

static int run_reverse(WamState *s, WamValue in, WamValue out) {
    WamValue args[2] = { in, out };
    return wam_run_predicate(s, "wam_reverse_q/2", args, 2);
}

static void run_ground(WamState *s, const char *id, WamValue in) {
    ensure_h(s, 1);
    int r_addr = s->H++;
    s->H_array[r_addr] = val_unbound("R");
    WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
    int rc = run_reverse(s, in, r_ref);
    if (rc == 0 && s->error == 0) {
        emit_case(id, "ok", s, &r_ref);
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

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_reverse_q_2(&state);
    setup_wam_reverse_continuation_3(&state);
    setup_wam_reverse_bind_control_1(&state);
    setup_wam_reverse_backtrack_1(&state);
    setup_wam_reverse_mismatch_positive_1(&state);
    setup_wam_reverse_mismatch_rollback_1(&state);

    WamValue nil = nil_atom();

    /* 1. Ground cases matching SWI reverse/2 */
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

    /* 2. Prebound matching output */
    {
        WamValue want = cons(&state, val_int(2),
                        cons(&state, val_int(1),
                        cons(&state, val_int(3), nil)));
        int rc = run_reverse(&state,
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

    /* 3. Prebound mismatching output (logical failure) */
    {
        WamValue want = cons(&state, val_int(3),
                        cons(&state, val_int(1),
                        cons(&state, val_int(2), nil)));
        WamValue input = cons(&state, val_int(3),
                         cons(&state, val_int(1),
                         cons(&state, val_int(2), nil)));
        int base_h = state.H;
        int base_tr = state.TR;
        int rc = WAM_HALT;
        bool stable = true;
        for (int i = 0; i < 100; i++) {
            rc = run_reverse(&state, input, want);
            if (rc != WAM_HALT || state.error != 0 ||
                state.H != base_h || state.TR != base_tr) {
                stable = false;
                break;
            }
        }
        if (stable)
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
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, in, r_ref);
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

    /* 5. Heap growth: output variable handle preserved on heap across reallocation */
    {
        WamState growing;
        wam_state_init(&growing);
        setup_wam_reverse_q_2(&growing);
        WamValue input = nil_atom();
        for (int i = 1; i <= 1024; i++) input = cons(&growing, val_int(i), input);
        /* input is [1024, 1023, ..., 1] */
        int old_cap = growing.H_cap;
        ensure_h(&growing, 1);
        int r_addr = growing.H++;
        growing.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&growing, input, r_ref);
        /* reversed is [1, 2, ..., 1024] */
        int ok = rc == 0 && growing.H_cap > old_cap;
        WamValue cursor = r_ref;
        for (int i = 1; ok && i <= 1024; i++) {
            int head = wam_cons_head_addr(&growing, wam_deref_ptr(&growing, &cursor));
            ok = head >= 0 && same_int(wam_deref_ptr(&growing, &growing.H_array[head]), i);
            if (ok) cursor = growing.H_array[head + 1];
        }
        ok = ok && is_nil_cell(wam_deref_ptr(&growing, &cursor));
        /* check input list preserved: 1024 down to 1 */
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

    /* 6. Shared variable identity (preserved output handle) */
    {
        WamValue x = wam_make_ref(&state);
        WamValue lst = cons(&state, x,
                       cons(&state, val_atom("a"),
                       cons(&state, x, nil)));
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, lst, r_ref);
        WamValue *heads[8];
        int n = 0;
        int ok = 0;
        if (rc == 0 && list_heads(&state, r_ref, heads, 8, &n) && n == 3) {
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

    /* 7. Distinct variables preserved (preserved output handle) */
    {
        WamValue x = wam_make_ref(&state);
        WamValue y = wam_make_ref(&state);
        WamValue lst = cons(&state, y,
                       cons(&state, x,
                       cons(&state, y, nil)));
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, lst, r_ref);
        WamValue *heads[8];
        int n = 0;
        int ok = 0;
        if (rc == 0 && list_heads(&state, r_ref, heads, 8, &n) && n == 3) {
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
            fail_check("distinct_vars", "distinct variables collapsed or swapped incorrectly");
            emit_token("distinct_vars", "fail", "distinct_bad");
        }
    }

    /* 8. Cell vars in output list can be independently bound (preserved output handle) */
    {
        WamValue in = cons(&state, val_unbound("X"),
                      cons(&state, val_unbound("Y"), nil));
        int xaddr = in.data.ref_addr;
        int yaddr = state.H_array[xaddr + 1].data.ref_addr;
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, in, r_ref);
        WamValue *heads[2];
        int n = 0;
        int ok = rc == 0 && list_heads(&state, r_ref, heads, 2, &n) && n == 2;
        if (ok) {
            WamValue *a = wam_deref_ptr(&state, heads[0]);
            WamValue *b = wam_deref_ptr(&state, heads[1]);
            WamValue *x = &state.H_array[xaddr];
            WamValue *y = &state.H_array[yaddr];
            /* in was [X, Y], reversed must be [Y, X] */
            ok = a != b && a == y && b == x;
            if (ok) {
                WamValue bound = val_atom("bound_val");
                ok = wam_unify(&state, a, &bound) &&
                     same_atom(wam_deref_ptr(&state, a), "bound_val") &&
                     same_atom(wam_deref_ptr(&state, y), "bound_val") &&
                     val_is_unbound(*wam_deref_ptr(&state, b)) &&
                     val_is_unbound(*wam_deref_ptr(&state, x));
            }
        }
        emit_token("cell_vars", ok ? "ok" : "fail", ok ? "cell_vars_ok" : "cell_vars_bad");
    }

    /* 9. Compound terms sharing variables (preserved output handle) */
    {
        WamValue x = wam_make_ref(&state);
        WamValue lst = cons(&state, mkstr1(&state, "f/1", x),
                       cons(&state, mkstr1(&state, "g/1", x), nil));
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, lst, r_ref);
        WamValue *heads[2];
        int n = 0;
        int ok = rc == 0 && list_heads(&state, r_ref, heads, 2, &n) && n == 2;
        if (ok) {
            WamValue *g_term = wam_deref_ptr(&state, heads[0]);
            WamValue *f_term = wam_deref_ptr(&state, heads[1]);
            ok = g_term->tag == VAL_STR && f_term->tag == VAL_STR;
            if (ok) {
                WamValue *g_arg = &state.H_array[g_term->data.ref_addr + 1];
                WamValue *f_arg = &state.H_array[f_term->data.ref_addr + 1];
                WamValue *gx = wam_deref_ptr(&state, g_arg);
                WamValue *fx = wam_deref_ptr(&state, f_arg);
                WamValue *xcell = wam_deref_ptr(&state, &x);
                ok = gx == xcell && fx == xcell;
            }
        }
        emit_token("compound_shared_var", ok ? "ok" : "fail", ok ? "compound_shared_ok" : "compound_shared_bad");
    }

    /* 10a. Positive control: reverse([1, X], [2, 1]) succeeds and binds X = 2 */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_reverse_bind_control/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_int(wam_deref_ptr(&state, &out_ref), 2);
        emit_token("bind_control", ok ? "ok" : "fail", ok ? "bind_control_ok" : "bind_control_bad");
    }

    /* 10b. Backtracking rollback: downstream failure rolls back X=2 to unbound */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_reverse_backtrack/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("backtrack_rollback", ok ? "ok" : "fail", ok ? "backtrack_ok" : "backtrack_bad");
    }

    /* 11a. Positive mismatch control: reverse([a, X], [b, a]) succeeds and binds X = b */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_reverse_mismatch_positive/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "b");
        emit_token("mismatch_positive", ok ? "ok" : "fail", ok ? "mismatch_positive_ok" : "mismatch_positive_bad");
    }

    /* 11b. Mismatch rollback: reverse([a, X], [b, c]) binds X=b then fails on c\==a, rolling back X */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_reverse_mismatch_rollback/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("mismatch_rollback", ok ? "ok" : "fail", ok ? "mismatch_rollback_ok" : "mismatch_rollback_bad");
    }

    /* 11c. Direct C unifier rollback on mismatch: binds X=b in traversal then unwinds on mismatch */
    {
        WamValue x = wam_make_ref(&state);
        WamValue in = cons(&state, val_atom("a"), cons(&state, x, nil));
        /* Positive: reverse([a, X], [b, a]) succeeds and binds x to b */
        WamValue target_pos = cons(&state, val_atom("b"), cons(&state, val_atom("a"), nil));
        int tr_before = state.TR;
        int rc_pos = run_reverse(&state, in, target_pos);
        int c_bind_ok = rc_pos == 0 && same_atom(wam_deref_ptr(&state, &x), "b");
        unwind_trail(&state, tr_before);

        /* Mismatch: reverse([a, X], [b, c]) unifies X=b then fails on a\==c; unwinds x */
        WamValue target_mis = cons(&state, val_atom("b"), cons(&state, val_atom("c"), nil));
        int rc_mis = run_reverse(&state, in, target_mis);
        int c_unwind_ok = rc_mis == WAM_HALT && val_is_unbound(*wam_deref_ptr(&state, &x));
        int ok = c_bind_ok && c_unwind_ok;
        emit_token("c_unifier_rollback", ok ? "ok" : "fail", ok ? "c_rollback_ok" : "c_rollback_bad");
    }

    /* 12. Caller continuation: goals before and after reverse execute normally */
    {
        WamValue l = cons(&state, val_int(1), cons(&state, val_int(2), nil));
        ensure_h(&state, 2);
        int s1_addr = state.H++;
        state.H_array[s1_addr] = val_unbound("S1");
        WamValue s1_ref = { .tag = VAL_REF, .data = { .ref_addr = s1_addr } };
        int s2_addr = state.H++;
        state.H_array[s2_addr] = val_unbound("S2");
        WamValue s2_ref = { .tag = VAL_REF, .data = { .ref_addr = s2_addr } };
        WamValue args[3] = { l, s1_ref, s2_ref };
        int rc = wam_run_predicate(&state, "wam_reverse_continuation/3", args, 3);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &s1_ref), "start") &&
                 same_atom(wam_deref_ptr(&state, &s2_ref), "done");
        emit_token("caller_continuation", ok ? "ok" : "fail", ok ? "continuation_ok" : "continuation_bad");
    }

    /* 13. Unsupported shape checks */
    /* 13a. Unbound first argument */
    {
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, val_unbound("Var"), r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "reverse/2") == 0 &&
            state.error_arity == 2)
            emit_token("unbound_first_arg", "runtime_error", "unbound_ok");
        else {
            fail_check("unbound_first_arg", "unbound first arg not diagnosed");
            emit_token("unbound_first_arg", rc == 0 ? "ok" : "fail", "unbound_bad");
        }
        wam_clear_error(&state);
    }

    /* 13b. Cyclic list */
    {
        ensure_h(&state, 3);
        int base = state.H;
        state.H_array[state.H++] = val_int(1);
        WamValue cyclic;
        cyclic.tag = VAL_LIST;
        cyclic.data.ref_addr = base;
        state.H_array[state.H++] = cyclic;
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, cyclic, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "reverse/2") == 0 &&
            state.error_arity == 2)
            emit_token("cyclic", "runtime_error", "cyclic_ok");
        else {
            fail_check("cyclic", "cyclic list not diagnosed");
            emit_token("cyclic", rc == 0 ? "ok" : "fail", "cyclic_bad");
        }
        wam_clear_error(&state);
    }

    /* 13c. Open list */
    {
        WamValue open = cons(&state, val_int(1), val_unbound("Tail"));
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, open, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "reverse/2") == 0 &&
            state.error_arity == 2)
            emit_token("open_list", "runtime_error", "open_ok");
        else {
            fail_check("open_list", "open list not diagnosed");
            emit_token("open_list", rc == 0 ? "ok" : "fail", "open_bad");
        }
        wam_clear_error(&state);
    }

    /* 13d. Non-list */
    {
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, val_atom("not_a_list"), r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "reverse/2") == 0 &&
            state.error_arity == 2)
            emit_token("non_list", "runtime_error", "non_list_ok");
        else {
            fail_check("non_list", "non-list not diagnosed");
            emit_token("non_list", rc == 0 ? "ok" : "fail", "non_list_bad");
        }
        wam_clear_error(&state);
    }

    /* 13e. Improper list */
    {
        WamValue improper = cons(&state, val_int(1), val_atom("end"));
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_reverse(&state, improper, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "reverse/2") == 0 &&
            state.error_arity == 2)
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
