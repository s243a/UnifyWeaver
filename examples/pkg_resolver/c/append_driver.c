/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Driver for tests/test_wam_c_append.pl. Builds append/3 queries on the
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

void setup_wam_append_q_3(WamState *state);
void setup_wam_append_continuation_4(WamState *state);
void setup_wam_append_bind_control_1(WamState *state);
void setup_wam_append_backtrack_1(WamState *state);
void setup_wam_append_mismatch_positive_1(WamState *state);
void setup_wam_append_mismatch_rollback_1(WamState *state);

static int g_fail = 0;

static void fail_check(const char *id, const char *why) {
    fprintf(stderr, "append_driver fail %s: %s\n", id, why);
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
        WamValue *functor_cell = &state->H_array[d->data.ref_addr];
        const char *raw_name = (functor_cell->tag == VAL_ATOM) ? functor_cell->data.atom : "";
        char name[128];
        int arity = 0;
        if (!split_functor(raw_name, name, sizeof(name), &arity)) {
            print_atom(raw_name);
            return;
        }
        if (is_list_functor(name) && arity == 2) {
            print_list_from_cells(state,
                                  state->H_array[d->data.ref_addr + 1],
                                  state->H_array[d->data.ref_addr + 2],
                                  depth);
            return;
        }
        if (strcmp(name, "-") == 0 && arity == 2) {
            print_term(state, state->H_array[d->data.ref_addr + 1], depth + 1);
            putchar('-');
            print_term(state, state->H_array[d->data.ref_addr + 2], depth + 1);
            return;
        }
        print_atom(name);
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
        printf("<ref %d>", d->data.ref_addr);
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

static int run_append(WamState *s, WamValue l1, WamValue l2, WamValue out) {
    WamValue args[3] = { l1, l2, out };
    return wam_run_predicate(s, "wam_append_q/3", args, 3);
}

static void run_ground(WamState *s, const char *id, WamValue l1, WamValue l2) {
    ensure_h(s, 1);
    int r_addr = s->H++;
    s->H_array[r_addr] = val_unbound("R");
    WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
    int rc = run_append(s, l1, l2, r_ref);
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
    setup_wam_append_q_3(&state);
    setup_wam_append_continuation_4(&state);
    setup_wam_append_bind_control_1(&state);
    setup_wam_append_backtrack_1(&state);
    setup_wam_append_mismatch_positive_1(&state);
    setup_wam_append_mismatch_rollback_1(&state);

    WamValue nil = nil_atom();

    /* 1. Ground cases matching SWI append/3 */
    run_ground(&state, "empty_empty", nil, nil);
    run_ground(&state, "empty_nonempty", nil, cons(&state, val_atom("a"), cons(&state, val_atom("b"), nil)));
    run_ground(&state, "singleton_empty", cons(&state, val_atom("a"), nil), nil);
    run_ground(&state, "singleton_singleton", cons(&state, val_atom("a"), nil), cons(&state, val_atom("b"), nil));
    run_ground(&state, "duplicates",
               cons(&state, val_atom("b"), cons(&state, val_atom("a"), cons(&state, val_atom("b"), nil))),
               cons(&state, val_atom("a"), cons(&state, val_atom("c"), nil)));
    run_ground(&state, "unordered_ints",
               cons(&state, val_int(3), cons(&state, val_int(1), nil)),
               cons(&state, val_int(2), cons(&state, val_int(4), nil)));
    run_ground(&state, "unordered_atoms",
               cons(&state, val_atom("c"), cons(&state, val_atom("a"), nil)),
               cons(&state, val_atom("b"), cons(&state, val_atom("d"), nil)));
    run_ground(&state, "negatives",
               cons(&state, val_int(3), cons(&state, val_int(-1), nil)),
               cons(&state, val_int(0), cons(&state, val_int(-5), nil)));
    run_ground(&state, "ints_and_atoms",
               cons(&state, val_atom("b"), cons(&state, val_int(2), nil)),
               cons(&state, val_atom("a"), cons(&state, val_int(1), nil)));
    run_ground(&state, "pairs",
               cons(&state, mkstr(&state, "-/2", val_atom("c"), val_int(1)), nil),
               cons(&state, mkstr(&state, "-/2", val_atom("a"), val_int(1)),
               cons(&state, mkstr(&state, "-/2", val_atom("b"), val_int(1)), nil)));
    run_ground(&state, "compounds",
               cons(&state, mkstr(&state, "g/2", val_atom("a"), val_atom("b")), nil),
               cons(&state, mkstr1(&state, "f/1", val_atom("b")),
               cons(&state, mkstr1(&state, "f/1", val_atom("a")), nil)));
    run_ground(&state, "lists_and_compounds",
               cons(&state, mkstr1(&state, "f/1", val_atom("a")), nil),
               cons(&state, cons(&state, val_atom("a"), nil),
               cons(&state, val_atom("a"), nil)));
    run_ground(&state, "nested_lists",
               cons(&state, cons(&state, val_int(3), cons(&state, val_int(1), nil)), nil),
               cons(&state, cons(&state, val_int(1), cons(&state, val_int(2), nil)),
               cons(&state, cons(&state, val_int(3), cons(&state, val_int(1), nil)), nil)));
    run_ground(&state, "five_elements",
               cons(&state, val_int(1), cons(&state, val_int(2), nil)),
               cons(&state, val_int(3), cons(&state, val_int(4), cons(&state, val_int(5), nil))));
    run_ground(&state, "arbitrary_tail",
               cons(&state, val_int(1), cons(&state, val_int(2), nil)),
               val_atom("tail_atom"));

    /* 2. Empty tail sharing: append([], Tail, Out) unifies Out with Tail without copying Tail */
    {
        WamValue tail = cons(&state, val_atom("x"), cons(&state, val_atom("y"), nil));
        int h_before = state.H;
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        int rc = run_append(&state, nil, tail, out_ref);
        WamValue *d = wam_deref_ptr(&state, &out_ref);
        int ok = rc == 0 && state.error == 0 &&
                 d->tag == VAL_LIST && d->data.ref_addr == tail.data.ref_addr &&
                 (state.H == h_before + 1);
        emit_token("empty_tail_sharing", ok ? "ok" : "fail",
                   ok ? "empty_tail_sharing_ok" : "empty_tail_sharing_bad");
    }

    /* 3. Spine copy tail share: copy only spine of List1, share Tail without copying */
    {
        WamValue tail = cons(&state, val_atom("t1"), nil);
        WamValue l1 = cons(&state, val_atom("h1"), cons(&state, val_atom("h2"), nil));
        int h_before = state.H;
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        int rc = run_append(&state, l1, tail, out_ref);
        WamValue *d = wam_deref_ptr(&state, &out_ref);
        int slots_allocated = state.H - h_before - 1;
        int ok = rc == 0 && state.error == 0 && slots_allocated == 4 && d->tag == VAL_LIST;
        if (ok) {
            int c1 = d->data.ref_addr;
            WamValue *h1 = wam_deref_ptr(&state, &state.H_array[c1]);
            WamValue *t1 = wam_deref_ptr(&state, &state.H_array[c1 + 1]);
            ok = same_atom(h1, "h1") && t1->tag == VAL_LIST;
            if (ok) {
                int c2 = t1->data.ref_addr;
                WamValue *h2 = wam_deref_ptr(&state, &state.H_array[c2]);
                WamValue *t2 = wam_deref_ptr(&state, &state.H_array[c2 + 1]);
                ok = same_atom(h2, "h2") && t2->tag == VAL_LIST &&
                     t2->data.ref_addr == tail.data.ref_addr;
            }
        }
        emit_token("spine_copy_tail_share", ok ? "ok" : "fail",
                   ok ? "spine_copy_tail_share_ok" : "spine_copy_tail_share_bad");
    }

    /* 4. Prebound matching output */
    {
        WamValue want = cons(&state, val_int(1),
                        cons(&state, val_int(2),
                        cons(&state, val_int(3), nil)));
        int rc = run_append(&state,
                            cons(&state, val_int(1), cons(&state, val_int(2), nil)),
                            cons(&state, val_int(3), nil),
                            want);
        if (rc == 0 && state.error == 0)
            emit_token("prebound_match", "ok", "prebound_ok");
        else
            emit_token("prebound_match", rc == WAM_HALT ? "fail" : "runtime_error",
                       "prebound_bad");
    }

    /* 5. Prebound mismatching output (logical failure) */
    {
        WamValue want = cons(&state, val_int(1),
                        cons(&state, val_int(2),
                        cons(&state, val_int(4), nil)));
        int rc = run_append(&state,
                            cons(&state, val_int(1), cons(&state, val_int(2), nil)),
                            cons(&state, val_int(3), nil),
                            want);
        if (rc == WAM_HALT && state.error == 0)
            emit_token("prebound_mismatch", "fail", "prebound_mismatch_ok");
        else
            emit_token("prebound_mismatch", rc == 0 ? "ok" : "runtime_error",
                       "prebound_mismatch_bad");
    }

    /* 6. Preserve input cells */
    {
        WamValue in1 = cons(&state, val_int(3), cons(&state, val_int(1), nil));
        WamValue in2 = cons(&state, val_int(2), nil);
        int base1 = in1.data.ref_addr;
        int h0 = state.H_array[base1].data.integer;
        int t1addr = state.H_array[base1 + 1].data.ref_addr;
        int h1 = state.H_array[t1addr].data.integer;
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, in1, in2, r_ref);
        int still0 = state.H_array[base1].tag == VAL_INT &&
                     state.H_array[base1].data.integer == h0;
        int still1 = state.H_array[t1addr].tag == VAL_INT &&
                     state.H_array[t1addr].data.integer == h1;
        if (rc == 0 && still0 && still1 && h0 == 3 && h1 == 1)
            emit_token("preserve_input", "ok", "preserve_ok");
        else {
            fail_check("preserve_input", "input cons cells changed");
            emit_token("preserve_input", "fail", "preserve_bad");
        }
    }

    /* 7. Heap growth: output variable handle preserved on heap across reallocation */
    {
        WamState growing;
        wam_state_init(&growing);
        setup_wam_append_q_3(&growing);
        WamValue input = nil_atom();
        for (int i = 1024; i >= 1; i--) input = cons(&growing, val_int(i), input);
        /* input is [1, 2, ..., 1024] */
        int old_cap = growing.H_cap;
        ensure_h(&growing, 1);
        int r_addr = growing.H++;
        growing.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&growing, input, cons(&growing, val_int(1025), nil), r_ref);
        /* output is [1, 2, ..., 1025] */
        int ok = rc == 0 && growing.H_cap > old_cap;
        WamValue cursor = r_ref;
        for (int i = 1; ok && i <= 1025; i++) {
            int head = wam_cons_head_addr(&growing, wam_deref_ptr(&growing, &cursor));
            ok = head >= 0 && same_int(wam_deref_ptr(&growing, &growing.H_array[head]), i);
            if (ok) cursor = growing.H_array[head + 1];
        }
        ok = ok && is_nil_cell(wam_deref_ptr(&growing, &cursor));
        emit_token("heap_growth", ok ? "ok" : "fail", ok ? "heap_growth_ok" : "heap_growth_bad");
        wam_free_state(&growing);
    }

    /* 8. Shared variable identity */
    {
        WamValue x = wam_make_ref(&state);
        WamValue lst1 = cons(&state, x, cons(&state, val_atom("a"), nil));
        WamValue lst2 = cons(&state, x, nil);
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, lst1, lst2, r_ref);
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

    /* 9. Distinct variables preserved */
    {
        WamValue x = wam_make_ref(&state);
        WamValue y = wam_make_ref(&state);
        WamValue lst1 = cons(&state, y, cons(&state, x, nil));
        WamValue lst2 = cons(&state, y, nil);
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, lst1, lst2, r_ref);
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

    /* 10. Cell vars in output list can be independently bound */
    {
        WamValue in1 = cons(&state, val_unbound("X"), nil);
        WamValue in2 = cons(&state, val_unbound("Y"), nil);
        int xaddr = in1.data.ref_addr;
        int yaddr = in2.data.ref_addr;
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, in1, in2, r_ref);
        WamValue *heads[2];
        int n = 0;
        int ok = rc == 0 && list_heads(&state, r_ref, heads, 2, &n) && n == 2;
        if (ok) {
            WamValue *a = wam_deref_ptr(&state, heads[0]);
            WamValue *b = wam_deref_ptr(&state, heads[1]);
            WamValue *x = &state.H_array[xaddr];
            WamValue *y = &state.H_array[yaddr];
            ok = a != b && a == x && b == y;
            if (ok) {
                WamValue bound = val_atom("bound_val");
                ok = wam_unify(&state, a, &bound) &&
                     same_atom(wam_deref_ptr(&state, a), "bound_val") &&
                     same_atom(wam_deref_ptr(&state, x), "bound_val") &&
                     val_is_unbound(*wam_deref_ptr(&state, b)) &&
                     val_is_unbound(*wam_deref_ptr(&state, y));
            }
        }
        emit_token("cell_vars", ok ? "ok" : "fail", ok ? "cell_vars_ok" : "cell_vars_bad");
    }

    /* 11. Compound terms sharing variables */
    {
        WamValue x = wam_make_ref(&state);
        WamValue lst1 = cons(&state, mkstr1(&state, "f/1", x), nil);
        WamValue lst2 = cons(&state, mkstr1(&state, "g/1", x), nil);
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, lst1, lst2, r_ref);
        WamValue *heads[2];
        int n = 0;
        int ok = rc == 0 && list_heads(&state, r_ref, heads, 2, &n) && n == 2;
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
                ok = fx == xcell && gx == xcell;
            }
        }
        emit_token("compound_shared_var", ok ? "ok" : "fail", ok ? "compound_shared_ok" : "compound_shared_bad");
    }

    /* 12. Positive control: append([1, X], [3], [1, 2, 3]) succeeds and binds X = 2 */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_append_bind_control/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_int(wam_deref_ptr(&state, &out_ref), 2);
        emit_token("bind_control", ok ? "ok" : "fail", ok ? "bind_control_ok" : "bind_control_bad");
    }

    /* 13. Backtracking rollback: downstream failure rolls back X=2 to unbound */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_append_backtrack/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("backtrack_rollback", ok ? "ok" : "fail", ok ? "backtrack_ok" : "backtrack_bad");
    }

    /* 14. Positive mismatch control: append([a, X], [c], [a, b, c]) succeeds and binds X = b */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_append_mismatch_positive/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "b");
        emit_token("mismatch_positive", ok ? "ok" : "fail", ok ? "mismatch_positive_ok" : "mismatch_positive_bad");
    }

    /* 15. Mismatch rollback: append([a, X], [c], [a, b, d]) binds X=b then fails on c\==d, rolling back X */
    {
        ensure_h(&state, 1);
        int out_addr = state.H++;
        state.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_append_mismatch_rollback/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("mismatch_rollback", ok ? "ok" : "fail", ok ? "mismatch_rollback_ok" : "mismatch_rollback_bad");
    }

    /* 16. Repeated mismatch rollback on one WamState */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_append_q_3(&st);

        WamValue x = wam_make_ref(&st);
        WamValue l1 = cons(&st, val_atom("a"), cons(&st, x, nil));
        WamValue l2 = cons(&st, val_atom("c"), nil);
        WamValue target_bad = cons(&st, val_atom("a"),
                              cons(&st, val_atom("b"),
                              cons(&st, val_atom("mismatch"), nil)));
        WamValue target_good = cons(&st, val_atom("a"),
                               cons(&st, val_atom("b"),
                               cons(&st, val_atom("c"), nil)));

        int baseline_H = st.H;
        int baseline_TR = st.TR;

        int all_failed = 1;
        for (int iter = 0; iter < 5; iter++) {
            int rc = run_append(&st, l1, l2, target_bad);
            if (rc != WAM_HALT || st.error != 0 || !val_is_unbound(*wam_deref_ptr(&st, &x))) {
                all_failed = 0;
                break;
            }
            if (st.H != baseline_H || st.TR != baseline_TR) {
                all_failed = 0;
                break;
            }
        }

        int rc_good = run_append(&st, l1, l2, target_good);
        int good_ok = rc_good == 0 && st.error == 0 &&
                      same_atom(wam_deref_ptr(&st, &x), "b");

        int ok = all_failed && good_ok;
        emit_token("repeated_mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "repeated_rollback_ok" : "repeated_rollback_bad");
        wam_free_state(&st);
    }

    /* 17. Direct C unifier rollback on mismatch */
    {
        WamValue x = wam_make_ref(&state);
        WamValue in1 = cons(&state, val_atom("a"), cons(&state, x, nil));
        WamValue in2 = cons(&state, val_atom("c"), nil);
        /* Positive: append([a, X], [c], [a, b, c]) succeeds and binds x to b */
        WamValue target_pos = cons(&state, val_atom("a"),
                              cons(&state, val_atom("b"),
                              cons(&state, val_atom("c"), nil)));
        int tr_before = state.TR;
        int rc_pos = run_append(&state, in1, in2, target_pos);
        int c_bind_ok = rc_pos == 0 && same_atom(wam_deref_ptr(&state, &x), "b");
        unwind_trail(&state, tr_before);

        /* Mismatch: append([a, X], [c], [a, b, d]) unifies X=b then fails on c\==d; unwinds x */
        WamValue target_mis = cons(&state, val_atom("a"),
                              cons(&state, val_atom("b"),
                              cons(&state, val_atom("d"), nil)));
        int rc_mis = run_append(&state, in1, in2, target_mis);
        int c_unwind_ok = rc_mis == WAM_HALT && val_is_unbound(*wam_deref_ptr(&state, &x));
        int ok = c_bind_ok && c_unwind_ok;
        emit_token("c_unifier_rollback", ok ? "ok" : "fail", ok ? "c_rollback_ok" : "c_rollback_bad");
    }

    /* 18. Caller continuation: goals before and after append execute normally */
    {
        WamValue l1 = cons(&state, val_int(1), nil);
        WamValue l2 = cons(&state, val_int(2), nil);
        ensure_h(&state, 2);
        int s1_addr = state.H++;
        state.H_array[s1_addr] = val_unbound("S1");
        WamValue s1_ref = { .tag = VAL_REF, .data = { .ref_addr = s1_addr } };
        int s2_addr = state.H++;
        state.H_array[s2_addr] = val_unbound("S2");
        WamValue s2_ref = { .tag = VAL_REF, .data = { .ref_addr = s2_addr } };
        WamValue args[4] = { l1, l2, s1_ref, s2_ref };
        int rc = wam_run_predicate(&state, "wam_append_continuation/4", args, 4);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &s1_ref), "start") &&
                 same_atom(wam_deref_ptr(&state, &s2_ref), "done");
        emit_token("caller_continuation", ok ? "ok" : "fail", ok ? "continuation_ok" : "continuation_bad");
    }

    /* 19. Unsupported shape checks */
    /* 19a. Unbound first argument */
    {
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, val_unbound("Var"), nil, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "append/3") == 0 &&
            state.error_arity == 3)
            emit_token("unbound_first_arg", "runtime_error", "unbound_ok");
        else {
            fail_check("unbound_first_arg", "unbound first arg not diagnosed");
            emit_token("unbound_first_arg", rc == 0 ? "ok" : "fail", "unbound_bad");
        }
        wam_clear_error(&state);
    }

    /* 19b. Cyclic list */
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
        int rc = run_append(&state, cyclic, nil, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "append/3") == 0 &&
            state.error_arity == 3)
            emit_token("cyclic", "runtime_error", "cyclic_ok");
        else {
            fail_check("cyclic", "cyclic list not diagnosed");
            emit_token("cyclic", rc == 0 ? "ok" : "fail", "cyclic_bad");
        }
        wam_clear_error(&state);
    }

    /* 19c. Open list */
    {
        WamValue open = cons(&state, val_int(1), val_unbound("Tail"));
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, open, nil, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "append/3") == 0 &&
            state.error_arity == 3)
            emit_token("open_list", "runtime_error", "open_ok");
        else {
            fail_check("open_list", "open list not diagnosed");
            emit_token("open_list", rc == 0 ? "ok" : "fail", "open_bad");
        }
        wam_clear_error(&state);
    }

    /* 19d. Non-list */
    {
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, val_atom("not_a_list"), nil, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "append/3") == 0 &&
            state.error_arity == 3)
            emit_token("non_list", "runtime_error", "non_list_ok");
        else {
            fail_check("non_list", "non-list not diagnosed");
            emit_token("non_list", rc == 0 ? "ok" : "fail", "non_list_bad");
        }
        wam_clear_error(&state);
    }

    /* 19e. Improper list */
    {
        WamValue improper = cons(&state, val_int(1), val_atom("end"));
        ensure_h(&state, 1);
        int r_addr = state.H++;
        state.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        int rc = run_append(&state, improper, nil, r_ref);
        if (rc == WAM_ERR_UNSUPPORTED &&
            state.error == WAM_ERR_UNSUPPORTED &&
            state.error_op && strcmp(state.error_op, "append/3") == 0 &&
            state.error_arity == 3)
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
