/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Driver for tests/test_wam_c_member.pl. Builds member/2 queries on the
 * heap, prints write_canonical-style ground answers, and checks
 * variable identity, backtracking, cut, findall, and invalid lists.
 */

#include "wam_runtime.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_wam_member_q_2(WamState *state);
void setup_wam_member_all_2(WamState *state);
void setup_wam_member_then_3(WamState *state);
void setup_wam_member_cut_then_3(WamState *state);
void setup_wam_member_nested_1(WamState *state);
void setup_wam_member_cut_all_2(WamState *state);
void setup_wam_member_grow_4(WamState *state);

static int g_fail = 0;

static void fail_check(const char *id, const char *why) {
    fprintf(stderr, "member_driver fail %s: %s\n", id, why);
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
    fflush(stdout);
}

static void emit_token(const char *id, const char *status, const char *token) {
    printf("CASE %s\n", id);
    printf("STATUS %s\n", status);
    printf("TERM %s\n", token);
    fflush(stdout);
}

static int same_atom(WamValue *cell, const char *name) {
    return cell->tag == VAL_ATOM && cell->data.atom &&
           strcmp(cell->data.atom, name) == 0;
}

static int same_int(WamValue *cell, int n) {
    return cell->tag == VAL_INT && cell->data.integer == n;
}

static void setup_all(WamState *s) {
    setup_wam_member_q_2(s);
    setup_wam_member_all_2(s);
    setup_wam_member_then_3(s);
    setup_wam_member_cut_then_3(s);
    setup_wam_member_nested_1(s);
    setup_wam_member_cut_all_2(s);
    setup_wam_member_grow_4(s);
}

static int run_all(WamState *s, WamValue in) {
    WamValue args[2] = { in, val_unbound("Xs") };
    return wam_run_predicate(s, "wam_member_all/2", args, 2);
}

static int run_q(WamState *s, WamValue elem, WamValue list) {
    WamValue args[2] = { elem, list };
    return wam_run_predicate(s, "wam_member_q/2", args, 2);
}

static void run_all_case(WamState *s, const char *id, WamValue in) {
    int rc = run_all(s, in);
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

static void run_q_elem(WamState *s, const char *id, WamValue elem, WamValue list) {
    int rc = run_q(s, elem, list);
    if (rc == 0 && s->error == 0) {
        emit_case(id, "ok", s, &s->A[0]);
        return;
    }
    if (rc == WAM_HALT && s->error == 0) {
        emit_case(id, "fail", s, NULL);
        return;
    }
    emit_case(id, "runtime_error", s, NULL);
}

static int expect_unsupported(WamState *s, WamValue elem, WamValue list) {
    int rc = run_q(s, elem, list);
    return rc == WAM_ERR_UNSUPPORTED && s->error == WAM_ERR_UNSUPPORTED &&
           s->error_op && strcmp(s->error_op, "member/2") == 0;
}

static void print_probe_state(const char *id, WamState *s, int rc,
                              int old_cap, int old_h) {
    WamValue *a0 = wam_deref_ptr(s, &s->A[0]);
    fprintf(stderr,
            "%s rc=%d error=%d B=%d E=%d TR=%d H %d->%d cap %d->%d A0tag=%d",
            id, rc, s->error, s->B, s->E, s->TR, old_h, s->H, old_cap, s->H_cap,
            a0->tag);
    if (a0->tag == VAL_ATOM && a0->data.atom)
        fprintf(stderr, " A0=%s", a0->data.atom);
    else if (a0->tag == VAL_INT)
        fprintf(stderr, " A0=%d", a0->data.integer);
    fprintf(stderr, "\n");
}

static int run_member_grow(WamState *s, WamValue lst, WamValue pad) {
    WamValue args[4] = { val_unbound("E"), lst, val_atom("c"), pad };
    return wam_run_predicate(s, "wam_member_grow/4", args, 4);
}

static void check_forced_trail_relocation(void) {
    WamState state;
    wam_state_init(&state);
    ensure_h(&state, 1);
    state.H = 1;
    state.H_array[0] = val_unbound("Heap");
    state.A[0] = val_unbound("Register");
    trail_binding(&state, &state.H_array[0]);
    trail_binding(&state, &state.A[0]);
    if (state.TR_array[0].heap_addr != 0 ||
        state.TR_array[1].heap_addr != -1 ||
        state.TR_array[1].cell != &state.A[0]) {
        fail_check("forced_trail_relocation", "heap/register classification");
        wam_free_state(&state);
        return;
    }
    state.H_array[0] = val_atom("bound_heap");
    state.A[0] = val_atom("bound_register");
    /* Allocate while the old heap is live, so relocation is guaranteed rather
       than depending on realloc moving the allocation on this machine. */
    WamValue *moved = malloc(sizeof(WamValue) * (size_t)state.H_cap);
    if (!moved) {
        fail_check("forced_trail_relocation", "allocation failed");
        wam_free_state(&state);
        return;
    }
    memcpy(moved, state.H_array, sizeof(WamValue) * (size_t)state.H);
    free(state.H_array);
    state.H_array = moved;
    unwind_trail(&state, 0);
    if (state.TR != 0 || state.H_array[0].tag != VAL_UNBOUND ||
        state.A[0].tag != VAL_UNBOUND)
        fail_check("forced_trail_relocation", "heap/register restoration");
    wam_free_state(&state);
}

int main(void) {
    check_forced_trail_relocation();
    WamState state;
    wam_state_init(&state);
    setup_all(&state);

    WamValue nil = nil_atom();

    run_all_case(&state, "empty", nil);
    run_all_case(&state, "singleton", cons(&state, val_atom("a"), nil));
    run_all_case(&state, "duplicates",
                 cons(&state, val_atom("a"),
                 cons(&state, val_atom("a"),
                 cons(&state, val_atom("b"), nil))));
    run_all_case(&state, "abc",
                 cons(&state, val_atom("a"),
                 cons(&state, val_atom("b"),
                 cons(&state, val_atom("c"), nil))));
    run_all_case(&state, "compounds",
                 cons(&state, mkstr1(&state, "f/1", val_atom("a")),
                 cons(&state, val_atom("b"), nil)));

    {
        WamValue lst = cons(&state, val_atom("a"),
                       cons(&state, val_atom("b"),
                       cons(&state, val_atom("c"), nil)));
        run_q_elem(&state, "later", val_atom("b"), lst);
        run_q_elem(&state, "missing", val_atom("d"), lst);
    }
    run_q_elem(&state, "empty_q", val_unbound("X"), nil);

    {
        WamValue lst = cons(&state, val_atom("a"),
                       cons(&state, val_atom("b"),
                       cons(&state, val_atom("c"), nil)));
        int rc = run_q(&state, val_atom("b"), lst);
        if (rc == 0 && state.error == 0)
            emit_token("prebound_match", "ok", "prebound_ok");
        else
            emit_token("prebound_match", rc == WAM_HALT ? "fail" : "runtime_error",
                       "prebound_bad");
        rc = run_q(&state, val_atom("d"), lst);
        if (rc == WAM_HALT && state.error == 0)
            emit_token("prebound_mismatch", "fail", "prebound_mismatch_ok");
        else
            emit_token("prebound_mismatch", rc == 0 ? "ok" : "runtime_error",
                       "prebound_mismatch_bad");
    }

    {
        WamValue in = cons(&state, val_atom("a"),
                      cons(&state, val_atom("b"), nil));
        int base = in.data.ref_addr;
        const char *h0 = state.H_array[base].data.atom;
        int rc = run_all(&state, in);
        int still = state.H_array[base].tag == VAL_ATOM &&
                    state.H_array[base].data.atom == h0;
        if (rc == 0 && still)
            emit_token("preserve_input", "ok", "preserve_ok");
        else {
            fail_check("preserve_input", "input cons cells changed");
            emit_token("preserve_input", "fail", "preserve_bad");
        }
    }

    {
        WamValue lst = cons(&state, val_atom("a"),
                       cons(&state, val_atom("b"),
                       cons(&state, val_atom("c"), nil)));
        WamValue args[3] = { val_unbound("E"), lst, val_atom("c") };
        int rc = wam_run_predicate(&state, "wam_member_then/3", args, 3);
        if (rc == 0 && state.error == 0 &&
            same_atom(wam_deref_ptr(&state, &state.A[0]), "c"))
            emit_token("downstream", "ok", "downstream_ok");
        else {
            fail_check("downstream", "did not resume to c");
            emit_token("downstream", rc == WAM_HALT ? "fail" : "runtime_error",
                       "downstream_bad");
        }
    }

    {
        WamValue lst = cons(&state, val_atom("a"),
                       cons(&state, val_atom("b"),
                       cons(&state, val_atom("c"), nil)));
        WamValue args[3] = { val_unbound("E"), lst, val_atom("c") };
        int rc = wam_run_predicate(&state, "wam_member_cut_then/3", args, 3);
        if (rc == WAM_HALT && state.error == 0)
            emit_token("cut_commit", "fail", "cut_ok");
        else {
            fail_check("cut_commit", "cut did not commit first solution");
            emit_token("cut_commit", rc == 0 ? "ok" : "runtime_error", "cut_bad");
        }
    }

    {
        WamValue x = wam_make_ref(&state);
        WamValue elem = mkstr(&state, "f/2", x, val_int(2));
        WamValue lst = cons(&state, mkstr(&state, "f/2", val_atom("a"), val_int(1)),
                       cons(&state, mkstr(&state, "f/2", val_atom("b"), val_int(2)),
                            nil));
        int rc = run_q(&state, elem, lst);
        if (rc == 0 && state.error == 0 &&
            same_atom(wam_deref_ptr(&state, &x), "b"))
            emit_token("partial", "ok", "partial_ok");
        else {
            fail_check("partial", "failed candidate leaked or skipped match");
            emit_token("partial", rc == WAM_HALT ? "fail" : "runtime_error",
                       "partial_bad");
        }
    }

    {
        WamValue args[1] = { val_unbound("P") };
        int rc = wam_run_predicate(&state, "wam_member_nested/1", args, 1);
        WamValue *heads[8];
        int n = 0;
        int ok = rc == 0 && state.error == 0 &&
                 list_heads(&state, state.A[0], heads, 8, &n) && n == 4;
        if (ok) {
            for (int i = 0; ok && i < 4; i++) {
                WamValue *d = wam_deref_ptr(&state, heads[i]);
                if (d->tag != VAL_STR) {
                    ok = 0;
                    break;
                }
                WamValue *fn = &state.H_array[d->data.ref_addr];
                char name[32];
                int arity = 0;
                if (fn->tag != VAL_ATOM || fn->data.atom == NULL ||
                    !split_functor(fn->data.atom, name, sizeof name, &arity) ||
                    arity != 2 || strcmp(name, "-") != 0) {
                    ok = 0;
                    break;
                }
                WamValue *left = wam_deref_ptr(&state, &state.H_array[d->data.ref_addr + 1]);
                WamValue *right = wam_deref_ptr(&state, &state.H_array[d->data.ref_addr + 2]);
                const char *want_l = (i < 2) ? "a" : "b";
                int want_r = (i % 2) ? 2 : 1;
                ok = same_atom(left, want_l) && same_int(right, want_r);
            }
        }
        if (ok)
            emit_token("nested", "ok", "nested_ok");
        else {
            fail_check("nested", "cartesian bag mismatch");
            emit_token("nested", "fail", "nested_bad");
        }
    }

    {
        WamValue e = wam_make_ref(&state);
        WamValue x = wam_make_ref(&state);
        WamValue lst = cons(&state, x,
                       cons(&state, val_atom("a"),
                       cons(&state, x, nil)));
        int rc = run_q(&state, e, lst);
        WamValue *ecell = wam_deref_ptr(&state, &e);
        WamValue *xcell = wam_deref_ptr(&state, &x);
        int ok = rc == 0 && state.error == 0 &&
                 val_is_unbound(*ecell) && ecell == xcell;
        if (ok)
            emit_token("shared_var", "ok", "shared_ok");
        else {
            fail_check("shared_var", "first solution lost shared cell identity");
            emit_token("shared_var", "fail", "shared_bad");
        }
    }

    {
        WamValue e = wam_make_ref(&state);
        WamValue x = wam_make_ref(&state);
        WamValue y = wam_make_ref(&state);
        WamValue lst = cons(&state, x, cons(&state, y, nil));
        int rc = run_q(&state, e, lst);
        WamValue *ecell = wam_deref_ptr(&state, &e);
        WamValue *xcell = wam_deref_ptr(&state, &x);
        WamValue *ycell = wam_deref_ptr(&state, &y);
        int ok = rc == 0 && state.error == 0 && val_is_unbound(*ecell) &&
                 ecell == xcell && ecell != ycell;
        if (ok)
            emit_token("distinct_vars", "ok", "distinct_ok");
        else {
            fail_check("distinct_vars", "distinct variables collapsed");
            emit_token("distinct_vars", "fail", "distinct_bad");
        }
    }

    {
        WamValue e = wam_make_ref(&state);
        WamValue x = wam_make_ref(&state);
        WamValue y = wam_make_ref(&state);
        WamValue lst = cons(&state, x, cons(&state, y, nil));
        int rc = run_q(&state, e, lst);
        WamValue *ecell = wam_deref_ptr(&state, &e);
        WamValue bound = val_atom("bound");
        int ok = rc == 0 && state.error == 0 && val_is_unbound(*ecell) &&
                 wam_unify(&state, ecell, &bound) &&
                 same_atom(wam_deref_ptr(&state, &x), "bound") &&
                 val_is_unbound(*wam_deref_ptr(&state, &y));
        if (ok)
            emit_token("cell_vars", "ok", "cell_vars_ok");
        else {
            fail_check("cell_vars", "binding through member did not alias cell");
            emit_token("cell_vars", "fail", "cell_vars_bad");
        }
    }

    {
        WamState growing;
        wam_state_init(&growing);
        setup_all(&growing);
        WamValue input = nil_atom();
        for (int i = 1024; i >= 1; i--)
            input = cons(&growing, val_int(i), input);
        int old_cap = growing.H_cap;
        int rc = run_all(&growing, input);
        int ok = rc == 0 && growing.error == 0 && growing.H_cap >= old_cap;
        WamValue cursor = growing.A[1];
        for (int i = 1; ok && i <= 1024; i++) {
            int head = wam_cons_head_addr(&growing, wam_deref_ptr(&growing, &cursor));
            ok = head >= 0 && same_int(wam_deref_ptr(&growing, &growing.H_array[head]), i);
            if (ok) cursor = growing.H_array[head + 1];
        }
        ok = ok && is_nil_cell(wam_deref_ptr(&growing, &cursor));
        cursor = input;
        for (int i = 1; ok && i <= 1024; i++) {
            int head = wam_cons_head_addr(&growing, wam_deref_ptr(&growing, &cursor));
            ok = head >= 0 && same_int(wam_deref_ptr(&growing, &growing.H_array[head]), i);
            if (ok) cursor = growing.H_array[head + 1];
        }
        ok = ok && is_nil_cell(wam_deref_ptr(&growing, &cursor));
        emit_token("heap_growth", ok ? "ok" : "fail",
                   ok ? "heap_growth_ok" : "heap_growth_bad");
        wam_free_state(&growing);
    }

    {
        ensure_h(&state, 2);
        int base = state.H;
        state.H_array[state.H++] = val_int(1);
        WamValue cyclic;
        cyclic.tag = VAL_LIST;
        cyclic.data.ref_addr = base;
        state.H_array[state.H++] = cyclic;
        if (expect_unsupported(&state, val_unbound("X"), cyclic))
            emit_token("cyclic", "runtime_error", "cyclic_ok");
        else {
            fail_check("cyclic", "cyclic list not diagnosed");
            emit_token("cyclic", "fail", "cyclic_bad");
        }
    }

    {
        WamValue open = cons(&state, val_int(1), val_unbound("Tail"));
        if (expect_unsupported(&state, val_unbound("X"), open))
            emit_token("open_list", "runtime_error", "open_ok");
        else {
            fail_check("open_list", "open list not diagnosed");
            emit_token("open_list", "fail", "open_bad");
        }
    }

    {
        if (expect_unsupported(&state, val_atom("a"), val_atom("not_a_list")))
            emit_token("non_list", "runtime_error", "non_list_ok");
        else {
            fail_check("non_list", "non-list not diagnosed");
            emit_token("non_list", "fail", "non_list_bad");
        }
    }

    {
        WamValue improper = cons(&state, val_int(1), val_atom("end"));
        if (expect_unsupported(&state, val_unbound("X"), improper))
            emit_token("improper", "runtime_error", "improper_ok");
        else {
            fail_check("improper", "improper list not diagnosed");
            emit_token("improper", "fail", "improper_bad");
        }
    }

    {
        if (expect_unsupported(&state, val_unbound("X"), val_unbound("L")))
            emit_token("unbound_list", "runtime_error", "unbound_ok");
        else {
            fail_check("unbound_list", "unbound list not diagnosed");
            emit_token("unbound_list", "fail", "unbound_bad");
        }
    }

    {
        WamValue lst = cons(&state, val_atom("a"),
                       cons(&state, val_atom("b"), nil));
        int rc1 = run_q(&state, val_atom("a"), lst);
        int rc2 = run_q(&state, val_atom("z"), lst);
        int rc3 = run_all(&state, lst);
        int rc4 = run_q(&state, val_unbound("X"), val_atom("nope"));
        int rc5 = run_q(&state, val_atom("b"), lst);
        int ok = rc1 == 0 && rc2 == WAM_HALT && rc3 == 0 &&
                 rc4 == WAM_ERR_UNSUPPORTED && rc5 == 0 &&
                 state.error == 0 && state.error_op == NULL;
        if (ok)
            emit_token("repeated", "ok", "repeated_ok");
        else {
            fail_check("repeated", "query reuse left error or wrong status");
            emit_token("repeated", "fail", "repeated_bad");
        }
    }

    {
        WamValue lst = cons(&state, val_atom("a"),
                       cons(&state, val_atom("b"),
                       cons(&state, val_atom("c"), nil)));
        WamValue args[2] = { lst, val_unbound("Xs") };
        int rc = wam_run_predicate(&state, "wam_member_cut_all/2", args, 2);
        WamValue *heads[2];
        int n = 0;
        int ok = rc == 0 && state.error == 0 &&
                 list_heads(&state, state.A[1], heads, 2, &n) && n == 1 &&
                 same_atom(wam_deref_ptr(&state, heads[0]), "a");
        if (!ok) {
            fprintf(stderr, "cut_all rc=%d error=%d n=%d B=%d E=%d TR=%d H=%d\n",
                    rc, state.error, n, state.B, state.E, state.TR, state.H);
            fail_check("cut_all", "aggregate cut did not return [a]");
        }
        emit_token("cut_all", ok ? "ok" : "fail",
                   ok ? "cut_all_ok" : "cut_all_bad");
    }

    /* Same member(E,L), sort(Pad,_), E=C shape without heap growth.
       Distinguishes register/environment restoration from realloc/trail. */
    {
        WamState nogrow;
        wam_state_init(&nogrow);
        setup_all(&nogrow);
        WamValue lst = cons(&nogrow, val_atom("a"),
                       cons(&nogrow, val_atom("b"),
                       cons(&nogrow, val_atom("c"), nil_atom())));
        WamValue pad = cons(&nogrow, val_int(1),
                       cons(&nogrow, val_int(2),
                       cons(&nogrow, val_int(3), nil_atom())));
        int old_cap = nogrow.H_cap;
        int old_h = nogrow.H;
        int rc = run_member_grow(&nogrow, lst, pad);
        int grew = nogrow.H_cap > old_cap;
        int ok = rc == 0 && nogrow.error == 0 && !grew &&
                 same_atom(wam_deref_ptr(&nogrow, &nogrow.A[0]), "c");
        print_probe_state("trail_no_growth", &nogrow, rc, old_cap, old_h);
        if (ok)
            emit_token("trail_no_growth", "ok", "trail_nogrow_ok");
        else {
            fail_check("trail_no_growth",
                       grew ? "small pad grew the heap"
                            : "retry without growth returned the wrong result");
            emit_token("trail_no_growth", "fail", "trail_nogrow_bad");
        }
        wam_free_state(&nogrow);
    }

    /* Growth after binding followed by retry. Pair with trail_no_growth:
       both-fail => register/CP/Y restoration; no-growth-only-pass =>
       realloc/trail; both-pass => continuation survived allocation. */
    {
        WamState grow;
        wam_state_init(&grow);
        setup_all(&grow);
        WamValue lst = cons(&grow, val_atom("a"),
                       cons(&grow, val_atom("b"),
                       cons(&grow, val_atom("c"), nil_atom())));
        WamValue pad = nil_atom();
        for (int i = 1; i <= 1024; i++)
            pad = cons(&grow, val_int(i), pad);
        int old_cap = grow.H_cap;
        int old_h = grow.H;
        int rc = run_member_grow(&grow, lst, pad);
        int grew = grow.H_cap > old_cap;
        int ok = rc == 0 && grow.error == 0 && grew &&
                 same_atom(wam_deref_ptr(&grow, &grow.A[0]), "c");
        print_probe_state("trail_after_growth", &grow, rc, old_cap, old_h);
        fprintf(stderr, "trail_after_growth grew=%d\n", grew);
        if (ok)
            emit_token("trail_after_growth", "ok", "trail_ok");
        else {
            fail_check("trail_after_growth", "growth/retry returned the wrong result");
            emit_token("trail_after_growth", "fail", "trail_bad");
        }
        wam_free_state(&grow);
    }

    /* A choicepoint retains live Y registers even if younger calls deallocate
       and reuse the same EnvFrame slots before backtracking. */
    {
        WamState env_restore;
        wam_state_init(&env_restore);
        env_restore.E = 1;
        env_restore.E_array[0].y_regs[0] = val_atom("outer");
        env_restore.E_array[1].y_regs[0] = val_atom("saved");
        push_choice_point(&env_restore, 17, 0);
        env_restore.E_array[0].y_regs[0] = val_atom("overwritten_outer");
        env_restore.E_array[1].y_regs[0] = val_atom("overwritten_saved");
        restore_choice_point(&env_restore,
                             &env_restore.B_array[env_restore.B - 1]);
        int ok = env_restore.E == 1 &&
                 same_atom(&env_restore.E_array[0].y_regs[0], "outer") &&
                 same_atom(&env_restore.E_array[1].y_regs[0], "saved");
        emit_token("choicepoint_env_restore", ok ? "ok" : "fail",
                   ok ? "choicepoint_env_restore_ok"
                      : "choicepoint_env_restore_bad");
        pop_choice_point(&env_restore);
        wam_free_state(&env_restore);
    }

    wam_free_state(&state);
    return g_fail ? 20 : 0;
}
