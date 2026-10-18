/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Driver for tests/test_wam_c_indexed_dispatch.pl. Prints
 * write_canonical-style first/all-solution terms for indexed
 * try/retry/trust cases. Exit 0 means the driver ran; comparison
 * against SWI is done by the Prolog harness.
 */

#include "wam_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "setup_all.inc"

static int is_plain_atom(const char *s) {
    if (s == NULL || s[0] == '\0')
        return 0;
    if (!(s[0] >= 'a' && s[0] <= 'z'))
        return 0;
    for (const char *p = s + 1; *p; p++) {
        if (!(isalnum((unsigned char)*p) || *p == '_'))
            return 0;
    }
    return 1;
}

static void print_atom(const char *s) {
    if (s == NULL) {
        fputs("''", stdout);
        return;
    }
    if (strcmp(s, "[]") == 0) {
        fputs("[]", stdout);
        return;
    }
    if (is_plain_atom(s)) {
        fputs(s, stdout);
        return;
    }
    putchar('\'');
    for (const char *p = s; *p; p++) {
        if (*p == '\\' || *p == '\'')
            putchar('\\');
        putchar(*p);
    }
    putchar('\'');
}

static void print_term(WamState *state, WamValue v, int depth);

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

static int is_list_functor(const char *name) {
    return strcmp(name, ".") == 0 || strcmp(name, "[|]") == 0;
}

static void print_list_from_cells(WamState *state, WamValue head, WamValue tail, int depth) {
    putchar('[');
    print_term(state, head, depth + 1);
    for (;;) {
        WamValue *td = wam_deref_ptr(state, &tail);
        if (td->tag == VAL_ATOM && td->data.atom && strcmp(td->data.atom, "[]") == 0)
            break;
        if (td->tag == VAL_LIST) {
            putchar(',');
            print_term(state, state->H_array[td->data.ref_addr], depth + 1);
            tail = state->H_array[td->data.ref_addr + 1];
            continue;
        }
        if (td->tag == VAL_STR) {
            WamValue *fn = &state->H_array[td->data.ref_addr];
            char name[128];
            int ar = 0;
            if (fn->tag == VAL_ATOM && fn->data.atom &&
                split_functor(fn->data.atom, name, sizeof name, &ar) &&
                ar == 2 && is_list_functor(name)) {
                putchar(',');
                print_term(state, state->H_array[td->data.ref_addr + 1], depth + 1);
                tail = state->H_array[td->data.ref_addr + 2];
                continue;
            }
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

static void emit_outcome(const char *id, int rc, WamState *state, int result_reg) {
    printf("CASE %s\n", id);
    if (rc == WAM_ERR_OOB) {
        printf("STATUS missing_predicate\n");
        return;
    }
    if (rc == WAM_HALT) {
        printf("STATUS fail\n");
        return;
    }
    if (rc != 0) {
        printf("STATUS runtime_error\n");
        printf("RC %d\n", rc);
        if (rc == WAM_ERR_UNSUPPORTED || state->error == WAM_ERR_UNSUPPORTED) {
            printf("KIND unsupported_builtin\n");
            if (state->error_op != NULL)
                printf("BUILTIN %s\n", state->error_op);
            printf("ARITY %d\n", state->error_arity);
        }
        return;
    }
    printf("STATUS ok\n");
    printf("TERM ");
    print_term(state, state->A[result_reg], 0);
    putchar('\n');
}

static void reset_state(WamState *state) {
    wam_free_state(state);
    wam_state_init(state);
    setup_all_predicates(state);
}

static WamValue make_list1(WamState *state, WamValue head) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = head;
    state->H_array[state->H++] = val_atom("[]");
    return list;
}

static WamValue make_struct1(WamState *state, const char *functor, WamValue arg) {
    WamValue structure;
    structure.tag = VAL_STR;
    structure.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(functor);
    state->H_array[state->H++] = arg;
    return structure;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_all_predicates(&state);

    {
        WamValue args[2] = { val_atom("b"), val_unbound("Y") };
        int rc = wam_run_predicate(&state, "idx_p/2", args, 2);
        emit_outcome("first_b", rc, &state, 1);
    }

    reset_state(&state);
    {
        WamValue args[2] = { val_unbound("X"), val_unbound("Y") };
        int rc = wam_run_predicate(&state, "idx_p/2", args, 2);
        emit_outcome("first_var", rc, &state, 0);
        if (rc == 0) {
            printf("TERM2 ");
            print_term(&state, state.A[1], 0);
            putchar('\n');
        }
    }

    reset_state(&state);
    {
        WamValue args[2] = { val_atom("z"), val_unbound("Y") };
        int rc = wam_run_predicate(&state, "idx_p/2", args, 2);
        emit_outcome("miss_z", rc, &state, 1);
    }

    reset_state(&state);
    {
        WamValue args[1] = { val_unbound("Ys") };
        int rc = wam_run_predicate(&state, "idx_all_b/1", args, 1);
        emit_outcome("all_b", rc, &state, 0);
    }

    reset_state(&state);
    {
        WamValue args[1] = { val_unbound("Ps") };
        int rc = wam_run_predicate(&state, "idx_all_var/1", args, 1);
        emit_outcome("all_var", rc, &state, 0);
    }

    reset_state(&state);
    {
        WamValue args[1] = { val_unbound("Ps") };
        int rc = wam_run_predicate(&state, "idx_all_wrap/1", args, 1);
        emit_outcome("all_wrap", rc, &state, 0);
    }

    reset_state(&state);
    {
        WamValue args[1] = { val_unbound("Ys") };
        int rc = wam_run_predicate(&state, "idx_all_guard/1", args, 1);
        emit_outcome("all_guard", rc, &state, 0);
    }

    reset_state(&state);
    {
        WamValue lst = make_list1(&state, val_atom("x"));
        WamValue args[2] = { lst, val_unbound("Ys") };
        int rc = wam_run_predicate(&state, "idx_all_list/2", args, 2);
        emit_outcome("all_list", rc, &state, 1);
    }

    reset_state(&state);
    {
        WamValue args[1] = { val_unbound("Ys") };
        int rc = wam_run_predicate(&state, "idx_all_s/1", args, 1);
        emit_outcome("all_s", rc, &state, 0);
    }

    reset_state(&state);
    {
        WamValue lst = make_list1(&state, val_atom("x"));
        WamValue args[2] = { lst, val_unbound("Y") };
        int rc = wam_run_predicate(&state, "idx_list/2", args, 2);
        emit_outcome("first_list", rc, &state, 1);
    }

    reset_state(&state);
    {
        WamValue bar = make_struct1(&state, "bar/1", val_atom("z"));
        WamValue args[2] = { bar, val_unbound("Y") };
        int rc = wam_run_predicate(&state, "idx_s/2", args, 2);
        emit_outcome("first_struct", rc, &state, 1);
    }

    wam_free_state(&state);
    return 0;
}
