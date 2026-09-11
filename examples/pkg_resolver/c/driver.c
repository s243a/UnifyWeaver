/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * driver.c -- run one named smoke wrapper and print the grounded
 * selection term. Query arguments live in the compiled wrappers; Sel
 * is unbound. Exit 0 is a produced logical outcome (ok or fail), not
 * a comparison pass.
 *
 * Exit status:
 *   0  STATUS ok / STATUS fail (WAM success or legitimate failure)
 *   2  usage / unknown case
 *   3  missing predicate (WAM_ERR_OOB / generator omission)
 *   4  unexpected runtime status (including unsupported builtin)
 */

#include "wam_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "setup_all.inc"

static const char *pred_for_case(const char *id) {
    if (strcmp(id, "empty_requests") == 0)
        return "smoke_empty_requests/1";
    if (strcmp(id, "single_package") == 0)
        return "smoke_single_package/1";
    if (strcmp(id, "backtrack_conflict_deeper") == 0)
        return "smoke_backtrack_conflict_deeper/1";
    if (strcmp(id, "unsatisfiable_missing") == 0)
        return "smoke_unsatisfiable_missing/1";
    return NULL;
}

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

static int is_graphic_atom(const char *s) {
    if (s == NULL || s[0] == '\0')
        return 0;
    for (const char *p = s; *p; p++) {
        if (strchr("#$&*+-./:<=>?@^~\\\\", *p) == NULL)
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
    if (is_plain_atom(s) || is_graphic_atom(s)) {
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: smoke_runner CASE_ID\n");
        return 2;
    }
    const char *case_id = argv[1];
    const char *pred = pred_for_case(case_id);
    if (pred == NULL) {
        fprintf(stderr, "unknown case: %s\n", case_id);
        return 2;
    }

    printf("CASE %s\n", case_id);

    WamState state;
    wam_state_init(&state);
    setup_all_predicates(&state);

    WamValue args[1];
    args[0] = val_unbound("Sel");
    int rc = wam_run_predicate(&state, pred, args, 1);

    if (rc == WAM_ERR_OOB) {
        printf("STATUS missing_predicate\n");
        printf("PRED %s\n", pred);
        wam_free_state(&state);
        return 3;
    }
    if (rc == WAM_HALT) {
        printf("STATUS fail\n");
        wam_free_state(&state);
        return 0;
    }
    if (rc != 0) {
        printf("STATUS runtime_error\n");
        printf("RC %d\n", rc);
        if (rc == WAM_ERR_UNSUPPORTED || state.error == WAM_ERR_UNSUPPORTED) {
            printf("KIND unsupported_builtin\n");
            if (state.error_op != NULL)
                printf("BUILTIN %s\n", state.error_op);
            printf("ARITY %d\n", state.error_arity);
        }
        wam_free_state(&state);
        return 4;
    }

    printf("STATUS ok\n");
    printf("TERM ");
    print_term(&state, state.A[0], 0);
    putchar('\n');
    wam_free_state(&state);
    return 0;
}
