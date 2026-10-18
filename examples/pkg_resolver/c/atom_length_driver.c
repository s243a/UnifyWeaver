/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Driver for tests/test_wam_c_atom_length.pl. Builds atom_length/2 queries,
 * prints ground lengths, and checks rollback, continuation, and rejected
 * input shapes in compiled C. Optional argv "oracle" emits a representative
 * f64 grid for comparison against atom_length_float_oracle.rs.
 */

#include "wam_runtime.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_wam_atom_length_q_2(WamState *state);
void setup_wam_atom_length_continuation_3(WamState *state);
void setup_wam_atom_length_bind_control_1(WamState *state);
void setup_wam_atom_length_backtrack_1(WamState *state);
void setup_wam_atom_length_mismatch_positive_1(WamState *state);
void setup_wam_atom_length_mismatch_rollback_1(WamState *state);

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

static WamValue mkstr1(WamState *s, const char *functor, WamValue a) {
    ensure_h(s, 2);
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom(functor);
    s->H_array[s->H++] = a;
    return term;
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

static void print_term(WamState *state, WamValue v, int depth);

static void print_list_from_cells(WamState *state, WamValue head, WamValue tail, int depth) {
    putchar('[');
    print_term(state, head, depth + 1);
    for (;;) {
        WamValue *td = wam_deref_ptr(state, &tail);
        if (td->tag == VAL_ATOM && td->data.atom && strcmp(td->data.atom, "[]") == 0)
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
        if (fn->tag == VAL_ATOM && fn->data.atom) {
            print_atom(fn->data.atom);
            putchar('(');
            print_term(state, state->H_array[d->data.ref_addr + 1], depth + 1);
            putchar(')');
        } else {
            fputs("'<struct>'", stdout);
        }
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

static WamValue make_out_ref(WamState *s, const char *name) {
    ensure_h(s, 1);
    int addr = s->H++;
    s->H_array[addr] = val_unbound(name);
    WamValue ref;
    ref.tag = VAL_REF;
    ref.data.ref_addr = addr;
    return ref;
}

static int run_atom_length(WamState *s, WamValue atom, WamValue len) {
    WamValue args[2] = { atom, len };
    return wam_run_predicate(s, "wam_atom_length_q/2", args, 2);
}

static int same_int(WamValue *cell, int n) {
    return cell->tag == VAL_INT && cell->data.integer == n;
}

static int same_atom(WamValue *cell, const char *name) {
    return cell->tag == VAL_ATOM && cell->data.atom &&
           strcmp(cell->data.atom, name) == 0;
}

static void run_ground(WamState *s, const char *id, WamValue atom) {
    WamValue n_ref = make_out_ref(s, "N");
    int rc = run_atom_length(s, atom, n_ref);
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

static void run_float_oracle_grid(WamState *s) {
    static const uint64_t mants[] = {0, 1, 2, 3, 1ULL << 51, (1ULL << 52) - 1};
    uint32_t exp;
    size_t i;
    for (exp = 0; exp < 2048u; exp++) {
        for (i = 0; i < sizeof(mants) / sizeof(mants[0]); i++) {
            uint64_t frac = mants[i];
            uint64_t bits;
            double f;
            char id[40];
            if (exp == 0x7FFu && frac > 1)
                frac = 1;
            bits = ((uint64_t)exp << 52) | frac;
            memcpy(&f, &bits, sizeof f);
            snprintf(id, sizeof id, "oracle_%016llx", (unsigned long long)bits);
            run_ground(s, id, val_float(f));
            if (exp != 0x7FFu || frac == 0) {
                bits |= 1ULL << 63;
                memcpy(&f, &bits, sizeof f);
                snprintf(id, sizeof id, "oracle_%016llx", (unsigned long long)bits);
                run_ground(s, id, val_float(f));
            }
        }
    }
    {
        static const double powers[] = {1e-6, 1.0, 1e6, 1e23, 1e24, 1e28, 1e100};
        for (i = 0; i < sizeof(powers) / sizeof(powers[0]); i++) {
            int delta;
            uint64_t center;
            memcpy(&center, &powers[i], sizeof center);
            for (delta = -1; delta <= 1; delta++) {
                uint64_t bits = (uint64_t)((int64_t)center + delta);
                double f;
                char id[40];
                memcpy(&f, &bits, sizeof f);
                snprintf(id, sizeof id, "power_%016llx", (unsigned long long)bits);
                run_ground(s, id, val_float(f));
                bits |= 1ULL << 63;
                memcpy(&f, &bits, sizeof f);
                snprintf(id, sizeof id, "power_%016llx", (unsigned long long)bits);
                run_ground(s, id, val_float(f));
            }
        }
    }
}

int main(int argc, char **argv) {
    WamState state;
    wam_state_init(&state);
    setup_wam_atom_length_q_2(&state);
    setup_wam_atom_length_continuation_3(&state);
    setup_wam_atom_length_bind_control_1(&state);
    setup_wam_atom_length_backtrack_1(&state);
    setup_wam_atom_length_mismatch_positive_1(&state);
    setup_wam_atom_length_mismatch_rollback_1(&state);

    if (argc >= 2 && strcmp(argv[1], "oracle") == 0) {
        run_float_oracle_grid(&state);
        wam_free_state(&state);
        return 0;
    }

    /* 1. Ground atom cases (SWI oracle) */
    run_ground(&state, "ascii", val_atom("hello"));
    run_ground(&state, "empty", val_atom(""));
    run_ground(&state, "multibyte", val_atom("caf\u00e9"));

    /* 2. Rust-parity atomic text forms */
    run_ground(&state, "integer_input", val_int(42));
    run_ground(&state, "float_input", val_float(3.5));
    run_ground(&state, "true_atom", val_atom("true"));
    run_ground(&state, "false_atom", val_atom("false"));
    run_ground(&state, "empty_list_atom", val_atom("[]"));

    /* 2b. Rust Display length probes (named; grid is argv oracle) */
    run_ground(&state, "float_1e6", val_float(1000000.0));
    run_ground(&state, "float_1e_minus_6", val_float(0.000001));
    run_ground(&state, "float_1_23456789", val_float(1.23456789));
    run_ground(&state, "float_1e100", val_float(1e100));
    {
        double min_sub;
        uint64_t bits = 1;
        memcpy(&min_sub, &bits, sizeof min_sub);
        run_ground(&state, "float_min_subnormal", val_float(min_sub));
    }
    run_ground(&state, "float_neg", val_float(-3.5));
    run_ground(&state, "float_zero", val_float(0.0));
    run_ground(&state, "float_neg_zero", val_float(-0.0));
    run_ground(&state, "float_inf", val_float(INFINITY));
    run_ground(&state, "float_neg_inf", val_float(-INFINITY));
    run_ground(&state, "float_nan", val_float(NAN));
    run_ground(&state, "float_tenth", val_float(0.1));
    run_ground(&state, "float_one", val_float(1.0));
    run_ground(&state, "float_pow2", val_float(2.0));
    run_ground(&state, "float_three_tenths", val_float(0.3));

    /* 3. Prebound matching length */
    {
        int rc = run_atom_length(&state, val_atom("hello"), val_int(5));
        emit_token("prebound_match", (rc == 0 && state.error == 0) ? "ok" : "fail",
                   (rc == 0 && state.error == 0) ? "prebound_ok" : "prebound_bad");
    }

    /* 4. Prebound mismatching length */
    {
        int h0 = state.H;
        int tr0 = state.TR;
        int rc = run_atom_length(&state, val_atom("hello"), val_int(4));
        int rolled = state.H == h0 && state.TR == tr0 && state.error == 0;
        if (rc == WAM_HALT && rolled)
            emit_token("prebound_mismatch", "fail", "prebound_mismatch_ok");
        else
            emit_token("prebound_mismatch", rc == 0 ? "ok" : "runtime_error",
                       "prebound_mismatch_bad");
    }

    /* 5. Preserve input atom cell */
    {
        WamValue atom = val_atom("abc");
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_atom_length(&state, atom, n_ref);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &atom), "abc") &&
                 same_int(wam_deref_ptr(&state, &n_ref), 3);
        emit_token("preserve_input", ok ? "ok" : "fail",
                   ok ? "preserve_ok" : "preserve_bad");
    }

    /* 6. Reference chain on first argument */
    {
        WamValue alias = wam_make_ref(&state);
        WamValue bound = val_atom("hi");
        int alias_ok = wam_unify(&state, &alias, &bound);
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = alias_ok ? run_atom_length(&state, alias, n_ref) : WAM_HALT;
        int ok = alias_ok && rc == 0 && state.error == 0 &&
                 same_int(wam_deref_ptr(&state, &n_ref), 2);
        emit_token("reference_chain", ok ? "ok" : "fail",
                   ok ? "ref_chain_ok" : "ref_chain_bad");
    }

    /* 7. Repeated calls on one state */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_atom_length_q_2(&st);
        int ok = 1;
        for (int i = 0; i < 3; i++) {
            WamValue n_ref = make_out_ref(&st, "N");
            int rc = run_atom_length(&st, val_atom("ab"), n_ref);
            if (rc != 0 || st.error != 0 || !same_int(wam_deref_ptr(&st, &n_ref), 2)) {
                ok = 0;
                break;
            }
        }
        emit_token("repeated_calls", ok ? "ok" : "fail",
                   ok ? "repeated_ok" : "repeated_bad");
        wam_free_state(&st);
    }

    /* 8. Rejected inputs */
    {
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_atom_length(&state, val_unbound("A"), n_ref);
        emit_token("unbound_first", (rc == WAM_HALT && state.error == 0) ? "fail" : "runtime_error",
                   (rc == WAM_HALT && state.error == 0) ? "unbound_ok" : "unbound_bad");
    }
    {
        WamValue n_ref = make_out_ref(&state, "N");
        int rc = run_atom_length(&state, mkstr1(&state, "f/1", val_atom("a")), n_ref);
        emit_token("compound_input", (rc == WAM_HALT && state.error == 0) ? "fail" : "runtime_error",
                   (rc == WAM_HALT && state.error == 0) ? "compound_ok" : "compound_bad");
    }
    {
        WamValue n_ref = make_out_ref(&state, "N");
        WamValue lst = cons(&state, val_atom("a"), val_atom("[]"));
        int rc = run_atom_length(&state, lst, n_ref);
        emit_token("nonempty_list", (rc == WAM_HALT && state.error == 0) ? "fail" : "runtime_error",
                   (rc == WAM_HALT && state.error == 0) ? "list_ok" : "list_bad");
    }

    /* 9. Bind control via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_length_bind_control/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_int(wam_deref_ptr(&state, &out_ref), 3);
        emit_token("bind_control", ok ? "ok" : "fail",
                   ok ? "bind_control_ok" : "bind_control_bad");
    }

    /* 10. Backtracking rollback */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_length_backtrack/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("backtrack_rollback", ok ? "ok" : "fail",
                   ok ? "backtrack_ok" : "backtrack_bad");
    }

    /* 11. Mismatch positive control: successful unify binds Out to 5 */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_length_mismatch_positive/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_int(wam_deref_ptr(&state, &out_ref), 5);
        emit_token("mismatch_positive", ok ? "ok" : "fail",
                   ok ? "mismatch_positive_ok" : "mismatch_positive_bad");
    }

    /* 12. Mismatch rollback via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_length_mismatch_rollback/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "mismatch_rollback_ok" : "mismatch_rollback_bad");
    }

    /* 13. Direct C rollback: prebound mismatch must not grow heap/trail */
    {
        WamValue n = wam_make_ref(&state);
        int tr_before = state.TR;
        int rc_pos = run_atom_length(&state, val_atom("a"), n);
        int bind_ok = rc_pos == 0 && same_int(wam_deref_ptr(&state, &n), 1);
        unwind_trail(&state, tr_before);

        int h_before = state.H;
        int tr_mis = state.TR;
        int rc_mis = run_atom_length(&state, val_atom("hello"), val_int(3));
        int rollback_ok = rc_mis == WAM_HALT && state.error == 0 &&
                          state.H == h_before && state.TR == tr_mis;
        int ok = bind_ok && rollback_ok;
        emit_token("c_unifier_rollback", ok ? "ok" : "fail",
                   ok ? "c_rollback_ok" : "c_rollback_bad");
    }

    /* 14. Caller continuation */
    {
        WamValue s1_ref = make_out_ref(&state, "S1");
        WamValue s2_ref = make_out_ref(&state, "S2");
        WamValue args[3] = { val_atom("foo"), s1_ref, s2_ref };
        int rc = wam_run_predicate(&state, "wam_atom_length_continuation/3", args, 3);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &s1_ref), "start") &&
                 same_atom(wam_deref_ptr(&state, &s2_ref), "done");
        emit_token("caller_continuation", ok ? "ok" : "fail",
                   ok ? "continuation_ok" : "continuation_bad");
    }

    /* 15. Repeated mismatch rollback */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_atom_length_q_2(&st);
        WamValue n = wam_make_ref(&st);
        int baseline_H = st.H;
        int baseline_TR = st.TR;
        int all_failed = 1;
        for (int iter = 0; iter < 5; iter++) {
            int rc = run_atom_length(&st, val_atom("hello"), val_int(3));
            if (rc != WAM_HALT || st.error != 0 ||
                st.H != baseline_H || st.TR != baseline_TR) {
                all_failed = 0;
                break;
            }
        }
        int rc_good = run_atom_length(&st, val_atom("hello"), n);
        int good_ok = rc_good == 0 && st.error == 0 &&
                      same_int(wam_deref_ptr(&st, &n), 5);
        int ok = all_failed && good_ok;
        emit_token("repeated_mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "repeated_rollback_ok" : "repeated_rollback_bad");
        wam_free_state(&st);
    }

    wam_free_state(&state);
    return 0;
}
