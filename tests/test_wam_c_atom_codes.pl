:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C atom_codes/2: generated/compiled C vs SWI for forward
% atom-to-codes mode (ASCII, empty, multibyte code points).
% Compatible and incompatible prebound outputs, rollback after mismatch,
% caller continuation, and variable identity are verified.
% Unbound atom, unsupported reverse mode, and non-atom inputs
% explicitly diagnose WAM_ERR_UNSUPPORTED, not silent logical failure.
%
%   swipl -q -g run_tests -t halt tests/test_wam_c_atom_codes.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).
:- use_module(library(lists), [member/2]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.
:- dynamic user:wam_atom_codes_q/2.
:- dynamic user:wam_atom_codes_continuation/3.
:- dynamic user:wam_atom_codes_bind_control/1.
:- dynamic user:wam_atom_codes_backtrack/1.
:- dynamic user:wam_atom_codes_mismatch_positive/1.
:- dynamic user:wam_atom_codes_mismatch_rollback/1.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

wam_c_temp_root('/tmp').

wam_c_temp_path(Prefix, Stamp, Path) :-
    wam_c_temp_root(Root),
    format(atom(Path), '~w/~w_~w', [Root, Prefix, Stamp]).

gcc_available :-
    catch(process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          _, fail),
    process_wait(Pid, exit(0)).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).

cleanup_atom_codes_preds :-
    retractall(user:wam_atom_codes_q(_, _)),
    retractall(user:wam_atom_codes_continuation(_, _, _)),
    retractall(user:wam_atom_codes_bind_control(_)),
    retractall(user:wam_atom_codes_backtrack(_)),
    retractall(user:wam_atom_codes_mismatch_positive(_)),
    retractall(user:wam_atom_codes_mismatch_rollback(_)).

setup_atom_codes_preds :-
    cleanup_atom_codes_preds,
    assertz((user:wam_atom_codes_q(A, C) :- atom_codes(A, C))),
    assertz((user:wam_atom_codes_continuation(A, S1, S2) :-
                 S1 = start, atom_codes(A, _), S2 = done)),
    assertz((user:wam_atom_codes_bind_control(Out) :-
                 atom_codes(foo, C), Out = C)),
    assertz((user:wam_atom_codes_backtrack(Out) :-
                 ( atom_codes(foo, C), C == [99, 99], fail
                 ; ( var(C) -> Out = restored ; Out = bad )
                 ), Out == restored)),
    assertz((user:wam_atom_codes_mismatch_positive(Out) :-
                 atom_codes(hello, C), Out = C)),
    assertz((user:wam_atom_codes_mismatch_rollback(Out) :-
                 ( atom_codes(hello, C), C = [99]
                 ; ( var(C) -> Out = restored ; Out = bad )
                 ), Out == restored)).

ground_input(ascii, hello).
ground_input(empty, '').
ground_input(multibyte_cafe, 'café').
ground_input(multibyte_greek, 'λ').
ground_input(multibyte_cjk, '日本語').
ground_input(multibyte_emoji, '🚀').

ground_swi(Id, Expected) :-
    ground_input(Id, Atom),
    atom_codes(Atom, Expected).

token_swi(prebound_match, ok, prebound_ok) :-
    atom_codes(hello, [104, 101, 108, 108, 111]).
token_swi(prebound_empty_match, ok, prebound_empty_ok) :-
    atom_codes('', []).
token_swi(aliased_var_match, ok, aliased_match_ok) :-
    atom_codes(aba, [X, 98, X]), X == 97.
token_swi(prebound_mismatch_content, fail, mismatch_content_ok) :-
    \+ atom_codes(hello, [104, 101, 108, 108, 99]).
token_swi(prebound_mismatch_length, fail, mismatch_length_ok) :-
    \+ atom_codes(hello, [104, 101, 108, 108]).
token_swi(prebound_empty_mismatch, fail, empty_mismatch_ok) :-
    \+ atom_codes('', [97]).
token_swi(aliased_var_mismatch, fail, aliased_mismatch_ok) :-
    \+ atom_codes(abc, [X, 98, X]).
token_swi(preserve_input, ok, preserve_ok).
token_swi(reference_chain, ok, ref_chain_ok).
token_swi(repeated_calls, ok, repeated_ok).
token_swi(bind_control, ok, bind_control_ok) :-
    user:wam_atom_codes_bind_control([102, 111, 111]).
token_swi(backtrack_rollback, ok, backtrack_ok) :-
    user:wam_atom_codes_backtrack(restored).
token_swi(mismatch_positive, ok, mismatch_positive_ok) :-
    user:wam_atom_codes_mismatch_positive([104, 101, 108, 108, 111]).
token_swi(mismatch_rollback, ok, mismatch_rollback_ok) :-
    user:wam_atom_codes_mismatch_rollback(restored).
token_swi(c_unifier_rollback, ok, c_rollback_ok).
token_swi(caller_continuation, ok, continuation_ok) :-
    S1 = start, atom_codes(foo, _), S2 = done,
    S1 == start, S2 == done.
token_swi(repeated_mismatch_rollback, ok, repeated_rollback_ok).
token_swi(unbound_atom, ok, unbound_atom_ok).
token_swi(reverse_mode, ok, reverse_mode_ok).
token_swi(compound_input, ok, compound_ok).
token_swi(nonempty_list, ok, nonempty_list_ok).
token_swi(integer_input, ok, integer_ok).
token_swi(distinguish_mismatch_vs_unsupported, ok, distinguish_ok).
token_swi(prebound_heap_stable, ok, heap_stable_ok).
token_swi(malformed_utf8, ok, malformed_ok).
token_swi(heap_alloc_failure, ok, alloc_failure_ok).

parse_c_cases([], []).
parse_c_cases([Line|Rest], Cases) :-
    (   sub_string(Line, 0, _, _, "CASE ")
    ->  sub_string(Line, 5, _, 0, Id0),
        atom_string(Id, Id0),
        parse_c_status(Rest, Id, Case, Rest2),
        Cases = [Case|Cases2],
        parse_c_cases(Rest2, Cases2)
    ;   parse_c_cases(Rest, Cases)
    ).

parse_c_status([], Id, c_case(Id, missing_status, _), []).
parse_c_status([Line|Rest], Id, Case, RestOut) :-
    (   sub_string(Line, 0, _, _, "STATUS ")
    ->  sub_string(Line, 7, _, 0, St0),
        atom_string(Status, St0),
        parse_c_term(Rest, Id, Status, Case, RestOut)
    ;   sub_string(Line, 0, _, _, "CASE ")
    ->  Case = c_case(Id, missing_status, _),
        RestOut = [Line|Rest]
    ;   parse_c_status(Rest, Id, Case, RestOut)
    ).

parse_c_term([], Id, Status, c_case(Id, Status, _), []).
parse_c_term([Line|Rest], Id, Status, Case, RestOut) :-
    (   sub_string(Line, 0, _, _, "TERM ")
    ->  sub_string(Line, 5, _, 0, TermStr),
        (   catch(term_string(Term, TermStr), _, fail)
        ->  true
        ;   Term = parse_error(TermStr)
        ),
        Case = c_case(Id, Status, Term),
        RestOut = Rest
    ;   Status == fail
    ->  Case = c_case(Id, Status, _),
        RestOut = [Line|Rest]
    ;   sub_string(Line, 0, _, _, "CASE ")
    ->  Case = c_case(Id, Status, _),
        RestOut = [Line|Rest]
    ;   parse_c_term(Rest, Id, Status, Case, RestOut)
    ).

same_success(Expected, Actual) :-
    ground(Expected),
    ground(Actual),
    Actual == Expected.

compare_ground(Id, CCases) :-
    ground_swi(Id, Expected),
    (   member(c_case(Id, CStatus, CTerm), CCases)
    ->  true
    ;   fail_test(Id, 'missing C output'),
        fail
    ),
    (   CStatus == ok,
        same_success(Expected, CTerm)
    ->  pass(Id)
    ;   format(atom(R), 'SWI ~q vs C ~w ~q', [Expected, CStatus, CTerm]),
        fail_test(Id, R),
        fail
    ).

compare_token(Id, CCases) :-
    (   token_swi(Id, SWIStatus, Token)
    ->  true
    ;   fail_test(Id, 'missing SWI token oracle'),
        fail
    ),
    (   member(c_case(Id, CStatus, CTerm), CCases)
    ->  true
    ;   fail_test(Id, 'missing C output'),
        fail
    ),
    (   CStatus == SWIStatus,
        CTerm == Token
    ->  pass(Id)
    ;   format(atom(R), 'expected ~w ~q vs C ~w ~q',
               [SWIStatus, Token, CStatus, CTerm]),
        fail_test(Id, R),
        fail
    ).

test_generation_atom_codes_builtin :-
    Test = 'atom_codes/2: generated runtime contains atom_codes handler',
    (   compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(HelpersS, _, _, _, 'strcmp(op, "atom_codes/2")'),
        sub_string(HelpersS, _, _, _, 'wam_execute_atom_codes'),
        sub_string(HelpersS, _, _, _, 'wam_decode_utf8_codes'),
        sub_string(HelpersS, _, _, _, 'wam_build_code_list')
    ->  pass(Test)
    ;   fail_test(Test, 'atom_codes/2 handler missing from generated runtime')
    ).

test_wam_emits_atom_codes_builtin :-
    Test = 'atom_codes/2: WAM text emits builtin_call atom_codes/2',
    setup_atom_codes_preds,
    (   compile_predicate_to_wam(user:wam_atom_codes_q/2, [], Wam),
        sub_string(Wam, _, _, _, 'builtin_call atom_codes/2')
    ->  pass(Test)
    ;   fail_test(Test, 'wam_atom_codes_q/2 WAM missing builtin_call atom_codes/2')
    ).

test_wrong_answer_rejected :-
    Test = 'atom_codes/2: SWI comparator rejects a wrong answer',
    atom_codes(hello, SWI),
    Wrong = [104, 101, 108, 108],
    (   same_success(SWI, SWI),
        \+ same_success(SWI, Wrong),
        \+ same_success(SWI, _)
    ->  pass(Test)
    ;   fail_test(Test, 'actual comparator failed positive or negative controls')
    ).

compile_one(Pred, Code) :-
    compile_predicate_to_wam(Pred, [], Wam),
    compile_wam_predicate_to_c(Pred, Wam, [], Code).

test_compiled_c_atom_codes_matches_swi :-
    Test = 'atom_codes/2: compiled C solutions match SWI',
    (   gcc_available
    ->  (   run_compiled_c_atom_codes
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C atom_codes executable failed or mismatched')
        )
    ;   fail_test(Test, 'gcc unavailable; compiled behavior not verified')
    ).

run_process_output(Args, OutStr, Status) :-
    process_create(path(timeout), Args,
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, Status),
    (   ErrStr == ""
    ->  true
    ;   format(user_error, '~w', [ErrStr])
    ).

run_compiled_c_atom_codes :-
    setup_atom_codes_preds,
    compile_one(user:wam_atom_codes_q/2, QCode),
    compile_one(user:wam_atom_codes_continuation/3, ContCode),
    compile_one(user:wam_atom_codes_bind_control/1, BindCode),
    compile_one(user:wam_atom_codes_backtrack/1, BackCode),
    compile_one(user:wam_atom_codes_mismatch_positive/1, MisPosCode),
    compile_one(user:wam_atom_codes_mismatch_rollback/1, MismatchCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_atom_codes', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(DriverPath), '~w_driver.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    atomic_list_concat([QCode, "\n\n", ContCode, "\n\n", BindCode, "\n\n",
                        BackCode, "\n\n", MisPosCode, "\n\n", MismatchCode],
                       AllPredCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [AllPredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    atom_codes_driver_c_source(DriverCode),
    write_text_file(DriverPath, DriverCode),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc),
                   ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
                    RuntimePath, PredPath, DriverPath, '-Wl,--wrap=realloc', '-lm', '-o', ExePath],
                   [process(Pid)]),
    process_wait(Pid, GccStatus),
    format('gcc orig_exit=~w~n', [GccStatus]),
    GccStatus == exit(0),
    run_process_output(['10', ExePath], OutStr, RunStatus),
    format('runner orig_exit=~w~n', [RunStatus]),
    RunStatus == exit(0),
    split_string(OutStr, "\n", "", Lines),
    parse_c_cases(Lines, CCases),
    Ground = [ascii, empty, multibyte_cafe, multibyte_greek, multibyte_cjk, multibyte_emoji],
    Tokens = [prebound_match, prebound_empty_match, aliased_var_match,
              prebound_mismatch_content, prebound_mismatch_length,
              prebound_empty_mismatch,
              aliased_var_mismatch, preserve_input, reference_chain,
              repeated_calls, bind_control, backtrack_rollback,
              mismatch_positive, mismatch_rollback, c_unifier_rollback,
              caller_continuation, repeated_mismatch_rollback,
              unbound_atom, reverse_mode, compound_input, nonempty_list,
              integer_input, distinguish_mismatch_vs_unsupported,
              prebound_heap_stable, malformed_utf8, heap_alloc_failure],
    findall(Id, (member(Id, Ground), \+ compare_ground(Id, CCases)), GroundBads),
    findall(Id, (member(Id, Tokens), \+ compare_token(Id, CCases)), TokenBads),
    GroundBads == [],
    TokenBads == [].

atom_codes_driver_c_source(
'#include "wam_runtime.h"
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_wam_atom_codes_q_2(WamState *state);
void setup_wam_atom_codes_continuation_3(WamState *state);
void setup_wam_atom_codes_bind_control_1(WamState *state);
void setup_wam_atom_codes_backtrack_1(WamState *state);
void setup_wam_atom_codes_mismatch_positive_1(WamState *state);
void setup_wam_atom_codes_mismatch_rollback_1(WamState *state);

static int fail_next_realloc;
void *__real_realloc(void *ptr, size_t size);
void *__wrap_realloc(void *ptr, size_t size) {
    if (fail_next_realloc) {
        fail_next_realloc = 0;
        return NULL;
    }
    return __real_realloc(ptr, size);
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

static WamValue mkstr1(WamState *s, const char *functor, WamValue a) {
    ensure_h(s, 2);
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom(functor);
    s->H_array[s->H++] = a;
    return term;
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

static WamValue build_int_list(WamState *s, const int *items, int n) {
    WamValue cur = val_atom("[]");
    for (int i = n - 1; i >= 0; i--) {
        cur = cons(s, val_int(items[i]), cur);
    }
    return cur;
}

static int is_plain_atom(const char *str) {
    if (str == NULL || str[0] == 0)
        return 0;
    if (!(str[0] >= 97 && str[0] <= 122))
        return 0;
    for (const char *p = str + 1; *p; p++) {
        if (!((p[0] >= 97 && p[0] <= 122) ||
              (p[0] >= 65 && p[0] <= 90) ||
              (p[0] >= 48 && p[0] <= 57) ||
              p[0] == 95))
            return 0;
    }
    return 1;
}

static void print_atom(const char *str) {
    if (str == NULL) {
        putchar(39); putchar(39);
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
    putchar(39);
    for (const char *p = str; *p; p++) {
        if (*p == 92 || *p == 39)
            putchar(92);
        putchar(*p);
    }
    putchar(39);
}

static void print_term(WamState *state, WamValue v, int depth);

static void print_list_from_cells(WamState *state, WamValue head, WamValue tail, int depth) {
    putchar(91);
    print_term(state, head, depth + 1);
    for (;;) {
        WamValue *td = wam_deref_ptr(state, &tail);
        if (td->tag == VAL_ATOM && td->data.atom && strcmp(td->data.atom, "[]") == 0)
            break;
        int c = wam_cons_head_addr(state, td);
        if (c >= 0) {
            putchar(44);
            print_term(state, state->H_array[c], depth + 1);
            tail = state->H_array[c + 1];
            continue;
        }
        putchar(124);
        print_term(state, tail, depth + 1);
        break;
    }
    putchar(93);
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
        putchar(95);
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
            putchar(40);
            print_term(state, state->H_array[d->data.ref_addr + 1], depth + 1);
            putchar(41);
        } else {
            fputs("<struct>", stdout);
        }
        return;
    }
    case VAL_REF:
        putchar(95);
        return;
    default:
        fputs("<unknown>", stdout);
        return;
    }
}

static void emit_case(const char *id, const char *status, WamState *state, WamValue *term) {
    printf("CASE %s\\n", id);
    printf("STATUS %s\\n", status);
    if (term) {
        fputs("TERM ", stdout);
        print_term(state, *term, 0);
        putchar(10);
    }
}

static void emit_token(const char *id, const char *status, const char *token) {
    printf("CASE %s\\n", id);
    printf("STATUS %s\\n", status);
    printf("TERM %s\\n", token);
}

static int run_atom_codes(WamState *s, WamValue atom, WamValue codes) {
    WamValue args[2] = { atom, codes };
    return wam_run_predicate(s, "wam_atom_codes_q/2", args, 2);
}

static int same_atom(WamValue *cell, const char *name) {
    return cell->tag == VAL_ATOM && cell->data.atom &&
           strcmp(cell->data.atom, name) == 0;
}

static void run_ground(WamState *s, const char *id, WamValue atom) {
    WamValue c_ref = make_out_ref(s, "C");
    int rc = run_atom_codes(s, atom, c_ref);
    if (rc == 0 && s->error == 0) {
        emit_case(id, "ok", s, &c_ref);
        return;
    }
    if (rc == WAM_HALT && s->error == 0) {
        emit_case(id, "fail", s, NULL);
        return;
    }
    emit_case(id, "runtime_error", s, NULL);
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_atom_codes_q_2(&state);
    setup_wam_atom_codes_continuation_3(&state);
    setup_wam_atom_codes_bind_control_1(&state);
    setup_wam_atom_codes_backtrack_1(&state);
    setup_wam_atom_codes_mismatch_positive_1(&state);
    setup_wam_atom_codes_mismatch_rollback_1(&state);

    /* 1. Ground atom cases (SWI oracle) */
    run_ground(&state, "ascii", val_atom("hello"));
    run_ground(&state, "empty", val_atom(""));
    run_ground(&state, "multibyte_cafe", val_atom("café"));
    run_ground(&state, "multibyte_greek", val_atom("λ"));
    run_ground(&state, "multibyte_cjk", val_atom("日本語"));
    run_ground(&state, "multibyte_emoji", val_atom("🚀"));

    /* 2. Compatible prebound match */
    {
        static const int hello_codes[] = {104, 101, 108, 108, 111};
        WamValue pre = build_int_list(&state, hello_codes, 5);
        int rc = run_atom_codes(&state, val_atom("hello"), pre);
        emit_token("prebound_match", (rc == 0 && state.error == 0) ? "ok" : "fail",
                   (rc == 0 && state.error == 0) ? "prebound_ok" : "prebound_bad");
    }

    /* 3. Compatible prebound empty match */
    {
        int rc = run_atom_codes(&state, val_atom(""), val_atom("[]"));
        emit_token("prebound_empty_match", (rc == 0 && state.error == 0) ? "ok" : "fail",
                   (rc == 0 && state.error == 0) ? "prebound_empty_ok" : "prebound_empty_bad");
    }

    /* 4. Compatible prebound with shared variable */
    {
        WamValue x = wam_make_ref(&state);
        WamValue pre = cons(&state, x, cons(&state, val_int(98), cons(&state, x, val_atom("[]"))));
        int rc = run_atom_codes(&state, val_atom("aba"), pre);
        WamValue *x_cell = wam_deref_ptr(&state, &x);
        int ok = rc == 0 && state.error == 0 && x_cell->tag == VAL_INT && x_cell->data.integer == 97;
        emit_token("aliased_var_match", ok ? "ok" : "fail",
                   ok ? "aliased_match_ok" : "aliased_match_bad");
    }

    /* 5. Incompatible prebound content mismatch */
    {
        static const int bad_codes[] = {104, 101, 108, 108, 99};
        WamValue pre = build_int_list(&state, bad_codes, 5);
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_atom_codes(&state, val_atom("hello"), pre);
        int rolled = (state.H == h_before && state.TR == tr_before && state.error == 0);
        emit_token("prebound_mismatch_content", (rc == WAM_HALT && rolled) ? "fail" : "ok",
                   (rc == WAM_HALT && rolled) ? "mismatch_content_ok" : "mismatch_content_bad");
    }

    /* 6. Incompatible prebound length mismatch (short) */
    {
        static const int short_codes[] = {104, 101, 108, 108};
        WamValue pre = build_int_list(&state, short_codes, 4);
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_atom_codes(&state, val_atom("hello"), pre);
        int rolled = (state.H == h_before && state.TR == tr_before && state.error == 0);
        emit_token("prebound_mismatch_length", (rc == WAM_HALT && rolled) ? "fail" : "ok",
                   (rc == WAM_HALT && rolled) ? "mismatch_length_ok" : "mismatch_length_bad");
    }

    /* 7. Incompatible prebound non-list */
    {
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_atom_codes(&state, val_atom("hello"), val_int(42));
        int rolled = (state.H == h_before && state.TR == tr_before && state.error == 0);
        emit_token("prebound_mismatch_nonlist", (rc == WAM_HALT && rolled) ? "fail" : "ok",
                   (rc == WAM_HALT && rolled) ? "mismatch_nonlist_ok" : "mismatch_nonlist_bad");
    }

    /* 8. Incompatible prebound empty mismatch */
    {
        WamValue pre = cons(&state, val_int(97), val_atom("[]"));
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_atom_codes(&state, val_atom(""), pre);
        int rolled = (state.H == h_before && state.TR == tr_before && state.error == 0);
        emit_token("prebound_empty_mismatch", (rc == WAM_HALT && rolled) ? "fail" : "ok",
                   (rc == WAM_HALT && rolled) ? "empty_mismatch_ok" : "empty_mismatch_bad");
    }

    /* 9. Incompatible prebound with shared variable conflict */
    {
        WamValue x = wam_make_ref(&state);
        WamValue pre = cons(&state, x, cons(&state, val_int(98), cons(&state, x, val_atom("[]"))));
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_atom_codes(&state, val_atom("abc"), pre);
        WamValue *x_cell = wam_deref_ptr(&state, &x);
        int rolled = (rc == WAM_HALT && state.error == 0 &&
                      state.H == h_before && state.TR == tr_before &&
                      val_is_unbound(*x_cell));
        emit_token("aliased_var_mismatch", rolled ? "fail" : "ok",
                   rolled ? "aliased_mismatch_ok" : "aliased_mismatch_bad");
    }

    /* 10. Preserve input atom cell */
    {
        WamValue atom = val_atom("abc");
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_atom_codes(&state, atom, c_ref);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &atom), "abc");
        emit_token("preserve_input", ok ? "ok" : "fail",
                   ok ? "preserve_ok" : "preserve_bad");
    }

    /* 11. Reference chain on input atom */
    {
        WamValue alias = wam_make_ref(&state);
        WamValue bound = val_atom("hi");
        int alias_ok = wam_unify(&state, &alias, &bound);
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = alias_ok ? run_atom_codes(&state, alias, c_ref) : WAM_HALT;
        int ok = alias_ok && rc == 0 && state.error == 0;
        emit_token("reference_chain", ok ? "ok" : "fail",
                   ok ? "ref_chain_ok" : "ref_chain_bad");
    }

    /* 12. Repeated calls on one state */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_atom_codes_q_2(&st);
        int ok = 1;
        for (int i = 0; i < 3; i++) {
            WamValue c_ref = make_out_ref(&st, "C");
            int rc = run_atom_codes(&st, val_atom("ab"), c_ref);
            if (rc != 0 || st.error != 0) {
                ok = 0;
                break;
            }
        }
        emit_token("repeated_calls", ok ? "ok" : "fail",
                   ok ? "repeated_ok" : "repeated_bad");
        wam_free_state(&st);
    }

    /* 13. Bind control via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_codes_bind_control/1", args, 1);
        int ok = rc == 0 && state.error == 0;
        emit_token("bind_control", ok ? "ok" : "fail",
                   ok ? "bind_control_ok" : "bind_control_bad");
    }

    /* 14. Backtracking rollback via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_codes_backtrack/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("backtrack_rollback", ok ? "ok" : "fail",
                   ok ? "backtrack_ok" : "backtrack_bad");
    }

    /* 15. Mismatch positive control via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_codes_mismatch_positive/1", args, 1);
        int ok = rc == 0 && state.error == 0;
        emit_token("mismatch_positive", ok ? "ok" : "fail",
                   ok ? "mismatch_positive_ok" : "mismatch_positive_bad");
    }

    /* 16. Mismatch rollback via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_atom_codes_mismatch_rollback/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "mismatch_rollback_ok" : "mismatch_rollback_bad");
    }

    /* 17. Direct C unifier rollback */
    {
        WamValue c = wam_make_ref(&state);
        int tr_before = state.TR;
        int rc_pos = run_atom_codes(&state, val_atom("a"), c);
        int bind_ok = (rc_pos == 0 && state.error == 0);
        unwind_trail(&state, tr_before);

        int h_before = state.H;
        int tr_mis = state.TR;
        int rc_mis = run_atom_codes(&state, val_atom("hello"), val_int(3));
        int rollback_ok = (rc_mis == WAM_HALT && state.error == 0 &&
                           state.H == h_before && state.TR == tr_mis);
        int ok = bind_ok && rollback_ok;
        emit_token("c_unifier_rollback", ok ? "ok" : "fail",
                   ok ? "c_rollback_ok" : "c_rollback_bad");
    }

    /* 18. Caller continuation */
    {
        WamValue s1_ref = make_out_ref(&state, "S1");
        WamValue s2_ref = make_out_ref(&state, "S2");
        WamValue args[3] = { val_atom("foo"), s1_ref, s2_ref };
        int rc = wam_run_predicate(&state, "wam_atom_codes_continuation/3", args, 3);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &s1_ref), "start") &&
                 same_atom(wam_deref_ptr(&state, &s2_ref), "done");
        emit_token("caller_continuation", ok ? "ok" : "fail",
                   ok ? "continuation_ok" : "continuation_bad");
    }

    /* 19. Repeated mismatch rollback */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_atom_codes_q_2(&st);
        WamValue c = wam_make_ref(&st);
        int baseline_H = st.H;
        int baseline_TR = st.TR;
        int all_failed = 1;
        for (int iter = 0; iter < 5; iter++) {
            int rc = run_atom_codes(&st, val_atom("hello"), val_int(3));
            if (rc != WAM_HALT || st.error != 0 ||
                st.H != baseline_H || st.TR != baseline_TR) {
                all_failed = 0;
                break;
            }
        }
        int rc_good = run_atom_codes(&st, val_atom("hello"), c);
        int good_ok = (rc_good == 0 && st.error == 0);
        int ok = all_failed && good_ok;
        emit_token("repeated_mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "repeated_rollback_ok" : "repeated_rollback_bad");
        wam_free_state(&st);
    }

    /* 20. Unbound atom (both unbound) -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_atom_codes(&state, val_unbound("A"), c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED &&
                  state.error_op != NULL && strcmp(state.error_op, "atom_codes/2") == 0);
        emit_token("unbound_atom", ok ? "ok" : "fail",
                   ok ? "unbound_atom_ok" : "unbound_atom_bad");
        wam_clear_error(&state);
    }

    /* 21. Reverse mode (unbound atom, bound codes) -> WAM_ERR_UNSUPPORTED */
    {
        static const int ab_codes[] = {97, 98};
        WamValue codes = build_int_list(&state, ab_codes, 2);
        int rc = run_atom_codes(&state, val_unbound("A"), codes);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED &&
                  state.error_op != NULL && strcmp(state.error_op, "atom_codes/2") == 0);
        emit_token("reverse_mode", ok ? "ok" : "fail",
                   ok ? "reverse_mode_ok" : "reverse_mode_bad");
        wam_clear_error(&state);
    }

    /* 22. Compound input -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_atom_codes(&state, mkstr1(&state, "f/1", val_atom("a")), c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("compound_input", ok ? "ok" : "fail",
                   ok ? "compound_ok" : "compound_bad");
        wam_clear_error(&state);
    }

    /* 23. Non-empty list input -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        WamValue lst = cons(&state, val_int(97), val_atom("[]"));
        int rc = run_atom_codes(&state, lst, c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("nonempty_list", ok ? "ok" : "fail",
                   ok ? "nonempty_list_ok" : "nonempty_list_bad");
        wam_clear_error(&state);
    }

    /* 24. Integer input -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_atom_codes(&state, val_int(42), c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("integer_input", ok ? "ok" : "fail",
                   ok ? "integer_ok" : "integer_bad");
        wam_clear_error(&state);
    }

    /* 25. Distinguish mismatch vs unsupported */
    {
        int rc_mis = run_atom_codes(&state, val_atom("hello"), val_int(99));
        int mis_ok = (rc_mis == WAM_HALT && state.error == 0);

        WamValue c_ref = make_out_ref(&state, "C");
        int rc_uns = run_atom_codes(&state, val_unbound("A"), c_ref);
        int uns_ok = (rc_uns == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        wam_clear_error(&state);

        int ok = mis_ok && uns_ok;
        emit_token("distinguish_mismatch_vs_unsupported", ok ? "ok" : "fail",
                   ok ? "distinguish_ok" : "distinguish_bad");
    }

    /* 26. Matching a fully bound list must not retain temporary heap cells. */
    {
        WamState st;
        wam_state_init(&st);
        int item = 97;
        WamValue list = build_int_list(&st, &item, 1);
        int before = st.H;
        int ok = 1;
        for (int i = 0; i < 1000; i++) {
            st.A[0] = val_atom("a");
            st.A[1] = list;
            if (!wam_execute_builtin(&st, "atom_codes/2", 2) || st.error != 0) {
                ok = 0;
                break;
            }
        }
        ok = ok && st.H == before;
        emit_token("prebound_heap_stable", ok ? "ok" : "fail",
                   ok ? "heap_stable_ok" : "heap_stable_bad");
        wam_free_state(&st);
    }

    /* 27. Invalid C-provided bytes cannot be mistaken for the empty atom. */
    {
        char invalid[2] = { (char)0x80, 0 };
        state.A[0] = val_atom(invalid);
        state.A[1] = val_atom("[]");
        int ok = !wam_execute_builtin(&state, "atom_codes/2", 2) &&
                 state.error == WAM_ERR_UNSUPPORTED;
        emit_token("malformed_utf8", ok ? "ok" : "fail",
                   ok ? "malformed_ok" : "malformed_bad");
        wam_clear_error(&state);
    }

    /* 28. Failed growth must not advertise capacity the heap never acquired. */
    {
        WamState st;
        wam_state_init(&st);
        int cap = st.H_cap;
        st.H = cap - 1;
        st.A[0] = val_atom("ab");
        st.A[1] = val_unbound("C");
        fail_next_realloc = 1;
        int failed = !wam_execute_builtin(&st, "atom_codes/2", 2);
        int stable = st.H_cap == cap && st.H == cap - 1;
        st.H = 0;
        st.A[0] = val_atom("ab");
        st.A[1] = val_unbound("C");
        int retry = wam_execute_builtin(&st, "atom_codes/2", 2);
        int ok = failed && stable && retry && st.error == 0;
        emit_token("heap_alloc_failure", ok ? "ok" : "fail",
                   ok ? "alloc_failure_ok" : "alloc_failure_bad");
        wam_free_state(&st);
    }

    wam_free_state(&state);
    return 0;
}
').

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
    format('~n=== WAM-C atom_codes/2 Tests ===~n~n'),
    setup_atom_codes_preds,
    test_generation_atom_codes_builtin,
    test_wam_emits_atom_codes_builtin,
    test_wrong_answer_rejected,
    test_compiled_c_atom_codes_matches_swi,
    cleanup_atom_codes_preds,
    format('~n=== WAM-C atom_codes/2 Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
