:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C char_code/2: generated/compiled C vs SWI for forward
% char-to-code and reverse code-to-char modes (ASCII, Greek, CJK, emoji).
% Compatible and incompatible prebound outputs, rollback after mismatch,
% caller continuation, and variable identity are verified.
% Invalid UTF-8, empty/multi-character atoms, negative/surrogate/out-of-range
% codes, non-integer code input, and both-variables mode explicitly
% diagnose WAM_ERR_UNSUPPORTED, not silent logical failure.
%
%   swipl -q -g run_tests -t halt tests/test_wam_c_char_code.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).
:- use_module(library(lists), [member/2]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.
:- dynamic user:wam_char_code_q/2.
:- dynamic user:wam_char_code_continuation/3.
:- dynamic user:wam_char_code_bind_control/1.
:- dynamic user:wam_char_code_backtrack/1.
:- dynamic user:wam_char_code_mismatch_positive/1.
:- dynamic user:wam_char_code_mismatch_rollback/1.

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

cleanup_char_code_preds :-
    retractall(user:wam_char_code_q(_, _)),
    retractall(user:wam_char_code_continuation(_, _, _)),
    retractall(user:wam_char_code_bind_control(_)),
    retractall(user:wam_char_code_backtrack(_)),
    retractall(user:wam_char_code_mismatch_positive(_)),
    retractall(user:wam_char_code_mismatch_rollback(_)).

setup_char_code_preds :-
    cleanup_char_code_preds,
    assertz((user:wam_char_code_q(A, C) :- char_code(A, C))),
    assertz((user:wam_char_code_continuation(A, S1, S2) :-
                 S1 = start, char_code(A, _), S2 = done)),
    assertz((user:wam_char_code_bind_control(Out) :-
                 char_code(f, C), Out = C)),
    assertz((user:wam_char_code_backtrack(Out) :-
                 ( char_code(f, C), C == 99, fail
                 ; ( var(C) -> Out = restored ; Out = bad )
                 ), Out == restored)),
    assertz((user:wam_char_code_mismatch_positive(Out) :-
                 char_code(a, C), Out = C)),
    assertz((user:wam_char_code_mismatch_rollback(Out) :-
                 ( char_code(a, C), C = 99
                 ; ( var(C) -> Out = restored ; Out = bad )
                 ), Out == restored)).

ground_swi(fwd_ascii, Expected) :- char_code(a, Expected).
ground_swi(fwd_greek, Expected) :- char_code('λ', Expected).
ground_swi(fwd_cjk, Expected) :- char_code('日', Expected).
ground_swi(fwd_emoji, Expected) :- char_code('🚀', Expected).

ground_swi(rev_ascii, Expected) :- char_code(Expected, 97).
ground_swi(rev_greek, Expected) :- char_code(Expected, 955).
ground_swi(rev_cjk, Expected) :- char_code(Expected, 26085).
ground_swi(rev_emoji, Expected) :- char_code(Expected, 128640).

token_swi(prebound_ascii_match, ok, prebound_ascii_ok) :-
    char_code(a, 97).
token_swi(prebound_greek_match, ok, prebound_greek_ok) :-
    char_code('λ', 955).
token_swi(prebound_cjk_match, ok, prebound_cjk_ok) :-
    char_code('日', 26085).
token_swi(prebound_emoji_match, ok, prebound_emoji_ok) :-
    char_code('🚀', 128640).
token_swi(aliased_var_match, ok, aliased_match_ok) :-
    char_code(X, 97), X == a.
token_swi(prebound_mismatch_code, fail, mismatch_code_ok) :-
    \+ char_code(a, 98).
token_swi(prebound_mismatch_char, fail, mismatch_char_ok) :-
    \+ char_code(b, 97).
token_swi(prebound_mismatch_nonint, fail, mismatch_nonint_ok) :-
    \+ catch(char_code(a, foo), _, fail).
token_swi(aliased_var_mismatch, fail, aliased_mismatch_ok) :-
    X = b, \+ char_code(X, 97).
token_swi(preserve_input, ok, preserve_ok).
token_swi(reference_chain, ok, ref_chain_ok).
token_swi(repeated_calls, ok, repeated_ok).
token_swi(bind_control, ok, bind_control_ok) :-
    user:wam_char_code_bind_control(102).
token_swi(backtrack_rollback, ok, backtrack_ok) :-
    user:wam_char_code_backtrack(restored).
token_swi(mismatch_positive, ok, mismatch_positive_ok) :-
    user:wam_char_code_mismatch_positive(97).
token_swi(mismatch_rollback, ok, mismatch_rollback_ok) :-
    user:wam_char_code_mismatch_rollback(restored).
token_swi(c_unifier_rollback, ok, c_rollback_ok).
token_swi(caller_continuation, ok, continuation_ok) :-
    S1 = start, char_code(a, _), S2 = done,
    S1 == start, S2 == done.
token_swi(repeated_mismatch_rollback, ok, repeated_rollback_ok).
token_swi(both_unbound, ok, both_unbound_ok).
token_swi(empty_atom, ok, empty_atom_ok).
token_swi(empty_atom_prebound, ok, empty_prebound_ok).
token_swi(multi_char_atom, ok, multi_char_ok).
token_swi(multi_char_atom_prebound, ok, multi_char_prebound_ok).
token_swi(non_atom_first, ok, non_atom_first_ok).
token_swi(compound_first, ok, compound_first_ok).
token_swi(list_first, ok, list_first_ok).
token_swi(reverse_non_integer, ok, reverse_non_int_ok).
token_swi(reverse_compound, ok, reverse_compound_ok).
token_swi(reverse_list, ok, reverse_list_ok).
token_swi(negative_code, ok, negative_code_ok).
token_swi(zero_code, ok, zero_code_ok).
token_swi(surrogate_code_low, ok, surrogate_low_ok).
token_swi(surrogate_code_high, ok, surrogate_high_ok).
token_swi(out_of_range_code, ok, out_of_range_ok).
token_swi(malformed_utf8, ok, malformed_ok).
token_swi(distinguish_mismatch_vs_unsupported, ok, distinguish_ok).
token_swi(prebound_heap_stable, ok, heap_stable_ok).
token_swi(intern_alloc_failure, ok, alloc_failure_ok).

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

test_generation_char_code_builtin :-
    Test = 'char_code/2: generated runtime contains char_code handler',
    (   compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(HelpersS, _, _, _, 'strcmp(op, "char_code/2")'),
        sub_string(HelpersS, _, _, _, 'wam_execute_char_code'),
        sub_string(HelpersS, _, _, _, 'wam_decode_single_utf8_char'),
        sub_string(HelpersS, _, _, _, 'wam_encode_single_utf8_char')
    ->  pass(Test)
    ;   fail_test(Test, 'char_code/2 handler missing from generated runtime')
    ).

test_wam_emits_char_code_builtin :-
    Test = 'char_code/2: WAM text emits builtin_call char_code/2',
    setup_char_code_preds,
    (   compile_predicate_to_wam(user:wam_char_code_q/2, [], Wam),
        sub_string(Wam, _, _, _, 'builtin_call char_code/2')
    ->  pass(Test)
    ;   fail_test(Test, 'wam_char_code_q/2 WAM missing builtin_call char_code/2')
    ).

test_wrong_answer_rejected :-
    Test = 'char_code/2: SWI comparator rejects a wrong answer',
    char_code(a, SWI),
    Wrong = 98,
    (   same_success(SWI, SWI),
        \+ same_success(SWI, Wrong),
        \+ same_success(SWI, _)
    ->  pass(Test)
    ;   fail_test(Test, 'actual comparator failed positive or negative controls')
    ).

compile_one(Pred, Code) :-
    compile_predicate_to_wam(Pred, [], Wam),
    compile_wam_predicate_to_c(Pred, Wam, [], Code).

test_compiled_c_char_code_matches_swi :-
    Test = 'char_code/2: compiled C solutions match SWI',
    (   gcc_available
    ->  (   run_compiled_c_char_code
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C char_code executable failed or mismatched')
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

run_compiled_c_char_code :-
    setup_char_code_preds,
    compile_one(user:wam_char_code_q/2, QCode),
    compile_one(user:wam_char_code_continuation/3, ContCode),
    compile_one(user:wam_char_code_bind_control/1, BindCode),
    compile_one(user:wam_char_code_backtrack/1, BackCode),
    compile_one(user:wam_char_code_mismatch_positive/1, MisPosCode),
    compile_one(user:wam_char_code_mismatch_rollback/1, MismatchCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_char_code', Stamp, TmpBase),
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
    char_code_driver_c_source(DriverCode),
    write_text_file(DriverPath, DriverCode),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc),
                   ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
                    RuntimePath, PredPath, DriverPath, '-Wl,--wrap=malloc', '-lm', '-o', ExePath],
                   [process(Pid)]),
    process_wait(Pid, GccStatus),
    format('gcc orig_exit=~w~n', [GccStatus]),
    GccStatus == exit(0),
    run_process_output(['10', ExePath], OutStr, RunStatus),
    format('runner orig_exit=~w~n', [RunStatus]),
    RunStatus == exit(0),
    split_string(OutStr, "\n", "", Lines),
    parse_c_cases(Lines, CCases),
    Ground = [fwd_ascii, fwd_greek, fwd_cjk, fwd_emoji,
              rev_ascii, rev_greek, rev_cjk, rev_emoji],
    Tokens = [prebound_ascii_match, prebound_greek_match, prebound_cjk_match, prebound_emoji_match,
              aliased_var_match,
              prebound_mismatch_code, prebound_mismatch_char, prebound_mismatch_nonint,
              aliased_var_mismatch,
              preserve_input, reference_chain, repeated_calls, bind_control,
              backtrack_rollback, mismatch_positive, mismatch_rollback,
              c_unifier_rollback, caller_continuation, repeated_mismatch_rollback,
              both_unbound, empty_atom, empty_atom_prebound, multi_char_atom,
              multi_char_atom_prebound, non_atom_first, compound_first, list_first,
              reverse_non_integer, reverse_compound, reverse_list,
              negative_code, zero_code, surrogate_code_low, surrogate_code_high, out_of_range_code,
              malformed_utf8, distinguish_mismatch_vs_unsupported,
              prebound_heap_stable, intern_alloc_failure],
    findall(Id, (member(Id, Ground), \+ compare_ground(Id, CCases)), GroundBads),
    findall(Id, (member(Id, Tokens), \+ compare_token(Id, CCases)), TokenBads),
    GroundBads == [],
    TokenBads == [].

char_code_driver_c_source(
'#include "wam_runtime.h"
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_wam_char_code_q_2(WamState *state);
void setup_wam_char_code_continuation_3(WamState *state);
void setup_wam_char_code_bind_control_1(WamState *state);
void setup_wam_char_code_backtrack_1(WamState *state);
void setup_wam_char_code_mismatch_positive_1(WamState *state);
void setup_wam_char_code_mismatch_rollback_1(WamState *state);

static int fail_next_malloc;
void *__real_malloc(size_t size);
void *__wrap_malloc(size_t size) {
    if (fail_next_malloc) {
        fail_next_malloc = 0;
        return NULL;
    }
    return __real_malloc(size);
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

static int run_char_code(WamState *s, WamValue ch, WamValue code) {
    WamValue args[2] = { ch, code };
    return wam_run_predicate(s, "wam_char_code_q/2", args, 2);
}

static int same_atom(WamValue *cell, const char *name) {
    return cell->tag == VAL_ATOM && cell->data.atom &&
           strcmp(cell->data.atom, name) == 0;
}

static void run_ground_fwd(WamState *s, const char *id, WamValue atom) {
    WamValue c_ref = make_out_ref(s, "C");
    int rc = run_char_code(s, atom, c_ref);
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

static void run_ground_rev(WamState *s, const char *id, int code) {
    WamValue a_ref = make_out_ref(s, "A");
    int rc = run_char_code(s, a_ref, val_int(code));
    if (rc == 0 && s->error == 0) {
        emit_case(id, "ok", s, &a_ref);
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
    setup_wam_char_code_q_2(&state);
    setup_wam_char_code_continuation_3(&state);
    setup_wam_char_code_bind_control_1(&state);
    setup_wam_char_code_backtrack_1(&state);
    setup_wam_char_code_mismatch_positive_1(&state);
    setup_wam_char_code_mismatch_rollback_1(&state);

    /* 1. Ground forward cases (atom -> code) */
    run_ground_fwd(&state, "fwd_ascii", val_atom("a"));
    run_ground_fwd(&state, "fwd_greek", val_atom("λ"));
    run_ground_fwd(&state, "fwd_cjk", val_atom("日"));
    run_ground_fwd(&state, "fwd_emoji", val_atom("🚀"));

    /* 2. Ground reverse cases (code -> atom) */
    run_ground_rev(&state, "rev_ascii", 97);
    run_ground_rev(&state, "rev_greek", 955);
    run_ground_rev(&state, "rev_cjk", 26085);
    run_ground_rev(&state, "rev_emoji", 128640);

    /* 3. Compatible prebound match cases */
    {
        int rc = run_char_code(&state, val_atom("a"), val_int(97));
        int ok = rc == 0 && state.error == 0;
        emit_token("prebound_ascii_match", ok ? "ok" : "fail",
                   ok ? "prebound_ascii_ok" : "prebound_ascii_bad");
    }
    {
        int rc = run_char_code(&state, val_atom("λ"), val_int(955));
        int ok = rc == 0 && state.error == 0;
        emit_token("prebound_greek_match", ok ? "ok" : "fail",
                   ok ? "prebound_greek_ok" : "prebound_greek_bad");
    }
    {
        int rc = run_char_code(&state, val_atom("日"), val_int(26085));
        int ok = rc == 0 && state.error == 0;
        emit_token("prebound_cjk_match", ok ? "ok" : "fail",
                   ok ? "prebound_cjk_ok" : "prebound_cjk_bad");
    }
    {
        int rc = run_char_code(&state, val_atom("🚀"), val_int(128640));
        int ok = rc == 0 && state.error == 0;
        emit_token("prebound_emoji_match", ok ? "ok" : "fail",
                   ok ? "prebound_emoji_ok" : "prebound_emoji_bad");
    }

    /* 4. Compatible prebound with shared variable */
    {
        WamValue x = wam_make_ref(&state);
        int rc = run_char_code(&state, x, val_int(97));
        WamValue *x_cell = wam_deref_ptr(&state, &x);
        int ok = rc == 0 && state.error == 0 && same_atom(x_cell, "a");
        emit_token("aliased_var_match", ok ? "ok" : "fail",
                   ok ? "aliased_match_ok" : "aliased_match_bad");
    }

    /* 5. Prebound mismatch on code */
    {
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_char_code(&state, val_atom("a"), val_int(98));
        int rolled = (state.H == h_before && state.TR == tr_before && state.error == 0);
        emit_token("prebound_mismatch_code", (rc == WAM_HALT && rolled) ? "fail" : "ok",
                   (rc == WAM_HALT && rolled) ? "mismatch_code_ok" : "mismatch_code_bad");
    }

    /* 6. Prebound mismatch on char */
    {
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_char_code(&state, val_atom("b"), val_int(97));
        int rolled = (state.H == h_before && state.TR == tr_before && state.error == 0);
        emit_token("prebound_mismatch_char", (rc == WAM_HALT && rolled) ? "fail" : "ok",
                   (rc == WAM_HALT && rolled) ? "mismatch_char_ok" : "mismatch_char_bad");
    }

    /* 7. Incompatible prebound non-integer code */
    {
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_char_code(&state, val_atom("a"), val_atom("foo"));
        int rolled = (state.H == h_before && state.TR == tr_before && state.error == 0);
        emit_token("prebound_mismatch_nonint", (rc == WAM_HALT && rolled) ? "fail" : "ok",
                   (rc == WAM_HALT && rolled) ? "mismatch_nonint_ok" : "mismatch_nonint_bad");
    }

    /* 8. Incompatible prebound with bound variable conflict */
    {
        WamValue x = wam_make_ref(&state);
        WamValue b = val_atom("b");
        wam_unify(&state, &x, &b);
        int h_before = state.H;
        int tr_before = state.TR;
        int rc = run_char_code(&state, x, val_int(97));
        WamValue *x_cell = wam_deref_ptr(&state, &x);
        int rolled = (rc == WAM_HALT && state.error == 0 &&
                      state.H == h_before && state.TR == tr_before &&
                      same_atom(x_cell, "b"));
        emit_token("aliased_var_mismatch", rolled ? "fail" : "ok",
                   rolled ? "aliased_mismatch_ok" : "aliased_mismatch_bad");
    }

    /* 9. Preserve input atom cell */
    {
        WamValue atom = val_atom("a");
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_char_code(&state, atom, c_ref);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &atom), "a");
        emit_token("preserve_input", ok ? "ok" : "fail",
                   ok ? "preserve_ok" : "preserve_bad");
    }

    /* 10. Reference chain on input atom */
    {
        WamValue alias = wam_make_ref(&state);
        WamValue bound = val_atom("a");
        int alias_ok = wam_unify(&state, &alias, &bound);
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = alias_ok ? run_char_code(&state, alias, c_ref) : WAM_HALT;
        int ok = alias_ok && rc == 0 && state.error == 0 &&
                 wam_deref_ptr(&state, &c_ref)->tag == VAL_INT &&
                 wam_deref_ptr(&state, &c_ref)->data.integer == 97;
        emit_token("reference_chain", ok ? "ok" : "fail",
                   ok ? "ref_chain_ok" : "ref_chain_bad");
    }

    /* 11. Repeated calls on one state */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_char_code_q_2(&st);
        int ok = 1;
        for (int i = 0; i < 3; i++) {
            WamValue c_ref = make_out_ref(&st, "C");
            int rc = run_char_code(&st, val_atom("a"), c_ref);
            if (rc != 0 || st.error != 0 ||
                wam_deref_ptr(&st, &c_ref)->data.integer != 97) {
                ok = 0;
                break;
            }
        }
        emit_token("repeated_calls", ok ? "ok" : "fail",
                   ok ? "repeated_ok" : "repeated_bad");
        wam_free_state(&st);
    }

    /* 12. Bind control via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_char_code_bind_control/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 wam_deref_ptr(&state, &out_ref)->tag == VAL_INT &&
                 wam_deref_ptr(&state, &out_ref)->data.integer == 102;
        emit_token("bind_control", ok ? "ok" : "fail",
                   ok ? "bind_control_ok" : "bind_control_bad");
    }

    /* 13. Backtracking rollback via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_char_code_backtrack/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("backtrack_rollback", ok ? "ok" : "fail",
                   ok ? "backtrack_ok" : "backtrack_bad");
    }

    /* 14. Mismatch positive control via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_char_code_mismatch_positive/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 wam_deref_ptr(&state, &out_ref)->tag == VAL_INT &&
                 wam_deref_ptr(&state, &out_ref)->data.integer == 97;
        emit_token("mismatch_positive", ok ? "ok" : "fail",
                   ok ? "mismatch_positive_ok" : "mismatch_positive_bad");
    }

    /* 15. Mismatch rollback via compiled predicate */
    {
        WamValue out_ref = make_out_ref(&state, "Out");
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&state, "wam_char_code_mismatch_rollback/1", args, 1);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &out_ref), "restored");
        emit_token("mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "mismatch_rollback_ok" : "mismatch_rollback_bad");
    }

    /* 16. Direct C unifier rollback */
    {
        WamValue c = wam_make_ref(&state);
        int tr_before = state.TR;
        int rc_pos = run_char_code(&state, val_atom("a"), c);
        int bind_ok = (rc_pos == 0 && state.error == 0);
        unwind_trail(&state, tr_before);

        int h_before = state.H;
        int tr_mis = state.TR;
        int rc_mis = run_char_code(&state, val_atom("a"), val_int(99));
        int rollback_ok = (rc_mis == WAM_HALT && state.error == 0 &&
                           state.H == h_before && state.TR == tr_mis);
        int ok = bind_ok && rollback_ok;
        emit_token("c_unifier_rollback", ok ? "ok" : "fail",
                   ok ? "c_rollback_ok" : "c_rollback_bad");
    }

    /* 17. Caller continuation */
    {
        WamValue s1_ref = make_out_ref(&state, "S1");
        WamValue s2_ref = make_out_ref(&state, "S2");
        WamValue args[3] = { val_atom("a"), s1_ref, s2_ref };
        int rc = wam_run_predicate(&state, "wam_char_code_continuation/3", args, 3);
        int ok = rc == 0 && state.error == 0 &&
                 same_atom(wam_deref_ptr(&state, &s1_ref), "start") &&
                 same_atom(wam_deref_ptr(&state, &s2_ref), "done");
        emit_token("caller_continuation", ok ? "ok" : "fail",
                   ok ? "continuation_ok" : "continuation_bad");
    }

    /* 18. Repeated mismatch rollback */
    {
        WamState st;
        wam_state_init(&st);
        setup_wam_char_code_q_2(&st);
        WamValue c = wam_make_ref(&st);
        int baseline_H = st.H;
        int baseline_TR = st.TR;
        int all_failed = 1;
        for (int iter = 0; iter < 5; iter++) {
            int rc = run_char_code(&st, val_atom("a"), val_int(99));
            if (rc != WAM_HALT || st.error != 0 ||
                st.H != baseline_H || st.TR != baseline_TR) {
                all_failed = 0;
                break;
            }
        }
        int rc_good = run_char_code(&st, val_atom("a"), c);
        int good_ok = (rc_good == 0 && st.error == 0 &&
                       wam_deref_ptr(&st, &c)->data.integer == 97);
        int ok = all_failed && good_ok;
        emit_token("repeated_mismatch_rollback", ok ? "ok" : "fail",
                   ok ? "repeated_rollback_ok" : "repeated_rollback_bad");
        wam_free_state(&st);
    }

    /* 19. Both unbound -> WAM_ERR_UNSUPPORTED */
    {
        int rc = run_char_code(&state, val_unbound("A"), val_unbound("C"));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED &&
                  state.error_op != NULL && strcmp(state.error_op, "char_code/2") == 0);
        emit_token("both_unbound", ok ? "ok" : "fail",
                   ok ? "both_unbound_ok" : "both_unbound_bad");
        wam_clear_error(&state);
    }

    /* 20. Empty atom -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_char_code(&state, val_atom(""), c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("empty_atom", ok ? "ok" : "fail",
                   ok ? "empty_atom_ok" : "empty_atom_bad");
        wam_clear_error(&state);
    }

    /* 21. Empty atom prebound -> WAM_ERR_UNSUPPORTED */
    {
        int rc = run_char_code(&state, val_atom(""), val_int(97));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("empty_atom_prebound", ok ? "ok" : "fail",
                   ok ? "empty_prebound_ok" : "empty_prebound_bad");
        wam_clear_error(&state);
    }

    /* 22. Multi-character atom -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_char_code(&state, val_atom("ab"), c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("multi_char_atom", ok ? "ok" : "fail",
                   ok ? "multi_char_ok" : "multi_char_bad");
        wam_clear_error(&state);
    }

    /* 23. Multi-character atom prebound -> WAM_ERR_UNSUPPORTED */
    {
        int rc = run_char_code(&state, val_atom("ab"), val_int(97));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("multi_char_atom_prebound", ok ? "ok" : "fail",
                   ok ? "multi_char_prebound_ok" : "multi_char_prebound_bad");
        wam_clear_error(&state);
    }

    /* 24. Non-atom first argument -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_char_code(&state, val_int(42), c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("non_atom_first", ok ? "ok" : "fail",
                   ok ? "non_atom_first_ok" : "non_atom_first_bad");
        wam_clear_error(&state);
    }

    /* 25. Compound first argument -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        int rc = run_char_code(&state, mkstr1(&state, "f/1", val_atom("a")), c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("compound_first", ok ? "ok" : "fail",
                   ok ? "compound_first_ok" : "compound_first_bad");
        wam_clear_error(&state);
    }

    /* 26. List first argument -> WAM_ERR_UNSUPPORTED */
    {
        WamValue c_ref = make_out_ref(&state, "C");
        WamValue lst = cons(&state, val_int(97), val_atom("[]"));
        int rc = run_char_code(&state, lst, c_ref);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("list_first", ok ? "ok" : "fail",
                   ok ? "list_first_ok" : "list_first_bad");
        wam_clear_error(&state);
    }

    /* 27. Reverse mode non-integer code -> WAM_ERR_UNSUPPORTED */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        int rc = run_char_code(&state, a_ref, val_atom("foo"));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("reverse_non_integer", ok ? "ok" : "fail",
                   ok ? "reverse_non_int_ok" : "reverse_non_int_bad");
        wam_clear_error(&state);
    }

    /* 28. Reverse mode compound code -> WAM_ERR_UNSUPPORTED */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        int rc = run_char_code(&state, a_ref, mkstr1(&state, "f/1", val_int(97)));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("reverse_compound", ok ? "ok" : "fail",
                   ok ? "reverse_compound_ok" : "reverse_compound_bad");
        wam_clear_error(&state);
    }

    /* 29. Reverse mode list code -> WAM_ERR_UNSUPPORTED */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        WamValue lst = cons(&state, val_int(97), val_atom("[]"));
        int rc = run_char_code(&state, a_ref, lst);
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("reverse_list", ok ? "ok" : "fail",
                   ok ? "reverse_list_ok" : "reverse_list_bad");
        wam_clear_error(&state);
    }

    /* 30. Negative code -> WAM_ERR_UNSUPPORTED */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        int rc = run_char_code(&state, a_ref, val_int(-1));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("negative_code", ok ? "ok" : "fail",
                   ok ? "negative_code_ok" : "negative_code_bad");
        wam_clear_error(&state);
    }

    /* U+0000 cannot be represented by the runtime's NUL-terminated atoms. */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        int rc = run_char_code(&state, a_ref, val_int(0));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("zero_code", ok ? "ok" : "fail",
                   ok ? "zero_code_ok" : "zero_code_bad");
        wam_clear_error(&state);
    }

    /* 31. Surrogate low (0xD800) -> WAM_ERR_UNSUPPORTED */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        int rc = run_char_code(&state, a_ref, val_int(0xD800));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("surrogate_code_low", ok ? "ok" : "fail",
                   ok ? "surrogate_low_ok" : "surrogate_low_bad");
        wam_clear_error(&state);
    }

    /* 32. Surrogate high (0xDFFF) -> WAM_ERR_UNSUPPORTED */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        int rc = run_char_code(&state, a_ref, val_int(0xDFFF));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("surrogate_code_high", ok ? "ok" : "fail",
                   ok ? "surrogate_high_ok" : "surrogate_high_bad");
        wam_clear_error(&state);
    }

    /* 33. Out of range (> 0x10FFFF) -> WAM_ERR_UNSUPPORTED */
    {
        WamValue a_ref = make_out_ref(&state, "A");
        int rc = run_char_code(&state, a_ref, val_int(0x110000));
        int ok = (rc == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        emit_token("out_of_range_code", ok ? "ok" : "fail",
                   ok ? "out_of_range_ok" : "out_of_range_bad");
        wam_clear_error(&state);
    }

    /* 34. Malformed UTF-8 in atom -> WAM_ERR_UNSUPPORTED */
    {
        char invalid[2] = { (char)0x80, 0 };
        state.A[0] = val_atom(invalid);
        state.A[1] = val_unbound("C");
        int ok = !wam_execute_builtin(&state, "char_code/2", 2) &&
                 state.error == WAM_ERR_UNSUPPORTED;
        emit_token("malformed_utf8", ok ? "ok" : "fail",
                   ok ? "malformed_ok" : "malformed_bad");
        wam_clear_error(&state);
    }

    /* 35. Distinguish mismatch vs unsupported */
    {
        int rc_mis = run_char_code(&state, val_atom("a"), val_int(99));
        int mis_ok = (rc_mis == WAM_HALT && state.error == 0);

        int rc_uns = run_char_code(&state, val_unbound("A"), val_unbound("C"));
        int uns_ok = (rc_uns == WAM_ERR_UNSUPPORTED && state.error == WAM_ERR_UNSUPPORTED);
        wam_clear_error(&state);

        int ok = mis_ok && uns_ok;
        emit_token("distinguish_mismatch_vs_unsupported", ok ? "ok" : "fail",
                   ok ? "distinguish_ok" : "distinguish_bad");
    }

    /* 36. Matching prebound does not consume heap space */
    {
        WamState st;
        wam_state_init(&st);
        int before = st.H;
        int ok = 1;
        for (int i = 0; i < 1000; i++) {
            st.A[0] = val_atom("a");
            st.A[1] = val_int(97);
            if (!wam_execute_builtin(&st, "char_code/2", 2) || st.error != 0) {
                ok = 0;
                break;
            }
        }
        ok = ok && (st.H == before);
        emit_token("prebound_heap_stable", ok ? "ok" : "fail",
                   ok ? "heap_stable_ok" : "heap_stable_bad");
        wam_free_state(&st);
    }

    /* 37. Atom interning allocation failure returns false via sentinel without stack pointer leak */
    {
        WamState st;
        wam_state_init(&st);
        st.A[0] = val_unbound("A");
        st.A[1] = val_int(120); /* ''x'' */
        fail_next_malloc = 1;
        int failed = !wam_execute_builtin(&st, "char_code/2", 2);
        st.A[0] = val_unbound("A");
        st.A[1] = val_int(120);
        int retry = wam_execute_builtin(&st, "char_code/2", 2);
        int ok = failed && retry && st.error == 0 &&
                 same_atom(wam_deref_ptr(&st, &st.A[0]), "x");
        emit_token("intern_alloc_failure", ok ? "ok" : "fail",
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
    format('~n=== WAM-C char_code/2 Tests ===~n~n'),
    setup_char_code_preds,
    test_generation_char_code_builtin,
    test_wam_emits_char_code_builtin,
    test_wrong_answer_rejected,
    test_compiled_c_char_code_matches_swi,
    cleanup_char_code_preds,
    format('~n=== WAM-C char_code/2 Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
