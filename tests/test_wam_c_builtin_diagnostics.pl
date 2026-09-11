:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C builtin diagnostics: unsupported ops are a distinct runtime
% error, not logical fail. Does not implement missing builtin semantics.
%
%   swipl -g run_tests -t halt tests/test_wam_c_builtin_diagnostics.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).
:- use_module(library(filesex), [directory_file_path/3]).
:- use_module(library(readutil), [read_file_to_string/3]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.

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

compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath) :-
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc),
                   ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
                    RuntimePath, PredPath, MainPath, '-lm', '-o', ExePath],
                   [process(Pid)]),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, 'gcc failed with status ~w~n', [Status]),
        fail
    ).

run_c_smoke_plain(ExePath) :-
    process_create(path(timeout), ['10', ExePath], [process(Pid)]),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, 'generated executable failed with status ~w~n', [Status]),
        fail
    ).

diag_raw_wam(WamCode) :-
    WamCode = 'diag_true/0:
    builtin_call true/0, 0
    proceed
diag_fail/0:
    builtin_call fail/0, 0
    proceed
diag_gt_false/0:
    put_constant 1, A1
    put_constant 2, A2
    builtin_call >/2, 2
    proceed
diag_gt_true/0:
    put_constant 2, A1
    put_constant 1, A2
    builtin_call >/2, 2
    proceed
diag_unknown_ints/0:
    put_constant 1, A1
    put_constant 2, A2
    builtin_call reverse/2, 2
    proceed
diag_unknown_atoms/0:
    put_constant a, A1
    put_constant b, A2
    builtin_call reverse/2, 2
    proceed
diag_unknown_arity3/0:
    put_constant a, A1
    put_constant b, A2
    put_constant c, A3
    builtin_call append/3, 3
    proceed
diag_choice/0:
    try_me_else L_diag_choice_2
    put_constant 1, A1
    put_constant 2, A2
    builtin_call reverse/2, 2
    proceed
L_diag_choice_2:
    trust_me
    builtin_call true/0, 0
    proceed
diag_meta_find/1:
    put_variable X1, A1
    put_structure reverse/2, A2
    set_constant 1
    set_constant 2
    put_variable X2, A3
    call findall/3, 3
    proceed
'.

cleanup_diag_agg :-
    retractall(user:diag_agg_item(_)),
    retractall(user:diag_agg(_)).

setup_diag_agg :-
    cleanup_diag_agg,
    assertz(user:diag_agg_item(a)),
    assertz((user:diag_agg(L) :-
                 findall(X, (diag_agg_item(X), reverse(1, 2)), L))).

test_generation_error_paths :-
    Test = 'diagnostics: generated runtime classifies unsupported ops',
    (   compile_step_wam_to_c([], StepCode),
        atom_string(StepCode, StepS),
        compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(StepS, _, _, _, 'if (state->error != 0)'),
        sub_string(HelpersS, _, _, _, 'wam_clear_error(state)'),
        sub_string(HelpersS, _, _, _, 'wam_set_unsupported_builtin(state, op, arity)'),
        sub_string(HelpersS, _, _, _, 'Classify arithmetic comparisons by operator first'),
        sub_string(HelpersS, _, _, _, 'arity == 2 &&')
    ->  pass(Test)
    ;   fail_test(Test, 'generated runtime missing unsupported-builtin classification')
    ).

test_compiled_c_diagnostics :-
    Test = 'diagnostics: compiled C unknown/true/fail/compare/backtrack/agg/reuse',
    (   gcc_available
    ->  (   run_compiled_c_diagnostics
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C diagnostics executable failed')
        )
    ;   fail_test(Test, 'gcc unavailable; compiled behavior not verified')
    ).

run_compiled_c_diagnostics :-
    diag_raw_wam(RawWam),
    compile_wam_predicate_to_c(user:diag_true/0, RawWam, [], RawPredCode),
    setup_diag_agg,
    (   compile_predicate_to_wam(user:diag_agg_item/1, [], ItemWam),
        compile_predicate_to_wam(user:diag_agg/1, [], AggWam),
        sub_string(AggWam, _, _, _, 'begin_aggregate collect'),
        sub_string(AggWam, _, _, _, 'builtin_call reverse/2, 2'),
        compile_wam_predicate_to_c(user:diag_agg_item/1, ItemWam, [], ItemCode),
        compile_wam_predicate_to_c(user:diag_agg/1, AggWam, [], AggCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_builtin_diag', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        atomic_list_concat([RawPredCode, ItemCode, AggCode], '\n\n', PredCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        diag_main_c(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_diag_agg
    ;   cleanup_diag_agg,
        fail
    ).

diag_main_c(
'#include "wam_runtime.h"
#include <string.h>

void setup_diag_true_0(WamState* state);
void setup_diag_agg_item_1(WamState* state);
void setup_diag_agg_1(WamState* state);

static int expect_unsupported(WamState *state, const char *pred,
                              const char *op, int fail_code) {
    int rc = wam_run_predicate(state, pred, NULL, 0);
    if (rc != WAM_ERR_UNSUPPORTED) return fail_code;
    if (state->error != WAM_ERR_UNSUPPORTED) return fail_code + 1;
    if (state->error_op == NULL || strcmp(state->error_op, op) != 0)
        return fail_code + 2;
    return 0;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_diag_true_0(&state);
    setup_diag_agg_item_1(&state);
    setup_diag_agg_1(&state);

    int rc = wam_run_predicate(&state, "diag_true/0", NULL, 0);
    if (rc != 0 || state.error != 0) {
        wam_free_state(&state);
        return 10;
    }

    rc = wam_run_predicate(&state, "diag_fail/0", NULL, 0);
    if (rc != WAM_HALT || state.error != 0) {
        wam_free_state(&state);
        return 20;
    }

    rc = wam_run_predicate(&state, "diag_gt_false/0", NULL, 0);
    if (rc != WAM_HALT || state.error != 0) {
        wam_free_state(&state);
        return 30;
    }

    rc = wam_run_predicate(&state, "diag_gt_true/0", NULL, 0);
    if (rc != 0 || state.error != 0) {
        wam_free_state(&state);
        return 40;
    }

    int bad = expect_unsupported(&state, "diag_unknown_ints/0", "reverse/2", 50);
    if (bad) {
        wam_free_state(&state);
        return bad;
    }

    bad = expect_unsupported(&state, "diag_unknown_atoms/0", "reverse/2", 60);
    if (bad) {
        wam_free_state(&state);
        return bad;
    }

    bad = expect_unsupported(&state, "diag_unknown_arity3/0", "append/3", 70);
    if (bad) {
        wam_free_state(&state);
        return bad;
    }

    bad = expect_unsupported(&state, "diag_choice/0", "reverse/2", 80);
    if (bad) {
        wam_free_state(&state);
        return bad;
    }

    rc = wam_run_predicate(&state, "diag_true/0", NULL, 0);
    if (rc != 0 || state.error != 0) {
        wam_free_state(&state);
        return 90;
    }

    int base_e = state.E;
    int base_h = state.H;
    int base_tr = state.TR;
    WamValue agg_args[1] = { val_unbound("L") };
    rc = wam_run_predicate(&state, "diag_agg/1", agg_args, 1);
    if (rc != WAM_ERR_UNSUPPORTED || state.error != WAM_ERR_UNSUPPORTED ||
        state.error_op == NULL || strcmp(state.error_op, "reverse/2") != 0) {
        wam_free_state(&state);
        return 100;
    }
    if (state.aggregate_top != 0 || state.B != 0 || state.E != base_e ||
        state.H != base_h || state.TR != base_tr) return 101;

    rc = wam_run_predicate(&state, "diag_true/0", NULL, 0);
    if (rc != 0 || state.error != 0) {
        wam_free_state(&state);
        return 110;
    }

    WamValue meta_args[1] = { val_unbound("L") };
    rc = wam_run_predicate(&state, "diag_meta_find/1", meta_args, 1);
    if (rc != WAM_ERR_UNSUPPORTED || state.error != WAM_ERR_UNSUPPORTED ||
        state.error_op == NULL || strcmp(state.error_op, "reverse/2") != 0) {
        wam_free_state(&state);
        return 120;
    }

    rc = wam_run_predicate(&state, "diag_fail/0", NULL, 0);
    if (rc != WAM_HALT || state.error != 0) {
        wam_free_state(&state);
        return 130;
    }

    for (int i = 0; i < WAM_AGGREGATE_STACK_SIZE + 2; i++) {
        rc = wam_run_predicate(&state, "diag_meta_find/1", meta_args, 1);
        if (rc != WAM_ERR_UNSUPPORTED || state.aggregate_top != 0 ||
            state.B != 0 || state.E != base_e || state.H != base_h ||
            state.TR != base_tr) return 140;
    }
    rc = wam_run_predicate(&state, "diag_true/0", NULL, 0);
    if (rc != 0 || state.error != 0 || state.error_op != NULL ||
        state.error_arity != 0) return 150;

    wam_free_state(&state);
    return 0;
}
').

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        format('~n=== WAM-C Builtin Diagnostics ===~n~n'),
        test_generation_error_paths,
        test_compiled_c_diagnostics,
        format('~n=== WAM-C Builtin Diagnostics Complete ===~n'),
        (   test_failed -> halt(1) ; true )
    ).

:- initialization(run_tests, main).
