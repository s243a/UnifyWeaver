:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C atom_length/2: generated/compiled C vs SWI for atom text
% (Unicode code points) and WAM-Rust value_atomic_text forms. Compounds,
% non-empty lists, and unbound first arguments fail logically.
%
%   swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_atom_length.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).
:- use_module(library(lists), [member/2]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.
:- dynamic user:wam_atom_length_q/2.
:- dynamic user:wam_atom_length_continuation/3.
:- dynamic user:wam_atom_length_bind_control/1.
:- dynamic user:wam_atom_length_backtrack/1.
:- dynamic user:wam_atom_length_mismatch_positive/1.
:- dynamic user:wam_atom_length_mismatch_rollback/1.

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

rustc_available :-
    catch(process_create(path(rustc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          _, fail),
    process_wait(Pid, exit(0)).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).

cleanup_atom_length_preds :-
    retractall(user:wam_atom_length_q(_, _)),
    retractall(user:wam_atom_length_continuation(_, _, _)),
    retractall(user:wam_atom_length_bind_control(_)),
    retractall(user:wam_atom_length_backtrack(_)),
    retractall(user:wam_atom_length_mismatch_positive(_)),
    retractall(user:wam_atom_length_mismatch_rollback(_)).

setup_atom_length_preds :-
    cleanup_atom_length_preds,
    assertz((user:wam_atom_length_q(A, N) :- atom_length(A, N))),
    assertz((user:wam_atom_length_continuation(A, S1, S2) :-
                 S1 = start, atom_length(A, _), S2 = done)),
    assertz((user:wam_atom_length_bind_control(Out) :-
                 atom_length(foo, N), Out = N)),
    assertz((user:wam_atom_length_backtrack(Out) :-
                 ( atom_length(foo, N), N =:= 99, fail
                 ; ( var(N) -> Out = restored ; Out = bad )
                 ), Out == restored)),
    assertz((user:wam_atom_length_mismatch_positive(Out) :-
                 atom_length(hello, N), Out = N)),
    assertz((user:wam_atom_length_mismatch_rollback(Out) :-
                 ( atom_length(hello, N), N = 3
                 ; ( var(N) -> Out = restored ; Out = bad )
                 ), Out == restored)).

ground_input(ascii, hello).
ground_input(empty, '').
ground_input(multibyte, 'café').

ground_swi(Id, Expected) :-
    ground_input(Id, Atom),
    atom_length(Atom, Expected).

rust_parity_expected(integer_input, 2).
rust_parity_expected(float_input, 3).
rust_parity_expected(true_atom, 4).
rust_parity_expected(false_atom, 5).
rust_parity_expected(empty_list_atom, 2).
% Rust format!("{}", f) Display lengths (fixed-point, not %g).
rust_parity_expected(float_1e6, 7).
rust_parity_expected(float_1e_minus_6, 8).
rust_parity_expected(float_1_23456789, 10).
rust_parity_expected(float_1e100, 101).
rust_parity_expected(float_min_subnormal, 326).
rust_parity_expected(float_neg, 4).
rust_parity_expected(float_zero, 1).
rust_parity_expected(float_neg_zero, 2).
rust_parity_expected(float_inf, 3).
rust_parity_expected(float_neg_inf, 4).
rust_parity_expected(float_nan, 3).
rust_parity_expected(float_tenth, 3).
rust_parity_expected(float_one, 1).
rust_parity_expected(float_pow2, 1).
rust_parity_expected(float_three_tenths, 3).

token_swi(prebound_match, ok, prebound_ok) :-
    atom_length(hello, 5).
token_swi(prebound_mismatch, fail, prebound_mismatch_ok) :-
    \+ atom_length(hello, 4).
token_swi(preserve_input, ok, preserve_ok).
token_swi(reference_chain, ok, ref_chain_ok).
token_swi(repeated_calls, ok, repeated_ok).
token_swi(unbound_first, fail, unbound_ok).
token_swi(compound_input, fail, compound_ok).
token_swi(nonempty_list, fail, list_ok).
token_swi(bind_control, ok, bind_control_ok) :-
    user:wam_atom_length_bind_control(3).
token_swi(backtrack_rollback, ok, backtrack_ok) :-
    user:wam_atom_length_backtrack(restored).
token_swi(mismatch_positive, ok, mismatch_positive_ok) :-
    user:wam_atom_length_mismatch_positive(5).
token_swi(mismatch_rollback, ok, mismatch_rollback_ok) :-
    user:wam_atom_length_mismatch_rollback(restored).
token_swi(c_unifier_rollback, ok, c_rollback_ok).
token_swi(caller_continuation, ok, continuation_ok) :-
    S1 = start, atom_length(foo, _), S2 = done,
    S1 == start, S2 == done.
token_swi(repeated_mismatch_rollback, ok, repeated_rollback_ok).

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

compare_rust_parity(Id, CCases) :-
    rust_parity_expected(Id, Expected),
    (   member(c_case(Id, CStatus, CTerm), CCases)
    ->  true
    ;   fail_test(Id, 'missing C output'),
        fail
    ),
    (   CStatus == ok,
        same_success(Expected, CTerm)
    ->  pass(Id)
    ;   format(atom(R), 'Rust-parity ~q vs C ~w ~q', [Expected, CStatus, CTerm]),
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

test_generation_atom_length_builtin :-
    Test = 'atom_length/2: generated runtime contains atom_length handler',
    (   compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(HelpersS, _, _, _, 'strcmp(op, "atom_length/2")'),
        sub_string(HelpersS, _, _, _, 'wam_execute_atom_length'),
        sub_string(HelpersS, _, _, _, 'wam_value_atomic_length'),
        sub_string(HelpersS, _, _, _, 'wam_utf8_codepoint_count'),
        sub_string(HelpersS, _, _, _, 'wam_rust_display_float_len')
    ->  pass(Test)
    ;   fail_test(Test, 'atom_length/2 handler missing from generated runtime')
    ).

test_wam_emits_atom_length_builtin :-
    Test = 'atom_length/2: WAM text emits builtin_call atom_length/2',
    setup_atom_length_preds,
    (   compile_predicate_to_wam(user:wam_atom_length_q/2, [], Wam),
        sub_string(Wam, _, _, _, 'builtin_call atom_length/2')
    ->  pass(Test)
    ;   fail_test(Test, 'wam_atom_length_q/2 WAM missing builtin_call atom_length/2')
    ).

test_wrong_answer_rejected :-
    Test = 'atom_length/2: SWI comparator rejects a wrong answer',
    atom_length(hello, SWI),
    Wrong = 4,
    (   same_success(SWI, SWI),
        \+ same_success(SWI, Wrong),
        \+ same_success(SWI, _)
    ->  pass(Test)
    ;   fail_test(Test, 'actual comparator failed positive or negative controls')
    ).

compile_one(Pred, Code) :-
    compile_predicate_to_wam(Pred, [], Wam),
    compile_wam_predicate_to_c(Pred, Wam, [], Code).

test_compiled_c_atom_length_matches_swi :-
    Test = 'atom_length/2: compiled C solutions match SWI and Rust parity',
    (   gcc_available
    ->  (   run_compiled_c_atom_length
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C atom_length executable failed or mismatched')
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

compare_oracle_cases(RustCases, CCases) :-
    findall(Id, (member(c_case(Id, RustSt, RustT), RustCases),
                 \+ (member(c_case(Id, CSt, CT), CCases),
                     CSt == RustSt, CT == RustT)),
            Bads),
    length(RustCases, NRust),
    length(CCases, NC),
    (   Bads == [], NRust > 0, NRust == NC
    ->  format('[PASS] rust oracle grid (~w cases)~n', [NRust])
    ;   format(atom(R), 'oracle mismatch bads=~w rust=~w c=~w',
               [Bads, NRust, NC]),
        fail_test('float rust oracle grid', R),
        fail
    ).

run_compiled_c_atom_length :-
    setup_atom_length_preds,
    compile_one(user:wam_atom_length_q/2, QCode),
    compile_one(user:wam_atom_length_continuation/3, ContCode),
    compile_one(user:wam_atom_length_bind_control/1, BindCode),
    compile_one(user:wam_atom_length_backtrack/1, BackCode),
    compile_one(user:wam_atom_length_mismatch_positive/1, MisPosCode),
    compile_one(user:wam_atom_length_mismatch_rollback/1, MismatchCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_atom_length', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    atomic_list_concat([QCode, "\n\n", ContCode, "\n\n", BindCode, "\n\n",
                        BackCode, "\n\n", MisPosCode, "\n\n", MismatchCode],
                       AllPredCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [AllPredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    Driver = 'examples/pkg_resolver/c/atom_length_driver.c',
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc),
                   ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
                    RuntimePath, PredPath, Driver, '-lm', '-o', ExePath],
                   [process(Pid)]),
    process_wait(Pid, GccStatus),
    format('gcc orig_exit=~w~n', [GccStatus]),
    GccStatus == exit(0),
    run_process_output(['10', ExePath], OutStr, RunStatus),
    format('runner orig_exit=~w~n', [RunStatus]),
    RunStatus == exit(0),
    split_string(OutStr, "\n", "", Lines),
    parse_c_cases(Lines, CCases),
    Ground = [ascii, empty, multibyte],
    RustParity = [integer_input, float_input, true_atom, false_atom, empty_list_atom,
                  float_1e6, float_1e_minus_6, float_1_23456789, float_1e100,
                  float_min_subnormal, float_neg, float_zero, float_neg_zero,
                  float_inf, float_neg_inf, float_nan,
                  float_tenth, float_one, float_pow2, float_three_tenths],
    Tokens = [prebound_match, prebound_mismatch, preserve_input, reference_chain,
              repeated_calls, unbound_first, compound_input, nonempty_list,
              bind_control, backtrack_rollback, mismatch_positive, mismatch_rollback,
              c_unifier_rollback, caller_continuation, repeated_mismatch_rollback],
    findall(Id, (member(Id, Ground), \+ compare_ground(Id, CCases)), GroundBads),
    findall(Id, (member(Id, RustParity), \+ compare_rust_parity(Id, CCases)), ParityBads),
    findall(Id, (member(Id, Tokens), \+ compare_token(Id, CCases)), TokenBads),
    GroundBads == [],
    ParityBads == [],
    TokenBads == [],
    run_rust_float_oracle(TmpBase, ExePath).

run_rust_float_oracle(TmpBase, ExePath) :-
    Test = 'atom_length/2: C float lengths match rustc Display oracle grid',
    (   rustc_available
    ->  format(atom(RustExe), '~w_oracle', [TmpBase]),
        process_create(path(rustc),
                       ['examples/pkg_resolver/c/atom_length_float_oracle.rs',
                        '-O', '-o', RustExe],
                       [process(Pid)]),
        process_wait(Pid, RustcStatus),
        (   RustcStatus == exit(0)
        ->  run_process_output(['30', RustExe, 'grid'], RustOut, RustRun),
            run_process_output(['30', ExePath, 'oracle'], COut, CRun),
            RustRun == exit(0),
            CRun == exit(0),
            split_string(RustOut, "\n", "", RustLines),
            split_string(COut, "\n", "", CLines),
            parse_c_cases(RustLines, RustCases),
            parse_c_cases(CLines, CCases),
            compare_oracle_cases(RustCases, CCases)
        ;   fail_test(Test, 'rustc failed to build float oracle')
        )
    ;   format('[SKIP] ~w (rustc unavailable)~n', [Test])
    ).

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
    format('~n=== WAM-C atom_length/2 Tests ===~n~n'),
    setup_atom_length_preds,
    test_generation_atom_length_builtin,
    test_wam_emits_atom_length_builtin,
    test_wrong_answer_rejected,
    test_compiled_c_atom_length_matches_swi,
    cleanup_atom_length_preds,
    format('~n=== WAM-C atom_length/2 Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
