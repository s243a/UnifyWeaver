:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C append/3: generated/compiled C vs SWI for finite proper first lists.
% Unbound, open, improper, cyclic, or non-lists are diagnosed with
% WAM_ERR_UNSUPPORTED, not treated as silent logical failure.
%
%   swipl -q -g run_tests -t halt tests/test_wam_c_append.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).
:- use_module(library(lists), [member/2, append/3, numlist/3]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.
:- dynamic user:wam_append_q/3.
:- dynamic user:wam_append_continuation/4.
:- dynamic user:wam_append_bind_control/1.
:- dynamic user:wam_append_backtrack/1.
:- dynamic user:wam_append_mismatch_positive/1.
:- dynamic user:wam_append_mismatch_rollback/1.

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

cleanup_append_preds :-
    retractall(user:wam_append_q(_, _, _)),
    retractall(user:wam_append_continuation(_, _, _, _)),
    retractall(user:wam_append_bind_control(_)),
    retractall(user:wam_append_backtrack(_)),
    retractall(user:wam_append_mismatch_positive(_)),
    retractall(user:wam_append_mismatch_rollback(_)).

setup_append_preds :-
    cleanup_append_preds,
    assertz((user:wam_append_q(L1, L2, R) :- append(L1, L2, R))),
    assertz((user:wam_append_continuation(L1, L2, S1, S2) :- S1 = start, append(L1, L2, _), S2 = done)),
    assertz((user:wam_append_bind_control(Out) :- append([1, X], [3], [1, 2, 3]), Out = X)),
    assertz((user:wam_append_backtrack(Out) :- ( append([1, X], [3], [1, 2, 3]), X == 2, fail ; ( var(X) -> Out = restored ; Out = bad ) ), Out == restored)),
    assertz((user:wam_append_mismatch_positive(Out) :- append([a, X], [c], [a, b, c]), Out = X)),
    assertz((user:wam_append_mismatch_rollback(Out) :- ( append([a, X], [c], [a, b, d]) ; ( var(X) -> Out = restored ; Out = bad ) ), Out == restored)).

ground_input(empty_empty, [], []).
ground_input(empty_nonempty, [], [a, b]).
ground_input(singleton_empty, [a], []).
ground_input(singleton_singleton, [a], [b]).
ground_input(duplicates, [b, a, b], [a, c]).
ground_input(unordered_ints, [3, 1], [2, 4]).
ground_input(unordered_atoms, [c, a], [b, d]).
ground_input(negatives, [3, -1], [0, -5]).
ground_input(ints_and_atoms, [b, 2], [a, 1]).
ground_input(pairs, [c-1], [a-1, b-1]).
ground_input(compounds, [g(a, b)], [f(b), f(a)]).
ground_input(lists_and_compounds, [f(a)], [[a], a]).
ground_input(nested_lists, [[3, 1]], [[1, 2], [3, 1]]).
ground_input(five_elements, [1, 2], [3, 4, 5]).
ground_input(arbitrary_tail, [1, 2], tail_atom).

ground_swi(Id, Expected) :-
    ground_input(Id, In1, In2),
    append(In1, In2, Expected).

token_swi(empty_tail_sharing, ok, empty_tail_sharing_ok).
token_swi(spine_copy_tail_share, ok, spine_copy_tail_share_ok).
token_swi(prebound_match, ok, prebound_ok) :-
    append([1, 2], [3], [1, 2, 3]).
token_swi(prebound_mismatch, fail, prebound_mismatch_ok) :-
    \+ append([1, 2], [3], [1, 2, 4]).
token_swi(preserve_input, ok, preserve_ok).
token_swi(heap_growth, ok, heap_growth_ok) :-
    numlist(1, 1024, L1), append(L1, [1025], Out),
    length(Out, 1025).
token_swi(shared_var, ok, shared_ok) :-
    append([X, a], [X], [X, a, X]).
token_swi(distinct_vars, ok, distinct_ok) :-
    append([Y, X], [Y], [Y, X, Y]).
token_swi(cell_vars, ok, cell_vars_ok) :-
    append([X], [Y], [X, Y]).
token_swi(compound_shared_var, ok, compound_shared_ok) :-
    append([f(X)], [g(X)], [f(X), g(X)]).
token_swi(bind_control, ok, bind_control_ok) :-
    user:wam_append_bind_control(2).
token_swi(backtrack_rollback, ok, backtrack_ok) :-
    user:wam_append_backtrack(restored).
token_swi(mismatch_positive, ok, mismatch_positive_ok) :-
    user:wam_append_mismatch_positive(b).
token_swi(mismatch_rollback, ok, mismatch_rollback_ok) :-
    user:wam_append_mismatch_rollback(restored).
token_swi(repeated_mismatch_rollback, ok, repeated_rollback_ok).
token_swi(c_unifier_rollback, ok, c_rollback_ok).
token_swi(caller_continuation, ok, continuation_ok) :-
    S1 = start, append([1], [2], _), S2 = done,
    S1 == start, S2 == done.
token_swi(unbound_first_arg, runtime_error, unbound_ok).
token_swi(cyclic, runtime_error, cyclic_ok).
token_swi(open_list, runtime_error, open_ok).
token_swi(non_list, runtime_error, non_list_ok).
token_swi(improper, runtime_error, improper_ok).

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
    ->  Case = c_case(Id, fail, _),
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

test_generation_append_builtin :-
    Test = 'append/3: generated runtime contains append handler',
    (   compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(HelpersS, _, _, _, 'strcmp(op, "append/3")'),
        sub_string(HelpersS, _, _, _, 'wam_execute_append'),
        sub_string(HelpersS, _, _, _, 'wam_collect_append_list'),
        sub_string(HelpersS, _, _, _, 'wam_build_list_from_append_items')
    ->  pass(Test)
    ;   fail_test(Test, 'append/3 handler missing from generated runtime')
    ).

test_wam_emits_append_builtin :-
    Test = 'append/3: WAM text emits builtin_call append/3',
    setup_append_preds,
    (   compile_predicate_to_wam(user:wam_append_q/3, [], Wam),
        sub_string(Wam, _, _, _, 'builtin_call append/3')
    ->  pass(Test)
    ;   fail_test(Test, 'wam_append_q/3 WAM missing builtin_call append/3')
    ).

test_wrong_answer_rejected :-
    Test = 'append/3: SWI comparator rejects a wrong answer',
    append([c, a], [b, d], SWI),
    Wrong = [c, a, b, x],
    (   same_success(SWI, SWI),
        \+ same_success(SWI, Wrong),
        \+ same_success(SWI, _)
    ->  pass(Test)
    ;   fail_test(Test, 'actual comparator failed positive or negative controls')
    ).

compile_one(Pred, Code) :-
    compile_predicate_to_wam(Pred, [], Wam),
    compile_wam_predicate_to_c(Pred, Wam, [], Code).

test_compiled_c_append_matches_swi :-
    Test = 'append/3: compiled C solutions match SWI',
    (   gcc_available
    ->  (   run_compiled_c_append
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C append executable failed or mismatched SWI')
        )
    ;   fail_test(Test, 'gcc unavailable; compiled behavior not verified')
    ).

run_compiled_c_append :-
    setup_append_preds,
    compile_one(user:wam_append_q/3, QCode),
    compile_one(user:wam_append_continuation/4, ContCode),
    compile_one(user:wam_append_bind_control/1, BindCode),
    compile_one(user:wam_append_backtrack/1, BackCode),
    compile_one(user:wam_append_mismatch_positive/1, MisPosCode),
    compile_one(user:wam_append_mismatch_rollback/1, MismatchCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_append', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    atomic_list_concat([QCode, "\n\n", ContCode, "\n\n", BindCode, "\n\n", BackCode, "\n\n", MisPosCode, "\n\n", MismatchCode], AllPredCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [AllPredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    Driver = 'examples/pkg_resolver/c/append_driver.c',
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc),
                   ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
                    RuntimePath, PredPath, Driver, '-lm', '-o', ExePath],
                   [process(Pid)]),
    process_wait(Pid, GccStatus),
    format('gcc orig_exit=~w~n', [GccStatus]),
    GccStatus == exit(0),
    process_create(path(timeout), ['10', ExePath],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(RunPid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(RunPid, RunStatus),
    format('runner orig_exit=~w~n', [RunStatus]),
    (   ErrStr == ""
    ->  true
    ;   format(user_error, '~w', [ErrStr])
    ),
    RunStatus == exit(0),
    split_string(OutStr, "\n", "", Lines),
    parse_c_cases(Lines, CCases),
    Ground = [empty_empty, empty_nonempty, singleton_empty, singleton_singleton,
              duplicates, unordered_ints, unordered_atoms,
              negatives, ints_and_atoms, pairs, compounds, lists_and_compounds,
              nested_lists, five_elements, arbitrary_tail],
    Tokens = [empty_tail_sharing, spine_copy_tail_share,
              prebound_match, prebound_mismatch, preserve_input, heap_growth,
              shared_var, distinct_vars, cell_vars, compound_shared_var,
              bind_control, backtrack_rollback,
              mismatch_positive, mismatch_rollback, repeated_mismatch_rollback,
              c_unifier_rollback, caller_continuation,
              unbound_first_arg, cyclic, open_list, non_list, improper],
    findall(Id, (member(Id, Ground), \+ compare_ground(Id, CCases)), GroundBads),
    findall(Id, (member(Id, Tokens), \+ compare_token(Id, CCases)), TokenBads),
    (   member(c_case(unordered_atoms, ok, Atoms), CCases)
    ->  Atoms \== [c, a, b, wrong]
    ;   true
    ),
    GroundBads == [],
    TokenBads == [].

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
    format('~n=== WAM-C append/3 Tests ===~n~n'),
    setup_append_preds,
    test_generation_append_builtin,
    test_wam_emits_append_builtin,
    test_wrong_answer_rejected,
    test_compiled_c_append_matches_swi,
    cleanup_append_preds,
    format('~n=== WAM-C append/3 Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
