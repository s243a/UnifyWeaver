:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C member/2: generated/compiled C vs SWI for finite proper lists.
% Open/improper/cyclic/non-list/unbound lists are diagnosed, not claimed
% supported. Nondeterministic solutions are ordered and keep duplicates.
%
%   swipl -g run_tests -t halt tests/test_wam_c_member.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).
:- use_module(library(lists), [member/2]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.
:- dynamic user:wam_member_q/2.
:- dynamic user:wam_member_all/2.
:- dynamic user:wam_member_then/3.
:- dynamic user:wam_member_cut_then/3.
:- dynamic user:wam_member_nested/1.
:- dynamic user:wam_member_cut_all/2.
:- dynamic user:wam_member_grow/4.

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

cleanup_member_preds :-
    retractall(user:wam_member_q(_, _)),
    retractall(user:wam_member_all(_, _)),
    retractall(user:wam_member_then(_, _, _)),
    retractall(user:wam_member_cut_then(_, _, _)),
    retractall(user:wam_member_nested(_)),
    retractall(user:wam_member_cut_all(_, _)),
    retractall(user:wam_member_grow(_, _, _, _)).

setup_member_preds :-
    cleanup_member_preds,
    assertz((user:wam_member_q(E, L) :- member(E, L))),
    assertz((user:wam_member_all(L, Xs) :- findall(X, member(X, L), Xs))),
    assertz((user:wam_member_then(E, L, C) :- member(E, L), E = C)),
    assertz((user:wam_member_cut_then(E, L, C) :- member(E, L), !, E = C)),
    assertz((user:wam_member_nested(P) :-
                 findall(X-Y, (member(X, [a, b]), member(Y, [1, 2])), P))),
    assertz((user:wam_member_cut_all(L, Xs) :-
                 findall(X, (member(X, L), !), Xs))),
    assertz((user:wam_member_grow(E, L, C, Pad) :-
                 member(E, L), sort(Pad, _), E = C)).

ground_all_input(empty, []).
ground_all_input(singleton, [a]).
ground_all_input(duplicates, [a, a, b]).
ground_all_input(abc, [a, b, c]).
ground_all_input(compounds, [f(a), b]).

ground_all_swi(Id, Expected) :-
    ground_all_input(Id, Input),
    findall(X, member(X, Input), Expected).

ground_q_swi(later, ok, b) :-
    member(b, [a, b, c]).
ground_q_swi(missing, fail, _) :-
    \+ member(d, [a, b, c]).
ground_q_swi(empty_q, fail, _) :-
    \+ member(_, []).

token_swi(prebound_match, ok, prebound_ok) :-
    member(b, [a, b, c]).
token_swi(prebound_mismatch, fail, prebound_mismatch_ok) :-
    \+ member(d, [a, b, c]).
token_swi(preserve_input, ok, preserve_ok).
token_swi(downstream, ok, downstream_ok) :-
    member(E, [a, b, c]), E = c.
token_swi(cut_commit, fail, cut_ok) :-
    \+ (member(E, [a, b, c]), !, E = c).
token_swi(partial, ok, partial_ok) :-
    member(f(X, 2), [f(a, 1), f(b, 2)]), X == b.
token_swi(nested, ok, nested_ok) :-
    findall(X-Y, (member(X, [a, b]), member(Y, [1, 2])),
            [a-1, a-2, b-1, b-2]).
token_swi(shared_var, ok, shared_ok).
token_swi(cut_all, ok, cut_all_ok) :-
    findall(X, (member(X, [a, b, c]), !), [a]).
token_swi(trail_no_growth, ok, trail_nogrow_ok) :-
    member(E, [a, b, c]), sort([1, 2, 3], _), E = c.
token_swi(distinct_vars, ok, distinct_ok).
token_swi(cell_vars, ok, cell_vars_ok).
token_swi(heap_growth, ok, heap_growth_ok) :-
    numlist(1, 1024, L),
    findall(X, member(X, L), L).
token_swi(cyclic, runtime_error, cyclic_ok).
token_swi(open_list, runtime_error, open_ok).
token_swi(non_list, runtime_error, non_list_ok).
token_swi(improper, runtime_error, improper_ok).
token_swi(unbound_list, runtime_error, unbound_ok).
token_swi(repeated, ok, repeated_ok).

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

compare_ground_all(Id, CCases) :-
    ground_all_swi(Id, Expected),
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

compare_ground_q(Id, CCases) :-
    ground_q_swi(Id, SWIStatus, Expected),
    (   member(c_case(Id, CStatus, CTerm), CCases)
    ->  true
    ;   fail_test(Id, 'missing C output'),
        fail
    ),
    (   SWIStatus == fail
    ->  (   CStatus == fail
        ->  pass(Id)
        ;   format(atom(R), 'expected fail vs C ~w ~q', [CStatus, CTerm]),
            fail_test(Id, R),
            fail
        )
    ;   CStatus == ok,
        same_success(Expected, CTerm)
    ->  pass(Id)
    ;   format(atom(R), 'SWI ~q vs C ~w ~q', [Expected, CStatus, CTerm]),
        fail_test(Id, R),
        fail
    ).

report_trail_no_growth(CCases) :-
    Test = 'trail_no_growth',
    token_swi(trail_no_growth, ok, trail_nogrow_ok),
    (   member(c_case(trail_no_growth, ok, trail_nogrow_ok), CCases)
    ->  pass(Test)
    ;   member(c_case(trail_no_growth, fail, trail_nogrow_bad), CCases)
    ->  fail_test(Test, 'retry without heap growth returned the wrong result (register/CP/Y, not realloc)')
    ;   fail_test(Test, 'missing C output for no-growth member/sort/retry control')
    ).

report_trail_after_growth(CCases) :-
    Test = 'trail_after_growth',
    (   member(c_case(trail_after_growth, ok, trail_ok), CCases)
    ->  pass(Test)
    ;   member(c_case(trail_after_growth, fail, trail_bad), CCases)
    ->  fail_test(Test, 'growth followed by retry returned the wrong result')
    ;   fail_test(Test, 'missing C output for realloc-after-bind probe')
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

compile_one(PI, Code) :-
    compile_predicate_to_wam(PI, [], Wam),
    compile_wam_predicate_to_c(PI, Wam, [], Code).

test_generation_member_builtin :-
    Test = 'member/2: generated runtime contains backtracking member',
    (   compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        compile_step_wam_to_c([], StepCode),
        atom_string(StepCode, StepS),
        sub_string(HelpersS, _, _, _, 'strcmp(op, "member/2")'),
        sub_string(HelpersS, _, _, _, 'wam_execute_member'),
        sub_string(HelpersS, _, _, _, 'wam_member_bind_from'),
        sub_string(HelpersS, _, _, _, 'wam_sort_identity_value'),
        sub_string(StepS, _, _, _, 'WAM_MEMBER_NEXT')
    ->  pass(Test)
    ;   fail_test(Test, 'member/2 handler missing from generated runtime')
    ).

test_wam_emits_member_builtin :-
    Test = 'member/2: WAM text emits builtin_call member/2',
    setup_member_preds,
    (   compile_predicate_to_wam(user:wam_member_q/2, [], Wam),
        sub_string(Wam, _, _, _, 'builtin_call member/2')
    ->  pass(Test)
    ;   fail_test(Test, 'wam_member_q/2 WAM missing builtin_call member/2')
    ).

test_wrong_answer_rejected :-
    Test = 'member/2: SWI comparator rejects a wrong answer',
    findall(X, member(X, [a, a, b]), SWI),
    Wrong = [a, b],
    DroppedDup = [a, b],
    Reordered = [b, a, a],
    (   same_success(SWI, SWI),
        \+ same_success(SWI, Wrong),
        \+ same_success(SWI, DroppedDup),
        \+ same_success(SWI, Reordered),
        \+ same_success(SWI, _)
    ->  pass(Test)
    ;   fail_test(Test, 'actual comparator failed positive or negative controls')
    ).

test_compiled_c_member_matches_swi :-
    Test = 'member/2: compiled C solutions match SWI',
    (   gcc_available
    ->  (   run_compiled_c_member
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C member executable failed or mismatched SWI')
        )
    ;   fail_test(Test, 'gcc unavailable; compiled behavior not verified')
    ).

run_compiled_c_member :-
    setup_member_preds,
    compile_predicate_to_wam(user:wam_member_q/2, [], QWam),
    sub_string(QWam, _, _, _, 'builtin_call member/2'),
    compile_predicate_to_wam(user:wam_member_all/2, [], AllWam),
    sub_string(AllWam, _, _, _, 'builtin_call member/2'),
    compile_one(user:wam_member_q/2, QCode),
    compile_one(user:wam_member_all/2, AllCode),
    compile_one(user:wam_member_then/3, ThenCode),
    compile_one(user:wam_member_cut_then/3, CutThenCode),
    compile_one(user:wam_member_nested/1, NestedCode),
    compile_one(user:wam_member_cut_all/2, CutAllCode),
    compile_one(user:wam_member_grow/4, GrowCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_member', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    atomic_list_concat([QCode, AllCode, ThenCode, CutThenCode, NestedCode,
                        GrowCode, CutAllCode],
                       '\n\n', PredCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    Driver = 'examples/pkg_resolver/c/member_driver.c',
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
    split_string(OutStr, "\n", "", Lines),
    parse_c_cases(Lines, CCases),
    GroundAll = [empty, singleton, duplicates, abc, compounds],
    GroundQ = [later, missing, empty_q],
    Tokens = [prebound_match, prebound_mismatch, preserve_input, downstream,
              cut_commit, cut_all, partial, nested, shared_var, distinct_vars,
              cell_vars, heap_growth, cyclic, open_list, non_list, improper,
              unbound_list, repeated],
    findall(Id, (member(Id, GroundAll), \+ compare_ground_all(Id, CCases)), AllBads),
    findall(Id, (member(Id, GroundQ), \+ compare_ground_q(Id, CCases)), QBads),
    findall(Id, (member(Id, Tokens), \+ compare_token(Id, CCases)), TokenBads),
    report_trail_no_growth(CCases),
    report_trail_after_growth(CCases),
    (   member(c_case(duplicates, ok, Dups), CCases)
    ->  Dups \== [a, b]
    ;   true
    ),
    AllBads == [],
    QBads == [],
    TokenBads == [],
    RunStatus == exit(0).

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
    format('~n=== WAM-C member/2 Tests ===~n~n'),
    setup_member_preds,
    test_generation_member_builtin,
    test_wam_emits_member_builtin,
    test_wrong_answer_rejected,
    test_compiled_c_member_matches_swi,
    cleanup_member_preds,
    format('~n=== WAM-C member/2 Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
