:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C sort/2: generated/compiled C vs SWI for finite proper lists.
% Invalid/open/cyclic lists are diagnosed, not claimed supported.
%
%   swipl -g run_tests -t halt tests/test_wam_c_sort.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).
:- use_module(library(lists), [member/2]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.
:- dynamic user:wam_sort_q/2.

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

cleanup_sort_pred :-
    retractall(user:wam_sort_q(_, _)).

setup_sort_pred :-
    cleanup_sort_pred,
    assertz((user:wam_sort_q(L, S) :- sort(L, S))).

ground_input(empty, []).
ground_input(singleton, [a]).
ground_input(duplicates, [b, a, b, a, c]).
ground_input(unordered_ints, [3, 1, 2]).
ground_input(unordered_atoms, [c, a, b]).
ground_input(negatives, [3, -1, 0, -5]).
ground_input(ints_and_atoms, [b, 2, a, 1]).
ground_input(pairs, [c-1, a-1, b-1]).
ground_input(compounds, [g(a, b), f(b), f(a)]).
ground_input(lists_and_compounds, [f(a), [a], a]).
ground_input(nested_lists, [[3, 1], [1, 2], [3, 1]]).
ground_input(list_binary_order, [[a], a(a, b), z(a), z(a, b)]).

ground_swi(Id, Expected) :- ground_input(Id, Input), sort(Input, Expected).

token_swi(prebound_match, ok, prebound_ok) :-
    sort([3, 1, 2], [1, 2, 3]).
token_swi(prebound_mismatch, fail, prebound_mismatch_ok) :-
    \+ sort([3, 1, 2], [3, 2, 1]).
token_swi(preserve_input, ok, preserve_ok).
token_swi(heap_growth, ok, heap_growth_ok) :-
    numlist(1, 1024, Ascending), reverse(Ascending, Descending),
    sort(Descending, Ascending).
token_swi(mixed_numeric, ok, mixed_ok) :-
    sort([1, 1.0, 2, 0], [0, 1.0, 1, 2]).
token_swi(shared_var, ok, shared_ok) :-
    sort([X, a, X], [X, a]).
token_swi(cell_vars, ok, cell_vars_ok) :-
    sort([X, Y], Sorted),
    Sorted = [A, B], A \== B,
    ( A == X, B == Y ; A == Y, B == X ).
token_swi(distinct_vars, ok, distinct_ok) :-
    sort([Y, X, Y], L),
    L = [A, B],
    A \== B,
    (   A == X, B == Y
    ;   A == Y, B == X
    ).
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

test_generation_sort_builtin :-
    Test = 'sort/2: generated runtime contains standard-order unique sort',
    (   compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(HelpersS, _, _, _, 'strcmp(op, "sort/2")'),
        sub_string(HelpersS, _, _, _, 'wam_execute_sort'),
        sub_string(HelpersS, _, _, _, 'wam_sort_identity_value'),
        sub_string(HelpersS, _, _, _, 'Does not reuse the aggregate stored-term comparator')
    ->  pass(Test)
    ;   fail_test(Test, 'sort/2 handler missing from generated runtime')
    ).

test_wam_emits_sort_builtin :-
    Test = 'sort/2: WAM text emits builtin_call sort/2',
    setup_sort_pred,
    (   compile_predicate_to_wam(user:wam_sort_q/2, [], Wam),
        sub_string(Wam, _, _, _, 'builtin_call sort/2')
    ->  pass(Test)
    ;   fail_test(Test, 'wam_sort_q/2 WAM missing builtin_call sort/2')
    ).

test_wrong_answer_rejected :-
    Test = 'sort/2: SWI comparator rejects a wrong answer',
    sort([c, a, b], SWI),
    Wrong = [c, a, b],
    (   same_success(SWI, SWI),
        \+ same_success(SWI, Wrong),
        \+ same_success(SWI, _)
    ->  pass(Test)
    ;   fail_test(Test, 'actual comparator failed positive or negative controls')
    ).

test_compiled_c_sort_matches_swi :-
    Test = 'sort/2: compiled C solutions match SWI',
    (   gcc_available
    ->  (   run_compiled_c_sort
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C sort executable failed or mismatched SWI')
        )
    ;   fail_test(Test, 'gcc unavailable; compiled behavior not verified')
    ).

run_compiled_c_sort :-
    setup_sort_pred,
    compile_predicate_to_wam(user:wam_sort_q/2, [], Wam),
    sub_string(Wam, _, _, _, 'builtin_call sort/2'),
    compile_wam_predicate_to_c(user:wam_sort_q/2, Wam, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_sort', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    Driver = 'examples/pkg_resolver/c/sort_driver.c',
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
    Ground = [empty, singleton, duplicates, unordered_ints, unordered_atoms,
              negatives, ints_and_atoms, pairs, compounds, lists_and_compounds,
              nested_lists, list_binary_order],
    Tokens = [prebound_match, prebound_mismatch, preserve_input, heap_growth, mixed_numeric,
              shared_var, cell_vars, distinct_vars, cyclic, open_list, non_list, improper],
    findall(Id, (member(Id, Ground), \+ compare_ground(Id, CCases)), GroundBads),
    findall(Id, (member(Id, Tokens), \+ compare_token(Id, CCases)), TokenBads),
    (   member(c_case(unordered_atoms, ok, Atoms), CCases)
    ->  Atoms \== [c, a, b]
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
    format('~n=== WAM-C sort/2 Tests ===~n~n'),
    setup_sort_pred,
    test_generation_sort_builtin,
    test_wam_emits_sort_builtin,
    test_wrong_answer_rejected,
    test_compiled_c_sort_matches_swi,
    cleanup_sort_pred,
    format('~n=== WAM-C sort/2 Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
