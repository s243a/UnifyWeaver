:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Dedicated C indexed-dispatch regression: try/retry/trust (not
% try_me_else/retry_me_else/trust_me). Compares compiled C solutions
% with SWI, not merely emitted instruction strings.
%
%   swipl -g run_tests -t halt tests/test_wam_c_indexed_dispatch.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(lists), [member/2, nth1/3]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

idx_out_dir(Dir) :-
    Dir = 'examples/pkg_resolver/c/generated/indexed_dispatch'.

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

cleanup_idx_preds :-
    retractall(user:idx_p(_, _)),
    retractall(user:idx_choice(_)),
    retractall(user:idx_wrap(_, _)),
    retractall(user:idx_guard(_, _)),
    retractall(user:idx_list(_, _)),
    retractall(user:idx_s(_, _)),
    retractall(user:idx_all_b(_)),
    retractall(user:idx_all_var(_)),
    retractall(user:idx_all_wrap(_)),
    retractall(user:idx_all_guard(_)),
    retractall(user:idx_all_list(_, _)),
    retractall(user:idx_all_s(_)),
    retractall(user:idx_order_lt(_, _)),
    retractall(user:idx_order_val(_, _)).

setup_idx_preds :-
    cleanup_idx_preds,
    assertz(user:idx_p(a, 1)),
    assertz(user:idx_p(b, 2)),
    assertz(user:idx_p(b, 3)),
    assertz(user:idx_p(b, 4)),
    assertz(user:idx_p(c, 5)),
    assertz(user:idx_choice(10)),
    assertz(user:idx_choice(20)),
    assertz((user:idx_wrap(X, Y) :- idx_choice(X), idx_p(b, Y))),
    assertz(user:idx_guard(a, 1)),
    assertz((user:idx_guard(b, 2) :- fail)),
    assertz(user:idx_guard(b, 3)),
    assertz(user:idx_guard(b, 4)),
    assertz(user:idx_list(a, 0)),
    assertz(user:idx_list([_|_], 1)),
    assertz(user:idx_list([_|_], 2)),
    assertz(user:idx_s(n, 0)),
    assertz(user:idx_s(bar(_), 1)),
    assertz(user:idx_s(bar(_), 2)),
    assertz((user:idx_all_b(Ys) :- findall(Y, idx_p(b, Y), Ys))),
    assertz((user:idx_all_var(Ps) :- findall(X-Y, idx_p(X, Y), Ps))),
    assertz((user:idx_all_wrap(Ps) :- findall(X-Y, idx_wrap(X, Y), Ps))),
    assertz((user:idx_all_guard(Ys) :- findall(Y, idx_guard(b, Y), Ys))),
    assertz((user:idx_all_list(L, Ys) :- findall(Y, idx_list(L, Y), Ys))),
    assertz((user:idx_all_s(Ys) :- findall(Y, idx_s(bar(z), Y), Ys))),
    assertz((user:idx_order_lt([], []) :- !, fail)),
    assertz((user:idx_order_lt([], [C|_]) :-
                 idx_order_val(C, V), 0 < V)),
    assertz((user:idx_order_lt([C|_], []) :-
                 idx_order_val(C, V), V < 0)),
    assertz((user:idx_order_lt([A|As], [B|Bs]) :-
                 idx_order_val(A, VA),
                 idx_order_val(B, VB),
                 (   VA < VB
                 ->  true
                 ;   VA =:= VB,
                     idx_order_lt(As, Bs)
                 ))),
    assertz((user:idx_order_val(126, -1) :- !)),
    assertz((user:idx_order_val(C, C) :- C >= 65, C =< 90, !)),
    assertz((user:idx_order_val(C, C) :- C >= 97, C =< 122, !)),
    assertz((user:idx_order_val(C, V) :- V is C + 256)).

idx_pred_list([
    idx_p/2,
    idx_choice/1,
    idx_wrap/2,
    idx_guard/2,
    idx_list/2,
    idx_s/2,
    idx_all_b/1,
    idx_all_var/1,
    idx_all_wrap/1,
    idx_all_guard/1,
    idx_all_list/2,
    idx_all_s/1,
    idx_order_lt/2,
    idx_order_val/2
]).

write_setup_inc(Preds, Path) :-
    setup_call_cleanup(
        open(Path, write, S),
        write_setup_inc_stream(S, Preds),
        close(S)
    ).

write_setup_inc_stream(S, Preds) :-
    format(S, "/* Generated -- do not edit. */~n", []),
    forall(member(N/A, Preds),
           format(S, "void setup_~w_~w(WamState* state);~n", [N, A])),
    format(S, "void setup_detected_wam_c_kernels(WamState* state);~n", []),
    format(S, "void setup_lowered_wam_c_helpers(WamState* state);~n~n", []),
    format(S, "static void setup_all_predicates(WamState *state) {~n", []),
    forall(member(N/A, Preds),
           format(S, "    setup_~w_~w(state);~n", [N, A])),
    format(S, "    setup_detected_wam_c_kernels(state);~n", []),
    format(S, "    setup_lowered_wam_c_helpers(state);~n", []),
    format(S, "}~n", []).

qualify_user(P/A, user:P/A).

swi_case(first_b, ok(Y)) :-
    user:idx_p(b, Y), !.
swi_case(first_var, ok(X)) :-
    user:idx_p(X, _), !.
swi_case(miss_z, fail) :-
    \+ user:idx_p(z, _).
swi_case(all_b, ok(Ys)) :-
    user:idx_all_b(Ys).
swi_case(all_var, ok(Ps)) :-
    user:idx_all_var(Ps).
swi_case(all_wrap, ok(Ps)) :-
    user:idx_all_wrap(Ps).
swi_case(all_guard, ok(Ys)) :-
    user:idx_all_guard(Ys).
swi_case(all_list, ok(Ys)) :-
    user:idx_all_list([x], Ys).
swi_case(all_s, ok(Ys)) :-
    user:idx_all_s(Ys).
swi_case(first_list, ok(Y)) :-
    user:idx_list([x], Y), !.
swi_case(first_struct, ok(Y)) :-
    user:idx_s(bar(z), Y), !.

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

compare_ok(Id, CCases) :-
    (   swi_case(Id, SWI)
    ->  true
    ;   fail_test(Id, 'missing SWI oracle'),
        fail
    ),
    (   member(c_case(Id, CStatus, CTerm), CCases)
    ->  true
    ;   fail_test(Id, 'missing C output'),
        fail
    ),
    (   SWI = fail
    ->  (   CStatus == fail
        ->  pass(Id)
        ;   format(atom(R), 'SWI fail vs C ~w ~q', [CStatus, CTerm]),
            fail_test(Id, R),
            fail
        )
    ;   SWI = ok(Expected)
    ->  (   CStatus == ok,
            same_success(Expected, CTerm)
        ->  pass(Id)
        ;   format(atom(R), 'SWI ~q vs C ~w ~q', [Expected, CStatus, CTerm]),
            fail_test(Id, R),
            fail
        )
    ).

test_wam_emits_indexed_chain :-
    Test = 'indexed dispatch: WAM text has try/retry/trust not only try_me_else',
    setup_idx_preds,
    (   compile_predicate_to_wam(user:idx_p/2, [], Wam),
        sub_string(Wam, _, _, _, 'try L_idx_p_2_'),
        sub_string(Wam, _, _, _, 'retry L_idx_p_2_'),
        sub_string(Wam, _, _, _, 'trust L_idx_p_2_'),
        sub_string(Wam, _, _, _, 'try_me_else')
    ->  pass(Test)
    ;   fail_test(Test, 'idx_p/2 WAM missing indexed try/retry/trust chain')
    ).

test_c_emits_distinct_tags :-
    Test = 'indexed dispatch: C emits INSTR_TRY distinct from INSTR_TRY_ME_ELSE',
    setup_idx_preds,
    (   compile_predicate_to_wam(user:idx_p/2, [], Wam),
        compile_wam_predicate_to_c(user:idx_p/2, Wam, [], CCode),
        atom_string(CCode, S),
        sub_string(S, _, _, _, 'INSTR_TRY,'),
        sub_string(S, _, _, _, 'INSTR_RETRY,'),
        sub_string(S, _, _, _, 'INSTR_TRUST,'),
        sub_string(S, _, _, _, 'INSTR_TRY_ME_ELSE')
    ->  pass(Test)
    ;   fail_test(Test, 'idx_p/2 C missing distinct INSTR_TRY/RETRY/TRUST')
    ).

test_order_lt_compiles :-
    Test = 'indexed dispatch: order_lt-shaped list chain compiles',
    setup_idx_preds,
    (   compile_predicate_to_wam(user:idx_order_lt/2, [], Wam),
        sub_string(Wam, _, _, _, 'try L_idx_order_lt_2_'),
        compile_wam_predicate_to_c(user:idx_order_lt/2, Wam, [], CCode),
        atom_string(CCode, S),
        sub_string(S, _, _, _, 'INSTR_TRY,')
    ->  pass(Test)
    ;   fail_test(Test, 'idx_order_lt/2 still fails C codegen')
    ).

test_compiled_c_matches_swi :-
    Test = 'indexed dispatch: compiled C solutions match SWI',
    (   gcc_available
    ->  (   run_compiled_c_matches_swi
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C vs SWI mismatch or build failure')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable)~n', [Test])
    ).

run_compiled_c_matches_swi :-
    setup_idx_preds,
    idx_out_dir(Dir),
    make_directory_path(Dir),
    idx_pred_list(PredsBare),
    maplist(qualify_user, PredsBare, Preds),
    write_wam_c_project(Preds,
                        [no_kernels(true), lowered_helpers(false)],
                        Dir),
    directory_file_path(Dir, 'lib.c', LibPath),
    read_file_to_string(LibPath, Lib, []),
    (   sub_string(Lib, _, _, _, ': compilation failed')
    ->  format(user_error, 'lib.c contains compilation failed~n', []),
        fail
    ;   true
    ),
    sub_string(Lib, _, _, _, 'INSTR_TRY,'),
    sub_string(Lib, _, _, _, 'INSTR_RETRY,'),
    sub_string(Lib, _, _, _, 'INSTR_TRUST,'),
    directory_file_path(Dir, 'setup_all.inc', SetupPath),
    write_setup_inc(PredsBare, SetupPath),
    Driver = 'examples/pkg_resolver/c/indexed_dispatch_driver.c',
    directory_file_path(Dir, 'idx_runner', ExePath),
    directory_file_path(Dir, 'gcc.log', GccLog),
    directory_file_path(Dir, 'run.out', RunOut),
    directory_file_path(Dir, 'run.err', RunErr),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    directory_file_path(Dir, 'wam_runtime.c', RuntimePath),
    format(atom(GccCmd),
           'gcc -std=c11 -Wall -Wextra -I ~w -I ~w ~w ~w ~w -lm -o ~w > ~w 2>&1',
           [IncludeDir, Dir, RuntimePath, LibPath, Driver, ExePath, GccLog]),
    shell(GccCmd, GccStatus),
    format('gcc orig_exit=~w (log ~w)~n', [GccStatus, GccLog]),
    !,
    GccStatus =:= 0,
    format(atom(RunCmd), 'timeout 10 ~w > ~w 2> ~w', [ExePath, RunOut, RunErr]),
    shell(RunCmd, RunStatus),
    format('runner orig_exit=~w (out ~w)~n', [RunStatus, RunOut]),
    !,
    RunStatus =:= 0,
    read_file_to_string(RunOut, Out, []),
    split_string(Out, "\n", "", Lines),
    parse_c_cases(Lines, CCases),
    Cases = [first_b, first_var, miss_z,
             all_b, all_var, all_wrap, all_guard, all_list, all_s,
             first_list, first_struct],
    findall(Id, (member(Id, Cases), \+ compare_ok(Id, CCases)), Bads),
    (   member(c_case(all_b, ok, AllB), CCases)
    ->  AllB \== [9, 9, 9]
    ;   true
    ),
    Bads == [].

test_wrong_answer_rejected :-
    Test = 'indexed dispatch: SWI comparator rejects a wrong answer',
    setup_idx_preds,
    user:idx_all_b(SWI),
    Wrong = [9,9,9],
    (   same_success(SWI, SWI),
        \+ same_success(SWI, Wrong),
        \+ same_success(SWI, _)
    ->  pass(Test)
    ;   fail_test(Test, 'actual comparator failed positive or negative controls')
    ).

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
    format('~n=== WAM-C Indexed Dispatch Tests ===~n~n'),
    setup_idx_preds,
    test_wam_emits_indexed_chain,
    test_c_emits_distinct_tags,
    test_order_lt_compiles,
    test_wrong_answer_rejected,
    test_compiled_c_matches_swi,
    cleanup_idx_preds,
    format('~n=== WAM-C Indexed Dispatch Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
