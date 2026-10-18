:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_scala_classic_programs.pl
%
% End-to-end regression tests for the WAM Scala target running
% classic Prolog programs (list reverse, naive reverse, Ackermann,
% Fibonacci, simple n-queens). Each test compiles a small Prolog
% program to a Scala project and verifies a known answer.
%
% These complement the smoke tests by exercising real recursive
% Prolog programs end-to-end, not just synthetic predicate shapes.
% Gated on `scalac`/`scala` being on PATH (or SCALA_SMOKE_TESTS=1).

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').

% ============================================================
% Sample programs
% ============================================================

% --- 0-arity comparison-only predicates ---
% Regression for the Scala-target gap the cross-target conformance doc
% flagged: a 0-arity body like `p :- 3 > 2.` used to break the backend.
% emit_scala_wrapper/4 called numlist(1, 0, _) (which fails when Low >
% High), silently failing the whole project write; and the CLI driver's
% single-query dispatch required at least one argument, so even a
% compiled 0-arity predicate could not be invoked. Both are fixed.
:- dynamic user:zarity_true/0.
:- dynamic user:zarity_false/0.
user:zarity_true  :- 3 > 2.
user:zarity_false :- 3 > 4.

% --- List reverse via accumulator (linear time) ---
:- dynamic user:rev_acc/3.
:- dynamic user:list_reverse/2.
user:rev_acc([], A, A).
user:rev_acc([H|T], A, R) :- user:rev_acc(T, [H|A], R).
user:list_reverse(L, R) :- user:rev_acc(L, [], R).

% --- Naive reverse (O(n²) — uses append/3) ---
:- dynamic user:nrev/2.
user:nrev([], []).
user:nrev([H|T], R) :- user:nrev(T, T2), append(T2, [H], R).

% --- Ackermann (heavy recursion + arithmetic) ---
:- dynamic user:ack/3.
user:ack(0, N, R)  :- R is N + 1.
user:ack(M, 0, R)  :- M > 0, M1 is M - 1, user:ack(M1, 1, R).
user:ack(M, N, R)  :- M > 0, N > 0,
                     M1 is M - 1, N1 is N - 1,
                     user:ack(M, N1, R1),
                     user:ack(M1, R1, R).

% --- Fibonacci (naïve doubly-recursive) ---
:- dynamic user:fib/2.
user:fib(0, 0).
user:fib(1, 1).
user:fib(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                  user:fib(N1, R1), user:fib(N2, R2),
                  R is R1 + R2.

% --- Small expression evaluator ---
% Recursive evaluator for arithmetic expressions built as compound
% terms (+, *, -). Tests structure pattern matching on heads with
% non-list compound terms, plus arithmetic via is/2.
:- dynamic user:wam_eval/2.
user:wam_eval(N, R)     :- number(N), !, R = N.
user:wam_eval(A + B, R) :- user:wam_eval(A, RA),
                           user:wam_eval(B, RB),
                           R is RA + RB.
user:wam_eval(A * B, R) :- user:wam_eval(A, RA),
                           user:wam_eval(B, RB),
                           R is RA * RB.
user:wam_eval(A - B, R) :- user:wam_eval(A, RA),
                           user:wam_eval(B, RB),
                           R is RA - RB.

% --- N-queens via permutation + safe ---
% Uses between/3 to build the row list (1..N), permutation/2 to try
% column assignments, and safe/1 to reject conflicts. Combinatorial
% test that exercises switch_on_term, multi-solution backtracking,
% arithmetic comparisons, and recursive predicates all together.

:- dynamic user:numlist_uw/3.
:- dynamic user:select_uw/3.
:- dynamic user:permutation_uw/2.
:- dynamic user:safe_q/1.
:- dynamic user:safe_q_aux/3.
:- dynamic user:queens_q/2.

user:numlist_uw(L, H, [L|T]) :- L =< H, L1 is L + 1, user:numlist_uw(L1, H, T).
user:numlist_uw(L, H, [])    :- L > H.

user:select_uw(H, [H|T], T).
user:select_uw(H, [X|T], [X|T2]) :- user:select_uw(H, T, T2).

user:permutation_uw([], []).
user:permutation_uw(L, [H|T]) :- user:select_uw(H, L, Rest), user:permutation_uw(Rest, T).

user:safe_q([]).
user:safe_q([Q|Qs]) :- user:safe_q_aux(Qs, Q, 1), user:safe_q(Qs).
user:safe_q_aux([], _, _).
user:safe_q_aux([Q|Qs], Q0, D0) :-
    Q =\= Q0 + D0,
    Q =\= Q0 - D0,
    D1 is D0 + 1,
    user:safe_q_aux(Qs, Q0, D1).

user:queens_q(N, Qs) :-
    user:numlist_uw(1, N, L),
    user:permutation_uw(L, Qs),
    user:safe_q(Qs).

% --- Nested head shapes + strict (in)equality ---
% Both defects below were live while every classic program above was
% green, because each of those queries is GROUND: a head is always
% matched, never used to bind the caller's term.
%
% (1) A structure nested inside a list pattern. The WAM emits the
%     nested term interleaved with the enclosing one:
%         get_list A1        % queue = [head, tail]
%         unify_variable X1  % X1 := head,  queue = [tail]
%         get_structure tk/1, X1
%         unify_variable X2  % tk's argument
%         unify_variable X3  % <- the enclosing list's TAIL
%     A single unify queue lost `[tail]` at the get_structure, so the
%     final unify_variable found an empty queue and failed the clause.
%     nest_pair/2 is the shape; nest_body/2 is the same match written
%     as an explicit body goal and passed throughout — it is the
%     control that proves the defect is in nested head matching.
% (2) ==/2 and \==/2 were absent from the Scala builtin table
%     entirely, so the shared compiler's builtin_call failed closed.
:- dynamic user:nest_pair/2.
:- dynamic user:nest_body/2.
:- dynamic user:one_slot/2.
:- dynamic user:bind_one/1.
:- dynamic user:same_slots/2.
:- dynamic user:repeat_head/1.
:- dynamic user:strict_eq/1.
:- dynamic user:strict_neq/1.
:- dynamic user:strict_eq_compound/1.
user:nest_pair([tk(X)|_], X).
user:nest_body([H|_], X) :- H = tk(X).
% A head must BIND the caller's unbound structure slot, not just match it.
user:one_slot(q(U), U).
user:bind_one(X) :- T = q(A), user:one_slot(T, X), A == X.
% A repeated variable inside the head structure must bind BOTH slots.
user:same_slots(p(U, U), U).
user:repeat_head(X) :- T = p(_, _), user:same_slots(T, X), T == p(X, X).
% Strict (in)equality, atomic and compound.
user:strict_eq(X)          :- X == foo.
user:strict_neq(X)         :- X \== foo.
user:strict_eq_compound(X) :- Y = f(X, b), Y == f(foo, b).

% ============================================================
% scala_available — same gating pattern as the smoke tests
% ============================================================

scala_available :-
    (   getenv('SCALA_SMOKE_TESTS', "1") -> true
    ;   catch(
            process_create(path(scalac), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            _,
            fail
        ),
        process_wait(Pid, exit(0))
    ).

% ============================================================
% Tests
% ============================================================

:- begin_tests(wam_scala_classic_programs,
               [ condition(scala_available) ]).

% 0-arity comparison-only predicates: compile + dispatch with no args.
test(zero_arity_comparison) :-
    with_scala_project(
        [user:zarity_true/0, user:zarity_false/0],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'zarity_true/0',  [], "true"),
            verify_scala_args(TmpDir, 'zarity_false/0', [], "false")
        )).

test(list_reverse) :-
    with_scala_project(
        [user:rev_acc/3, user:list_reverse/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'list_reverse/2',
                              ['[a,b,c]', '[c,b,a]'], "true"),
            verify_scala_args(TmpDir, 'list_reverse/2',
                              ['[a,b,c,d]', '[d,c,b,a]'], "true"),
            verify_scala_args(TmpDir, 'list_reverse/2',
                              ['[]', '[]'], "true"),
            verify_scala_args(TmpDir, 'list_reverse/2',
                              ['[a]', '[a]'], "true"),
            verify_scala_args(TmpDir, 'list_reverse/2',
                              ['[a,b,c]', '[a,b,c]'], "false")
        )).

test(naive_reverse) :-
    with_scala_project(
        [user:nrev/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'nrev/2',
                              ['[a,b,c]', '[c,b,a]'], "true"),
            verify_scala_args(TmpDir, 'nrev/2',
                              ['[a,b,c,d]', '[d,c,b,a]'], "true"),
            verify_scala_args(TmpDir, 'nrev/2',
                              ['[]', '[]'], "true"),
            verify_scala_args(TmpDir, 'nrev/2',
                              ['[a,b,c]', '[a,c,b]'], "false")
        )).

test(ackermann) :-
    with_scala_project(
        [user:ack/3],
        _Opts,
        TmpDir,
        (
            % Standard Ackermann values:
            % ack(0, n) = n+1
            % ack(1, n) = n+2
            % ack(2, n) = 2n+3
            % ack(3, 3) = 61
            verify_scala_args(TmpDir, 'ack/3', ['0', '5', '6'],   "true"),
            verify_scala_args(TmpDir, 'ack/3', ['1', '5', '7'],   "true"),
            verify_scala_args(TmpDir, 'ack/3', ['2', '3', '9'],   "true"),
            verify_scala_args(TmpDir, 'ack/3', ['3', '3', '61'],  "true"),
            verify_scala_args(TmpDir, 'ack/3', ['3', '3', '60'],  "false")
        )).

test(fibonacci) :-
    with_scala_project(
        [user:fib/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'fib/2', ['0',  '0'],    "true"),
            verify_scala_args(TmpDir, 'fib/2', ['1',  '1'],    "true"),
            verify_scala_args(TmpDir, 'fib/2', ['10', '55'],   "true"),
            verify_scala_args(TmpDir, 'fib/2', ['15', '610'],  "true"),
            verify_scala_args(TmpDir, 'fib/2', ['10', '54'],   "false")
        )).

% Expression evaluator — exercises structure pattern matching with
% non-list compound terms, recursion, the cut from clause 1, and
% arithmetic via is/2.
test(expression_evaluator) :-
    with_scala_project(
        [user:wam_eval/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_eval/2', ['5', '5'],          "true"),
            verify_scala_args(TmpDir, 'wam_eval/2', ['+(2,3)', '5'],     "true"),
            verify_scala_args(TmpDir, 'wam_eval/2', ['*(+(2,3),4)', '20'], "true"),
            verify_scala_args(TmpDir, 'wam_eval/2', ['-(10,3)', '7'],    "true"),
            verify_scala_args(TmpDir, 'wam_eval/2', ['+(2,*(3,4))', '14'], "true"),
            verify_scala_args(TmpDir, 'wam_eval/2', ['*(+(2,3),4)', '19'], "false")
        )).

% N-queens — exercises permutation, recursion, comparison, switch_on_term,
% and multi-solution backtracking. The two valid 4-queens solutions are
% [2,4,1,3] and [3,1,4,2]; [1,2,3,4] is invalid (queens on same diagonal).
test(nqueens) :-
    with_scala_project(
        [user:numlist_uw/3, user:select_uw/3, user:permutation_uw/2,
         user:safe_q/1, user:safe_q_aux/3, user:queens_q/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'queens_q/2', ['4', '[2,4,1,3]'], "true"),
            verify_scala_args(TmpDir, 'queens_q/2', ['4', '[3,1,4,2]'], "true"),
            verify_scala_args(TmpDir, 'queens_q/2', ['4', '[1,2,3,4]'], "false"),
            verify_scala_args(TmpDir, 'queens_q/2', ['4', '[1,3,2,4]'], "false")
        )).

% One project for both fixes — scalac dominates the runtime, so the
% predicates share a build rather than getting a test each.
test(nested_heads_and_strict_equality) :-
    with_scala_project(
        [user:nest_pair/2, user:nest_body/2,
         user:one_slot/2, user:bind_one/1,
         user:same_slots/2, user:repeat_head/1,
         user:strict_eq/1, user:strict_neq/1, user:strict_eq_compound/1],
        _Opts,
        TmpDir,
        (
            % Structure nested inside a list pattern, and the
            % explicit-body control that always worked.
            verify_scala_args(TmpDir, 'nest_pair/2', ['[tk(a)]', 'a'], "true"),
            verify_scala_args(TmpDir, 'nest_pair/2', ['[tk(a),tk(b)]', 'a'], "true"),
            verify_scala_args(TmpDir, 'nest_pair/2', ['[tk(a)]', 'b'], "false"),
            verify_scala_args(TmpDir, 'nest_body/2', ['[tk(a)]', 'a'], "true"),
            % A head binding the caller's unbound slots.
            verify_scala_args(TmpDir, 'bind_one/1',   ['a'], "true"),
            verify_scala_args(TmpDir, 'repeat_head/1', ['a'], "true"),
            % ==/2 and \==/2.
            verify_scala_args(TmpDir, 'strict_eq/1',  ['foo'], "true"),
            verify_scala_args(TmpDir, 'strict_eq/1',  ['bar'], "false"),
            verify_scala_args(TmpDir, 'strict_neq/1', ['foo'], "false"),
            verify_scala_args(TmpDir, 'strict_neq/1', ['bar'], "true"),
            verify_scala_args(TmpDir, 'strict_eq_compound/1', ['foo'], "true"),
            verify_scala_args(TmpDir, 'strict_eq_compound/1', ['bar'], "false")
        )).

:- end_tests(wam_scala_classic_programs).

% ============================================================
% Test fixture helpers (mirrors test_wam_scala_runtime_smoke.pl)
% ============================================================

with_scala_project(Preds, ExtraOpts0, TmpDir, Goal) :-
    (   var(ExtraOpts0) -> ExtraOpts = [] ; ExtraOpts = ExtraOpts0 ),
    unique_scala_tmp_dir('tmp_scala_classic', TmpDir),
    BaseOpts = [ package('generated.wam_scala_classic.core'),
                 runtime_package('generated.wam_scala_classic.core'),
                 module_name('wam-scala-classic') ],
    append(ExtraOpts, BaseOpts, AllOpts),
    write_wam_scala_project(Preds, AllOpts, TmpDir),
    compile_scala_project(TmpDir),
    setup_call_cleanup(
        true,
        call(Goal),
        delete_directory_and_contents(TmpDir)).

compile_scala_project(ProjectDir) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    find_scala_sources(AbsProjectDir, Sources),
    Sources \= [],
    process_create(path(scalac),
                   ['-d', ClassDir | Sources],
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, _OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(scala_compile_failed(ExitCode, ErrStr), _))
    ).

find_scala_sources(AbsProjectDir, Sources) :-
    directory_file_path(AbsProjectDir, 'src', SrcDir),
    findall(F,
        ( directory_member(SrcDir, RelF,
              [extensions([scala]), recursive(true)]),
          directory_file_path(SrcDir, RelF, F)
        ),
        Sources).

verify_scala_args(ProjectDir, PredKey, Args, Expected) :-
    run_scala_predicate_args(ProjectDir, PredKey, Args, Actual),
    (   Actual == Expected
    ->  true
    ;   throw(error(assertion_error(PredKey, Args, Expected, Actual), _))
    ).

run_scala_predicate_args(ProjectDir, PredKey, Args, Output) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['-classpath', ClassDir,
            'generated.wam_scala_classic.core.GeneratedProgram',
            PredStr], ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    normalize_space(string(Output), OutStr0),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(scala_run_failed(ExitCode, PredKey, Args, ErrStr), _))
    ).

unique_scala_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).
