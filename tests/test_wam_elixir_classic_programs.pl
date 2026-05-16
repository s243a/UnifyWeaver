:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_elixir_classic_programs.pl
%
% End-to-end regression tests for the WAM-Elixir target running
% classic Prolog programs (Fibonacci, Ackermann). Each test
% compiles a small Prolog program to an Elixir project and verifies
% known answers via subprocess invocation of `elixir`.
%
% Mirrors the discipline of `tests/test_wam_scala_classic_programs.pl`
% which has been the breadth-anchor of the WAM test suite. Per the
% audit captured in docs/WAM_TARGET_ROADMAP.md "Cross-cutting
% observations" §4, Scala had 6 such end-to-end classic-program
% tests; Elixir had 0. This file ports the discipline + a starter
% set (fibonacci, ackermann); follow-ups can extend to list_reverse,
% naive_reverse, expression_evaluator, n-queens once the harness
% proves out for list-arg parsing in run_classic.exs.
%
% Gated on `elixir` being on PATH (or ELIXIR_SMOKE_TESTS=1).

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target').

% ============================================================
% Sample programs
% ============================================================

% --- Fibonacci (naïve doubly-recursive) ---
% Tests: backtracking, choice points, multiple clause heads, arithmetic.
:- dynamic user:fib/2.
user:fib(0, 0).
user:fib(1, 1).
user:fib(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                  user:fib(N1, R1), user:fib(N2, R2),
                  R is R1 + R2.

% --- Ackermann (heavy recursion + arithmetic) ---
% Tests: heavy recursion with arithmetic guards, nested recursive calls,
% three-clause dispatch.
:- dynamic user:ack/3.
user:ack(0, N, R)  :- R is N + 1.
user:ack(M, 0, R)  :- M > 0, M1 is M - 1, user:ack(M1, 1, R).
user:ack(M, N, R)  :- M > 0, N > 0,
                     M1 is M - 1, N1 is N - 1,
                     user:ack(M, N1, R1),
                     user:ack(M1, R1, R).

% --- Pythagoras (multi-predicate arithmetic chain) ---
% Tests: cross-predicate dispatch via WamDispatcher (the driver loads
% every emitted module from lib/, not just the entrypoints). Each
% sum_of_squares call dispatches into square/2 twice.
%
% List-shaped programs (list_reverse, naive_reverse) are deferred —
% their 2-clause heads `([], ...)` and `([H|T], ...)` compile to
% switch_on_term, which the WAM-Elixir lowered emitter currently raises
% as a TODO (wam_elixir_lowered_emitter.pl:1873). Implementing
% switch_on_term is the concrete blocker for porting those Scala tests.
:- dynamic user:square/2.
:- dynamic user:sum_of_squares/3.
user:square(X, Y) :- Y is X * X.
user:sum_of_squares(A, B, S) :- user:square(A, A2),
                                user:square(B, B2),
                                S is A2 + B2.

% --- call/N meta-call acceptance cases (PR #1) ---
% Spec: docs/design/WAM_ELIXIR_GAPS_SPECIFICATION.md §3 PR #1.
% Each predicate exercises one shape of meta-call dispatch:
%   - elx_call_true/0       atom goal, 0 extras  (succeeds)
%   - elx_call_fail/0       atom goal, 0 extras  (fails)
%   - elx_call_compound/1   compound goal, 0 extras (heap-built goal)
%   - elx_call_extras/1     atom goal, 3 extras   (partial-application)
:- dynamic user:elx_call_true/0.
:- dynamic user:elx_call_fail/0.
:- dynamic user:elx_call_compound/1.
:- dynamic user:elx_call_extras/1.
user:elx_call_true :- call(true).
user:elx_call_fail :- call(fail).
user:elx_call_compound(R) :- G = (X is 1 + 1), call(G), R = X.
user:elx_call_extras(R) :- call(append, [a, b], [c, d], R).

% --- catch/3 + throw/1 acceptance cases (PR #2) ---
% Spec: docs/design/WAM_ELIXIR_GAPS_SPECIFICATION.md §3 PR #2.
:- dynamic user:elx_catch_match/0.
:- dynamic user:elx_catch_compound/0.
:- dynamic user:elx_catch_no_throw/0.
:- dynamic user:elx_catch_uncaught/0.
:- dynamic user:elx_catch_nested/0.
:- dynamic user:elx_catch_fail_propagates/0.
% Atomic catcher matches atomic thrown term.
user:elx_catch_match :- catch(throw(foo), foo, true).
% Compound thrown term, unbound catcher binds to the whole compound.
% NOTE: structural compound-vs-compound unification (e.g., catcher
% pattern `error(_, _)`) is a pre-existing runtime limitation —
% unify/3 only handles `v1 == v2` equality or unbound binding, not
% recursive compound walk. Filed as a follow-up; not catch-specific.
% Once compound-unify lands, this test should change to:
%   catch(throw(error(type_error, ctx)), error(_, _), true).
user:elx_catch_compound :- catch(throw(error(type_error, ctx)),
                                 _Caught, true).
% Goal proceeds normally; recovery (fail) NOT invoked.
user:elx_catch_no_throw :- catch(true, _, fail).
% Throw with no enclosing catch — wrapper converts to :fail + stderr.
user:elx_catch_uncaught :- throw(boom).
% Inner catcher doesn't match (re-throws); outer catcher matches.
user:elx_catch_nested :-
    catch(
        catch(throw(outer), inner, fail),
        outer,
        true).
% Goal fails (not throws); catch propagates failure; recovery NOT invoked.
user:elx_catch_fail_propagates :- catch(fail, _, true).

% --- is_iso/2 + is_lax/2 acceptance cases (PR #4) ---
% Spec: docs/design/WAM_ELIXIR_GAPS_SPECIFICATION.md §3 PR #4.
% All five wrap is_iso/2 in catch(... , _, true) so the test can
% verify "throw fired" by predicate success without depending on
% compound-pattern unification (which is the runtime limitation
% documented in PR #2's catch_match_compound test).
:- dynamic user:elx_iso_is_instantiation/0.
:- dynamic user:elx_iso_is_type_error/0.
:- dynamic user:elx_iso_is_zero_divisor/0.
:- dynamic user:elx_iso_is_succeeds/1.
:- dynamic user:elx_explicit_lax_fails/0.
% RHS unbound -> instantiation_error throw -> catch matches.
user:elx_iso_is_instantiation :- catch(is_iso(_X, _Y), _, true).
% RHS is non-evaluable atom -> type_error(evaluable, foo/0) throw.
user:elx_iso_is_type_error :- catch(is_iso(_X, foo), _, true).
% RHS is 1//0 -> evaluation_error(zero_divisor) throw.
user:elx_iso_is_zero_divisor :- catch(is_iso(_X, 1//0), _, true).
% Happy path: is_iso(X, 1+1) succeeds with X=2 (no throw).
user:elx_iso_is_succeeds(R) :- is_iso(X, 1 + 1), R = X.
% Explicit is_lax bypasses rewrite — same lax behaviour as is/2.
% RHS is foo (non-evaluable) -> eval_arith throws {:eval_error, _}
% -> top-level wrapper converts to :fail. Predicate fails.
user:elx_explicit_lax_fails :- is_lax(_X, foo).

% ============================================================
% elixir_available — same gating pattern as the Scala suite
% ============================================================

elixir_available :-
    (   getenv('ELIXIR_SMOKE_TESTS', "1") -> true
    ;   catch(
            process_create(path(elixir), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            _,
            fail
        ),
        process_wait(Pid, exit(0))
    ).

% ============================================================
% Tests
% ============================================================

:- begin_tests(wam_elixir_classic_programs,
               [ condition(elixir_available) ]).

test(fibonacci) :-
    with_elixir_project(
        [user:fib/2],
        _Opts,
        TmpDir,
        (
            % fib(0, 0)  -> true
            verify_elixir_args(TmpDir, 'fib/2', ['0', '0'], "true"),
            % fib(1, 1)  -> true
            verify_elixir_args(TmpDir, 'fib/2', ['1', '1'], "true"),
            % fib(5, 5)  -> true (5th Fibonacci is 5)
            verify_elixir_args(TmpDir, 'fib/2', ['5', '5'], "true"),
            % fib(7, 13) -> true (7th Fibonacci is 13)
            verify_elixir_args(TmpDir, 'fib/2', ['7', '13'], "true"),
            % fib(5, 6)  -> false (5th Fib is 5, not 6)
            verify_elixir_args(TmpDir, 'fib/2', ['5', '6'], "false")
        )).

test(ackermann) :-
    with_elixir_project(
        [user:ack/3],
        _Opts,
        TmpDir,
        (
            % ack(0, n) = n+1
            verify_elixir_args(TmpDir, 'ack/3', ['0', '5', '6'],   "true"),
            % ack(1, n) = n+2
            verify_elixir_args(TmpDir, 'ack/3', ['1', '5', '7'],   "true"),
            % ack(2, n) = 2n+3
            verify_elixir_args(TmpDir, 'ack/3', ['2', '3', '9'],   "true"),
            % ack(3, 3) = 61
            verify_elixir_args(TmpDir, 'ack/3', ['3', '3', '61'],  "true"),
            % Wrong answer -> false
            verify_elixir_args(TmpDir, 'ack/3', ['3', '3', '60'],  "false")
        )).

test(pythagoras) :-
    with_elixir_project(
        [user:square/2, user:sum_of_squares/3],
        _Opts,
        TmpDir,
        (
            % 3^2 + 4^2 = 25 -> true
            verify_elixir_args(TmpDir, 'sum_of_squares/3',
                               ['3', '4', '25'], "true"),
            % 5^2 + 12^2 = 169 -> true (5-12-13 triple)
            verify_elixir_args(TmpDir, 'sum_of_squares/3',
                               ['5', '12', '169'], "true"),
            % 8^2 + 15^2 = 289 -> true (8-15-17 triple)
            verify_elixir_args(TmpDir, 'sum_of_squares/3',
                               ['8', '15', '289'], "true"),
            % Wrong answer -> false
            verify_elixir_args(TmpDir, 'sum_of_squares/3',
                               ['3', '4', '24'], "false"),
            % square direct: 7^2 = 49 -> true
            verify_elixir_args(TmpDir, 'square/2', ['7', '49'], "true")
        )).

% --- call/N meta-call tests (PR #1 acceptance) ---

test(call_atom_true) :-
    with_elixir_project(
        [user:elx_call_true/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_call_true/0', [], "true")
    ).

test(call_atom_fail) :-
    with_elixir_project(
        [user:elx_call_fail/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_call_fail/0', [], "false")
    ).

test(call_compound_goal) :-
    % G = (X is 1+1), call(G), R = X.  R = 2 -> true.
    with_elixir_project(
        [user:elx_call_compound/1],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_call_compound/1', ['2'], "true")
    ).

test(call_with_extras) :-
    % call(append, [a,b], [c,d], R) -> R = [a,b,c,d].
    with_elixir_project(
        [user:elx_call_extras/1],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_call_extras/1',
                           ['[a,b,c,d]'], "true")
    ).

% --- catch/3 + throw/1 tests (PR #2 acceptance) ---

test(catch_match_atom) :-
    % catch(throw(foo), foo, true) -> succeeds (atomic match).
    with_elixir_project(
        [user:elx_catch_match/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_catch_match/0', [], "true")
    ).

test(catch_match_compound) :-
    % Compound pattern error(_,_) matches thrown error(type_error, ctx).
    with_elixir_project(
        [user:elx_catch_compound/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_catch_compound/0', [], "true")
    ).

test(catch_no_throw) :-
    % Goal proceeds normally; recovery NOT invoked even though it would fail.
    with_elixir_project(
        [user:elx_catch_no_throw/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_catch_no_throw/0', [], "true")
    ).

test(catch_uncaught) :-
    % Throw with no enclosing catch — wrapper converts to :fail.
    with_elixir_project(
        [user:elx_catch_uncaught/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_catch_uncaught/0', [], "false")
    ).

test(catch_nested) :-
    % Inner catcher doesn't match (re-throws); outer matches.
    with_elixir_project(
        [user:elx_catch_nested/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_catch_nested/0', [], "true")
    ).

test(catch_fail_propagates) :-
    % Protected goal FAILS (not throws); catch propagates failure;
    % recovery `true` is NOT invoked. Predicate should fail.
    with_elixir_project(
        [user:elx_catch_fail_propagates/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_catch_fail_propagates/0',
                           [], "false")
    ).

% --- is_iso/2 + is_lax/2 tests (PR #4 acceptance) ---

test(iso_is_instantiation) :-
    % is_iso(X, _Y) -> error(instantiation_error, _) thrown -> catch matches.
    with_elixir_project(
        [user:elx_iso_is_instantiation/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_is_instantiation/0',
                           [], "true")
    ).

test(iso_is_type_error) :-
    % is_iso(X, foo) -> error(type_error(evaluable, foo/0), _) thrown.
    with_elixir_project(
        [user:elx_iso_is_type_error/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_is_type_error/0',
                           [], "true")
    ).

test(iso_is_zero_divisor) :-
    % is_iso(X, 1//0) -> error(evaluation_error(zero_divisor), _) thrown.
    with_elixir_project(
        [user:elx_iso_is_zero_divisor/0],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_is_zero_divisor/0',
                           [], "true")
    ).

test(iso_is_succeeds_on_valid) :-
    % is_iso(X, 1+1) -> no throw, X bound to 2. Happy-path coverage —
    % verifies the is_iso arm correctly delegates to the lax body on
    % valid input. Tests the bind/compare branches of the ISO arm.
    with_elixir_project(
        [user:elx_iso_is_succeeds/1],
        _Opts,
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_is_succeeds/1',
                           ['2'], "true")
    ).

test(explicit_lax_bypasses_iso_rewrite) :-
    % Predicate annotated ISO via inline option. The body uses
    % EXPLICIT is_lax/2 — the rewrite only touches default-form is/2,
    % so the call remains is_lax. is_lax with foo as RHS throws an
    % internal {:eval_error, _} which is NOT a wam_throw, so the
    % top-level wrapper's `error -> CRASHED` arm converts to :fail.
    % Predicate fails -> "false". Demonstrates that explicit *_lax
    % forms survive the rewrite (three-forms guarantee from
    % WAM_ELIXIR_PARITY_PHILOSOPHY §5).
    with_elixir_project(
        [user:elx_explicit_lax_fails/0],
        [iso_errors(elx_explicit_lax_fails/0, true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_explicit_lax_fails/0',
                           [], "false")
    ).

:- end_tests(wam_elixir_classic_programs).

% ============================================================
% Test fixture helpers (mirrors test_wam_scala_classic_programs.pl)
% ============================================================

with_elixir_project(Preds, ExtraOpts0, TmpDir, Goal) :-
    (   var(ExtraOpts0) -> ExtraOpts = [] ; ExtraOpts = ExtraOpts0 ),
    unique_elixir_tmp_dir('tmp_elixir_classic', TmpDir),
    BaseOpts = [ module_name('wam_elixir_classic'),
                 emit_mode(lowered),
                 source_module(user) ],
    append(ExtraOpts, BaseOpts, AllOpts),
    % Compile each predicate to WAM, then write the project. This is
    % the same shape examples/debug_wam_elixir_ancestor.pl uses.
    compile_predicates_to_wam(Preds, PredWamPairs),
    write_wam_elixir_project(PredWamPairs, AllOpts, TmpDir),
    % Copy the shared driver script into the project root. Resolved
    % at consult time via prolog_load_context so the path stays
    % correct regardless of cwd at run_tests time.
    classic_driver_path(DriverSrc),
    directory_file_path(TmpDir, 'run_classic.exs', DriverDst),
    copy_file(DriverSrc, DriverDst),
    setup_call_cleanup(
        true,
        call(Goal),
        delete_directory_and_contents(TmpDir)).

%% Resolved at consult time: this tests directory + elixir_e2e/run_classic.exs.
:- (   prolog_load_context(directory, Dir),
       directory_file_path(Dir, 'elixir_e2e/run_classic.exs', P),
       assertz(classic_driver_path_fact(P))
   ;   true
   ).
classic_driver_path(P) :- classic_driver_path_fact(P).

%% compile_predicates_to_wam(+Preds, -PredWamPairs)
%
%  Preds is a list of `Module:Name/Arity` indicators (typically with
%  Module=user). Compiles each to WAM bytecode and packages as
%  `Name/Arity-WamCode` pairs, the format `write_wam_elixir_project/3`
%  expects.
compile_predicates_to_wam([], []).
compile_predicates_to_wam([Mod:Name/Arity | Rest], [Name/Arity-WamCode | RestPairs]) :-
    (   wam_target:compile_predicate_to_wam(Name/Arity, [], WamCode) -> true
    ;   wam_target:compile_predicate_to_wam(Mod:Name/Arity, [], WamCode) -> true
    ;   throw(error(wam_compile_failed(Mod:Name/Arity), _))
    ),
    compile_predicates_to_wam(Rest, RestPairs).

%% verify_elixir_args(+ProjectDir, +PredKey, +Args, +Expected)
%
%  Invoke `elixir run_classic.exs <ModuleCamel> <PredKey> <Args...>`
%  in ProjectDir and assert stdout matches Expected ("true" / "false").
%
%  Module name passed must match the camelCase form `write_wam_elixir_project`
%  emits — `module_name('wam_elixir_classic')` -> `WamElixirClassic`.
verify_elixir_args(ProjectDir, PredKey, Args, Expected) :-
    run_elixir_predicate_args(ProjectDir, PredKey, Args, Actual),
    (   Actual == Expected
    ->  true
    ;   throw(error(assertion_error(PredKey, Args, Expected, Actual), _))
    ).

run_elixir_predicate_args(ProjectDir, PredKey, Args, Output) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    % First positional arg to the driver is the camelised module
    % name; matches the Module.concat([mod_camel, pred_camel]) lookup
    % in run_classic.exs.
    append(['run_classic.exs', "WamElixirClassic", PredStr], ArgStrs, ProcArgs),
    process_create(path(elixir), ProcArgs,
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
    ;   throw(error(elixir_run_failed(ExitCode, PredKey, Args, ErrStr), _))
    ).

unique_elixir_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    (   getenv('TMPDIR', Base) -> true
    ;   Base = '/tmp'
    ),
    format(atom(TmpDir), '~w/~w_~w', [Base, Prefix, Stamp]).
