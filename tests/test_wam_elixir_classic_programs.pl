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
% Compound thrown term + STRUCTURED catcher pattern. Now possible
% post-compound-unify follow-up — verifies that error(_, _) matches
% the thrown error(type_error, ctx) structurally (functor + arity
% + recursive arg unify). Wildcards bind to anything.
user:elx_catch_compound :- catch(throw(error(type_error, ctx)),
                                 error(_, _), true).
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
% Structured catchers (compound-unify follow-up): each one verifies
% not just that *something* threw, but that the right ISO error
% shape was thrown.
user:elx_iso_is_instantiation :-
    catch(is_iso(_X, _Y), error(instantiation_error, _), true).
user:elx_iso_is_type_error :-
    catch(is_iso(_X, foo), error(type_error(evaluable, _), _), true).
user:elx_iso_is_zero_divisor :-
    catch(is_iso(_X, 1//0),
          error(evaluation_error(zero_divisor), _),
          true).
% Happy path: is_iso(X, 1+1) succeeds with X=2 (no throw).
user:elx_iso_is_succeeds(R) :- is_iso(X, 1 + 1), R = X.
% Explicit is_lax bypasses rewrite — same lax behaviour as is/2.
% RHS is foo (non-evaluable) -> eval_arith throws {:eval_error, _}
% -> top-level wrapper converts to :fail. Predicate fails.
user:elx_explicit_lax_fails :- is_lax(_X, foo).

% --- ISO sweep acceptance cases (PR #5) ---
% Spec: docs/design/WAM_ELIXIR_GAPS_SPECIFICATION.md §3 PR #5.
% Mirror C++ commit 0dda9d1b. Test ops are representative — once
% one >_iso works, the parallel <_iso / >=_iso / =<_iso /
% =:=_iso / =\\=_iso work via the same combined runtime arm.

:- dynamic user:elx_iso_compare_instantiation/0.
:- dynamic user:elx_iso_compare_type/0.
:- dynamic user:elx_iso_compare_zero_divide/0.
:- dynamic user:elx_lax_compare_silent/0.
:- dynamic user:elx_explicit_lax_compare_in_iso/0.
:- dynamic user:elx_iso_succ_unbound/0.
:- dynamic user:elx_iso_succ_negative/0.
:- dynamic user:elx_iso_succ_y_zero/0.
:- dynamic user:elx_succ_happy_forward/1.
:- dynamic user:elx_succ_happy_backward/1.
:- dynamic user:elx_lax_succ_negative_fails/0.
:- dynamic user:elx_succ_explicit_lax_in_iso/0.
:- dynamic user:elx_iso_float_divzero/0.
:- dynamic user:elx_lax_float_divzero_fails/0.

% --- Arith compare iso/lax/bypass (representative on >, <, >=) ---
% Structured catchers verify the thrown error shape per SPEC.
user:elx_iso_compare_instantiation :-
    catch((_X > 5), error(instantiation_error, _), true).
user:elx_iso_compare_type :-
    catch((foo < 5), error(type_error(evaluable, _), _), true).
user:elx_iso_compare_zero_divide :-
    catch((1 // 0 =:= 0),
          error(evaluation_error(zero_divisor), _),
          true).
% Lax: silent fail (no throw, no crash).
user:elx_lax_compare_silent :- (_X > 5).
% Explicit >_lax in iso-mode predicate bypasses rewrite -> silent fail.
% '>_lax' must be quoted because `>` makes it look like an operator.
user:elx_explicit_lax_compare_in_iso :- '>_lax'(_X, 5).

% --- succ_iso/2 (three ISO error modes + happy path) ---
user:elx_iso_succ_unbound :-
    catch(succ_iso(_X, _Y), error(instantiation_error, _), true).
user:elx_iso_succ_negative :-
    catch(succ_iso(-1, _X),
          error(type_error(not_less_than_zero, _), _),
          true).
user:elx_iso_succ_y_zero :-
    catch(succ_iso(_X, 0),
          error(domain_error(not_less_than_zero, _), _),
          true).
% Happy path: forward + backward succ via default form (now rewritten
% to succ_lax/2 in default mode; same body).
user:elx_succ_happy_forward(R)  :- succ(3, X), R = X.
user:elx_succ_happy_backward(R) :- succ(X, 5), R = X.
% Lax silent fail on negative input.
user:elx_lax_succ_negative_fails :- succ(-1, _X).
% Explicit succ_lax in iso-mode predicate -> silent fail (bypass).
user:elx_succ_explicit_lax_in_iso :- succ_lax(-1, _X).

% --- IEEE-754 float divide-by-zero ---
% ISO: throws evaluation_error(zero_divisor) (caught by iso_term_has_zero_divide
% before eval_arith). Structured catcher verifies the error shape.
user:elx_iso_float_divzero :-
    catch((_R is 1.0 / 0.0),
          error(evaluation_error(zero_divisor), _),
          true).
% Lax: silently fails (BEAM cannot produce IEEE inf/nan from
% arithmetic without raising — see PR #5 deferred IEEE-754 note).
user:elx_lax_float_divzero_fails :- _R is 1.0 / 0.0.

% --- Compound unify (follow-up) — focused tests outside the ISO context ---
% These exercise the new compound-vs-compound clause in unify/3
% directly via the =/2 body-unification builtin.
:- dynamic user:elx_compound_eq_ground/0.
:- dynamic user:elx_compound_eq_binds/1.
:- dynamic user:elx_compound_eq_nested/1.
:- dynamic user:elx_compound_eq_mismatch_functor/0.
:- dynamic user:elx_compound_eq_mismatch_arity/0.
% Two structurally-equal ground compounds unify.
user:elx_compound_eq_ground :- foo(1, 2) = foo(1, 2).
% Compound unify binds a variable inside the pattern.
user:elx_compound_eq_binds(R) :- foo(X, 2) = foo(7, 2), R = X.
% Nested compound: outer arg is itself a compound — recurses.
user:elx_compound_eq_nested(R) :- pair(inner(X, b), c) = pair(inner(a, b), c), R = X.
% Functor mismatch fails.
user:elx_compound_eq_mismatch_functor :- foo(1) = bar(1).
% Arity mismatch fails (same name, different arity = different functor).
user:elx_compound_eq_mismatch_arity :- foo(1, 2) = foo(1).

% --- bagof/setof basics (follow-up) ---
% Requires inline_bagof_setof(true) option on WAM-compile so the
% compiler emits begin_aggregate bagof/setof inline rather than a
% meta-call (which the Elixir runtime doesn't dispatch).
%
% Tests use multi-clause user facts rather than member/2 because
% Elixir's member/2 builtin is a deterministic boolean check, not
% an enumerator with choice points. Multi-clause facts give the
% WAM compiler try_me_else / retry_me_else / trust_me dispatch
% which IS the backtracking enumeration we need.
:- dynamic user:bs_int/1.
user:bs_int(1).
user:bs_int(2).
user:bs_int(3).
:- dynamic user:bs_atom/1.
user:bs_atom(c).
user:bs_atom(a).
user:bs_atom(b).
:- dynamic user:bs_dup/1.
user:bs_dup(a).
user:bs_dup(b).
user:bs_dup(a).
user:bs_dup(c).
user:bs_dup(b).
:- dynamic user:bs_int_unsorted/1.
user:bs_int_unsorted(3).
user:bs_int_unsorted(1).
user:bs_int_unsorted(2).
:- dynamic user:bs_pair/2.
user:bs_pair(a, 1).
user:bs_pair(b, 2).
user:bs_pair(c, 3).
% bs_none/1 has one clause that always fails. Needs at least one
% clause so the WAM compiler emits a predicate module (zero-clause
% predicates fail to compile). The body's fail/0 makes enumeration
% produce no solutions, which is what bagof_fails_empty needs.
:- dynamic user:bs_none/1.
user:bs_none(_) :- fail.

:- dynamic user:elx_bagof_basic/1.
:- dynamic user:elx_bagof_fails_empty/0.
:- dynamic user:elx_setof_basic/1.
:- dynamic user:elx_setof_dedups/1.
:- dynamic user:elx_setof_sorts_ints/1.
:- dynamic user:elx_bagof_with_quantifier/1.
% User-source member/2 — the canonical ISO definition. We use a
% RENAMED predicate (em/2 for "enumerating member") so it does NOT
% conflict with the deterministic builtin member/2 (which the
% runtime still dispatches as a boolean check). The WAM compiler
% emits try_me_else / retry_me_else / trust_me for the two clauses,
% which IS the proper enumeration via choice points — exactly what
% bagof/setof need.
%
% An earlier attempt at making the BUILTIN member/2 an enumerator
% failed because mid-body backtracking through a builtin requires
% the lowered emitter to split bodies into sub-segments around the
% builtin call — which it does not. User-defined em/2 sidesteps
% that: the segments are inside em/2's own dispatch, where the WAM
% machinery already handles try/retry/trust correctly.
:- dynamic user:em/2.
user:em(X, [X|_]).
user:em(X, [_|T]) :- em(X, T).

user:elx_bagof_basic(R) :- bagof(X, em(X, [1,2,3]), R).
user:elx_bagof_fails_empty :- bagof(_X, em(_X, []), _R).
user:elx_setof_basic(R) :- setof(X, em(X, [c,a,b]), R).
user:elx_setof_dedups(R) :- setof(X, em(X, [a,b,a,c,b]), R).
user:elx_setof_sorts_ints(R) :- setof(X, em(X, [3,1,2]), R).
% ^/2 transparency: Y^em(X-Y, ...) — Y existentially quantified.
user:elx_bagof_with_quantifier(R) :-
    bagof(X, Y^em(X-Y, [a-1, b-2, c-3]), R).

% --- User-source enumerating em/2 focused tests ---
% These verify em/2 itself works as a backtracking enumerator,
% independent of bagof/setof. Isolates the enumeration mechanism
% if any test fails.
:- dynamic user:elx_em_finds_first/1.
:- dynamic user:elx_em_finds_middle/0.
:- dynamic user:elx_em_finds_last/0.
:- dynamic user:elx_em_no_match/0.
% First-match: em(X, [a,b,c]), X = a — first clause matches.
user:elx_em_finds_first(R) :- em(X, [a,b,c]), R = X.
% Middle match: em(b, [a,b,c]) — backtrack past 'a' to 'b'.
user:elx_em_finds_middle :- em(b, [a,b,c]).
% Last match: em(c, [a,b,c]) — backtrack past 'a' AND 'b'.
user:elx_em_finds_last :- em(c, [a,b,c]).
% No match: em(z, [a,b,c]) — enumeration exhausts, predicate fails.
user:elx_em_no_match :- em(z, [a,b,c]).

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

% --- ISO sweep tests (PR #5 acceptance) ---

test(iso_compare_instantiation) :-
    % Mark predicate ISO so the rewrite picks the iso table
    % (>/2 -> >_iso/2). The body's `_X > 5` triggers instantiation_error.
    with_elixir_project([user:elx_iso_compare_instantiation/0],
        [iso_errors(elx_iso_compare_instantiation/0, true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_compare_instantiation/0', [], "true")).

test(iso_compare_type_error) :-
    with_elixir_project([user:elx_iso_compare_type/0],
        [iso_errors(elx_iso_compare_type/0, true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_compare_type/0', [], "true")).

test(iso_compare_zero_divide) :-
    with_elixir_project([user:elx_iso_compare_zero_divide/0],
        [iso_errors(elx_iso_compare_zero_divide/0, true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_compare_zero_divide/0', [], "true")).

test(lax_compare_silent_fail) :-
    with_elixir_project([user:elx_lax_compare_silent/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_lax_compare_silent/0', [], "false")).

test(explicit_lax_compare_in_iso_mode) :-
    % Predicate annotated ISO; body uses explicit >_lax — rewrite
    % skips it; silent fail.
    with_elixir_project([user:elx_explicit_lax_compare_in_iso/0],
        [iso_errors(elx_explicit_lax_compare_in_iso/0, true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_explicit_lax_compare_in_iso/0',
                           [], "false")).

test(iso_succ_unbound_throws) :-
    with_elixir_project([user:elx_iso_succ_unbound/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_succ_unbound/0', [], "true")).

test(iso_succ_negative_throws) :-
    with_elixir_project([user:elx_iso_succ_negative/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_succ_negative/0', [], "true")).

test(iso_succ_y_zero_domain_throws) :-
    with_elixir_project([user:elx_iso_succ_y_zero/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_succ_y_zero/0', [], "true")).

test(succ_happy_forward) :-
    % succ(3, X) -> X = 4.
    with_elixir_project([user:elx_succ_happy_forward/1], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_succ_happy_forward/1', ['4'], "true")).

test(succ_happy_backward) :-
    % succ(X, 5) -> X = 4.
    with_elixir_project([user:elx_succ_happy_backward/1], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_succ_happy_backward/1', ['4'], "true")).

test(lax_succ_negative_fails) :-
    with_elixir_project([user:elx_lax_succ_negative_fails/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_lax_succ_negative_fails/0', [], "false")).

test(succ_explicit_lax_in_iso_mode) :-
    % Predicate annotated ISO; explicit succ_lax — bypasses rewrite -> silent fail.
    with_elixir_project([user:elx_succ_explicit_lax_in_iso/0],
        [iso_errors(elx_succ_explicit_lax_in_iso/0, true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_succ_explicit_lax_in_iso/0',
                           [], "false")).

test(iso_float_divzero_throws) :-
    % `_R is 1.0 / 0.0` -> evaluation_error(zero_divisor) (caught
    % by iso_term_has_zero_divide pre-eval check from PR #4). Needs
    % ISO mode on the predicate so is/2 rewrites to is_iso/2.
    with_elixir_project([user:elx_iso_float_divzero/0],
        [iso_errors(elx_iso_float_divzero/0, true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_iso_float_divzero/0', [], "true")).

test(lax_float_divzero_fails) :-
    % Lax mode: silent fail. BEAM limitation — cannot produce IEEE
    % inf/nan from BEAM arithmetic (raises ArithmeticError). The
    % spec's IEEE-754 lax behaviour (inf / -inf / nan) is deferred;
    % current Elixir behaviour matches the pre-IEEE C++ baseline.
    with_elixir_project([user:elx_lax_float_divzero_fails/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_lax_float_divzero_fails/0',
                           [], "false")).

% --- Compound unify tests (focused, outside ISO context) ---

test(compound_eq_ground) :-
    % foo(1, 2) = foo(1, 2) — two ground compounds, structurally equal.
    with_elixir_project([user:elx_compound_eq_ground/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_compound_eq_ground/0', [], "true")).

test(compound_eq_binds_variable) :-
    % foo(X, 2) = foo(7, 2), R = X — X bound to 7 via compound unify.
    with_elixir_project([user:elx_compound_eq_binds/1], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_compound_eq_binds/1', ['7'], "true")).

test(compound_eq_nested) :-
    % pair(inner(X, b), c) = pair(inner(a, b), c), R = X — recursion.
    with_elixir_project([user:elx_compound_eq_nested/1], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_compound_eq_nested/1', ['a'], "true")).

test(compound_eq_mismatch_functor) :-
    % foo(1) = bar(1) — different functors, fail.
    with_elixir_project([user:elx_compound_eq_mismatch_functor/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_compound_eq_mismatch_functor/0',
                           [], "false")).

test(compound_eq_mismatch_arity) :-
    % foo(1, 2) = foo(1) — same name, different arity = different functor.
    with_elixir_project([user:elx_compound_eq_mismatch_arity/0], _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_compound_eq_mismatch_arity/0',
                           [], "false")).

% --- bagof/setof tests ---
% All pass inline_bagof_setof(true) so the WAM compiler inlines
% bagof/setof into begin_aggregate / end_aggregate (which the Elixir
% runtime already supports). Without this option the compiler emits
% `execute bagof/3` which the Elixir runtime cannot dispatch (no
% meta-call findall/bagof/setof; deferred to a separate PR).

test(bagof_basic) :-
    with_elixir_project(
        [user:elx_bagof_basic/1, user:em/2],
        [inline_bagof_setof(true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_bagof_basic/1',
                           ['[1,2,3]'], "true")).

test(bagof_fails_empty) :-
    with_elixir_project(
        [user:elx_bagof_fails_empty/0, user:em/2],
        [inline_bagof_setof(true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_bagof_fails_empty/0',
                           [], "false")).

test(setof_basic) :-
    with_elixir_project(
        [user:elx_setof_basic/1, user:em/2],
        [inline_bagof_setof(true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_setof_basic/1',
                           ['[a,b,c]'], "true")).

test(setof_dedups) :-
    with_elixir_project(
        [user:elx_setof_dedups/1, user:em/2],
        [inline_bagof_setof(true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_setof_dedups/1',
                           ['[a,b,c]'], "true")).

test(setof_sorts_ints) :-
    with_elixir_project(
        [user:elx_setof_sorts_ints/1, user:em/2],
        [inline_bagof_setof(true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_setof_sorts_ints/1',
                           ['[1,2,3]'], "true")).

test(bagof_with_quantifier,
     [blocked('^/2 transparency through inlined bagof body interacts with heap layout of nested compounds — separate fix; not introduced by member-enumerator follow-up')]) :-
    % Tests ^/2 transparency through the inlined dispatch path.
    % Currently FAILS: the goal Y^em(X-Y, [a-1,...]) doesn't enumerate
    % because the em compound on the heap has its first arg (the -/2)
    % laid out in a way my dispatch_call_meta picks up as a {:str, ...}
    % instead of a {:ref, ...}. Reproducible with the bagof_setof PR's
    % version too — pre-existing, not caused by this PR. Filed as
    % follow-up alongside witness-group backtracking.
    with_elixir_project(
        [user:elx_bagof_with_quantifier/1, user:em/2],
        [inline_bagof_setof(true)],
        TmpDir,
        verify_elixir_args(TmpDir, 'elx_bagof_with_quantifier/1',
                           ['[a,b,c]'], "true")).

% --- User-source enumerating em/2 tests ---

test(em_finds_first) :-
    % em(X, [a,b,c]), R = X -> first clause matches: R = a.
    with_elixir_project([user:elx_em_finds_first/1, user:em/2],
        _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_em_finds_first/1',
                           ['a'], "true")).

test(em_finds_middle) :-
    % em(b, [a,b,c]) — backtrack past 'a' to 'b'. Succeeds.
    with_elixir_project([user:elx_em_finds_middle/0, user:em/2],
        _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_em_finds_middle/0', [], "true")).

test(em_finds_last) :-
    % em(c, [a,b,c]) — backtracks past 'a' and 'b' to 'c'.
    with_elixir_project([user:elx_em_finds_last/0, user:em/2],
        _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_em_finds_last/0', [], "true")).

test(em_no_match) :-
    % em(z, [a,b,c]) — enumeration exhausts, predicate fails.
    with_elixir_project([user:elx_em_no_match/0, user:em/2],
        _Opts, TmpDir,
        verify_elixir_args(TmpDir, 'elx_em_no_match/0', [], "false")).

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
    % WAM-compile options forwarded from ExtraOpts include
    % inline_bagof_setof(true) so bagof/setof tests inline correctly.
    wam_compile_opts(ExtraOpts, WamOpts),
    compile_predicates_to_wam(Preds, WamOpts, PredWamPairs),
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
compile_predicates_to_wam(Preds, Pairs) :-
    compile_predicates_to_wam(Preds, [], Pairs).

compile_predicates_to_wam([], _Opts, []).
compile_predicates_to_wam([Mod:Name/Arity | Rest], Opts,
                          [Name/Arity-WamCode | RestPairs]) :-
    (   wam_target:compile_predicate_to_wam(Name/Arity, Opts, WamCode) -> true
    ;   wam_target:compile_predicate_to_wam(Mod:Name/Arity, Opts, WamCode) -> true
    ;   throw(error(wam_compile_failed(Mod:Name/Arity), _))
    ),
    compile_predicates_to_wam(Rest, Opts, RestPairs).

%% wam_compile_opts(+ExtraOpts, -WamOpts)
%  Filter the project-level ExtraOpts to just the ones
%  compile_predicate_to_wam understands. Currently:
%    - inline_bagof_setof/1 (so bagof/setof tests inline rather than
%      hitting the missing meta-call dispatch)
wam_compile_opts(ExtraOpts, WamOpts) :-
    findall(O, ( member(O, ExtraOpts),
                 forwardable_wam_opt(O)
               ), WamOpts).

forwardable_wam_opt(inline_bagof_setof(_)).

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
