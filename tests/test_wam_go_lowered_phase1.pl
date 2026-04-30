:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Phase 1 tests for the WAM-lowered Go path.
%
% Phase 1 covers the low-level helpers exposed by wam_go_target.pl that
% enable the lowered emitter to produce correct, deduplicated Go code:
%
%   - intern_atom_go/2 / init_atom_intern_table_go/0 / emit_atom_table_go/1
%     Compile-time atom deduplication. Repeated atom literals in the
%     lowered output share a single package-level *Atom value.
%
%   - resolve_dimension_n_go/2
%     Codegen-time resolution of user:dimension_n/1 (mirrors the
%     equivalent Haskell helper added in commit c8c47d5 to fix the
%     hardcoded 5 in the effective-distance template).
%
% These helpers each exist in isolation; they don't require a Go
% toolchain or a running WAM. The real emitter behavior is exercised in
% Phase 3.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl

:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic test_failed/0.
:- dynamic user:dimension_n/1.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% ---------------------------------------------------------------------
%% Atom interning
%% ---------------------------------------------------------------------

test_intern_returns_stable_var_name :-
    Test = 'WAM-Go-Lowered Phase 1: intern_atom_go returns the same Go var name on repeat',
    init_atom_intern_table_go,
    intern_atom_go("foo", Var1),
    intern_atom_go("foo", Var2),
    (   Var1 == Var2
    ->  pass(Test)
    ;   fail_test(Test, ['expected stable name; got ', Var1, ' and ', Var2])
    ).

test_intern_assigns_distinct_names :-
    Test = 'WAM-Go-Lowered Phase 1: distinct atoms get distinct Go var names',
    init_atom_intern_table_go,
    intern_atom_go("foo", VarFoo),
    intern_atom_go("bar", VarBar),
    (   VarFoo \== VarBar
    ->  pass(Test)
    ;   fail_test(Test, ['expected distinct names; both got ', VarFoo])
    ).

test_intern_var_name_shape :-
    Test = 'WAM-Go-Lowered Phase 1: interned var follows wamAtom_<sanitized>_<seq> shape',
    init_atom_intern_table_go,
    intern_atom_go("hello", Var),
    atom_string(Var, VarS),
    (   sub_string(VarS, 0, 8, _, "wamAtom_")
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected name shape: ', Var])
    ).

test_intern_sanitizes_special_chars :-
    Test = 'WAM-Go-Lowered Phase 1: special chars in atoms map to underscores in var name',
    init_atom_intern_table_go,
    intern_atom_go("[]", Var),
    atom_string(Var, VarS),
    % Var name must be a valid Go identifier — only [A-Za-z0-9_].
    string_codes(VarS, Codes),
    (   forall(member(C, Codes),
               (   (C >= 0'a, C =< 0'z) ; (C >= 0'A, C =< 0'Z)
               ;   (C >= 0'0, C =< 0'9) ; C =:= 0'_
               ))
    ->  pass(Test)
    ;   fail_test(Test, ['var contains non-identifier chars: ', Var])
    ).

test_init_resets_table :-
    Test = 'WAM-Go-Lowered Phase 1: init_atom_intern_table_go resets state',
    init_atom_intern_table_go,
    intern_atom_go("first", _),
    init_atom_intern_table_go,
    emit_atom_table_go(Code),
    (   Code == ""
    ->  pass(Test)
    ;   fail_test(Test, ['expected empty table after reset; got: ', Code])
    ).

test_emit_table_lists_all_atoms :-
    Test = 'WAM-Go-Lowered Phase 1: emit_atom_table_go declares each interned atom',
    init_atom_intern_table_go,
    intern_atom_go("alpha", VarA),
    intern_atom_go("beta",  VarB),
    emit_atom_table_go(Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "var ("),
        sub_string(S, _, _, _, "&Atom{Name: \"alpha\"}"),
        sub_string(S, _, _, _, "&Atom{Name: \"beta\"}"),
        atom_string(VarA, VarAS),
        atom_string(VarB, VarBS),
        sub_string(S, _, _, _, VarAS),
        sub_string(S, _, _, _, VarBS)
    ->  pass(Test)
    ;   fail_test(Test, ['emit_atom_table_go output missing entries: ', S])
    ).

test_emit_table_empty_when_none_interned :-
    Test = 'WAM-Go-Lowered Phase 1: emit_atom_table_go yields empty string when no atoms interned',
    init_atom_intern_table_go,
    emit_atom_table_go(Code),
    (   Code == ""
    ->  pass(Test)
    ;   fail_test(Test, ['expected empty; got ', Code])
    ).

%% ---------------------------------------------------------------------
%% dimension_n resolution
%% ---------------------------------------------------------------------

test_dimn_default_is_5 :-
    Test = 'WAM-Go-Lowered Phase 1: resolve_dimension_n_go default is 5',
    retractall(user:dimension_n(_)),
    resolve_dimension_n_go([], N),
    (   N == 5
    ->  pass(Test)
    ;   fail_test(Test, ['expected 5; got ', N])
    ).

test_dimn_option_overrides_default :-
    Test = 'WAM-Go-Lowered Phase 1: dimension_n(N) option overrides the default',
    retractall(user:dimension_n(_)),
    resolve_dimension_n_go([dimension_n(7)], N),
    (   N == 7
    ->  pass(Test)
    ;   fail_test(Test, ['expected 7; got ', N])
    ).

test_dimn_user_fact_fallback :-
    Test = 'WAM-Go-Lowered Phase 1: user:dimension_n/1 fact is honored',
    retractall(user:dimension_n(_)),
    assertz(user:dimension_n(3)),
    resolve_dimension_n_go([], N),
    retractall(user:dimension_n(_)),
    (   N == 3
    ->  pass(Test)
    ;   fail_test(Test, ['expected 3 from user fact; got ', N])
    ).

test_dimn_option_beats_user_fact :-
    Test = 'WAM-Go-Lowered Phase 1: dimension_n(N) option wins over user:dimension_n/1',
    retractall(user:dimension_n(_)),
    assertz(user:dimension_n(3)),
    resolve_dimension_n_go([dimension_n(11)], N),
    retractall(user:dimension_n(_)),
    (   N == 11
    ->  pass(Test)
    ;   fail_test(Test, ['expected 11 from option; got ', N])
    ).

test_dimn_rejects_non_positive :-
    Test = 'WAM-Go-Lowered Phase 1: non-positive dimension_n falls through to default',
    retractall(user:dimension_n(_)),
    resolve_dimension_n_go([dimension_n(0)], N),
    (   N == 5
    ->  pass(Test)
    ;   fail_test(Test, ['expected fallback to 5 on dimension_n(0); got ', N])
    ).

%% ---------------------------------------------------------------------
%% Runner
%% ---------------------------------------------------------------------

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Go-Lowered Phase 1 tests ===~n', []),
    test_intern_returns_stable_var_name,
    test_intern_assigns_distinct_names,
    test_intern_var_name_shape,
    test_intern_sanitizes_special_chars,
    test_init_resets_table,
    test_emit_table_lists_all_atoms,
    test_emit_table_empty_when_none_interned,
    test_dimn_default_is_5,
    test_dimn_option_overrides_default,
    test_dimn_user_fact_fallback,
    test_dimn_option_beats_user_fact,
    test_dimn_rejects_non_positive,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 1 tests passed ===~n', [])
    ).
