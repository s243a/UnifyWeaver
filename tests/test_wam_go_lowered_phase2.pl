:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Phase 2 codegen-block tests for the WAM-lowered Go path.
%
% Phase 2 covers the structure of the emitted atom intern table —
% the chunk of Go that gets injected at the top of lowered.go before
% any lowered predicate functions. Where Phase 1 tests the helpers in
% isolation, Phase 2 tests how multiple intern_atom_go/2 calls compose
% into a single var(...) block.
%
% These are codegen-only assertions. They do NOT compile the resulting
% Go. The build verification for lowered.go output is exercised via the
% Go effective-distance benchmark in CI.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl

:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% ---------------------------------------------------------------------
%% Atom table block structure
%% ---------------------------------------------------------------------

test_atom_table_has_var_block_header :-
    Test = 'WAM-Go-Lowered Phase 2: atom table opens a var(...) block',
    init_atom_intern_table_go,
    intern_atom_go("hello", _),
    emit_atom_table_go(Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "var ("),
        sub_string(S, _, _, _, ")")
    ->  pass(Test)
    ;   fail_test(Test, ['no var (...) block: ', S])
    ).

test_atom_table_includes_dedup_comment :-
    Test = 'WAM-Go-Lowered Phase 2: atom table block carries a dedup-purpose comment',
    init_atom_intern_table_go,
    intern_atom_go("c", _),
    emit_atom_table_go(Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "Interned atom literals")
    ->  pass(Test)
    ;   fail_test(Test, ['comment missing: ', S])
    ).

test_atom_table_preserves_insertion_order :-
    Test = 'WAM-Go-Lowered Phase 2: atom decls appear in insertion order, not alphabetical',
    init_atom_intern_table_go,
    intern_atom_go("zeta",  _),
    intern_atom_go("alpha", _),
    intern_atom_go("mu",    _),
    emit_atom_table_go(Code),
    atom_string(Code, S),
    sub_string(S, ZetaIdx,  _, _, "wamAtom_zeta_"),
    sub_string(S, AlphaIdx, _, _, "wamAtom_alpha_"),
    sub_string(S, MuIdx,    _, _, "wamAtom_mu_"),
    (   ZetaIdx < AlphaIdx, AlphaIdx < MuIdx
    ->  pass(Test)
    ;   fail_test(Test, ['out of order: zeta=', ZetaIdx, ' alpha=', AlphaIdx, ' mu=', MuIdx])
    ).

test_atom_table_is_valid_go_syntax_shape :-
    Test = 'WAM-Go-Lowered Phase 2: each decl line follows `<name> = &Atom{Name: "..."}` shape',
    init_atom_intern_table_go,
    intern_atom_go("one",   VarOne),
    intern_atom_go("two",   VarTwo),
    intern_atom_go("three", VarThree),
    emit_atom_table_go(Code),
    atom_string(Code, S),
    atom_string(VarOne,   VOS),
    atom_string(VarTwo,   VTS),
    atom_string(VarThree, VRS),
    format(string(LineOne),   "~w = &Atom{Name: \"one\"}",   [VOS]),
    format(string(LineTwo),   "~w = &Atom{Name: \"two\"}",   [VTS]),
    format(string(LineThree), "~w = &Atom{Name: \"three\"}", [VRS]),
    (   sub_string(S, _, _, _, LineOne),
        sub_string(S, _, _, _, LineTwo),
        sub_string(S, _, _, _, LineThree)
    ->  pass(Test)
    ;   fail_test(Test, ['decl shape mismatch in: ', S])
    ).

test_atom_table_escapes_backslashes :-
    Test = 'WAM-Go-Lowered Phase 2: backslashes in atom names are escaped for Go',
    init_atom_intern_table_go,
    % Atom containing a backslash — atom_table must double-escape.
    intern_atom_go("a\\b", _),
    emit_atom_table_go(Code),
    atom_string(Code, S),
    % Source string in Go output should contain "a\\b" (literal four chars).
    (   sub_string(S, _, _, _, "a\\\\b")
    ->  pass(Test)
    ;   fail_test(Test, ['backslash not escaped: ', S])
    ).

%% ---------------------------------------------------------------------
%% Reset / regenerate behavior
%% ---------------------------------------------------------------------

test_regeneration_starts_clean :-
    Test = 'WAM-Go-Lowered Phase 2: re-init clears prior intern state',
    init_atom_intern_table_go,
    intern_atom_go("first",  _),
    intern_atom_go("second", _),
    init_atom_intern_table_go,
    intern_atom_go("third", _),
    emit_atom_table_go(Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "wamAtom_third_"),
        \+ sub_string(S, _, _, _, "wamAtom_first_"),
        \+ sub_string(S, _, _, _, "wamAtom_second_")
    ->  pass(Test)
    ;   fail_test(Test, ['regenerate did not clean prior state: ', S])
    ).

%% ---------------------------------------------------------------------
%% Runner
%% ---------------------------------------------------------------------

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Go-Lowered Phase 2 tests ===~n', []),
    test_atom_table_has_var_block_header,
    test_atom_table_includes_dedup_comment,
    test_atom_table_preserves_insertion_order,
    test_atom_table_is_valid_go_syntax_shape,
    test_atom_table_escapes_backslashes,
    test_regeneration_starts_clean,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 2 tests passed ===~n', [])
    ).
