:- encoding(utf8).
% Test suite for WAM-Elixir Utilities
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_utils.pl

:- use_module('../src/unifyweaver/targets/wam_elixir_utils').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

% --- is_label_part/1 tests ---

test_is_label_part_valid :-
    Test = 'is_label_part/1: valid labels',
    (   is_label_part("label:"),
        is_label_part("L1:"),
        is_label_part("test_pred/2:"),
        is_label_part('atom_label:')
    ->  pass(Test)
    ;   fail_test(Test, 'failed to identify valid labels')
    ).

test_is_label_part_invalid :-
    Test = 'is_label_part/1: invalid labels',
    (   \+ is_label_part("not_a_label"),
        \+ is_label_part("put_variable"),
        \+ is_label_part("A1"),
        \+ is_label_part("% comment:"),
        \+ is_label_part("%:")
    ->  pass(Test)
    ;   fail_test(Test, 'incorrectly identified invalid labels')
    ).

% --- reg_id/2 tests ---

test_reg_id_x_a :-
    Test = 'reg_id/2: X and A registers',
    (   reg_id("X1", 1),
        reg_id("A2", 2),
        reg_id('X99', 99),
        reg_id('A100', 100)
    ->  pass(Test)
    ;   fail_test(Test, 'failed mapping X/A registers')
    ).

test_reg_id_y :-
    Test = 'reg_id/2: Y registers',
    (   reg_id("Y1", 101),
        reg_id("Y2", 102),
        reg_id('Y99', 199)
    ->  pass(Test)
    ;   fail_test(Test, 'failed mapping Y registers with 100 offset')
    ).

test_reg_id_fallback :-
    Test = 'reg_id/2: fallback for non-registers',
    (   reg_id("not_a_reg", "not_a_reg"),
        reg_id('foo', 'foo'),
        reg_id(42, 42)
    ->  pass(Test)
    ;   fail_test(Test, 'failed fallback mapping')
    ).

% --- clean_comma/2 tests ---

test_clean_comma_trailing :-
    Test = 'clean_comma/2: strips trailing comma',
    (   clean_comma("A1,", "A1"),
        clean_comma("foo/2,", "foo/2"),
        clean_comma('label,', 'label')
    ->  pass(Test)
    ;   fail_test(Test, 'failed to strip trailing comma')
    ).

test_clean_comma_no_comma :-
    Test = 'clean_comma/2: leaves strings without comma unchanged',
    (   clean_comma("A1", "A1"),
        clean_comma("foo/2", "foo/2"),
        clean_comma('label', 'label')
    ->  pass(Test)
    ;   fail_test(Test, 'incorrectly modified string without trailing comma')
    ).

run_tests :-
    format('~n=== WAM-Elixir Utils Target Tests ===~n~n'),
    test_is_label_part_valid,
    test_is_label_part_invalid,
    test_reg_id_x_a,
    test_reg_id_y,
    test_reg_id_fallback,
    test_clean_comma_trailing,
    test_clean_comma_no_comma,
    format('~n=== WAM-Elixir Utils Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).
