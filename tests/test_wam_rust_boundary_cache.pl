:- encoding(utf8).
% Guard test for the boundary distribution cache P1 core in the Rust WAM target.
%
% P1 is the exact path-length-histogram splice (boundary_cache.rs.mustache):
% suffix_histogram (recurrence), splice_truncated (convolution), and the linear
% functionals (mass / moment1 / weighted_power). Correctness is proven by the
% Rust unit tests in that module (splice == full DFS, the Rust analog of the
% Python exact-match splice validation), run via `cargo test --lib`. This test
% pins the template strings and the lib wiring so a future edit can't silently
% drop the module. See WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md.
%
% Usage: swipl -q -g run_tests -t halt tests/test_wam_rust_boundary_cache.pl

:- dynamic test_failed/0.

pass(Test) :- format('[PASS] ~w~n', [Test]).
fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (test_failed -> true ; assert(test_failed)).

assert_contains(Code, Needle, Test) :-
    (   sub_string(Code, _, _, _, Needle)
    ->  pass(Test)
    ;   fail_test(Test, Needle)
    ).

test_boundary_cache_template_has_splice_core :-
    read_file_to_string('templates/targets/rust_wam/boundary_cache.rs.mustache', Code, []),
    assert_contains(Code, "pub fn suffix_histogram",
        'WAM-Rust: boundary_cache defines suffix_histogram (recurrence)'),
    assert_contains(Code, "pub fn splice_truncated",
        'WAM-Rust: boundary_cache defines splice_truncated (convolution)'),
    assert_contains(Code, "pub fn f_weighted_power",
        'WAM-Rust: boundary_cache exposes the weighted_power functional'),
    assert_contains(Code, "splice_equals_full_dfs_histogram",
        'WAM-Rust: boundary_cache carries the splice==full-DFS identity test').

test_lib_template_declares_module :-
    read_file_to_string('src/unifyweaver/core/template_system.pl', Code, []),
    assert_contains(Code, "pub mod boundary_cache;",
        'WAM-Rust: rust_wam_lib declares pub mod boundary_cache').

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Rust Boundary Cache Guard Test ===~n', []),
    test_boundary_cache_template_has_splice_core,
    test_lib_template_declares_module,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All boundary cache guard tests passed ===~n', [])
    ).
