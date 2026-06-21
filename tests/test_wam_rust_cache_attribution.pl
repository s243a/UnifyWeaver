:- encoding(utf8).
% Guard test for cached-side runtime attribution in the Rust WAM target.
%
% Rust-side analog of the Python boundary-cache attribution from PR #3120
% (scripts/lmdb_parent_boundary_cache_benchmark.py). The CachedLookup hot
% path in state.rs.mustache records which cache tier served each
% lookup_parents (L1/L2/miss) and how long the inner LMDB seek took on a
% miss; the benchmark main surfaces it as key=value stderr lines when
% UW_WAM_CACHE_ATTRIBUTION is set. This test pins those strings in the
% templates so a future edit can't silently drop the instrumentation.
%
% Behavioural coverage lives in the Rust unit tests
% (state.rs.mustache mod cache_attribution_tests), run via `cargo test`.
%
% Usage: swipl -q -g run_tests -t halt tests/test_wam_rust_cache_attribution.pl

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (test_failed -> true ; assert(test_failed)).

assert_contains(Code, Needle, Test) :-
    (   sub_string(Code, _, _, _, Needle)
    ->  pass(Test)
    ;   fail_test(Test, Needle)
    ).

test_state_template_has_attribution_scaffold :-
    read_file_to_string('templates/targets/rust_wam/state.rs.mustache', Code, []),
    assert_contains(Code, "pub struct CacheAttribution",
        'WAM-Rust: state.rs defines CacheAttribution'),
    assert_contains(Code, "pub fn cache_attribution",
        'WAM-Rust: state.rs exposes process-global cache_attribution()'),
    assert_contains(Code, "attribution: Option<Arc<CacheAttribution>>",
        'WAM-Rust: CachedLookup carries an opt-in attribution sink'),
    assert_contains(Code, "cache_attr_lookups",
        'WAM-Rust: attribution renders key=value report lines').

test_main_template_surfaces_attribution :-
    read_file_to_string('templates/targets/rust_wam/main.rs.mustache', Code, []),
    assert_contains(Code, "::state::cache_attribution()",
        'WAM-Rust: benchmark main reads back the shared attribution sink').

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Rust Cache Attribution Guard Test ===~n', []),
    test_state_template_has_attribution_scaffold,
    test_main_template_surfaces_attribution,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All cache attribution guard tests passed ===~n', [])
    ).
