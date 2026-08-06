:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_go_lmdb_materialisation.pl — LMDB-GO option plumbing
%
% The WAM-Go LMDB atom-fact source gained eager/lazy/cached tiers
% (mirroring F#'s LmdbFactSource). This file covers the Prolog side:
% how lmdb_materialisation/1 and lmdb_l2_capacity/1 resolve into the
% emitted registerLmdbAtomFact2 call.
%
% The runtime behaviour of the three tiers — helper-invocation counts
% per tier, L1/L2 dispatch, miss caching — is covered by
% TestLmdbMaterialisationTiers in tests/test_wam_go_foreign_lowering.pl,
% which builds and runs the generated Go.

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- begin_tests(go_lmdb_materialisation).

setup_line(Options, Line) :-
    wam_go_target:go_foreign_setup_line(
        register_lmdb_atom_fact2(edge/2, '/tmp/edge_artifact'), Options, Line).

% --- explicit tiers ---------------------------------------------------

test(default_is_cached_with_4096_l2) :-
    setup_line([], Line),
    assertion(sub_atom(Line, _, _, _, '"cached", 4096')).

test(explicit_eager) :-
    setup_line([lmdb_materialisation(eager)], Line),
    assertion(sub_atom(Line, _, _, _, '"eager", 0')).

test(explicit_lazy) :-
    setup_line([lmdb_materialisation(lazy)], Line),
    assertion(sub_atom(Line, _, _, _, '"lazy", 0')).

test(explicit_cached_with_capacity) :-
    setup_line([lmdb_materialisation(cached), lmdb_l2_capacity(512)], Line),
    assertion(sub_atom(Line, _, _, _, '"cached", 512')).

% Only the cached tier has a cache to size, so a capacity passed
% alongside eager/lazy is reported as 0 rather than silently implying a
% cache the generated source will not build.
test(capacity_ignored_for_non_cached_tiers) :-
    setup_line([lmdb_materialisation(eager), lmdb_l2_capacity(512)], Line),
    assertion(sub_atom(Line, _, _, _, '"eager", 0')).

test(artifact_dir_and_pred_key_preserved) :-
    setup_line([lmdb_materialisation(lazy)], Line),
    assertion(sub_atom(Line, _, _, _, 'vm.registerLmdbAtomFact2("edge/2", "/tmp/edge_artifact"')).

% --- auto resolution --------------------------------------------------

% auto defers to the shared cost model in core/cost_model.pl (the same
% rule F#, Haskell and R use), so a workload described once resolves the
% same way across targets. A demand set that cannot fit the declared
% memory budget must not pick eager.
test(auto_with_oversized_demand_set_is_not_eager) :-
    setup_line([lmdb_materialisation(auto),
                fact_count(50_000_000),
                demand_set_estimate(50_000_000),
                memory_budget(1_000_000)], Line),
    assertion(\+ sub_atom(Line, _, _, _, '"eager"')),
    assertion(( sub_atom(Line, _, _, _, '"cached"')
              ; sub_atom(Line, _, _, _, '"lazy"') )).

% A segregated workload has no cross-query key reuse to exploit, so the
% lazy form is bare lazy rather than cached.
test(auto_segregated_oversized_demand_set_is_lazy) :-
    setup_line([lmdb_materialisation(auto),
                fact_count(50_000_000),
                demand_set_estimate(50_000_000),
                memory_budget(1_000_000),
                workload_segregated(true)], Line),
    assertion(sub_atom(Line, _, _, _, '"lazy"')).

test(auto_resolves_to_a_concrete_tier) :-
    setup_line([lmdb_materialisation(auto), fact_count(1000)], Line),
    assertion(( sub_atom(Line, _, _, _, '"eager"')
              ; sub_atom(Line, _, _, _, '"cached"')
              ; sub_atom(Line, _, _, _, '"lazy"') )),
    assertion(\+ sub_atom(Line, _, _, _, '"auto"')).

% auto capacity scales with the demand set, clamped to [256, 65536].
test(auto_capacity_scales_with_demand_set) :-
    setup_line([lmdb_materialisation(cached),
                lmdb_l2_capacity(auto),
                demand_set_estimate(100_000)], Line),
    assertion(sub_atom(Line, _, _, _, '"cached", 10000')).

test(auto_capacity_clamped_low) :-
    setup_line([lmdb_materialisation(cached),
                lmdb_l2_capacity(auto),
                demand_set_estimate(10)], Line),
    assertion(sub_atom(Line, _, _, _, '"cached", 256')).

test(auto_capacity_clamped_high) :-
    setup_line([lmdb_materialisation(cached),
                lmdb_l2_capacity(auto),
                demand_set_estimate(100_000_000)], Line),
    assertion(sub_atom(Line, _, _, _, '"cached", 65536')).

% --- rejected values --------------------------------------------------

test(unknown_materialisation_is_a_domain_error,
     [throws(error(domain_error(lmdb_materialisation, sometimes), _))]) :-
    setup_line([lmdb_materialisation(sometimes)], _).

test(non_positive_capacity_is_a_domain_error,
     [throws(error(domain_error(lmdb_l2_capacity_positive_integer, 0), _))]) :-
    setup_line([lmdb_materialisation(cached), lmdb_l2_capacity(0)], _).

% --- generated runtime shape -----------------------------------------

test(runtime_carries_all_three_tiers) :-
    wam_go_target:compile_wam_runtime_to_go([], RuntimeCode),
    atom_string(RuntimeCode, Runtime),
    assertion(sub_string(Runtime, _, _, _,
        'func newLmdbAtomFact2Source(predKey string, artifactDir string, mode string, l2Capacity int) *lmdbAtomFact2Source')),
    % eager materialises once at construction
    assertion(sub_string(Runtime, _, _, _, 'func (source *lmdbAtomFact2Source) materialise()')),
    assertion(sub_string(Runtime, _, _, _, 'source.materialise()')),
    % cached goes through the two-level dispatch
    assertion(sub_string(Runtime, _, _, _, 'func (source *lmdbAtomFact2Source) lookupCached(left string) []AtomPair')),
    assertion(sub_string(Runtime, _, _, _, 'source.l1[left]')),
    assertion(sub_string(Runtime, _, _, _, 'source.l2[left]')),
    assertion(sub_string(Runtime, _, _, _, 'len(source.l2) < source.l2Capacity')),
    % lazy keeps the original per-lookup helper invocation
    assertion(sub_string(Runtime, _, _, _, 'return source.run("get", source.artifactDir, source.predKey, left)')).

:- end_tests(go_lmdb_materialisation).
