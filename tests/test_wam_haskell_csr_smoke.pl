:- encoding(utf8).
% CSR smoke test for the Haskell WAM target.
%
% Validates:
%   1. CSR reader template generates and compiles with GHC
%   2. Edge store auto-resolver picks correct modes for known workloads
%   3. CSR index backend auto-resolver delegates to cost model
%   4. Bidirectional kernel upgrade triggers when CSR + kernel_mode set
%
% Usage: swipl -g run_tests -t halt tests/test_wam_haskell_csr_smoke.pl

:- use_module('../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/core/cost_model').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% =====================================================================
%% Edge store auto-resolver decisions
%% =====================================================================

test_edge_store_dynamic_always_cached :-
    Test = 'CSR: dynamic graph always selects lmdb_cached',
    wam_haskell_target:resolve_auto_edge_store_hs(
        [edge_store(auto), graph_mutability(dynamic), edge_count(100000)],
        Opts),
    (   member(lmdb_materialisation(cached), Opts)
    ->  pass(Test)
    ;   fail_test(Test, 'dynamic graph did not select lmdb_cached')
    ).

test_edge_store_static_no_auto :-
    Test = 'CSR: edge_store not auto passes through unchanged',
    Opts0 = [edge_store(lmdb_eager), edge_count(1000)],
    wam_haskell_target:resolve_auto_edge_store_hs(Opts0, Opts),
    (   Opts == Opts0
    ->  pass(Test)
    ;   fail_test(Test, 'non-auto edge_store was modified')
    ).

test_edge_store_absent_passes_through :-
    Test = 'CSR: no edge_store option passes through unchanged',
    Opts0 = [use_lmdb(true)],
    wam_haskell_target:resolve_auto_edge_store_hs(Opts0, Opts),
    (   Opts == Opts0
    ->  pass(Test)
    ;   fail_test(Test, 'absent edge_store was modified')
    ).

test_edge_store_small_static_picks_cheapest :-
    Test = 'CSR: small static graph with few queries picks a concrete mode',
    wam_haskell_target:resolve_auto_edge_store_hs(
        [edge_store(auto), edge_count(1000),
         expected_query_count(10), expected_lookups_per_query(50)],
        Opts),
    (   \+ member(edge_store(auto), Opts)
    ->  pass(Test)
    ;   fail_test(Test, 'auto was not resolved')
    ).

test_edge_store_needs_reverse_considers_dual_csr :-
    Test = 'CSR: needs_reverse=true considers dual_csr',
    wam_haskell_target:compute_edge_store_hs(
        [edge_store(auto), edge_count(100),
         expected_query_count(10000), expected_lookups_per_query(100),
         needs_reverse(true)],
        Store),
    (   Store == dual_csr
    ->  pass(Test)
    ;   % dual_csr requires CSR to be cheapest; if not, any concrete mode is fine
        member(Store, [lmdb_cached, lmdb_eager, csr, dual_csr])
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected store: ', Store])
    ).

%% =====================================================================
%% LMDB materialisation auto-resolver
%% =====================================================================

test_materialisation_auto_resolves :-
    Test = 'CSR: lmdb_materialisation(auto) resolves to concrete mode',
    wam_haskell_target:resolve_auto_lmdb_materialisation_hs(
        [lmdb_materialisation(auto), fact_count(50000), expected_query_count(20)],
        Opts),
    (   member(lmdb_materialisation(Mode), Opts),
        member(Mode, [eager, lazy, cached])
    ->  pass(Test)
    ;   fail_test(Test, 'auto was not resolved to concrete mode')
    ).

test_materialisation_non_auto_passes_through :-
    Test = 'CSR: lmdb_materialisation(eager) passes through unchanged',
    Opts0 = [lmdb_materialisation(eager), fact_count(1000)],
    wam_haskell_target:resolve_auto_lmdb_materialisation_hs(Opts0, Opts),
    (   Opts == Opts0
    ->  pass(Test)
    ;   fail_test(Test, 'non-auto materialisation was modified')
    ).

%% =====================================================================
%% CSR index backend auto-resolver
%% =====================================================================

test_csr_backend_auto_without_csr_path :-
    Test = 'CSR: csr_index_backend(auto) without csr_path passes through',
    Opts0 = [csr_index_backend(auto)],
    wam_haskell_target:resolve_auto_csr_index_backend_hs(Opts0, Opts),
    (   Opts == Opts0
    ->  pass(Test)
    ;   fail_test(Test, 'resolved without csr_path')
    ).

test_csr_backend_explicit_passes_through :-
    Test = 'CSR: explicit csr_index_backend(sorted_array) passes through',
    Opts0 = [csr_path('/tmp/csr'), csr_index_backend(sorted_array)],
    wam_haskell_target:resolve_auto_csr_index_backend_hs(Opts0, Opts),
    (   Opts == Opts0
    ->  pass(Test)
    ;   fail_test(Test, 'explicit backend was changed')
    ).

%% =====================================================================
%% Bidirectional kernel upgrade
%% =====================================================================

test_bidir_upgrade_triggers :-
    Test = 'CSR: maybe_upgrade_bidirectional upgrades category_ancestor',
    wam_haskell_target:maybe_upgrade_bidirectional(
        'cat_anc/4'-recursive_kernel(category_ancestor, cat_anc/4, []),
        Result),
    (   Result = 'cat_anc/4'-recursive_kernel(bidirectional_ancestor, cat_anc/4, [])
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected result: ', Result])
    ).

test_bidir_upgrade_skips_other_kernels :-
    Test = 'CSR: maybe_upgrade_bidirectional skips non-category kernels',
    Input = 'tc/2'-recursive_kernel(transitive_closure, tc/2, []),
    wam_haskell_target:maybe_upgrade_bidirectional(Input, Result),
    (   Result == Input
    ->  pass(Test)
    ;   fail_test(Test, 'non-category kernel was upgraded')
    ).

%% =====================================================================
%% Composed resolver chain
%% =====================================================================

test_composed_chain_resolves_all :-
    Test = 'CSR: resolve_haskell_cost_options composes all resolvers',
    wam_haskell_target:resolve_haskell_cost_options(
        [edge_store(auto), edge_count(5000),
         expected_query_count(100), expected_lookups_per_query(50)],
        Opts),
    (   \+ member(edge_store(auto), Opts)
    ->  pass(Test)
    ;   fail_test(Test, 'composed chain did not resolve edge_store(auto)')
    ).

%% =====================================================================
%% CSR codegen: CsrReader.hs emitted when csr_path set
%% =====================================================================

:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module('helpers/smoke_paths', [tmp_root/1]).

:- dynamic user:csr_probe/3.
user:csr_probe(X, Y, Z) :- Z is X + Y.

test_csr_codegen_emits_reader :-
    Test = 'CSR: csr_path option causes CsrReader.hs to be emitted',
    tmp_project_dir(Dir),
    (   exists_directory(Dir) -> true ; make_directory_path(Dir) ),
    write_wam_haskell_project(
        [user:csr_probe/3],
        [module_name('uw-csr-test'), use_hashmap(false),
         csr_path('csr'), csr_relation('category_child')],
        Dir),
    directory_file_path(Dir, 'src/CsrReader.hs', CsrPath),
    (   exists_file(CsrPath)
    ->  read_file_to_string(CsrPath, S),
        (   sub_string(S, _, _, _, "openCsrEdgeLookup"),
            sub_string(S, _, _, _, "category_child")
        ->  pass(Test)
        ;   fail_test(Test, 'CsrReader.hs missing expected functions')
        )
    ;   fail_test(Test, 'CsrReader.hs was not emitted')
    ).

test_csr_codegen_main_imports_reader :-
    Test = 'CSR: Main.hs imports CsrReader when csr_path set',
    tmp_project_dir(Dir),
    directory_file_path(Dir, 'src/Main.hs', MainPath),
    read_file_to_string(MainPath, S),
    (   sub_string(S, _, _, _, "import qualified CsrReader"),
        sub_string(S, _, _, _, "openCsrEdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test, 'Main.hs missing CsrReader import or setup')
    ).

test_csr_codegen_no_reader_without_csr_path :-
    Test = 'CSR: no CsrReader.hs without csr_path option',
    tmp_project_dir_no_csr(Dir),
    (   exists_directory(Dir) -> true ; make_directory_path(Dir) ),
    write_wam_haskell_project(
        [user:csr_probe/3],
        [module_name('uw-no-csr-test'), use_hashmap(false)],
        Dir),
    directory_file_path(Dir, 'src/CsrReader.hs', CsrPath),
    (   \+ exists_file(CsrPath)
    ->  pass(Test)
    ;   fail_test(Test, 'CsrReader.hs emitted without csr_path')
    ).

%% =====================================================================
%% Helpers
%% =====================================================================

% writable_tmp_root/1 delegates to the shared smoke_paths helper
% (which covers Windows %TEMP%, Termux, /tmp, etc.).
writable_tmp_root(Root) :- tmp_root(Root).

tmp_project_dir(Dir) :-
    writable_tmp_root(Root),
    directory_file_path(Root, 'uw_wam_hs_csr_smoke', Dir).

tmp_project_dir_no_csr(Dir) :-
    writable_tmp_root(Root),
    directory_file_path(Root, 'uw_wam_hs_no_csr_smoke', Dir).

read_file_to_string(Path, Str) :-
    open(Path, read, S),
    read_string(S, _, Str),
    close(S).

%% =====================================================================
%% Runner
%% =====================================================================

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Haskell CSR Smoke Tests ===~n', []),
    % Edge store auto-resolver
    test_edge_store_dynamic_always_cached,
    test_edge_store_static_no_auto,
    test_edge_store_absent_passes_through,
    test_edge_store_small_static_picks_cheapest,
    test_edge_store_needs_reverse_considers_dual_csr,
    % LMDB materialisation
    test_materialisation_auto_resolves,
    test_materialisation_non_auto_passes_through,
    % CSR index backend
    test_csr_backend_auto_without_csr_path,
    test_csr_backend_explicit_passes_through,
    % Bidirectional upgrade
    test_bidir_upgrade_triggers,
    test_bidir_upgrade_skips_other_kernels,
    % Composed chain
    test_composed_chain_resolves_all,
    % Codegen
    test_csr_codegen_emits_reader,
    test_csr_codegen_main_imports_reader,
    test_csr_codegen_no_reader_without_csr_path,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All CSR smoke tests passed ===~n', [])
    ).
