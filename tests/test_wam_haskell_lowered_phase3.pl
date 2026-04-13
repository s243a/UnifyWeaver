:- encoding(utf8).
% Phase 3 smoke tests for the WAM-lowered Haskell path.
%
% Phase 3 adds:
%   - wam_haskell_lowered_emitter.pl with a real wam_haskell_lowerable/3
%     (whitelist: get_constant + proceed) and lower_predicate_to_haskell/4
%   - Re-export through wam_haskell_target.pl
%   - Lowered.hs emits actual function definitions when the lowered
%     partition is non-empty, plus a non-empty loweredPredicates map
%
% These are codegen-only assertions. The build+run verification is done
% manually via a temporary project directory with the effective_distance
% 10k benchmark, output byte-identical to the Prolog reference (verified
% to match Phase 2 output via diff, Temperature=2.840088).
%
% Usage: swipl -g run_tests -t halt tests/test_wam_haskell_lowered_phase3.pl

:- use_module('../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../src/unifyweaver/targets/wam_haskell_lowered_emitter').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% A probe predicate with a single-clause integer fact body. Compiles to
%% get_constant + proceed, exactly what the Phase 3 whitelist accepts.
:- dynamic user:phase3_constant/1.
user:phase3_constant(42).

%% A probe predicate with multiple clauses — its WAM uses try_me_else /
%% trust_me and must NOT be lowerable under the Phase 3 whitelist.
:- dynamic user:phase3_multi/1.
user:phase3_multi(a).
user:phase3_multi(b).

tmp_root_candidate(Root) :-
    member(Env, ['TMPDIR', 'TMP', 'TEMP']),
    getenv(Env, Root),
    Root \== ''.
tmp_root_candidate(Root) :-
    getenv('PREFIX', Prefix),
    Prefix \== '',
    directory_file_path(Prefix, tmp, Root).
tmp_root_candidate('output').

writable_tmp_root(Root) :-
    tmp_root_candidate(Root),
    catch(make_directory_path(Root), _, fail),
    access_file(Root, write),
    !.

project_dir(Dir) :-
    writable_tmp_root(Root),
    directory_file_path(Root, 'uw_wam_hs_lowered_phase3_test', Dir).

read_file_to_string(Path, Str) :-
    open(Path, read, S),
    read_string(S, _, Str),
    close(S).

%% ---------------------------------------------------------------------
%% Emitter unit tests
%% ---------------------------------------------------------------------

test_lowerable_accepts_single_clause_fact :-
    Test = 'Phase 3: wam_haskell_lowerable accepts single-clause fact',
    wam_target:compile_predicate_to_wam(phase3_constant/1, [], WamCode),
    (   wam_haskell_lowerable(user:phase3_constant/1, WamCode, _)
    ->  pass(Test)
    ;   fail_test(Test, 'single-clause integer fact was rejected')
    ).

test_lowerable_accepts_multi_clause :-
    Test = 'Phase 3: wam_haskell_lowerable accepts multi-clause fact (try_me_else)',
    wam_target:compile_predicate_to_wam(phase3_multi/1, [], WamCode),
    (   wam_haskell_lowerable(user:phase3_multi/1, WamCode, _)
    ->  pass(Test)
    ;   fail_test(Test, 'multi-clause fact was rejected — should be lowerable now')
    ).

test_lower_predicate_produces_function :-
    Test = 'Phase 3: lower_predicate_to_haskell emits expected function',
    wam_target:compile_predicate_to_wam(phase3_constant/1, [], WamCode),
    lower_predicate_to_haskell(user:phase3_constant/1, WamCode, [],
                               lowered(PredName, FuncName, HsCode)),
    (   PredName == 'phase3_constant/1',
        FuncName == lowered_phase3_constant_1,
        sub_string(HsCode, _, _, _, "lowered_phase3_constant_1 :: WamContext -> WamState -> Maybe WamState"),
        sub_string(HsCode, _, _, _, "GetConstant (Integer 42)")
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected emitter output; PredName=', PredName, ' FuncName=', FuncName])
    ).

%% ---------------------------------------------------------------------
%% Partition behavior
%% ---------------------------------------------------------------------

test_partition_mixed_lowers_supported :-
    Test = 'Phase 3: partition in mixed mode routes supported preds to LoweredList',
    wam_haskell_partition_predicates(
        mixed([phase3_constant/1]),
        [user:phase3_constant/1, user:phase3_multi/1],
        [],
        Interpreted,
        Lowered),
    %% phase3_constant/1 is hot + lowerable → Lowered
    %% phase3_multi/1 is cold → Interpreted (not in HotPreds)
    (   Lowered   == [user:phase3_constant/1],
        Interpreted == [user:phase3_multi/1]
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected partition Interp=', Interpreted, ' Lower=', Lowered])
    ).

test_partition_functions_lowers_all_supported :-
    Test = 'Phase 3: partition in functions mode lowers all supported preds',
    wam_haskell_partition_predicates(
        functions,
        [user:phase3_constant/1, user:phase3_multi/1],
        [],
        Interpreted,
        Lowered),
    %% Both are lowerable now (multi-clause restriction removed)
    (   Lowered   == [user:phase3_constant/1, user:phase3_multi/1],
        Interpreted == []
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected partition Interp=', Interpreted, ' Lower=', Lowered])
    ).

%% ---------------------------------------------------------------------
%% End-to-end project-file codegen
%% ---------------------------------------------------------------------

generate_phase3_project :-
    project_dir(Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ),
    % Only phase3_constant/1 — keep the interpreted partition empty so
    % the test does not exercise the pre-existing wam_value_to_haskell
    % parser bug on switch_on_constant with atom keys (out of scope
    % for this branch).
    write_wam_haskell_project(
        [user:phase3_constant/1],
        [module_name('uw-wam-hs-lowered-phase3-test'),
         use_hashmap(false),
         emit_mode(mixed([phase3_constant/1]))],
        Dir).

test_lowered_hs_has_function_definition :-
    Test = 'Phase 3: Lowered.hs contains the emitted function',
    generate_phase3_project,
    project_dir(Dir),
    atom_concat(Dir, '/src/Lowered.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "lowered_phase3_constant_1 :: WamContext -> WamState -> Maybe WamState"),
        sub_string(S, _, _, _, "GetConstant (Integer 42)")
    ->  pass(Test)
    ;   fail_test(Test, 'Lowered.hs missing expected function body')
    ).

test_lowered_hs_dispatch_map_populated :-
    Test = 'Phase 3: Lowered.hs loweredPredicates dispatch map is non-empty',
    project_dir(Dir),
    atom_concat(Dir, '/src/Lowered.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "loweredPredicates = Map.fromList"),
        sub_string(S, _, _, _, "(\"phase3_constant/1\", lowered_phase3_constant_1)")
    ->  pass(Test)
    ;   fail_test(Test, 'loweredPredicates missing expected entry')
    ).

test_predicates_hs_still_emitted :-
    Test = 'Phase 3: Predicates.hs still emitted (empty instruction array) so Main.hs compiles',
    project_dir(Dir),
    atom_concat(Dir, '/src/Predicates.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "allCode :: [Instruction]"),
        sub_string(S, _, _, _, "allLabels")
    ->  pass(Test)
    ;   fail_test(Test, 'Predicates.hs missing allCode/allLabels even in pure-lowered mode')
    ).

%% ---------------------------------------------------------------------
%% Runner
%% ---------------------------------------------------------------------

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Haskell-Lowered Phase 3 tests ===~n', []),
    test_lowerable_accepts_single_clause_fact,
    test_lowerable_accepts_multi_clause,
    test_lower_predicate_produces_function,
    test_partition_mixed_lowers_supported,
    test_partition_functions_lowers_all_supported,
    test_lowered_hs_has_function_definition,
    test_lowered_hs_dispatch_map_populated,
    test_predicates_hs_still_emitted,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 3 tests passed ===~n', [])
    ).
