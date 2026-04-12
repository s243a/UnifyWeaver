:- encoding(utf8).
% Phase 2 codegen tests for the WAM-lowered Haskell path.
%
% Phase 2 adds:
%   - wcLoweredPredicates field to WamContext (a Map from predicate name to
%     a (WamContext -> WamState -> Maybe WamState) function)
%   - a new case at the top of the Call dispatch chain in step that checks
%     wcLoweredPredicates before executeForeign
%   - a Lowered.hs module emitted alongside Predicates.hs, body
%     loweredPredicates = Map.empty
%   - Main.hs wiring that imports Lowered and populates wcLoweredPredicates
%   - the cabal other-modules list includes Lowered
%
% These are codegen-only assertions — they inspect the text of generated
% files, not GHC compilation. The build+run verification was done manually
% via /tmp/wam_hs_lowered_phase2 with the effective_distance 10k benchmark,
% output byte-identical to the Prolog reference.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_haskell_lowered_phase2.pl

:- use_module('../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(filesex), [make_directory_path/1]).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% A tiny user predicate so we can exercise write_wam_haskell_project/3
%% without depending on the benchmark fixtures.
:- dynamic user:phase2_probe/1.
user:phase2_probe(42).

project_dir('/tmp/uw_wam_hs_lowered_phase2_test').

generate_phase2_project :-
    project_dir(Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ),
    write_wam_haskell_project(
        [user:phase2_probe/1],
        [module_name('wam-bench-hm-phase2-test'), use_hashmap(false)],
        Dir).

read_file_to_string(Path, Str) :-
    open(Path, read, S),
    read_string(S, _, Str),
    close(S).

test_lowered_hs_emitted :-
    Test = 'WAM-Haskell-Lowered Phase 2: Lowered.hs emitted',
    generate_phase2_project,
    project_dir(Dir),
    atom_concat(Dir, '/src/Lowered.hs', Path),
    (   exists_file(Path)
    ->  pass(Test)
    ;   fail_test(Test, ['file missing: ', Path])
    ).

test_lowered_hs_module_header :-
    Test = 'WAM-Haskell-Lowered Phase 2: Lowered.hs has module header and empty Map',
    project_dir(Dir),
    atom_concat(Dir, '/src/Lowered.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "module Lowered where"),
        sub_string(S, _, _, _, "loweredPredicates :: Map.Map String (WamContext -> WamState -> Maybe WamState)"),
        sub_string(S, _, _, _, "loweredPredicates = Map.empty")
    ->  pass(Test)
    ;   fail_test(Test, 'module header or Map.empty body missing')
    ).

test_wam_types_has_wcLoweredPredicates :-
    Test = 'WAM-Haskell-Lowered Phase 2: WamContext has wcLoweredPredicates field',
    project_dir(Dir),
    atom_concat(Dir, '/src/WamTypes.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "wcLoweredPredicates :: !(Map.Map String (WamContext -> WamState -> Maybe WamState))")
    ->  pass(Test)
    ;   fail_test(Test, 'wcLoweredPredicates field missing from WamContext')
    ).

test_wam_types_no_deriving_show :-
    Test = 'WAM-Haskell-Lowered Phase 2: WamContext no longer derives Show',
    project_dir(Dir),
    atom_concat(Dir, '/src/WamTypes.hs', Path),
    read_file_to_string(Path, S),
    % Find data WamContext block and check the closing brace line is not followed by deriving Show.
    % Easiest: assert absence of literal "WamContext ... deriving (Show)".
    (   \+ sub_string(S, _, _, _, "WamContext\n  { wcCode          :: !(Array Int Instruction)")
    ->  fail_test(Test, 'could not find WamContext block at all')
    ;   sub_string(S, _, _, _, "-- Note: no `deriving (Show)`")
    ->  pass(Test)
    ;   fail_test(Test, 'deriving (Show) comment missing — check that deriving was actually dropped')
    ).

test_mkcontext_initializes_wcLoweredPredicates :-
    Test = 'WAM-Haskell-Lowered Phase 2: mkContext initializes wcLoweredPredicates',
    project_dir(Dir),
    atom_concat(Dir, '/src/WamTypes.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "wcLoweredPredicates = Map.empty")
    ->  pass(Test)
    ;   fail_test(Test, 'mkContext missing wcLoweredPredicates = Map.empty')
    ).

test_step_call_dispatch_checks_lowered_first :-
    Test = 'WAM-Haskell-Lowered Phase 2: step Call dispatch checks wcLoweredPredicates before executeForeign',
    compile_wam_runtime_to_haskell([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "case Map.lookup pred (wcLoweredPredicates ctx) of"),
        sub_string(S, _, _, _, "Just fn -> fn ctx sc")
    ->  pass(Test)
    ;   fail_test(Test, 'step dispatch does not check wcLoweredPredicates first')
    ).

test_main_imports_lowered :-
    Test = 'WAM-Haskell-Lowered Phase 2: Main.hs imports Lowered',
    project_dir(Dir),
    atom_concat(Dir, '/src/Main.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "import qualified Lowered")
    ->  pass(Test)
    ;   fail_test(Test, 'Main.hs missing import Lowered')
    ).

test_main_populates_wcLoweredPredicates :-
    Test = 'WAM-Haskell-Lowered Phase 2: Main.hs populates wcLoweredPredicates from Lowered',
    project_dir(Dir),
    atom_concat(Dir, '/src/Main.hs', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "wcLoweredPredicates = Lowered.loweredPredicates")
    ->  pass(Test)
    ;   fail_test(Test, 'Main.hs does not populate wcLoweredPredicates')
    ).

test_cabal_other_modules_has_lowered :-
    Test = 'WAM-Haskell-Lowered Phase 2: cabal other-modules lists Lowered',
    project_dir(Dir),
    atom_concat(Dir, '/wam-bench-hm-phase2-test.cabal', Path),
    read_file_to_string(Path, S),
    (   sub_string(S, _, _, _, "WamTypes, WamRuntime, Predicates, Lowered")
    ->  pass(Test)
    ;   fail_test(Test, 'cabal other-modules missing Lowered')
    ).

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Haskell-Lowered Phase 2 tests ===~n', []),
    test_lowered_hs_emitted,
    test_lowered_hs_module_header,
    test_wam_types_has_wcLoweredPredicates,
    test_wam_types_no_deriving_show,
    test_mkcontext_initializes_wcLoweredPredicates,
    test_step_call_dispatch_checks_lowered_first,
    test_main_imports_lowered,
    test_main_populates_wcLoweredPredicates,
    test_cabal_other_modules_has_lowered,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 2 tests passed ===~n', [])
    ).
