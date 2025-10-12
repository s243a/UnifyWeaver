% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% init_template.pl - UnifyWeaver testing environment initialization
% This template is used by init_testing.sh and Init-TestEnvironment.ps1
%
% Features:
% - Manual test helpers (reliable fallback)
% - Auto-discovery of test modules (convenient for development)
% - test_all command (runs both manual and auto-discovered tests)

:- dynamic user:library_directory/1.
:- dynamic user:file_search_path/2.
:- dynamic user:unifyweaver_root/1.

% Load necessary library for path manipulation
:- use_module(library(filesex)).

%% ============================================================================
%% PATH SETUP
%% ============================================================================

unifyweaver_init :-
    prolog_load_context(directory, Here),
    format('Here: ~w~n',[Here]),
    retractall(user:unifyweaver_root(_)),
    assertz(user:unifyweaver_root(Here)),
    directory_file_path(Here, 'src', AbsSrcDir),
    directory_file_path(AbsSrcDir, 'unifyweaver', AbsUnifyweaverDir),
    asserta(user:library_directory(AbsSrcDir)),
    asserta(user:file_search_path(unifyweaver, AbsUnifyweaverDir)),
    format('[UnifyWeaver] Absolute paths configured:~n', []),
    format('  src: ~w~n', [AbsSrcDir]),
    format('  unifyweaver: ~w~n', [AbsUnifyweaverDir]),

    % Auto-discover tests (but don't fail if it doesn't work)
    catch(
        auto_discover_tests,
        Error,
        format('[UnifyWeaver] Auto-discovery failed (using manual tests only): ~w~n', [Error])
    ),

    % Mark as initialized
    assertz(unifyweaver_initialized),

    help.

%% ============================================================================
%% MANUAL TEST HELPERS (Hardcoded - Always Available)
%% ============================================================================

% Core module loaders
load_recursive :-
    ( use_module(unifyweaver(core/recursive_compiler))
    -> format('recursive_compiler module loaded successfully!~n', [])
    ; format('Failed to load recursive_compiler~n', [])
    ).

load_stream :-
    ( use_module(unifyweaver(core/stream_compiler))
    -> format('stream_compiler module loaded successfully!~n', [])
    ; format('Failed to load stream_compiler~n', [])
    ).

load_template :-
    ( use_module(unifyweaver(core/template_system))
    -> format('template_system module loaded successfully!~n', [])
    ; format('Failed to load template_system~n', [])
    ).

load_all_core :-
    use_module(unifyweaver(core/recursive_compiler)),
    use_module(unifyweaver(core/stream_compiler)),
    use_module(unifyweaver(core/template_system)),
    format('All core modules loaded successfully!~n', []).

% Manual test helpers (guaranteed to work)
test_stream :-
    ( use_module(unifyweaver(core/stream_compiler))
    -> test_stream_compiler
    ; format('Failed to load stream_compiler~n', [])
    ).

test_recursive :-
    ( use_module(unifyweaver(core/recursive_compiler))
    -> test_recursive_compiler
    ; format('Failed to load recursive_compiler~n', [])
    ).

test_advanced :-
    ( use_module(unifyweaver(core/advanced/test_advanced))
    -> test_all_advanced
    ; format('Failed to load advanced tests~n', [])
    ).

test_constraints :-
    ( use_module(unifyweaver(core/test_constraints))
    -> test_constraints
    ; format('Failed to load constraint tests~n', [])
    ).

%% ============================================================================
%% AUTO-DISCOVERY (Optional - Development Convenience)
%% ============================================================================

:- dynamic auto_test_helper/2.  % auto_test_helper(Name, ModulePath)

%% auto_discover_tests
%  Scan directories for test_*.pl files and create helpers
auto_discover_tests :-
    file_search_path(unifyweaver, UnifyweaverDir),
    atom_concat(UnifyweaverDir, '/core', CoreDir),

    % Discover tests in core/
    (   exists_directory(CoreDir) ->
        directory_files(CoreDir, Files),
        forall(
            (   member(File, Files),
                atom_string(File, FileStr),
                sub_string(FileStr, 0, _, _, "test_"),
                sub_string(FileStr, _, _, 0, ".pl"),
                \+ sub_string(FileStr, _, _, _, "test_advanced")  % Skip advanced (in subdirectory)
            ),
            register_auto_test(File, 'core')
        )
    ;   true
    ),

    % Discover tests in core/advanced/
    atom_concat(CoreDir, '/advanced', AdvDir),
    (   exists_directory(AdvDir) ->
        directory_files(AdvDir, AdvFiles),
        forall(
            (   member(File, AdvFiles),
                atom_string(File, FileStr),
                sub_string(FileStr, 0, _, _, "test_"),
                sub_string(FileStr, _, _, 0, ".pl")
            ),
            register_auto_test(File, 'core/advanced')
        )
    ;   true
    ),

    % Report discovered tests
    findall(Name, auto_test_helper(Name, _), Names),
    length(Names, Count),
    (   Count > 0 ->
        format('[UnifyWeaver] Auto-discovered ~w additional test modules~n', [Count])
    ;   format('[UnifyWeaver] No additional tests auto-discovered~n', [])
    ).

%% register_auto_test(+File, +SubDir)
%  Register a test file for auto-discovery
register_auto_test(File, SubDir) :-
    % Extract name from test_<name>.pl
    atom_string(File, FileStr),
    sub_string(FileStr, 5, _, 3, NameStr),  % Skip "test_" prefix and ".pl" suffix
    atom_string(Name, NameStr),

    % Build module path
    atomic_list_concat([SubDir, '/', File], '', ModuleFile),
    atom_string(ModuleFile, ModuleFileStr),
    sub_string(ModuleFileStr, 0, _, 3, ModulePathStr),  % Remove .pl
    atom_string(ModulePath, ModulePathStr),

    % Don't register if it's one of our manual tests
    \+ member(Name, [stream, recursive, advanced, constraints]),

    % Store for later use
    assertz(auto_test_helper(Name, ModulePath)),

    % Create dynamic helper predicate
    atom_concat('test_', Name, HelperName),
    atom_concat('test_', Name, TestPred),

    % Assert the helper dynamically
    assertz((
        call(HelperName) :-
            (   use_module(unifyweaver(ModulePath))
            ->  call(TestPred)
            ;   format('[ERROR] Failed to load auto-discovered test: ~w~n', [ModulePath])
            )
    )).

%% test_auto
%  Run all auto-discovered tests
test_auto :-
    findall(Name, auto_test_helper(Name, _), Names),
    (   Names = [] ->
        format('~n[INFO] No auto-discovered tests available~n', [])
    ;   format('~n╔════════════════════════════════════════╗~n', []),
        format('║  Auto-Discovered Tests                 ║~n', []),
        format('╚════════════════════════════════════════╝~n~n', []),
        length(Names, Count),
        format('[INFO] Running ~w auto-discovered tests~n~n', [Count]),
        forall(member(Name, Names),
            (   atom_concat('test_', Name, Helper),
                format('~n┌─ ~w ~50+┐~n', [Name]),
                catch(
                    call(Helper),
                    Error,
                    format('[ERROR] Test ~w failed: ~w~n', [Name, Error])
                ),
                format('└─ ~w Complete ~42+┘~n', [Name])
            )
        )
    ).

%% ============================================================================
%% TEST ALL (Manual + Auto-Discovered)
%% ============================================================================

test_all :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Running All UnifyWeaver Tests        ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Run manual tests first (most important)
    format('~n═══ Manual Tests (Core) ═══~n~n', []),

    format('┌─ Stream Compiler ~38+┐~n', []),
    catch(test_stream, E1, format('[ERROR] ~w~n', [E1])),
    format('└─ Stream Compiler Complete ~30+┘~n~n', []),

    format('┌─ Recursive Compiler ~34+┐~n', []),
    catch(test_recursive, E2, format('[ERROR] ~w~n', [E2])),
    format('└─ Recursive Compiler Complete ~26+┘~n~n', []),

    format('┌─ Advanced Recursion ~34+┐~n', []),
    catch(test_advanced, E3, format('[ERROR] ~w~n', [E3])),
    format('└─ Advanced Recursion Complete ~26+┘~n~n', []),

    format('┌─ Constraint System ~36+┐~n', []),
    catch(test_constraints, E4, format('[ERROR] ~w~n', [E4])),
    format('└─ Constraint System Complete ~28+┘~n~n', []),

    % Run auto-discovered tests (if any)
    findall(Name, auto_test_helper(Name, _), AutoNames),
    (   AutoNames \= [] ->
        format('~n═══ Auto-Discovered Tests ═══~n', []),
        test_auto
    ;   true
    ),

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All Tests Complete                    ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).

%% ============================================================================
%% HELP
%% ============================================================================

help :-
    format('~n=== UnifyWeaver Testing Help ===~n', []),
    format('~nCore Commands:~n', []),
    format(' load_stream.      - Load stream compiler~n', []),
    format(' load_recursive.   - Load recursive compiler~n', []),
    format(' load_template.    - Load template system~n', []),
    format(' load_all_core.    - Load all core modules~n', []),
    format('~nManual Tests (Always Available):~n', []),
    format(' test_stream.      - Test stream compiler~n', []),
    format(' test_recursive.   - Test recursive compiler~n', []),
    format(' test_advanced.    - Test advanced recursion~n', []),
    format(' test_constraints. - Test constraint system~n', []),
    format('~nAuto-Discovery:~n', []),
    format(' test_auto.        - Run auto-discovered tests~n', []),

    % Show auto-discovered tests if any
    findall(Name, auto_test_helper(Name, _), Names),
    (   Names \= [] ->
        format('~n Auto-discovered test modules:~n', []),
        forall(member(Name, Names),
            (   atom_concat('test_', Name, Helper),
                format('  ~w.~20+ - Test ~w~n', [Helper, Name])
            )
        )
    ;   format('  (none found - using manual tests only)~n', [])
    ),

    format('~nTest Suites:~n', []),
    format(' test_all.         - Run ALL tests (manual + auto)~n', []),
    format('~nOther:~n', []),
    format(' help.             - Show this help~n', []),
    format('~n', []).

%% ============================================================================
%% INITIALIZATION (Must be at end after all predicates defined)
%% ============================================================================

:- dynamic unifyweaver_initialized/0.
:- initialization(unifyweaver_init, now).
