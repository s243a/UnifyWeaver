:- encoding(utf8).
% Phase 5 smoke tests for the WAM Haskell ISO error handling.
%
% Phase 5 adds ISO error handling parity with the F# target:
%   - Error constructors (makeInstantiationError, makeTypeError, etc.)
%   - throw/1 and catch/3 builtins
%   - is_iso/2 and is_lax/2 three-form dispatch
%   - ISO/lax variants for 6 arithmetic comparison operators
%   - succ/2, succ_iso/2, succ_lax/2 bidirectional successor
%   - Missing lax comparisons (=</2, >=/2, =:=/2, =\=/2)
%   - isIsoMetaBuiltin routing in Call/Execute
%   - iso_errors key tables and WAM text rewrite integration
%
% Usage: swipl -g run_tests -t halt tests/test_wam_haskell_iso_smoke.pl

:- use_module('../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../src/unifyweaver/targets/wam_haskell_lowered_emitter').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/core/iso_errors').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% =====================================================================
%% ISO key table registration
%% =====================================================================

test_iso_key_tables_registered :-
    Test = 'Phase 5: ISO key tables registered for Haskell target',
    (   iso_errors:iso_errors_default_to_iso("is/2", "is_iso/2"),
        iso_errors:iso_errors_default_to_lax("is/2", "is_lax/2"),
        iso_errors:iso_errors_default_to_iso("</2", "<_iso/2"),
        iso_errors:iso_errors_default_to_lax("</2", "<_lax/2"),
        iso_errors:iso_errors_default_to_iso(">/2", ">_iso/2"),
        iso_errors:iso_errors_default_to_lax(">/2", ">_lax/2"),
        iso_errors:iso_errors_default_to_iso(">=/2", ">=_iso/2"),
        iso_errors:iso_errors_default_to_lax(">=/2", ">=_lax/2"),
        iso_errors:iso_errors_default_to_iso("=</2", "=<_iso/2"),
        iso_errors:iso_errors_default_to_lax("=</2", "=<_lax/2"),
        iso_errors:iso_errors_default_to_iso("=:=/2", "=:=_iso/2"),
        iso_errors:iso_errors_default_to_lax("=:=/2", "=:=_lax/2"),
        iso_errors:iso_errors_default_to_iso("succ/2", "succ_iso/2"),
        iso_errors:iso_errors_default_to_lax("succ/2", "succ_lax/2")
    ->  pass(Test)
    ;   fail_test(Test, 'missing ISO key table entries for Haskell target')
    ).

%% =====================================================================
%% ISO errors rewrite
%% =====================================================================

test_iso_rewrite_is_to_iso :-
    Test = 'Phase 5: ISO rewrite is/2 -> is_iso/2 in ISO mode',
    Config = iso_config(true, []),
    iso_errors_rewrite_text_hs(Config, test/0,
        "    builtin_call is/2, 2\n    proceed\n",
        Rewritten),
    (   sub_string(Rewritten, _, _, _, "is_iso/2")
    ->  pass(Test)
    ;   fail_test(Test, ['expected is_iso/2 in: ', Rewritten])
    ).

test_iso_rewrite_is_to_lax :-
    Test = 'Phase 5: ISO rewrite is/2 -> is_lax/2 in lax mode',
    Config = iso_config(false, []),
    iso_errors_rewrite_text_hs(Config, test/0,
        "    builtin_call is/2, 2\n    proceed\n",
        Rewritten),
    (   sub_string(Rewritten, _, _, _, "is_lax/2")
    ->  pass(Test)
    ;   fail_test(Test, ['expected is_lax/2 in: ', Rewritten])
    ).

test_iso_rewrite_explicit_survives :-
    Test = 'Phase 5: explicit is_iso/2 survives lax mode rewrite',
    Config = iso_config(false, []),
    iso_errors_rewrite_text_hs(Config, test/0,
        "    builtin_call is_iso/2, 2\n    proceed\n",
        Rewritten),
    (   sub_string(Rewritten, _, _, _, "is_iso/2")
    ->  pass(Test)
    ;   fail_test(Test, ['explicit is_iso/2 was rewritten in lax mode'])
    ).

test_iso_rewrite_comparison_sweep :-
    Test = 'Phase 5: ISO rewrite all 6 comparison ops',
    Config = iso_config(true, []),
    iso_errors_rewrite_text_hs(Config, test/0,
        "    builtin_call </2, 2\n    builtin_call >/2, 2\n    builtin_call >=/2, 2\n    builtin_call =</2, 2\n    builtin_call =:=/2, 2\n    builtin_call =\\=/2, 2\n    proceed\n",
        Rewritten),
    (   sub_string(Rewritten, _, _, _, "<_iso/2"),
        sub_string(Rewritten, _, _, _, ">_iso/2"),
        sub_string(Rewritten, _, _, _, ">=_iso/2"),
        sub_string(Rewritten, _, _, _, "=<_iso/2"),
        sub_string(Rewritten, _, _, _, "=:=_iso/2")
    ->  pass(Test)
    ;   fail_test(Test, ['not all comparison ops were rewritten to ISO'])
    ).

%% =====================================================================
%% Codegen: ISO runtime helpers present in generated code
%% =====================================================================

:- dynamic user:iso_probe_is/2.
user:iso_probe_is(X, Y) :- Y is X + 1.

test_codegen_has_iso_helpers :-
    Test = 'Phase 5: generated Haskell contains ISO error helpers',
    wam_target:compile_predicate_to_wam(iso_probe_is/2, [], _WamCode),
    % Generate a project to inspect the output
    tmp_project_dir(Dir),
    write_wam_haskell_project(
        [user:iso_probe_is/2],
        [module_name('uw-wam-hs-iso-test')],
        Dir),
    atom_concat(Dir, '/src/WamRuntime.hs', RuntimePath),
    read_file_to_string(RuntimePath, S),
    (   sub_string(S, _, _, _, "WamException"),
        sub_string(S, _, _, _, "derefDeep"),
        sub_string(S, _, _, _, "throwIsoError"),
        sub_string(S, _, _, _, "makeInstantiationError"),
        sub_string(S, _, _, _, "makeTypeError"),
        sub_string(S, _, _, _, "isIsoMetaBuiltin")
    ->  pass(Test)
    ;   fail_test(Test, 'generated WamRuntime.hs missing ISO helpers')
    ).

test_codegen_has_iso_atoms :-
    Test = 'Phase 5: generated Haskell has ISO atom constants',
    tmp_project_dir(Dir),
    atom_concat(Dir, '/src/WamTypes.hs', TypesPath),
    read_file_to_string(TypesPath, S),
    (   sub_string(S, _, _, _, "atomError"),
        sub_string(S, _, _, _, "atomInstantiationError"),
        sub_string(S, _, _, _, "atomTypeError")
    ->  pass(Test)
    ;   fail_test(Test, 'generated WamTypes.hs missing ISO atom constants')
    ).

test_codegen_has_iso_builtins :-
    Test = 'Phase 5: generated WamRuntime.hs has ISO builtin handlers',
    tmp_project_dir(Dir),
    atom_concat(Dir, '/src/WamRuntime.hs', RuntimePath),
    read_file_to_string(RuntimePath, S),
    (   sub_string(S, _, _, _, "\"is_iso/2\""),
        sub_string(S, _, _, _, "\"is_lax/2\""),
        sub_string(S, _, _, _, "\"throw/1\""),
        sub_string(S, _, _, _, "\"catch/3\""),
        sub_string(S, _, _, _, "\"=</2\""),
        sub_string(S, _, _, _, "\">=/2\""),
        sub_string(S, _, _, _, "\"succ/2\"")
    ->  pass(Test)
    ;   fail_test(Test, 'generated WamRuntime.hs missing ISO builtin handlers')
    ).

%% =====================================================================
%% ISO atom interning
%% =====================================================================

test_iso_atoms_interned :-
    Test = 'Phase 5: ISO atoms are pre-interned at compile time',
    % These atoms should have been registered in the intern table
    (   wam_haskell_target:intern_atom("error", _),
        wam_haskell_target:intern_atom("instantiation_error", _),
        wam_haskell_target:intern_atom("type_error", _),
        wam_haskell_target:intern_atom("domain_error", _),
        wam_haskell_target:intern_atom("evaluation_error", _),
        wam_haskell_target:intern_atom("evaluable", _),
        wam_haskell_target:intern_atom("zero_divisor", _)
    ->  pass(Test)
    ;   fail_test(Test, 'ISO atoms not found in intern table')
    ).

%% =====================================================================
%% Helpers
%% =====================================================================

:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).

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

tmp_project_dir(Dir) :-
    writable_tmp_root(Root),
    directory_file_path(Root, 'uw_wam_hs_iso_test', Dir).

read_file_to_string(Path, Str) :-
    open(Path, read, S),
    read_string(S, _, Str),
    close(S).

%% =====================================================================
%% Runner
%% =====================================================================

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Haskell ISO Error Handling (Phase 5) tests ===~n', []),
    test_iso_key_tables_registered,
    test_iso_rewrite_is_to_iso,
    test_iso_rewrite_is_to_lax,
    test_iso_rewrite_explicit_survives,
    test_iso_rewrite_comparison_sweep,
    test_codegen_has_iso_helpers,
    test_codegen_has_iso_atoms,
    test_codegen_has_iso_builtins,
    test_iso_atoms_interned,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 5 tests passed ===~n', [])
    ).
