:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% Codegen parity tests for WAM-to-F# transpilation.
%
% These tests assert that the generated F# source contains the expected
% identifiers, patterns, and BuiltinCall dispatch cases — they do not
% drive an `fsharpc` / `dotnet build` per case. A full .NET build per
% test would be prohibitively slow, so most runtime correctness for the
% term-inspection builtins (functor/3, arg/3, =../2, copy_term/2) is
% validated via the parallel WAM-Haskell integration tests
% (tests/test_wam_haskell_target.pl) and the WAM-Rust suite
% (tests/test_wam_rust_target.pl).
%
% Scope: WAM hybrid parity audit against the Haskell/Rust/C++ baseline
% — confirms that the F# step function exposes the same set of term
% inspection / negation-as-failure builtins as the other hybrid WAM
% targets, so the same source Prolog program compiles to a functioning
% F# WAM project.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_fsharp_target.pl

:- use_module('../src/unifyweaver/targets/wam_fsharp_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/bindings/fsharp_wam_bindings').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% ----------------------------------------------------------------------
%% Phase 5 parity: term inspection builtins (functor/3, arg/3, =../2,
%% copy_term/2). The Haskell target had these since Phase 5; the F#
%% target is being brought to parity with Rust / C++ / Go / Haskell.
%% ----------------------------------------------------------------------

test_fsharp_helper_functions_present :-
    %% The helpers live in WamTypes.fs (the type-header preamble),
    %% not in WamRuntime.fs, so we drive fsharp_wam_type_header/1 here
    %% rather than compile_wam_runtime_to_fsharp/3. This mirrors the
    %% Haskell parity test, but accounts for the F# target's module
    %% split (types module vs runtime module).
    Test = 'WAM-FSharp: term-builtin helpers generated',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        %% bindOutput: non-PC-advancing register-binding helper used
        %% by functor/3, arg/3, =../2 for output positions.
        sub_string(S, _, _, _, "bindOutput"),
        %% copyTermWalk: recursive walker for copy_term/2 that threads
        %% (counter, varMap) to preserve variable sharing.
        sub_string(S, _, _, _, "copyTermWalk"),
        sub_string(S, _, _, _, "copyTermArgs")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing bindOutput / copyTermWalk / copyTermArgs helpers')
    ).

test_fsharp_functor_builtin_present :-
    Test = 'WAM-FSharp: functor/3 step case generated',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"functor/3\""),
        %% Construct mode: allocates fresh Unbound cells.
        sub_string(S, _, _, _, "Unbound (c0 + i)"),
        %% Read mode: pattern matches Str and VList branches.
        sub_string(S, _, _, _, "Str (fn, args)")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing functor/3 step case patterns')
    ).

test_fsharp_arg_builtin_present :-
    Test = 'WAM-FSharp: arg/3 step case generated',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"arg/3\""),
        %% 1-based indexing into Str args list.
        sub_string(S, _, _, _, "List.item (idx - 1) args"),
        %% bindOutput on register A3.
        sub_string(S, _, _, _, "bindOutput 3 a s")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing arg/3 step case patterns')
    ).

test_fsharp_univ_builtin_present :-
    Test = 'WAM-FSharp: =../2 step case generated',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"=../2\""),
        %% Decompose mode: prepends functor atom to arg list.
        sub_string(S, _, _, _, "VList ((Atom fn) :: args)"),
        %% Compose mode: rebuilds Str from list head + tail.
        sub_string(S, _, _, _, "(Atom fname) :: rest -> Some (Str (fname, rest))")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing =../2 step case patterns')
    ).

test_fsharp_copy_term_builtin_present :-
    Test = 'WAM-FSharp: copy_term/2 step case generated',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"copy_term/2\""),
        %% Drives copyTermWalk with the current var counter and an
        %% empty Map (the var map scopes per call).
        sub_string(S, _, _, _, "copyTermWalk s.WsVarCounter Map.empty tVal")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing copy_term/2 step case patterns')
    ).

%% ----------------------------------------------------------------------
%% Phase 4.4 parity: negation-as-failure (\+/1) in the step function.
%% Mirrors the Haskell baseline (member fast-path + true/fail fast-path
%% + general goal-snapshot path). F# Atom is string-based (no interned
%% atom table at the runtime level), so the implementation uses string
%% comparisons directly.
%% ----------------------------------------------------------------------

test_fsharp_negation_step_handler_present :-
    Test = 'WAM-FSharp: \\+/1 step case generated',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        %% The runtime pattern uses an F# string literal "\\+/1" which
        %% contains exactly one backslash at runtime — matching the
        %% interned op name produced by the WAM compiler.
        sub_string(S, _, _, _, "BuiltinCall (\"\\\\+/1\""),
        %% Member fast path inline list walk.
        sub_string(S, _, _, _, "Str (\"member\", [needle; haystack])"),
        %% true / fail fast paths.
        sub_string(S, _, _, _, "Atom \"true\""),
        sub_string(S, _, _, _, "Atom \"fail\""),
        %% General path resolves the goal label via WcLabels and
        %% executes the snapshot through the run loop.
        sub_string(S, _, _, _, "Map.tryFind goalKey ctx.WcLabels"),
        sub_string(S, _, _, _, "run ctx snapshot")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing \\+/1 step case patterns')
    ).

test_fsharp_no_regressions :-
    %% Sanity that adding the term-builtins did not break the
    %% pre-existing builtins (is/2, length/2, member/2, comparison).
    Test = 'WAM-FSharp: pre-existing builtins still present',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"!/0\""),
        sub_string(S, _, _, _, "BuiltinCall (\"is/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"length/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"member/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"</2\""),
        sub_string(S, _, _, _, "BuiltinCall (\">/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"nonvar/1\""),
        sub_string(S, _, _, _, "BuiltinCall (\"var/1\""),
        sub_string(S, _, _, _, "BuiltinCall (\"atom/1\""),
        sub_string(S, _, _, _, "BuiltinCall (\"integer/1\""),
        sub_string(S, _, _, _, "BuiltinCall (\"number/1\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Pre-existing builtin dispatch cases missing')
    ).

%% ----------------------------------------------------------------------
%% Phase B parity: arithmetic comparison operators (>=, =<, =:=, =\=)
%% present in the Rust target. The Haskell baseline currently has only
%% < and >, so this brings the F# target one step further toward the
%% Rust/C++ comparison-builtin coverage.
%% ----------------------------------------------------------------------

test_fsharp_arith_comparison_builtins :-
    Test = 'WAM-FSharp: arithmetic comparisons (>=, =<, =:=, =\\=)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\">=/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"=</2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"=:=/2\""),
        %% =\=/2 emits as F# pattern "=\\=/2" (one backslash at F# runtime).
        sub_string(S, _, _, _, "BuiltinCall (\"=\\\\=/2\""),
        %% EPSILON tolerance bridges integer/float comparisons.
        sub_string(S, _, _, _, "System.Double.Epsilon")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing arithmetic comparison step cases')
    ).

%% ----------------------------------------------------------------------
%% Term equality (==, \\==): structural equality on the dereferenced
%% value, no unification or binding. Mirrors the Rust execute_arith ==/2
%% case and the C++ ==/2 / \\==/2 fused case.
%% ----------------------------------------------------------------------

test_fsharp_term_equality_builtins :-
    Test = 'WAM-FSharp: term equality (==/2, \\==/2)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"==/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"\\\\==/2\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing term equality / inequality step cases')
    ).

%% ----------------------------------------------------------------------
%% Trivial control: true/0 (succeed + advance) and fail/0 (always fail).
%% These show up in source Prolog programs as call(true) / call(fail)
%% and via aggregate-clause guards.  Rust has them as the first arms of
%% execute_control.
%% ----------------------------------------------------------------------

test_fsharp_trivial_control_builtins :-
    Test = 'WAM-FSharp: trivial control (true/0, fail/0)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"true/0\""),
        sub_string(S, _, _, _, "BuiltinCall (\"fail/0\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing trivial control step cases')
    ).

%% ----------------------------------------------------------------------
%% Type checks: compound/1, float/1, is_list/1. Brings the F# type-check
%% set to parity with the Rust execute_type_builtin coverage. atom/1,
%% integer/1, number/1, var/1, nonvar/1 were already present.
%% ----------------------------------------------------------------------

test_fsharp_type_check_builtins :-
    Test = 'WAM-FSharp: type checks (compound/1, float/1, is_list/1)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"compound/1\""),
        %% compound matches Str OR non-empty VList.
        sub_string(S, _, _, _, "Some (Str _)"),
        sub_string(S, _, _, _, "Some (VList (_::_))"),
        sub_string(S, _, _, _, "BuiltinCall (\"float/1\""),
        sub_string(S, _, _, _, "BuiltinCall (\"is_list/1\""),
        %% is_list accepts the empty-list atom convention too.
        sub_string(S, _, _, _, "Some (Atom \"[]\")")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing type check step cases')
    ).

%% ----------------------------------------------------------------------
%% I/O builtins: write/1, display/1, nl/0. F# uses sprintf / printfn for
%% Display output. Mirrors the Rust execute_io_builtin set.
%% ----------------------------------------------------------------------

test_fsharp_io_builtins :-
    Test = 'WAM-FSharp: I/O (write/1, display/1, nl/0)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"write/1\""),
        sub_string(S, _, _, _, "BuiltinCall (\"display/1\""),
        sub_string(S, _, _, _, "BuiltinCall (\"nl/0\""),
        sub_string(S, _, _, _, "printfn \"\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing I/O step cases')
    ).

%% ----------------------------------------------------------------------
%% Phase G parity: atom / string builtins (parity with the Go target).
%% Brings the F# step function to coverage parity with the Go target''s
%% atom_*, char_code, upcase_atom, downcase_atom, atom_string,
%% string_to_atom, atom_number, and succ builtins.
%% Reference: templates/targets/go_wam/state.go.mustache.
%% ----------------------------------------------------------------------

test_fsharp_atom_concat_builtin :-
    Test = 'WAM-FSharp: atom_concat/3',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"atom_concat/3\""),
        %% .NET string concatenation, both A1 and A2 must be Atoms.
        sub_string(S, _, _, _, "Atom (a + b)"),
        sub_string(S, _, _, _, "bindOutput 3 (Atom (a + b))")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing atom_concat/3 step case')
    ).

test_fsharp_atom_length_builtin :-
    Test = 'WAM-FSharp: atom_length/2 + string_length/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        %% Both names alias to the same OR-pattern arm in the step case.
        sub_string(S, _, _, _, "BuiltinCall (\"atom_length/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"string_length/2\""),
        sub_string(S, _, _, _, "Integer a.Length")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing atom_length/2 / string_length/2 step case')
    ).

test_fsharp_char_code_builtin :-
    Test = 'WAM-FSharp: char_code/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"char_code/2\""),
        %% Forward: single-char Atom -> Integer code via int a.[0].
        sub_string(S, _, _, _, "Integer (int a.[0])"),
        %% Reverse: Integer code -> single-char Atom via string (char c).
        sub_string(S, _, _, _, "Atom (string (char c))"),
        %% BMP guard to keep parity with the Go target''s 0..65535 range.
        sub_string(S, _, _, _, "c >= 0 && c <= 65535")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing char_code/2 step case patterns')
    ).

test_fsharp_atom_codes_builtin :-
    Test = 'WAM-FSharp: atom_codes/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"atom_codes/2\""),
        %% Forward: Atom -> VList of Integer code points.
        sub_string(S, _, _, _, "Integer (int c)"),
        sub_string(S, _, _, _, "VList codes"),
        %% Reverse: VList of Integer -> Atom via StringBuilder fold.
        sub_string(S, _, _, _, "System.Text.StringBuilder")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing atom_codes/2 step case patterns')
    ).

test_fsharp_atom_chars_builtin :-
    Test = 'WAM-FSharp: atom_chars/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"atom_chars/2\""),
        %% Forward: Atom -> VList of single-char Atom values.
        sub_string(S, _, _, _, "Atom (string c)"),
        sub_string(S, _, _, _, "VList chars")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing atom_chars/2 step case patterns')
    ).

test_fsharp_atom_string_aliases :-
    Test = 'WAM-FSharp: atom_string/2 + string_to_atom/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"atom_string/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"string_to_atom/2\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing atom_string/2 / string_to_atom/2 step cases')
    ).

test_fsharp_case_conversion_builtins :-
    Test = 'WAM-FSharp: upcase_atom/2 + downcase_atom/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"upcase_atom/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"downcase_atom/2\""),
        %% .NET invariant-culture transforms (Unicode-safe).
        sub_string(S, _, _, _, "a.ToUpperInvariant()"),
        sub_string(S, _, _, _, "a.ToLowerInvariant()")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing upcase / downcase step cases')
    ).

test_fsharp_atom_number_builtin :-
    Test = 'WAM-FSharp: atom_number/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"atom_number/2\""),
        %% Integer parse first, then Float fallback (invariant culture).
        sub_string(S, _, _, _, "System.Int64.TryParse"),
        sub_string(S, _, _, _, "System.Double.TryParse"),
        sub_string(S, _, _, _, "System.Globalization.CultureInfo.InvariantCulture")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing atom_number/2 step case patterns')
    ).

test_fsharp_succ_builtin :-
    Test = 'WAM-FSharp: succ/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"succ/2\""),
        %% Forward: Integer x (x >= 0) -> Integer (x + 1).
        sub_string(S, _, _, _, "Integer (x + 1)"),
        %% Reverse: Integer y (y > 0) -> Integer (y - 1).
        sub_string(S, _, _, _, "Integer (y - 1)")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing succ/2 step case patterns')
    ).

%% ----------------------------------------------------------------------
%% Phase F parity smoke: fact-shape classification helpers exposed by
%% the F# target (parity infra used by Haskell / Elixir hybrid targets).
%% ----------------------------------------------------------------------

test_fsharp_fact_shape_helpers_exported :-
    Test = 'WAM-FSharp: fact shape classification exports present',
    (   catch((
            current_predicate(wam_fsharp_target:classify_fact_predicate_fs/4),
            current_predicate(wam_fsharp_target:fsharp_fact_only/2),
            current_predicate(wam_fsharp_target:fsharp_first_arg_groundness/3),
            current_predicate(wam_fsharp_target:fsharp_pick_layout/5),
            current_predicate(wam_fsharp_target:split_wam_into_segments_fs/2)
        ), _, fail)
    ->  pass(Test)
    ;   fail_test(Test, 'classify_fact_predicate_fs / fsharp_fact_only / ... missing')
    ).

test_fsharp_emit_mode_resolution :-
    %% Default emit mode must remain the safer interpreter mode (matches
    %% the Haskell baseline). functions(...) / mixed(...) should round-trip.
    Test = 'WAM-FSharp: emit mode resolution parity',
    (   wam_fsharp_resolve_emit_mode([], Default),
        Default == interpreter,
        wam_fsharp_resolve_emit_mode([emit_mode(functions)], Funcs),
        Funcs == functions,
        wam_fsharp_resolve_emit_mode([emit_mode(mixed([foo/1]))], Mixed),
        Mixed == mixed([foo/1])
    ->  pass(Test)
    ;   fail_test(Test, 'emit_mode hierarchy or values mismatch parity')
    ).

%% ----------------------------------------------------------------------
%% Phase 3 parity: lowered emitter end-to-end smoke. Confirms the
%% F# lowered emitter accepts the same instruction whitelist as the
%% Haskell / Rust / Clojure lowered emitters for deterministic
%% single-clause bodies.
%% ----------------------------------------------------------------------

test_fsharp_lowerable_single_clause_proceed :-
    %% Smallest deterministic body: just `proceed`. All lowered
    %% emitters in the audit accept this.
    Test = 'WAM-FSharp: lowerable trivial single-clause body',
    Wam = 'p/0:\n  proceed',
    (   wam_fsharp_lowerable(p/0, Wam, _Reason)
    ->  pass(Test)
    ;   fail_test(Test, 'Trivial proceed-only body should be lowerable')
    ).

test_fsharp_lowerable_rejects_aggregate :-
    %% The lowered F# emitter explicitly excludes begin_aggregate /
    %% end_aggregate (see wam_fsharp_lowered_emitter.pl supported_fs/1
    %% header comment) because aggregate collection needs the run loop
    %% for backtrack-driven traversal. Matches the Haskell baseline.
    Test = 'WAM-FSharp: lowerable rejects aggregate instructions',
    Wam = 'p/2:\n  begin_aggregate sum, A1, A2\n  proceed\n  end_aggregate A1\n  proceed',
    (   \+ wam_fsharp_lowerable(p/2, Wam, _Reason)
    ->  pass(Test)
    ;   fail_test(Test, 'Body containing begin_aggregate should not be lowerable')
    ).

test_fsharp_lowerable_multi_clause :-
    %% Multi-clause bodies ARE lowerable in F# (clause 1 lowered, clauses
    %% 2+ stay in interpreter, reached via backtrack from the lowered
    %% function — see wam_fsharp_target.pl:68-74 header comment).
    %% This matches the Haskell baseline, and is the expected behavior
    %% for the F# / Haskell / Clojure family.
    Test = 'WAM-FSharp: lowerable accepts multi-clause via clause-1 lowering',
    Wam = 'p/1:\n  try_me_else L1\n  get_constant a, A1\n  proceed\nL1:\n  trust_me\n  get_constant b, A1\n  proceed',
    (   wam_fsharp_lowerable(p/1, Wam, _Reason)
    ->  pass(Test)
    ;   fail_test(Test, 'Multi-clause body should be lowerable (clause 1 + backtrack fallback)')
    ).

%% ----------------------------------------------------------------------
%% Project generation smoke: exercise write_wam_fsharp_project/3 in a
%% throwaway directory with no kernels and assert the expected files
%% are produced. Does not invoke `dotnet build`.
%% ----------------------------------------------------------------------

test_fsharp_project_generation_smoke :-
    Test = 'WAM-FSharp: write_wam_fsharp_project produces expected files',
    setup_call_cleanup(
        tmp_dir_fs(ProjectDir),
        (   write_wam_fsharp_project([], [no_kernels(true), module_name('parity_smoke')], ProjectDir),
            directory_file_path(ProjectDir, 'WamTypes.fs',  TypesP),
            directory_file_path(ProjectDir, 'WamRuntime.fs', RunP),
            directory_file_path(ProjectDir, 'Predicates.fs', PredsP),
            directory_file_path(ProjectDir, 'Lowered.fs',    LoweredP),
            directory_file_path(ProjectDir, 'Program.fs',    ProgP),
            directory_file_path(ProjectDir, 'parity_smoke.fsproj', FsprojP),
            (   exists_file(TypesP),
                exists_file(RunP),
                exists_file(PredsP),
                exists_file(LoweredP),
                exists_file(ProgP),
                exists_file(FsprojP)
            ->  pass(Test)
            ;   fail_test(Test, 'One or more generated project files missing')
            )
        ),
        cleanup_dir_fs(ProjectDir)
    ).

tmp_dir_fs(Dir) :-
    tmp_file(wam_fsharp_parity, Base),
    atom_concat(Base, '_dir', Dir).

cleanup_dir_fs(Dir) :-
    (   exists_directory(Dir)
    ->  delete_directory_and_contents_fs(Dir)
    ;   true
    ).

delete_directory_and_contents_fs(Dir) :-
    directory_files(Dir, Files),
    forall(
        (   member(F, Files),
            F \== '.', F \== '..',
            directory_file_path(Dir, F, Path)
        ),
        (   exists_directory(Path)
        ->  delete_directory_and_contents_fs(Path)
        ;   catch(delete_file(Path), _, true)
        )
    ),
    catch(delete_directory(Dir), _, true).

%% ----------------------------------------------------------------------
%% Test runner
%% ----------------------------------------------------------------------

run_tests :-
    format('~n========================================~n'),
    format('WAM-FSharp parity tests (vs Haskell/Rust/C++)~n'),
    format('========================================~n~n'),
    test_fsharp_helper_functions_present,
    test_fsharp_functor_builtin_present,
    test_fsharp_arg_builtin_present,
    test_fsharp_univ_builtin_present,
    test_fsharp_copy_term_builtin_present,
    test_fsharp_negation_step_handler_present,
    test_fsharp_no_regressions,
    test_fsharp_arith_comparison_builtins,
    test_fsharp_term_equality_builtins,
    test_fsharp_trivial_control_builtins,
    test_fsharp_type_check_builtins,
    test_fsharp_io_builtins,
    test_fsharp_atom_concat_builtin,
    test_fsharp_atom_length_builtin,
    test_fsharp_char_code_builtin,
    test_fsharp_atom_codes_builtin,
    test_fsharp_atom_chars_builtin,
    test_fsharp_atom_string_aliases,
    test_fsharp_case_conversion_builtins,
    test_fsharp_atom_number_builtin,
    test_fsharp_succ_builtin,
    test_fsharp_fact_shape_helpers_exported,
    test_fsharp_emit_mode_resolution,
    test_fsharp_lowerable_single_clause_proceed,
    test_fsharp_lowerable_rejects_aggregate,
    test_fsharp_lowerable_multi_clause,
    test_fsharp_project_generation_smoke,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).
