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
%% Phase H parity: list / sort / order / unification builtins.  Brings
%% the F# target to the remaining Go-target builtin coverage (append/3,
%% reverse/2, last/2, nth0/3, nth1/3, memberchk/2, delete/3, select/3,
%% numlist/3, sort/2, msort/2, compare/3, @</2 family, =/2, \\=/2).
%% Reference: templates/targets/go_wam/state.go.mustache.
%% ----------------------------------------------------------------------

test_fsharp_compare_value_helper_present :-
    %% compareValue + compareValueList live in WamTypes.fs (the type
    %% header preamble), invoked by compare/3, the @ comparisons, and
    %% sort/2 / msort/2. The helper implements Prolog-standard term
    %% ordering, which is NOT the same as F#''s default structural
    %% comparison (F# orders DU variants by declaration position).
    Test = 'WAM-FSharp: compareValue helper present in type header',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "let rec compareValue"),
        sub_string(S, _, _, _, "compareValueList"),
        %% Standard ordering: Var < Number < Atom < Compound.
        sub_string(S, _, _, _, "| Unbound _           -> 0"),
        sub_string(S, _, _, _, "| Atom _              -> 2"),
        %% Numeric mixing via float-promotion.
        sub_string(S, _, _, _, "| Integer x, Float y   -> compare (float x) y")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing compareValue / compareValueList helpers')
    ).

test_fsharp_value_du_no_buggy_overrides :-
    %% Regression guard for the Value DU cleanup. The earlier definition
    %% carried [<CustomEquality; CustomComparison>] plus an Equals /
    %% CompareTo override that recursed through `this = other` /
    %% `compare this other` — both would stack-overflow if exercised on
    %% the hot path. Removing them lets F# auto-generate proper
    %% structural equality + comparison for the DU.
    %%
    %% Checks for the actual F# syntax that would re-introduce the bug,
    %% not just keyword mentions (the cleanup commentary in compareValue''s
    %% docstring intentionally references the removed pattern by name).
    Test = 'WAM-FSharp: Value DU has no recursive equality / comparison override',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        %% The interface block opener is gone.
        \+ sub_string(S, _, _, _, "interface System.IComparable with"),
        %% The CompareTo / Equals member declarations are gone.
        \+ sub_string(S, _, _, _, "member this.CompareTo(obj)"),
        \+ sub_string(S, _, _, _, "override this.Equals(obj)"),
        \+ sub_string(S, _, _, _, "override this.GetHashCode()"),
        %% Value DU itself still emitted.
        sub_string(S, _, _, _, "type Value =")
    ->  pass(Test)
    ;   fail_test(Test, 'Buggy override or attribute still present on Value DU')
    ).

test_fsharp_append_builtin :-
    Test = 'WAM-FSharp: append/3',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"append/3\""),
        %% F# list concatenation operator on the two VList contents.
        sub_string(S, _, _, _, "VList (xs @ ys)")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing append/3 step case')
    ).

test_fsharp_reverse_builtin :-
    Test = 'WAM-FSharp: reverse/2 (bidirectional)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"reverse/2\""),
        sub_string(S, _, _, _, "List.rev items")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing reverse/2 step case')
    ).

test_fsharp_last_builtin :-
    Test = 'WAM-FSharp: last/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"last/2\""),
        %% Non-empty list guard.
        sub_string(S, _, _, _, "not (List.isEmpty items)"),
        sub_string(S, _, _, _, "List.last items")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing last/2 step case')
    ).

test_fsharp_nth_builtins :-
    Test = 'WAM-FSharp: nth0/3 + nth1/3',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"nth0/3\""),
        sub_string(S, _, _, _, "BuiltinCall (\"nth1/3\""),
        %% Shared dispatch with a base offset derived from the op name.
        sub_string(S, _, _, _, "List.item idx items")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing nth0/3 / nth1/3 step case')
    ).

test_fsharp_memberchk_builtin :-
    Test = 'WAM-FSharp: memberchk/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"memberchk/2\""),
        %% Deterministic: no choice point, just List.exists.
        sub_string(S, _, _, _, "items |> List.exists"),
        sub_string(S, _, _, _, "derefVar s.WsBindings v = elem_")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing memberchk/2 step case')
    ).

test_fsharp_delete_builtin :-
    Test = 'WAM-FSharp: delete/3',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"delete/3\""),
        %% Removes ALL occurrences via List.filter.
        sub_string(S, _, _, _, "List.filter")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing delete/3 step case')
    ).

test_fsharp_select_builtin :-
    %% select/3 now enumerates all solutions via SelectRetry choice
    %% points (parity with the Go target's SelectResults).
    Test = 'WAM-FSharp: select/3 (full backtracking)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"select/3\""),
        %% Computes all (selected, rest) splits upfront.
        sub_string(S, _, _, _, "let rec splits prefix rest ="),
        %% Pushes SelectRetry CP with the remaining candidates.
        sub_string(S, _, _, _, "Some (SelectRetry (1, 3, more, retPC))")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing select/3 backtracking patterns')
    ).

test_fsharp_select_retry_builtin_state :-
    %% SelectRetry must be declared in the BuiltinState DU alongside
    %% FactRetry / HopsRetry / FFIStreamRetry.
    Test = 'WAM-FSharp: SelectRetry arm in BuiltinState DU',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| SelectRetry"),
        %% Carries elemReg, outReg, remaining (Value, Value list) pairs,
        %% and retPC.
        sub_string(S, _, _, _, "elemReg: int * outReg: int * remaining: (Value * Value list) list * retPC: int")
    ->  pass(Test)
    ;   fail_test(Test, 'SelectRetry arm missing from BuiltinState')
    ).

test_fsharp_select_retry_resume_handler :-
    %% resumeBuiltin must dispatch on SelectRetry and restore the CP
    %% snapshot before trying the next candidate.
    Test = 'WAM-FSharp: resumeBuiltin handles SelectRetry',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| SelectRetry (_, _, [], _) ->"),
        sub_string(S, _, _, _, "| SelectRetry (elemReg, outReg, candidates, retPC) ->"),
        %% Snapshot restoration before unification.
        sub_string(S, _, _, _, "WsBindings = cp.CpBindings"),
        %% Iterates candidates and chains via tryNext.
        sub_string(S, _, _, _, "let rec tryNext pairs =")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing SelectRetry resume handler patterns')
    ).

test_fsharp_numlist_builtin :-
    Test = 'WAM-FSharp: numlist/3',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"numlist/3\""),
        %% Range comprehension generates the Integer list.
        sub_string(S, _, _, _, "for n in lo .. hi -> Integer n")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing numlist/3 step case')
    ).

test_fsharp_sort_msort_builtins :-
    Test = 'WAM-FSharp: sort/2 + msort/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"sort/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"msort/2\""),
        %% Both call List.sortWith compareValue.
        sub_string(S, _, _, _, "List.sortWith compareValue"),
        %% sort/2 has a dedup pass; msort/2 does not.
        sub_string(S, _, _, _, "let rec dedup")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing sort/2 / msort/2 step cases')
    ).

test_fsharp_compare_builtin :-
    Test = 'WAM-FSharp: compare/3',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"compare/3\""),
        %% Result is an Atom dispatching on the sign of compareValue.
        sub_string(S, _, _, _, "if c < 0 then Atom \"<\" elif c > 0 then Atom \">\" else Atom \"=\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing compare/3 step case')
    ).

test_fsharp_standard_order_comparison :-
    Test = 'WAM-FSharp: @</2, @=</2, @>/2, @>=/2',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"@</2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"@=</2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"@>/2\""),
        sub_string(S, _, _, _, "BuiltinCall (\"@>=/2\""),
        %% All four call compareValue.
        sub_string(S, _, _, _, "compareValue a b < 0"),
        sub_string(S, _, _, _, "compareValue a b <= 0"),
        sub_string(S, _, _, _, "compareValue a b > 0"),
        sub_string(S, _, _, _, "compareValue a b >= 0")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing @ comparison step cases')
    ).

test_fsharp_unify_builtin :-
    Test = 'WAM-FSharp: =/2 (explicit unification)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall (\"=/2\""),
        %% Delegates to the local unifyVal helper that lives at the
        %% end of the step function (binding-trail aware).
        sub_string(S, _, _, _, "unifyVal a b s")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing =/2 step case')
    ).

test_fsharp_not_unifiable_builtin :-
    Test = 'WAM-FSharp: \\=/2 (non-unifiable)',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        %% F# pattern emits "\\=/2" (one backslash at runtime),
        %% matching the WAM op name.
        sub_string(S, _, _, _, "BuiltinCall (\"\\\\=/2\""),
        %% Inverted-unify pattern: Some _ (unified) => None (fail).
        sub_string(S, _, _, _, "| Some _ -> None")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing \\=/2 step case')
    ).

%% ----------------------------------------------------------------------
%% Phase I parity: Haskell-only specialized instructions ported to F#.
%% These are performance optimizations emitted by the WAM compiler''s
%% binding-analysis pass; their semantics live in step (parity with
%% src/unifyweaver/targets/wam_haskell_target.pl).
%% ----------------------------------------------------------------------

test_fsharp_vset_value_variant :-
    %% F# Value DU gains a VSet of Set<string> variant. Haskell uses
    %% IS.IntSet of interned atom IDs; F# uses Set<string> directly
    %% because F# atoms are string-based at the runtime level.
    Test = 'WAM-FSharp: VSet variant on Value DU',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| VSet    of Set<string>"),
        %% compareValue treats VSet at compound rank.
        sub_string(S, _, _, _, "| VSet _              -> 3"),
        sub_string(S, _, _, _, "| VSet x,    VSet y    -> compare x y")
    ->  pass(Test)
    ;   fail_test(Test, 'VSet variant missing or compareValue not extended')
    ).

test_fsharp_specialized_instructions_declared :-
    %% The 7 new Phase-I instructions must be declared in the
    %% Instruction DU.
    Test = 'WAM-FSharp: Phase-I instructions in Instruction DU',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| PutStructureDyn       of nameReg: int * arityReg: int * targetReg: int"),
        sub_string(S, _, _, _, "| Arg                   of n: int * tReg: int * aReg: int"),
        sub_string(S, _, _, _, "| NotMemberList         of xReg: int * lReg: int"),
        sub_string(S, _, _, _, "| NotMemberConstAtoms   of xReg: int * atoms: string list"),
        sub_string(S, _, _, _, "| BuildEmptySet         of reg: int"),
        sub_string(S, _, _, _, "| SetInsert             of elemReg: int * inReg: int * outReg: int"),
        sub_string(S, _, _, _, "| NotMemberSet          of elemReg: int * setReg: int")
    ->  pass(Test)
    ;   fail_test(Test, 'One or more Phase-I instructions missing from DU')
    ).

test_fsharp_put_structure_dyn_step :-
    Test = 'WAM-FSharp: PutStructureDyn step case',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| PutStructureDyn (nameReg, arityReg, targetReg) ->"),
        %% Functor name must dereference to Atom; arity to non-neg Integer.
        sub_string(S, _, _, _, "Some (Atom fname), Some (Integer arity) when arity >= 0"),
        %% Pushes a BuildStruct into the WamBuilder slot at targetReg.
        sub_string(S, _, _, _, "BuildStruct (fname, targetReg, arity, [])")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing PutStructureDyn step case')
    ).

test_fsharp_arg_specialized_step :-
    Test = 'WAM-FSharp: Arg (specialized arg/3) step case',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| Arg (n, tReg, aReg) when n >= 1 ->"),
        %% Pattern-matches Str / VList for the subterm.
        sub_string(S, _, _, _, "Str (_, args) when n <= List.length args -> Some (List.item (n - 1) args)"),
        sub_string(S, _, _, _, "VList (x :: _)  when n = 1 -> Some x"),
        sub_string(S, _, _, _, "VList (_ :: xs) when n = 2 -> Some (VList xs)"),
        %% Fall-through guard for n < 1 / non-compound T.
        sub_string(S, _, _, _, "| Arg (_, _, _) -> None")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Arg specialized step case')
    ).

test_fsharp_not_member_list_step :-
    Test = 'WAM-FSharp: NotMemberList step case',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| NotMemberList (xReg, lReg) ->"),
        %% Inline list walk via List.exists + derefVar.
        sub_string(S, _, _, _, "items |> List.exists (fun item -> derefVar s.WsBindings item = x)")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing NotMemberList step case')
    ).

test_fsharp_not_member_const_atoms_step :-
    Test = 'WAM-FSharp: NotMemberConstAtoms step case',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| NotMemberConstAtoms (xReg, atoms) ->"),
        %% Atom case: List.contains on the baked-in atom strings.
        sub_string(S, _, _, _, "if List.contains name atoms then None"),
        %% Could-unify cases (Unbound, Ref) fail.
        sub_string(S, _, _, _, "| Some (Unbound _) -> None"),
        sub_string(S, _, _, _, "| Some (Ref _)     -> None"),
        %% Non-atom ground succeeds.
        sub_string(S, _, _, _, "| Some _           -> Some { s with WsPC = s.WsPC + 1 }")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing NotMemberConstAtoms step case')
    ).

test_fsharp_vset_step_cases :-
    Test = 'WAM-FSharp: BuildEmptySet / SetInsert / NotMemberSet step cases',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        %% BuildEmptySet writes VSet Set.empty into the target register.
        sub_string(S, _, _, _, "| BuildEmptySet reg ->"),
        sub_string(S, _, _, _, "r.[reg] <- VSet Set.empty"),
        %% SetInsert reads Atom + VSet, writes VSet with Set.add.
        sub_string(S, _, _, _, "| SetInsert (elemReg, inReg, outReg) ->"),
        sub_string(S, _, _, _, "Some (Atom name), Some (VSet s0)"),
        sub_string(S, _, _, _, "VSet (Set.add name s0)"),
        %% NotMemberSet uses Set.contains for the O(log N) check.
        sub_string(S, _, _, _, "| NotMemberSet (elemReg, setReg) ->"),
        sub_string(S, _, _, _, "Set.contains name s0")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing VSet step case patterns')
    ).

%% ----------------------------------------------------------------------
%% Phase J parity: parallel WAM TPL wiring.  Mirrors the Haskell baseline''s
%% enumerateParBranches + runNegationParallel helpers.  The forkable-
%% aggregate path (forkParBranches + MergeStrategy) is deliberately out
%% of scope for this round — it requires the MergeStrategy / ForkContext
%% machinery the Haskell target has but F# does not yet.
%% ----------------------------------------------------------------------

test_fsharp_enumerate_par_branches_present :-
    Test = 'WAM-FSharp: enumerateParBranches helper emitted',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "and enumerateParBranches (ctx: WamContext) (parPC: int) (elsePC: int) : int list ="),
        %% Walks the Par* chain via ParRetryMeElse / ParRetryMeElsePc, stops at ParTrustMe.
        sub_string(S, _, _, _, "| ParRetryMeElse label ->"),
        sub_string(S, _, _, _, "| ParRetryMeElsePc nextPC ->"),
        sub_string(S, _, _, _, "| ParTrustMe ->"),
        %% Pre-Par variants terminate the chain (mixed sequential/parallel).
        sub_string(S, _, _, _, "| RetryMeElse _ | RetryMeElsePc _ | TrustMe ->")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing enumerateParBranches helper or chain handling')
    ).

test_fsharp_run_negation_parallel_present :-
    Test = 'WAM-FSharp: runNegationParallel helper emitted',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "and runNegationParallel (ctx: WamContext) (s: WamState) (entryPC: int) (elsePC: int) : bool ="),
        %% forkMinBranches threshold for forking overhead.
        sub_string(S, _, _, _, "and forkMinBranches : int = 3"),
        sub_string(S, _, _, _, "List.length branchPCs >= forkMinBranches"),
        %% Async.Choice gives soft race-to-cancel: returns first Some,
        %% wall time bounded by the first successful branch (rather
        %% than the slowest, which Async.Parallel + Array.exists would).
        sub_string(S, _, _, _, "|> Async.Choice"),
        sub_string(S, _, _, _, "|> Async.RunSynchronously"),
        sub_string(S, _, _, _, "result.IsSome"),
        %% Hard-cancel: each branch pulls its CancellationToken via
        %% Async.CancellationToken and wires it into a per-branch ctx
        %% so the inner `run` loop can short-circuit.
        sub_string(S, _, _, _, "let! token = Async.CancellationToken"),
        sub_string(S, _, _, _, "let ctxC = { ctx with WcCancellationToken = Some token }"),
        %% Fallback to sequential when too few branches.
        sub_string(S, _, _, _, "// Too few branches for fork overhead to pay off")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing runNegationParallel helper patterns')
    ).

test_fsharp_wam_context_has_cancellation_token :-
    %% Phase K: hard-cancel for runNegationParallel.  WamContext gains
    %% an optional CancellationToken field that the runtime checks at
    %% each iteration of the `run` loop.
    Test = 'WAM-FSharp: WamContext.WcCancellationToken field emitted',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "WcCancellationToken: System.Threading.CancellationToken option")
    ->  pass(Test)
    ;   fail_test(Test, 'WcCancellationToken field missing from WamContext')
    ).

test_fsharp_run_loop_checks_cancellation_token :-
    %% The `run` loop must consult the token each iteration and return
    %% None on cancellation request.
    Test = 'WAM-FSharp: run loop checks WcCancellationToken',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "and run (ctx: WamContext) (s: WamState) : WamState option ="),
        %% Per-iteration token check.
        sub_string(S, _, _, _, "| Some t when t.IsCancellationRequested -> true"),
        sub_string(S, _, _, _, "elif (match ctx.WcCancellationToken with")
    ->  pass(Test)
    ;   fail_test(Test, 'run loop missing cancellation-token check')
    ).

test_fsharp_negation_dispatches_through_parallel :-
    %% \+/1 must dispatch through runNegationParallel when the goal''s
    %% entry instruction is ParTryMeElse / ParTryMeElsePc.  Sequential
    %% fallback for other entry shapes.
    Test = 'WAM-FSharp: \\+/1 dispatches through runNegationParallel for Par* goals',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "| ParTryMeElse elseLabel ->"),
        sub_string(S, _, _, _, "| ParTryMeElsePc elsePC ->"),
        sub_string(S, _, _, _, "if runNegationParallel ctx snap pc elsePC then None")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing \\+/1 parallel dispatch patterns')
    ).

test_fsharp_specialized_instructions_wam_parse :-
    %% Round-trip the WAM-text mnemonics through wam_instr_to_fsharp/2
    %% to make sure they map to the right Instruction constructors.
    Test = 'WAM-FSharp: WAM-text parse rules for Phase-I instructions',
    (   wam_fsharp_target:wam_instr_to_fsharp(
            ["put_structure_dyn", "A1", "A2", "A3"], F1),
        wam_fsharp_target:wam_instr_to_fsharp(
            ["arg", "2", "A1", "A3"], F2),
        wam_fsharp_target:wam_instr_to_fsharp(
            ["not_member_list", "A1", "A2"], F3),
        wam_fsharp_target:wam_instr_to_fsharp(
            ["not_member_const_atoms", "A1", "foo", "bar"], F4),
        wam_fsharp_target:wam_instr_to_fsharp(
            ["build_empty_set", "A1"], F5),
        wam_fsharp_target:wam_instr_to_fsharp(
            ["set_insert", "A1", "A2", "A3"], F6),
        wam_fsharp_target:wam_instr_to_fsharp(
            ["not_member_set", "A1", "A2"], F7),
        F1 == "PutStructureDyn (1, 2, 3)",
        F2 == "Arg (2, 1, 3)",
        F3 == "NotMemberList (1, 2)",
        F4 == "NotMemberConstAtoms (1, [\"foo\"; \"bar\"])",
        F5 == "BuildEmptySet 1",
        F6 == "SetInsert (1, 2, 3)",
        F7 == "NotMemberSet (1, 2)"
    ->  pass(Test)
    ;   fail_test(Test, 'WAM-text parse rules produced unexpected output')
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
%% Phase I — lowered-emitter coverage for the Haskell-only specialized
%% instructions.  Before this round these instructions were missing from
%% supported_fs/1 / is_match_instr_fs/1 / emit_one_fs/6, so any
%% predicate whose WAM-text body contained PutStructureDyn / Arg /
%% NotMember* / VSet-family ops silently fell back to the interpreter.
%% These tests assert (a) the lowered emitter now accepts them, and
%% (b) the emitted F# code matches the expected step-delegation shape.
%% ----------------------------------------------------------------------

test_fsharp_lowered_emitter_phase_i_accepted :-
    Test = 'WAM-FSharp lowered: Phase-I body is lowerable',
    Wam = 'p_phase_i/3:\n  build_empty_set A1\n  put_structure_dyn A1 A2 A3\n  arg 2 A1 A3\n  not_member_list A1 A2\n  set_insert A1 A2 A3\n  not_member_set A1 A2\n  not_member_const_atoms A1 foo bar baz\n  proceed',
    (   wam_fsharp_lowerable(p_phase_i/3, Wam, _Reason)
    ->  pass(Test)
    ;   fail_test(Test, 'Phase-I body should be lowerable')
    ).

test_fsharp_lowered_emitter_phase_i_emits_step_delegation :-
    Test = 'WAM-FSharp lowered: Phase-I instructions delegate to step',
    Wam = 'p_phase_i/3:\n  build_empty_set A1\n  put_structure_dyn A1 A2 A3\n  arg 2 A1 A3\n  not_member_list A1 A2\n  set_insert A1 A2 A3\n  not_member_set A1 A2\n  not_member_const_atoms A1 foo bar baz\n  proceed',
    (   wam_fsharp_target:lower_predicate_to_fsharp(p_phase_i/3, Wam,
            [base_pc(1), foreign_preds([])],
            lowered(_, _FuncName, Code)),
        %% Every Phase-I instruction must delegate to `step` with the
        %% correct constructor; the |> Some sv -> continuation chain
        %% must be present.
        sub_string(Code, _, _, _, "(BuildEmptySet 1)"),
        sub_string(Code, _, _, _, "(PutStructureDyn (1, 2, 3))"),
        sub_string(Code, _, _, _, "(Arg (2, 1, 3))"),
        sub_string(Code, _, _, _, "(NotMemberList (1, 2))"),
        sub_string(Code, _, _, _, "(SetInsert (1, 2, 3))"),
        sub_string(Code, _, _, _, "(NotMemberSet (1, 2))"),
        sub_string(Code, _, _, _, "(NotMemberConstAtoms (1, [\"foo\"; \"bar\"; \"baz\"]))"),
        %% Step-delegation chain shape: match step ctx { ... } (Instr) with | Some _ ->
        sub_string(Code, _, _, _, "match step ctx { s_init with WsPC = 1 } (BuildEmptySet 1) with")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Phase-I step-delegation patterns')
    ).

test_fsharp_lowered_emitter_phase_i_pc_offsets :-
    %% base_pc(N) offsets the local PCs so they match the global merged
    %% instruction array.  With base_pc(10), the first instruction''s
    %% WsPC should be 10 (not 1).
    Test = 'WAM-FSharp lowered: Phase-I respects base_pc offset',
    Wam = 'p_phase_i_off/2:\n  build_empty_set A1\n  not_member_set A1 A2\n  proceed',
    (   wam_fsharp_target:lower_predicate_to_fsharp(p_phase_i_off/2, Wam,
            [base_pc(10), foreign_preds([])],
            lowered(_, _, Code)),
        %% First instruction at offset 10.
        sub_string(Code, _, _, _, "WsPC = 10 } (BuildEmptySet 1)"),
        %% Second at 11.
        sub_string(Code, _, _, _, "WsPC = 11 } (NotMemberSet (1, 2))")
    ->  pass(Test)
    ;   fail_test(Test, 'base_pc offset not threaded through Phase-I emitters')
    ).

%% ----------------------------------------------------------------------
%% Phase J-β — forkParBranches + MergeStrategy machinery.  Mirrors the
%% Haskell baseline''s forkable-aggregate fork path.  Per-branch
%% aggregation is computed normally (via finalizeAggregate in each
%% branch), then the per-branch results are merged by
%% combineParBranchResults at the cross-branch level.  Sum / count /
%% bag / set / findall are forkable; anything else falls back to
%% sequential TryMeElse semantics.
%% ----------------------------------------------------------------------

test_fsharp_merge_strategy_du :-
    Test = 'WAM-FSharp: MergeStrategy DU + AggFrame.AggMergeStrategy',
    (   fsharp_wam_type_header(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "type MergeStrategy ="),
        sub_string(S, _, _, _, "| MergeSum"),
        sub_string(S, _, _, _, "| MergeCount"),
        sub_string(S, _, _, _, "| MergeBag"),
        sub_string(S, _, _, _, "| MergeSet"),
        sub_string(S, _, _, _, "| MergeFindall"),
        sub_string(S, _, _, _, "| MergeSequential"),
        sub_string(S, _, _, _, "AggMergeStrategy:  MergeStrategy")
    ->  pass(Test)
    ;   fail_test(Test, 'MergeStrategy DU or AggFrame field missing')
    ).

test_fsharp_infer_merge_strategy :-
    Test = 'WAM-FSharp: inferMergeStrategy helper',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "and inferMergeStrategy (aggType: string) : MergeStrategy ="),
        sub_string(S, _, _, _, "| \"sum\"     -> MergeSum"),
        sub_string(S, _, _, _, "| \"count\"   -> MergeCount"),
        sub_string(S, _, _, _, "| \"bag\"     -> MergeBag"),
        sub_string(S, _, _, _, "| \"set\"     -> MergeSet"),
        sub_string(S, _, _, _, "| \"findall\" -> MergeFindall"),
        sub_string(S, _, _, _, "| \"collect\" -> MergeFindall"),
        sub_string(S, _, _, _, "| _         -> MergeSequential")
    ->  pass(Test)
    ;   fail_test(Test, 'inferMergeStrategy patterns missing')
    ).

test_fsharp_begin_aggregate_records_strategy :-
    %% BeginAggregate must thread inferMergeStrategy(aggType) into the
    %% new AggMergeStrategy field on the pushed AggFrame.
    Test = 'WAM-FSharp: BeginAggregate stores AggMergeStrategy',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "AggMergeStrategy = inferMergeStrategy aggType")
    ->  pass(Test)
    ;   fail_test(Test, 'BeginAggregate does not record AggMergeStrategy')
    ).

test_fsharp_forkable_strategy_helpers :-
    Test = 'WAM-FSharp: isForkableStrategy / currentAggMergeStrategy / currentAggFrame helpers',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "and isForkableStrategy (ms: MergeStrategy) : bool ="),
        sub_string(S, _, _, _, "| MergeSum | MergeCount | MergeBag | MergeSet | MergeFindall -> true"),
        sub_string(S, _, _, _, "| MergeSequential -> false"),
        sub_string(S, _, _, _, "and currentAggMergeStrategy (s: WamState) : MergeStrategy option ="),
        sub_string(S, _, _, _, "and currentAggFrame (s: WamState) : AggFrame option =")
    ->  pass(Test)
    ;   fail_test(Test, 'forkable-strategy helpers missing')
    ).

test_fsharp_fork_par_branches_helpers :-
    Test = 'WAM-FSharp: removeNearestAggFrame / findOuterEndAggregate / combineParBranchResults / forkParBranches',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "and removeNearestAggFrame (cps: ChoicePoint list) : ChoicePoint list ="),
        sub_string(S, _, _, _, "and findOuterEndAggregate (ctx: WamContext) (startPC: int) : int ="),
        sub_string(S, _, _, _, "| EndAggregate _ -> pc + 1"),
        sub_string(S, _, _, _, "and combineParBranchResults (ms: MergeStrategy) (results: Value list) : Value ="),
        %% Per-strategy combiners.
        sub_string(S, _, _, _, "| MergeSum ->"),
        sub_string(S, _, _, _, "| MergeCount ->"),
        sub_string(S, _, _, _, "| MergeBag | MergeFindall ->"),
        sub_string(S, _, _, _, "| MergeSet ->"),
        sub_string(S, _, _, _, "List.distinct allItems"),
        sub_string(S, _, _, _, "and forkParBranches (ctx: WamContext) (s: WamState) (af: AggFrame)"),
        sub_string(S, _, _, _, "|> Async.Parallel"),
        sub_string(S, _, _, _, "combineParBranchResults af.AggMergeStrategy perBranchValues")
    ->  pass(Test)
    ;   fail_test(Test, 'forkParBranches helpers missing or incomplete')
    ).

test_fsharp_fork_or_sequential_dispatcher :-
    Test = 'WAM-FSharp: forkOrSequential dispatcher + Par*MeElse routing',
    (   compile_wam_runtime_to_fsharp([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "and forkOrSequential (ctx: WamContext) (s: WamState)"),
        %% Forkable + enough branches => fork.
        sub_string(S, _, _, _, "isForkableStrategy af.AggMergeStrategy"),
        sub_string(S, _, _, _, "forkParBranches ctx s af s.WsPC elsePC"),
        %% Otherwise fall back to sequential TryMeElse semantics.
        sub_string(S, _, _, _, "step ctx s (TryMeElse lbl)"),
        sub_string(S, _, _, _, "step ctx s (TryMeElsePc pc)"),
        %% Par*MeElse step arms dispatch through forkOrSequential.
        sub_string(S, _, _, _, "| ParTryMeElse label    -> forkOrSequential ctx s (Choice1Of2 label)"),
        sub_string(S, _, _, _, "| ParTryMeElsePc pc     -> forkOrSequential ctx s (Choice2Of2 pc)")
    ->  pass(Test)
    ;   fail_test(Test, 'forkOrSequential dispatcher or ParTryMeElse routing missing')
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
    %% Phase H — list / sort / order / unification (Go parity)
    test_fsharp_compare_value_helper_present,
    test_fsharp_value_du_no_buggy_overrides,
    test_fsharp_append_builtin,
    test_fsharp_reverse_builtin,
    test_fsharp_last_builtin,
    test_fsharp_nth_builtins,
    test_fsharp_memberchk_builtin,
    test_fsharp_delete_builtin,
    test_fsharp_select_builtin,
    test_fsharp_select_retry_builtin_state,
    test_fsharp_select_retry_resume_handler,
    test_fsharp_numlist_builtin,
    test_fsharp_sort_msort_builtins,
    test_fsharp_compare_builtin,
    test_fsharp_standard_order_comparison,
    test_fsharp_unify_builtin,
    test_fsharp_not_unifiable_builtin,
    %% Phase I — Haskell-only specialized instructions
    test_fsharp_vset_value_variant,
    test_fsharp_specialized_instructions_declared,
    test_fsharp_put_structure_dyn_step,
    test_fsharp_arg_specialized_step,
    test_fsharp_not_member_list_step,
    test_fsharp_not_member_const_atoms_step,
    test_fsharp_vset_step_cases,
    %% Phase J — parallel WAM TPL wiring
    test_fsharp_enumerate_par_branches_present,
    test_fsharp_run_negation_parallel_present,
    test_fsharp_wam_context_has_cancellation_token,
    test_fsharp_run_loop_checks_cancellation_token,
    test_fsharp_negation_dispatches_through_parallel,
    test_fsharp_specialized_instructions_wam_parse,
    test_fsharp_fact_shape_helpers_exported,
    test_fsharp_emit_mode_resolution,
    test_fsharp_lowerable_single_clause_proceed,
    test_fsharp_lowerable_rejects_aggregate,
    test_fsharp_lowerable_multi_clause,
    %% Phase I — lowered emitter
    test_fsharp_lowered_emitter_phase_i_accepted,
    test_fsharp_lowered_emitter_phase_i_emits_step_delegation,
    test_fsharp_lowered_emitter_phase_i_pc_offsets,
    %% Phase J-β — forkParBranches + MergeStrategy
    test_fsharp_merge_strategy_du,
    test_fsharp_infer_merge_strategy,
    test_fsharp_begin_aggregate_records_strategy,
    test_fsharp_forkable_strategy_helpers,
    test_fsharp_fork_par_branches_helpers,
    test_fsharp_fork_or_sequential_dispatcher,
    test_fsharp_project_generation_smoke,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).
