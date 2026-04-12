:- encoding(utf8).
% Codegen tests for WAM-to-Haskell transpilation.
%
% Unlike the WAM-Rust and WAM-WAT targets, Haskell does not have a
% functional execution harness in this project — there is no GHC on
% the CI/dev environment and building a full stack/cabal project per
% test would be prohibitively slow. These tests therefore assert only
% that the generated Haskell source contains the expected identifiers,
% patterns, and dispatch cases. Runtime correctness of the new term
% inspection builtins (functor/3, arg/3, =../2, copy_term/2) is
% validated via the parallel WAM-Rust integration tests in
% tests/test_wam_rust_target.pl + the manual cargo-test suite.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_haskell_target.pl

:- use_module('../src/unifyweaver/targets/wam_haskell_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Phase 5: Term inspection builtins codegen
%% --------------------------------------------

test_haskell_helper_functions_present :-
    Test = 'WAM-Haskell: term-builtin helpers generated',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        %% bindOutput: non-PC-advancing register-binding helper used
        %% by functor/3, arg/3, =../2 for output positions.
        sub_string(S, _, _, _, "bindOutput :: Int -> Value -> WamState"),
        %% copyTermWalk: recursive walker for copy_term/2 that threads
        %% (counter, varMap) to preserve variable sharing.
        sub_string(S, _, _, _, "copyTermWalk :: Int -> IM.IntMap Int"),
        sub_string(S, _, _, _, "copyTermArgs :: Int -> IM.IntMap Int")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing bindOutput/copyTermWalk/copyTermArgs helpers')
    ).

test_haskell_functor_builtin_present :-
    Test = 'WAM-Haskell: functor/3 step case generated',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall \"functor/3\""),
        %% Construct mode: allocates fresh Unbound cells.
        sub_string(S, _, _, _, "Unbound (c0 + i)"),
        %% Read mode: pattern matches Str and VList branches.
        sub_string(S, _, _, _, "Str fn args -> Just (Atom fn, length args)")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing functor/3 step case patterns')
    ).

test_haskell_arg_builtin_present :-
    Test = 'WAM-Haskell: arg/3 step case generated',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall \"arg/3\""),
        %% 1-based indexing into Str args list.
        sub_string(S, _, _, _, "args !! (idx - 1)")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing arg/3 step case patterns')
    ).

test_haskell_univ_builtin_present :-
    Test = 'WAM-Haskell: =../2 step case generated',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall \"=../2\""),
        %% Decompose mode: prepends functor atom to arg list.
        sub_string(S, _, _, _, "VList (Atom fn : args)"),
        %% Compose mode: rebuilds Str from list head+tail.
        sub_string(S, _, _, _, "(Atom fname : rest) -> Just (Str fname rest)")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing =../2 step case patterns')
    ).

test_haskell_copy_term_builtin_present :-
    Test = 'WAM-Haskell: copy_term/2 step case generated',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuiltinCall \"copy_term/2\""),
        %% Drives copyTermWalk with the current var counter and an
        %% empty IntMap (the var map scopes per call).
        sub_string(S, _, _, _, "copyTermWalk (wsVarCounter s) IM.empty tVal"),
        %% The walker''s Unbound branch: reuse existing mapping, else
        %% allocate next counter and extend the map.
        sub_string(S, _, _, _, "IM.insert vid c m")
    ->  pass(Test)
    ;   fail_test(Test, 'Missing copy_term/2 step case patterns')
    ).

test_haskell_no_regressions :-
    %% Smoke-test that adding Phase 5 has not broken pre-existing
    %% generated helpers (unifyVal, is/2, length/2 dispatch).
    Test = 'WAM-Haskell: pre-existing builtins still present',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "unifyVal :: Value -> Value -> WamState"),
        sub_string(S, _, _, _, "BuiltinCall \"is/2\""),
        sub_string(S, _, _, _, "BuiltinCall \"length/2\""),
        sub_string(S, _, _, _, "BuiltinCall \"member/2\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Pre-existing builtin dispatch cases missing')
    ).

%% Phase 6: Parameterized executeForeign codegen
%% ------------------------------------------------

:- use_module('../src/unifyweaver/core/recursive_kernel_detection').

test_parameterized_execute_foreign_category_ancestor :-
    Test = 'WAM-Haskell: parameterized executeForeign for category_ancestor/4',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4, [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_execute_foreign(
            ['category_ancestor/4'-Kernel], Code),
        %% Type signature
        sub_string(Code, _, _, _, "executeForeign :: WamContext -> String -> WamState -> Maybe WamState"),
        %% Dispatch on predicate indicator
        sub_string(Code, _, _, _, "executeForeign !ctx \"category_ancestor/4\" s ="),
        %% Input register reads
        sub_string(Code, _, _, _, "IM.lookup 1 (wsRegs s)"),
        sub_string(Code, _, _, _, "IM.lookup 2 (wsRegs s)"),
        sub_string(Code, _, _, _, "IM.lookup 4 (wsRegs s)"),
        %% Config bindings derived from metadata
        sub_string(Code, _, _, _, "category_parent_facts"),
        sub_string(Code, _, _, _, "max_depth_cfg"),
        %% Native call
        sub_string(Code, _, _, _, "nativeKernel_category_ancestor"),
        %% Output register binding
        sub_string(Code, _, _, _, "IM.lookup 3 (wsRegs s)"),
        sub_string(Code, _, _, _, "Integer (fromIntegral rv)"),
        %% Choice point creation for stream results
        sub_string(Code, _, _, _, "HopsRetry"),
        %% Fallback
        sub_string(Code, _, _, _, "executeForeign _ _ _ = Nothing")
    ->  pass(Test)
    ;   fail_test(Test, 'Parameterized executeForeign missing expected patterns')
    ).

test_parameterized_execute_foreign_empty :-
    Test = 'WAM-Haskell: executeForeign with no kernels is a no-op',
    (   wam_haskell_target:generate_kernel_haskell([], _KF, EF),
        sub_string(EF, _, _, _, "executeForeign _ _ _ = Nothing"),
        \+ sub_string(EF, _, _, _, "executeForeign !ctx")
    ->  pass(Test)
    ;   fail_test(Test, 'Empty kernel list should produce trivial executeForeign')
    ).

test_parameterized_render_kernel_function :-
    Test = 'WAM-Haskell: render_kernel_function uses metadata',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4, [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:render_kernel_function('category_ancestor/4'-Kernel, Code),
        %% Should contain the kernel function from the Mustache template
        sub_string(Code, _, _, _, "nativeKernel_category_ancestor"),
        %% Should have resolved the edge_pred placeholder
        sub_string(Code, _, _, _, "category_parent")
    ->  pass(Test)
    ;   fail_test(Test, 'render_kernel_function failed for category_ancestor')
    ).

test_transitive_closure_kernel_function :-
    Test = 'WAM-Haskell: transitive_closure2 kernel template renders',
    Kernel = recursive_kernel(transitive_closure2, closure/2, [edge_pred(edge/2)]),
    (   wam_haskell_target:render_kernel_function('closure/2'-Kernel, Code),
        sub_string(Code, _, _, _, "nativeKernel_transitive_closure"),
        sub_string(Code, _, _, _, "Map.Map String [String] -> String -> [String]"),
        %% edge_pred placeholder resolved
        sub_string(Code, _, _, _, "Edge predicate: edge")
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_closure2 template rendering failed')
    ).

test_transitive_closure_execute_foreign :-
    Test = 'WAM-Haskell: transitive_closure2 executeForeign generated',
    Kernel = recursive_kernel(transitive_closure2, closure/2, [edge_pred(edge/2)]),
    (   wam_haskell_target:generate_execute_foreign(['closure/2'-Kernel], Code),
        %% Dispatch on predicate indicator
        sub_string(Code, _, _, _, "executeForeign !ctx \"closure/2\" s ="),
        %% Single input register (reg 1)
        sub_string(Code, _, _, _, "IM.lookup 1 (wsRegs s)"),
        %% config_facts_from resolved to edge pred name
        sub_string(Code, _, _, _, "edge_facts"),
        %% Native call with correct args
        sub_string(Code, _, _, _, "nativeKernel_transitive_closure edge_facts r1S"),
        %% Output is atom (not integer)
        sub_string(Code, _, _, _, "Atom rv"),
        %% Single-input case pattern (not tuple)
        sub_string(Code, _, _, _, "case r1 of"),
        sub_string(Code, _, _, _, "Atom r1S ->")
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_closure2 executeForeign generation failed')
    ).

test_multi_kernel_execute_foreign :-
    Test = 'WAM-Haskell: executeForeign with multiple kernels',
    K1 = recursive_kernel(category_ancestor, 'category_ancestor'/4, [max_depth(10), edge_pred(category_parent/2)]),
    K2 = recursive_kernel(transitive_closure2, closure/2, [edge_pred(edge/2)]),
    (   wam_haskell_target:generate_execute_foreign(
            ['category_ancestor/4'-K1, 'closure/2'-K2], Code),
        %% Both dispatch entries present
        sub_string(Code, _, _, _, "executeForeign !ctx \"category_ancestor/4\""),
        sub_string(Code, _, _, _, "executeForeign !ctx \"closure/2\""),
        %% Both native calls present
        sub_string(Code, _, _, _, "nativeKernel_category_ancestor"),
        sub_string(Code, _, _, _, "nativeKernel_transitive_closure"),
        %% Fallback
        sub_string(Code, _, _, _, "executeForeign _ _ _ = Nothing")
    ->  pass(Test)
    ;   fail_test(Test, 'Multi-kernel executeForeign generation failed')
    ).

run_tests :-
    format('~n========================================~n'),
    format('WAM-Haskell target: Phase 5+6+7 codegen tests~n'),
    format('========================================~n~n'),
    test_haskell_helper_functions_present,
    test_haskell_functor_builtin_present,
    test_haskell_arg_builtin_present,
    test_haskell_univ_builtin_present,
    test_haskell_copy_term_builtin_present,
    test_haskell_no_regressions,
    test_parameterized_execute_foreign_category_ancestor,
    test_parameterized_execute_foreign_empty,
    test_parameterized_render_kernel_function,
    test_transitive_closure_kernel_function,
    test_transitive_closure_execute_foreign,
    test_multi_kernel_execute_foreign,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).

:- initialization(run_tests, main).
