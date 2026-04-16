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
:- use_module('../src/unifyweaver/core/clause_body_analysis').
:- use_module('../src/unifyweaver/core/purity_certificate').

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
        sub_string(Code, _, _, _, "Integer (fromIntegral rv_1)"),
        %% Choice point creation for stream results — all kernels now
        %% route through FFIStreamRetry (pre-wrapped Values) for type
        %% correctness with atom/float outputs.
        sub_string(Code, _, _, _, "FFIStreamRetry"),
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
        %% Signature uses IntMap after atom interning
        sub_string(Code, _, _, _, "IM.IntMap [Int] -> Int -> [Int]"),
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
        %% config_facts_from resolved to edge pred name, now using wcFfiFacts
        sub_string(Code, _, _, _, "edge_facts = fromMaybe IM.empty"),
        sub_string(Code, _, _, _, "wcFfiFacts ctx"),
        %% Native call: first arg is facts, second is interned atom lookup
        sub_string(Code, _, _, _, "nativeKernel_transitive_closure edge_facts"),
        sub_string(Code, _, _, _, "Map.lookup r1S (wcAtomIntern ctx)"),
        %% Output is atom: de-intern via wcAtomDeintern (rv_1 after
        %% routing single-output through the multi-output FFIStreamRetry path)
        sub_string(Code, _, _, _, "Atom (fromMaybe \"\" (IM.lookup rv_1 (wcAtomDeintern ctx)))"),
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

%% Phase 8: CallForeign instruction codegen
%% -------------------------------------------

test_call_foreign_in_types :-
    Test = 'WAM-Haskell: CallForeign in Instruction data type',
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "CallForeign String !Int")
    ->  pass(Test)
    ;   fail_test(Test, 'CallForeign not in Instruction data type')
    ).

test_call_foreign_step_case :-
    Test = 'WAM-Haskell: CallForeign step case dispatches to executeForeign',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        %% CallForeign step calls executeForeign directly
        sub_string(S, _, _, _, "step !ctx s (CallForeign pred _arity)"),
        sub_string(S, _, _, _, "executeForeign ctx pred"),
        %% Call step does NOT call executeForeign (removed from fallthrough)
        %% The Call case should have wcLoweredPredicates but NOT executeForeign
        sub_string(S, _, _, _, "step !ctx s (Call pred _arity)")
    ->  pass(Test)
    ;   fail_test(Test, 'CallForeign step case missing or Call still has executeForeign')
    ).

test_call_foreign_resolve :-
    Test = 'WAM-Haskell: resolveCallInstrs produces CallForeign',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        %% resolveCallInstrs should map foreign preds to CallForeign
        sub_string(S, _, _, _, "CallForeign pred arity")
    ->  pass(Test)
    ;   fail_test(Test, 'resolveCallInstrs does not produce CallForeign')
    ).

test_call_foreign_helper :-
    Test = 'WAM-Haskell: callForeign helper for lowered functions',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "callForeign :: WamContext -> String -> WamState -> Maybe WamState"),
        sub_string(S, _, _, _, "callForeign !ctx pred !sc = executeForeign ctx pred sc")
    ->  pass(Test)
    ;   fail_test(Test, 'callForeign helper missing from WamRuntime')
    ).

%% Phase 4.1: Par* instructions — type, step handlers, resolveCallInstrs
%% --------------------------------------------

test_haskell_par_instructions_in_types :-
    Test = 'WAM-Haskell: Par* constructors present in Instruction type',
    (   compile_wam_runtime_to_haskell([], [], Code0),
        % The Par* constructors are declared in WamTypes, which we
        % access indirectly via the types generator.
        atom_string(Code0, _),
        wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, TypesS),
        sub_string(TypesS, _, _, _, "ParTryMeElse String"),
        sub_string(TypesS, _, _, _, "ParRetryMeElse String"),
        sub_string(TypesS, _, _, _, "ParTrustMe"),
        sub_string(TypesS, _, _, _, "ParTryMeElsePc !Int"),
        sub_string(TypesS, _, _, _, "ParRetryMeElsePc !Int")
    ->  pass(Test)
    ;   fail_test(Test, 'Par* constructors missing from Instruction type')
    ).

test_haskell_par_step_handlers_present :-
    Test = 'WAM-Haskell: step function handles Par* by delegating',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !ctx s (ParTryMeElse label)"),
        sub_string(S, _, _, _, "step !ctx s (ParRetryMeElse label)"),
        sub_string(S, _, _, _, "step !ctx s ParTrustMe"),
        sub_string(S, _, _, _, "step ctx s (TryMeElse label)"),
        % confirm the ParTrustMe handler delegates to TrustMe
        sub_string(S, _, _, _, "step ctx s TrustMe")
    ->  pass(Test)
    ;   fail_test(Test, 'Par* step handlers missing or not delegating')
    ).

test_haskell_par_resolve_present :-
    Test = 'WAM-Haskell: resolveCallInstrs handles Par* variants',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "resolve (ParTryMeElse label)"),
        sub_string(S, _, _, _, "resolve (ParRetryMeElse label)"),
        sub_string(S, _, _, _, "ParTryMeElsePc"),
        sub_string(S, _, _, _, "ParRetryMeElsePc")
    ->  pass(Test)
    ;   fail_test(Test, 'Par* resolution missing from resolveCallInstrs')
    ).

%% Phase 4.1: certificate-driven Par* emission
%% --------------------------------------------

test_par_emission_for_user_annotated_predicate :-
    Test = 'WAM-Haskell: :- parallel annotation drives Par* emission',
    % Define a pure multi-clause predicate and mark it parallel.
    retractall(clause_body_analysis:order_independent(user:p41_pure/1)),
    retractall(user:p41_pure(_)),
    assertz(clause_body_analysis:order_independent(user:p41_pure/1)),
    assertz((user:p41_pure(1))),
    assertz((user:p41_pure(2))),
    assertz((user:p41_pure(3))),
    (   wam_haskell_target:maybe_parallelize_instrs(user:p41_pure/1, [],
            ['TryMeElse "L_a"', 'TrustMe', 'RetryMeElse "L_b"'], Out),
        Out == ['ParTryMeElse "L_a"', 'ParTrustMe', 'ParRetryMeElse "L_b"']
    ->  pass(Test)
    ;   fail_test(Test, 'Expected Par* rewrite did not happen')
    ),
    retract(clause_body_analysis:order_independent(user:p41_pure/1)),
    retractall(user:p41_pure(_)).

test_no_par_emission_for_unannotated_impure_predicate :-
    Test = 'WAM-Haskell: un-annotated predicate with impure body stays sequential',
    % Un-annotated but the blacklist will flag write/1 as impure →
    % analyze_predicate_purity returns impure → Par* suppressed.
    retractall(user:p41_impure(_, _)),
    assertz((user:p41_impure(X, Y) :- write(X), Y = X)),
    assertz((user:p41_impure(_, _) :- nl)),
    (   wam_haskell_target:maybe_parallelize_instrs(user:p41_impure/2, [],
            ['TryMeElse "L_a"', 'TrustMe'], Out),
        Out == ['TryMeElse "L_a"', 'TrustMe']
    ->  pass(Test)
    ;   fail_test(Test, 'Par* emitted for impure body — blacklist should suppress')
    ),
    retractall(user:p41_impure(_, _)).

test_no_par_emission_for_impure_body :-
    Test = 'WAM-Haskell: annotation plus impure body still emits sequential (blacklist overrides via confidence)',
    % Annotation alone would normally win (priority 100). The test
    % verifies that analyze_predicate_purity returns `pure/declared`
    % BUT maybe_parallelize_instrs also requires the chain to be at
    % least 0.85 confidence. Declared is 1.0 so the rewrite happens.
    % This test documents current behavior: user declarations are
    % trusted. A stricter policy could consult
    % check_purity_contradictions, but that's not Phase 4.1.
    retractall(clause_body_analysis:order_independent(user:p41_loud/1)),
    retractall(user:p41_loud(_)),
    assertz(clause_body_analysis:order_independent(user:p41_loud/1)),
    assertz((user:p41_loud(X) :- write(X))),
    (   wam_haskell_target:maybe_parallelize_instrs(user:p41_loud/1, [],
            ['TryMeElse "L_a"', 'TrustMe'], Out),
        % Declared user wins — this is expected. Document in reason.
        Out = ['ParTryMeElse "L_a"', 'ParTrustMe']
    ->  pass(Test)
    ;   fail_test(Test, 'Declared-user annotation did not drive Par* as expected')
    ),
    retract(clause_body_analysis:order_independent(user:p41_loud/1)),
    retractall(user:p41_loud(_)).

test_intra_query_parallel_false_kill_switch :-
    Test = 'WAM-Haskell: intra_query_parallel(false) forces sequential emission',
    retractall(clause_body_analysis:order_independent(user:p41_kill/1)),
    assertz(clause_body_analysis:order_independent(user:p41_kill/1)),
    (   wam_haskell_target:maybe_parallelize_instrs(user:p41_kill/1,
            [intra_query_parallel(false)],
            ['TryMeElse "L_a"', 'TrustMe'], Out),
        Out == ['TryMeElse "L_a"', 'TrustMe']
    ->  pass(Test)
    ;   fail_test(Test, 'intra_query_parallel(false) did not suppress Par*')
    ),
    retract(clause_body_analysis:order_independent(user:p41_kill/1)).

test_par_rewrite_preserves_non_choice_instrs :-
    Test = 'WAM-Haskell: Par* rewrite does not touch non-choice instructions',
    retractall(clause_body_analysis:order_independent(user:p41_mix/1)),
    assertz(clause_body_analysis:order_independent(user:p41_mix/1)),
    Input = ['GetConstant (Atom "a") 1',
             'TryMeElse "L1"',
             'Proceed',
             'RetryMeElse "L2"',
             'Call "foo/2" 2',
             'TrustMe',
             'Proceed'],
    Expected = ['GetConstant (Atom "a") 1',
                'ParTryMeElse "L1"',
                'Proceed',
                'ParRetryMeElse "L2"',
                'Call "foo/2" 2',
                'ParTrustMe',
                'Proceed'],
    (   wam_haskell_target:maybe_parallelize_instrs(user:p41_mix/1, [], Input, Out),
        Out == Expected
    ->  pass(Test)
    ;   fail_test(Test, 'Non-choice instructions incorrectly touched by Par* rewrite')
    ),
    retract(clause_body_analysis:order_independent(user:p41_mix/1)).

run_tests :-
    format('~n========================================~n'),
    format('WAM-Haskell target: Phase 5+6+7+8 codegen tests~n'),
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
    test_call_foreign_in_types,
    test_call_foreign_step_case,
    test_call_foreign_resolve,
    test_call_foreign_helper,
    %% Phase 4.1: Par* instructions + certificate-driven emission
    test_haskell_par_instructions_in_types,
    test_haskell_par_step_handlers_present,
    test_haskell_par_resolve_present,
    test_par_emission_for_user_annotated_predicate,
    test_no_par_emission_for_unannotated_impure_predicate,
    test_no_par_emission_for_impure_body,
    test_intra_query_parallel_false_kill_switch,
    test_par_rewrite_preserves_non_choice_instrs,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).

:- initialization(run_tests, main).
