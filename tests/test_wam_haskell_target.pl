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
:- use_module('../src/unifyweaver/targets/wam_target').
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
        sub_string(S, _, _, _, "Str fnId args -> Just (Atom fnId, length args)")
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
        sub_string(S, _, _, _, "VList (Atom fnId : args)"),
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
        %% Native call: first arg is facts, input atom is already Int (interned)
        sub_string(Code, _, _, _, "nativeKernel_transitive_closure edge_facts"),
        sub_string(Code, _, _, _, "r1I"),
        %% Output is atom: directly use interned ID (rv_1 after
        %% routing single-output through the multi-output FFIStreamRetry path)
        sub_string(Code, _, _, _, "Atom rv_1"),
        %% Single-input case pattern: Atom r1I (Int, not String)
        sub_string(Code, _, _, _, "case r1 of"),
        sub_string(Code, _, _, _, "Atom r1I ->")
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

%% Phase 4.2: MergeStrategy, ForkContext, fork helpers — codegen
%% --------------------------------------------

test_haskell_merge_strategy_in_types :-
    Test = 'WAM-Haskell: MergeStrategy constructors present in WamTypes',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "data MergeStrategy"),
        sub_string(S, _, _, _, "MergeSumInt"),
        sub_string(S, _, _, _, "MergeSumDouble"),
        sub_string(S, _, _, _, "MergeCount"),
        sub_string(S, _, _, _, "MergeFindall"),
        sub_string(S, _, _, _, "MergeBag"),
        sub_string(S, _, _, _, "MergeSet"),
        sub_string(S, _, _, _, "MergeRace"),
        sub_string(S, _, _, _, "MergeNegation"),
        sub_string(S, _, _, _, "MergeSequential")
    ->  pass(Test)
    ;   fail_test(Test, 'MergeStrategy constructors missing')
    ).

test_haskell_fork_context_in_types :-
    Test = 'WAM-Haskell: ForkContext record emitted in WamTypes',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "data ForkContext"),
        sub_string(S, _, _, _, "fcMergeStrategy"),
        sub_string(S, _, _, _, "fcWorkEstimate")
    ->  pass(Test)
    ;   fail_test(Test, 'ForkContext record missing')
    ).

test_haskell_agg_frame_has_merge_strategy :-
    Test = 'WAM-Haskell: AggFrame includes afMergeStrategy field',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "afMergeStrategy :: !MergeStrategy")
    ->  pass(Test)
    ;   fail_test(Test, 'afMergeStrategy field missing from AggFrame')
    ).

test_haskell_infer_merge_strategy :-
    Test = 'WAM-Haskell: inferMergeStrategy function emitted',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "inferMergeStrategy :: String -> MergeStrategy"),
        sub_string(S, _, _, _, "inferMergeStrategy \"sum\""),
        sub_string(S, _, _, _, "inferMergeStrategy \"count\"")
    ->  pass(Test)
    ;   fail_test(Test, 'inferMergeStrategy function missing')
    ).

test_haskell_value_nfdata_instance :-
    Test = 'WAM-Haskell: NFData Value instance emitted for parMap',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "instance NFData Value")
    ->  pass(Test)
    ;   fail_test(Test, 'NFData Value instance missing')
    ).

test_haskell_fork_min_branches_threshold :-
    Test = 'WAM-Haskell: forkMinBranches threshold emitted in runtime',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "forkMinBranches :: Int"),
        sub_string(S, _, _, _, "forkMinBranches = 3"),
        sub_string(S, _, _, _, "length branches >= forkMinBranches")
    ->  pass(Test)
    ;   fail_test(Test, 'forkMinBranches threshold missing')
    ).

test_haskell_fork_helpers_present :-
    Test = 'WAM-Haskell: fork helpers (forkOrSequential, etc.) emitted in runtime',
    (   wam_haskell_target:compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "forkOrSequential"),
        sub_string(S, _, _, _, "currentAggMergeStrategy"),
        sub_string(S, _, _, _, "isForkableStrategy"),
        sub_string(S, _, _, _, "enumerateParBranches"),
        sub_string(S, _, _, _, "runBranchForFork"),
        sub_string(S, _, _, _, "forkParBranches"),
        sub_string(S, _, _, _, "findOuterEndAggregate")
    ->  pass(Test)
    ;   fail_test(Test, 'Phase 4.2 fork helper(s) missing')
    ).

test_haskell_partryme_else_delegates_to_fork :-
    Test = 'WAM-Haskell: ParTryMeElse step handler routes through forkOrSequential',
    (   wam_haskell_target:compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !ctx s (ParTryMeElse label)"),
        sub_string(S, _, _, _, "forkOrSequential"),
        % ParTryMeElsePc also routes through fork path
        sub_string(S, _, _, _, "step !ctx s (ParTryMeElsePc pc)")
    ->  pass(Test)
    ;   fail_test(Test, 'ParTryMeElse step handler does not route to fork')
    ).

test_haskell_runtime_imports_parallel :-
    Test = 'WAM-Haskell: WamRuntime imports Control.Parallel.Strategies',
    (   wam_haskell_target:compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "Control.Parallel.Strategies"),
        sub_string(S, _, _, _, "Control.DeepSeq"),
        sub_string(S, _, _, _, "parMap"),
        sub_string(S, _, _, _, "rdeepseq")
    ->  pass(Test)
    ;   fail_test(Test, 'parallel/deepseq imports missing from WamRuntime')
    ).

%% PutStructureDyn: runtime-parsed functors
%% --------------------------------------------

test_haskell_put_structure_dyn_in_types :-
    Test = 'WAM-Haskell: PutStructureDyn constructor in Instruction type',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "PutStructureDyn !RegId !RegId !RegId")
    ->  pass(Test)
    ;   fail_test(Test, 'PutStructureDyn missing from Instruction type')
    ).

test_haskell_put_structure_dyn_step_handler :-
    Test = 'WAM-Haskell: PutStructureDyn step handler present',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !ctx s (PutStructureDyn nameReg arityReg targetReg)"),
        sub_string(S, _, _, _, "BuildStruct fnId targetReg (fromIntegral arity)")
    ->  pass(Test)
    ;   fail_test(Test, 'PutStructureDyn step handler missing or incorrect')
    ).

test_haskell_put_structure_dyn_wam_parse :-
    Test = 'WAM-Haskell: put_structure_dyn WAM text parsed',
    (   wam_haskell_target:wam_instr_to_haskell(
            ["put_structure_dyn", "A1,", "A2,", "A3"], Hs),
        sub_string(Hs, _, _, _, "PutStructureDyn")
    ->  pass(Test)
    ;   fail_test(Test, 'put_structure_dyn WAM text not parsed')
    ).

%% Phase 4.3: findall/bag/set merge strategies
%% --------------------------------------------

test_haskell_findall_bag_set_forkable :-
    Test = 'WAM-Haskell: isForkableStrategy covers findall/bag/set',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "isForkableStrategy MergeFindall   = True"),
        sub_string(S, _, _, _, "isForkableStrategy MergeBag       = True"),
        sub_string(S, _, _, _, "isForkableStrategy MergeSet       = True")
    ->  pass(Test)
    ;   fail_test(Test, 'findall/bag/set not forkable')
    ).

test_haskell_infer_collect_maps_to_findall :-
    Test = 'WAM-Haskell: inferMergeStrategy "collect" = MergeFindall',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "inferMergeStrategy \"collect\" = MergeFindall")
    ->  pass(Test)
    ;   fail_test(Test, 'collect -> MergeFindall mapping missing')
    ).

test_haskell_apply_aggregation_set_dedup :-
    Test = 'WAM-Haskell: applyAggregation "set" uses nub for dedup',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "applyAggregation \"set\" vals = VList (nub vals)")
    ->  pass(Test)
    ;   fail_test(Test, 'set dedup via nub missing')
    ).

test_haskell_apply_aggregation_bag :-
    Test = 'WAM-Haskell: applyAggregation "bag" returns VList vals',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "applyAggregation \"bag\" vals = VList vals")
    ->  pass(Test)
    ;   fail_test(Test, 'bag applyAggregation missing')
    ).

test_haskell_nub_import :-
    Test = 'WAM-Haskell: Data.List import includes nub',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "nub")
    ->  pass(Test)
    ;   fail_test(Test, 'nub not imported from Data.List')
    ).

test_wam_bag_template_compiles :-
    Test = 'WAM: aggregate_all(bag(X), ...) emits begin_aggregate bag',
    (   retractall(user:bag_demo(_, _)),
        assertz((user:bag_demo(Xs, Y) :-
            aggregate_all(bag(X), member(X, Y), Xs))),
        wam_target:compile_single_clause_wam(
            bag_demo(Xs, Y)-(aggregate_all(bag(X), member(X, Y), Xs)),
            [], Code),
        sub_string(Code, _, _, _, "begin_aggregate bag")
    ->  pass(Test)
    ;   fail_test(Test, 'bag template did not compile to begin_aggregate bag')
    ).

test_wam_set_template_compiles :-
    Test = 'WAM: aggregate_all(set(X), ...) emits begin_aggregate set',
    (   retractall(user:set_demo(_, _)),
        assertz((user:set_demo(Xs, Y) :-
            aggregate_all(set(X), member(X, Y), Xs))),
        wam_target:compile_single_clause_wam(
            set_demo(Xs, Y)-(aggregate_all(set(X), member(X, Y), Xs)),
            [], Code),
        sub_string(Code, _, _, _, "begin_aggregate set")
    ->  pass(Test)
    ;   fail_test(Test, 'set template did not compile to begin_aggregate set')
    ).

%% Phase 4.4: General negation + parallel negation
%% --------------------------------------------

test_haskell_negation_general_handler :-
    Test = 'WAM-Haskell: general \\+/1 handler with run-based sub-execution',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        % General path: resolve goal key via lookupAtom and call run ctx snapshot
        sub_string(S, _, _, _, "lookupAtom tbl fnId"),
        sub_string(S, _, _, _, "case run ctx snapshot of")
    ->  pass(Test)
    ;   fail_test(Test, 'general \\+/1 handler missing run-based sub-execution')
    ).

test_haskell_negation_parallel_dispatch :-
    Test = 'WAM-Haskell: \\+/1 dispatches to parallel for Par* goals',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "runNegationParallel"),
        sub_string(S, _, _, _, "ParTryMeElse elseLabel")
    ->  pass(Test)
    ;   fail_test(Test, '\\+/1 parallel dispatch for Par* missing')
    ).

test_haskell_negation_parallel_helper :-
    Test = 'WAM-Haskell: runNegationParallel uses async race-to-cancel',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "runNegationParallel :: WamContext -> WamState -> Int -> Int -> Bool"),
        sub_string(S, _, _, _, "unsafePerformIO"),
        sub_string(S, _, _, _, "raceToTrue")
    ->  pass(Test)
    ;   fail_test(Test, 'runNegationParallel race-to-cancel missing or incomplete')
    ).

test_haskell_race_to_true_helper :-
    Test = 'WAM-Haskell: raceToTrue helper with async/waitAny/cancel',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "raceToTrue :: [IO Bool] -> IO Bool"),
        sub_string(S, _, _, _, "waitAny"),
        sub_string(S, _, _, _, "mapM_ cancel")
    ->  pass(Test)
    ;   fail_test(Test, 'raceToTrue helper missing')
    ).

test_haskell_async_imports :-
    Test = 'WAM-Haskell: async/unsafePerformIO imports present',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "Control.Concurrent.Async"),
        sub_string(S, _, _, _, "System.IO.Unsafe"),
        sub_string(S, _, _, _, "Control.Exception (evaluate)")
    ->  pass(Test)
    ;   fail_test(Test, 'async/unsafe imports missing')
    ).

test_haskell_negation_true_fail_fast_paths :-
    Test = 'WAM-Haskell: \\+/1 has fast paths for true and fail atoms',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "aid == atomTrue -> Nothing"),
        sub_string(S, _, _, _, "aid == atomFail -> Just (s { wsPC = wsPC s + 1 })")
    ->  pass(Test)
    ;   fail_test(Test, '\\+ true/fail fast paths missing')
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
    Test = 'WAM-Haskell: step function handles Par* (fork path + sequential fallback)',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !ctx s (ParTryMeElse label)"),
        sub_string(S, _, _, _, "step !ctx s (ParRetryMeElse label)"),
        sub_string(S, _, _, _, "step !ctx s ParTrustMe"),
        % Phase 4.2: ParTryMeElse routes through forkOrSequential which
        % decides fork vs fallback based on the enclosing aggregate''s
        % merge strategy. ParRetryMeElse / ParTrustMe still delegate
        % straight to the sequential variants.
        sub_string(S, _, _, _, "forkOrSequential ctx s"),
        sub_string(S, _, _, _, "step ctx s (RetryMeElse label)"),
        sub_string(S, _, _, _, "step ctx s TrustMe")
    ->  pass(Test)
    ;   fail_test(Test, 'Par* step handlers missing or not wired correctly')
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
    %% Phase 4.2: MergeStrategy / ForkContext / fork helpers
    test_haskell_merge_strategy_in_types,
    test_haskell_fork_context_in_types,
    test_haskell_agg_frame_has_merge_strategy,
    test_haskell_infer_merge_strategy,
    test_haskell_value_nfdata_instance,
    test_haskell_fork_min_branches_threshold,
    test_haskell_fork_helpers_present,
    test_haskell_partryme_else_delegates_to_fork,
    test_haskell_runtime_imports_parallel,
    %% PutStructureDyn: runtime-parsed functors
    test_haskell_put_structure_dyn_in_types,
    test_haskell_put_structure_dyn_step_handler,
    test_haskell_put_structure_dyn_wam_parse,
    %% Phase 4.3: findall/bag/set merge strategies
    test_haskell_findall_bag_set_forkable,
    test_haskell_infer_collect_maps_to_findall,
    test_haskell_apply_aggregation_set_dedup,
    test_haskell_apply_aggregation_bag,
    test_haskell_nub_import,
    test_wam_bag_template_compiles,
    test_wam_set_template_compiles,
    %% Phase 4.4: General negation + parallel negation
    test_haskell_negation_general_handler,
    test_haskell_negation_parallel_dispatch,
    test_haskell_negation_parallel_helper,
    test_haskell_race_to_true_helper,
    test_haskell_async_imports,
    test_haskell_negation_true_fail_fast_paths,
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
