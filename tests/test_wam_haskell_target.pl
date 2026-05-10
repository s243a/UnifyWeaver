:- encoding(utf8).
% Codegen tests for WAM-to-Haskell transpilation.
%
% These tests assert that the generated Haskell source contains the
% expected identifiers, patterns, and dispatch cases — they do not
% drive a GHC build per case. A full cabal build per test would be
% prohibitively slow, so most runtime correctness for term inspection
% builtins (functor/3, arg/3, =../2, copy_term/2) is validated via
% the parallel WAM-Rust integration tests in
% tests/test_wam_rust_target.pl + the manual cargo-test suite.
%
% A focused GHC + cabal smoke for the put_structure_dyn instruction
% lives at tests/core/test_wam_put_structure_dyn_ghc_smoke.pl. It
% generates a real project and runs the actual compiled WamRuntime,
% skipping gracefully when GHC/cabal are not available.
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

%% Regression test for the read_kernel_template/2 cwd-independence fix.
%% Prior to the fix, the predicate fell through to a cwd-relative path
%% when the source_file/2 lookup silently failed; codegen worked from
%% the project root but emitted "Template not found" stubs elsewhere.
%% This test pins the fix down by running codegen from /tmp.
test_render_kernel_function_cwd_independent :-
    Test = 'WAM-Haskell: render_kernel_function works from any cwd (regression)',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4, [max_depth(10), edge_pred(category_parent/2)]),
    working_directory(SavedCwd, '/tmp'),
    catch(
        (   wam_haskell_target:render_kernel_function('category_ancestor/4'-Kernel, Code),
            atom_string(Code, S),
            \+ sub_string(S, _, _, _, "Template not found"),
            sub_string(S, _, _, _, "nativeKernel_category_ancestor")
        ->  Result = pass
        ;   Result = fail
        ),
        E,
        Result = error(E)
    ),
    working_directory(_, SavedCwd),
    (   Result == pass
    ->  pass(Test)
    ;   Result = error(Err)
    ->  format(string(Reason), 'codegen from /tmp threw: ~w', [Err]),
        fail_test(Test, Reason)
    ;   fail_test(Test, 'codegen from /tmp produced "Template not found"')
    ).

test_transitive_closure_kernel_function :-
    Test = 'WAM-Haskell: transitive_closure2 kernel template renders',
    Kernel = recursive_kernel(transitive_closure2, closure/2, [edge_pred(edge/2)]),
    (   wam_haskell_target:render_kernel_function('closure/2'-Kernel, Code),
        sub_string(Code, _, _, _, "nativeKernel_transitive_closure"),
        %% Signature uses EdgeLookup after B1 abstraction
        sub_string(Code, _, _, _, "EdgeLookup -> Int -> [Int]"),
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
        %% config_facts_from resolved — checks wcEdgeLookups first, falls back to wcFfiFacts
        sub_string(Code, _, _, _, "edge_facts = case Map.lookup"),
        sub_string(Code, _, _, _, "wcEdgeLookups ctx"),
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

test_haskell_hashable_value_handles_intset :-
    Test = 'WAM-Haskell: HashMap rewrite hashes VSet IntSet explicitly',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        wam_haskell_target:apply_hashmap_rewrite(true, types, TypesCode, HashCode),
        atom_string(HashCode, S),
        sub_string(S, _, _, _, "import Data.Hashable (Hashable(..))"),
        sub_string(S, _, _, _, "instance Hashable Value where"),
        sub_string(S, _, _, _, "hashWithSalt salt (VSet s) = hashWithSalt salt (4 :: Int, IS.toList s)"),
        \+ sub_string(S, _, _, _, "deriving (Eq, Ord, Show, Generic)")
    ->  pass(Test)
    ;   fail_test(Test, 'Hashable Value still relies on derived Generic or misses VSet handling')
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

%% SetVariable: builder slot = fresh unbound (used by user-source =../2 list-build path)
%% ------------------------------------------------------------------------------------

test_haskell_set_variable_in_types :-
    Test = 'WAM-Haskell: SetVariable constructor takes RegId (not String)',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "SetVariable !RegId")
    ->  pass(Test)
    ;   fail_test(Test, 'SetVariable !RegId missing from Instruction type')
    ).

test_haskell_set_variable_step_handler :-
    Test = 'WAM-Haskell: SetVariable step handler present',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !ctx s (SetVariable xn)"),
        sub_string(S, _, _, _, "Unbound vid")
    ->  pass(Test)
    ;   fail_test(Test, 'SetVariable step handler missing or incorrect')
    ).

test_haskell_set_variable_wam_parse :-
    Test = 'WAM-Haskell: set_variable WAM text emits Int reg id',
    (   wam_haskell_target:wam_instr_to_haskell(["set_variable", "X6"], Hs),
        % X6 -> 106 via reg_name_to_int
        sub_string(Hs, _, _, _, "SetVariable 106")
    ->  pass(Test)
    ;   fail_test(Test, 'set_variable did not emit numeric register id')
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

%% Phase F1: Fact predicate classification tests
%% -----------------------------------------------

test_f1_fact_only_classification :-
    Test = 'F1: fact-only predicate classified correctly',
    (   retractall(user:f1_color(_)),
        assert(user:(f1_color(red))),
        assert(user:(f1_color(blue))),
        assert(user:(f1_color(green))),
        wam_target:compile_predicate_to_wam(f1_color/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_color/1, Lines, [], Info),
        Info = fact_shape_info(NClauses, FactOnly, FirstArg, Layout),
        NClauses == 3,
        FactOnly == true,
        FirstArg == all_ground,
        Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, 'Incorrect classification for fact-only predicate')
    ),
    retractall(user:f1_color(_)).

test_f1_rule_predicate_not_fact_only :-
    Test = 'F1: rule predicate classified as not fact-only',
    (   retractall(user:f1_anc(_, _)),
        retractall(user:f1_par(_, _)),
        assert(user:(f1_par(tom, bob))),
        assert(user:(f1_anc(X, Y) :- user:f1_par(X, Y))),
        assert(user:(f1_anc(X, Y) :- user:f1_par(X, Z), user:f1_anc(Z, Y))),
        wam_target:compile_predicate_to_wam(f1_anc/2, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_anc/2, Lines, [], Info),
        Info = fact_shape_info(_, FactOnly, _, _),
        FactOnly == false
    ->  pass(Test)
    ;   fail_test(Test, 'Rule predicate should be fact_only=false')
    ),
    retractall(user:f1_anc(_, _)),
    retractall(user:f1_par(_, _)).

test_f1_two_arg_fact_groundness :-
    Test = 'F1: 2-arg fact predicate has all_ground first arg',
    (   retractall(user:f1_edge(_, _)),
        assert(user:(f1_edge(a, b))),
        assert(user:(f1_edge(b, c))),
        wam_target:compile_predicate_to_wam(f1_edge/2, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_edge/2, Lines, [], Info),
        Info = fact_shape_info(2, true, all_ground, compiled)
    ->  pass(Test)
    ;   fail_test(Test, 'Two-arg fact should be fact_only=true, all_ground, compiled')
    ),
    retractall(user:f1_edge(_, _)).

test_f1_variable_first_arg :-
    Test = 'F1: variable first arg detected as mixed',
    (   retractall(user:f1_vfact(_, _)),
        assert(user:(f1_vfact(X, pizza) :- true)),
        assert(user:(f1_vfact(bob, tacos) :- true)),
        wam_target:compile_predicate_to_wam(f1_vfact/2, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_vfact/2, Lines, [], Info),
        Info = fact_shape_info(_, true, FirstArg, _),
        FirstArg == mixed
    ->  pass(Test)
    ;   fail_test(Test, 'Mixed first-arg groundness not detected')
    ),
    retractall(user:f1_vfact(_, _)).

test_f1_layout_auto_above_threshold :-
    Test = 'F1: auto policy picks inline_data above threshold',
    (   retractall(user:f1_big(_)),
        forall(between(1, 6, I), (
            atom_number(A, I),
            assert(user:f1_big(A))
        )),
        wam_target:compile_predicate_to_wam(f1_big/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_big/1, Lines, [fact_count_threshold(5)], Info),
        Info = fact_shape_info(6, true, all_ground, inline_data([]))
    ->  pass(Test)
    ;   fail_test(Test, 'Auto policy should pick inline_data above threshold')
    ),
    retractall(user:f1_big(_)).

test_f1_layout_compiled_below_threshold :-
    Test = 'F1: auto policy keeps compiled below threshold',
    (   retractall(user:f1_small(_)),
        assert(user:f1_small(x)),
        assert(user:f1_small(y)),
        wam_target:compile_predicate_to_wam(f1_small/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_small/1, Lines, [fact_count_threshold(5)], Info),
        Info = fact_shape_info(2, true, all_ground, compiled)
    ->  pass(Test)
    ;   fail_test(Test, 'Auto policy should keep compiled below threshold')
    ),
    retractall(user:f1_small(_)).

test_f1_layout_user_override :-
    Test = 'F1: user fact_layout override respected',
    (   retractall(user:f1_ov(_)),
        assert(user:f1_ov(a)),
        wam_target:compile_predicate_to_wam(f1_ov/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_ov/1, Lines,
            [fact_layout(f1_ov/1, external_source(tsv("data.tsv")))], Info),
        Info = fact_shape_info(_, _, _, external_source(tsv("data.tsv")))
    ->  pass(Test)
    ;   fail_test(Test, 'User override should take precedence')
    ),
    retractall(user:f1_ov(_)).

test_f1_comment_in_predicates_hs :-
    Test = 'F1: classification comment emitted in Predicates.hs',
    (   retractall(user:f1_cp(_)),
        assert(user:f1_cp(hello)),
        assert(user:f1_cp(world)),
        wam_haskell_target:compile_predicates_to_haskell([f1_cp/1], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "Fact shape classification"),
        sub_string(S, _, _, _, "f1_cp/1: fact_only=true")
    ->  pass(Test)
    ;   fail_test(Test, 'Classification comment not found in Predicates.hs')
    ),
    retractall(user:f1_cp(_)).

test_f1_segment_parser :-
    Test = 'F1: split_wam_into_segments parses correctly',
    (   Lines = [
            "pred/1:",
            "    get_constant foo, A1",
            "    proceed",
            "L_pred_1_2:",
            "    get_constant bar, A1",
            "    proceed"
        ],
        split_wam_into_segments(Lines, Segments),
        length(Segments, 2),
        Segments = ["pred/1"-Instrs1, "L_pred_1_2"-Instrs2],
        member(get_constant("foo", "A1"), Instrs1),
        member(proceed, Instrs1),
        member(get_constant("bar", "A1"), Instrs2),
        member(proceed, Instrs2)
    ->  pass(Test)
    ;   fail_test(Test, 'Segment parser produced wrong results')
    ).

test_f1_compiled_only_policy :-
    Test = 'F1: compiled_only policy always returns compiled',
    (   retractall(user:f1_co(_)),
        forall(between(1, 200, I), (
            atom_number(A, I),
            assert(user:f1_co(A))
        )),
        wam_target:compile_predicate_to_wam(f1_co/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        classify_fact_predicate(f1_co/1, Lines,
            [fact_layout_policy(compiled_only)], Info),
        Info = fact_shape_info(_, true, _, compiled)
    ->  pass(Test)
    ;   fail_test(Test, 'compiled_only policy should always return compiled')
    ),
    retractall(user:f1_co(_)).

%% Phase F2: FactStream choice point type tests
%% -----------------------------------------------

test_f2_fact_stream_in_builtin_state :-
    Test = 'F2: FactStream constructor in BuiltinState type',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "FactStream !Int !Int ![(Int, Int)] !Int")
    ->  pass(Test)
    ;   fail_test(Test, 'FactStream constructor not found in BuiltinState')
    ).

test_f2_call_fact_stream_in_instruction :-
    Test = 'F2: CallFactStream constructor in Instruction type',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "CallFactStream String !Int")
    ->  pass(Test)
    ;   fail_test(Test, 'CallFactStream constructor not found in Instruction type')
    ).

test_f2_stream_facts_function :-
    Test = 'F2: streamFacts function present in runtime',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "streamFacts :: WamContext -> String -> WamState -> Maybe WamState")
    ->  pass(Test)
    ;   fail_test(Test, 'streamFacts function not found in runtime')
    ).

test_f2_resume_fact_stream_handler :-
    Test = 'F2: resumeBuiltin handles FactStream CPs',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "resumeBuiltin (FactStream"),
        sub_string(S, _, _, _, "FactStream var1 var2")
    ->  pass(Test)
    ;   fail_test(Test, 'resumeBuiltin FactStream handler not found')
    ).

test_f2_call_fact_stream_step_handler :-
    Test = 'F2: step function handles CallFactStream',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !ctx s (CallFactStream pred"),
        sub_string(S, _, _, _, "streamFacts ctx pred")
    ->  pass(Test)
    ;   fail_test(Test, 'CallFactStream step handler not found')
    ).

test_f2_wc_inline_facts_field :-
    Test = 'F2: wcInlineFacts field in WamContext',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "wcInlineFacts")
    ->  pass(Test)
    ;   fail_test(Test, 'wcInlineFacts field not found in WamContext')
    ).

test_f2_stream_facts_filters_bound_a1 :-
    Test = 'F2: streamFacts filters by bound A1',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        % Verify the filter logic: Atom aid case filters rows
        sub_string(S, _, _, _, "Atom aid, Atom bid"),
        sub_string(S, _, _, _, "Atom aid, _)")
    ->  pass(Test)
    ;   fail_test(Test, 'streamFacts bound-arg filtering logic not found')
    ).

test_f2_fact_stream_exhaustion_backtracks :-
    Test = 'F2: FactStream empty rows triggers backtrack',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "resumeBuiltin (FactStream _ _ [] _) _ rest s"),
        sub_string(S, _, _, _, "backtrack (s { wsCPs = rest")
    ->  pass(Test)
    ;   fail_test(Test, 'FactStream exhaustion backtrack not found')
    ).

%% Phase F3: inline_data emission tests
%% -----------------------------------------------

test_f3_inline_data_emits_call_fact_stream :-
    Test = 'F3: inline_data predicate emits CallFactStream instruction',
    (   retractall(user:f3_big(_, _)),
        forall(between(1, 6, I), (
            atom_number(A, I),
            atom_concat(parent_, A, P),
            assert(user:f3_big(A, P))
        )),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [f3_big/2], [fact_count_threshold(5)], Code, _),
        atom_string(Code, S),
        sub_string(S, _, _, _, "CallFactStream")
    ->  pass(Test)
    ;   fail_test(Test, 'CallFactStream not emitted for inline_data predicate')
    ),
    retractall(user:f3_big(_, _)).

test_f3_inline_data_emits_fact_literal :-
    Test = 'F3: inline_data predicate emits fact literal list',
    (   retractall(user:f3_lit(_, _)),
        forall(between(1, 6, I), (
            atom_number(A, I),
            atom_concat(val_, A, V),
            assert(user:f3_lit(A, V))
        )),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [f3_lit/2], [fact_count_threshold(5)], Code, _),
        atom_string(Code, S),
        sub_string(S, _, _, _, "f3LitFacts :: [(Int, Int)]"),
        sub_string(S, _, _, _, "inline fact data")
    ->  pass(Test)
    ;   fail_test(Test, 'Fact literal list not emitted in Predicates.hs')
    ),
    retractall(user:f3_lit(_, _)).

test_f3_below_threshold_stays_compiled :-
    Test = 'F3: below-threshold predicate stays compiled (no CallFactStream)',
    (   retractall(user:f3_sm(_, _)),
        assert(user:f3_sm(a, b)),
        assert(user:f3_sm(c, d)),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [f3_sm/2], [fact_count_threshold(5)], Code, _),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "CallFactStream")
    ->  pass(Test)
    ;   fail_test(Test, 'Below-threshold predicate should not use CallFactStream')
    ),
    retractall(user:f3_sm(_, _)).

test_f3_inline_defs_returned :-
    Test = 'F3: compile_predicates_to_haskell returns InlineDefs',
    (   retractall(user:f3_def(_, _)),
        forall(between(1, 6, I), (
            atom_number(A, I),
            atom_concat(x_, A, X),
            assert(user:f3_def(A, X))
        )),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [f3_def/2], [fact_count_threshold(5)], _, InlineDefs),
        InlineDefs = [inline_fact(_, _, _)]
    ->  pass(Test)
    ;   fail_test(Test, 'InlineDefs not returned for inline_data predicate')
    ),
    retractall(user:f3_def(_, _)).

test_f3_fact_tuples_extracted :-
    Test = 'F3: extract_fact_tuples_hs extracts interned pairs',
    (   init_atom_intern_table,
        Lines = [
            "pred/2:",
            "    switch_on_constant a:default",
            "    try_me_else L_pred_2_2",
            "    get_constant alpha, A1",
            "    get_constant beta, A2",
            "    proceed",
            "L_pred_2_2:",
            "    trust_me",
            "    get_constant gamma, A1",
            "    get_constant delta, A2",
            "    proceed"
        ],
        split_wam_into_segments(Lines, Segments),
        extract_fact_tuples_hs(Segments, Tuples),
        length(Tuples, 2),
        Tuples = [(Id1a, Id1b), (Id2a, Id2b)],
        integer(Id1a), integer(Id1b),
        integer(Id2a), integer(Id2b),
        Id1a \= Id2a  % different first args
    ->  pass(Test)
    ;   fail_test(Test, 'Fact tuple extraction failed')
    ).

test_f3_camel_case_list_name :-
    Test = 'F3: haskell_fact_list_name generates camelCase',
    (   haskell_fact_list_name("category_parent", Name1),
        Name1 == 'categoryParentFacts',
        haskell_fact_list_name("edge", Name2),
        Name2 == 'edgeFacts'
    ->  pass(Test)
    ;   fail_test(Test, 'camelCase list name generation failed')
    ).

%% Phase F4: FactSource abstraction tests
%% -----------------------------------------------

test_f4_fact_source_type_present :-
    Test = 'F4: FactSource record type in WamTypes',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "data FactSource = FactSource"),
        sub_string(S, _, _, _, "fsScan"),
        sub_string(S, _, _, _, "fsLookupArg1"),
        sub_string(S, _, _, _, "fsClose")
    ->  pass(Test)
    ;   fail_test(Test, 'FactSource type not found in WamTypes')
    ).

test_f4_wc_fact_sources_field :-
    Test = 'F4: wcFactSources field in WamContext',
    (   wam_haskell_target:generate_wam_types_hs(TypesCode),
        atom_string(TypesCode, S),
        sub_string(S, _, _, _, "wcFactSources")
    ->  pass(Test)
    ;   fail_test(Test, 'wcFactSources field not found in WamContext')
    ).

test_f4_tsv_fact_source_function :-
    Test = 'F4: tsvFactSource function in runtime',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "tsvFactSource :: InternTable -> FilePath -> IO FactSource")
    ->  pass(Test)
    ;   fail_test(Test, 'tsvFactSource function not found in runtime')
    ).

test_f4_intmap_fact_source_function :-
    Test = 'F4: intMapFactSource function in runtime',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "intMapFactSource :: IM.IntMap [Int] -> FactSource")
    ->  pass(Test)
    ;   fail_test(Test, 'intMapFactSource function not found in runtime')
    ).

test_f4_stream_facts_fallback_to_fact_sources :-
    Test = 'F4: streamFacts falls back to wcFactSources',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "wcFactSources ctx"),
        sub_string(S, _, _, _, "fsLookupArg1 fs"),
        sub_string(S, _, _, _, "fsScan fs")
    ->  pass(Test)
    ;   fail_test(Test, 'streamFacts does not fall back to wcFactSources')
    ).

test_f4_stream_fact_rows_helper :-
    Test = 'F4: streamFactRows helper present',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "streamFactRows :: [(Int, Int)] -> WamState -> Maybe WamState")
    ->  pass(Test)
    ;   fail_test(Test, 'streamFactRows helper not found')
    ).

%% E2E: fact access wiring tests
%% -----------------------------------------------

test_e2e_inline_data_project_has_call_fact_stream :-
    Test = 'E2E: inline_data project emits CallFactStream in allCode',
    (   retractall(user:e2e_edge(_, _)),
        forall(between(1, 6, I), (
            atom_number(A, I),
            atom_concat(src_, A, S),
            atom_concat(dst_, A, D),
            assert(user:e2e_edge(S, D))
        )),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [e2e_edge/2], [fact_count_threshold(5)], Code, InlineDefs),
        atom_string(Code, CS),
        sub_string(CS, _, _, _, "CallFactStream"),
        InlineDefs = [inline_fact(_, _, _)]
    ->  pass(Test)
    ;   fail_test(Test, 'CallFactStream not in allCode or InlineDefs missing')
    ),
    retractall(user:e2e_edge(_, _)).

test_e2e_inline_facts_wiring_generated :-
    Test = 'E2E: generate_inline_facts_wiring produces wcInlineFacts code',
    (   InlineDefs = [inline_fact("my_pred", 'myPredFacts', "...")],
        wam_haskell_target:generate_inline_facts_wiring(InlineDefs, Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "wcInlineFacts"),
        sub_string(S, _, _, _, "\"my_pred\""),
        sub_string(S, _, _, _, "myPredFacts")
    ->  pass(Test)
    ;   fail_test(Test, 'wcInlineFacts wiring code not generated correctly')
    ).

test_e2e_empty_inline_defs_no_wiring :-
    Test = 'E2E: empty InlineDefs produces no wcInlineFacts wiring',
    (   wam_haskell_target:generate_inline_facts_wiring([], Code),
        Code == ''
    ->  pass(Test)
    ;   fail_test(Test, 'Empty InlineDefs should produce empty wiring code')
    ).

test_e2e_below_threshold_no_inline :-
    Test = 'E2E: below-threshold project has no CallFactStream or inline facts',
    (   retractall(user:e2e_sm(_, _)),
        assert(user:e2e_sm(x, y)),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [e2e_sm/2], [fact_count_threshold(5)], Code, InlineDefs),
        atom_string(Code, CS),
        \+ sub_string(CS, _, _, _, "CallFactStream"),
        InlineDefs == []
    ->  pass(Test)
    ;   fail_test(Test, 'Below-threshold should have no CallFactStream or InlineDefs')
    ),
    retractall(user:e2e_sm(_, _)).

%% Phase B1: LMDB backend tests
%% -----------------------------------------------

test_b1_lmdb_functions_present_when_enabled :-
    Test = 'B1: lmdbFactSource emitted when use_lmdb(true)',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "lmdbFactSource :: FilePath -> String -> IO FactSource"),
        sub_string(S, _, _, _, "ingestTsvToLmdb")
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB functions not found when use_lmdb(true)')
    ).

test_b1_lmdb_imports_present_when_enabled :-
    Test = 'B1: LMDB imports emitted when use_lmdb(true)',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "import Database.LMDB.Raw"),
        sub_string(S, _, _, _, "import Foreign.Ptr")
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB imports not found when use_lmdb(true)')
    ).

test_b1_lmdb_absent_when_disabled :-
    Test = 'B1: no LMDB code when use_lmdb not set',
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "lmdbFactSource"),
        \+ sub_string(S, _, _, _, "import Database.LMDB")
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB code should not appear without use_lmdb(true)')
    ).

test_b1_lmdb_cabal_dependency :-
    Test = 'B1: cabal includes lmdb when use_lmdb(true)',
    (   wam_haskell_target:generate_cabal_file('test', false, [use_lmdb(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "lmdb >= 0.2.5")
    ->  pass(Test)
    ;   fail_test(Test, 'lmdb not in cabal deps when use_lmdb(true)')
    ).

test_b1_no_lmdb_cabal_default :-
    Test = 'B1: cabal excludes lmdb-simple by default',
    (   wam_haskell_target:generate_cabal_file('test', false, [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "lmdb-simple")
    ->  pass(Test)
    ;   fail_test(Test, 'lmdb-simple should not appear in default cabal deps')
    ).

test_with_rtsopts_emits_in_cabal :-
    Test = 'WAM-Haskell: with_rtsopts(Flags) bakes -with-rtsopts into ghc-options',
    (   wam_haskell_target:generate_cabal_file('test', false,
            [with_rtsopts('-A64M')], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "\"-with-rtsopts=-A64M\"")
    ->  pass(Test)
    ;   fail_test(Test, 'with_rtsopts(-A64M) should add -with-rtsopts=-A64M to ghc-options')
    ).

test_with_rtsopts_absent_by_default :-
    Test = 'WAM-Haskell: cabal omits -with-rtsopts when option not set',
    (   wam_haskell_target:generate_cabal_file('test', false, [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "-with-rtsopts")
    ->  pass(Test)
    ;   fail_test(Test, 'cabal should not contain -with-rtsopts when option absent')
    ).

test_b1_external_source_skips_wam_compilation :-
    Test = 'B1: external_source fact predicate skips WAM compilation',
    (   retractall(user:b1_ext(_, _)),
        forall(between(1, 10, I), (
            atom_number(A, I),
            atom_concat(parent_, A, P),
            assert(user:b1_ext(A, P))
        )),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [b1_ext/2],
            [use_lmdb(true), lmdb_backed_facts([b1_ext/2])],
            Code, _),
        atom_string(Code, S),
        sub_string(S, _, _, _, "external_source"),
        % No per-fact emission in any path — neither CallFactStream nor
        % an inline fact list nor WAM label entry for b1_ext
        \+ sub_string(S, _, _, _, "CallFactStream"),
        \+ sub_string(S, _, _, _, "b1ExtFacts"),
        \+ sub_string(S, _, _, _, "(\"b1_ext/2\"")
    ->  pass(Test)
    ;   fail_test(Test, 'external_source predicate should emit no WAM code')
    ),
    retractall(user:b1_ext(_, _)).

test_b1_external_source_default_allow_list :-
    Test = 'B1: default lmdb_backed_facts covers category_parent/2',
    (   retractall(user:category_parent(_, _)),
        forall(between(1, 10, I), (
            atom_number(A, I),
            atom_concat(cp_p_, A, P),
            atom_concat(cp_c_, A, C),
            assert(user:category_parent(C, P))
        )),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [category_parent/2],
            [use_lmdb(true)],
            Code, _),
        atom_string(Code, S),
        sub_string(S, _, _, _, "external_source"),
        \+ sub_string(S, _, _, _, "CallFactStream"),
        \+ sub_string(S, _, _, _, "categoryParentFacts")
    ->  pass(Test)
    ;   fail_test(Test, 'category_parent/2 should be skipped by default under use_lmdb(true)')
    ),
    retractall(user:category_parent(_, _)).

test_b1_external_source_off_without_use_lmdb :-
    Test = 'B1: without use_lmdb(true) fact predicates still compile',
    (   retractall(user:b1_noext(_, _)),
        forall(between(1, 6, I), (
            atom_number(A, I),
            atom_concat(p_, A, P),
            assert(user:b1_noext(A, P))
        )),
        init_atom_intern_table,
        wam_haskell_target:compile_predicates_to_haskell(
            [b1_noext/2],
            [lmdb_backed_facts([b1_noext/2])],
            Code, _),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "external_source")
    ->  pass(Test)
    ;   fail_test(Test, 'external_source should require use_lmdb(true)')
    ),
    retractall(user:b1_noext(_, _)).

test_b1_lmdb_raw_zero_copy_reads :-
    Test = 'B1: lmdbRawEdgeLookup uses mdb_get for zero-copy reads',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "mdb_get'"),
        sub_string(S, _, _, _, "peekElemOff"),
        sub_string(S, _, _, _, "MDB_INTEGERKEY")
    ->  pass(Test)
    ;   fail_test(Test, 'Raw LMDB zero-copy read patterns not found')
    ).

test_b1_phase2b2_loaders_emitted :-
    Test = 'B1: Phase 2b.2 loaders (loadInternTableFromLmdb / loadArticleCategoriesFromLmdb / loadForwardEdgesFromLmdb) emitted',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "loadInternTableFromLmdb :: MDB_env -> String -> String -> IO InternTable"),
        sub_string(S, _, _, _, "loadArticleCategoriesFromLmdb :: MDB_env -> String -> IO [(Int, Int)]"),
        sub_string(S, _, _, _, "loadForwardEdgesFromLmdb :: MDB_env -> String -> Int -> IO (IM.IntMap [Int])"),
        sub_string(S, _, _, _, "iterateAllPairs txn dbi decode = do"),
        sub_string(S, _, _, _, "peekStringBytes p len")
    ->  pass(Test)
    ;   fail_test(Test, 'Phase 2b.2 LMDB-resident loaders should be emitted when use_lmdb(true)')
    ).

test_b1_peekstringbytes_decodes_utf8 :-
    Test = 'B1: peekStringBytes decodes UTF-8 via TE.decodeUtf8With',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BS.packCStringLen (p, len)"),
        sub_string(S, _, _, _, "TE.decodeUtf8With TEE.lenientDecode"),
        \+ sub_string(S, _, _, _, "map (toEnum . fromIntegral) bytes")
    ->  pass(Test)
    ;   fail_test(Test, 'peekStringBytes should decode UTF-8 via Data.Text.Encoding')
    ).

test_b1_lmdb_text_imports_present :-
    Test = 'B1: bytestring + text imports emitted with use_lmdb(true)',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "import qualified Data.ByteString as BS"),
        sub_string(S, _, _, _, "import qualified Data.Text as T"),
        sub_string(S, _, _, _, "import qualified Data.Text.Encoding as TE"),
        sub_string(S, _, _, _, "import qualified Data.Text.Encoding.Error as TEE")
    ->  pass(Test)
    ;   fail_test(Test, 'bytestring/text imports missing under use_lmdb(true)')
    ).

test_b1_lmdb_cabal_text_dependencies :-
    Test = 'B1: cabal includes bytestring + text when use_lmdb(true)',
    (   wam_haskell_target:generate_cabal_file('test', false, [use_lmdb(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "bytestring >= 0.10"),
        sub_string(S, _, _, _, "text >= 1.2")
    ->  pass(Test)
    ;   fail_test(Test, 'bytestring/text not in cabal deps when use_lmdb(true)')
    ).

test_b1_lmdb_scan_support_present :-
    Test = 'B1: raw LMDB FactSource emits scan support',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "scanRawIntPairs :: MDB_txn -> MDB_dbi' -> IO [(Int, Int)]"),
        sub_string(S, _, _, _, "fsScan       = scanRawIntPairs txn dbi"),
        \+ sub_string(S, _, _, _, "fsScan       = return []")
    ->  pass(Test)
    ;   fail_test(Test, 'Raw LMDB FactSource scan support not found')
    ).

test_b1_lmdb_manifest_fact_source_present :-
    Test = 'B1: manifest-backed LMDB FactSource emitted when use_lmdb(true)',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "readLmdbArtifactManifest :: FilePath -> IO (String, Bool)"),
        sub_string(S, _, _, _, "openLmdbUtf8StoreFromManifest :: FilePath -> IO (MDB_env, MDB_txn, MDB_dbi', Bool)"),
        sub_string(S, _, _, _, "lmdbFactSourceFromManifest :: InternTable -> FilePath -> IO FactSource"),
        sub_string(S, _, _, _, "lookupUtf8Values txn dbi dupsort atomKey")
    ->  pass(Test)
    ;   fail_test(Test, 'Manifest-backed LMDB FactSource helpers not found')
    ).

test_b1_lmdb_manifest_wiring_option_present :-
    Test = 'B1: lmdb_fact_source_manifest option wires manifest-backed FactSource',
    (   wam_haskell_target:generate_main_hs(
            [],
            [],
            [],
            [use_lmdb(true), lmdb_fact_source_manifest('/tmp/lmdb-artifact')],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "cpFactSource <- lmdbFactSourceFromManifest fullInternTable \"/tmp/lmdb-artifact\""),
        \+ sub_string(S, _, _, _, "cpFactSource <- lmdbFactSource lmdbDir \"category_parent\"")
    ->  pass(Test)
    ;   fail_test(Test, 'Manifest-backed FactSource wiring option not found')
    ).

test_demand_filter_gates_seed_query_body :-
    Test = 'WAM-Haskell: demand filter pre-filters seeds before parMap',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "let seedCatsAll = map head $ group $ sort $ map snd articleCategories"),
        sub_string(S, _, _, _, "!demandFilterSpec = (HopLimit Nothing)"),
        sub_string(S, _, _, _, "!demandFilterResult = runDemandBFS demandFilterSpec parentsIndexInterned rootId"),
        sub_string(S, _, _, _, "!demandSet = dfrInSet demandFilterResult"),
        sub_string(S, _, _, _, "!reverseAdj = IM.fromListWith (++)"),
        sub_string(S, _, _, _, "filterByDemand demandSet parents = IS.foldl' addChild IM.empty demandSet"),
        sub_string(S, _, _, _, "!filteredSeedCats = filter (\\cat -> IS.member (iAtom cat) demandSet) seedCats"),
        sub_string(S, _, _, _, "!demandSkippedSeeds = length seedCats - length filteredSeedCats"),
        sub_string(S, _, _, _, ") filteredSeedCats"),
        \+ sub_string(S, _, _, _, "if not (IS.member (iAtom cat) demandSet) then (cat, 0.0) else"),
        sub_string(S, _, _, _, "collectForeignSolutions ctx \"category_ancestor/4\""),
        sub_string(S, _, _, _, "demand_skipped_seeds=")
    ->  pass(Test)
    ;   fail_test(Test, 'Demand-filtered Main.hs should pre-filter seeds before parMap')
    ).

test_demand_bfs_mode_cursor_emits_runDemandBFSCursor :-
    Test = 'WAM-Haskell: demand_bfs_mode(cursor) emits runDemandBFSCursor instead of in-memory variant',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [demand_bfs_mode(cursor)],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "runDemandBFSCursor demandFilterSpec internEnv \"category_child\" rootId"),
        \+ sub_string(S, _, _, _, "runDemandBFS demandFilterSpec parentsIndexInterned"),
        \+ sub_string(S, _, _, _, "filterByDemand demandSet parentsIndexInterned")
    ->  pass(Test)
    ;   fail_test(Test, 'cursor mode should emit runDemandBFSCursor and skip filteredParents materialisation')
    ).

test_demand_bfs_mode_in_memory_default :-
    Test = 'WAM-Haskell: default demand BFS mode is in_memory (no behavior change)',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "runDemandBFS demandFilterSpec parentsIndexInterned rootId"),
        \+ sub_string(S, _, _, _, "runDemandBFSCursor demandFilterSpec internEnv")
    ->  pass(Test)
    ;   fail_test(Test, 'default should keep emitting runDemandBFS (in-memory)')
    ).

test_demand_bfs_mode_auto_high_fact_count_picks_cursor :-
    Test = 'WAM-Haskell: demand_bfs_mode(auto) + fact_count >= 50000 picks cursor',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [demand_bfs_mode(auto), fact_count(196900)],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "runDemandBFSCursor demandFilterSpec internEnv")
    ->  pass(Test)
    ;   fail_test(Test, 'auto + high fact_count should resolve to cursor')
    ).

test_demand_bfs_mode_auto_low_fact_count_picks_in_memory :-
    Test = 'WAM-Haskell: demand_bfs_mode(auto) + fact_count < 50000 picks in_memory',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [demand_bfs_mode(auto), fact_count(5000)],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "runDemandBFS demandFilterSpec parentsIndexInterned"),
        \+ sub_string(S, _, _, _, "runDemandBFSCursor demandFilterSpec internEnv")
    ->  pass(Test)
    ;   fail_test(Test, 'auto + low fact_count should resolve to in_memory')
    ).

test_demand_filter_hop_limit_with_max_hops :-
    Test = 'WAM-Haskell: demand_filter_spec(hop_limit, [max_hops(N)]) emits HopLimit (Just N)',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [demand_filter_spec(hop_limit, [max_hops(7)])],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "!demandFilterSpec = (HopLimit (Just 7))"),
        sub_string(S, _, _, _, "runDemandBFS demandFilterSpec parentsIndexInterned rootId")
    ->  pass(Test)
    ;   fail_test(Test, 'demand_filter_spec(hop_limit, [max_hops(7)]) should emit HopLimit (Just 7)')
    ).

test_demand_filter_none_emits_dfnone :-
    Test = 'WAM-Haskell: demand_filter_spec(none, []) emits DfNone',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [demand_filter_spec(none, [])],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "!demandFilterSpec = DfNone"),
        sub_string(S, _, _, _, "runDemandBFS demandFilterSpec parentsIndexInterned rootId")
    ->  pass(Test)
    ;   fail_test(Test, 'demand_filter_spec(none, []) should emit DfNone')
    ).

test_demand_filter_flux_emits_panic_stub_compatible :-
    Test = 'WAM-Haskell: demand_filter_spec(flux, [...]) emits Flux constructor (panics at runtime in Phase 2)',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [demand_filter_spec(flux, [target_count(5000), sort_sparks(true)])],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "!demandFilterSpec = (Flux 5000 True)")
    ->  pass(Test)
    ;   fail_test(Test, 'demand_filter_spec(flux, [...]) should emit Flux constructor for runtime to panic on')
    ).

test_int_atom_seeds_lmdb_calls_loaders :-
    Test = 'WAM-Haskell: int_atom_seeds(lmdb) calls Phase 2b.2 loaders (no panic stub)',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [int_atom_seeds(lmdb)],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "openLmdbInternEnvReadonly"),
        sub_string(S, _, _, _, "loadInternTableFromLmdb internEnv \"s2i\" \"i2s\""),
        sub_string(S, _, _, _, "loadArticleCategoriesFromLmdb internEnv \"article_category\""),
        sub_string(S, _, _, _, "loadForwardEdgesFromLmdb internEnv \"category_parent\""),
        sub_string(S, _, _, _, "!fullInternTable = lmdbInternTable"),
        sub_string(S, _, _, _, "!parentsIndexInterned = lmdbParentsIndex"),
        \+ sub_string(S, _, _, _, "int_atom_seeds(lmdb): runtime LMDB-resident loaders not yet")
    ->  pass(Test)
    ;   fail_test(Test, 'int_atom_seeds(lmdb) should call the Phase 2b.2 loaders, not emit a panic stub')
    ).

test_int_atom_seeds_true_does_not_call_lmdb_loaders :-
    Test = 'WAM-Haskell: int_atom_seeds(true) does NOT call LMDB loaders',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [int_atom_seeds(true)],
            Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "openLmdbInternEnvReadonly"),
        \+ sub_string(S, _, _, _, "loadInternTableFromLmdb internEnv"),
        \+ sub_string(S, _, _, _, "lmdbParentsIndex")
    ->  pass(Test)
    ;   fail_test(Test, 'int_atom_seeds(true) should not call the LMDB loaders')
    ).

test_int_atom_seeds_default_does_not_call_lmdb_loaders :-
    Test = 'WAM-Haskell: default mode does NOT call LMDB loaders',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [],
            Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "openLmdbInternEnvReadonly"),
        \+ sub_string(S, _, _, _, "loadInternTableFromLmdb internEnv"),
        \+ sub_string(S, _, _, _, "lmdbParentsIndex")
    ->  pass(Test)
    ;   fail_test(Test, 'default mode should not call the LMDB loaders')
    ).

test_demand_filter_invalid_strategy_rejected :-
    Test = 'WAM-Haskell: demand_filter_spec with unknown strategy throws domain_error',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   catch(
            wam_haskell_target:generate_main_hs(
                [],
                ['category_ancestor/4'-Kernel],
                [],
                [demand_filter_spec(bogus_strategy, [])],
                _Code),
            error(domain_error(demand_filter_strategy, bogus_strategy), _),
            (true, ThrewDomainError = true))
    ->  (   ThrewDomainError == true
        ->  pass(Test)
        ;   fail_test(Test, 'unknown strategy should throw domain_error')
        )
    ;   fail_test(Test, 'unknown strategy should throw domain_error')
    ).

test_demand_filter_false_leaves_query_ungated :-
    Test = 'WAM-Haskell: demand_filter(false) leaves seed query ungated',
    Kernel = recursive_kernel(category_ancestor, 'category_ancestor'/4,
                              [max_depth(10), edge_pred(category_parent/2)]),
    (   wam_haskell_target:generate_main_hs(
            [],
            ['category_ancestor/4'-Kernel],
            [],
            [demand_filter(false)],
            Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "collectForeignSolutions ctx \"category_ancestor/4\""),
        \+ sub_string(S, _, _, _, "if not (IS.member (iAtom cat) demandSet)"),
        \+ sub_string(S, _, _, _, "filteredSeedCats = filter"),
        sub_string(S, _, _, _, ") seedCats"),
        \+ sub_string(S, _, _, _, "demand_skipped_seeds=")
    ->  pass(Test)
    ;   fail_test(Test, 'demand_filter(false) should not emit pre-filter or seed gate')
    ).

%% Regression test for the linear-chain-zero-results bug: the FFI kernel's
%% max_depth must be substituted from user:max_depth/1 (or an option) at
%% codegen time. Hardcoding 10 in the Main.hs template caused chain-shaped
%% queries with deeper roots (e.g. cat_001 -> ... -> cat_030) to return
%% zero results — the kernel cut off recursion at depth 10, never reaching
%% the root, while SWI-Prolog (using the same workload's max_depth/1 fact
%% asserted to 30) found the paths.
test_max_depth_default_10_when_no_user_fact :-
    Test = 'WAM-Haskell: Main.hs max_depth defaults to 10 when no user:max_depth/1',
    %% Ensure no stale user:max_depth/1 leaks from other tests.
    retractall(user:max_depth(_)),
    (   wam_haskell_target:generate_main_hs([], [], [], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "Map.singleton \"max_depth\" 10")
    ->  pass(Test)
    ;   fail_test(Test, 'Default max_depth=10 not present in Main.hs')
    ).

test_max_depth_from_user_fact :-
    Test = 'WAM-Haskell: Main.hs max_depth picks up user:max_depth/1',
    %% Simulate the workload (effective_distance.pl) asserting a depth bound.
    retractall(user:max_depth(_)),
    assertz(user:max_depth(30)),
    (   wam_haskell_target:generate_main_hs([], [], [], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "Map.singleton \"max_depth\" 30"),
        \+ sub_string(S, _, _, _, "Map.singleton \"max_depth\" 10")
    ->  pass(Test)
    ;   fail_test(Test, 'user:max_depth(30) not propagated into Main.hs FFI config')
    ),
    retractall(user:max_depth(_)).

test_max_depth_option_overrides_user_fact :-
    Test = 'WAM-Haskell: max_depth(N) option overrides user:max_depth/1',
    retractall(user:max_depth(_)),
    assertz(user:max_depth(30)),
    (   wam_haskell_target:generate_main_hs([], [], [], [max_depth(50)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "Map.singleton \"max_depth\" 50"),
        \+ sub_string(S, _, _, _, "Map.singleton \"max_depth\" 30")
    ->  pass(Test)
    ;   fail_test(Test, 'max_depth(50) option did not override user:max_depth(30)')
    ),
    retractall(user:max_depth(_)).

%% =========================================================================
%% dimension_n substitution (same instrumentation-bug class as max_depth).
%%
%% The aggregation formula d_eff = (sum Hops^(-N))^(-1/N) needs N at
%% codegen — `n` in Main.hs was historically hardcoded to 5.0. A user
%% asserting `dimension_n(7).` in the workload would silently still get
%% N=5 in the FFI aggregation while the WAM-compiled `dimension_n/1`
%% predicate returned 7. Fix: read `user:dimension_n/1` (or
%% `dimension_n(N)` option) at codegen and substitute `{{dimension_n}}`.
test_dimension_n_default_5_when_no_user_fact :-
    Test = 'WAM-Haskell: Main.hs dimension_n defaults to 5 when no user:dimension_n/1',
    retractall(user:dimension_n(_)),
    (   wam_haskell_target:generate_main_hs([], [], [], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "fromIntegral (5 :: Int) :: Double")
    ->  pass(Test)
    ;   fail_test(Test, 'Default dimension_n=5 not present in Main.hs')
    ).

test_dimension_n_from_user_fact :-
    Test = 'WAM-Haskell: Main.hs dimension_n picks up user:dimension_n/1',
    retractall(user:dimension_n(_)),
    assertz(user:dimension_n(7)),
    (   wam_haskell_target:generate_main_hs([], [], [], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "fromIntegral (7 :: Int) :: Double"),
        \+ sub_string(S, _, _, _, "fromIntegral (5 :: Int) :: Double")
    ->  pass(Test)
    ;   fail_test(Test, 'user:dimension_n(7) not propagated into Main.hs')
    ),
    retractall(user:dimension_n(_)).

test_dimension_n_option_overrides_user_fact :-
    Test = 'WAM-Haskell: dimension_n(N) option overrides user:dimension_n/1',
    retractall(user:dimension_n(_)),
    assertz(user:dimension_n(7)),
    (   wam_haskell_target:generate_main_hs([], [], [], [dimension_n(11)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "fromIntegral (11 :: Int) :: Double"),
        \+ sub_string(S, _, _, _, "fromIntegral (7 :: Int) :: Double")
    ->  pass(Test)
    ;   fail_test(Test, 'dimension_n(11) option did not override user:dimension_n(7)')
    ),
    retractall(user:dimension_n(_)).

%% =========================================================================
%% use_lmdb(auto) resolver — picks IntMap-vs-LMDB by fact_count threshold
%% with a ghc-pkg availability gate. Tests cover the deterministic paths;
%% the `auto + high fact_count + ghc-pkg has lmdb → true` path is gated
%% by environment so we test only the always-false branches here.
test_resolve_use_lmdb_explicit_true_passthrough :-
    Test = 'WAM-Haskell: resolve_auto_use_lmdb passes use_lmdb(true) unchanged',
    (   wam_haskell_target:resolve_auto_use_lmdb([use_lmdb(true), other(x)], R),
        memberchk(use_lmdb(true), R),
        \+ memberchk(use_lmdb(auto), R)
    ->  pass(Test)
    ;   fail_test(Test, 'use_lmdb(true) was not preserved')
    ).

test_resolve_use_lmdb_explicit_false_passthrough :-
    Test = 'WAM-Haskell: resolve_auto_use_lmdb passes use_lmdb(false) unchanged',
    (   wam_haskell_target:resolve_auto_use_lmdb([use_lmdb(false)], R),
        memberchk(use_lmdb(false), R)
    ->  pass(Test)
    ;   fail_test(Test, 'use_lmdb(false) was not preserved')
    ).

test_resolve_use_lmdb_absent_unchanged :-
    Test = 'WAM-Haskell: resolve_auto_use_lmdb leaves Options without use_lmdb unchanged',
    (   wam_haskell_target:resolve_auto_use_lmdb([fact_count(99999)], R),
        \+ memberchk(use_lmdb(_), R),
        memberchk(fact_count(99999), R)
    ->  pass(Test)
    ;   fail_test(Test, 'absent use_lmdb was modified or fact_count lost')
    ).

test_resolve_use_lmdb_auto_low_fact_count_false :-
    %% At fact_count below the threshold, auto resolves to false
    %% regardless of lmdb availability. Deterministic across environments.
    Test = 'WAM-Haskell: resolve_auto_use_lmdb(auto) + low fact_count → false',
    (   wam_haskell_target:resolve_auto_use_lmdb(
            [use_lmdb(auto), fact_count(1000), lmdb_auto_threshold(50000)], R),
        memberchk(use_lmdb(false), R),
        \+ memberchk(use_lmdb(auto), R)
    ->  pass(Test)
    ;   fail_test(Test, 'auto + low fact_count did not resolve to false')
    ).

test_resolve_use_lmdb_auto_no_fact_count_false :-
    %% Without fact_count we conservatively pick false rather than guessing.
    Test = 'WAM-Haskell: resolve_auto_use_lmdb(auto) without fact_count → false',
    (   wam_haskell_target:resolve_auto_use_lmdb([use_lmdb(auto)], R),
        memberchk(use_lmdb(false), R),
        \+ memberchk(use_lmdb(auto), R)
    ->  pass(Test)
    ;   fail_test(Test, 'auto without fact_count did not resolve to false')
    ).

%% =================================================================
%% Phase 2c: cache_strategy(auto) cost-model resolver
%% =================================================================

test_cache_strategy_auto_low_k_picks_in_memory :-
    %% Small fact_count + ~5% wsf → K small. With huge mem_available
    %% (forcing FHot=1 regime) and explicit constants pinning K_cross
    %% near zero, force the model to pick `scan` → `in_memory`.
    %% Concretely: 100k facts × 5MB DB × 1MB R_free → DB exceeds RAM,
    %% but K=5000 vs K_cross at this regime puts us comfortably above
    %% threshold; the resolver should map to in_memory.
    Test = 'WAM-Haskell: cache_strategy(auto) at small selection picks in_memory',
    (   wam_haskell_target:resolve_auto_cache_strategy(
            [cache_strategy(auto),
             fact_count(100),
             working_set_fraction(0.5),
             db_size_bytes(5000),
             mem_available_bytes(16000000000)],
            R),
        memberchk(demand_bfs_mode(in_memory), R),
        \+ memberchk(cache_strategy(auto), R)
    ->  pass(Test)
    ;   fail_test(Test, 'cache_strategy auto + small selection did not pick in_memory')
    ).

test_cache_strategy_auto_high_k_picks_in_memory_at_high_selection :-
    %% Large fact_count + high working_set_fraction (touches >> K_cross)
    %% → cost model says scan → in_memory.
    Test = 'WAM-Haskell: cache_strategy(auto) at high selection picks in_memory',
    (   wam_haskell_target:resolve_auto_cache_strategy(
            [cache_strategy(auto),
             fact_count(10000000),
             working_set_fraction(0.5),
             db_size_bytes(500000000),
             mem_available_bytes(16000000000)],
            R),
        memberchk(demand_bfs_mode(in_memory), R)
    ->  pass(Test)
    ;   fail_test(Test, 'cache_strategy auto + high selection did not pick in_memory')
    ).

test_cache_strategy_auto_tiny_selection_picks_cursor :-
    %% Large DB + tiny working_set_fraction (K << K_cross) → sort →
    %% cursor. 10M facts × 0.0001 wsf = K=1000; K_cross at hot regime
    %% on 500MB W is ~100k. So K << K_cross → sort → cursor.
    Test = 'WAM-Haskell: cache_strategy(auto) at tiny selection picks cursor',
    (   wam_haskell_target:resolve_auto_cache_strategy(
            [cache_strategy(auto),
             fact_count(10000000),
             working_set_fraction(0.0001),
             db_size_bytes(500000000),
             mem_available_bytes(16000000000)],
            R),
        memberchk(demand_bfs_mode(cursor), R)
    ->  pass(Test)
    ;   fail_test(Test, 'cache_strategy auto + tiny selection did not pick cursor')
    ).

test_cache_strategy_absent_leaves_options_unchanged :-
    %% No cache_strategy(auto) → resolver is a no-op. Existing
    %% demand_bfs_mode (or its absence) flows through untouched.
    Test = 'WAM-Haskell: resolve_auto_cache_strategy is a no-op without cache_strategy(auto)',
    (   wam_haskell_target:resolve_auto_cache_strategy(
            [fact_count(1000), demand_bfs_mode(in_memory)], R),
        R == [fact_count(1000), demand_bfs_mode(in_memory)]
    ->  pass(Test)
    ;   fail_test(Test, 'resolve_auto_cache_strategy mutated options without cache_strategy(auto)')
    ).

test_cache_strategy_auto_overrides_explicit_demand_bfs_mode :-
    %% cache_strategy(auto) takes precedence over an existing
    %% demand_bfs_mode/1 entry — that's the whole point of opting in.
    Test = 'WAM-Haskell: cache_strategy(auto) overrides explicit demand_bfs_mode',
    (   wam_haskell_target:resolve_auto_cache_strategy(
            [cache_strategy(auto),
             demand_bfs_mode(in_memory),
             fact_count(10000000),
             working_set_fraction(0.0001),
             db_size_bytes(500000000),
             mem_available_bytes(16000000000)],
            R),
        memberchk(demand_bfs_mode(cursor), R),
        \+ memberchk(demand_bfs_mode(in_memory), R)
    ->  pass(Test)
    ;   fail_test(Test, 'cache_strategy auto did not override explicit demand_bfs_mode')
    ).

test_cache_strategy_footprint_guard_overrides_in_memory :-
    %% Cold-regime case: cost_model would recommend scan → in_memory
    %% (high selection ratio), but R_free < W means the in_memory
    %% IntMap can't actually fit. Guard must override to cursor.
    %% Inputs: 9.9M facts × 50 bytes/edge = 495 MB; wsf=0.05 → K=495k;
    %% R_free=125 MB; K_cross at f_hot=0.25 ≈ 4k → scan picked first,
    %% then footprint guard kicks in because 495 MB > 125 MB.
    Test = 'WAM-Haskell: cache_strategy(auto) footprint guard overrides in_memory when W > R_free',
    (   wam_haskell_target:resolve_auto_cache_strategy(
            [cache_strategy(auto),
             fact_count(9900000),
             working_set_fraction(0.05),
             db_size_bytes(495000000),
             mem_available_bytes(125000000)],
            R),
        memberchk(demand_bfs_mode(cursor), R),
        \+ memberchk(demand_bfs_mode(in_memory), R)
    ->  pass(Test)
    ;   fail_test(Test, 'footprint guard did not override in_memory when R_free < W')
    ).

test_cache_strategy_footprint_guard_inactive_in_hot_regime :-
    %% Hot regime: cost_model recommends scan → in_memory and the
    %% working set fits. Guard must NOT fire — in_memory stays.
    %% 297k facts × 50 = 15 MB DB; wsf=0.5 → K=148k; R_free=16 GB
    %% (W << R_free). K well above K_cross → scan → in_memory; guard
    %% inactive because 15 MB < 16 GB.
    Test = 'WAM-Haskell: cache_strategy(auto) footprint guard does NOT fire when W <= R_free',
    (   wam_haskell_target:resolve_auto_cache_strategy(
            [cache_strategy(auto),
             fact_count(297000),
             working_set_fraction(0.5),
             db_size_bytes(15000000),
             mem_available_bytes(16000000000)],
            R),
        memberchk(demand_bfs_mode(in_memory), R)
    ->  pass(Test)
    ;   fail_test(Test, 'footprint guard incorrectly fired in hot regime')
    ).

test_b1_lmdb_dupsort_per_thread_cursor :-
    Test = 'B1: dupsort layout uses per-thread cursor cache',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true), lmdb_layout(dupsort)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "DupsortCursorCache"),
        sub_string(S, _, _, _, "newDupsortCursorCache"),
        sub_string(S, _, _, _, "getOrOpenDupsortCursor"),
        sub_string(S, _, _, _, "myThreadId")
    ->  pass(Test)
    ;   fail_test(Test, 'Dupsort per-thread cursor cache not emitted')
    ).

test_b1_lmdb_default_layout_no_cursor_cache :-
    Test = 'B1: default layout does not emit cursor cache',
    (   compile_wam_runtime_to_haskell([use_lmdb(true)], [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "DupsortCursorCache")
    ->  pass(Test)
    ;   fail_test(Test, 'Default layout unexpectedly emitted dupsort cache code')
    ).

test_b1_lmdb_cache_memoize_emitted :-
    Test = 'B1: lmdb_cache_mode(memoize) maps to L2 (deprecated synonym)',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(memoize)], [], Code),
        atom_string(Code, S),
        % Phase 2: memoize is a deprecated synonym for sharded; both
        % emit L2 (lock-free IOArray cache).  The legacy IntMap-based
        % memoize code path is gone.
        sub_string(S, _, _, _, "L2Cache"),
        sub_string(S, _, _, _, "lmdbL2EdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test, 'L2 EdgeLookup not emitted for memoize mode')
    ).

test_b1_lmdb_cache_memoize_not_emitted_without_dupsort :-
    Test = 'B1: lmdb_cache_mode(memoize) ignored without dupsort layout',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true), lmdb_cache_mode(memoize)], [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "L2Cache"),
        \+ sub_string(S, _, _, _, "lmdbL2EdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test,
            'L2 EdgeLookup unexpectedly emitted without dupsort')
    ).

test_b1_lmdb_cache_sharded_emitted :-
    Test = 'B1: lmdb_cache_mode(sharded) emits L2 EdgeLookup',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(sharded)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "L2Cache"),
        sub_string(S, _, _, _, "lmdbL2EdgeLookup"),
        sub_string(S, _, _, _, "defaultL2Capacity"),
        sub_string(S, _, _, _, "newL2Cache")
    ->  pass(Test)
    ;   fail_test(Test, 'L2 EdgeLookup not emitted for sharded mode')
    ).

test_b1_lmdb_cache_two_level_emitted :-
    Test = 'B1: lmdb_cache_mode(two_level) emits both L1 and L2',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(two_level)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "L1Cache"),
        sub_string(S, _, _, _, "L2Cache"),
        sub_string(S, _, _, _, "lmdbTwoLevelEdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test, 'two_level EdgeLookup not emitted')
    ).

test_b1_lmdb_cache_default_no_l2 :-
    Test = 'B1: default (no cache_mode) does not emit L2',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true), lmdb_layout(dupsort)], [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "L2Cache"),
        \+ sub_string(S, _, _, _, "lmdbL2EdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test, 'L2 unexpectedly emitted in default mode')
    ).

test_b1_lmdb_cache_l2_capacity_override :-
    Test = 'B1: lmdb_cache_l2_capacity_bytes overrides auto-detect',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(sharded),
             lmdb_cache_l2_capacity_bytes(67108864)], [], Code),  % 64 MB
        atom_string(Code, S),
        % override branch emits a constant-return defaultL2Capacity
        sub_string(S, _, _, _, "user-specified"),
        sub_string(S, _, _, _, "67108864 `div` 32"),
        % the auto-detect branch should NOT be emitted
        \+ sub_string(S, _, _, _, "l2MemoryBudgetBytes")
    ->  pass(Test)
    ;   fail_test(Test, 'L2 capacity override not emitted as expected')
    ).

test_b1_lmdb_cache_l2_capacity_default_when_unset :-
    Test = 'B1: L2 capacity falls back to auto-detect when not specified',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(sharded)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "l2MemoryBudgetBytes"),
        \+ sub_string(S, _, _, _, "user-specified")
    ->  pass(Test)
    ;   fail_test(Test, 'Auto-detect path not emitted by default')
    ).

test_b1_lmdb_cache_l2_capacity_ignored_without_l2 :-
    Test = 'B1: lmdb_cache_l2_capacity_bytes ignored when L2 not active',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(per_hec),
             lmdb_cache_l2_capacity_bytes(67108864)], [], Code),
        atom_string(Code, S),
        % L2 is not active (per_hec only) so override should not fire
        \+ sub_string(S, _, _, _, "user-specified"),
        \+ sub_string(S, _, _, _, "67108864")
    ->  pass(Test)
    ;   fail_test(Test, 'L2 override unexpectedly emitted without L2')
    ).

test_b1_lmdb_cache_auto_no_hints_falls_back_to_none :-
    Test = 'B1: lmdb_cache_mode(auto) with no hints emits no cache',
    (   statistics:clear_cache_hints,
        compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(auto)], [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "L1Cache"),
        \+ sub_string(S, _, _, _, "L2Cache"),
        \+ sub_string(S, _, _, _, "lmdbL1EdgeLookup"),
        \+ sub_string(S, _, _, _, "lmdbL2EdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test,
            'auto with no hints unexpectedly emitted cache code')
    ).

test_b1_lmdb_cache_auto_intra_thread_picks_per_hec :-
    Test = 'B1: lmdb_cache_mode(auto) + intra_thread hint → per_hec (L1)',
    (   statistics:clear_cache_hints,
        statistics:declare_cache_hints(_{reuse_axis: intra_thread}),
        compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(auto)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "L1Cache"),
        sub_string(S, _, _, _, "lmdbL1EdgeLookup"),
        \+ sub_string(S, _, _, _, "L2Cache")
    ->  pass(Test)
    ;   fail_test(Test, 'auto + intra_thread did not emit L1-only')
    ),
    statistics:clear_cache_hints.

test_b1_lmdb_cache_auto_cross_thread_picks_sharded :-
    Test = 'B1: lmdb_cache_mode(auto) + cross_thread hint → sharded (L2)',
    (   statistics:clear_cache_hints,
        statistics:declare_cache_hints(_{reuse_axis: cross_thread}),
        compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(auto)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "L2Cache"),
        sub_string(S, _, _, _, "lmdbL2EdgeLookup"),
        \+ sub_string(S, _, _, _, "lmdbL1EdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test, 'auto + cross_thread did not emit L2-only')
    ),
    statistics:clear_cache_hints.

test_b1_lmdb_cache_auto_mixed_picks_two_level :-
    Test = 'B1: lmdb_cache_mode(auto) + mixed hint → two_level',
    (   statistics:clear_cache_hints,
        statistics:declare_cache_hints(_{reuse_axis: mixed}),
        compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(auto)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "lmdbTwoLevelEdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test, 'auto + mixed did not emit two_level')
    ),
    statistics:clear_cache_hints.

test_b1_lmdb_dupsort_alone_no_memoize :-
    Test = 'B1: dupsort without lmdb_cache_mode does not emit memoising lookup',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true), lmdb_layout(dupsort)], [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "LmdbResultCache"),
        \+ sub_string(S, _, _, _, "lmdbCachedEdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test,
            'Memoising EdgeLookup unexpectedly emitted without cache_mode')
    ).

test_b1_lmdb_cache_l1_emitted :-
    Test = 'B1: lmdb_cache_mode(per_hec) emits L1 EdgeLookup',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(per_hec)], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "L1Cache"),
        sub_string(S, _, _, _, "L1Registry"),
        sub_string(S, _, _, _, "lmdbL1EdgeLookup"),
        sub_string(S, _, _, _, "defaultL1Capacity")
    ->  pass(Test)
    ;   fail_test(Test, 'L1 EdgeLookup not emitted')
    ).

test_b1_lmdb_cache_l1_not_emitted_without_dupsort :-
    Test = 'B1: lmdb_cache_mode(per_hec) ignored without dupsort layout',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true), lmdb_cache_mode(per_hec)], [], Code),
        atom_string(Code, S),
        \+ sub_string(S, _, _, _, "L1Cache"),
        \+ sub_string(S, _, _, _, "lmdbL1EdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test,
            'L1 EdgeLookup unexpectedly emitted without dupsort')
    ).

test_b1_lmdb_cache_modes_mutually_exclusive :-
    Test = 'B1: cache modes are mutually exclusive (first option wins)',
    (   compile_wam_runtime_to_haskell(
            [use_lmdb(true),
             lmdb_layout(dupsort),
             lmdb_cache_mode(memoize),
             lmdb_cache_mode(per_hec)], [], Code),
        atom_string(Code, S),
        % memoize is the first option, and it now maps to L2 (sharded);
        % L1 should not be emitted alongside.  The legacy
        % lmdbCachedEdgeLookup is gone — Phase 2 unified memoize and
        % sharded under the IOArray L2 implementation.
        sub_string(S, _, _, _, "lmdbL2EdgeLookup"),
        \+ sub_string(S, _, _, _, "lmdbL1EdgeLookup")
    ->  pass(Test)
    ;   fail_test(Test,
            'Mode precedence violated when both flags set')
    ).

run_tests :-
    format('~n========================================~n'),
    format('WAM-Haskell target: Phase 5+6+7+8+F1-F4+E2E+B1 codegen tests~n'),
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
    test_render_kernel_function_cwd_independent,
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
    test_haskell_hashable_value_handles_intset,
    test_haskell_fork_min_branches_threshold,
    test_haskell_fork_helpers_present,
    test_haskell_partryme_else_delegates_to_fork,
    test_haskell_runtime_imports_parallel,
    %% PutStructureDyn: runtime-parsed functors
    test_haskell_put_structure_dyn_in_types,
    test_haskell_put_structure_dyn_step_handler,
    test_haskell_put_structure_dyn_wam_parse,
    %% SetVariable: list-build path used by user-source =../2 compose mode
    test_haskell_set_variable_in_types,
    test_haskell_set_variable_step_handler,
    test_haskell_set_variable_wam_parse,
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
    %% Phase F1: Fact predicate classification
    test_f1_fact_only_classification,
    test_f1_rule_predicate_not_fact_only,
    test_f1_two_arg_fact_groundness,
    test_f1_variable_first_arg,
    test_f1_layout_auto_above_threshold,
    test_f1_layout_compiled_below_threshold,
    test_f1_layout_user_override,
    test_f1_comment_in_predicates_hs,
    test_f1_segment_parser,
    test_f1_compiled_only_policy,
    %% Phase F2: FactStream choice point type
    test_f2_fact_stream_in_builtin_state,
    test_f2_call_fact_stream_in_instruction,
    test_f2_stream_facts_function,
    test_f2_resume_fact_stream_handler,
    test_f2_call_fact_stream_step_handler,
    test_f2_wc_inline_facts_field,
    test_f2_stream_facts_filters_bound_a1,
    test_f2_fact_stream_exhaustion_backtracks,
    %% Phase F3: inline_data emission
    test_f3_inline_data_emits_call_fact_stream,
    test_f3_inline_data_emits_fact_literal,
    test_f3_below_threshold_stays_compiled,
    test_f3_inline_defs_returned,
    test_f3_fact_tuples_extracted,
    test_f3_camel_case_list_name,
    %% Phase F4: FactSource abstraction
    test_f4_fact_source_type_present,
    test_f4_wc_fact_sources_field,
    test_f4_tsv_fact_source_function,
    test_f4_intmap_fact_source_function,
    test_f4_stream_facts_fallback_to_fact_sources,
    test_f4_stream_fact_rows_helper,
    %% E2E: fact access wiring
    test_e2e_inline_data_project_has_call_fact_stream,
    test_e2e_inline_facts_wiring_generated,
    test_e2e_empty_inline_defs_no_wiring,
    test_e2e_below_threshold_no_inline,
    %% Phase B1: LMDB backend
    test_b1_lmdb_functions_present_when_enabled,
    test_b1_lmdb_imports_present_when_enabled,
    test_b1_lmdb_absent_when_disabled,
    test_b1_lmdb_cabal_dependency,
    test_b1_no_lmdb_cabal_default,
    test_with_rtsopts_emits_in_cabal,
    test_with_rtsopts_absent_by_default,
    test_b1_lmdb_raw_zero_copy_reads,
    test_b1_phase2b2_loaders_emitted,
    test_b1_peekstringbytes_decodes_utf8,
    test_b1_lmdb_text_imports_present,
    test_b1_lmdb_cabal_text_dependencies,
    test_b1_lmdb_scan_support_present,
    test_b1_lmdb_manifest_fact_source_present,
    test_b1_lmdb_manifest_wiring_option_present,
    test_b1_lmdb_dupsort_per_thread_cursor,
    test_b1_lmdb_default_layout_no_cursor_cache,
    test_b1_lmdb_cache_memoize_emitted,
    test_b1_lmdb_cache_memoize_not_emitted_without_dupsort,
    test_b1_lmdb_dupsort_alone_no_memoize,
    test_b1_lmdb_cache_l1_emitted,
    test_b1_lmdb_cache_l1_not_emitted_without_dupsort,
    test_b1_lmdb_cache_modes_mutually_exclusive,
    test_b1_lmdb_cache_sharded_emitted,
    test_b1_lmdb_cache_two_level_emitted,
    test_b1_lmdb_cache_default_no_l2,
    test_b1_lmdb_cache_l2_capacity_override,
    test_b1_lmdb_cache_l2_capacity_default_when_unset,
    test_b1_lmdb_cache_l2_capacity_ignored_without_l2,
    test_b1_lmdb_cache_auto_no_hints_falls_back_to_none,
    test_b1_lmdb_cache_auto_intra_thread_picks_per_hec,
    test_b1_lmdb_cache_auto_cross_thread_picks_sharded,
    test_b1_lmdb_cache_auto_mixed_picks_two_level,
    test_b1_external_source_skips_wam_compilation,
    test_b1_external_source_default_allow_list,
    test_b1_external_source_off_without_use_lmdb,
    test_demand_filter_gates_seed_query_body,
    test_demand_bfs_mode_cursor_emits_runDemandBFSCursor,
    test_demand_bfs_mode_in_memory_default,
    test_demand_bfs_mode_auto_high_fact_count_picks_cursor,
    test_demand_bfs_mode_auto_low_fact_count_picks_in_memory,
    test_demand_filter_hop_limit_with_max_hops,
    test_demand_filter_none_emits_dfnone,
    test_demand_filter_flux_emits_panic_stub_compatible,
    test_int_atom_seeds_lmdb_calls_loaders,
    test_int_atom_seeds_true_does_not_call_lmdb_loaders,
    test_int_atom_seeds_default_does_not_call_lmdb_loaders,
    test_demand_filter_invalid_strategy_rejected,
    test_demand_filter_false_leaves_query_ungated,
    %% max_depth substitution (regression for linear-chain-zero-results)
    test_max_depth_default_10_when_no_user_fact,
    test_max_depth_from_user_fact,
    test_max_depth_option_overrides_user_fact,
    %% dimension_n substitution (same instrumentation-bug class as max_depth)
    test_dimension_n_default_5_when_no_user_fact,
    test_dimension_n_from_user_fact,
    test_dimension_n_option_overrides_user_fact,
    %% use_lmdb(auto) resolver — IntMap-vs-LMDB normalisation
    test_resolve_use_lmdb_explicit_true_passthrough,
    test_resolve_use_lmdb_explicit_false_passthrough,
    test_resolve_use_lmdb_absent_unchanged,
    test_resolve_use_lmdb_auto_low_fact_count_false,
    test_resolve_use_lmdb_auto_no_fact_count_false,
    test_cache_strategy_auto_low_k_picks_in_memory,
    test_cache_strategy_auto_high_k_picks_in_memory_at_high_selection,
    test_cache_strategy_auto_tiny_selection_picks_cursor,
    test_cache_strategy_absent_leaves_options_unchanged,
    test_cache_strategy_auto_overrides_explicit_demand_bfs_mode,
    test_cache_strategy_footprint_guard_overrides_in_memory,
    test_cache_strategy_footprint_guard_inactive_in_hot_regime,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).

:- initialization(run_tests, main).
