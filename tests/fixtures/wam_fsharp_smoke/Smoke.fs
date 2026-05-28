module Program

// Runtime smoke for the F# WAM target.
//
// Replaces the default Program.fs (the category-ancestor benchmark driver)
// when the test harness wants to actually execute the runtime rather than
// just verify it builds.  Imports the *real* generated WamTypes +
// WamRuntime — same build flags, same module structure — and drives a
// handful of representative scenarios that exercise:
//   - Basic step instructions (PutConstant, Proceed, GetConstant)
//   - Backtracking via try_me_else / trust_me
//   - Standard-order term comparison (compareValue)
//   - Variable dereferencing through binding chains (derefVar)
//   - Parallel WAM helpers (enumerateParBranches)
//
// The smoke does NOT depend on Predicates.fs or Lowered.fs content —
// it constructs Instruction arrays inline.  That keeps the test
// independent of which predicates the project happened to be generated
// with.  Tests/fixtures uses the same approach as the Haskell GHC
// smoke (tests/fixtures/wam_put_structure_dyn_smoke/Smoke.hs).
//
// Output contract:
//   - Each scenario prints `[PASS] <name>` or `[FAIL] <name>`.
//   - A final `RESULT <passes>/<total>` line is parsed by the
//     Prolog test driver to determine overall outcome.
//   - Exit code is 0 iff every scenario passes.

open WamTypes
open WamRuntime

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn "[PASS] %s" name
    else
        fails <- fails + 1
        printfn "[FAIL] %s" name

let mkEmptyState () =
    { WsPC         = 0
      WsRegs       = Array.create MaxRegs (Unbound -1)
      WsStack      = []
      WsHeap       = []
      WsHeapLen    = 0
      WsTrail      = []
      WsTrailLen   = 0
      WsCP         = 0
      WsCPs        = []
      WsCPsLen     = 0
      WsBindings   = Map.empty
      WsCutBar     = 0
      WsVarCounter = 0
      WsBuilder    = None
      WsBuilderStack = []
      WsAggAccum   = []
      WsB0Stack    = []
      WsCatchers   = [] }

let mkContext (code: Instruction array) (labels: Map<string, int>) =
    { WcCode              = code
      WcLabels            = labels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = Map.empty
      WcAtomDeintern      = Map.empty
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcCancellationToken = None }

// -- Scenario 1: PutConstant writes to register and advances PC ------------

let scenario_put_constant () =
    let code = [| PutConstant (Atom "foo", 1) |]
    let ctx = mkContext code Map.empty
    let s0 = mkEmptyState ()
    match step ctx s0 code.[0] with
    | Some s ->
        assertTrue "PutConstant: register 1 = Atom \"foo\""
                   (s.WsRegs.[1] = Atom "foo")
        assertTrue "PutConstant: PC advanced by 1"
                   (s.WsPC = 1)
    | None ->
        assertTrue "PutConstant should succeed" false

// -- Scenario 2: GetConstant unifies with bound register --------------------

let scenario_get_constant_match () =
    let code = [| GetConstant (Atom "foo", 1) |]
    let ctx = mkContext code Map.empty
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom "foo"
    let s0 = { mkEmptyState () with WsRegs = regs }
    match step ctx s0 code.[0] with
    | Some s ->
        assertTrue "GetConstant (matched): PC advanced"
                   (s.WsPC = 1)
    | None ->
        assertTrue "GetConstant (matched) should succeed" false

let scenario_get_constant_mismatch () =
    let code = [| GetConstant (Atom "foo", 1) |]
    let ctx = mkContext code Map.empty
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom "bar"
    let s0 = { mkEmptyState () with WsRegs = regs }
    match step ctx s0 code.[0] with
    | Some _ ->
        assertTrue "GetConstant (mismatch) should fail" false
    | None ->
        assertTrue "GetConstant (mismatch): returns None" true

// -- Scenario 3: Proceed returns to CP --------------------------------------

let scenario_proceed_returns_to_cp () =
    let code = [| Proceed |]
    let ctx = mkContext code Map.empty
    let s0 = { mkEmptyState () with WsPC = 0; WsCP = 42 }
    match step ctx s0 code.[0] with
    | Some s ->
        assertTrue "Proceed: PC <- CP" (s.WsPC = 42)
        assertTrue "Proceed: CP cleared" (s.WsCP = 0)
    | None ->
        assertTrue "Proceed should succeed" false

// -- Scenario 4: compareValue gives Prolog standard order ------------------

let scenario_compare_value () =
    assertTrue "compareValue: Unbound < Integer"
               (compareValue (Unbound 0) (Integer 1) < 0)
    assertTrue "compareValue: Integer < Atom"
               (compareValue (Integer 5) (Atom "a") < 0)
    assertTrue "compareValue: Atom < Str (compound)"
               (compareValue (Atom "z") (Str ("foo", [])) < 0)
    assertTrue "compareValue: structural equality on Atoms"
               (compareValue (Atom "x") (Atom "x") = 0)
    assertTrue "compareValue: numeric mixing (Integer 1 < Float 2.0)"
               (compareValue (Integer 1) (Float 2.0) < 0)
    assertTrue "compareValue: Atom alpha order"
               (compareValue (Atom "a") (Atom "b") < 0)

// -- Scenario 5: derefVar follows the binding chain ------------------------

let scenario_deref_var () =
    let bs = Map.ofList [ (1, Unbound 2); (2, Atom "x") ]
    assertTrue "derefVar: 1 -> 2 -> Atom \"x\""
               (derefVar bs (Unbound 1) = Atom "x")
    assertTrue "derefVar: non-Unbound passthrough"
               (derefVar bs (Atom "y") = Atom "y")
    assertTrue "derefVar: unbound stays Unbound"
               (derefVar Map.empty (Unbound 99) = Unbound 99)

// -- Scenario 6: enumerateParBranches walks Par* chains --------------------

let scenario_enumerate_par_branches () =
    // PC 0: ParTryMeElsePc 3   (branch 1 entry at PC 1)
    // PC 1: Proceed             (branch 1 body)
    // PC 2: (unused)
    // PC 3: ParTrustMe          (last branch entry at PC 4)
    // PC 4: Proceed             (branch 2 body)
    let code = [|
        ParTryMeElsePc 3      // PC 0
        Proceed                // PC 1
        Proceed                // PC 2
        ParTrustMe             // PC 3
        Proceed                // PC 4
    |]
    let ctx = mkContext code Map.empty
    let branches = enumerateParBranches ctx 0 3
    assertTrue "enumerateParBranches: yields both branch entry PCs"
               (branches = [1; 4])

    // 3-branch chain via ParRetryMeElsePc in the middle.
    let code3 = [|
        ParTryMeElsePc 3      // PC 0
        Proceed                // PC 1 (branch 1)
        Proceed                // PC 2
        ParRetryMeElsePc 5    // PC 3
        Proceed                // PC 4 (branch 2)
        ParTrustMe             // PC 5
        Proceed                // PC 6 (branch 3)
    |]
    let ctx3 = mkContext code3 Map.empty
    let branches3 = enumerateParBranches ctx3 0 3
    assertTrue "enumerateParBranches: 3-branch chain via ParRetryMeElsePc"
               (branches3 = [1; 4; 6])

// -- Scenario 7: Try/Trust backtracking via run ----------------------------

let scenario_backtrack_run () =
    // Two-clause predicate emulating:
    //   p(a). p(b).
    // Layout:
    //   PC 0: TryMeElsePc 3       — push CP for clause 2
    //   PC 1: GetConstant (Atom "a", 1)
    //   PC 2: Proceed
    //   PC 3: TrustMe             — pop CP (last clause)
    //   PC 4: GetConstant (Atom "b", 1)
    //   PC 5: Proceed
    let code = [|
        TryMeElsePc 3
        GetConstant (Atom "a", 1)
        Proceed
        TrustMe
        GetConstant (Atom "b", 1)
        Proceed
    |]
    let ctx = mkContext code Map.empty
    // Query: p(b) — should succeed on clause 2 via backtrack.
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom "b"
    let s0 = { mkEmptyState () with WsPC = 1; WsCP = 0; WsRegs = regs }
    // Drive the first try: GetConstant a fails -> backtrack to clause 2.
    // Use run to exercise the full backtracking machinery, NOT just one
    // step.  But run terminates on WsPC = 0, so we'd need a sentinel.
    // Instead, just check that backtrack works on a single failure:
    match step ctx s0 (TryMeElsePc 3) with
    | Some s1 ->
        // First clause head: GetConstant a vs Atom "b" -> mismatch -> None.
        let s2 = { s1 with WsPC = 1 }
        match step ctx s2 code.[1] with
        | None ->
            // Now backtrack.
            match backtrack s2 with
            | Some s3 ->
                assertTrue "Backtrack: returns to clause 2 (PC = 3)"
                           (s3.WsPC = 3)
                // Restored regs from CP snapshot
                assertTrue "Backtrack: regs restored"
                           (s3.WsRegs.[1] = Atom "b")
            | None ->
                assertTrue "Backtrack should succeed" false
        | Some _ ->
            assertTrue "GetConstant (mismatch) should fail" false
    | None ->
        assertTrue "TryMeElsePc should succeed" false

// -- Scenario 8: unifyVal binds and trails ---------------------------------

let scenario_unify_val () =
    let s0 = mkEmptyState ()
    // unifyVal (Unbound 1) (Atom "x") s0 should bind 1 -> Atom "x"
    match unifyVal (Unbound 1) (Atom "x") s0 with
    | Some s1 ->
        assertTrue "unifyVal: bindings[1] = Atom \"x\""
                   (Map.tryFind 1 s1.WsBindings = Some (Atom "x"))
        assertTrue "unifyVal: trail entry created"
                   (s1.WsTrailLen = 1)
    | None ->
        assertTrue "unifyVal (Unbound, Atom) should succeed" false

    // unifyVal (Atom "x") (Atom "y") s0 should fail
    match unifyVal (Atom "x") (Atom "y") s0 with
    | Some _ ->
        assertTrue "unifyVal (Atom x, Atom y) should fail" false
    | None ->
        assertTrue "unifyVal (Atom x, Atom y): returns None" true

// -- Scenario 9: PutStructureDyn (Phase-I specialized instruction) ---------

let scenario_put_structure_dyn () =
    // PutStructureDyn nameReg=1 arityReg=2 targetReg=3
    // A1 = Atom "foo", A2 = Integer 2  =>  WsBuilder = BuildStruct ("foo", 3, 2, [])
    let code = [| PutStructureDyn (1, 2, 3) |]
    let ctx = mkContext code Map.empty
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom "foo"
    regs.[2] <- Integer 2
    let s0 = { mkEmptyState () with WsRegs = regs }
    match step ctx s0 code.[0] with
    | Some s ->
        assertTrue "PutStructureDyn: PC advanced" (s.WsPC = 1)
        match s.WsBuilder with
        | Some (BuildStruct (fn, reg, arity, args)) ->
            assertTrue "PutStructureDyn: builder functor = \"foo\"" (fn = "foo")
            assertTrue "PutStructureDyn: builder target reg = 3" (reg = 3)
            assertTrue "PutStructureDyn: builder arity = 2" (arity = 2)
            assertTrue "PutStructureDyn: builder args = []" (args = [])
        | _ ->
            assertTrue "PutStructureDyn: builder should be BuildStruct" false
    | None ->
        assertTrue "PutStructureDyn should succeed with Atom + non-neg Integer" false

    // Negative arity: should fail.
    let regs2 = Array.create MaxRegs (Unbound -1)
    regs2.[1] <- Atom "foo"
    regs2.[2] <- Integer -1
    let s0b = { mkEmptyState () with WsRegs = regs2 }
    match step ctx s0b code.[0] with
    | Some _ ->
        assertTrue "PutStructureDyn: negative arity should fail" false
    | None ->
        assertTrue "PutStructureDyn: negative arity returns None" true

    // Non-Atom name: should fail.
    let regs3 = Array.create MaxRegs (Unbound -1)
    regs3.[1] <- Integer 42
    regs3.[2] <- Integer 0
    let s0c = { mkEmptyState () with WsRegs = regs3 }
    match step ctx s0c code.[0] with
    | Some _ ->
        assertTrue "PutStructureDyn: non-Atom name should fail" false
    | None ->
        assertTrue "PutStructureDyn: non-Atom name returns None" true

// -- Scenario 10: Arg (specialized arg/3) ----------------------------------

let scenario_arg_specialized () =
    // Arg 2 tReg=1 aReg=2 on a Str ("foo", [Atom "a"; Atom "b"; Atom "c"])
    // should extract Atom "b" into register 2.
    let code = [| Arg (2, 1, 2) |]
    let ctx = mkContext code Map.empty
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Str ("foo", [Atom "a"; Atom "b"; Atom "c"])
    let s0 = { mkEmptyState () with WsRegs = regs }
    match step ctx s0 code.[0] with
    | Some s ->
        assertTrue "Arg: PC advanced" (s.WsPC = 1)
        assertTrue "Arg: aReg = 2nd subterm (Atom \"b\")"
                   (s.WsRegs.[2] = Atom "b")
    | None ->
        assertTrue "Arg should succeed on Str + valid N" false

    // VList head-or-tail virtualization: N=1 head, N=2 tail
    let codeHead = [| Arg (1, 1, 2) |]
    let regsH = Array.create MaxRegs (Unbound -1)
    regsH.[1] <- VList [Atom "h"; Atom "t1"; Atom "t2"]
    let s0H = { mkEmptyState () with WsRegs = regsH }
    let ctxH = mkContext codeHead Map.empty
    match step ctxH s0H codeHead.[0] with
    | Some s ->
        assertTrue "Arg: VList N=1 head"
                   (s.WsRegs.[2] = Atom "h")
    | None ->
        assertTrue "Arg should succeed for VList N=1" false

    let codeTail = [| Arg (2, 1, 2) |]
    let regsT = Array.create MaxRegs (Unbound -1)
    regsT.[1] <- VList [Atom "h"; Atom "t1"; Atom "t2"]
    let s0T = { mkEmptyState () with WsRegs = regsT }
    let ctxT = mkContext codeTail Map.empty
    match step ctxT s0T codeTail.[0] with
    | Some s ->
        assertTrue "Arg: VList N=2 tail (VList [t1; t2])"
                   (s.WsRegs.[2] = VList [Atom "t1"; Atom "t2"])
    | None ->
        assertTrue "Arg should succeed for VList N=2" false

    // Out-of-range N: should fail.
    let codeOOR = [| Arg (5, 1, 2) |]
    let regsOOR = Array.create MaxRegs (Unbound -1)
    regsOOR.[1] <- Str ("foo", [Atom "a"; Atom "b"])
    let s0OOR = { mkEmptyState () with WsRegs = regsOOR }
    let ctxOOR = mkContext codeOOR Map.empty
    match step ctxOOR s0OOR codeOOR.[0] with
    | Some _ ->
        assertTrue "Arg: out-of-range N should fail" false
    | None ->
        assertTrue "Arg: out-of-range N returns None" true

// -- Scenario 11: NotMemberList (Phase-I) ----------------------------------

let scenario_not_member_list () =
    // NotMemberList xReg=1 lReg=2 with X = Atom "z", L = [a, b, c]
    // -> succeed (z not in list).
    let code = [| NotMemberList (1, 2) |]
    let ctx = mkContext code Map.empty
    let regsOK = Array.create MaxRegs (Unbound -1)
    regsOK.[1] <- Atom "z"
    regsOK.[2] <- VList [Atom "a"; Atom "b"; Atom "c"]
    let s0OK = { mkEmptyState () with WsRegs = regsOK }
    match step ctx s0OK code.[0] with
    | Some s ->
        assertTrue "NotMemberList: X not in list -> success"
                   (s.WsPC = 1)
    | None ->
        assertTrue "NotMemberList: X not in list should succeed" false

    // X = Atom "b" is in [a, b, c] -> fail.
    let regsBad = Array.create MaxRegs (Unbound -1)
    regsBad.[1] <- Atom "b"
    regsBad.[2] <- VList [Atom "a"; Atom "b"; Atom "c"]
    let s0Bad = { mkEmptyState () with WsRegs = regsBad }
    match step ctx s0Bad code.[0] with
    | Some _ ->
        assertTrue "NotMemberList: X in list should fail" false
    | None ->
        assertTrue "NotMemberList: X in list returns None" true

// -- Scenario 12: NotMemberConstAtoms (Phase-I) -----------------------------

let scenario_not_member_const_atoms () =
    // X = Atom "z" not in [a; b; c] -> succeed.
    let code = [| NotMemberConstAtoms (1, ["a"; "b"; "c"]) |]
    let ctx = mkContext code Map.empty
    let regs1 = Array.create MaxRegs (Unbound -1)
    regs1.[1] <- Atom "z"
    let s1 = { mkEmptyState () with WsRegs = regs1 }
    match step ctx s1 code.[0] with
    | Some _ ->
        assertTrue "NotMemberConstAtoms: Atom \"z\" notin [a;b;c]" true
    | None ->
        assertTrue "NotMemberConstAtoms (z): should succeed" false

    // X = Atom "b" is in the list -> fail.
    let regs2 = Array.create MaxRegs (Unbound -1)
    regs2.[1] <- Atom "b"
    let s2 = { mkEmptyState () with WsRegs = regs2 }
    match step ctx s2 code.[0] with
    | Some _ ->
        assertTrue "NotMemberConstAtoms: Atom \"b\" in list should fail" false
    | None ->
        assertTrue "NotMemberConstAtoms (b): returns None" true

    // Non-atom ground (Integer): cannot unify with atoms -> succeed.
    let regs3 = Array.create MaxRegs (Unbound -1)
    regs3.[1] <- Integer 42
    let s3 = { mkEmptyState () with WsRegs = regs3 }
    match step ctx s3 code.[0] with
    | Some _ ->
        assertTrue "NotMemberConstAtoms: Integer can't unify with atoms" true
    | None ->
        assertTrue "NotMemberConstAtoms (Integer): should succeed" false

    // Unbound: could-unify -> fail (matches Prolog \+ member(X, [a,b,c])
    // when X is unbound).  Use a non-sentinel register slot first.
    let regs4 = Array.create MaxRegs (Unbound -1)
    regs4.[1] <- Unbound 100  // not the -1 sentinel
    let s4 = { mkEmptyState () with WsRegs = regs4 }
    match step ctx s4 code.[0] with
    | Some _ ->
        assertTrue "NotMemberConstAtoms: Unbound should fail (could-unify)" false
    | None ->
        assertTrue "NotMemberConstAtoms (Unbound): returns None" true

// -- Scenario 13: VSet family (BuildEmptySet / SetInsert / NotMemberSet) ---

let scenario_vset_family () =
    // BuildEmptySet 1 -> WsRegs.[1] = VSet (empty Set<string>)
    let code1 = [| BuildEmptySet 1 |]
    let ctx1 = mkContext code1 Map.empty
    let s0 = mkEmptyState ()
    match step ctx1 s0 code1.[0] with
    | Some s ->
        assertTrue "BuildEmptySet: WsRegs[1] = VSet Set.empty"
                   (s.WsRegs.[1] = VSet Set.empty)
    | None ->
        assertTrue "BuildEmptySet should succeed" false

    // SetInsert elem=2, in=1, out=3 on (VSet {}) -> VSet {"x"}
    let code2 = [| SetInsert (2, 1, 3) |]
    let ctx2 = mkContext code2 Map.empty
    let regs2 = Array.create MaxRegs (Unbound -1)
    regs2.[1] <- VSet Set.empty
    regs2.[2] <- Atom "x"
    let s2 = { mkEmptyState () with WsRegs = regs2 }
    match step ctx2 s2 code2.[0] with
    | Some s ->
        assertTrue "SetInsert: outReg = VSet {\"x\"}"
                   (s.WsRegs.[3] = VSet (Set.ofList ["x"]))
    | None ->
        assertTrue "SetInsert should succeed on Atom + VSet" false

    // NotMemberSet: elem=1, set=2 with set = {"a"; "b"} and elem = "z"
    // -> succeed (z not in set).
    let code3 = [| NotMemberSet (1, 2) |]
    let ctx3 = mkContext code3 Map.empty
    let regs3 = Array.create MaxRegs (Unbound -1)
    regs3.[1] <- Atom "z"
    regs3.[2] <- VSet (Set.ofList ["a"; "b"])
    let s3 = { mkEmptyState () with WsRegs = regs3 }
    match step ctx3 s3 code3.[0] with
    | Some _ ->
        assertTrue "NotMemberSet: \"z\" notin {\"a\";\"b\"} -> success" true
    | None ->
        assertTrue "NotMemberSet (z): should succeed" false

    // elem = "a" IS in the set -> fail.
    let regs4 = Array.create MaxRegs (Unbound -1)
    regs4.[1] <- Atom "a"
    regs4.[2] <- VSet (Set.ofList ["a"; "b"])
    let s4 = { mkEmptyState () with WsRegs = regs4 }
    match step ctx3 s4 code3.[0] with
    | Some _ ->
        assertTrue "NotMemberSet: \"a\" in set should fail" false
    | None ->
        assertTrue "NotMemberSet (a): returns None" true

// -- Scenario 14: runNegationParallel end-to-end ---------------------------
//
// Builds a Par* chain inside a context with labels so \+/1 routes through
// runNegationParallel.  The "goal" is a 3-branch chain where every branch
// dead-ends via Fail; expected: negation succeeds (no branch goal succeeded).

let scenario_run_negation_parallel_all_fail () =
    // Labels: "g/0" -> PC 0 (the ParTryMeElsePc).
    // PC 0: ParTryMeElsePc 3
    // PC 1: Fail
    // PC 2: Proceed  (unreachable past Fail)
    // PC 3: ParRetryMeElsePc 5
    // PC 4: Fail
    // PC 5: ParTrustMe
    // PC 6: Fail
    let code = [|
        ParTryMeElsePc 3         // 0
        Fail                      // 1
        Proceed                   // 2
        ParRetryMeElsePc 5       // 3
        Fail                      // 4
        ParTrustMe                // 5
        Fail                      // 6
    |]
    let labels = Map.ofList [ ("g/0", 0) ]
    let ctx = mkContext code labels
    let s0 = mkEmptyState ()
    // runNegationParallel directly: all branches Fail -> false (no goal
    // succeeded -> \+ would succeed).
    let anySucceeded = runNegationParallel ctx s0 0 3
    assertTrue "runNegationParallel: all-Fail chain => false (negation succeeds)"
               (anySucceeded = false)

let scenario_run_negation_parallel_one_succeeds () =
    // Same shape but the second branch reaches Proceed.
    // PC 0: ParTryMeElsePc 3
    // PC 1: Fail
    // PC 2: (filler)
    // PC 3: ParRetryMeElsePc 5
    // PC 4: Proceed              // <-- this branch succeeds
    // PC 5: ParTrustMe
    // PC 6: Fail
    let code = [|
        ParTryMeElsePc 3
        Fail
        Proceed
        ParRetryMeElsePc 5
        Proceed
        ParTrustMe
        Fail
    |]
    let labels = Map.ofList [ ("g/0", 0) ]
    let ctx = mkContext code labels
    let s0 = mkEmptyState ()
    let anySucceeded = runNegationParallel ctx s0 0 3
    assertTrue "runNegationParallel: one-Proceed branch => true (negation fails)"
               (anySucceeded = true)

// -- Scenario 15: ParTryMeElsePc OUTSIDE an aggregate aliases to sequential
//
// forkOrSequential checks for a forkable aggregate frame in WsCPs; absent
// one, it falls back to the sequential TryMeElse step.  This scenario
// locks in that contract.

let scenario_par_step_aliases_to_sequential () =
    let code = [| ParTryMeElsePc 99 |]
    let ctx = mkContext code Map.empty
    let s0 = mkEmptyState ()
    match step ctx s0 code.[0] with
    | Some s ->
        assertTrue "ParTryMeElsePc (no agg frame): PC advanced"
                   (s.WsPC = 1)
        assertTrue "ParTryMeElsePc (no agg frame): one CP pushed"
                   (s.WsCPsLen = 1)
        match s.WsCPs with
        | cp :: _ ->
            assertTrue "ParTryMeElsePc (no agg frame): CP next PC = else target"
                       (cp.CpNextPC = 99)
        | [] ->
            assertTrue "ParTryMeElsePc: CP list should be non-empty" false
    | None ->
        assertTrue "ParTryMeElsePc should succeed" false

// -- Scenario 16: inferMergeStrategy + isForkableStrategy ------------------

let scenario_merge_strategy_helpers () =
    assertTrue "inferMergeStrategy: \"sum\" -> MergeSum"
               (inferMergeStrategy "sum" = MergeSum)
    assertTrue "inferMergeStrategy: \"count\" -> MergeCount"
               (inferMergeStrategy "count" = MergeCount)
    assertTrue "inferMergeStrategy: \"bag\" -> MergeBag"
               (inferMergeStrategy "bag" = MergeBag)
    assertTrue "inferMergeStrategy: \"set\" -> MergeSet"
               (inferMergeStrategy "set" = MergeSet)
    assertTrue "inferMergeStrategy: \"findall\" -> MergeFindall"
               (inferMergeStrategy "findall" = MergeFindall)
    assertTrue "inferMergeStrategy: \"collect\" -> MergeFindall (alias)"
               (inferMergeStrategy "collect" = MergeFindall)
    assertTrue "inferMergeStrategy: unknown -> MergeSequential"
               (inferMergeStrategy "unknown_strategy" = MergeSequential)
    assertTrue "isForkableStrategy: MergeSum"     (isForkableStrategy MergeSum)
    assertTrue "isForkableStrategy: MergeCount"   (isForkableStrategy MergeCount)
    assertTrue "isForkableStrategy: MergeBag"     (isForkableStrategy MergeBag)
    assertTrue "isForkableStrategy: MergeSet"     (isForkableStrategy MergeSet)
    assertTrue "isForkableStrategy: MergeFindall" (isForkableStrategy MergeFindall)
    assertTrue "isForkableStrategy: MergeSequential -> false"
               (not (isForkableStrategy MergeSequential))

// -- Scenario 17: combineParBranchResults per-strategy semantics ----------

let scenario_combine_par_branch_results () =
    // Sum: numeric sum, demotes to Integer when whole.
    let sumResult = combineParBranchResults MergeSum [Integer 1; Integer 2; Integer 3]
    assertTrue "combineParBranchResults sum [1; 2; 3] = Integer 6"
               (sumResult = Integer 6)
    // Count: sum of per-branch counts.
    let countResult = combineParBranchResults MergeCount [Integer 1; Integer 1; Integer 1]
    assertTrue "combineParBranchResults count [1; 1; 1] = Integer 3"
               (countResult = Integer 3)
    // Bag: concat of per-branch VLists.
    let bagResult =
        combineParBranchResults MergeBag
            [VList [Atom "a"; Atom "b"]; VList [Atom "c"]]
    let bagExpected = VList [Atom "a"; Atom "b"; Atom "c"]
    assertTrue "combineParBranchResults bag [[a;b]; [c]] = VList [a;b;c]"
               (bagResult = bagExpected)
    // Set: dedup'd concat.
    let setResult =
        combineParBranchResults MergeSet
            [VList [Atom "a"; Atom "b"]; VList [Atom "b"; Atom "c"]]
    let setExpected = VList [Atom "a"; Atom "b"; Atom "c"]
    assertTrue "combineParBranchResults set [[a;b]; [b;c]] = VList [a;b;c] (dedup)"
               (setResult = setExpected)
    // Findall: same as bag (preserves order + duplicates).
    let findallResult =
        combineParBranchResults MergeFindall
            [VList [Atom "a"]; VList [Atom "a"]; VList [Atom "b"]]
    let findallExpected = VList [Atom "a"; Atom "a"; Atom "b"]
    assertTrue "combineParBranchResults findall [[a]; [a]; [b]] = VList [a;a;b]"
               (findallResult = findallExpected)

// -- Scenario 18: findOuterEndAggregate scans forward to EndAggregate -----

let scenario_find_outer_end_aggregate () =
    let code = [|
        BeginAggregate ("sum", 2, 1)   // PC 0
        ParTryMeElsePc 3               // PC 1
        Proceed                         // PC 2
        ParTrustMe                      // PC 3
        Proceed                         // PC 4
        EndAggregate 2                  // PC 5
        Proceed                         // PC 6 — retPC
    |]
    let ctx = mkContext code Map.empty
    let retPC = findOuterEndAggregate ctx 1   // scan from ParTryMeElsePc
    assertTrue "findOuterEndAggregate: scans forward to EndAggregate's next PC"
               (retPC = 6)
    // Overrun: scan from past-the-end => 0 (halt).
    let retOverrun = findOuterEndAggregate ctx 100
    assertTrue "findOuterEndAggregate: overrun returns 0"
               (retOverrun = 0)

// -- Scenario 19: removeNearestAggFrame drops the nearest agg-frame CP ----

let scenario_remove_nearest_agg_frame () =
    let plainCP : ChoicePoint = {
        CpNextPC = 10; CpRegs = Array.empty; CpStack = []
        CpCP = 0; CpTrailLen = 0; CpHeapLen = 0
        CpBindings = Map.empty; CpCutBar = 0
        CpB0StackLen = 0
        CpAggFrame = None; CpBuiltin = None }
    let aggCP : ChoicePoint = {
        plainCP with CpAggFrame = Some { AggType = "sum"; AggValReg = 2
                                         AggResReg = 1; AggReturnPC = 0
                                         AggMergeStrategy = MergeSum } }
    let cps = [plainCP; aggCP; plainCP]
    let trimmed = removeNearestAggFrame cps
    assertTrue "removeNearestAggFrame: drops the aggregate-frame CP"
               (List.length trimmed = 2)
    assertTrue "removeNearestAggFrame: neither remaining CP has an AggFrame"
               (trimmed |> List.forall (fun cp -> cp.CpAggFrame.IsNone))

// -- Scenario 20: forkOrSequential dispatches forkable vs sequential -----
//
// Inside a non-forkable (MergeSequential) aggregate frame, ParTryMeElsePc
// must still fall back to sequential TryMeElse semantics (a CP gets pushed,
// PC advances past the Par* instruction).  No aggregate frame at all =>
// same fallback.  Inside a forkable aggregate, the fallback only kicks in
// if branch count < forkMinBranches (= 3).

let scenario_par_step_inside_sequential_aggregate () =
    let code = [| ParTryMeElsePc 99 |]
    let ctx = mkContext code Map.empty
    let aggCP : ChoicePoint = {
        CpNextPC = 0; CpRegs = Array.empty; CpStack = []
        CpCP = 0; CpTrailLen = 0; CpHeapLen = 0
        CpBindings = Map.empty; CpCutBar = 0
        CpB0StackLen = 0
        CpAggFrame = Some { AggType = "unknown"; AggValReg = 2
                            AggResReg = 1; AggReturnPC = 0
                            AggMergeStrategy = MergeSequential }
        CpBuiltin = None }
    let s0 = { mkEmptyState () with WsCPs = [aggCP]; WsCPsLen = 1 }
    match step ctx s0 code.[0] with
    | Some s ->
        assertTrue "ParTryMeElsePc (MergeSequential agg): fallback adds 1 CP"
                   (s.WsCPsLen = 2)
    | None ->
        assertTrue "ParTryMeElsePc inside agg should succeed" false

// -- Scenarios 21-23: end-to-end forkParBranches via run ------------------
//
// Drive the full BeginAggregate -> ParTryMeElsePc -> branch chain ->
// EndAggregate cycle through `run`, asserting on the aggregated result.
// This is the first test that exercises forkOrSequential's fork path
// from start to finish — earlier scenarios only unit-tested the helpers.
//
// IMPORTANT: AggResReg must be initialized to `Unbound vid` (a real
// var ID, not the -1 sentinel) BEFORE BeginAggregate fires, otherwise
// finalizeAggregate's `match resVal with | Some (Unbound vid) -> ...`
// arm doesn't bind anything (the sentinel branch via getReg returns
// None, falling through to the no-op `| _ ->` case).  In real Prolog
// codegen, the caller does this via a PutVariable that precedes the
// aggregate's BeginAggregate.
//
// Layout (PCs are positions in the WcCode array):
//   PC 0: BeginAggregate "sum" 2 1
//   PC 1: ParTryMeElsePc 5       -- forkable point; 3 branches
//   PC 2: PutConstant (Integer 10) 2   -- branch 1 body
//   PC 3: EndAggregate 2          -- finalize per-branch (AggReturnPC = 4)
//   PC 4: Proceed                 -- fork's post-EndAgg continuation
//   PC 5: ParRetryMeElsePc 9      -- branch 2 entry = 6
//   PC 6: PutConstant (Integer 20) 2
//   PC 7: EndAggregate 2
//   PC 8: Proceed
//   PC 9: ParTrustMe              -- branch 3 entry = 10
//   PC 10: PutConstant (Integer 30) 2
//   PC 11: EndAggregate 2
//   PC 12: Proceed
//
// Expected: WsRegs.[1] = Integer 60 (10 + 20 + 30) after run halts.

// mkResRegState returns a state with the result register at `resReg`
// initialized to Unbound 100 (a real var id) and WsPC = 1.  PC 0 is
// `run`'s halt sentinel, so all WAM layouts must start their first
// real instruction at PC 1 (or higher).
let mkResRegState (resReg: int) =
    let r = Array.create MaxRegs (Unbound -1)
    r.[resReg] <- Unbound 100   // valid Unbound vid for finalizeAggregate
    { mkEmptyState () with WsRegs = r; WsVarCounter = 101; WsPC = 1 }

let scenario_fork_sum_via_run () =
    let code = [|
        Proceed                         // PC 0 — unused (halt sentinel)
        BeginAggregate ("sum", 2, 1)   // PC 1
        ParTryMeElsePc 6                // PC 2 — elsePC = 6
        PutConstant (Integer 10, 2)    // PC 3 — branch 1 body
        EndAggregate 2                  // PC 4
        Proceed                         // PC 5 — branch 1 halts (AggReturnPC = 5)
        ParRetryMeElsePc 10             // PC 6 — branch 2 entry = 7
        PutConstant (Integer 20, 2)    // PC 7
        EndAggregate 2                  // PC 8
        Proceed                         // PC 9
        ParTrustMe                      // PC 10 — branch 3 entry = 11
        PutConstant (Integer 30, 2)    // PC 11
        EndAggregate 2                  // PC 12
        Proceed                         // PC 13
    |]
    let ctx = mkContext code Map.empty
    let s0 = mkResRegState 1
    match run ctx s0 with
    | Some sf ->
        let actual = derefVar sf.WsBindings sf.WsRegs.[1]
        assertTrue "fork sum via run: AggResReg (deref'd) = Integer 60 (10+20+30)"
                   (actual = Integer 60)
    | None ->
        assertTrue "fork sum via run should succeed" false

let scenario_fork_count_via_run () =
    // aggType = "count".  Each branch's EndAggregate finalizes a count
    // of 1 (one value pushed); fork combines via MergeCount which sums
    // per-branch counts => Integer 3.
    let code = [|
        Proceed                              // PC 0 — halt sentinel
        BeginAggregate ("count", 2, 1)      // PC 1
        ParTryMeElsePc 6                     // PC 2
        PutConstant (Integer 10, 2)         // PC 3
        EndAggregate 2                       // PC 4
        Proceed                              // PC 5
        ParRetryMeElsePc 10                  // PC 6
        PutConstant (Integer 20, 2)         // PC 7
        EndAggregate 2                       // PC 8
        Proceed                              // PC 9
        ParTrustMe                           // PC 10
        PutConstant (Integer 30, 2)         // PC 11
        EndAggregate 2                       // PC 12
        Proceed                              // PC 13
    |]
    let ctx = mkContext code Map.empty
    let s0 = mkResRegState 1
    match run ctx s0 with
    | Some sf ->
        let actual = derefVar sf.WsBindings sf.WsRegs.[1]
        assertTrue "fork count via run: AggResReg = Integer 3 (3 branches)"
                   (actual = Integer 3)
    | None ->
        assertTrue "fork count via run should succeed" false

let scenario_fork_bag_via_run () =
    // aggType = "bag".  Each branch's EndAggregate finalizes a one-
    // element VList [Integer X]; fork concats them.
    let code = [|
        Proceed                              // PC 0
        BeginAggregate ("bag", 2, 1)
        ParTryMeElsePc 6
        PutConstant (Integer 10, 2)
        EndAggregate 2
        Proceed
        ParRetryMeElsePc 10
        PutConstant (Integer 20, 2)
        EndAggregate 2
        Proceed
        ParTrustMe
        PutConstant (Integer 30, 2)
        EndAggregate 2
        Proceed
    |]
    let ctx = mkContext code Map.empty
    let s0 = mkResRegState 1
    let expected = VList [Integer 10; Integer 20; Integer 30]
    match run ctx s0 with
    | Some sf ->
        let actual = derefVar sf.WsBindings sf.WsRegs.[1]
        assertTrue "fork bag via run: AggResReg = VList [10; 20; 30]"
                   (actual = expected)
    | None ->
        assertTrue "fork bag via run should succeed" false

// -- Scenario 24: forkOrSequential dispatch for non-forkable strategy -----
//
// aggType = "unknown" → MergeSequential → not forkable.  We don't
// drive full run here because the multi-branch SEQUENTIAL dispatch
// path via Par* aliasing has a deeper issue (RetryMeElse modifies the
// top CP, but backtrack already popped it — leaving the agg frame as
// the modified target).  That's a pre-existing bug outside this PR's
// scope.  Instead we verify the DISPATCH DECISION: after the ParTryMeElsePc
// step in an "unknown" aggregate, the agg frame should still be on
// WsCPs (fork didn't consume it) AND a new TryMeElsePc CP should be on
// top (the sequential fallback pushed one).

let scenario_fork_sequential_fallback_dispatch () =
    // aggType = "unknown" → MergeSequential → not forkable.  We verify
    // the DISPATCH DECISION: after the ParTryMeElsePc step in an
    // "unknown" aggregate, the agg frame should still be on WsCPs (fork
    // didn't consume it) AND a new TryMeElsePc CP should be on top
    // (the sequential fallback pushed one).
    //
    // (Driving the full multi-branch sequential WAM via `run` exposes a
    // pre-existing bug — RetryMeElse modifies the top CP, but backtrack
    // already popped it — leaving the agg frame as the modified target.
    // Out of scope for this PR; tested at the dispatch level instead.)
    let code = [|
        Proceed                             // PC 0 — halt sentinel
        BeginAggregate ("unknown", 2, 1)   // PC 1
        ParTryMeElsePc 4                    // PC 2 — elsePC = 4
        Proceed                             // PC 3 — branch 1 placeholder
    |]
    let ctx = mkContext code Map.empty
    let s0 = mkResRegState 1
    // Step BeginAggregate (PC 1), then ParTryMeElsePc (PC 2).
    match step ctx s0 code.[1] with
    | Some s1 ->
        // After BeginAggregate: WsCPs = [aggCP], WsCPsLen = 1.
        assertTrue "BeginAggregate (unknown): pushes aggCP"
                   (s1.WsCPsLen = 1)
        match step ctx s1 code.[2] with
        | Some s2 ->
            // After ParTryMeElsePc on non-forkable agg: fallback pushed
            // a TryMeElsePc CP => 2 CPs total.
            assertTrue "ParTryMeElsePc (MergeSequential agg): fallback pushes a 2nd CP"
                       (s2.WsCPsLen = 2)
            assertTrue "ParTryMeElsePc (MergeSequential agg): top CP is non-agg (TryMeElse fallback)"
                       (s2.WsCPs.Head.CpAggFrame.IsNone)
            assertTrue "ParTryMeElsePc (MergeSequential agg): agg frame still present below"
                       (s2.WsCPs |> List.exists (fun cp -> cp.CpAggFrame.IsSome))
        | None ->
            assertTrue "ParTryMeElsePc should not fail" false
    | None ->
        assertTrue "BeginAggregate should not fail" false

// -- Scenario 25: forkOrSequential dispatch below forkMinBranches ---------
//
// 2 branches: forkable strategy but |branches| = 2 < forkMinBranches (3),
// so falls back to sequential.  Same dispatch-level assertion as scenario 24.

let scenario_fork_below_min_branches_dispatch () =
    let code = [|
        Proceed                             // PC 0
        BeginAggregate ("sum", 2, 1)       // PC 1
        ParTryMeElsePc 5                    // PC 2 — only 2 branches (entry 3 + entry 6)
        Proceed                             // PC 3
        Proceed                             // PC 4
        ParTrustMe                          // PC 5
        Proceed                             // PC 6
    |]
    let ctx = mkContext code Map.empty
    let s0 = mkResRegState 1
    match step ctx s0 code.[1] with
    | Some s1 ->
        match step ctx s1 code.[2] with
        | Some s2 ->
            // enumerateParBranches returns [3; 6] (length 2 < 3 threshold).
            assertTrue "ParTryMeElsePc (below threshold): fallback pushes TryMeElsePc CP"
                       (s2.WsCPsLen = 2)
            assertTrue "ParTryMeElsePc (below threshold): top CP is non-agg"
                       (s2.WsCPs.Head.CpAggFrame.IsNone)
        | None ->
            assertTrue "ParTryMeElsePc should not fail (below threshold)" false
    | None ->
        assertTrue "BeginAggregate should not fail" false

[<EntryPoint>]
let main _argv =
    scenario_put_constant ()
    scenario_get_constant_match ()
    scenario_get_constant_mismatch ()
    scenario_proceed_returns_to_cp ()
    scenario_compare_value ()
    scenario_deref_var ()
    scenario_enumerate_par_branches ()
    scenario_backtrack_run ()
    scenario_unify_val ()
    scenario_put_structure_dyn ()
    scenario_arg_specialized ()
    scenario_not_member_list ()
    scenario_not_member_const_atoms ()
    scenario_vset_family ()
    scenario_run_negation_parallel_all_fail ()
    scenario_run_negation_parallel_one_succeeds ()
    scenario_par_step_aliases_to_sequential ()
    scenario_merge_strategy_helpers ()
    scenario_combine_par_branch_results ()
    scenario_find_outer_end_aggregate ()
    scenario_remove_nearest_agg_frame ()
    scenario_par_step_inside_sequential_aggregate ()
    scenario_fork_sum_via_run ()
    scenario_fork_count_via_run ()
    scenario_fork_bag_via_run ()
    scenario_fork_sequential_fallback_dispatch ()
    scenario_fork_below_min_branches_dispatch ()
    let total = passes + fails
    printfn "RESULT %d/%d" passes total
    if fails > 0 then 1 else 0
