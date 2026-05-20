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
      WsAggAccum   = [] }

let mkContext (code: Instruction array) (labels: Map<string, int>) =
    { WcCode              = code
      WcLabels            = labels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = Map.empty
      WcAtomDeintern      = Map.empty
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty }

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
    let total = passes + fails
    printfn "RESULT %d/%d" passes total
    if fails > 0 then 1 else 0
