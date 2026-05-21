module Program

// Runtime query smoke for the F# WAM target.
//
// Drives queries against a 3-clause `parent/2` predicate compiled from
// real Prolog source through write_wam_fsharp_project/3.  The first
// end-to-end test of multi-clause predicate dispatch — earlier smokes
// constructed Instruction arrays inline (no WAM compiler in the loop)
// or verified the project just builds.
//
// Predicate definition (compiled to Predicates.fs by the harness):
//   parent(tom, bob).
//   parent(bob, ann).
//   parent(ann, eve).
//
// Each query scenario:
//   - Sets up A1, A2 in registers
//   - Calls dispatchCall ctx "parent_query_smoke/2" state
//   - Asserts on the resulting state
//
// Two pre-existing F# WAM dispatch bugs surfaced through this smoke
// (see PR description for details).  Scenarios that exercise these
// bugs assert the CURRENT (buggy) behavior so the smoke passes,
// documenting the bugs as living regression markers.  When the bugs
// are fixed in follow-up PRs, those assertions will need to flip.
//
//   Bug A — SwitchOnConstantPc at array index 0 unreachable: the WAM
//     compiler emits it as the first instruction, but the F# array
//     stores it at index 0, where `run`'s halt sentinel
//     (`if s.WsPC = 0 then Some s`) short-circuits before fetching.
//     The label "parent_query_smoke/2" → 1 points to the SECOND emitted
//     instruction (TryMeElsePc), bypassing indexed dispatch.
//
//   Bug B — Multi-backtrack chain breaks: `backtrack` pops the CP and
//     restores from it; `RetryMeElse`/`TrustMe` then expect to modify
//     /pop the top CP, but it's gone.  Chains of 3+ clauses (where
//     the match needs 2+ backtracks) fail — the second backtrack
//     finds an empty CP stack.
//
// Net effect: clauses 1 and 2 are reachable; clause 3+ is not.

open WamTypes
open WamRuntime
open Predicates

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then passes <- passes + 1; printfn "[PASS] %s" name
    else fails <- fails + 1; printfn "[FAIL] %s" name

let mkContext () =
    // Resolve TryMeElse / RetryMeElse / Jump labels to PC indices,
    // mirroring what the auto-generated Program.fs benchmark driver
    // does at startup.
    let foreignPreds : string list = []
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = Map.empty
      WcAtomDeintern      = Map.empty
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcCancellationToken = None }

let mkQueryState (a1: Value) (a2: Value) (vidStart: int) =
    let r = Array.create MaxRegs (Unbound -1)
    r.[1] <- a1
    r.[2] <- a2
    { WsPC         = 0
      WsRegs       = r
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
      WsVarCounter = vidStart
      WsBuilder    = None
      WsAggAccum   = [] }

// -- Working queries: clauses 1-2 reachable --------------------------------

let query_parent_tom_X () =
    // Clause 1 — direct match without backtrack.
    let ctx = mkContext ()
    let s = mkQueryState (Atom "tom") (Unbound 100) 101
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some result ->
        let answer = derefVar result.WsBindings result.WsRegs.[2]
        assertTrue "parent(tom, X): X = Atom \"bob\""
                   (answer = Atom "bob")
    | None ->
        assertTrue "parent(tom, X) should succeed (clause 1)" false

let query_parent_bob_X () =
    // Clause 2 — reached via one backtrack from clause 1.
    let ctx = mkContext ()
    let s = mkQueryState (Atom "bob") (Unbound 200) 201
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some result ->
        let answer = derefVar result.WsBindings result.WsRegs.[2]
        assertTrue "parent(bob, X): X = Atom \"ann\" (clause 2 via 1 backtrack)"
                   (answer = Atom "ann")
    | None ->
        assertTrue "parent(bob, X) should succeed" false

let query_parent_bob_ann () =
    // Both ground, clause 2 exact match.
    let ctx = mkContext ()
    let s = mkQueryState (Atom "bob") (Atom "ann") 300
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some _ ->
        assertTrue "parent(bob, ann) succeeds (exact ground match)" true
    | None ->
        assertTrue "parent(bob, ann) should succeed" false

let query_parent_bob_eve () =
    // Both ground, no match (bob's child is ann, not eve).
    let ctx = mkContext ()
    let s = mkQueryState (Atom "bob") (Atom "eve") 400
    match dispatchCall ctx "parent_query_smoke/2" s with
    | None ->
        assertTrue "parent(bob, eve) returns None (no match in any clause)" true
    | Some _ ->
        assertTrue "parent(bob, eve) should fail" false

let query_parent_zebra_X () =
    // A1 not in any clause — chain fully fails out.
    let ctx = mkContext ()
    let s = mkQueryState (Atom "zebra") (Unbound 500) 501
    match dispatchCall ctx "parent_query_smoke/2" s with
    | None ->
        assertTrue "parent(zebra, X) returns None (no matching clause)" true
    | Some _ ->
        assertTrue "parent(zebra, X) should fail" false

let query_X_parent_bob () =
    // A1 Unbound; first clause matches (tom, bob).
    let ctx = mkContext ()
    let s = mkQueryState (Unbound 600) (Atom "bob") 601
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some result ->
        let answer = derefVar result.WsBindings result.WsRegs.[1]
        assertTrue "parent(X, bob): X = Atom \"tom\" (clause 1)"
                   (answer = Atom "tom")
    | None ->
        assertTrue "parent(X, bob) should succeed" false

let query_X_parent_ann () =
    // A1 Unbound; match in clause 2 (bob, ann) — one backtrack from
    // clause 1.  Exercises the TryMe → backtrack → clause-2-body path.
    let ctx = mkContext ()
    let s = mkQueryState (Unbound 700) (Atom "ann") 701
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some result ->
        let answer = derefVar result.WsBindings result.WsRegs.[1]
        assertTrue "parent(X, ann): X = Atom \"bob\" (clause 2 via 1 backtrack)"
                   (answer = Atom "bob")
    | None ->
        assertTrue "parent(X, ann) should succeed" false

// -- Clause-3 queries: now reachable via indexed dispatch for ground A1
//                       (Bug A fixed in this PR), still broken for unbound
//                       A1 because of Bug B (chain pop bug).
//
// Bug A (sentinel + Retry/Trust no-op-on-empty + SwitchOnConstantPc
// fall-through-on-miss) lets ground A1 reach any clause via the indexed
// dispatch table.  parent(ann, X) and parent(ann, eve) now succeed.
//
// Bug B (backtrack pops CP, RetryMeElse expects to modify it) still bites
// when A1 is Unbound and we need to walk past clause 2 — there's no
// indexed path available, the chain runs out after the first backtrack.
// parent(X, eve) still asserts [KNOWN BUG].

let query_parent_ann_X () =
    // Ground A1 = ann: indexed dispatch via SwitchOnConstantPc jumps
    // directly to clause 3.  Should now succeed with X = eve.
    let ctx = mkContext ()
    let s = mkQueryState (Atom "ann") (Unbound 800) 801
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some result ->
        let answer = derefVar result.WsBindings result.WsRegs.[2]
        assertTrue "parent(ann, X): X = Atom \"eve\" (clause 3 via indexed dispatch)"
                   (answer = Atom "eve")
    | None ->
        assertTrue "parent(ann, X) should succeed (clause 3 indexed)" false

let query_parent_ann_eve () =
    // Both ground, clause 3 exact match via indexed dispatch.
    let ctx = mkContext ()
    let s = mkQueryState (Atom "ann") (Atom "eve") 900
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some _ ->
        assertTrue "parent(ann, eve) succeeds (clause 3 exact match via indexed dispatch)"
                   true
    | None ->
        assertTrue "parent(ann, eve) should succeed" false

let query_X_parent_eve () =
    // A1 Unbound, A2 ground.  No indexed dispatch (A1 unbound → falls
    // through to the linear chain).  Match is in clause 3 (X = ann),
    // reached via 2 backtracks: clause 1 (tom, bob) fails on A2,
    // clause 2 (bob, ann) fails on A2, clause 3 (ann, eve) matches.
    // Now works after Bug B fix (PR follow-up to #2351): backtrack
    // keeps the CP on the stack, so RetryMeElse can modify and
    // TrustMe can pop normally.
    let ctx = mkContext ()
    let s = mkQueryState (Unbound 1000) (Atom "eve") 1001
    match dispatchCall ctx "parent_query_smoke/2" s with
    | Some result ->
        let answer = derefVar result.WsBindings result.WsRegs.[1]
        assertTrue "parent(X, eve): X = Atom \"ann\" (clause 3 via 2 backtracks)"
                   (answer = Atom "ann")
    | None ->
        assertTrue "parent(X, eve) should succeed via clause-3 chain walk" false

[<EntryPoint>]
let main _argv =
    // Working scenarios (clauses 1-2 reachable):
    query_parent_tom_X ()
    query_parent_bob_X ()
    query_parent_bob_ann ()
    query_parent_bob_eve ()
    query_parent_zebra_X ()
    query_X_parent_bob ()
    query_X_parent_ann ()
    // Currently-broken scenarios (clause 3 unreachable):
    query_parent_ann_X ()
    query_parent_ann_eve ()
    query_X_parent_eve ()
    let total = passes + fails
    printfn "RESULT %d/%d" passes total
    if fails > 0 then 1 else 0
