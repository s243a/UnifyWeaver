module Program

// Runtime smoke for the Phase-I lowered emitter.
//
// Replaces the default Program.fs (the category-ancestor benchmark
// driver) and drives lowered functions that were emitted by
// lower_predicate_to_fsharp/4 from hand-rolled WAM bodies containing
// Phase-I specialized instructions.
//
// The test driver (tests/core/test_wam_fsharp_dotnet_smoke.pl) splices
// the lowered functions into Lowered.fs at the predictable names below,
// then runs this Program.fs to call each one with crafted register
// inputs and assert on the resulting WamState.
//
// Predicate-name → lowered-function-name convention (from
// lower_predicate_to_fsharp's sanitizer):
//   phase_i_arg/3   -> lowered_phase_i_arg_3
//   phase_i_nml/2   -> lowered_phase_i_nml_2
//   phase_i_vset/2  -> lowered_phase_i_vset_2
//   phase_i_nmca/1  -> lowered_phase_i_nmca_1
//   phase_i_psd/3   -> lowered_phase_i_psd_3

open WamTypes
open WamRuntime
open Lowered

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn "[PASS] %s" name
    else
        fails <- fails + 1
        printfn "[FAIL] %s" name

let mkContext () =
    { WcCode              = [||]
      WcLabels            = Map.empty
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = Map.empty
      WcAtomDeintern      = Map.empty
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcCancellationToken = None }

let mkState (regs: (int * Value) list) =
    let r = Array.create MaxRegs (Unbound -1)
    for (i, v) in regs do
        r.[i] <- v
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
      WsVarCounter = 0
      WsBuilder    = None
      WsAggAccum   = [] }

// -- arg/3 specialized ----------------------------------------------------

let scenario_arg () =
    // phase_i_arg/3 body: `arg 2 A1 X1\n  proceed`
    // With A1 = Str("foo", [Atom "x"; Atom "y"; Atom "z"]),
    // X1 (reg 101) should be set to Atom "y" (2nd subterm).
    let ctx = mkContext ()
    let s = mkState [
        (1, Str ("foo", [Atom "x"; Atom "y"; Atom "z"]))
    ]
    match lowered_phase_i_arg_3 ctx s with
    | Some final ->
        assertTrue "lowered arg(2, A1, X1): X1 = Atom \"y\""
                   (final.WsRegs.[101] = Atom "y")
    | None ->
        assertTrue "lowered arg should succeed on a 3-arg Str" false

    // Mismatch case: N=2 on a 1-arg Str -> should fail (returns None).
    let sMiss = mkState [
        (1, Str ("foo", [Atom "x"]))
    ]
    match lowered_phase_i_arg_3 ctx sMiss with
    | Some _ ->
        assertTrue "lowered arg out-of-range should fail" false
    | None ->
        assertTrue "lowered arg out-of-range: returns None" true

// -- not_member_list ------------------------------------------------------

let scenario_nml () =
    // phase_i_nml/2 body: `not_member_list A1 A2\n  proceed`
    // A1 = Atom "z" (not in list), A2 = VList [a; b; c] -> succeed.
    let ctx = mkContext ()
    let s = mkState [
        (1, Atom "z")
        (2, VList [Atom "a"; Atom "b"; Atom "c"])
    ]
    match lowered_phase_i_nml_2 ctx s with
    | Some _ ->
        assertTrue "lowered not_member_list: z notin [a,b,c] -> success" true
    | None ->
        assertTrue "lowered not_member_list should succeed" false

    // A1 = Atom "b" (in list) -> fail.
    let sBad = mkState [
        (1, Atom "b")
        (2, VList [Atom "a"; Atom "b"; Atom "c"])
    ]
    match lowered_phase_i_nml_2 ctx sBad with
    | Some _ ->
        assertTrue "lowered not_member_list: b in list should fail" false
    | None ->
        assertTrue "lowered not_member_list: returns None when X in list" true

// -- VSet family: build_empty_set + set_insert + not_member_set -----------

let scenario_vset () =
    // phase_i_vset/2 body:
    //   build_empty_set X1
    //   set_insert A1 X1 X2
    //   not_member_set A2 X2
    //   proceed
    //
    // With A1 = Atom "a", A2 = Atom "b":
    //   X1 = VSet {}
    //   X2 = VSet {"a"}
    //   not_member_set "b" {"a"} -> success.
    let ctx = mkContext ()
    let s = mkState [
        (1, Atom "a")
        (2, Atom "b")
    ]
    match lowered_phase_i_vset_2 ctx s with
    | Some final ->
        assertTrue "lowered vset chain: succeeds"
                   true
        assertTrue "lowered vset chain: X1 (reg 101) = VSet Set.empty"
                   (final.WsRegs.[101] = VSet Set.empty)
        assertTrue "lowered vset chain: X2 (reg 102) = VSet {\"a\"}"
                   (final.WsRegs.[102] = VSet (Set.ofList ["a"]))
    | None ->
        assertTrue "lowered vset chain should succeed" false

    // With A2 = Atom "a" (same as inserted) -> not_member_set fails.
    let sBad = mkState [
        (1, Atom "a")
        (2, Atom "a")
    ]
    match lowered_phase_i_vset_2 ctx sBad with
    | Some _ ->
        assertTrue "lowered vset chain: A2 in set should fail" false
    | None ->
        assertTrue "lowered vset chain: returns None on hit" true

// -- not_member_const_atoms (variable-arity) -------------------------------

let scenario_nmca () =
    // phase_i_nmca/1 body: `not_member_const_atoms A1 foo bar baz\n  proceed`
    // A1 = Atom "qux" (not in {foo, bar, baz}) -> succeed.
    let ctx = mkContext ()
    let s = mkState [
        (1, Atom "qux")
    ]
    match lowered_phase_i_nmca_1 ctx s with
    | Some _ ->
        assertTrue "lowered not_member_const_atoms: qux notin {foo,bar,baz}" true
    | None ->
        assertTrue "lowered not_member_const_atoms should succeed" false

    // A1 = Atom "bar" (in list) -> fail.
    let sBad = mkState [
        (1, Atom "bar")
    ]
    match lowered_phase_i_nmca_1 ctx sBad with
    | Some _ ->
        assertTrue "lowered not_member_const_atoms: bar in list should fail" false
    | None ->
        assertTrue "lowered not_member_const_atoms: returns None on hit" true

    // A1 = Integer 42 (non-atom ground) -> succeed (can't unify with atoms).
    let sInt = mkState [
        (1, Integer 42)
    ]
    match lowered_phase_i_nmca_1 ctx sInt with
    | Some _ ->
        assertTrue "lowered not_member_const_atoms: Integer can't unify with atoms" true
    | None ->
        assertTrue "lowered not_member_const_atoms (Integer): should succeed" false

// -- PutStructureDyn ------------------------------------------------------

let scenario_psd () =
    // phase_i_psd/3 body: `put_structure_dyn A1 A2 A3\n  proceed`
    // With A1 = Atom "foo", A2 = Integer 2 (arity), A3 = target reg.
    // Effect: WsBuilder = BuildStruct("foo", 3, 2, [])
    let ctx = mkContext ()
    let s = mkState [
        (1, Atom "foo")
        (2, Integer 2)
    ]
    match lowered_phase_i_psd_3 ctx s with
    | Some final ->
        match final.WsBuilder with
        | Some (BuildStruct (fn, reg, arity, args)) ->
            assertTrue "lowered put_structure_dyn: builder functor = \"foo\""
                       (fn = "foo")
            assertTrue "lowered put_structure_dyn: builder target reg = 3"
                       (reg = 3)
            assertTrue "lowered put_structure_dyn: builder arity = 2"
                       (arity = 2)
            assertTrue "lowered put_structure_dyn: builder args = []"
                       (args = [])
        | _ ->
            assertTrue "lowered put_structure_dyn: builder should be BuildStruct" false
    | None ->
        assertTrue "lowered put_structure_dyn should succeed" false

    // Non-Atom name -> failure.
    let sBad = mkState [
        (1, Integer 42)
        (2, Integer 0)
    ]
    match lowered_phase_i_psd_3 ctx sBad with
    | Some _ ->
        assertTrue "lowered put_structure_dyn: non-Atom name should fail" false
    | None ->
        assertTrue "lowered put_structure_dyn (non-Atom): returns None" true

[<EntryPoint>]
let main _argv =
    scenario_arg ()
    scenario_nml ()
    scenario_vset ()
    scenario_nmca ()
    scenario_psd ()
    let total = passes + fails
    printfn "RESULT %d/%d" passes total
    if fails > 0 then 1 else 0
