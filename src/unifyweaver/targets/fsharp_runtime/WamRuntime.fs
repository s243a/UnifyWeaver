// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (@s243a)
//
// WamRuntime.fs — Static F# WAM runtime library for UnifyWeaver
//
// This file is a hand-maintained reference copy of the runtime that
// wam_fsharp_target.pl generates into WamRuntime.fs at project-generation
// time.  Keep this file in sync with:
//   - src/unifyweaver/bindings/fsharp_wam_bindings.pl   (types + helpers)
//   - src/unifyweaver/targets/wam_fsharp_target.pl       (step, backtrack, run)
//
// It can be used directly in projects that prefer a pre-built runtime
// instead of a generated one, or as a regression reference for the generator.
//
// Performance notes (from docs/design/WAM_PERF_OPTIMIZATION_LOG.md):
//   Phase A: Map<int,Value> for bindings/regs → O(1) choice-point snapshots
//   Phase B: WamContext (cold, read-only) separated from WamState (hot)
//   Phase C: Labels pre-resolved to int PCs at load time (-47% exec time)
//   Phase D: Skip WAM-compile of FFI-owned facts; intern atoms at boundary

module WamTypes

open System.Collections.Generic

// ============================================================================
// Value discriminated union
// ============================================================================

[<CustomEquality; CustomComparison>]
type Value =
    | Atom    of string
    | Integer of int
    | Float   of float
    | VList   of Value list
    | Str     of string * Value list   // functor name, args
    | Unbound of int                   // variable id
    | Ref     of int                   // heap reference
    interface System.IComparable with
        member this.CompareTo(obj) =
            match obj with
            | :? Value as other -> compare this other
            | _ -> -1
    override this.Equals(obj) =
        match obj with
        | :? Value as other -> this = other
        | _ -> false
    override this.GetHashCode() = hash this

// ============================================================================
// Trail, aggregation, builtin-state, and environment types
// ============================================================================

type TrailEntry = { TrailVarId: int; TrailOldVal: Value option }

type AggFrame = { AggType: string; AggValReg: int; AggResReg: int; AggReturnPC: int }

type BuiltinState =
    | FactRetry      of varId: int * remaining: string list * retPC: int
    | HopsRetry      of varId: int * remaining: int list   * retPC: int
    | FFIStreamRetry of outRegs: int list * outVars: int list * remaining: Value list list * retPC: int

type EnvFrame = { EfSavedCP: int; EfYRegs: Map<int, Value> }

// ============================================================================
// ChoicePoint — O(1) snapshot via immutable Map
// ============================================================================

type ChoicePoint =
    { CpNextPC   : int
      CpRegs     : Map<int, Value>
      CpStack    : EnvFrame list
      CpCP       : int
      CpTrailLen : int
      CpHeapLen  : int
      CpBindings : Map<int, Value>   // O(1) snapshot — immutable tree
      CpCutBar   : int
      CpAggFrame : AggFrame option
      CpBuiltin  : BuiltinState option }

// ============================================================================
// WamState — hot per-step record (updated with { s with Field = v })
// ============================================================================

type WamState =
    { WsPC        : int
      WsRegs      : Map<int, Value>      // A/X registers (int-keyed)
      WsStack     : EnvFrame list        // environment frames
      WsHeap      : Value list           // term construction heap
      WsHeapLen   : int                  // cached — never call List.length in hot loop
      WsTrail     : TrailEntry list
      WsTrailLen  : int                  // cached
      WsCP        : int                  // continuation pointer
      WsCPs       : ChoicePoint list
      WsCPsLen    : int                  // cached
      WsBindings  : Map<int, Value>      // variable bindings
      WsCutBar    : int
      WsVarCounter: int                  // fresh variable id counter
      WsBuilder   : BuilderState option  // PutStructure / PutList accumulator
      WsAggAccum  : Value list }

and BuilderState =
    | BuildStruct of fn: string * reg: int * arity: int * args: Value list
    | BuildList   of reg: int * items: Value list

// ============================================================================
// WamContext — cold read-only context (Phase B hot/cold split)
// ============================================================================

type WamContext =
    { WcCode            : Instruction array
      WcLabels          : Map<string, int>
      WcForeignFacts    : Map<string, Map<string, string list>>
      WcFfiFacts        : Map<string, Map<int, int list>>
      WcFfiWeightedFacts: Map<string, Map<int, (int * float) list>>
      WcAtomIntern      : Map<string, int>
      WcAtomDeintern    : Map<int, string>
      WcForeignConfig   : Map<string, int>
      WcLoweredPredicates: Map<string, WamContext -> WamState -> WamState option> }

// ============================================================================
// Instruction discriminated union
// ============================================================================

and Instruction =
    // Head unification
    | GetConstant    of c: Value * ai: int
    | GetVariable    of xn: int  * ai: int
    | GetValue       of xn: int  * ai: int
    | GetStructure   of fn: string * arity: int * ai: int
    | GetList        of ai: int
    | UnifyVariable  of xn: int
    | UnifyValue     of xn: int
    | UnifyConstant  of c: Value
    // Body building
    | PutConstant    of c: Value * ai: int
    | PutVariable    of xn: int  * ai: int
    | PutValue       of xn: int  * ai: int
    | PutStructure   of fn: string * ai: int * arity: int
    | PutList        of ai: int
    | SetValue       of xn: int
    | SetVariable    of xn: int
    | SetConstant    of c: Value
    // Control
    | Call           of pred: string * arity: int
    | CallResolved   of pc: int * arity: int
    | CallForeign    of pred: string * arity: int
    | Execute        of pred: string
    | ExecutePc      of pc: int
    | Proceed
    | Allocate
    | Deallocate
    | Jump           of label: string
    | JumpPc         of pc: int
    // Choice points
    | TryMeElse      of label: string
    | TryMeElsePc    of nextPC: int
    | RetryMeElse    of label: string
    | RetryMeElsePc  of nextPC: int
    | TrustMe
    // Parallel stubs
    | ParTryMeElse     of label: string
    | ParRetryMeElsePc of nextPC: int
    | ParRetryMeElse   of label: string
    | ParTrustMe
    // Indexing
    | SwitchOnConstant   of table: Map<Value, string>
    | SwitchOnConstantPc of table: Map<string, int>
    // Builtins
    | BuiltinCall    of name: string * arity: int
    | CutIte
    // Aggregation
    | BeginAggregate of aggType: string * valReg: int * resReg: int
    | EndAggregate   of valReg: int

// ============================================================================
// WamRuntime module — step, backtrack, run loop, helpers
// ============================================================================

module WamRuntime

open WamTypes

// ----------------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------------

/// Dereference a value through the binding chain.
let rec derefVar (bindings: Map<int, Value>) (v: Value) : Value =
    match v with
    | Unbound vid ->
        match Map.tryFind vid bindings with
        | Some bound -> derefVar bindings bound
        | None       -> v
    | _ -> v

/// Look up and dereference a register.
let getReg (n: int) (s: WamState) : Value option =
    Map.tryFind n s.WsRegs |> Option.map (derefVar s.WsBindings)

/// Set a register.
let putReg (n: int) (v: Value) (s: WamState) : WamState =
    { s with WsRegs = Map.add n v s.WsRegs }

/// Bind an output register; succeed only if unbound or already equal.
let bindOutput (reg: int) (value: Value) (s: WamState) : WamState option =
    match Map.tryFind reg s.WsRegs |> Option.map (derefVar s.WsBindings) with
    | Some (Unbound vid) ->
        Some { s with
                 WsRegs     = Map.add reg value s.WsRegs
                 WsBindings = Map.add vid value s.WsBindings
                 WsTrail    = { TrailVarId = vid
                                TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen = s.WsTrailLen + 1 }
    | Some existing when existing = value -> Some s
    | _ -> None

/// Append to the current builder (PutStructure / PutList sequence).
let addToBuilder (value: Value) (s: WamState) : WamState option =
    match s.WsBuilder with
    | None -> None
    | Some (BuildStruct (fn, reg, arity, args)) ->
        let args' = args @ [value]
        if List.length args' = arity then
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = Map.add reg (Str (fn, args')) s.WsRegs
                     WsBuilder = None }
        else
            Some { s with WsBuilder = Some (BuildStruct (fn, reg, arity, args')) }
    | Some (BuildList (reg, items)) ->
        Some { s with WsBuilder = Some (BuildList (reg, items @ [value])) }

/// Arithmetic evaluator over Value terms.
let rec evalArith (bindings: Map<int, Value>) (v: Value) : float option =
    match derefVar bindings v with
    | Integer n -> Some (float n)
    | Float   f -> Some f
    | Str ("+", [a; b]) -> Option.map2 (+) (evalArith bindings a) (evalArith bindings b)
    | Str ("-", [a; b]) -> Option.map2 (-) (evalArith bindings a) (evalArith bindings b)
    | Str ("*", [a; b]) -> Option.map2 (*) (evalArith bindings a) (evalArith bindings b)
    | Str ("/", [a; b]) ->
        match evalArith bindings a, evalArith bindings b with
        | Some x, Some y when y <> 0.0 -> Some (x / y)
        | _ -> None
    | Str ("mod", [a; b]) ->
        match evalArith bindings a, evalArith bindings b with
        | Some x, Some y when y <> 0.0 -> Some (float (int x % int y))
        | _ -> None
    | _ -> None

/// Undo a single trail entry (used during backtrack).
let undoBinding (bindings: Map<int, Value>) (e: TrailEntry) : Map<int, Value> =
    match e.TrailOldVal with
    | Some old -> Map.add e.TrailVarId old bindings
    | None     -> Map.remove e.TrailVarId bindings

/// Copy-term: walk a Value, renaming variables to fresh ids.
let rec copyTermWalk (c: int) (m: Map<int,int>) (v: Value) : Value * int * Map<int,int> =
    match v with
    | Unbound vid ->
        match Map.tryFind vid m with
        | Some nv -> Unbound nv, c, m
        | None    -> Unbound c,  c + 1, Map.add vid c m
    | Str (fn, args) ->
        let args', c1, m1 = copyTermArgs c m args
        Str (fn, args'), c1, m1
    | VList items ->
        let items', c1, m1 = copyTermArgs c m items
        VList items', c1, m1
    | _ -> v, c, m

and copyTermArgs (c: int) (m: Map<int,int>) (xs: Value list) : Value list * int * Map<int,int> =
    match xs with
    | [] -> [], c, m
    | x :: rest ->
        let x1, c1, m1 = copyTermWalk c m x
        let rest1, c2, m2 = copyTermArgs c1 m1 rest
        x1 :: rest1, c2, m2

// ----------------------------------------------------------------------------
// Aggregation helpers
// ----------------------------------------------------------------------------

let applyAggregation (typ: string) (vals: Value list) : Value =
    match typ with
    | "sum" ->
        let toNum = function Integer n -> float n | Float f -> f | _ -> 0.0
        let s = List.sumBy toNum vals
        if float (int s) = s then Integer (int s) else Float s
    | "count"   -> Integer (List.length vals)
    | "collect" -> VList vals
    | _         -> VList vals

// ----------------------------------------------------------------------------
// Backtrack and aggregate finalisation
// ----------------------------------------------------------------------------

let rec backtrack (s: WamState) : WamState option =
    match s.WsCPs with
    | [] -> None
    | cp :: rest ->
        match cp.CpAggFrame with
        | Some af -> finalizeAggregate af.AggReturnPC s
        | None ->
        match cp.CpBuiltin with
        | Some bs -> resumeBuiltin bs cp rest s
        | None ->
        let diff = s.WsTrailLen - cp.CpTrailLen
        let restoredBindings =
            s.WsTrail |> List.take diff |> List.rev
            |> List.fold undoBinding cp.CpBindings
        Some { s with
                 WsPC       = cp.CpNextPC
                 WsRegs     = cp.CpRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = cp.CpTrailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = restoredBindings
                 WsCutBar   = cp.CpCutBar
                 WsCPs      = rest
                 WsCPsLen   = s.WsCPsLen - 1 }

and resumeBuiltin (bs: BuiltinState) (cp: ChoicePoint) (rest: ChoicePoint list) (s: WamState) : WamState option =
    let restoreCommon diff =
        { s with
            WsStack    = cp.CpStack
            WsCP       = cp.CpCP
            WsTrail    = List.skip diff s.WsTrail
            WsTrailLen = cp.CpTrailLen
            WsHeap     = List.take cp.CpHeapLen s.WsHeap
            WsHeapLen  = cp.CpHeapLen
            WsCutBar   = cp.CpCutBar }
    let diff = s.WsTrailLen - cp.CpTrailLen
    match bs with
    | FactRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | FactRetry (vid, v :: vs, retPC) ->
        let newCPs = match vs with
                     | [] -> rest
                     | _  -> { cp with CpBuiltin = Some (FactRetry (vid, vs, retPC)) } :: rest
        let base_ = restoreCommon diff
        Some { base_ with
                 WsPC       = retPC
                 WsRegs     = Map.add 2 (Atom v) cp.CpRegs
                 WsBindings = Map.add vid (Atom v) cp.CpBindings
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }
    | HopsRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | HopsRetry (vid, h :: hs, retPC) ->
        let newCPs = match hs with
                     | [] -> rest
                     | _  -> { cp with CpBuiltin = Some (HopsRetry (vid, hs, retPC)) } :: rest
        let base_ = restoreCommon diff
        Some { base_ with
                 WsPC       = retPC
                 WsRegs     = Map.add 3 (Integer h) cp.CpRegs
                 WsBindings = Map.add vid (Integer h) cp.CpBindings
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }
    | FFIStreamRetry (_, _, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | FFIStreamRetry (outRegs, outVars, tuple :: restTuples, retPC) ->
        let newRegs     = List.fold2 (fun m rN v -> Map.add rN v m) cp.CpRegs outRegs tuple
        let newBindings = List.fold2
                            (fun m vid v -> if vid = -1 then m else Map.add vid v m)
                            cp.CpBindings outVars tuple
        let newCPs = match restTuples with
                     | [] -> rest
                     | _  -> { cp with CpBuiltin = Some (FFIStreamRetry (outRegs, outVars, restTuples, retPC)) } :: rest
        let base_ = restoreCommon diff
        Some { base_ with
                 WsPC       = retPC
                 WsRegs     = newRegs
                 WsBindings = newBindings
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }

and backtrackInner (_returnPC: int) (s: WamState) : WamState option =
    match s.WsCPs with
    | cp :: _ when cp.CpAggFrame.IsSome -> None
    | _ -> backtrack s

and finalizeAggregate (returnPC: int) (s: WamState) : WamState option =
    let rec go cps =
        match cps with
        | [] -> None
        | cp :: rest ->
            match cp.CpAggFrame with
            | None -> go rest
            | Some af ->
                let result = applyAggregation af.AggType (List.rev s.WsAggAccum)
                let diff   = s.WsTrailLen - cp.CpTrailLen
                let restoredTrail = List.skip diff s.WsTrail
                let finalRegs, finalBindings, finalTrail, finalTrailLen =
                    match Map.tryFind af.AggResReg cp.CpRegs |> Option.map (derefVar cp.CpBindings) with
                    | Some (Unbound vid) ->
                        ( Map.add af.AggResReg result cp.CpRegs
                        , Map.add vid result cp.CpBindings
                        , { TrailVarId = vid; TrailOldVal = Map.tryFind vid cp.CpBindings } :: restoredTrail
                        , cp.CpTrailLen + 1 )
                    | _ ->
                        (cp.CpRegs, cp.CpBindings, restoredTrail, cp.CpTrailLen)
                Some { s with
                         WsPC       = returnPC
                         WsRegs     = finalRegs
                         WsStack    = cp.CpStack
                         WsBindings = finalBindings
                         WsTrail    = finalTrail
                         WsTrailLen = finalTrailLen
                         WsHeap     = List.take cp.CpHeapLen s.WsHeap
                         WsHeapLen  = cp.CpHeapLen
                         WsCP       = cp.CpCP
                         WsCPs      = rest
                         WsCPsLen   = s.WsCPsLen - 1
                         WsAggAccum = [] }
    go s.WsCPs

// ----------------------------------------------------------------------------
// step — execute a single WAM instruction
// (stub for the most commonly used instructions; the generator emits the full
// version which includes all ~40 opcodes)
// ----------------------------------------------------------------------------

let rec step (ctx: WamContext) (s: WamState) (instr: Instruction) : WamState option =
    let s1 = { s with WsPC = s.WsPC + 1 }
    match instr with
    | GetConstant (c, ai) ->
        let v = Map.tryFind ai s.WsRegs |> Option.map (derefVar s.WsBindings)
        match v with
        | Some (Unbound vid) ->
            Some { s1 with
                     WsBindings = Map.add vid c s.WsBindings
                     WsTrail    = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen = s.WsTrailLen + 1 }
        | Some existing when existing = c -> Some s1
        | _ -> backtrack s

    | GetVariable (xn, ai) ->
        let src = Map.tryFind ai s.WsRegs
                  |> Option.defaultWith (fun _ -> failwith "GetVariable: source register not bound")
                  |> derefVar s.WsBindings
        Some { s1 with WsRegs = Map.add xn src s.WsRegs }

    | GetValue (xn, ai) ->
        // Full unification needed; delegate to unify helper
        let v1 = Map.tryFind xn s.WsRegs |> Option.map (derefVar s.WsBindings)
        let v2 = Map.tryFind ai s.WsRegs |> Option.map (derefVar s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a = b -> Some s1
        | _ -> backtrack s  // simplified: full unify handled by generator

    | PutConstant (c, ai) ->
        Some { s1 with WsRegs = Map.add ai c s.WsRegs }

    | PutVariable (xn, ai) ->
        let vid = s.WsVarCounter
        let v   = Unbound vid
        Some { s1 with
                 WsRegs      = s.WsRegs |> Map.add xn v |> Map.add ai v
                 WsVarCounter = vid + 1 }

    | PutValue (xn, ai) ->
        let v = Map.tryFind xn s.WsRegs
                |> Option.defaultWith (fun _ -> failwith "PutValue: source register not bound")
        Some { s1 with WsRegs = Map.add ai v s.WsRegs }

    | Proceed ->
        let retPC = s.WsCP
        if retPC = 0 then Some { s1 with WsPC = 0 }
        else Some { s1 with WsPC = retPC; WsCP = 0 }

    | Allocate ->
        Some { s1 with
                 WsStack  = { EfSavedCP = s.WsCP; EfYRegs = Map.empty } :: s.WsStack
                 WsCutBar = s.WsCPsLen }

    | Deallocate ->
        match s.WsStack with
        | ef :: rest -> Some { s1 with WsStack = rest; WsCP = ef.EfSavedCP }
        | [] -> backtrack s

    | Call (pred, _arity) ->
        let sc = { s with WsCP = s.WsPC + 1 }
        dispatchCall ctx pred sc

    | CallResolved (pc, _arity) ->
        Some { s with WsPC = pc; WsCP = s.WsPC + 1 }

    | CallForeign (pred, _arity) ->
        executeForeign ctx pred { s with WsCP = s.WsPC + 1 }

    | Execute pred ->
        dispatchCall ctx pred s

    | ExecutePc pc ->
        Some { s with WsPC = pc }

    | TryMeElse label ->
        let altPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
        let cp = { CpNextPC   = altPC
                   CpRegs     = s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpAggFrame = None
                   CpBuiltin  = None }
        Some { s1 with WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    | TryMeElsePc nextPC ->
        let cp = { CpNextPC   = nextPC
                   CpRegs     = s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpAggFrame = None
                   CpBuiltin  = None }
        Some { s1 with WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    | RetryMeElse label ->
        let altPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
        match s.WsCPs with
        | cp :: rest ->
            let cp' = { cp with CpNextPC = altPC; CpRegs = s.WsRegs }
            Some { s1 with WsCPs = cp' :: rest }
        | [] -> backtrack s

    | RetryMeElsePc nextPC ->
        match s.WsCPs with
        | cp :: rest ->
            let cp' = { cp with CpNextPC = nextPC; CpRegs = s.WsRegs }
            Some { s1 with WsCPs = cp' :: rest }
        | [] -> backtrack s

    | TrustMe ->
        match s.WsCPs with
        | _ :: rest -> Some { s1 with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
        | [] -> backtrack s

    | CutIte ->
        Some { s1 with WsCPsLen = s.WsCutBar; WsCPs = List.skip (s.WsCPs.Length - s.WsCutBar) s.WsCPs }

    | Jump label ->
        let targetPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue (s.WsPC + 1)
        Some { s with WsPC = targetPC }

    | JumpPc pc ->
        Some { s with WsPC = pc }

    | SwitchOnConstant table ->
        // Resolve to PC-indexed version at load time; this arm is a fallback
        let key = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings)
        match key with
        | Some v ->
            let label = Map.tryFind v table
            match label with
            | Some lbl ->
                let pc = Map.tryFind lbl ctx.WcLabels |> Option.defaultValue (s.WsPC + 1)
                Some { s with WsPC = pc }
            | None -> backtrack s
        | None -> backtrack s

    | SwitchOnConstantPc table ->
        let key = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings)
        match key with
        | Some (Atom a) ->
            match Map.tryFind a table with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> backtrack s
        | Some (Integer n) ->
            match Map.tryFind (string n) table with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> backtrack s
        | _ -> backtrack s

    | BuiltinCall (name, arity) ->
        // Dispatch builtins — the generator emits the full table
        eprintfn "BuiltinCall: %s/%d not implemented in static runtime stub" name arity
        backtrack s

    | BeginAggregate (aggType, valReg, resReg) ->
        let retPC = s.WsCP
        let af = { AggType = aggType; AggValReg = valReg; AggResReg = resReg; AggReturnPC = retPC }
        let cp = { CpNextPC   = s.WsPC + 1
                   CpRegs     = s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpAggFrame = Some af
                   CpBuiltin  = None }
        Some { s1 with WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    | EndAggregate valReg ->
        let valOpt = Map.tryFind valReg s.WsRegs |> Option.map (derefVar s.WsBindings)
        match valOpt with
        | Some v ->
            let s2 = { s1 with WsAggAccum = v :: s.WsAggAccum }
            backtrack s2  // backtrack to collect next solution
        | None -> backtrack s

    // Parallel stubs — alias to sequential
    | ParTryMeElse label     -> step ctx s (TryMeElse label)
    | ParRetryMeElse label   -> step ctx s (RetryMeElse label)
    | ParRetryMeElsePc pc    -> step ctx s (RetryMeElsePc pc)
    | ParTrustMe             -> step ctx s TrustMe

    // Structure/list building — handled by builder state
    | PutStructure (fn, ai, arity) ->
        Some { s1 with WsBuilder = Some (BuildStruct (fn, ai, arity, [])) }

    | PutList ai ->
        Some { s1 with WsBuilder = Some (BuildList (ai, [])) }

    | SetValue xn ->
        let v = Map.tryFind xn s.WsRegs |> Option.defaultValue (Unbound 0)
        addToBuilder v s

    | SetVariable xn ->
        let vid = s.WsVarCounter
        let v   = Unbound vid
        let s2  = { s1 with WsRegs = Map.add xn v s.WsRegs; WsVarCounter = vid + 1 }
        addToBuilder v s2

    | SetConstant c ->
        addToBuilder c s

    | GetStructure (fn, arity, ai) ->
        let v = Map.tryFind ai s.WsRegs |> Option.map (derefVar s.WsBindings)
        match v with
        | Some (Str (fn2, args)) when fn = fn2 && List.length args = arity ->
            Some s1  // start reading args via UnifyVariable/UnifyValue
        | _ -> backtrack s

    | GetList ai ->
        let v = Map.tryFind ai s.WsRegs |> Option.map (derefVar s.WsBindings)
        match v with
        | Some (VList _) -> Some s1
        | _ -> backtrack s

    | UnifyVariable xn ->
        Some { s1 with WsRegs = Map.add xn (Unbound s.WsVarCounter) s.WsRegs
                       WsVarCounter = s.WsVarCounter + 1 }

    | UnifyValue xn ->
        // simplified: real unification done by generator's full step
        Some s1

    | UnifyConstant c ->
        Some s1  // real check done in generator

// ----------------------------------------------------------------------------
// dispatchCall — look up predicate and call it
// ----------------------------------------------------------------------------

and dispatchCall (ctx: WamContext) (pred: string) (s: WamState) : WamState option =
    match Map.tryFind pred ctx.WcLoweredPredicates with
    | Some fn -> fn ctx s
    | None ->
    match callIndexedFact2 ctx pred s with
    | Some sr -> Some sr
    | None ->
    match Map.tryFind pred ctx.WcLabels with
    | Some pc -> Some { s with WsPC = pc }
    | None    -> backtrack s

/// Fast indexed fact lookup (first-argument indexing on foreign fact tables).
and callIndexedFact2 (ctx: WamContext) (pred: string) (s: WamState) : WamState option =
    match Map.tryFind pred ctx.WcForeignFacts with
    | None -> None
    | Some tbl ->
        let keyOpt =
            match Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) with
            | Some (Atom a)    -> Some a
            | Some (Integer n) -> Some (string n)
            | _                -> None
        match keyOpt with
        | None -> None
        | Some k ->
            match Map.tryFind k tbl with
            | None -> None
            | Some vals ->
                let retPC = s.WsCP
                match vals with
                | [] -> None
                | [v] ->
                    let vid = s.WsVarCounter
                    let newRegs = Map.add 2 (Atom v) s.WsRegs
                    let newBindings = Map.add vid (Atom v) s.WsBindings
                    Some { s with
                             WsPC       = retPC
                             WsRegs     = newRegs
                             WsBindings = newBindings
                             WsVarCounter = vid + 1 }
                | v :: rest ->
                    let vid = s.WsVarCounter
                    let newRegs = Map.add 2 (Atom v) s.WsRegs
                    let newBindings = Map.add vid (Atom v) s.WsBindings
                    let cp = { CpNextPC   = retPC
                               CpRegs     = s.WsRegs
                               CpStack    = s.WsStack
                               CpCP       = s.WsCP
                               CpTrailLen = s.WsTrailLen
                               CpHeapLen  = s.WsHeapLen
                               CpBindings = s.WsBindings
                               CpCutBar   = s.WsCutBar
                               CpAggFrame = None
                               CpBuiltin  = Some (FactRetry (vid, rest, retPC)) }
                    Some { s with
                             WsPC       = retPC
                             WsRegs     = newRegs
                             WsBindings = newBindings
                             WsVarCounter = vid + 1
                             WsCPs      = cp :: s.WsCPs
                             WsCPsLen   = s.WsCPsLen + 1 }

/// Foreign predicate dispatch — auto-generated from kernel detection.
/// This stub always fails; the generator emits the real version.
and executeForeign (_ctx: WamContext) (_pred: string) (_s: WamState) : WamState option = None

/// Foreign call entry point used from lowered functions.
and callForeign (ctx: WamContext) (pred: string) (sc: WamState) : WamState option =
    executeForeign ctx pred sc

// ----------------------------------------------------------------------------
// resolveCallInstrs — pre-resolve Call labels to PCs at load time (Phase C)
// ----------------------------------------------------------------------------

/// Convert Call/Jump/TryMeElse labels to direct PC variants.
/// Call this once after loading the instruction array.
let resolveCallInstrs (labels: Map<string, int>) (foreignPreds: string list) (instrs: Instruction list) : Instruction list =
    instrs |> List.map (fun instr ->
        match instr with
        | Call (pred, arity) ->
            if List.contains pred foreignPreds then CallForeign (pred, arity)
            else
                match Map.tryFind pred labels with
                | Some pc -> CallResolved (pc, arity)
                | None    -> instr
        | Execute pred ->
            match Map.tryFind pred labels with
            | Some pc -> ExecutePc pc
            | None    -> instr
        | Jump label ->
            match Map.tryFind label labels with
            | Some pc -> JumpPc pc
            | None    -> instr
        | TryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> TryMeElsePc pc
            | None    -> instr
        | RetryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> RetryMeElsePc pc
            | None    -> instr
        | SwitchOnConstant table ->
            let pcTable =
                table |> Map.toList
                      |> List.choose (fun (v, label) ->
                            Map.tryFind label labels
                            |> Option.map (fun pc ->
                                let k = match v with
                                        | Atom s -> s
                                        | Integer n -> string n
                                        | other -> sprintf "%A" other
                                k, pc))
                      |> Map.ofList
            SwitchOnConstantPc pcTable
        | i -> i)

// ----------------------------------------------------------------------------
// run — main interpreter loop
// ----------------------------------------------------------------------------

/// Run the WAM interpreter from the current state until halt (WsPC = 0) or failure.
let rec run (ctx: WamContext) (s: WamState) : WamState option =
    if s.WsPC = 0 then Some s
    else
        let pc = s.WsPC
        if pc < 1 || pc > ctx.WcCode.Length then backtrack s
        else
            let instr = ctx.WcCode.[pc - 1]  // 1-indexed
            match step ctx s instr with
            | Some s1 -> run ctx s1
            | None    -> None
