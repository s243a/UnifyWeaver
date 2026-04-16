:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% fsharp_wam_bindings.pl - WAM-specific F# bindings for UnifyWeaver
%
% Provides the register-level and state-management bindings used by the
% WAM-to-F# transpilation target. The F# design mirrors the Haskell WAM
% target closely:
%
%   - `Value` discriminated union (DU) replaces Haskell's `data Value`
%   - `WamState` record with Map.t fields replaces Haskell's WamState
%   - `WamContext` (read-only) carries code array, labels, FFI facts
%   - Immutable F# `Map<'K,'V>` (FSharp.Collections.Map) gives O(1)
%     structural-sharing snapshots for choice points — the same key
%     insight that brought the Haskell target from ~4700ms to 75ms.
%     See docs/design/WAM_PERF_OPTIMIZATION_LOG.md for the full timeline.
%
% Key differences from the Haskell target:
%   - F# uses `Map<int,Value>` instead of `Data.IntMap` for registers
%     (F# stdlib has no IntMap; Map<int,_> is the idiomatic equivalent)
%   - Choice points use `{ ... with field = v }` record update syntax
%   - Pattern matching is `match x with | Pat -> ...` not `case x of`
%   - Immutability is the default — no need for Bang patterns / UNPACK
%   - `option<'T>` replaces `Maybe a`; `Some`/`None` replace `Just`/`Nothing`
%   - The run loop uses tail-recursive style without GHC-specific pragmas
%
% Performance notes (inherited from Haskell/Rust opt log):
%   1. Pre-resolve Call labels to direct PC ints at load time
%   2. Cache trail/heap lengths as counters in WamState (don't use List.length)
%   3. Split WamState (hot, per-step) from WamContext (cold, read-only)
%   4. Skip WAM-compilation of FFI-owned fact predicates entirely
%   5. Intern atoms as ints at the FFI boundary (eliminates Map<string,_>
%      hashing inside the kernel's recursive hot loop)
%
% See: docs/design/WAM_PERF_OPTIMIZATION_LOG.md
%      docs/vision/HASKELL_TARGET_ROADMAP.md   (closest analog)

:- module(fsharp_wam_bindings, [
    init_fsharp_wam_bindings/0,
    fsharp_wam_value_type/0,          % Emit the Value DU
    fsharp_wam_state_type/0,          % Emit WamState record
    fsharp_wam_context_type/0,        % Emit WamContext record
    fsharp_wam_choicepoint_type/0,    % Emit ChoicePoint record
    fsharp_wam_helpers/0,             % Emit derefVar, getReg, putReg, evalArith
    fsharp_wam_type_header/1,         % -Code: full type + helper preamble
    fsharp_wam_reg_default/2,         % +Type, -FSharpExpr
    fsharp_wam_result_wrap/2,         % +Type, -FSharpExpr  (wraps `rv`)
    fsharp_wam_result_wrap_rv/3       % +Type, +Index, -FSharpExpr (wraps `rv_I`)
]).

:- use_module('../core/binding_registry').

%% init_fsharp_wam_bindings
init_fsharp_wam_bindings.

% ============================================================================
% Value discriminated union — mirrors Haskell `data Value`
% ============================================================================
%
% F# DUs are structurally equivalent to Haskell ADTs.
% Map<int,Value> replaces Data.IntMap for registers/bindings.
% Key insight: Map<int,Value> is immutable and persistent — saving a choice
% point snapshot is O(1) (pointer copy of the root), just like Haskell's
% Data.Map.  Backtracking = swap the reference back.

fsharp_wam_value_type :-
    writeln(
"[<CustomEquality; CustomComparison>]
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
    override this.GetHashCode() = hash this").

% ============================================================================
% TrailEntry — mirrors Haskell `data TrailEntry`
% ============================================================================

fsharp_wam_trail_entry_type :-
    writeln(
"type TrailEntry = { TrailVarId: int; TrailOldVal: Value option }").

% ============================================================================
% AggregateFrame — mirrors Haskell `AggFrame`
% ============================================================================

fsharp_wam_agg_frame_type :-
    writeln(
"type AggFrame = { AggType: string; AggValReg: int; AggResReg: int; AggReturnPC: int }").

% ============================================================================
% BuiltinState — mirrors Haskell `BuiltinState`
% ============================================================================

fsharp_wam_builtin_state_type :-
    writeln(
"type BuiltinState =
    | FactRetry      of varId: int * remaining: string list * retPC: int
    | HopsRetry      of varId: int * remaining: int list   * retPC: int
    | FFIStreamRetry of outRegs: int list * outVars: int list * remaining: Value list list * retPC: int").

% ============================================================================
% EnvFrame — mirrors Haskell `EnvFrame`
% ============================================================================

fsharp_wam_env_frame_type :-
    writeln(
"type EnvFrame = { EfSavedCP: int; EfYRegs: Map<int, Value> }").

% ============================================================================
% ChoicePoint — mirrors Haskell `ChoicePoint`
%
% Key: cpBindings is a Map<int,Value> snapshot — O(1) to save/restore
% because F# maps are persistent immutable trees (structural sharing).
% ============================================================================

fsharp_wam_choicepoint_type :-
    writeln(
"type ChoicePoint =
    { CpNextPC   : int
      CpRegs     : Map<int, Value>
      CpStack    : EnvFrame list
      CpCP       : int
      CpTrailLen : int
      CpHeapLen  : int
      CpBindings : Map<int, Value>   // O(1) snapshot — immutable tree
      CpCutBar   : int
      CpAggFrame : AggFrame option
      CpBuiltin  : BuiltinState option }").

% ============================================================================
% WamState — hot, per-step mutable state (F# record, updated with `{ s with ... }`)
%
% Performance notes:
%   - wsTrailLen / wsHeapLen / wsCPsLen are explicit counters; never call
%     List.length in the hot loop (lists don't cache length)
%   - wsVarCounter is bumped on PutVariable to avoid rehashing
%   - wsAggAccum accumulates values for aggregate/3
% ============================================================================

fsharp_wam_state_type :-
    writeln(
"type WamState =
    { WsPC        : int
      WsRegs      : Map<int, Value>      // A/X registers (Int-keyed)
      WsStack     : EnvFrame list        // environment frames
      WsHeap      : Value list           // term construction
      WsHeapLen   : int                  // cached List.length WsHeap
      WsTrail     : TrailEntry list      // binding history
      WsTrailLen  : int                  // cached List.length WsTrail
      WsCP        : int                  // continuation pointer
      WsCPs       : ChoicePoint list     // choice point stack
      WsCPsLen    : int                  // cached List.length WsCPs
      WsBindings  : Map<int, Value>      // variable bindings (O(1) snapshot)
      WsCutBar    : int                  // cut barrier depth
      WsVarCounter: int                  // fresh variable id counter
      WsBuilder   : BuilderState option  // PutStructure / PutList accumulator
      WsAggAccum  : Value list           // aggregate accumulator
    }

and BuilderState =
    | BuildStruct of fn: string * reg: int * arity: int * args: Value list
    | BuildList   of reg: int * items: Value list").

% ============================================================================
% WamContext — cold, read-only context threaded through the run loop
%
% Split from WamState so that the hot-path record updates don't copy the
% large code array or label map on every step — same lesson as the Haskell
% Phase-B hot/cold split (commit ab829695 in the perf log).
% ============================================================================

fsharp_wam_context_type :-
    writeln(
"type WamContext =
    { WcCode            : Instruction array    // instruction array (O(1) fetch)
      WcLabels          : Map<string, int>     // label → PC
      WcForeignFacts    : Map<string, Map<string, string list>>
                                               // pred → key → [value, ...]
      WcFfiFacts        : Map<string, Map<int, int list>>
                                               // interned facts for FFI kernels
      WcFfiWeightedFacts: Map<string, Map<int, (int * float) list>>
                                               // weighted FFI facts (Dijkstra etc.)
      WcAtomIntern      : Map<string, int>     // atom string → int id
      WcAtomDeintern    : Map<int, string>     // int id → atom string
      WcForeignConfig   : Map<string, int>     // config_int values
      WcLoweredPredicates: Map<string, WamContext -> WamState -> WamState option>
                                               // predicate name → lowered fn
    }").

% ============================================================================
% Instruction DU — mirrors Haskell `data Instruction`
% ============================================================================

fsharp_wam_instruction_type :-
    writeln(
"type Instruction =
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
    // Parallel variants (Phase 4.1 stubs — alias to sequential at runtime)
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
    | EndAggregate   of valReg: int").

% ============================================================================
% Helper functions — derefVar, getReg, putReg, addToBuilder, evalArith
% ============================================================================

fsharp_wam_helpers :-
    writeln(
"// ============================================================================
// Helper functions
// ============================================================================

/// Dereference a value through the binding chain.
let rec derefVar (bindings: Map<int, Value>) (v: Value) : Value =
    match v with
    | Unbound vid ->
        match Map.tryFind vid bindings with
        | Some bound -> derefVar bindings bound
        | None       -> v
    | _ -> v

/// Look up a register (A/X register), dereference through bindings.
let getReg (n: int) (s: WamState) : Value option =
    Map.tryFind n s.WsRegs |> Option.map (derefVar s.WsBindings)

/// Set a register value.
let putReg (n: int) (v: Value) (s: WamState) : WamState =
    { s with WsRegs = Map.add n v s.WsRegs }

/// Dereference a heap pointer.
let derefHeap (heap: Value list) (v: Value) : Value option =
    match v with
    | Ref i ->
        if i < List.length heap then Some (List.item i heap)
        else None
    | _ -> Some v

/// Bind an output register without advancing PC.
/// Succeeds if register is unbound or already equals val.
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

/// Append a value to the current builder (PutStructure / PutList).
let addToBuilder (value: Value) (s: WamState) : WamState option =
    match s.WsBuilder with
    | None -> None
    | Some (BuildStruct (fn, reg, arity, args)) ->
        let args' = args @ [value]
        if List.length args' = arity then
            let str = Str (fn, args')
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = Map.add reg str s.WsRegs
                     WsBuilder = None }
        else
            Some { s with WsBuilder = Some (BuildStruct (fn, reg, arity, args')) }
    | Some (BuildList (reg, items)) ->
        Some { s with WsBuilder = Some (BuildList (reg, items @ [value])) }

/// Arithmetic evaluator over Value terms.
let rec evalArith (bindings: Map<int, Value>) (v: Value) : float option =
    let d = derefVar bindings v
    match d with
    | Integer n -> Some (float n)
    | Float   f -> Some f
    | Str (\"+\", [a; b]) ->
        Option.map2 (+) (evalArith bindings a) (evalArith bindings b)
    | Str (\"-\", [a; b]) ->
        Option.map2 (-) (evalArith bindings a) (evalArith bindings b)
    | Str (\"*\", [a; b]) ->
        Option.map2 (*) (evalArith bindings a) (evalArith bindings b)
    | Str (\"/\", [a; b]) ->
        match evalArith bindings a, evalArith bindings b with
        | Some x, Some y when y <> 0.0 -> Some (x / y)
        | _ -> None
    | Str (\"mod\", [a; b]) ->
        match evalArith bindings a, evalArith bindings b with
        | Some x, Some y when y <> 0.0 -> Some (float (int x % int y))
        | _ -> None
    | _ -> None

/// Undo a single trail entry during backtrack.
let undoBinding (bindings: Map<int, Value>) (e: TrailEntry) : Map<int, Value> =
    match e.TrailOldVal with
    | Some old -> Map.add e.TrailVarId old bindings
    | None     -> Map.remove e.TrailVarId bindings

/// Walk a Value tree, copying variables to fresh ids (copy_term/2).
let rec copyTermWalk (c: int) (m: Map<int, int>) (v: Value)
        : Value * int * Map<int, int> =
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

and copyTermArgs (c: int) (m: Map<int, int>) (xs: Value list)
        : Value list * int * Map<int, int> =
    match xs with
    | [] -> [], c, m
    | x :: rest ->
        let x1, c1, m1 = copyTermWalk c m x
        let rest1, c2, m2 = copyTermArgs c1 m1 rest
        x1 :: rest1, c2, m2").

% ============================================================================
% Combined preamble emitter
% ============================================================================

fsharp_wam_type_header(Code) :-
    with_output_to(string(Code), (
        writeln("module WamTypes"),
        nl,
        writeln("open System.Collections.Generic"),
        nl,
        fsharp_wam_value_type,         nl,
        fsharp_wam_trail_entry_type,   nl,
        fsharp_wam_agg_frame_type,     nl,
        fsharp_wam_builtin_state_type, nl,
        fsharp_wam_env_frame_type,     nl,
        fsharp_wam_choicepoint_type,   nl,
        fsharp_wam_state_type,         nl,
        fsharp_wam_context_type,       nl,
        fsharp_wam_instruction_type,   nl,
        fsharp_wam_helpers
    )).

% ============================================================================
% Type-driven codegen helpers (used by wam_fsharp_target.pl)
% ============================================================================

%% fsharp_wam_reg_default(+Type, -FSharpExpr)
%  Default Value constructor when a register lookup returns None.
fsharp_wam_reg_default(atom,        'Atom ""').
fsharp_wam_reg_default(integer,     'Integer 0').
fsharp_wam_reg_default(vlist_atoms, 'VList []').

%% fsharp_wam_result_wrap(+Type, -FSharpExpr)
%  Wrap kernel result `rv` into a Value. For atoms, the kernel returns
%  an interned int id — de-intern via WcAtomDeintern before wrapping.
fsharp_wam_result_wrap(integer, 'Integer rv').
fsharp_wam_result_wrap(atom,    'Atom (Map.tryFind rv ctx.WcAtomDeintern |> Option.defaultValue "")').
fsharp_wam_result_wrap(float,   'Float rv').

%% fsharp_wam_result_wrap_rv(+Type, +Index, -FSharpExpr)
%  Like fsharp_wam_result_wrap but for rv_<Index> tuple components.
fsharp_wam_result_wrap_rv(integer, I, Expr) :-
    format(atom(Expr), 'Integer rv_~w', [I]).
fsharp_wam_result_wrap_rv(atom, I, Expr) :-
    format(atom(Expr), 'Atom (Map.tryFind rv_~w ctx.WcAtomDeintern |> Option.defaultValue "")', [I]).
fsharp_wam_result_wrap_rv(float, I, Expr) :-
    format(atom(Expr), 'Float rv_~w', [I]).
