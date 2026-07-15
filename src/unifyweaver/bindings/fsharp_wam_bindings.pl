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
%   - F# uses `Value array` (fixed-size 600) for registers — O(1) access
%     with Array.copy snapshots for choice points (matches Go's [512]Value)
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
"/// Discriminated union for runtime Prolog values.
///
/// F# auto-generates structural equality and structural comparison for
/// DUs, so we deliberately do NOT use [<CustomEquality; CustomComparison>]
/// here.  An earlier version did, with `Equals(obj) = match ... -> this = other`
/// and `IComparable.CompareTo = compare this other` overrides — both
/// recurse through the very operator they''re trying to define, which
/// would stack-overflow if ever exercised on the hot path.  The runtime
/// hot path matches Value variants pattern-by-pattern anyway, so the
/// override was unreachable; removing it is a pure cleanup.
///
/// For Prolog-standard term order (Var < Number < Atom < Compound) — as
/// required by compare/3, @</2 and friends, sort/2, msort/2 — call the
/// runtime helper compareValue (defined below) directly.  F#''s default
/// structural comparison is NOT Prolog-standard order: it follows the
/// DU constructor declaration order (Atom < Integer < Float < VList < ...).
type Value =
    | Atom    of string
    | Integer of int
    | Float   of float
    | VList   of Value list
    | Str     of string * Value list   // functor name, args
    | Unbound of int                   // variable id
    | Ref     of int                   // heap reference
    | VSet    of Set<string>           // visited-set: Set of atom strings
                                        // (F# atoms are string-based; Haskell
                                        //  uses IS.IntSet of interned int IDs)").

% ============================================================================
% TrailEntry — mirrors Haskell `data TrailEntry`
% ============================================================================

fsharp_wam_trail_entry_type :-
    writeln(
"type TrailEntry = { TrailVarId: int; TrailOldVal: Value option }").

% ============================================================================
% MergeStrategy + AggregateFrame — mirrors Haskell `MergeStrategy` + `AggFrame`
% ============================================================================
%
% AggMergeStrategy classifies an aggregate's combine semantics so the WAM
% runtime can decide whether to evaluate branches in parallel (forkable
% strategies: sum/count/bag/set/findall) or fall back to sequential
% choice-point backtracking (everything else).  Computed at BeginAggregate
% time from the aggType string via inferMergeStrategy (defined in
% WamRuntime).

fsharp_wam_agg_frame_type :-
    writeln(
"type MergeStrategy =
    | MergeSum
    | MergeCount
    | MergeBag
    | MergeSet
    | MergeFindall
    | MergeSequential   // not forkable: fall back to sequential backtrack

type AggFrame =
    { AggType:           string
      AggValReg:         int
      AggResReg:         int
      AggReturnPC:       int
      AggMergeStrategy:  MergeStrategy }").

% ============================================================================
% BuiltinState — mirrors Haskell `BuiltinState`
% ============================================================================

fsharp_wam_builtin_state_type :-
    writeln(
"type BuiltinState =
    | FactRetry      of varId: int * remaining: string list * retPC: int
    | HopsRetry      of varId: int * remaining: int list   * retPC: int
    | FFIStreamRetry of outRegs: int list * outVars: int list * remaining: Value list list * retPC: int
    /// select/3 enumeration choice point.  Each remaining candidate is
    /// a pair (selected, rest_items): on backtrack the runtime restores
    /// the CP snapshot, unifies elemReg with `selected`, and binds
    /// outReg to VList rest_items.  Mirrors the Go target's
    /// SelectResults field (templates/targets/go_wam/state.go.mustache).
    | SelectRetry    of elemReg: int * outReg: int * remaining: (Value * Value list) list * retPC: int
    /// member/2 backtracking choice point.  `remaining` is the unflattened
    /// list-tail still to be tried after the current success; on backtrack
    /// the runtime restores the CP snapshot and walks `remaining` looking
    /// for the next unifiable element.  Needed because the parser uses
    /// `member(op(Name, P, T), OpTable), is_op_type(T), !` and depends on
    /// backtracking into member when the type guard fails.
    | MemberRetry    of elemReg: int * remaining: Value list * retPC: int
    /// T9 fact-table enumeration choice point.  `args` are the query argument
    /// values (deref''d at call time); `remaining` are the candidate rows still
    /// to try, each a `VList` of column values.  On backtrack the runtime
    /// restores the CP snapshot and unifies every column of the next matching
    /// row against `args`, leaving a fresh FactTableRetry CP if more remain.
    | FactTableRetry of args: Value list * remaining: Value list * retPC: int").

% ============================================================================
% EnvFrame — mirrors Haskell `EnvFrame`
% ============================================================================

fsharp_wam_env_frame_type :-
    writeln(
"type EnvFrame = { EfSavedCP: int; EfYRegs: Map<int, Value>; EfSavedCutBar: int }

/// Exception carrier for Prolog throw/1.  The thrown term is
/// deep-dereferenced before being raised so the catcher sees the
/// value the thrower intended, not bindings that may have changed
/// during unwind.  Uncaught throws propagate to the top-level entry
/// point (dispatchCall / runPredicate).
exception WamException of Value").

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
      CpRegs     : Value array        // O(1) snapshot via Array.copy
      CpStack    : EnvFrame list
      CpCP       : int
      CpTrailLen : int
      CpHeapLen  : int
      CpBindings : Map<int, Value>   // O(1) snapshot — immutable tree
      CpCutBar   : int
      // B0 stack depth at CP creation time (issue #2400 fu).
      // Call/CallResolved push onto WsB0Stack; Proceed pops.  But
      // backtrack into this CP must also discard any pushes that
      // happened after the CP was created (the failed sub-calls'
      // pushes never reached their Proceed).  CpB0StackLen captures
      // the depth at TryMeElse time so backtrack can truncate
      // WsB0Stack back to it — without that the stack grows
      // unboundedly and subsequent Proceed pops the wrong entry.
      CpB0StackLen : int
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
"/// Register array size — matches Go's [512]Value + padding for X/Y regs.
[<Literal>]
let MaxRegs = 600

type WamState =
    { WsPC        : int
      WsRegs      : Value array          // A/X/Y registers — O(1) array access
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
      // Stack of outer build / read contexts saved when a GetStructure
      // or GetList nests into the active context (e.g. `[op(:-,1200,xfx)|T]`
      // needs the outer BuildList active while the inner BuildStruct
      // fills, then restores on inner materialization).  Both R
      // (templates/targets/r_wam/runtime.R.mustache build_stack) and
      // Python (src/unifyweaver/targets/wam_python_runtime/WamRuntime.py
      // write_stack/read_stack) maintain the same stack discipline.
      WsBuilderStack : BuilderState list
      WsAggAccum  : Value list           // aggregate accumulator
      // Cut-barrier (B0) stack — pushed by Call/CallResolved before
      // jumping into a callee and popped by Proceed on return.
      // Execute/ExecutePc (tail call) updates WsCutBar in place but
      // does NOT push or pop — the caller's frame is gone, so the
      // stack depth must remain matched with Call/Proceed pairs
      // only.  Standard-WAM B0 protocol: each Call saves the
      // caller's cut barrier here and sets WsCutBar = WsCPsLen
      // (the count BEFORE the callee's leading TryMeElse pushes
      // CP_self).  Without this, a `:- ..., !.` in the callee
      // never drops the predicate's own retry CP because the
      // barrier was set AFTER TryMeElse already incremented
      // WsCPsLen — see issue #2400 follow-up.
      WsB0Stack   : int list
      // ISO catcher stack for catch/3.  Each catch/3 pushes a frame
      // carrying the catcher pattern, recovery goal, and a snapshot
      // of state at catch-entry time.  throw/1 raises a WamException
      // which the nearest catch/3 try/with catches; the catcher
      // restores its snapshot, unifies catcher with thrown term, and
      // runs recovery.  Mirrors Python wam_runtime.catcher_frames.
      WsCatchers  : CatcherFrame list
    }

/// ISO catcher frame for catch/3.  CfSnapshot is the live state at
/// catch-entry time; on WamException unwind the catcher restores from
/// it.  CfSnapshotRegs is a defensive Array.copy of the snapshot''s
/// WsRegs so that putReg''s in-place mutation between catch and throw
/// can''t corrupt the saved frame (same discipline as ChoicePoint.CpRegs).
and CatcherFrame =
    { CfCatcherTerm  : Value
      CfRecoveryTerm : Value
      CfSnapshot     : WamState
      CfSnapshotRegs : Value array }

and BuilderState =
    | BuildStruct of fn: string * reg: int * arity: int * args: Value list
    | BuildList   of reg: int * items: Value list
    | ReadArgs    of items: Value list").

% ============================================================================
% WamContext — cold, read-only context threaded through the run loop
%
% Split from WamState so that the hot-path record updates don't copy the
% large code array or label map on every step — same lesson as the Haskell
% Phase-B hot/cold split (commit ab829695 in the perf log).
% ============================================================================

fsharp_wam_context_type :-
    writeln(
"/// Abstraction over fact-source access patterns.  Implementations:
///   - EagerLookupSource: wraps a pre-loaded Map<int, int list> (Phase 1 eager).
///   - LmdbCursorLookup: reads LMDB on demand per key (Phase 2 lazy).
/// The WAM kernel dispatch checks WcLookupSources before falling back
/// to the legacy WcFfiFacts path.  This lets lazy and cached modes
/// plug in without changing caller code.
type ILookupSource =
    abstract member Lookup : int -> int list

/// Eager implementation: wraps a fully-materialised Map built at startup.
type EagerLookupSource(data: Map<int, int list>) =
    member _.Data = data
    interface ILookupSource with
        member _.Lookup(key) =
            Map.tryFind key data |> Option.defaultValue []

/// Dictionary-backed eager lookup: O(1) amortized access without the
/// cost of building an immutable Map. Used when the caller only needs
/// ILookupSource (not a Map for kernel dispatch).
type DictLookupSource(data: System.Collections.Generic.Dictionary<int, int list>) =
    member _.Dict = data
    interface ILookupSource with
        member _.Lookup(key) =
            let ok, vs = data.TryGetValue(key)
            if ok then vs else []


type WamContext =
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
      WcLookupSources   : Map<string, ILookupSource>
                                               // pred → polymorphic fact source
                                               // (eager Map, lazy LMDB cursor, or
                                               // cached decorator). Checked BEFORE
                                               // WcFfiFacts by kernel dispatch.
      WcCancellationToken: System.Threading.CancellationToken option
                                               // optional hard-cancel signal.
                                               // `run` checks this each loop
                                               // iteration and returns None when
                                               // cancellation is requested.  Wired
                                               // by runNegationParallel via
                                               // Async.CancellationToken so the
                                               // first successful branch can
                                               // halt sibling branches'' work
                                               // (CPU/thread savings beyond the
                                               // wall-time win from Async.Choice).
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
    | Fail
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
    // Indexed-dispatch chain ops.  Standard-WAM `try` / `retry` /
    // `trust` (no `_me_else`).  Emitted into the synthesized
    // try/retry/trust chains that switch_on_term / switch_on_constant /
    // switch_on_structure target when a dispatch group has >1
    // matching clause (issue #2400).  Unlike the linear chain''s
    // _me_else variants, these CARRY the target body label and
    // store the in-chain fall-through PC in the CP.
    | Try            of label: string
    | TryPc          of targetPC: int
    | Retry          of label: string
    | RetryPc        of targetPC: int
    | Trust          of label: string
    | TrustPc        of targetPC: int
    // Parallel variants (Phase 4.1 stubs — alias to sequential at runtime)
    | ParTryMeElse     of label: string
    | ParTryMeElsePc   of nextPC: int
    | ParRetryMeElsePc of nextPC: int
    | ParRetryMeElse   of label: string
    | ParTrustMe
    // Indexing
    | SwitchOnConstant   of table: Map<Value, string>
    | SwitchOnConstantPc of table: (string * int) array  // sorted by key, binary search
    // Type-based indexing: dispatch on A1''s type.  Atomic A1 hits
    // constTable; Str (F, args) builds the F-slash-N key and hits
    // structTable; list-shaped A1 jumps to listLabel/listPc.  Every
    // miss falls through to PC+1 (the linear try_me_else chain).
    // Label form is emitted by the WAM text translator; resolveCallInstrs
    // rewrites to the PC form at load time.
    | SwitchOnTerm   of constTable: (string * string) array
                       * structTable: (string * string) array
                       * listLabel: string
    | SwitchOnTermPc of constTable: (string * int) array
                       * structTable: (string * int) array
                       * listPc: int
    // Builtins
    | BuiltinCall    of name: string * arity: int
    | CutIte
    // Aggregation
    | BeginAggregate of aggType: string * valReg: int * resReg: int
    | EndAggregate   of valReg: int
    // ----------------------------------------------------------------------
    // Phase I — Haskell-only specialized instructions ported to F#.
    // These are performance optimizations emitted by the WAM compiler's
    // analysis passes; their semantics live in step (see wam_fsharp_target.pl).
    //   PutStructureDyn: runtime-parsed functor (name and arity from registers).
    //   Arg: specialized arg/3 with literal N.
    //   NotMemberList: specialized \\+ member(X, L) with L a VList.
    //   NotMemberConstAtoms: specialized \\+ member(X, [a, b, c]) with the
    //     atoms baked into the instruction at compile time.
    //   BuildEmptySet / SetInsert / NotMemberSet: VSet visited-set support
    //     (uses Set<string>; Haskell uses IS.IntSet of interned atom IDs).
    | PutStructureDyn       of nameReg: int * arityReg: int * targetReg: int
    | Arg                   of n: int * tReg: int * aReg: int
    | NotMemberList         of xReg: int * lReg: int
    | NotMemberConstAtoms   of xReg: int * atoms: string list
    | BuildEmptySet         of reg: int
    | SetInsert             of elemReg: int * inReg: int * outReg: int
    | NotMemberSet          of elemReg: int * setReg: int").

% ============================================================================
% Helper functions — derefVar, getReg, putReg, addToBuilder, evalArith
% ============================================================================

fsharp_wam_helpers :-
    writeln(
"// ============================================================================
// Helper functions
// ============================================================================

/// T9 first-argument index key: a canonical string for an atomic value, or
/// None for compound / list / unbound values (which can only match via a full
/// scan).  Keys both the stored rows and the query arg, so a bound atomic first
/// argument selects exactly the matching index bucket.
let factIndexKey (v: Value) : string option =
    match v with
    | Atom s    -> Some s
    | Integer n -> Some (\"i\" + string n)
    | Float f   -> Some (\"f\" + string f)
    | _         -> None

/// Resolve a fact lookup function for kernel dispatch.  Returns
/// int -> int list, which kernels call directly instead of
/// Map.tryFind.  This avoids materialising the entire relation
/// into a Map — critical for lazy/cached modes at large scale
/// (enwiki 10M edges: Map.add would take ~140s).
///
/// Dispatch order:
///   1. WcLookupSources (ILookupSource.Lookup — works for eager,
///      lazy cursor, cached two-level, dict)
///   2. WcFfiFacts (legacy Map<int, int list> path)
///   3. empty (returns [] for any key)
let resolveFactLookup (pred: string) (ctx: WamContext) : (int -> int list) =
    match Map.tryFind pred ctx.WcLookupSources with
    | Some src -> src.Lookup
    | None ->
        match Map.tryFind pred ctx.WcFfiFacts with
        | Some factMap -> fun key -> Map.tryFind key factMap |> Option.defaultValue []
        | None -> fun _ -> []

/// Legacy: resolve a fact Map for callers that still need Map<int, int list>.
/// Prefer resolveFactLookup for new code.
let resolveFactMap (pred: string) (ctx: WamContext) : Map<int, int list> =
    match Map.tryFind pred ctx.WcLookupSources with
    | Some (:? EagerLookupSource as eager) -> eager.Data
    | Some (:? DictLookupSource as dict) ->
        dict.Dict |> Seq.fold (fun acc kv ->
            Map.add kv.Key kv.Value acc) Map.empty
    | Some _ ->
        Map.tryFind pred ctx.WcFfiFacts |> Option.defaultValue Map.empty
    | None ->
        Map.tryFind pred ctx.WcFfiFacts |> Option.defaultValue Map.empty

/// Merge two maps (right-biased: values from m2 overwrite m1).
let mapUnion (m1: Map<'k, 'v>) (m2: Map<'k, 'v>) : Map<'k, 'v> =
    Map.fold (fun acc k v -> Map.add k v acc) m1 m2

/// Dereference a value through the binding chain.
let rec derefVar (bindings: Map<int, Value>) (v: Value) : Value =
    match v with
    | Unbound vid ->
        match Map.tryFind vid bindings with
        | Some bound -> derefVar bindings bound
        | None       -> v
    | _ -> v

/// Walk a list-shaped value into a flat F# list of derefd elements.
/// Lists may be stored as VList (proper, ground-tail) OR as a chain of
/// Str(\"[|]\", [h; t]) cons cells (when materialized with a non-ground
/// tail in addToBuilder).  Builtins that consume lists -- atom_codes/2,
/// length/2, member/2, etc. -- only matched VList before this helper,
/// so anything built via the cons-cell shape (everything coming out of
/// take_digits / take_ident / parser accumulators) was opaque.  Returns
/// None for improper / partial lists.
let rec valueToProperList (bindings: Map<int, Value>) (v: Value) : Value list option =
    match derefVar bindings v with
    | Atom \"[]\"                  -> Some []
    | VList items                ->
        // Items may themselves contain Unbound vars or cons-cell tails;
        // surface them as-is for the caller to deref.
        Some items
    | Str (\"[|]\", [h; t])        ->
        match valueToProperList bindings t with
        | Some rest -> Some (h :: rest)
        | None      -> None
    | _ -> None

/// Look up a register (A/X register), dereference through bindings.
let getReg (n: int) (s: WamState) : Value option =
    // Y registers (n >= 201, encoded by fs_reg_name_to_int as Yk -> 200+k)
    // are mirrored in the current env frame -- prefer the frame copy
    // since the WsRegs entry may have been clobbered by a called
    // predicate using the same numeric index for its own temporaries.
    // R (env$perm_vars) and Python (_Y_BASE = 301) use the same
    // discipline.
    if n >= 201 then
        match s.WsStack with
        | frame :: _ when Map.containsKey (n - 200) frame.EfYRegs ->
            Some (derefVar s.WsBindings (Map.find (n - 200) frame.EfYRegs))
        | _ ->
            // No frame entry yet -- fall through to WsRegs.
            if n < s.WsRegs.Length then
                match s.WsRegs.[n] with
                | Unbound -1 -> None
                | v          -> Some (derefVar s.WsBindings v)
            else None
    elif n >= 0 && n < s.WsRegs.Length then
        match s.WsRegs.[n] with
        | Unbound -1 -> None   // sentinel = uninitialized
        | v -> Some (derefVar s.WsBindings v)
    else None

/// Set a register value.  Mutates WsRegs in place for X-regs and the WsRegs
/// mirror of Y-regs; the env-frame EfYRegs Map is updated via the normal
/// record-with allocation.
///
/// In-place mutation is safe because WAM state ownership is single-threaded
/// in this runtime: each step returns the new state, the caller uses it,
/// and the previous reference is dead.  CPs snapshot WsRegs via an explicit
/// Array.copy at TryMeElse / BeginAggregate / FactRetry / etc., so any
/// subsequent in-place write doesn't bleed back into the CP's saved regs.
/// Removing the previous per-write Array.copy of the 512-entry MaxRegs
/// array dropped putReg from ~32% of CPU time (per dotnet-trace on the
/// parser-heavy benchmark) to near zero.
let putReg (n: int) (v: Value) (s: WamState) : WamState =
    if n >= 201 then
        if n < s.WsRegs.Length then s.WsRegs.[n] <- v
        match s.WsStack with
        | frame :: rest ->
            let newFrame = { frame with EfYRegs = Map.add (n - 200) v frame.EfYRegs }
            { s with WsStack = newFrame :: rest }
        | [] -> s
    else
        if n >= 0 && n < s.WsRegs.Length then s.WsRegs.[n] <- v
        s

/// Set a register value in-place (use only when no snapshot is needed).
let setReg (n: int) (v: Value) (regs: Value array) : Value array =
    let r = Array.copy regs
    r.[n] <- v
    r

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
    let regVal = if reg >= 0 && reg < s.WsRegs.Length then s.WsRegs.[reg] else Unbound -1
    match regVal with
    | Unbound -1 -> None
    | v ->
        let dv = derefVar s.WsBindings v
        match dv with
        | Unbound vid ->
            let r = Array.copy s.WsRegs
            r.[reg] <- value
            Some { s with
                     WsRegs     = r
                     WsBindings = Map.add vid value s.WsBindings
                     WsTrail    = { TrailVarId = vid
                                    TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen = s.WsTrailLen + 1 }
        | existing when existing = value -> Some s
        | _ -> None

// ============================================================================
// ISO error-term constructors.  Mirror the Python/Elixir/C++ shapes:
//   error(ErrorTerm, Context) where Context is an unbound var for v1.
// throwIsoError wraps an ErrorTerm in the error/2 envelope, deep-derefs
// it (so the catcher sees what the thrower meant, not stale bindings),
// then raises WamException for the nearest catch/3 try/with to catch.
// ============================================================================

/// Build error(instantiation_error, _) inner.
let makeInstantiationError () : Value = Atom \"instantiation_error\"

/// Build error(type_error(Expected, Culprit), _) inner.
let makeTypeError (expected: string) (culprit: Value) : Value =
    Str (\"type_error\", [Atom expected; culprit])

/// Build error(domain_error(Domain, Culprit), _) inner.
let makeDomainError (domain: string) (culprit: Value) : Value =
    Str (\"domain_error\", [Atom domain; culprit])

/// Build error(evaluation_error(Kind), _) inner.
let makeEvaluationError (kind: string) : Value =
    Str (\"evaluation_error\", [Atom kind])

/// Deep-deref a Value through current bindings so the thrown term is
/// self-contained.  Mirrors Python _deep_copy_term''s deref pass.
let rec derefDeep (bindings: Map<int, Value>) (v: Value) : Value =
    match derefVar bindings v with
    | Str (fn, args) -> Str (fn, args |> List.map (derefDeep bindings))
    | VList items    -> VList (items |> List.map (derefDeep bindings))
    | other          -> other

/// Wrap an ISO error term in error(ErrorTerm, _) and raise WamException.
/// The fresh unbound Context slot follows the C++ spec §5 decision: the
/// standard catcher shape `error(Pattern, _)` will unify regardless.
let throwIsoError (s: WamState) (errTerm: Value) : 'a =
    let wrapped = Str (\"error\", [derefDeep s.WsBindings errTerm; Unbound s.WsVarCounter])
    raise (WamException wrapped)

/// Build a predicate-indicator term (Atom/Arity) for ISO error reports.
/// Used as the Culprit for type_error(evaluable, X/N).
let makePredIndicator (name: string) (arity: int) : Value =
    Str (\"/\", [Atom name; Integer arity])

/// True iff any subterm in v dereferences to an unbound variable.
/// Used by is_iso/2 and the ISO arithmetic-compare variants to
/// detect instantiation_error before evaluating.
let rec hasUnboundDeep (bindings: Map<int, Value>) (v: Value) : bool =
    match derefVar bindings v with
    | Unbound _      -> true
    | Str (_, args)  -> args  |> List.exists (hasUnboundDeep bindings)
    | VList items    -> items |> List.exists (hasUnboundDeep bindings)
    | _              -> false

/// Build the Name/Arity culprit term used by type_error(evaluable, ...).
/// Walks one level: Atom 'foo' -> foo/0; Str (f, args) -> f/(length args);
/// anything else -> unknown/0.  Mirrors Python iso_arith_culprit.
let arithCulprit (bindings: Map<int, Value>) (expr: Value) : Value =
    match derefVar bindings expr with
    | Atom name      -> makePredIndicator name 0
    | Str (fn, args) -> makePredIndicator fn (List.length args)
    | _              -> makePredIndicator \"unknown\" 0

/// True for predicate names the WAM compiler emits as Call/Execute
/// (meta-call shape) but that the F# step function handles as ISO
/// builtins.  The Call/Execute step arms check this to route through
/// BuiltinCall dispatch.
let isIsoMetaBuiltin (pred: string) : bool =
    match pred with
    | \"catch/3\" | \"throw/1\"
    | \"is_iso/2\" | \"is_lax/2\"
    | \"<_iso/2\" | \">_iso/2\" | \">=_iso/2\" | \"=<_iso/2\"
    | \"=:=_iso/2\" | \"=\\\\=_iso/2\"
    | \"<_lax/2\" | \">_lax/2\" | \">=_lax/2\" | \"=<_lax/2\"
    | \"=:=_lax/2\" | \"=\\\\=_lax/2\"
    | \"succ/2\" | \"succ_iso/2\" | \"succ_lax/2\" -> true
    | _ -> false

/// Arity of an ISO meta-builtin.  Used by the Execute step arm to
/// reconstruct the BuiltinCall instruction.
let isoMetaBuiltinArity (pred: string) : int =
    match pred with
    | \"throw/1\" -> 1
    | \"catch/3\" -> 3
    | _ -> 2  // is_*, comparison ops, succ all arity 2

/// Append a value to the current builder (PutStructure / PutList).
// popBuilderStack: when an inner builder is finished (BuildStruct/List
// arity filled, or ReadArgs exhausted), restore the outer builder
// from WsBuilderStack so the parent structure can keep filling.
// Both R (build_stack pop in append_build_arg) and Python (write_stack
// pop in _finish_write_ctx) do this on every fill completion -- not
// just the final one -- in case the parent is also at its limit.
// We don''t cascade-fill here (the parent already holds an Unbound
// var bound to the inner struct via its register, so derefVar
// resolves it transparently); we just unwind the saved contexts.
let popBuilderStack (s: WamState) : WamState =
    match s.WsBuilderStack with
    | outer :: rest -> { s with WsBuilder = Some outer; WsBuilderStack = rest }
    | []            -> { s with WsBuilder = None }

// pushBuilderIfActive: when starting a new inner build/read context
// (GetStructure / GetList on a register that produces nesting), save
// the current builder onto the stack so the inner can use WsBuilder
// without clobbering the outer.  No-op when there''s no active
// builder.  Returns the modified state; callers then set the new
// WsBuilder on the returned state.
let pushBuilderIfActive (s: WamState) : WamState =
    match s.WsBuilder with
    | Some b -> { s with WsBuilderStack = b :: s.WsBuilderStack }
    | None   -> s

let addToBuilder (value: Value) (s: WamState) : WamState option =
    match s.WsBuilder with
    | None -> None
    | Some (BuildStruct (fn, reg, arity, args)) ->
        let args' = args @ [value]
        if List.length args' = arity then
            let str = Str (fn, args')
            // Route the destination write through putReg so Y registers
            // (n >= 201) land in the env frame rather than WsRegs.
            // Snapshot the previous value first for binding trail.
            let regVal =
                match getReg reg s with
                | Some v -> v
                | None   -> Unbound -1
            let s0 = putReg reg str s
            // Cycle check: bind regVal''s vid to the new struct ONLY
            // when vid doesn''t appear inside the struct.  Two
            // patterns drive this (#2400 continuation):
            //   1. PutList ai + SetValue Yn where Yn=ai (cyclic):
            //      ai was the caller''s output var.  SetValue reads
            //      Yn = ai = Unbound vid, appends it as the tail.
            //      Materialization sees vid inside the new term ->
            //      skipping the bind prevents `vid -> Str(...,vid)`
            //      cycle.  (E.g. parse_primary''s tk_lparen clause
            //      building `[tk_rparen | Rest]` where Rest is the
            //      caller''s output reg.)
            //   2. SetVariable Yn + PutStructure Yn (non-cyclic):
            //      Yn was a fresh var, also referenced by some outer
            //      builder.  Materialization binds vid -> new struct
            //      so the outer ref derefs to the new term.  (E.g.
            //      parse_op_loop building `[OpName|[Left|[Right|[]]]]`
            //      via successive PutStructure on fresh tail vars.)
            let regValVid =
                match derefVar s.WsBindings regVal with
                | Unbound v -> v
                | _ -> -1
            let rec containsVid v =
                match derefVar s.WsBindings v with
                | Unbound v' when v' >= 0 -> v' = regValVid
                | Str (_, xs) | VList xs -> List.exists containsVid xs
                | _ -> false
            let s0' =
                match derefVar s.WsBindings regVal with
                // KNOWN OPEN BUG (bind-through sweep, side finding 4):
                // the cycle check alone is NOT sufficient — a goal
                // structure that does NOT contain the A-register
                // occupant still aliases it (probe:
                // `p(X) :- q(g(7)), var(X)` wrong-fails; see
                // docs/reports/wam_bindthrough_cross_target_sweep.md).
                // The register-class guard the other 8 targets carry
                // CANNOT be applied here: unlike their compilers, the
                // F# pipeline stages legitimate build placeholders in
                // A registers too (gating on reg >= 100 regressed the
                // compiled parser smoke from 42/42 to 1/42). Fixing
                // this needs F#-specific occupant provenance — left
                // open for the F# stream with the probe attached.
                | Unbound vid when vid >= 0 && not (containsVid str) ->
                    { s0 with
                         WsBindings= Map.add vid str s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
                | _ -> s0
            let s1 = popBuilderStack s0'
            Some { s1 with WsPC = s.WsPC + 1 }
        else
            Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildStruct (fn, reg, arity, args')) }
    | Some (BuildList (reg, items)) ->
        let items' = items @ [value]
        if List.length items' = 2 then
            let head = List.item 0 items'
            let tail = List.item 1 items'
            let listVal =
                // Cons cell with tail.  Atom \"[]\" is the empty-list
                // atom and must collapse to VList []; otherwise
                // `[53|[]]` materializes as VList [Integer 53; Atom \"[]\"]
                // which is a two-element list, not a singleton.  The
                // parser library''s tokenize / take_digits / etc. build
                // single-element accumulators via `[H|[]]` SetConstant
                // pairs, so this case is hit constantly.
                //
                // When the tail is anything else (Unbound, Str, ...),
                // use Str(\"[|]\", [h; t]) -- the proper Prolog cons-cell
                // representation -- so the tail can stay symbolic.
                // GetList recognizes both shapes.  Without this,
                // building `[H|T]` with T unbound produced a flat
                // VList [H; T] which represents the WRONG list.
                match derefVar s.WsBindings tail with
                | VList t      -> VList (head :: t)
                | Atom \"[]\"    -> VList [head]
                | derefdTail   -> Str (\"[|]\", [head; derefdTail])
            // Y-aware write of materialized list value to destination reg.
            let regVal =
                match getReg reg s with
                | Some v -> v
                | None   -> Unbound -1
            let s0 = putReg reg listVal s
            // Cycle check — see BuildStruct above for context.  Same
            // rule: bind regVal''s vid to listVal unless vid appears
            // inside listVal (which would create a self-reference).
            let regValVid =
                match derefVar s.WsBindings regVal with
                | Unbound v -> v
                | _ -> -1
            let rec containsVid v =
                match derefVar s.WsBindings v with
                | Unbound v' when v' >= 0 -> v' = regValVid
                | Str (_, xs) | VList xs -> List.exists containsVid xs
                | _ -> false
            let s0' =
                match derefVar s.WsBindings regVal with
                // Same KNOWN OPEN BUG as BuildStruct above (sweep
                // side finding 4): cycle check only; the register
                // guard is not applicable to the F# pipeline.
                | Unbound vid when vid >= 0 && not (containsVid listVal) ->
                    { s0 with
                         WsBindings= Map.add vid listVal s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
                | _ -> s0
            let s1 = popBuilderStack s0'
            Some { s1 with WsPC = s.WsPC + 1 }
        else
            Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildList (reg, items')) }
    | Some (ReadArgs _) -> None

let readNextArg (s: WamState) : (Value * WamState) option =
    match s.WsBuilder with
    | Some (ReadArgs (v :: rest)) ->
        let s' =
            if List.isEmpty rest then popBuilderStack s
            else { s with WsBuilder = Some (ReadArgs rest) }
        Some (derefVar s.WsBindings v, s')
    | _ -> None

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
    | Str (\"//\", [a; b]) ->
        // Prolog integer division: truncate quotient toward zero (SWI default,
        // integer_rounding_function = toward_zero).
        match evalArith bindings a, evalArith bindings b with
        | Some x, Some y when y <> 0.0 -> Some (float (truncate (x / y)))
        | _ -> None
    | Str (\"mod\", [a; b]) ->
        match evalArith bindings a, evalArith bindings b with
        | Some x, Some y when y <> 0.0 -> Some (float (int x % int y))
        | _ -> None
    | _ -> None

/// Binary search on a sorted (string * int) array. Returns Some pc or None.
let binarySearchStr (arr: (string * int) array) (key: string) : int option =
    let mutable lo = 0
    let mutable hi = arr.Length - 1
    let mutable result = -1
    while lo <= hi && result = -1 do
        let mid = (lo + hi) / 2
        let cmp = System.String.Compare(fst arr.[mid], key, System.StringComparison.Ordinal)
        if cmp = 0 then result <- mid
        elif cmp < 0 then lo <- mid + 1
        else hi <- mid - 1
    if result >= 0 then Some (snd arr.[result]) else None

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
        x1 :: rest1, c2, m2

/// Standard-order comparison on Value terms.  Mirrors the Go target's
/// compareValues helper (templates/targets/go_wam/state.go.mustache) and
/// the Prolog/ISO standard ordering of terms:
///
///   Var < Number < Atom < String < Compound (incl. VList)
///
/// Numbers (Integer / Float) are compared by numeric value, bridging int
/// and float via float-promotion.  Atoms compare by ordinal string order.
/// Compound terms (Str / VList) compare first by arity, then by functor
/// name, then element-wise.  VList behaves like a compound with functor
/// '.' and arity matching the list length.
///
/// F#''s auto-generated structural comparison for the Value DU follows
/// the constructor declaration order (Atom < Integer < Float < VList <
/// Str < Unbound < Ref) — which is NOT Prolog standard order. So
/// builtins that need standard order (compare/3, @</2, @=</2, @>/2,
/// @>=/2, sort/2, msort/2) call compareValue directly rather than
/// relying on the F# `compare` operator on Value.
let rec compareValue (a: Value) (b: Value) : int =
    let rankOf v =
        match v with
        | Unbound _           -> 0
        | Integer _ | Float _ -> 1
        | Atom _              -> 2
        | Str _ | VList _     -> 3
        | VSet _              -> 3  // VSet is a compound-ish visited set
        | Ref _               -> 4
    match a, b with
    | Unbound x, Unbound y -> compare x y
    | Integer x, Integer y -> compare x y
    | Float x,   Float y   -> compare x y
    | Integer x, Float y   -> compare (float x) y
    | Float x,   Integer y -> compare x (float y)
    | Atom x,    Atom y    -> System.String.Compare(x, y, System.StringComparison.Ordinal)
    | Ref x,     Ref y     -> compare x y
    | VSet x,    VSet y    -> compare x y  // F# Set has structural comparison
    | Str (fx, ax), Str (fy, ay) ->
        let ca = compare (List.length ax) (List.length ay)
        if ca <> 0 then ca
        else
            let cf = System.String.Compare(fx, fy, System.StringComparison.Ordinal)
            if cf <> 0 then cf
            else compareValueList ax ay
    | VList xs, VList ys -> compareValueList xs ys
    | Str (_, ax), VList ys ->
        let ca = compare (List.length ax) (List.length ys)
        if ca <> 0 then ca else compareValueList ax ys
    | VList xs, Str (_, ay) ->
        let ca = compare (List.length xs) (List.length ay)
        if ca <> 0 then ca else compareValueList xs ay
    | _ -> compare (rankOf a) (rankOf b)

and compareValueList (xs: Value list) (ys: Value list) : int =
    match xs, ys with
    | [], []           -> 0
    | [], _            -> -1
    | _, []            -> 1
    | x :: xt, y :: yt ->
        let c = compareValue x y
        if c <> 0 then c else compareValueList xt yt").

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
        fsharp_wam_instruction_type,   nl,
        fsharp_wam_context_type,       nl,
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
