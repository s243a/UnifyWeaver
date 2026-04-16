:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_fsharp_target.pl - WAM-to-F# Transpilation Target
%
% Compiles WAM instructions to F# code using F#'s native immutable Map<'K,'V>.
%
% Design mirrors the Haskell WAM target closely (use Haskell as primary
% reference, Rust as secondary — see PR #1378 perf log for lessons applied):
%
%   - Map<int,Value> for registers/bindings → O(1) structural-sharing
%     snapshots for choice points (same insight as Haskell Phase-A)
%   - WamContext separated from WamState (hot/cold split — Haskell Phase-B)
%   - Labels pre-resolved to int PCs at load time (Phase-C win: -47%)
%   - Skip WAM-compilation of FFI-owned fact predicates (Phase-D: -70%)
%   - Atom interning at FFI boundary (Phase-D: -48% query time)
%
% Key F# idioms vs Haskell:
%   - `match x with | Pat -> ...` instead of `case x of`
%   - `{ s with Field = v }` record update syntax
%   - `option<'T>` / `Some` / `None` instead of `Maybe` / `Just` / `Nothing`
%   - `Map.tryFind k m` instead of `Map.lookup k m`
%   - `Map.add k v m` instead of `Map.insert k v m`
%   - No Bang patterns / UNPACK — F# is strict by default (no laziness overhead)
%   - `Array.get arr i` instead of `arr ! i` for O(1) instruction fetch
%   - `Parallel.map` (TPL) for intra-query parallelism instead of `parMap`
%
% Pipeline:
%   Prolog source → fsharp_target.pl (native lowering, preferred)
%                 → wam_target.pl (WAM compilation, fallback)
%                 → wam_fsharp_target.pl (THIS FILE: WAM → F#)
%
% See: docs/design/WAM_PERF_OPTIMIZATION_LOG.md
%      src/unifyweaver/targets/wam_haskell_target.pl (primary reference)
%      src/unifyweaver/targets/wam_elixir_target.pl  (structural reference)

:- module(wam_fsharp_target, [
    compile_wam_predicate_to_fsharp/4,   % +Pred/Arity, +WamCode, +Options, -FSharpCode
    compile_wam_runtime_to_fsharp/3,     % +Options, +DetectedKernels, -Code
    write_wam_fsharp_project/3,          % +Predicates, +Options, +ProjectDir
    wam_fsharp_resolve_emit_mode/2,      % +Options, -Mode
    wam_fsharp_partition_predicates/5    % +Mode, +Predicates, +DK, -Interp, -Lowered
]).

:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/recursive_kernel_detection',
             [detect_recursive_kernel/4, kernel_metadata/4, kernel_config/2,
              kernel_register_layout/2, kernel_native_call/2, kernel_template_file/2]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../core/purity_certificate', [analyze_predicate_purity/2]).
:- use_module('../bindings/fsharp_wam_bindings').

% Phase 3 lowered emitter lives in wam_fsharp_lowered_emitter.
:- reexport('wam_fsharp_lowered_emitter',
            [wam_fsharp_lowerable/3, lower_predicate_to_fsharp/4]).

% ============================================================================
% Emit mode selector — identical hierarchy to Haskell target
% ============================================================================
%
% Three modes:
%   interpreter   — every predicate via the instruction-array interpreter
%   functions     — every predicate attempts lowering; falls back to interp
%   mixed(List)   — named predicates attempt lowering; rest use interpreter
%
% Selector hierarchy (checked in order):
%   1. emit_mode(Mode) option
%   2. user:wam_fsharp_emit_mode(Mode) dynamic fact
%   3. default: interpreter

:- multifile user:wam_fsharp_emit_mode/1.

%% wam_fsharp_resolve_emit_mode(+Options, -Mode)
wam_fsharp_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  wam_fsharp_validate_emit_mode(M0, Mode)
    ;   catch(user:wam_fsharp_emit_mode(M1), _, fail)
    ->  wam_fsharp_validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

wam_fsharp_validate_emit_mode(interpreter, interpreter) :- !.
wam_fsharp_validate_emit_mode(functions,   functions)   :- !.
wam_fsharp_validate_emit_mode(mixed(L), mixed(L)) :- is_list(L), !.
wam_fsharp_validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_fsharp_emit_mode, Other),
                wam_fsharp_resolve_emit_mode/2)).

%% wam_fsharp_partition_predicates(+Mode, +Predicates, +DK, -Interp, -Lowered)
wam_fsharp_partition_predicates(interpreter, Predicates, _, Predicates, []) :- !.
wam_fsharp_partition_predicates(functions, Predicates, DK, Interpreted, Lowered) :- !,
    pairs_keys(DK, KernelKeys),
    wam_fsharp_partition_try_lower(Predicates, KernelKeys, Interpreted, Lowered).
wam_fsharp_partition_predicates(mixed(HotPreds), Predicates, DK, Interpreted, Lowered) :- !,
    pairs_keys(DK, KernelKeys),
    wam_fsharp_partition_mixed(Predicates, HotPreds, KernelKeys, Interpreted, Lowered).

wam_fsharp_partition_try_lower([], _, [], []).
wam_fsharp_partition_try_lower([P|Rest], KK, Interpreted, Lowered) :-
    pred_key_fs(P, Key),
    (   member(Key, KK)
    ->  Interpreted = [P|IR],
        wam_fsharp_partition_try_lower(Rest, KK, IR, Lowered)
    ;   wam_fsharp_predicate_wamcode(P, WamCode),
        (   wam_fsharp_lowerable(P, WamCode, _Reason)
        ->  Lowered = [P|LR],
            wam_fsharp_partition_try_lower(Rest, KK, Interpreted, LR)
        ;   Interpreted = [P|IR],
            wam_fsharp_partition_try_lower(Rest, KK, IR, Lowered)
        )
    ).

wam_fsharp_partition_mixed([], _, _, [], []).
wam_fsharp_partition_mixed([P|Rest], HotPreds, KK, Interpreted, Lowered) :-
    pred_key_fs(P, Key),
    (   member(Key, KK)
    ->  Interpreted = [P|IR],
        wam_fsharp_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
    ;   wam_fsharp_indicator_in_list(P, HotPreds)
    ->  wam_fsharp_predicate_wamcode(P, WamCode),
        (   wam_fsharp_lowerable(P, WamCode, _Reason)
        ->  Lowered = [P|LR],
            wam_fsharp_partition_mixed(Rest, HotPreds, KK, Interpreted, LR)
        ;   Interpreted = [P|IR],
            wam_fsharp_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
        )
    ;   Interpreted = [P|IR],
        wam_fsharp_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
    ).

pred_key_fs(P, Key) :-
    (P = _Mod:Pred/Arity -> true ; P = Pred/Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]).

wam_fsharp_indicator_in_list(P, HotPreds) :- member(P, HotPreds), !.
wam_fsharp_indicator_in_list(_Mod:Pred/Arity, HotPreds) :-
    member(Pred/Arity, HotPreds), !.

wam_fsharp_predicate_wamcode(PI, WamCode) :-
    (   PI = _Module:Pred/Arity -> true ; PI = Pred/Arity ),
    wam_target:compile_predicate_to_wam(Pred/Arity, [], WamCode).

% ============================================================================
% Kernel detection (shared with Haskell target pattern)
% ============================================================================

detect_kernels_fs([], []).
detect_kernels_fs([PI|Rest], Kernels) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses \= [],
        detect_recursive_kernel(Pred, Arity, Clauses, Kernel)
    ->  format(atom(Key), '~w/~w', [Pred, Arity]),
        Kernels = [Key-Kernel|RestKernels]
    ;   Kernels = RestKernels
    ),
    detect_kernels_fs(Rest, RestKernels).

% ============================================================================
% Base PC computation (for lowered function PC offset correctness)
% ============================================================================

compute_base_pcs_fs(Predicates, Map) :-
    compute_base_pcs_fs_(Predicates, 1, Map).

compute_base_pcs_fs_([], _, []).
compute_base_pcs_fs_([PI|Rest], StartPC, [Key-StartPC|RestMap]) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_fsharp_predicate_wamcode(PI, WamCode),
    count_wam_instructions_fs(WamCode, Count),
    NextPC is StartPC + Count,
    compute_base_pcs_fs_(Rest, NextPC, RestMap).

count_wam_instructions_fs(WamCode, Count) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    include(is_wam_instruction_line_fs, Lines, InstrLines),
    length(InstrLines, Count).

is_wam_instruction_line_fs(Line) :-
    split_string(Line, "", " \t", [Trimmed]),
    Trimmed \== "",
    \+ sub_string(Trimmed, _, 1, 0, ":").

predicate_base_pc_fs(P, Map, PC) :-
    (   P = _Mod:Pred/Arity -> true ; P = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    (   member(Key-PC, Map) -> true ; PC = 1 ).

% ============================================================================
% lower_all: run the lowered emitter over LoweredList
% ============================================================================

lower_all_fs([], _, _, []).
lower_all_fs([P|Rest], BasePCMap, DetectedKernels, [Entry|RestEntries]) :-
    wam_fsharp_predicate_wamcode(P, WamCode),
    predicate_base_pc_fs(P, BasePCMap, BasePC),
    pairs_keys(DetectedKernels, ForeignKeys),
    lower_predicate_to_fsharp(P, WamCode,
        [base_pc(BasePC), foreign_preds(ForeignKeys)], Entry),
    lower_all_fs(Rest, BasePCMap, DetectedKernels, RestEntries).

% ============================================================================
% PHASE 1: WAM Instruction → F# Expression
%
% Each clause maps one WAM instruction to an F# state transformation
% of type: WamState -> WamState option
% None = failure (backtrack), Some s = success with updated state.
%
% F# idioms applied throughout:
%   - Map.tryFind instead of Map.lookup
%   - Map.add instead of Map.insert
%   - { s with ... } record update
%   - match ... with | Pat -> ... instead of case
%   - Option.defaultValue instead of fromMaybe
% ============================================================================

%% wam_to_fsharp(+Instruction, -FSharpExpr)

wam_to_fsharp(get_constant(C, Ai), Code) :-
    format(string(Code),
'    let valOpt = Map.tryFind ~w s.WsRegs |> Option.map (derefVar s.WsBindings)
    match valOpt with
    | Some v when v = ~w -> Some { s with WsPC = s.WsPC + 1 }
    | Some (Unbound vid) ->
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsRegs    = Map.add ~w ~w s.WsRegs
                 WsBindings= Map.add vid ~w s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | _ -> None', [Ai, C, Ai, C, C]).

wam_to_fsharp(get_variable(Xn, Ai), Code) :-
    format(string(Code),
'    match Map.tryFind ~w s.WsRegs with
    | Some v ->
        let dv = derefVar s.WsBindings v
        Some (putReg ~w dv { s with WsPC = s.WsPC + 1 })
    | None -> None', [Ai, Xn]).

wam_to_fsharp(get_value(Xn, Ai), Code) :-
    format(string(Code),
'    let va = Map.tryFind ~w s.WsRegs |> Option.map (derefVar s.WsBindings)
    let vx = getReg ~w s
    match va, vx with
    | Some a, Some x when a = x -> Some { s with WsPC = s.WsPC + 1 }
    | Some (Unbound vid), Some x ->
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsRegs    = Map.add ~w x s.WsRegs
                 WsBindings= Map.add vid x s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | _ -> None', [Ai, Xn, Ai]).

wam_to_fsharp(put_constant(C, Ai), Code) :-
    format(string(Code),
'    Some { s with WsPC = s.WsPC + 1; WsRegs = Map.add ~w ~w s.WsRegs }', [Ai, C]).

wam_to_fsharp(put_variable(Xn, Ai), Code) :-
    format(string(Code),
'    let vid = s.WsVarCounter
    let var = Unbound vid
    let s1  = putReg ~w var s
    Some { s1 with
             WsPC       = s.WsPC + 1
             WsRegs     = Map.add ~w var s1.WsRegs
             WsVarCounter= s.WsVarCounter + 1 }', [Xn, Ai]).

wam_to_fsharp(put_value(Xn, Ai), Code) :-
    format(string(Code),
'    match getReg ~w s with
    | Some v -> Some { s with WsPC = s.WsPC + 1; WsRegs = Map.add ~w v s.WsRegs }
    | None   -> None', [Xn, Ai]).

wam_to_fsharp(call(Pred, _Arity), Code) :-
    format(string(Code),
'    Some { s with WsPC = lookupLabel "~w" ctx; WsCP = s.WsPC + 1 }', [Pred]).

wam_to_fsharp(proceed, Code) :-
    Code = '    let ret = s.WsCP
    if ret = 0 then Some { s with WsPC = 0 }
    else Some { s with WsPC = ret; WsCP = 0 }'.

wam_to_fsharp(allocate, Code) :-
    Code = '    let frame = { EfSavedCP = s.WsCP; EfYRegs = Map.empty }
    Some { s with
             WsPC    = s.WsPC + 1
             WsStack = frame :: s.WsStack
             WsCutBar= s.WsCPsLen }'.

wam_to_fsharp(deallocate, Code) :-
    Code = '    match s.WsStack with
    | ef :: rest -> Some { s with WsPC = s.WsPC + 1; WsStack = rest; WsCP = ef.EfSavedCP }
    | []         -> None'.

wam_to_fsharp(try_me_else(Label), Code) :-
    format(string(Code),
'    let nextPC = lookupLabel "~w" ctx
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
    Some { s with WsPC = s.WsPC + 1; WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }',
    [Label]).

wam_to_fsharp(trust_me, Code) :-
    Code = '    match s.WsCPs with
    | _ :: rest -> Some { s with WsPC = s.WsPC + 1; WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | []        -> None'.

wam_to_fsharp(retry_me_else(Label), Code) :-
    format(string(Code),
'    match s.WsCPs with
    | cp :: rest ->
        let nextPC = lookupLabel "~w" ctx
        Some { s with WsPC = s.WsPC + 1; WsCPs = { cp with CpNextPC = nextPC } :: rest }
    | [] -> None', [Label]).

wam_to_fsharp(builtin_call('!/0', 0), Code) :-
    Code = '    Some { s with WsPC = s.WsPC + 1
                         WsCPs    = List.take s.WsCutBar s.WsCPs
                         WsCPsLen = s.WsCutBar }'.

wam_to_fsharp(builtin_call('is/2', 2), Code) :-
    Code = '    let expr   = Map.tryFind 2 s.WsRegs |> Option.defaultValue (Integer 0) |> derefVar s.WsBindings
    let result = evalArith s.WsBindings expr
    let lhs    = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings)
    match lhs, result with
    | Some (Unbound vid), Some r ->
        let v = if float (int r) = r then Integer (int r) else Float r
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsRegs    = Map.add 1 v s.WsRegs
                 WsBindings= Map.add vid v s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | Some (Integer n), Some r when float n = r -> Some { s with WsPC = s.WsPC + 1 }
    | Some (Float f),   Some r when f = r       -> Some { s with WsPC = s.WsPC + 1 }
    | _ -> None'.

wam_to_fsharp(builtin_call('</2', 2), Code) :-
    Code = '    let v1 = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
    let v2 = Map.tryFind 2 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
    match v1, v2 with
    | Some a, Some b when a < b -> Some { s with WsPC = s.WsPC + 1 }
    | _ -> None'.

wam_to_fsharp(builtin_call('>/2', 2), Code) :-
    Code = '    let v1 = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
    let v2 = Map.tryFind 2 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
    match v1, v2 with
    | Some a, Some b when a > b -> Some { s with WsPC = s.WsPC + 1 }
    | _ -> None'.

wam_to_fsharp(builtin_call('\\+/1', 1), Code) :-
    Code = '    let goal = Map.tryFind 1 s.WsRegs |> Option.bind (derefHeap s.WsHeap)
    match goal with
    | Some (Str ("member", [needle; haystack])) ->
        let n = derefVar s.WsBindings needle
        let h = derefVar s.WsBindings haystack
        let found = match h with
                    | VList items -> List.exists (fun item -> derefVar s.WsBindings item = n) items
                    | _ -> false
        if found then None else Some { s with WsPC = s.WsPC + 1 }
    | _ -> None'.

wam_to_fsharp(builtin_call('length/2', 2), Code) :-
    Code = '    let listVal = Map.tryFind 1 s.WsRegs |> Option.defaultValue (VList []) |> derefVar s.WsBindings
    match listVal with
    | VList items ->
        let len = List.length items
        let lhs = Map.tryFind 2 s.WsRegs |> Option.map (derefVar s.WsBindings)
        match lhs with
        | Some (Unbound vid) ->
            let v = Integer len
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = Map.add 2 v s.WsRegs
                     WsBindings= Map.add vid v s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | Some (Integer n) when n = len -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None
    | _ -> None'.

% ============================================================================
% PHASE 2: Backtrack Function
% ============================================================================

backtrack_fsharp(Code) :-
    Code = '/// Restore state from the top choice point.
/// Dispatches: aggregate frame → finalizeAggregate, builtin → resumeBuiltin, normal → restore.
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
        let trailLen  = cp.CpTrailLen
        let diff      = s.WsTrailLen - trailLen
        let newEntries= s.WsTrail |> List.take diff |> List.rev
        let restoredBindings = List.fold undoBinding cp.CpBindings newEntries
        Some { s with
                 WsPC       = cp.CpNextPC
                 WsRegs     = cp.CpRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = trailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = restoredBindings
                 WsCutBar   = cp.CpCutBar
                 WsCPs      = rest
                 WsCPsLen   = s.WsCPsLen - 1 }

and resumeBuiltin (bs: BuiltinState) (cp: ChoicePoint) (rest: ChoicePoint list) (s: WamState) : WamState option =
    match bs with
    | FactRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | FactRetry (vid, v :: vs, retPC) ->
        let newBindings = Map.add vid (Atom v) cp.CpBindings
        let newRegs     = Map.add 2 (Atom v) cp.CpRegs
        let newCPs      = match vs with
                          | [] -> rest
                          | _  -> { cp with CpBuiltin = Some (FactRetry (vid, vs, retPC)) } :: rest
        let diff = s.WsTrailLen - cp.CpTrailLen
        Some { s with
                 WsPC       = retPC
                 WsRegs     = newRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = cp.CpTrailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = newBindings
                 WsCutBar   = cp.CpCutBar
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }
    | HopsRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | HopsRetry (vid, h :: hs, retPC) ->
        let newBindings = Map.add vid (Integer h) cp.CpBindings
        let newRegs     = Map.add 3 (Integer h) cp.CpRegs
        let newCPs      = match hs with
                          | [] -> rest
                          | _  -> { cp with CpBuiltin = Some (HopsRetry (vid, hs, retPC)) } :: rest
        let diff = s.WsTrailLen - cp.CpTrailLen
        Some { s with
                 WsPC       = retPC
                 WsRegs     = newRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = cp.CpTrailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = newBindings
                 WsCutBar   = cp.CpCutBar
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
        let diff = s.WsTrailLen - cp.CpTrailLen
        Some { s with
                 WsPC       = retPC
                 WsRegs     = newRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = cp.CpTrailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = newBindings
                 WsCutBar   = cp.CpCutBar
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }

and backtrackInner (_returnPC: int) (s: WamState) : WamState option =
    match s.WsCPs with
    | cp :: _ when cp.CpAggFrame.IsSome -> None   // reached aggregate frame = done
    | _ -> backtrack s

and finalizeAggregate (returnPC: int) (s: WamState) : WamState option =
    let rec go cps =
        match cps with
        | [] -> None
        | cp :: rest ->
            match cp.CpAggFrame with
            | Some af ->
                let accum  = List.rev s.WsAggAccum
                let result = applyAggregation af.AggType accum
                let cpState= { s with WsRegs = cp.CpRegs; WsStack = cp.CpStack; WsBindings = cp.CpBindings }
                let resVal = getReg af.AggResReg cpState
                let diff   = s.WsTrailLen - cp.CpTrailLen
                let restoredTrail = List.skip diff s.WsTrail
                let finalRegs, finalBindings, finalTrail, finalTrailLen =
                    match resVal with
                    | Some (Unbound vid) ->
                        ( Map.add af.AggResReg result cp.CpRegs
                        , Map.add vid result cp.CpBindings
                        , { TrailVarId = vid; TrailOldVal = Map.tryFind vid cp.CpBindings } :: restoredTrail
                        , cp.CpTrailLen + 1 )
                    | _ -> (cp.CpRegs, cp.CpBindings, restoredTrail, cp.CpTrailLen)
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
            | None -> go rest
    go s.WsCPs

and applyAggregation (typ: string) (vals: Value list) : Value =
    match typ with
    | "sum" ->
        let toNum = function Integer n -> float n | Float f -> f | _ -> 0.0
        let s = List.sumBy toNum vals
        if float (int s) = s then Integer (int s) else Float s
    | "count"   -> Integer (List.length vals)
    | "collect" -> VList vals
    | _         -> VList vals'.

% ============================================================================
% PHASE 3: Step Function
% ============================================================================

step_function_fsharp(Code) :-
    Code = '/// Execute a single WAM instruction.
/// WamContext is read-only — threaded through without per-step allocation.
let rec step (ctx: WamContext) (s: WamState) (instr: Instruction) : WamState option =
    match instr with
    | GetConstant (c, ai) ->
        let valOpt = Map.tryFind ai s.WsRegs |> Option.map (derefVar s.WsBindings)
        match valOpt with
        | Some v when v = c -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Unbound vid) ->
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = Map.add ai c s.WsRegs
                     WsBindings= Map.add vid c s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | _ -> None

    | GetVariable (xn, ai) ->
        match Map.tryFind ai s.WsRegs with
        | Some v -> let dv = derefVar s.WsBindings v
                    Some (putReg xn dv { s with WsPC = s.WsPC + 1 })
        | None   -> None

    | GetValue (xn, ai) ->
        let va = Map.tryFind ai s.WsRegs |> Option.map (derefVar s.WsBindings)
        let vx = getReg xn s
        match va, vx with
        | Some a, Some x when a = x -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Unbound vid), Some x ->
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = Map.add ai x s.WsRegs
                     WsBindings= Map.add vid x s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | _ -> None

    | PutConstant (c, ai) ->
        Some { s with WsPC = s.WsPC + 1; WsRegs = Map.add ai c s.WsRegs }

    | PutVariable (xn, ai) ->
        let vid = s.WsVarCounter
        let var = Unbound vid
        let s1  = putReg xn var s
        Some { s1 with
                 WsPC        = s.WsPC + 1
                 WsRegs      = Map.add ai var s1.WsRegs
                 WsVarCounter= s.WsVarCounter + 1 }

    | PutValue (xn, ai) ->
        match getReg xn s with
        | Some v -> Some { s with WsPC = s.WsPC + 1; WsRegs = Map.add ai v s.WsRegs }
        | None   -> None

    | PutStructure (fn, ai, arity) ->
        Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildStruct (fn, ai, arity, [])) }

    | PutList ai ->
        Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildList (ai, [])) }

    | SetValue xn ->
        match getReg xn s with
        | Some v -> addToBuilder v s
        | None   -> None

    | SetConstant c -> addToBuilder c s

    | CallResolved (pc, _arity) ->
        Some { s with WsPC = pc; WsCP = s.WsPC + 1 }

    | CallForeign (pred, _arity) ->
        executeForeign ctx pred { s with WsCP = s.WsPC + 1 }

    | Call (pred, _arity) ->
        let sc = { s with WsCP = s.WsPC + 1 }
        match Map.tryFind pred ctx.WcLoweredPredicates with
        | Some fn -> fn ctx sc
        | None ->
        match callIndexedFact2 ctx pred sc with
        | Some sr -> Some sr
        | None ->
        match Map.tryFind pred ctx.WcLabels with
        | Some pc -> Some { sc with WsPC = pc }
        | None    -> None

    | Execute pred ->
        match Map.tryFind pred ctx.WcLoweredPredicates with
        | Some fn -> fn ctx s
        | None ->
        match callIndexedFact2 ctx pred s with
        | Some sr -> Some sr
        | None ->
        match Map.tryFind pred ctx.WcLabels with
        | Some pc -> Some { s with WsPC = pc }
        | None    -> None

    | ExecutePc pc -> Some { s with WsPC = pc }

    | Jump label ->
        match Map.tryFind label ctx.WcLabels with
        | Some pc -> Some { s with WsPC = pc }
        | None    -> None

    | JumpPc pc -> Some { s with WsPC = pc }

    | Proceed ->
        let ret = s.WsCP
        if ret = 0 then Some { s with WsPC = 0 }
        else Some { s with WsPC = ret; WsCP = 0 }

    | Allocate ->
        let frame = { EfSavedCP = s.WsCP; EfYRegs = Map.empty }
        Some { s with
                 WsPC    = s.WsPC + 1
                 WsStack = frame :: s.WsStack
                 WsCutBar= s.WsCPsLen }

    | Deallocate ->
        match s.WsStack with
        | ef :: rest -> Some { s with WsPC = s.WsPC + 1; WsStack = rest; WsCP = ef.EfSavedCP }
        | []         -> None

    | TryMeElse label ->
        let nextPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
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
        Some { s with WsPC = s.WsPC + 1; WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

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
        Some { s with WsPC = s.WsPC + 1; WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    | TrustMe ->
        match s.WsCPs with
        | _ :: rest -> Some { s with WsPC = s.WsPC + 1; WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
        | []        -> None

    | RetryMeElse label ->
        match s.WsCPs with
        | cp :: rest ->
            let nextPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
            Some { s with WsPC = s.WsPC + 1; WsCPs = { cp with CpNextPC = nextPC } :: rest }
        | [] -> None

    | RetryMeElsePc nextPC ->
        match s.WsCPs with
        | cp :: rest -> Some { s with WsPC = s.WsPC + 1; WsCPs = { cp with CpNextPC = nextPC } :: rest }
        | []         -> None

    // Phase 4.1 parallel stubs — alias to sequential counterparts
    | ParTryMeElse label    -> step ctx s (TryMeElse label)
    | ParRetryMeElse label  -> step ctx s (RetryMeElse label)
    | ParTrustMe            -> step ctx s TrustMe
    | ParTryMeElsePc pc     -> step ctx s (TryMeElsePc pc)
    | ParRetryMeElsePc pc   -> step ctx s (RetryMeElsePc pc)

    | SwitchOnConstantPc table ->
        let valOpt = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings)
        match valOpt with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Atom key) ->
            match Map.tryFind key table with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> None
        | Some (Integer n) ->
            match Map.tryFind (string n) table with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> None
        | _ -> None

    | SwitchOnConstant table ->
        let valOpt = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings)
        match valOpt with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | Some v ->
            match Map.tryFind v table with
            | Some label ->
                match Map.tryFind label ctx.WcLabels with
                | Some pc -> Some { s with WsPC = pc }
                | None    -> None
            | None -> None
        | None -> None

    | BuiltinCall ("!/0", _) ->
        Some { s with
                 WsPC    = s.WsPC + 1
                 WsCPs   = List.take s.WsCutBar s.WsCPs
                 WsCPsLen= s.WsCutBar }

    | CutIte ->
        match s.WsCPs with
        | _ :: rest -> Some { s with WsPC = s.WsPC + 1; WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
        | []        -> Some { s with WsPC = s.WsPC + 1 }

    | BuiltinCall ("nonvar/1", _) ->
        match Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) with
        | Some (Unbound _) -> None
        | Some _           -> Some { s with WsPC = s.WsPC + 1 }
        | None             -> None

    | BuiltinCall ("var/1", _) ->
        match Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                -> None

    | BuiltinCall ("atom/1", _) ->
        match Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) with
        | Some (Atom _) -> Some { s with WsPC = s.WsPC + 1 }
        | _             -> None

    | BuiltinCall ("integer/1", _) ->
        match Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) with
        | Some (Integer _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                -> None

    | BuiltinCall ("number/1", _) ->
        match Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) with
        | Some (Integer _) | Some (Float _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                                  -> None

    | BuiltinCall ("is/2", _) ->
        let expr   = Map.tryFind 2 s.WsRegs |> Option.defaultValue (Integer 0) |> derefVar s.WsBindings
        let result = evalArith s.WsBindings expr
        let lhs    = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings)
        match lhs, result with
        | Some (Unbound vid), Some r ->
            let v = if float (int r) = r then Integer (int r) else Float r
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = Map.add 1 v s.WsRegs
                     WsBindings= Map.add vid v s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | Some (Integer n), Some r when float n = r -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("length/2", _) ->
        let listVal = Map.tryFind 1 s.WsRegs |> Option.defaultValue (VList []) |> derefVar s.WsBindings
        match listVal with
        | VList items ->
            let len = List.length items
            match Map.tryFind 2 s.WsRegs |> Option.map (derefVar s.WsBindings) with
            | Some (Unbound vid) ->
                let v = Integer len
                Some { s with
                         WsPC      = s.WsPC + 1
                         WsRegs    = Map.add 2 v s.WsRegs
                         WsBindings= Map.add vid v s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
            | Some (Integer n) when n = len -> Some { s with WsPC = s.WsPC + 1 }
            | _ -> None
        | _ -> None

    | BuiltinCall ("</2", _) ->
        let v1 = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
        let v2 = Map.tryFind 2 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a < b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall (">/2", _) ->
        let v1 = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
        let v2 = Map.tryFind 2 s.WsRegs |> Option.map (derefVar s.WsBindings) |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a > b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("member/2", _) ->
        let elem_ = Map.tryFind 1 s.WsRegs |> Option.defaultValue (Unbound -1) |> derefVar s.WsBindings
        let list_ = Map.tryFind 2 s.WsRegs |> Option.defaultValue (VList []) |> derefVar s.WsBindings
        match list_ with
        | VList (x :: _) -> unifyVal elem_ x s
        | _              -> None

    | BeginAggregate (aggType, valReg, resReg) ->
        let cp = { CpNextPC   = s.WsPC
                   CpRegs     = s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpAggFrame = Some { AggType = aggType; AggValReg = valReg
                                       AggResReg = resReg; AggReturnPC = 0 }
                   CpBuiltin  = None }
        Some { s with
                 WsPC       = s.WsPC + 1
                 WsCPs      = cp :: s.WsCPs
                 WsCPsLen   = s.WsCPsLen + 1
                 WsAggAccum = [] }

    | EndAggregate valReg ->
        let v = getReg valReg s |> Option.defaultValue (Integer 0) |> derefVar s.WsBindings
        let returnPC  = s.WsPC + 1
        let updatedCPs= updateNearestAggFrame returnPC s.WsCPs
        let s1 = { s with WsAggAccum = v :: s.WsAggAccum; WsCPs = updatedCPs }
        match backtrackInner returnPC s1 with
        | Some s2 -> Some s2
        | None    -> finalizeAggregate returnPC s1

    | BuiltinCall ("functor/3", _) ->
        let t = Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings)
        match t with
        | Some (Unbound vid) ->
            let nArg = Map.tryFind 2 s.WsRegs |> Option.map (derefVar s.WsBindings)
            let aArg = Map.tryFind 3 s.WsRegs |> Option.map (derefVar s.WsBindings)
            match nArg, aArg with
            | Some nameVal, Some (Integer arity) when arity >= 0 ->
                let mBuilt =
                    if arity = 0 then Some (nameVal, s.WsVarCounter)
                    else match nameVal with
                         | Atom fname ->
                             let c0    = s.WsVarCounter
                             let args  = [ for i in 0 .. arity - 1 -> Unbound (c0 + i) ]
                             Some (Str (fname, args), c0 + arity)
                         | _ -> None
                match mBuilt with
                | None -> None
                | Some (built, newCtr) ->
                    Some { s with
                             WsPC        = s.WsPC + 1
                             WsRegs      = Map.add 1 built s.WsRegs
                             WsBindings  = Map.add vid built s.WsBindings
                             WsTrail     = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                             WsTrailLen  = s.WsTrailLen + 1
                             WsVarCounter= newCtr }
            | _ -> None
        | Some tVal ->
            let mInfo = match tVal with
                        | Str (fn, args)  -> Some (Atom fn, List.length args)
                        | VList []        -> Some (Atom "[]", 0)
                        | VList _         -> Some (Atom ".", 2)
                        | Atom _          -> Some (tVal, 0)
                        | Integer _       -> Some (tVal, 0)
                        | Float _         -> Some (tVal, 0)
                        | _               -> None
            match mInfo with
            | None -> None
            | Some (name, arity) ->
                match bindOutput 2 name s with
                | None   -> None
                | Some s1 ->
                    match bindOutput 3 (Integer arity) s1 with
                    | None    -> None
                    | Some s2 -> Some { s2 with WsPC = s2.WsPC + 1 }
        | None -> None

    | BuiltinCall ("copy_term/2", _) ->
        match Map.tryFind 1 s.WsRegs |> Option.map (derefVar s.WsBindings) with
        | Some tVal ->
            let copy, newCtr, _ = copyTermWalk s.WsVarCounter Map.empty tVal
            let s0 = { s with WsVarCounter = newCtr }
            match bindOutput 2 copy s0 with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | None -> None

    | _ -> None   // fallback for unhandled instructions

and updateNearestAggFrame (rpc: int) (cps: ChoicePoint list) : ChoicePoint list =
    match cps with
    | [] -> []
    | cp :: rest ->
        match cp.CpAggFrame with
        | Some af -> { cp with CpAggFrame = Some { af with AggReturnPC = rpc } } :: rest
        | None    -> cp :: updateNearestAggFrame rpc rest

and unifyVal (a: Value) (b: Value) (s: WamState) : WamState option =
    match a, b with
    | Unbound vid, v ->
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsBindings= Map.add vid v s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | v, Unbound vid ->
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsBindings= Map.add vid v s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | x, y when x = y -> Some { s with WsPC = s.WsPC + 1 }
    | _                -> None'.

% ============================================================================
% PHASE 4: Run Loop
% ============================================================================

run_loop_fsharp(Code) :-
    Code = '/// Main execution loop. Runs until halt (pc=0) or failure.
/// WamContext is read-only. Tail-recursive via use of trampolining.
let rec run (ctx: WamContext) (s: WamState) : WamState option =
    if s.WsPC = 0 then Some s
    else
        let instr = ctx.WcCode.[s.WsPC]
        match step ctx s instr with
        | Some s2 -> run ctx s2
        | None    ->
            match backtrack s with
            | Some s2 -> run ctx s2
            | None    -> None

/// Indexed fact dispatch for 2-arg facts via BuiltinState CP.
/// O(1) Map lookup; first match returned, FactRetry CP for the rest.
and callIndexedFact2 (ctx: WamContext) (pred: string) (s: WamState) : WamState option =
    let basePred = pred |> Seq.takeWhile ((<>) '"'"'/'"'"') |> System.String.Concat
    let retPC    = s.WsCP
    match Map.tryFind basePred ctx.WcForeignFacts with
    | None -> None
    | Some factIndex ->
        let a1 = Map.tryFind 1 s.WsRegs |> Option.defaultValue (Atom "") |> derefVar s.WsBindings
        let a2 = Map.tryFind 2 s.WsRegs |> Option.defaultValue (Unbound -1) |> derefVar s.WsBindings
        match a1 with
        | Atom key ->
            match Map.tryFind key factIndex with
            | Some (v :: rest) ->
                match a2 with
                | Unbound vid ->
                    let newRegs     = Map.add 2 (Atom v) s.WsRegs
                    let newBindings = Map.add vid (Atom v) s.WsBindings
                    let newTrail    = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                    let newCPs, newCPsLen =
                        match rest with
                        | [] -> s.WsCPs, s.WsCPsLen
                        | _  ->
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
                            cp :: s.WsCPs, s.WsCPsLen + 1
                    Some { s with
                             WsPC      = retPC
                             WsRegs    = newRegs
                             WsBindings= newBindings
                             WsTrail   = newTrail
                             WsTrailLen= s.WsTrailLen + 1
                             WsCPs     = newCPs
                             WsCPsLen  = newCPsLen }
                | Atom existing ->
                    if existing = v then Some { s with WsPC = retPC }
                    elif List.contains existing rest then Some { s with WsPC = retPC }
                    else None
                | _ -> None
            | _ -> None
        | _ -> None

/// Dispatch a Call to another predicate for use by lowered functions.
and dispatchCall (ctx: WamContext) (pred: string) (sc: WamState) : WamState option =
    match Map.tryFind pred ctx.WcLoweredPredicates with
    | Some fn -> fn ctx sc
    | None ->
    match callIndexedFact2 ctx pred sc with
    | Some sr -> Some sr
    | None ->
    match Map.tryFind pred ctx.WcLabels with
    | Some pc -> run ctx { sc with WsPC = pc }
    | None    -> None

/// Foreign call for lowered functions.
and callForeign (ctx: WamContext) (pred: string) (sc: WamState) : WamState option =
    executeForeign ctx pred sc

{{execute_foreign}}

/// Resolve Call instructions at load time:
///   - Foreign predicates → CallForeign
///   - Known labels → CallResolved (direct PC)
///   - Others → left as Call (runtime dispatch)
let resolveCallInstrs (labels: Map<string, int>) (foreignPreds: string list) (instrs: Instruction list) : Instruction list =
    instrs |> List.map (fun instr ->
        match instr with
        | Call (pred, arity) when List.contains pred foreignPreds ->
            CallForeign (pred, arity)
        | Call (pred, arity) ->
            match Map.tryFind pred labels with
            | Some pc -> CallResolved (pc, arity)
            | None    -> Call (pred, arity)
        | Execute pred ->
            match Map.tryFind pred labels with
            | Some pc -> ExecutePc pc
            | None    -> Execute pred
        | Jump label ->
            match Map.tryFind label labels with
            | Some pc -> JumpPc pc
            | None    -> Jump label
        | TryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> TryMeElsePc pc
            | None    -> TryMeElse label
        | RetryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> RetryMeElsePc pc
            | None    -> RetryMeElse label
        | ParTryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> ParTryMeElsePc pc
            | None    -> ParTryMeElse label
        | ParRetryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> ParRetryMeElsePc pc
            | None    -> ParRetryMeElse label
        | SwitchOnConstant table ->
            let extractKey = function
                | Atom s  -> s
                | Integer n -> string n
                | v -> sprintf "%A" v
            let pcTable =
                table |> Map.toList
                      |> List.choose (fun (v, label) ->
                            Map.tryFind label labels |> Option.map (fun pc -> (extractKey v, pc)))
                      |> Map.ofList
            SwitchOnConstantPc pcTable
        | i -> i)'.

% ============================================================================
% compile_wam_runtime_to_fsharp/3
% ============================================================================

%% compile_wam_runtime_to_fsharp(+Options, +DetectedKernels, -Code)
compile_wam_runtime_to_fsharp(_Options, DetectedKernels, Code) :-
    step_function_fsharp(StepCode),
    backtrack_fsharp(BacktrackCode),
    run_loop_fsharp(RunLoopTemplate),
    generate_kernel_fsharp(DetectedKernels, KernelFunctionsCode, ExecuteForeignCode),
    render_template(RunLoopTemplate,
                    [execute_foreign=ExecuteForeignCode],
                    RunLoopCode),
    fsharp_wam_type_header(TypeHeader),
    format(string(Code),
'~w

// ============================================================================
// WAM Runtime
// ============================================================================

module WamRuntime

open WamTypes

~w

~w

~w
', [TypeHeader, StepCode, BacktrackCode, RunLoopCode]).

% ============================================================================
% Kernel code generation — F# analogs of the Haskell kernel helpers
% ============================================================================

generate_kernel_fsharp([], KF, EF) :- !,
    KF = "// No kernels detected.",
    EF = "and executeForeign (_ctx: WamContext) (_pred: string) (_s: WamState) : WamState option = None".
generate_kernel_fsharp(DetectedKernels, KernelFunctionsCode, ExecuteForeignCode) :-
    maplist(render_kernel_function_fs, DetectedKernels, KernelParts),
    atomic_list_concat(KernelParts, '\n\n', KernelFunctionsCode),
    generate_execute_foreign_fs(DetectedKernels, ExecuteForeignCode).

render_kernel_function_fs(Key-Kernel, Code) :-
    Kernel = recursive_kernel(Kind, _, ConfigOps),
    (   kernel_template_file(Kind, TemplateFile)
    ->  atom_concat('templates/targets/fsharp_wam/', TemplateFile, RelPath),
        (   source_file(wam_fsharp_target, SrcFile)
        ->  file_directory_name(SrcFile, SrcDir),
            file_directory_name(SrcDir, TargetsDir),
            file_directory_name(TargetsDir, UnifyWeaverDir),
            file_directory_name(UnifyWeaverDir, ProjectDir),
            atom_concat(ProjectDir, '/', P1),
            atom_concat(P1, RelPath, AbsPath)
        ;   AbsPath = RelPath
        ),
        (   exists_file(AbsPath)
        ->  read_file_to_string(AbsPath, Template, []),
            config_ops_to_template_vars_fs(ConfigOps, TemplateVars),
            render_template(Template, TemplateVars, Code0),
            atom_string(Code0, Code)
        ;   format(atom(Code), '// Kernel ~w: template not found at ~w', [Key, AbsPath])
        )
    ;   format(atom(Code), '// Kernel ~w: no F# template available', [Key])
    ).

config_ops_to_template_vars_fs([], []).
config_ops_to_template_vars_fs([Op|Rest], [Key=Value|RestVars]) :-
    Op =.. [Key, RawValue],
    (   RawValue = Pred/_ -> Value = Pred ; Value = RawValue ),
    config_ops_to_template_vars_fs(Rest, RestVars).

generate_execute_foreign_fs(DetectedKernels, Code) :-
    with_output_to(string(Code), (
        format("and executeForeign (ctx: WamContext) (pred: string) (s: WamState) : WamState option =~n"),
        format("    match pred with~n"),
        forall(member(KV, DetectedKernels), emit_execute_foreign_entry_fs(KV)),
        format("    | _ -> None~n")
    )).

emit_execute_foreign_entry_fs(Key-Kernel) :-
    Kernel = recursive_kernel(Kind, _, ConfigOps),
    (   kernel_register_layout(Kind, RegSpecs),
        kernel_native_call(Kind, CallSpec)
    ->  resolve_call_spec_fs(CallSpec, ConfigOps, ResolvedCallSpec),
        emit_ef_clause_fs(Key, RegSpecs, ResolvedCallSpec)
    ;   format('    // executeForeign: no metadata for ~w~n', [Key])
    ).

resolve_call_spec_fs(call(Func, Args), ConfigOps, call(Func, ResolvedArgs)) :-
    maplist(resolve_arg_spec_fs(ConfigOps), Args, ResolvedArgs).

resolve_arg_spec_fs(ConfigOps, config_facts_from(ConfigKey), config_facts(FactName)) :- !,
    Op =.. [ConfigKey, RawValue],
    member(Op, ConfigOps),
    (   RawValue = Pred/_ -> FactName = Pred ; FactName = RawValue ).
resolve_arg_spec_fs(ConfigOps, config_weighted_facts_from(ConfigKey), config_weighted_facts(FactName)) :- !,
    Op =.. [ConfigKey, RawValue],
    member(Op, ConfigOps),
    (   RawValue = Pred/_ -> FactName = Pred ; FactName = RawValue ).
resolve_arg_spec_fs(_, Arg, Arg).

emit_ef_clause_fs(Key, RegSpecs, call(FuncName, ArgSpecs)) :-
    format('    | "~w" ->~n', [Key]),
    include(is_input_reg_fs, RegSpecs, InputRegs),
    format('        let '),
    emit_input_let_bindings_fs(InputRegs, first),
    emit_config_let_bindings_fs(ArgSpecs),
    include(is_output_reg_fs, RegSpecs, OutputRegs),
    emit_case_and_call_fs(InputRegs, OutputRegs, FuncName, ArgSpecs),
    format('~n').

is_input_reg_fs(input(_, _)).
is_output_reg_fs(output(_, _)).

emit_input_let_bindings_fs([], _).
emit_input_let_bindings_fs([input(RegN, Type)|Rest], Pos) :-
    reg_var_name_fs(RegN, VarName),
    fsharp_wam_reg_default(Type, Default),
    (   Pos = first -> true ; format('        let ') ),
    format('~w = Map.tryFind ~w s.WsRegs |> Option.defaultValue (~w) |> derefVar s.WsBindings~n',
           [VarName, RegN, Default]),
    emit_input_let_bindings_fs(Rest, rest).

reg_var_name_fs(N, Name) :- format(atom(Name), 'r~w', [N]).

emit_config_let_bindings_fs([]).
emit_config_let_bindings_fs([config_facts(FactKey)|Rest]) :-
    format('        let ~w_facts = Map.tryFind "~w" ctx.WcFfiFacts |> Option.defaultValue Map.empty~n',
           [FactKey, FactKey]),
    emit_config_let_bindings_fs(Rest).
emit_config_let_bindings_fs([config_weighted_facts(FactKey)|Rest]) :-
    format('        let ~w_facts = Map.tryFind "~w" ctx.WcFfiWeightedFacts |> Option.defaultValue Map.empty~n',
           [FactKey, FactKey]),
    emit_config_let_bindings_fs(Rest).
emit_config_let_bindings_fs([config_int(ConfigKey, Default)|Rest]) :-
    format('        let ~w_cfg = Map.tryFind "~w" ctx.WcForeignConfig |> Option.defaultValue ~w~n',
           [ConfigKey, ConfigKey, Default]),
    emit_config_let_bindings_fs(Rest).
emit_config_let_bindings_fs([_|Rest]) :-
    emit_config_let_bindings_fs(Rest).

emit_case_and_call_fs(InputRegs, OutputRegs, FuncName, ArgSpecs) :-
    length(InputRegs, NInputs),
    (   NInputs =:= 1
    ->  InputRegs = [input(RegN1, _)],
        reg_var_name_fs(RegN1, ScrutName),
        format('        match ~w with~n', [ScrutName]),
        emit_single_case_branch_fs(InputRegs, OutputRegs, FuncName, ArgSpecs)
    ;   format('        match ('),
        emit_scrutinee_tuple_fs(InputRegs, first),
        format(') with~n'),
        format('        | ('),
        emit_pattern_tuple_fs(InputRegs, first),
        format(') ->~n'),
        emit_native_call_and_binding_fs(OutputRegs, FuncName, ArgSpecs, InputRegs, "            ")
    ),
    format('        | _ -> None~n').

emit_single_case_branch_fs([input(RegN, Type)|_], OutputRegs, FuncName, ArgSpecs) :-
    reg_var_name_fs(RegN, VarName),
    type_pattern_fs(Type, VarName, Pattern),
    format('        | ~w ->~n', [Pattern]),
    emit_native_call_and_binding_fs(OutputRegs, FuncName, ArgSpecs, [input(RegN, Type)], "            ").

emit_scrutinee_tuple_fs([], _).
emit_scrutinee_tuple_fs([input(RegN, _)|Rest], Pos) :-
    reg_var_name_fs(RegN, VarName),
    (   Pos = first -> true ; format(', ') ),
    format('~w', [VarName]),
    emit_scrutinee_tuple_fs(Rest, rest).

emit_pattern_tuple_fs([], _).
emit_pattern_tuple_fs([input(RegN, Type)|Rest], Pos) :-
    reg_var_name_fs(RegN, VarName),
    type_pattern_fs(Type, VarName, Pattern),
    (   Pos = first -> true ; format(', ') ),
    format('~w', [Pattern]),
    emit_pattern_tuple_fs(Rest, rest).

type_pattern_fs(atom,        VarName, Pattern) :- format(atom(Pattern), 'Atom ~wS', [VarName]).
type_pattern_fs(integer,     VarName, Pattern) :- format(atom(Pattern), 'Integer ~wI', [VarName]).
type_pattern_fs(vlist_atoms, VarName, Pattern) :- format(atom(Pattern), 'VList ~wL', [VarName]).

emit_native_call_and_binding_fs(OutputRegs, FuncName, ArgSpecs, InputRegs, Indent) :-
    format('~wlet results = ~w', [Indent, FuncName]),
    emit_call_args_fs(ArgSpecs, InputRegs),
    format('~n'),
    emit_stream_binding_multi_fs(OutputRegs, Indent).

emit_call_args_fs([], _).
emit_call_args_fs([Spec|Rest], InputRegs) :-
    format(' '),
    emit_one_call_arg_fs(Spec, InputRegs),
    emit_call_args_fs(Rest, InputRegs).

emit_one_call_arg_fs(config_facts(FactKey), _) :-
    format('~w_facts', [FactKey]).
emit_one_call_arg_fs(config_weighted_facts(FactKey), _) :-
    format('~w_facts', [FactKey]).
emit_one_call_arg_fs(config_int(ConfigKey, _), _) :-
    format('~w_cfg', [ConfigKey]).
emit_one_call_arg_fs(reg(RegN), InputRegs) :-
    member(input(RegN, Type), InputRegs),
    reg_var_name_fs(RegN, VarName),
    emit_reg_extraction_fs(VarName, Type).
emit_one_call_arg_fs(derived(length, RegN), InputRegs) :-
    member(input(RegN, _Type), InputRegs),
    reg_var_name_fs(RegN, VarName),
    format('(~wL |> List.choose (function Atom v -> Some v | _ -> None) |> List.length)', [VarName]).

emit_reg_extraction_fs(VarName, atom) :-
    format('(Map.tryFind ~wS ctx.WcAtomIntern |> Option.defaultValue -1)', [VarName]).
emit_reg_extraction_fs(VarName, vlist_atoms) :-
    format('(~wL |> List.choose (function Atom v -> Map.tryFind v ctx.WcAtomIntern | _ -> None))', [VarName]).
emit_reg_extraction_fs(VarName, integer) :-
    format('~wI', [VarName]).

emit_stream_binding_multi_fs(OutputRegs, Indent) :-
    length(OutputRegs, NOuts),
    format('~w    let retPC = s.WsCP~n', [Indent]),
    emit_multi_out_derefs_fs(OutputRegs, Indent),
    format('~w    let bindResult ', [Indent]),
    emit_tuple_pattern_fs(NOuts),
    format(' =~n', []),
    format('~w        let ', [Indent]),
    emit_multi_wrap_bindings_fs(OutputRegs, 1),
    format('~w        { s with WsPC = retPC~n', [Indent]),
    emit_multi_reg_updates_fs(OutputRegs, Indent),
    emit_multi_binding_updates_fs(OutputRegs, Indent),
    emit_multi_trail_updates_fs(OutputRegs, Indent),
    format('~w                 WsTrailLen = s.WsTrailLen + ~w }~n', [Indent, NOuts]),
    format('~w    match results with~n', [Indent]),
    format('~w    | [] -> None~n', [Indent]),
    format('~w    | [h] -> Some (bindResult h)~n', [Indent]),
    format('~w    | h :: restResults ->~n', [Indent]),
    format('~w        let s1 = bindResult h~n', [Indent]),
    emit_multi_outvars_fs(OutputRegs, Indent),
    format('~w        let restWrapped = restResults |> List.map (fun ', [Indent]),
    emit_tuple_pattern_fs(NOuts),
    format(' -> [', []),
    emit_multi_wrap_list_fs(OutputRegs, 1),
    format('])~n', []),
    format('~w        let cp = { CpNextPC   = retPC~n', [Indent]),
    format('~w                   CpRegs     = s.WsRegs~n', [Indent]),
    format('~w                   CpStack    = s.WsStack~n', [Indent]),
    format('~w                   CpCP       = s.WsCP~n', [Indent]),
    format('~w                   CpTrailLen = s.WsTrailLen~n', [Indent]),
    format('~w                   CpHeapLen  = s.WsHeapLen~n', [Indent]),
    format('~w                   CpBindings = s.WsBindings~n', [Indent]),
    format('~w                   CpCutBar   = s.WsCutBar~n', [Indent]),
    format('~w                   CpAggFrame = None~n', [Indent]),
    format('~w                   CpBuiltin  = Some (FFIStreamRetry ', [Indent]),
    emit_outregs_list_fs(OutputRegs),
    format(' outVars restWrapped retPC) }~n', []),
    format('~w        Some { s1 with WsCPs = cp :: s1.WsCPs; WsCPsLen = s1.WsCPsLen + 1 }~n', [Indent]).

emit_multi_out_derefs_fs([], _).
emit_multi_out_derefs_fs([output(RegN, _)|Rest], Indent) :-
    format('~w    let outReg_~w = Map.tryFind ~w s.WsRegs |> Option.defaultValue (Unbound -1) |> derefVar s.WsBindings~n',
           [Indent, RegN, RegN]),
    emit_multi_out_derefs_fs(Rest, Indent).

emit_tuple_pattern_fs(1) :- format('rv_1', []).
emit_tuple_pattern_fs(N) :-
    N > 1,
    format('(', []),
    emit_tuple_pattern_args_fs(1, N),
    format(')', []).

emit_tuple_pattern_args_fs(I, N) :-
    I < N, !,
    format('rv_~w, ', [I]),
    I1 is I + 1,
    emit_tuple_pattern_args_fs(I1, N).
emit_tuple_pattern_args_fs(N, N) :- format('rv_~w', [N]).

emit_multi_wrap_bindings_fs([], _) :- format('~n', []).
emit_multi_wrap_bindings_fs([output(_, Type)|Rest], I) :-
    fsharp_wam_result_wrap_rv(Type, I, WrapExpr),
    (   I =:= 1 -> format('w_~w = ~w', [I, WrapExpr])
    ;              format('; w_~w = ~w', [I, WrapExpr])
    ),
    I1 is I + 1,
    emit_multi_wrap_bindings_fs(Rest, I1).

emit_multi_reg_updates_fs(OutputRegs, Indent) :-
    format('~w                 WsRegs = ', [Indent]),
    emit_reg_add_chain_fs(OutputRegs, 1),
    format('s.WsRegs~n', []).

emit_reg_add_chain_fs([], _).
emit_reg_add_chain_fs([output(RegN, _)|Rest], I) :-
    format('Map.add ~w w_~w (', [RegN, I]),
    I1 is I + 1,
    emit_reg_add_chain_fs(Rest, I1),
    format(')', []).

emit_multi_binding_updates_fs(OutputRegs, Indent) :-
    format('~w                 WsBindings = ', [Indent]),
    emit_binding_add_chain_fs(OutputRegs, 1),
    format('s.WsBindings~n', []).

emit_binding_add_chain_fs([], _).
emit_binding_add_chain_fs([output(RegN, _)|Rest], I) :-
    format('(match outReg_~w with | Unbound v -> Map.add v w_~w | _ -> id) (', [RegN, I]),
    I1 is I + 1,
    emit_binding_add_chain_fs(Rest, I1),
    format(')', []).

emit_multi_trail_updates_fs(OutputRegs, Indent) :-
    format('~w                 WsTrail = ', [Indent]),
    emit_trail_entry_chain_fs(OutputRegs, 1),
    format('s.WsTrail~n', []).

emit_trail_entry_chain_fs([], _).
emit_trail_entry_chain_fs([output(RegN, _)|Rest], I) :-
    format('(match outReg_~w of | Unbound v -> List.cons { TrailVarId = v; TrailOldVal = Map.tryFind v s.WsBindings } | _ -> id) (', [RegN]),
    I1 is I + 1,
    emit_trail_entry_chain_fs(Rest, I1),
    format(')', []).

emit_multi_outvars_fs(OutputRegs, Indent) :-
    format('~w        let outVars = [', [Indent]),
    emit_outvars_list_fs(OutputRegs, 1),
    format(']~n', []).

emit_outvars_list_fs([], _).
emit_outvars_list_fs([output(RegN, _)|Rest], I) :-
    (   I =:= 1 -> true ; format('; ', []) ),
    format('match outReg_~w with | Unbound v -> v | _ -> -1', [RegN]),
    I1 is I + 1,
    emit_outvars_list_fs(Rest, I1).

emit_outregs_list_fs(OutputRegs) :-
    format('[', []),
    emit_outregs_list_items_fs(OutputRegs, 1),
    format(']', []).

emit_outregs_list_items_fs([], _).
emit_outregs_list_items_fs([output(RegN, _)|Rest], I) :-
    (   I =:= 1 -> true ; format('; ', []) ),
    format('~w', [RegN]),
    I1 is I + 1,
    emit_outregs_list_items_fs(Rest, I1).

emit_multi_wrap_list_fs([], _).
emit_multi_wrap_list_fs([output(_, Type)|Rest], I) :-
    fsharp_wam_result_wrap_rv(Type, I, WrapExpr),
    (   I =:= 1 -> true ; format('; ', []) ),
    format('~w', [WrapExpr]),
    I1 is I + 1,
    emit_multi_wrap_list_fs(Rest, I1).

% ============================================================================
% compile_wam_predicate_to_fsharp/4
% ============================================================================

%% compile_wam_predicate_to_fsharp(+PredIndicator, +WamCode, +Options, -Code)
compile_wam_predicate_to_fsharp(PredIndicator, WamCode, _Options, Code) :-
    (   PredIndicator = _M:Pred/Arity -> true ; PredIndicator = Pred/Arity ),
    format(atom(FuncName), '~w_~w', [Pred, Arity]),
    % Split WamCode into instruction lines and translate each
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    maplist(translate_wam_line_fs, Lines, FsLines),
    atomic_list_concat(FsLines, '\n', InstrBody),
    format(string(Code),
'// ~w/~w
let ~w_code : Instruction list =
    [ ~w ]

let ~w_labels : Map<string, int> =
    Map.ofList [ ("~w/~w", 1) ]
', [Pred, Arity, FuncName, InstrBody, FuncName, Pred, Arity]).

translate_wam_line_fs(Line, FsLine) :-
    (   wam_line_to_fsharp_instr(Line, Instr)
    ->  FsLine = Instr
    ;   format(atom(FsLine), '// ~w', [Line])
    ).

% Basic instruction string translations (sufficient for interpreter path)
wam_line_to_fsharp_instr(Line, Instr) :-
    string_concat("get_constant(", _, Line), !,
    format(atom(Instr), 'GetConstant ~~parsed~~ (* ~w *)', [Line]).
wam_line_to_fsharp_instr(Line, Instr) :-
    string_concat("proceed", _, Line), !,
    Instr = 'Proceed'.
wam_line_to_fsharp_instr(Line, Instr) :-
    string_concat("allocate", _, Line), !,
    Instr = 'Allocate'.
wam_line_to_fsharp_instr(Line, Instr) :-
    string_concat("deallocate", _, Line), !,
    Instr = 'Deallocate'.

% ============================================================================
% write_wam_fsharp_project/3 — Project Generation
% ============================================================================

%% write_wam_fsharp_project(+Predicates, +Options, +ProjectDir)
%  Generates a complete F# project with:
%  - WamTypes.fs:   Value DU, WamState, WamContext, helpers
%  - WamRuntime.fs: step, backtrack, run loop, executeForeign
%  - Predicates.fs: compiled predicates (instruction arrays)
%  - Lowered.fs:    lowered predicate functions (Phase 3+)
%  - Program.fs:    benchmark driver
%  - wam-fsharp-bench.fsproj: project file
write_wam_fsharp_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),

    % Kernel detection (skip with no_kernels(true))
    (   option(no_kernels(true), Options)
    ->  DetectedKernels = [],
        format(user_error, '[WAM-FSharp] kernel detection suppressed~n', [])
    ;   detect_kernels_fs(Predicates, DetectedKernels),
        (   DetectedKernels \= []
        ->  pairs_keys(DetectedKernels, DetectedKeys),
            format(user_error, '[WAM-FSharp] detected kernels: ~w~n', [DetectedKeys])
        ;   true
        )
    ),

    % Resolve emit mode and partition
    wam_fsharp_resolve_emit_mode(Options, EmitMode),
    wam_fsharp_partition_predicates(EmitMode, Predicates, DetectedKernels, InterpList, LoweredList),
    length(InterpList, NInterp),
    length(LoweredList, NLower),
    format(user_error, '[WAM-FSharp] emit_mode=~w  interpreted=~w  lowered=~w~n',
           [EmitMode, NInterp, NLower]),

    % Generate WamRuntime.fs (includes types via fsharp_wam_type_header)
    compile_wam_runtime_to_fsharp(Options, DetectedKernels, RuntimeCode),
    directory_file_path(ProjectDir, 'WamRuntime.fs', RuntimePath),
    write_fs_file(RuntimePath, RuntimeCode),

    % Generate Predicates.fs
    compile_predicates_to_fsharp(Predicates, Options, PredsCode),
    directory_file_path(ProjectDir, 'Predicates.fs', PredsPath),
    write_fs_file(PredsPath, PredsCode),

    % Generate Lowered.fs
    compute_base_pcs_fs(Predicates, BasePCMap),
    lower_all_fs(LoweredList, BasePCMap, DetectedKernels, LoweredEntries),
    generate_lowered_fs(LoweredEntries, LoweredCode),
    directory_file_path(ProjectDir, 'Lowered.fs', LoweredPath),
    write_fs_file(LoweredPath, LoweredCode),

    % Generate Program.fs (benchmark driver)
    option(module_name(ModName), Options, 'wam-fsharp-bench'),
    generate_program_fs(Predicates, DetectedKernels, Options, ProgramCode),
    directory_file_path(ProjectDir, 'Program.fs', ProgramPath),
    write_fs_file(ProgramPath, ProgramCode),

    % Generate .fsproj
    generate_fsproj(ModName, Options, FsprojCode),
    format(atom(FsprojFile), '~w.fsproj', [ModName]),
    directory_file_path(ProjectDir, FsprojFile, FsprojPath),
    write_fs_file(FsprojPath, FsprojCode),

    format(user_error, '[WAM-FSharp] Generated project at: ~w~n', [ProjectDir]).

%% write_fs_file(+Path, +Content)
write_fs_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream).

%% compile_predicates_to_fsharp(+Predicates, +Options, -Code)
compile_predicates_to_fsharp(Predicates, Options, Code) :-
    maplist(compile_one_predicate_fs(Options), Predicates, PredCodes),
    atomic_list_concat(PredCodes, '\n\n', AllPredCode),
    % Build merged code list and label map
    maplist(pred_func_name_fs, Predicates, FuncNames),
    emit_merged_code_build_fs(FuncNames, MergedCodeBuild),
    format(string(Code),
'module Predicates

open WamTypes
open WamRuntime

~w

~w
', [AllPredCode, MergedCodeBuild]).

compile_one_predicate_fs(Options, PredIndicator, Code) :-
    (   PredIndicator = _M:Pred/Arity -> true ; PredIndicator = Pred/Arity ),
    wam_fsharp_predicate_wamcode(PredIndicator, WamCode),
    compile_wam_predicate_to_fsharp(PredIndicator, WamCode, Options, Code).

pred_func_name_fs(PI, FN) :-
    (   PI = _M:P/A -> true ; PI = P/A ),
    format(atom(FN), '~w_~w', [P, A]).

emit_merged_code_build_fs([], Code) :-
    Code = 'let allCode : Instruction array = [||]\nlet allLabels : Map<string, int> = Map.empty'.
emit_merged_code_build_fs(FuncNames, Code) :-
    FuncNames \= [],
    maplist([FN, Expr]>>(format(atom(Expr), '~w_code', [FN])), FuncNames, CodeExprs),
    atomic_list_concat(CodeExprs, ' @ ', CodeConcat),
    maplist([FN, Expr]>>(format(atom(Expr), '~w_labels', [FN])), FuncNames, LabelExprs),
    atomic_list_concat(LabelExprs, ' |> Map.union ', LabelUnion),
    format(string(Code),
'let allCode : Instruction array = (~w) |> List.toArray
let allLabels : Map<string, int> = ~w', [CodeConcat, LabelUnion]).

%% generate_lowered_fs(+LoweredEntries, -Code)
generate_lowered_fs([], Code) :- !,
    Code = 'module Lowered

open WamTypes
open WamRuntime

// No predicates lowered.
let loweredPredicates : Map<string, WamContext -> WamState -> WamState option> =
    Map.empty'.
generate_lowered_fs(LoweredEntries, Code) :-
    LoweredEntries \= [],
    with_output_to(string(FuncBodies), (
        forall(member(lowered(_, _, FsCode), LoweredEntries),
               format('~w~n', [FsCode]))
    )),
    with_output_to(string(MapEntries), (
        format('    Map.ofList~n'),
        format('        [ '),
        emit_lowered_entries_fs(LoweredEntries),
        format('        ]~n')
    )),
    format(string(Code),
'module Lowered

open WamTypes
open WamRuntime

~w
let loweredPredicates : Map<string, WamContext -> WamState -> WamState option> =
~w', [FuncBodies, MapEntries]).

emit_lowered_entries_fs([lowered(PredName, FuncName, _)|Rest]) :-
    format('("~w", ~w)~n', [PredName, FuncName]),
    emit_lowered_entries_rest_fs(Rest).
emit_lowered_entries_rest_fs([]).
emit_lowered_entries_rest_fs([lowered(PredName, FuncName, _)|Rest]) :-
    format('          ; ("~w", ~w)~n', [PredName, FuncName]),
    emit_lowered_entries_rest_fs(Rest).

%% generate_program_fs(+Predicates, +DetectedKernels, +Options, -Code)
generate_program_fs(_Predicates, DetectedKernels, Options, Code) :-
    pairs_keys(DetectedKernels, ForeignKeys),
    format_foreign_preds_fs(ForeignKeys, ForeignPredsStr),
    option(module_name(_ModName), Options, 'wam-fsharp-bench'),
    format(string(Code),
'module Program

open System
open WamTypes
open WamRuntime
open Predicates
open Lowered

[<EntryPoint>]
let main _argv =
    // Resolve call instructions (pre-resolve labels to PCs at load time)
    let foreignPreds = [ ~w ]
    let resolvedCode = resolveCallInstrs allLabels foreignPreds (Array.toList allCode) |> List.toArray

    let ctx =
        { WcCode             = resolvedCode
          WcLabels            = allLabels
          WcForeignFacts      = Map.empty   // populate from your dataset
          WcFfiFacts          = Map.empty
          WcFfiWeightedFacts  = Map.empty
          WcAtomIntern        = Map.empty
          WcAtomDeintern      = Map.empty
          WcForeignConfig     = Map.empty
          WcLoweredPredicates = loweredPredicates }

    let emptyState =
        { WsPC         = 0
          WsRegs       = Map.empty
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

    // Example: run a query. Adapt startPC and initial registers to your predicate.
    let startPC = Map.tryFind "main/0" allLabels |> Option.defaultValue 1
    let s0 = { emptyState with WsPC = startPC }
    match run ctx s0 with
    | Some sf -> printfn "Query succeeded. PC=%d" sf.WsPC
    | None    -> printfn "Query failed."
    0
', [ForeignPredsStr]).

format_foreign_preds_fs([], '').
format_foreign_preds_fs(Keys, Str) :-
    Keys \= [],
    maplist([Key, Q]>>(format(atom(Q), '"~w"', [Key])), Keys, Quoted),
    atomic_list_concat(Quoted, '; ', Str).

%% generate_fsproj(+ModName, +Options, -Code)
generate_fsproj(ModName, _Options, Code) :-
    format(string(Code),
'<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AssemblyName>~w</AssemblyName>
    <Optimize>true</Optimize>
    <Nullable>enable</Nullable>
    <Deterministic>true</Deterministic>
  </PropertyGroup>

  <ItemGroup>
    <Compile Include="WamRuntime.fs" />
    <Compile Include="Predicates.fs" />
    <Compile Include="Lowered.fs" />
    <Compile Include="Program.fs" />
  </ItemGroup>

</Project>
', [ModName]).
