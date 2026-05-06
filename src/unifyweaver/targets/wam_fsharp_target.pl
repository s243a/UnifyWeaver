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
%   - Value array for registers → O(1) array access with Array.copy snapshots
%     for choice points (same insight as Go's [512]Value)
%   - WamContext separated from WamState (hot/cold split — Haskell Phase-B)
%   - Labels pre-resolved to int PCs at load time (Phase-C win: -47%)
%   - Skip WAM-compilation of FFI-owned fact predicates (Phase-D: -70%)
%   - Atom interning at FFI boundary (Phase-D: -48% query time)
%     Implemented via buildAtomInternTable in generated Program.fs
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
% FFI-owned fact detection (Phase D: skip WAM-compilation for pure FFI facts)
%
% A predicate is FFI-owned if:
%   1. It was detected as a recursive kernel (has FFI bindings)
%   2. All its clauses are pure facts (body = true)
%
% Skipping WAM compilation of these predicates was the -70% total query
% time win in Haskell/Go — the FFI kernel path handles them directly.
% ============================================================================

%% is_ffi_owned_fact_fs(+PredIndicator, +DetectedKernels)
is_ffi_owned_fact_fs(PI, DetectedKernels) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    member(Key-_, DetectedKernels),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    forall(member(_-Body, Clauses), Body == true).

%% ffi_owned_fact_filter_fs(+DetectedKernels, +PI) — exclude/3 callback
ffi_owned_fact_filter_fs(DetectedKernels, PI) :-
    is_ffi_owned_fact_fs(PI, DetectedKernels).

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
'    let valOpt = getReg ~w s
    match valOpt with
    | Some v when v = ~w -> Some { s with WsPC = s.WsPC + 1 }
    | Some (Unbound vid) ->
        let r = Array.copy s.WsRegs
        r.[~w] <- ~w
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsRegs    = r
                 WsBindings= Map.add vid ~w s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | _ -> None', [Ai, C, Ai, C, C]).

wam_to_fsharp(get_variable(Xn, Ai), Code) :-
    format(string(Code),
'    match getReg ~w s with
    | Some dv -> Some (putReg ~w dv { s with WsPC = s.WsPC + 1 })
    | None    -> None', [Ai, Xn]).

wam_to_fsharp(get_value(Xn, Ai), Code) :-
    format(string(Code),
'    let va = getReg ~w s
    let vx = getReg ~w s
    match va, vx with
    | Some a, Some x when a = x -> Some { s with WsPC = s.WsPC + 1 }
    | Some (Unbound vid), Some x ->
        let r = Array.copy s.WsRegs
        r.[~w] <- x
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsRegs    = r
                 WsBindings= Map.add vid x s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | _ -> None', [Ai, Xn, Ai]).

wam_to_fsharp(put_constant(C, Ai), Code) :-
    format(string(Code),
'    let r = Array.copy s.WsRegs
    r.[~w] <- ~w
    Some { s with WsPC = s.WsPC + 1; WsRegs = r }', [Ai, C]).

wam_to_fsharp(put_variable(Xn, Ai), Code) :-
    format(string(Code),
'    let vid = s.WsVarCounter
    let var = Unbound vid
    let s1  = putReg ~w var s
    let r   = Array.copy s1.WsRegs
    r.[~w] <- var
    Some { s1 with
             WsPC       = s.WsPC + 1
             WsRegs     = r
             WsVarCounter= s.WsVarCounter + 1 }', [Xn, Ai]).

wam_to_fsharp(put_value(Xn, Ai), Code) :-
    format(string(Code),
'    match getReg ~w s with
    | Some v ->
        let r = Array.copy s.WsRegs
        r.[~w] <- v
        Some { s with WsPC = s.WsPC + 1; WsRegs = r }
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
               CpRegs     = Array.copy s.WsRegs
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
    Code = '    let expr   = s.WsRegs.[2] |> derefVar s.WsBindings
    let result = evalArith s.WsBindings expr
    let lhs    = getReg 1 s
    match lhs, result with
    | Some (Unbound vid), Some r ->
        let v = if float (int r) = r then Integer (int r) else Float r
        let regs = Array.copy s.WsRegs
        regs.[1] <- v
        Some { s with
                 WsPC      = s.WsPC + 1
                 WsRegs    = regs
                 WsBindings= Map.add vid v s.WsBindings
                 WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                 WsTrailLen= s.WsTrailLen + 1 }
    | Some (Integer n), Some r when float n = r -> Some { s with WsPC = s.WsPC + 1 }
    | Some (Float f),   Some r when f = r       -> Some { s with WsPC = s.WsPC + 1 }
    | _ -> None'.

wam_to_fsharp(builtin_call('</2', 2), Code) :-
    Code = '    let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
    let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
    match v1, v2 with
    | Some a, Some b when a < b -> Some { s with WsPC = s.WsPC + 1 }
    | _ -> None'.

wam_to_fsharp(builtin_call('>/2', 2), Code) :-
    Code = '    let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
    let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
    match v1, v2 with
    | Some a, Some b when a > b -> Some { s with WsPC = s.WsPC + 1 }
    | _ -> None'.

wam_to_fsharp(builtin_call('\\+/1', 1), Code) :-
    Code = '    let goal = Some s.WsRegs.[1] |> Option.bind (derefHeap s.WsHeap)
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
    Code = '    let listVal = s.WsRegs.[1] |> derefVar s.WsBindings
    match listVal with
    | VList items ->
        let len = List.length items
        let lhs = getReg 2 s
        match lhs with
        | Some (Unbound vid) ->
            let v = Integer len
            let regs = Array.copy s.WsRegs
            regs.[2] <- v
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = regs
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
and backtrack (s: WamState) : WamState option =
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
        let newRegs     = Array.copy cp.CpRegs
        newRegs.[2]    <- Atom v
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
        let newRegs     = Array.copy cp.CpRegs
        newRegs.[3]    <- Integer h
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
        let newRegs     = Array.copy cp.CpRegs
        List.iter2 (fun rN v -> newRegs.[rN] <- v) outRegs tuple
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
                        let r = Array.copy cp.CpRegs
                        r.[af.AggResReg] <- result
                        ( r
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
        let valOpt = getReg ai s
        match valOpt with
        | Some v when v = c -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Unbound vid) ->
            let r = Array.copy s.WsRegs
            r.[ai] <- c
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = r
                     WsBindings= Map.add vid c s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | _ -> None

    | GetVariable (xn, ai) ->
        match getReg ai s with
        | Some dv -> Some (putReg xn dv { s with WsPC = s.WsPC + 1 })
        | None    -> None

    | GetValue (xn, ai) ->
        let va = getReg ai s
        let vx = getReg xn s
        match va, vx with
        | Some a, Some x when a = x -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Unbound vid), Some x ->
            let r = Array.copy s.WsRegs
            r.[ai] <- x
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = r
                     WsBindings= Map.add vid x s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | _ -> None

    | GetStructure (fn, arity, ai) ->
        match getReg ai s with
        | Some (Str (fn0, args)) when fn0 = fn && List.length args = arity ->
            if arity = 0 then Some { s with WsPC = s.WsPC + 1; WsBuilder = None }
            else Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (ReadArgs args) }
        | Some (Unbound vid) when arity = 0 ->
            let str = Str (fn, [])
            let r = Array.copy s.WsRegs
            r.[ai] <- str
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = r
                     WsBindings= Map.add vid str s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1
                     WsBuilder = None }
        | Some (Unbound _) ->
            Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildStruct (fn, ai, arity, [])) }
        | _ -> None

    | GetList ai ->
        match getReg ai s with
        | Some (VList (h :: t)) ->
            Some { s with WsPC = s.WsPC + 1
                           WsBuilder = Some (ReadArgs [h; VList t]) }
        | Some (Unbound _) ->
            Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildList (ai, [])) }
        | _ -> None

    | UnifyVariable xn ->
        match readNextArg s with
        | Some (v, s1) -> Some (putReg xn v { s1 with WsPC = s.WsPC + 1 })
        | None         -> None

    | UnifyValue xn ->
        match readNextArg s, getReg xn s with
        | Some (v, s1), Some x -> unifyVal v x s1
        | _                    -> None

    | UnifyConstant c ->
        match readNextArg s with
        | Some (v, s1) -> unifyVal v c s1
        | None         -> None

    | PutConstant (c, ai) ->
        let r = Array.copy s.WsRegs
        r.[ai] <- c
        Some { s with WsPC = s.WsPC + 1; WsRegs = r }

    | PutVariable (xn, ai) ->
        let vid = s.WsVarCounter
        let var = Unbound vid
        let s1  = putReg xn var s
        let r   = Array.copy s1.WsRegs
        r.[ai] <- var
        Some { s1 with
                 WsPC        = s.WsPC + 1
                 WsRegs      = r
                 WsVarCounter= s.WsVarCounter + 1 }

    | PutValue (xn, ai) ->
        match getReg xn s with
        | Some v ->
            let r = Array.copy s.WsRegs
            r.[ai] <- v
            Some { s with WsPC = s.WsPC + 1; WsRegs = r }
        | None   -> None

    | PutStructure (fn, ai, arity) ->
        Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildStruct (fn, ai, arity, [])) }

    | PutList ai ->
        Some { s with WsPC = s.WsPC + 1; WsBuilder = Some (BuildList (ai, [])) }

    | SetValue xn ->
        match getReg xn s with
        | Some v -> addToBuilder v s
        | None   -> None

    | SetVariable xn ->
        let vid = s.WsVarCounter
        let var = Unbound vid
        let s1 = putReg xn var { s with WsVarCounter = s.WsVarCounter + 1 }
        addToBuilder var s1

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

    | Fail -> None

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
                   CpRegs     = Array.copy s.WsRegs
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
                   CpRegs     = Array.copy s.WsRegs
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
        let valOpt = getReg 1 s
        match valOpt with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Atom key) ->
            match binarySearchStr table key with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> None
        | Some (Integer n) ->
            match binarySearchStr table (string n) with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> None
        | _ -> None

    | SwitchOnConstant table ->
        let valOpt = getReg 1 s
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
        match getReg 1 s with
        | Some (Unbound _) -> None
        | Some _           -> Some { s with WsPC = s.WsPC + 1 }
        | None             -> None

    | BuiltinCall ("var/1", _) ->
        match getReg 1 s with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                -> None

    | BuiltinCall ("atom/1", _) ->
        match getReg 1 s with
        | Some (Atom _) -> Some { s with WsPC = s.WsPC + 1 }
        | _             -> None

    | BuiltinCall ("integer/1", _) ->
        match getReg 1 s with
        | Some (Integer _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                -> None

    | BuiltinCall ("number/1", _) ->
        match getReg 1 s with
        | Some (Integer _) | Some (Float _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                                  -> None

    | BuiltinCall ("is/2", _) ->
        let expr   = s.WsRegs.[2] |> derefVar s.WsBindings
        let result = evalArith s.WsBindings expr
        let lhs    = getReg 1 s
        match lhs, result with
        | Some (Unbound vid), Some r ->
            let v = if float (int r) = r then Integer (int r) else Float r
            let regs = Array.copy s.WsRegs
            regs.[1] <- v
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = regs
                     WsBindings= Map.add vid v s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | Some (Integer n), Some r when float n = r -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("length/2", _) ->
        let listVal = s.WsRegs.[1] |> derefVar s.WsBindings
        match listVal with
        | VList items ->
            let len = List.length items
            match getReg 2 s with
            | Some (Unbound vid) ->
                let v = Integer len
                let regs = Array.copy s.WsRegs
                regs.[2] <- v
                Some { s with
                         WsPC      = s.WsPC + 1
                         WsRegs    = regs
                         WsBindings= Map.add vid v s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
            | Some (Integer n) when n = len -> Some { s with WsPC = s.WsPC + 1 }
            | _ -> None
        | _ -> None

    | BuiltinCall ("</2", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a < b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall (">/2", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a > b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("member/2", _) ->
        let elem_ = s.WsRegs.[1] |> derefVar s.WsBindings
        let list_ = s.WsRegs.[2] |> derefVar s.WsBindings
        match list_ with
        | VList (x :: _) -> unifyVal elem_ x s
        | _              -> None

    | BeginAggregate (aggType, valReg, resReg) ->
        let cp = { CpNextPC   = s.WsPC
                   CpRegs     = Array.copy s.WsRegs
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
        let t = getReg 1 s
        match t with
        | Some (Unbound vid) ->
            let nArg = getReg 2 s
            let aArg = getReg 3 s
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
                    let regs = Array.copy s.WsRegs
                    regs.[1] <- built
                    Some { s with
                             WsPC        = s.WsPC + 1
                             WsRegs      = regs
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
        match getReg 1 s with
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
and run (ctx: WamContext) (s: WamState) : WamState option =
    if s.WsPC = 0 then Some s
    else
        let instr = ctx.WcCode.[s.WsPC]
        match step ctx s instr with
        | Some s2 -> run ctx s2
        | None    ->
            match backtrack s with
            | Some s2 -> run ctx s2
            | None    -> None

/// Run multiple seed states in parallel using TPL (System.Threading.Tasks.Parallel).
/// WamContext is read-only and safely shared across threads.
/// Each seed gets its own WamState copy — no shared mutable state.
and runParallel (ctx: WamContext) (seeds: WamState list) : WamState option list =
    seeds
    |> List.toArray
    |> Array.Parallel.map (fun seed -> run ctx seed)
    |> Array.toList

/// Indexed fact dispatch for 2-arg facts via BuiltinState CP.
/// O(1) Map lookup; first match returned, FactRetry CP for the rest.
and callIndexedFact2 (ctx: WamContext) (pred: string) (s: WamState) : WamState option =
    let basePred = pred |> Seq.takeWhile ((<>) ''/'') |> System.String.Concat
    let retPC    = s.WsCP
    match Map.tryFind basePred ctx.WcForeignFacts with
    | None -> None
    | Some factIndex ->
        let a1 = s.WsRegs.[1] |> derefVar s.WsBindings
        let a2 = s.WsRegs.[2] |> derefVar s.WsBindings
        match a1 with
        | Atom key ->
            match Map.tryFind key factIndex with
            | Some (v :: rest) ->
                match a2 with
                | Unbound vid ->
                    let newRegs     = Array.copy s.WsRegs
                    newRegs.[2]    <- Atom v
                    let newBindings = Map.add vid (Atom v) s.WsBindings
                    let newTrail    = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                    let newCPs, newCPsLen =
                        match rest with
                        | [] -> s.WsCPs, s.WsCPsLen
                        | _  ->
                            let cp = { CpNextPC   = retPC
                                       CpRegs     = Array.copy s.WsRegs
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
                      |> List.sortBy fst
                      |> List.toArray
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
    format(string(Code),
'module WamRuntime

open WamTypes

~w

~w

~w

~w
', [KernelFunctionsCode, StepCode, BacktrackCode, RunLoopCode]).

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
    (   kernel_template_file(Kind, HsTemplateFile),
        % Replace .hs.mustache with .fs.mustache for F# target
        atom_concat(Base, '.hs.mustache', HsTemplateFile),
        atom_concat(Base, '.fs.mustache', TemplateFile)
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
    format('~w = (let rv = s.WsRegs.[~w] in match rv with Unbound -1 -> ~w | _ -> rv) |> derefVar s.WsBindings~n',
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
    format('~wlet retPC = s.WsCP~n', [Indent]),
    emit_multi_out_derefs_fs(OutputRegs, Indent),
    format('~wlet bindResult ', [Indent]),
    emit_tuple_pattern_fs(NOuts),
    format(' =~n', []),
    format('~w    let ', [Indent]),
    emit_multi_wrap_bindings_fs(OutputRegs, 1),
    format('~w    { s with WsPC = retPC~n', [Indent]),
    emit_multi_reg_updates_fs(OutputRegs, Indent),
    emit_multi_binding_updates_fs(OutputRegs, Indent),
    emit_multi_trail_updates_fs(OutputRegs, Indent),
    format('~w             WsTrailLen = s.WsTrailLen + ~w }~n', [Indent, NOuts]),
    format('~wmatch results with~n', [Indent]),
    format('~w| [] -> None~n', [Indent]),
    format('~w| [h] -> Some (bindResult h)~n', [Indent]),
    format('~w| h :: restResults ->~n', [Indent]),
    format('~w    let s1 = bindResult h~n', [Indent]),
    emit_multi_outvars_fs(OutputRegs, Indent),
    format('~w    let restWrapped = restResults |> List.map (fun ', [Indent]),
    emit_tuple_pattern_fs(NOuts),
    format(' -> [', []),
    emit_multi_wrap_list_fs(OutputRegs, 1),
    format('])~n', []),
    format('~w    let cp = { CpNextPC   = retPC~n', [Indent]),
    format('~w               CpRegs     = Array.copy s.WsRegs~n', [Indent]),
    format('~w               CpStack    = s.WsStack~n', [Indent]),
    format('~w               CpCP       = s.WsCP~n', [Indent]),
    format('~w               CpTrailLen = s.WsTrailLen~n', [Indent]),
    format('~w               CpHeapLen  = s.WsHeapLen~n', [Indent]),
    format('~w               CpBindings = s.WsBindings~n', [Indent]),
    format('~w               CpCutBar   = s.WsCutBar~n', [Indent]),
    format('~w               CpAggFrame = None~n', [Indent]),
    format('~w               CpBuiltin  = Some (FFIStreamRetry (', [Indent]),
    emit_outregs_list_fs(OutputRegs),
    format(', outVars, restWrapped, retPC)) }~n', []),
    format('~w    Some { s1 with WsCPs = cp :: s1.WsCPs; WsCPsLen = s1.WsCPsLen + 1 }~n', [Indent]).

emit_multi_out_derefs_fs([], _).
emit_multi_out_derefs_fs([output(RegN, _)|Rest], Indent) :-
    format('~wlet outReg_~w = s.WsRegs.[~w] |> derefVar s.WsBindings~n',
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
    format('~w             WsRegs = (let r = Array.copy s.WsRegs in ', [Indent]),
    emit_reg_set_chain_fs(OutputRegs, 1),
    format('r)~n', []).

emit_reg_set_chain_fs([], _).
emit_reg_set_chain_fs([output(RegN, _)|Rest], I) :-
    format('r.[~w] <- w_~w; ', [RegN, I]),
    I1 is I + 1,
    emit_reg_set_chain_fs(Rest, I1).

emit_multi_binding_updates_fs(OutputRegs, Indent) :-
    format('~w             WsBindings = ', [Indent]),
    emit_binding_add_chain_fs(OutputRegs, 1),
    format('s.WsBindings~n', []).

emit_binding_add_chain_fs([], _).
emit_binding_add_chain_fs([output(RegN, _)|Rest], I) :-
    format('(match outReg_~w with | Unbound v -> Map.add v w_~w | _ -> id) ', [RegN, I]),
    I1 is I + 1,
    (   Rest = []
    ->  true
    ;   format('(', []),
        emit_binding_add_chain_fs(Rest, I1),
        format(')', [])
    ).

emit_multi_trail_updates_fs(OutputRegs, Indent) :-
    format('~w             WsTrail = ', [Indent]),
    emit_trail_entry_chain_fs(OutputRegs, 1),
    format('s.WsTrail~n', []).

emit_trail_entry_chain_fs([], _).
emit_trail_entry_chain_fs([output(RegN, _)|Rest], I) :-
    format('(match outReg_~w with | Unbound v -> (fun tl -> { TrailVarId = v; TrailOldVal = Map.tryFind v s.WsBindings } :: tl) | _ -> id) ', [RegN]),
    I1 is I + 1,
    (   Rest = []
    ->  true
    ;   format('(', []),
        emit_trail_entry_chain_fs(Rest, I1),
        format(')', [])
    ).

emit_multi_outvars_fs(OutputRegs, Indent) :-
    format('~w    let outVars = [', [Indent]),
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
%  Converts WAM assembly text for a single predicate into F# source defining
%  an Instruction list and a labels map.  Mirrors wam_haskell_target's
%  compile_wam_predicate_to_haskell/4 but emits F# discriminated-union syntax.
compile_wam_predicate_to_fsharp(PredIndicator, WamCode, Options, Code) :-
    (   PredIndicator = _M:Pred/Arity -> true ; PredIndicator = Pred/Arity ),
    (   string(WamCode) -> WamStr = WamCode ; atom_string(WamCode, WamStr) ),
    (   member(base_pc(BasePC), Options) -> true ; BasePC = 1 ),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_fsharp(Lines, BasePC, InstrExprs, LabelExprs),
    atomic_list_concat(InstrExprs, '\n      ; ', InstrCode),
    atomic_list_concat(LabelExprs, '; ', LabelCode),
    atom_string(Pred, PredStr),
    atomic_list_concat(PredParts, '$', PredStr),
    atomic_list_concat(PredParts, '_', PredSafe),
    format(atom(FuncName), '~w_~w', [PredSafe, Arity]),
    format(string(Code),
'// ~w/~w
let ~w_code : Instruction list =
    [ ~w
    ]

let ~w_labels : Map<string, int> =
    Map.ofList [ ~w ]
', [Pred, Arity, FuncName, InstrCode, FuncName, LabelCode]).

%% wam_lines_to_fsharp(+Lines, +PC, -InstrExprs, -LabelExprs)
wam_lines_to_fsharp([], _, [], []).
wam_lines_to_fsharp([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_fsharp(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LExpr), '("~w", ~w)', [LabelName, PC]),
            Labels = [LExpr|LR],
            wam_lines_to_fsharp(Rest, PC, Instrs, LR)
        ;   wam_instr_to_fsharp(CleanParts, FsExpr),
            NPC is PC + 1,
            Instrs = [FsExpr|IR],
            wam_lines_to_fsharp(Rest, NPC, IR, Labels)
        )
    ).

%% fs_reg_name_to_int(+RegName, -Int)
%  A1-A99 → 1-99, X1-X99 → 101-199, Y1-Y99 → 201-299.
fs_reg_name_to_int(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, Bank),
    sub_atom(RegA, 1, _, 0, NumA),
    atom_number(NumA, Num),
    (   Bank == 'A' -> Int = Num
    ;   Bank == 'X' -> Int is Num + 100
    ;   Bank == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).

%% fs_clean_comma(+Str, -Clean) — strip trailing comma
fs_clean_comma(Str, Clean) :-
    (   sub_string(Str, _, 1, 0, ",")
    ->  sub_string(Str, 0, _, 1, Clean)
    ;   Clean = Str
    ).

%% fs_wam_value(+WamVal, -FsExpr)
fs_wam_value(Val, Fs) :-
    (   number_string(N, Val), integer(N)
    ->  format(string(Fs), 'Integer ~w', [N])
    ;   number_string(F, Val), float(F)
    ->  format(string(Fs), 'Float ~w', [F])
    ;   Val == "[]"
    ->  Fs = "Atom \"[]\""
    ;   format(string(Fs), 'Atom "~w"', [Val])
    ).

%% fs_parse_functor(+FunctorString, -Name, -Arity)
%  Extract name and arity from "name/N" format.  E.g. "-/1" → ("-", 1).
fs_parse_functor(FN, Name, Arity) :-
    atom_string(FNA, FN),
    (   sub_atom(FNA, Before, 1, _, '/'),
        sub_atom(FNA, 0, Before, _, Name),
        After is Before + 1,
        sub_atom(FNA, After, _, 0, ArityStr),
        atom_number(ArityStr, Arity)
    ->  true
    ;   Name = FNA, Arity = 0
    ).

%% fs_escape_string(+In, -Out) — escape backslashes for F# string literals
fs_escape_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", Out).

%% wam_instr_to_fsharp(+Parts, -FSharpExpr)
wam_instr_to_fsharp(["get_constant", C, Ai], Fs) :-
    fs_clean_comma(C, CC), fs_clean_comma(Ai, CAi),
    fs_wam_value(CC, FsVal), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetConstant (~w, ~w)', [FsVal, AiI]).
wam_instr_to_fsharp(["get_variable", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetVariable (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["get_value", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetValue (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["get_structure", FN, Arity, Ai], Fs) :-
    fs_clean_comma(FN, CFN), fs_clean_comma(Arity, CA), fs_clean_comma(Ai, CAi),
    (   number_string(ANum, CA) -> true ; throw(error(domain_error(wam_integer, CA), wam_instr_to_fsharp/2)) ),
    fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetStructure ("~w", ~w, ~w)', [CFN, ANum, AiI]).
wam_instr_to_fsharp(["get_list", Ai], Fs) :-
    fs_clean_comma(Ai, CAi), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetList ~w', [AiI]).
wam_instr_to_fsharp(["get_nil", Ai], Fs) :-
    fs_clean_comma(Ai, CAi), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetConstant (Atom "[]", ~w)', [AiI]).
wam_instr_to_fsharp(["get_integer", N, Ai], Fs) :-
    fs_clean_comma(N, CN), fs_clean_comma(Ai, CAi),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetConstant (Integer ~w, ~w)', [Num, AiI]).
wam_instr_to_fsharp(["unify_variable", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'UnifyVariable ~w', [XnI]).
wam_instr_to_fsharp(["unify_value", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'UnifyValue ~w', [XnI]).
wam_instr_to_fsharp(["unify_constant", C], Fs) :-
    fs_wam_value(C, FsVal),
    format(string(Fs), 'UnifyConstant (~w)', [FsVal]).
wam_instr_to_fsharp(["put_constant", C, Ai], Fs) :-
    fs_clean_comma(C, CC), fs_clean_comma(Ai, CAi),
    fs_wam_value(CC, FsVal), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutConstant (~w, ~w)', [FsVal, AiI]).
wam_instr_to_fsharp(["put_variable", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutVariable (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["put_value", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutValue (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["put_structure", FN, Ai], Fs) :-
    fs_clean_comma(FN, CFN), fs_clean_comma(Ai, CAi),
    fs_parse_functor(CFN, FuncName, FsArity),
    fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutStructure ("~w", ~w, ~w)', [FuncName, AiI, FsArity]).
wam_instr_to_fsharp(["put_list", Ai], Fs) :-
    fs_clean_comma(Ai, CAi), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutList ~w', [AiI]).
wam_instr_to_fsharp(["set_value", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'SetValue ~w', [XnI]).
wam_instr_to_fsharp(["set_variable", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'SetVariable ~w', [XnI]).
wam_instr_to_fsharp(["set_constant", C], Fs) :-
    fs_wam_value(C, FsVal),
    format(string(Fs), 'SetConstant (~w)', [FsVal]).
wam_instr_to_fsharp(["allocate"], "Allocate").
wam_instr_to_fsharp(["deallocate"], "Deallocate").
wam_instr_to_fsharp(["call", P, N], Fs) :-
    fs_clean_comma(P, CP), fs_clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    format(string(Fs), 'Call ("~w", ~w)', [CP, Num]).
wam_instr_to_fsharp(["call_foreign", P, N], Fs) :-
    fs_clean_comma(P, CP), fs_clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    format(string(Fs), 'CallForeign ("~w", ~w)', [CP, Num]).
wam_instr_to_fsharp(["execute", P], Fs) :-
    format(string(Fs), 'Execute "~w"', [P]).
wam_instr_to_fsharp(["proceed"], "Proceed").
wam_instr_to_fsharp(["fail"], "Fail").
wam_instr_to_fsharp(["jump", Label], Fs) :-
    format(string(Fs), 'Jump "~w"', [Label]).
wam_instr_to_fsharp(["cut_ite"], "CutIte").
wam_instr_to_fsharp(["builtin_call", Op, N], Fs) :-
    fs_clean_comma(Op, COp), fs_clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    fs_escape_string(COp, ECOp),
    format(string(Fs), 'BuiltinCall ("~w", ~w)', [ECOp, Num]).
wam_instr_to_fsharp(["try_me_else", Label], Fs) :-
    format(string(Fs), 'TryMeElse "~w"', [Label]).
wam_instr_to_fsharp(["retry_me_else", Label], Fs) :-
    format(string(Fs), 'RetryMeElse "~w"', [Label]).
wam_instr_to_fsharp(["trust_me"], "TrustMe").
wam_instr_to_fsharp(["switch_on_constant"|Entries], Fs) :-
    fs_parse_switch_entries(Entries, FsPairs),
    atomic_list_concat(FsPairs, '; ', PairsStr),
    format(string(Fs), 'SwitchOnConstant (Map.ofList [~w])', [PairsStr]).
wam_instr_to_fsharp(["switch_on_constant_a2"|Entries], Fs) :-
    fs_parse_switch_entries(Entries, FsPairs),
    atomic_list_concat(FsPairs, '; ', PairsStr),
    format(string(Fs), 'SwitchOnConstant (Map.ofList [~w])', [PairsStr]).
wam_instr_to_fsharp(["begin_aggregate", Type, ValReg, ResReg], Fs) :-
    fs_clean_comma(Type, CT), fs_clean_comma(ValReg, CV), fs_clean_comma(ResReg, CR),
    fs_reg_name_to_int(CV, VI), fs_reg_name_to_int(CR, RI),
    format(string(Fs), 'BeginAggregate ("~w", ~w, ~w)', [CT, VI, RI]).
wam_instr_to_fsharp(["end_aggregate", ValReg], Fs) :-
    fs_clean_comma(ValReg, CV),
    fs_reg_name_to_int(CV, VI),
    format(string(Fs), 'EndAggregate ~w', [VI]).
% Fallback for unknown instructions
wam_instr_to_fsharp(Parts, Fs) :-
    atomic_list_concat(Parts, ' ', Joined),
    format(string(Fs), '(* UNKNOWN: ~w *) Proceed', [Joined]).

%% fs_parse_switch_entries(+Entries, -FSharpPairs)
fs_parse_switch_entries([], []).
fs_parse_switch_entries([Entry|Rest], [FsPair|FsRest]) :-
    fs_clean_comma(Entry, CEntry),
    (   sub_atom(CEntry, Before, 1, _, ':')
    ->  sub_atom(CEntry, 0, Before, _, Key),
        After is Before + 1,
        sub_atom(CEntry, After, _, 0, Label),
        fs_wam_value(Key, FsKey),
        format(string(FsPair), '(~w, "~w")', [FsKey, Label])
    ;   format(string(FsPair), '(Atom "~w", "default")', [CEntry])
    ),
    fs_parse_switch_entries(Rest, FsRest).

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

    % Generate WamTypes.fs (separate module for types)
    fsharp_wam_type_header(TypeHeader),
    directory_file_path(ProjectDir, 'WamTypes.fs', TypesPath),
    write_fs_file(TypesPath, TypeHeader),

    % Generate WamRuntime.fs (runtime functions only)
    compile_wam_runtime_to_fsharp(Options, DetectedKernels, RuntimeCode),
    directory_file_path(ProjectDir, 'WamRuntime.fs', RuntimePath),
    write_fs_file(RuntimePath, RuntimeCode),

    % Compute base PCs for all predicates (shared between Predicates.fs and Lowered.fs)
    compute_base_pcs_fs(Predicates, BasePCMap),

    % Generate Predicates.fs (skip FFI-owned facts — Phase D: -70%)
    compile_predicates_to_fsharp(Predicates, Options, DetectedKernels, BasePCMap, PredsCode),
    directory_file_path(ProjectDir, 'Predicates.fs', PredsPath),
    write_fs_file(PredsPath, PredsCode),

    % Generate Lowered.fs
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
    open(Path, write, Stream, [encoding(utf8)]),
    write(Stream, Content),
    close(Stream).

%% compile_predicates_to_fsharp(+Predicates, +Options, +DetectedKernels, +BasePCMap, -Code)
compile_predicates_to_fsharp(Predicates, Options, DetectedKernels, BasePCMap, Code) :-
    % Phase D: skip FFI-owned facts (predicates handled entirely by FFI kernel path)
    exclude(ffi_owned_fact_filter_fs(DetectedKernels), Predicates, WamPredicates),
    (   length(Predicates, NAll), length(WamPredicates, NWam), NSkipped is NAll - NWam,
        NSkipped > 0
    ->  format(user_error, '[WAM-FSharp] skipped ~w FFI-owned fact predicates~n', [NSkipped])
    ;   true
    ),
    maplist(compile_one_predicate_fs(Options, BasePCMap), WamPredicates, PredCodes),
    atomic_list_concat(PredCodes, '\n\n', AllPredCode),
    % Build merged code list and label map
    maplist(pred_func_name_fs, WamPredicates, FuncNames),
    emit_merged_code_build_fs(FuncNames, MergedCodeBuild),
    format(string(Code),
'module Predicates

open WamTypes
open WamRuntime

~w

~w
', [AllPredCode, MergedCodeBuild]).

compile_one_predicate_fs(Options, BasePCMap, PredIndicator, Code) :-
    (   PredIndicator = _M:Pred/Arity -> true ; PredIndicator = Pred/Arity ),
    wam_fsharp_predicate_wamcode(PredIndicator, WamCode),
    predicate_base_pc_fs(PredIndicator, BasePCMap, BasePC),
    compile_wam_predicate_to_fsharp(PredIndicator, WamCode, [base_pc(BasePC)|Options], Code).

pred_func_name_fs(PI, FN) :-
    (   PI = _M:P/A -> true ; PI = P/A ),
    atom_string(P, PStr),
    atomic_list_concat(Parts, '$', PStr),
    atomic_list_concat(Parts, '_', PSafe),
    format(atom(FN), '~w_~w', [PSafe, A]).

emit_merged_code_build_fs([], Code) :-
    Code = 'let allCode : Instruction array = [||]\nlet allLabels : Map<string, int> = Map.empty'.
emit_merged_code_build_fs(FuncNames, Code) :-
    FuncNames \= [],
    maplist([FN, Expr]>>(format(atom(Expr), '~w_code', [FN])), FuncNames, CodeExprs),
    atomic_list_concat(CodeExprs, ' @ ', CodeConcat),
    maplist([FN, Expr]>>(format(atom(Expr), '~w_labels', [FN])), FuncNames, LabelExprs),
    atomic_list_concat(LabelExprs, ' |> mapUnion ', LabelUnion),
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
open System.Diagnostics
open WamTypes
open WamRuntime
open Predicates
open Lowered

// -- TSV loading helpers --------------------------------------------------

let loadTsvPairs (path: string) : (string * string) array =
    System.IO.File.ReadAllLines(path)
    |> Array.skip 1
    |> Array.filter (fun l -> l.Trim().Length > 0 && not (l.StartsWith("#")))
    |> Array.choose (fun l ->
        let parts = l.Split(''\\t'')
        if parts.Length >= 2 then Some (parts.[0].Trim(), parts.[1].Trim())
        else None)

let loadSingleColumn (path: string) : string array =
    System.IO.File.ReadAllLines(path)
    |> Array.skip 1
    |> Array.filter (fun l -> l.Trim().Length > 0 && not (l.StartsWith("#")))
    |> Array.map (fun l -> l.Trim())

// -- Atom interning from all string sources --------------------------------

let buildFullAtomInternTable (code: Instruction array) (extraAtoms: string seq) : Map<string, int> * Map<int, string> =
    let atoms = System.Collections.Generic.HashSet<string>()
    for instr in code do
        match instr with
        | GetConstant (Atom a, _) -> atoms.Add(a) |> ignore
        | PutConstant (Atom a, _) -> atoms.Add(a) |> ignore
        | SwitchOnConstantPc table -> for (k, _) in table do atoms.Add(k) |> ignore
        | _ -> ()
    for a in extraAtoms do atoms.Add(a) |> ignore
    let sorted = atoms |> Seq.sort |> Seq.toArray
    let intern   = sorted |> Array.mapi (fun i a -> (a, i)) |> Map.ofArray
    let deintern = sorted |> Array.mapi (fun i a -> (i, a)) |> Map.ofArray
    intern, deintern

// -- Build interned FFI facts map -----------------------------------------

let buildFfiFacts (pairs: (string * string) array) (intern: Map<string, int>) : Map<int, int list> =
    let mutable m : Map<int, int list> = Map.empty
    for (child, parent) in pairs do
        match Map.tryFind child intern, Map.tryFind parent intern with
        | Some cid, Some pid ->
            let existing = Map.tryFind cid m |> Option.defaultValue []
            m <- Map.add cid (pid :: existing) m
        | _ -> ()
    m

// -- Extract double from WAM Value ----------------------------------------

let extractDouble (v: Value) : float option =
    match v with
    | Float f   -> Some f
    | Integer n -> Some (float n)
    | _         -> None

[<EntryPoint>]
let main argv =
    let factsDir = if argv.Length > 0 then argv.[0] else "."
    let numReps  = if argv.Length > 1 then (try int argv.[1] with _ -> 3) else 3

    let sw = Stopwatch.StartNew()

    // Load TSV facts
    let categoryParents  = loadTsvPairs (System.IO.Path.Combine(factsDir, "category_parent.tsv"))
    let articleCategories = loadTsvPairs (System.IO.Path.Combine(factsDir, "article_category.tsv"))
    let roots            = loadSingleColumn (System.IO.Path.Combine(factsDir, "root_categories.tsv"))
    let root = if roots.Length > 0 then roots.[0] else "Physics"

    let loadMs = sw.ElapsedMilliseconds
    eprintfn "load_ms=%d" loadMs

    // Resolve call instructions
    let foreignPreds = [ ~w ]
    let resolvedCode = resolveCallInstrs allLabels foreignPreds (Array.toList allCode) |> List.toArray

    // Build atom intern table from instructions + all TSV atoms
    let extraAtoms = seq {
        for (c, p) in categoryParents do yield c; yield p
        for (_, c) in articleCategories do yield c
        for r in roots do yield r
    }
    let atomIntern, atomDeintern = buildFullAtomInternTable resolvedCode extraAtoms

    // Build interned FFI facts: category_parent child -> [parent ids]
    let parentsInterned = buildFfiFacts categoryParents atomIntern

    // Seed categories (distinct second column of article_category)
    let seedCats =
        articleCategories
        |> Array.map snd
        |> Array.distinct
        |> Array.sort

    // Build WamContext
    let ctx =
        { WcCode             = resolvedCode
          WcLabels            = allLabels
          WcForeignFacts      = Map.empty
          WcFfiFacts          = Map.ofList [ ("category_parent", parentsInterned) ]
          WcFfiWeightedFacts  = Map.empty
          WcAtomIntern        = atomIntern
          WcAtomDeintern      = atomDeintern
          WcForeignConfig     = Map.ofList [ ("max_depth", 10) ]
          WcLoweredPredicates = loweredPredicates }

    let emptyState =
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

    // Find entry point: prefer lowered predicates, fall back to code array
    let queryPred =
        let candidates = [
            "category_ancestor$effective_distance_sum_selected/3"
            "category_ancestor$effective_distance_sum_bound/3"
            "category_ancestor/4"
        ]
        candidates
        |> List.tryFind (fun p ->
            Map.containsKey p ctx.WcLoweredPredicates ||
            Map.containsKey p ctx.WcLabels)
        |> Option.defaultValue "category_ancestor/4"

    let setupMs = sw.ElapsedMilliseconds
    eprintfn "setup_ms=%d queryPred=%s" setupMs queryPred

    // Benchmark loop: numReps repetitions
    let mutable totalSolutions = 0
    let mutable lastQueryMs = 0L

    for rep = 1 to numReps do
        let querySw = Stopwatch.StartNew()
        let mutable repSolutions = 0

        for cat in seedCats do
            let varId = 1000000
            let regs = Array.create MaxRegs (Unbound -1)
            regs.[1] <- Atom cat
            regs.[2] <- Atom root
            regs.[3] <- Unbound varId
            let s0 = { emptyState with WsPC = 0; WsRegs = regs; WsCP = 0 }
            match dispatchCall ctx queryPred s0 with
            | Some s1 ->
                repSolutions <- repSolutions + 1
            | None -> ()

        querySw.Stop()
        lastQueryMs <- querySw.ElapsedMilliseconds
        totalSolutions <- totalSolutions + repSolutions
        eprintfn "rep=%d query_ms=%d seeds=%d solutions=%d"
            rep lastQueryMs seedCats.Length repSolutions

    sw.Stop()
    let totalMs = sw.ElapsedMilliseconds

    // Summary output
    printfn "query_ms=%d seeds=%d solutions=%d reps=%d"
        lastQueryMs seedCats.Length (totalSolutions / numReps) numReps
    printfn "total_ms=%d" totalMs
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
    <Compile Include="WamTypes.fs" />
    <Compile Include="WamRuntime.fs" />
    <Compile Include="Predicates.fs" />
    <Compile Include="Lowered.fs" />
    <Compile Include="Program.fs" />
  </ItemGroup>

</Project>
', [ModName]).
